"""Hash-gated Bayesian analysis for the completed V2 evidence package."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v2_engineering import STATUS_VALID, V2_CONDITIONS, V2_MODELS
from src.v2_execution import (
    V2ExecutionContext,
    execution_blockers,
    load_execution_context,
    validate_v2_output_frames,
)

ROPE = (-0.01, 0.01)
MIN_COMPLETE_BLOCKS = 3


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def declared_contrasts(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Expand the preregistered contrast families without inspecting scores."""

    pairs: list[tuple[str, str]] = []
    if config["contrasts"].get("raw_vs_all_applicable"):
        pairs.extend((condition, "raw") for condition in V2_CONDITIONS if condition != "raw")
    pairs.extend(tuple(pair) for pair in config["contrasts"]["weighting_vs_sampling"])
    pairs.extend(tuple(pair) for pair in config["contrasts"]["balanced_random_forest_vs_rf_interventions"])
    pairs.extend(tuple(pair) for pair in config["contrasts"]["sampler_pairs"])
    return list(dict.fromkeys(pairs))


def paired_contrast_blocks(
    results: pd.DataFrame,
    left_condition: str,
    right_condition: str,
    classifier: str,
    feature_type: str | None = None,
) -> pd.DataFrame:
    """Return complete dataset/outer-fold paired differences for one contrast."""

    required = {
        "dataset_id",
        "outer_fold",
        "classifier",
        "condition",
        "feature_type",
        "macro_average_precision",
        "status",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"results are missing V2 columns: {missing}")
    subset = results[
        (results["status"] == STATUS_VALID)
        & (results["classifier"] == classifier)
        & (results["condition"].isin([left_condition, right_condition]))
    ].copy()
    if feature_type is not None:
        subset = subset[subset["feature_type"] == feature_type]
    if subset.empty:
        return pd.DataFrame(
            columns=["dataset_id", "outer_fold", "classifier", "feature_type", "left", "right", "delta"]
        )
    pivot = subset.pivot_table(
        index=["dataset_id", "outer_fold", "feature_type"],
        columns="condition",
        values="macro_average_precision",
        aggfunc="first",
    )
    if left_condition not in pivot or right_condition not in pivot:
        return pd.DataFrame()
    paired = pivot.dropna(subset=[left_condition, right_condition]).reset_index()
    if paired.empty:
        return pd.DataFrame()
    paired = paired.rename(columns={left_condition: "left", right_condition: "right"})
    paired["classifier"] = classifier
    paired["delta"] = paired["left"] - paired["right"]
    counts = paired.groupby("dataset_id")["outer_fold"].nunique()
    complete_ids = counts[counts == 5].index
    paired = paired[paired["dataset_id"].isin(complete_ids)].copy()
    return paired[
        ["dataset_id", "outer_fold", "classifier", "feature_type", "left", "right", "delta"]
    ].sort_values(["dataset_id", "outer_fold"])


def rope_category(probability_worse: float, probability_equivalent: float, probability_better: float) -> str:
    """Use the preregistered posterior ROPE category."""

    probabilities = {
        "practically_worse": probability_worse,
        "practically_equivalent": probability_equivalent,
        "practically_better": probability_better,
    }
    return max(probabilities, key=probabilities.get)


def fit_bayesian_contrast(
    blocks: pd.DataFrame,
    *,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 20260917,
) -> tuple[dict[str, Any], Any]:
    """Fit the preregistered exchangeable fold-correlated hierarchical model."""

    if blocks.empty:
        raise ValueError("cannot fit a Bayesian contrast without complete blocks")
    dataset_ids = sorted(blocks["dataset_id"].unique())
    if len(dataset_ids) < MIN_COMPLETE_BLOCKS:
        raise ValueError(f"fewer than {MIN_COMPLETE_BLOCKS} complete dataset blocks")
    observations = blocks.pivot(index="dataset_id", columns="outer_fold", values="delta")
    observations = observations.reindex(index=dataset_ids, columns=range(5))
    if observations.isna().any().any():
        raise ValueError("Bayesian input contains an incomplete five-fold block")
    values = observations.to_numpy(dtype=float)
    import arviz as az
    import pymc as pm

    with pm.Model():
        mu = pm.Normal("mu", mu=0.0, sigma=0.5)
        tau_dataset = pm.HalfNormal("tau_dataset", sigma=0.5)
        sigma_fold = pm.HalfNormal("sigma_fold", sigma=0.25)
        rho = pm.Beta("rho", alpha=2.0, beta=2.0)
        dataset_offset = pm.Normal("dataset_offset", mu=0.0, sigma=tau_dataset, shape=len(dataset_ids))
        identity = np.eye(5)
        ones = np.ones((5, 5))
        covariance = sigma_fold**2 * ((1.0 - rho) * identity + rho * ones)
        pm.MvNormal(
            "delta",
            mu=(mu + dataset_offset)[:, None] * np.ones((1, 5)),
            cov=covariance,
            observed=values,
        )
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
            return_inferencedata=True,
        )
    summary = az.summary(trace, var_names=["mu", "tau_dataset", "sigma_fold", "rho"])
    r_hat_max = float(summary["r_hat"].max())
    bulk_ess_min = float(summary["ess_bulk"].min())
    tail_ess_min = float(summary["ess_tail"].min())
    divergences = int(trace.sample_stats["diverging"].sum().item())
    diagnostics = {
        "r_hat_max": r_hat_max,
        "bulk_ess_minimum": bulk_ess_min,
        "tail_ess_minimum": tail_ess_min,
        "divergences": divergences,
    }
    if r_hat_max > 1.01 or bulk_ess_min < 400 or tail_ess_min < 400 or divergences != 0:
        raise RuntimeError(f"Bayesian diagnostics failed: {diagnostics}")
    mu_samples = trace.posterior["mu"].values.reshape(-1)
    probabilities = {
        "p_practically_worse": float(np.mean(mu_samples < ROPE[0])),
        "p_practically_equivalent": float(np.mean((mu_samples >= ROPE[0]) & (mu_samples <= ROPE[1]))),
        "p_practically_better": float(np.mean(mu_samples > ROPE[1])),
    }
    payload = {
        "status": "VALID",
        "complete_dataset_blocks": len(dataset_ids),
        "outer_folds": 5,
        "posterior_mean_mu": float(np.mean(mu_samples)),
        "posterior_median_mu": float(np.median(mu_samples)),
        "credible_interval_95": [
            float(np.quantile(mu_samples, 0.025)),
            float(np.quantile(mu_samples, 0.975)),
        ],
        **probabilities,
        "rope": list(ROPE),
        "posterior_category": rope_category(
            probabilities["p_practically_worse"],
            probabilities["p_practically_equivalent"],
            probabilities["p_practically_better"],
        ),
        "diagnostics": diagnostics,
    }
    return payload, trace


def _full_run_blockers(context: V2ExecutionContext) -> tuple[str, ...]:
    blockers = list(execution_blockers(context))
    manifest_path = context.output_dir / "v2_manifest.json"
    if not manifest_path.is_file():
        blockers.append("complete V2 full-run manifest is missing")
        return tuple(dict.fromkeys(blockers))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        blockers.append(f"full-run manifest is unreadable: {type(error).__name__}")
        return tuple(dict.fromkeys(blockers))
    if manifest.get("status") != "complete":
        blockers.append(f"full-run manifest status is {manifest.get('status')!r}")
    if manifest.get("config_sha256") != context.config_sha256:
        blockers.append("full-run manifest config hash is stale")
    if manifest.get("registry_sha256") != context.registry_sha256:
        blockers.append("full-run manifest registry hash is stale")
    aggregate = context.output_dir / "v2_results.csv"
    failures = context.output_dir / "v2_failures.csv"
    if not aggregate.is_file() or not failures.is_file():
        blockers.append("aggregate V2 result/failure tables are missing")
    return tuple(dict.fromkeys(blockers))


def analysis_dry_run(context: V2ExecutionContext) -> dict[str, Any]:
    blockers = _full_run_blockers(context)
    return {
        "status": "BLOCKED" if blockers else "READY",
        "dataset_count": len(context.datasets),
        "declared_contrasts": declared_contrasts(context.config),
        "rope_macro_average_precision": list(ROPE),
        "confirmatory_p_values": False,
        "blockers": list(blockers),
        "analysis_started": False,
    }


def run_v2_analysis(
    context: V2ExecutionContext,
    *,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or plan Bayesian contrasts from a complete, hash-matched V2 run."""

    output_dir = output_dir or context.root / "results/v2-analysis"
    summary = analysis_dry_run(context)
    if dry_run:
        return summary
    blockers = tuple(summary["blockers"])
    if blockers:
        raise RuntimeError(f"V2 analysis is blocked: {'; '.join(blockers)}")
    results = pd.read_csv(context.output_dir / "v2_results.csv")
    failures = pd.read_csv(context.output_dir / "v2_failures.csv")
    for spec in context.datasets:
        result_subset = results[results["dataset_id"] == spec.dataset_id]
        failure_subset = failures[failures["dataset_id"] == spec.dataset_id]
        validate_v2_output_frames(spec.dataset_id, result_subset, failure_subset)
    output_dir.mkdir(parents=True, exist_ok=True)
    contrast_rows: list[dict[str, Any]] = []
    posterior_dir = output_dir / "posteriors"
    posterior_dir.mkdir(exist_ok=True)
    for left, right in declared_contrasts(context.config):
        classifiers = ("random_forest",) if "balanced_random_forest" in {left, right} else V2_MODELS
        for classifier in classifiers:
            blocks = paired_contrast_blocks(results, left, right, classifier)
            row: dict[str, Any] = {
                "left_condition": left,
                "right_condition": right,
                "classifier": classifier,
                "complete_dataset_blocks": int(blocks["dataset_id"].nunique()) if not blocks.empty else 0,
            }
            if row["complete_dataset_blocks"] < MIN_COMPLETE_BLOCKS:
                row.update({"status": "SKIPPED", "reason": "fewer than three complete dataset blocks"})
            else:
                try:
                    payload, trace = fit_bayesian_contrast(blocks)
                    name = f"{left}_vs_{right}_{classifier}".replace("/", "_")
                    trace.to_netcdf(posterior_dir / f"{name}.nc")
                    row.update(payload)
                    row["posterior_path"] = str((posterior_dir / f"{name}.nc").relative_to(output_dir))
                except Exception as error:  # noqa: BLE001 - analysis failure is explicit
                    row.update({"status": "ANALYSIS_FAILED", "reason": str(error)})
            contrast_rows.append(row)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(output_dir / "bayesian_contrasts.csv", index=False)
    manifest = {
        "schema_version": "v2-analysis-v1",
        "status": "complete",
        "created_at_utc": _utc_now(),
        "config_sha256": context.config_sha256,
        "registry_sha256": context.registry_sha256,
        "results_sha256": _sha256_file(context.output_dir / "v2_results.csv"),
        "failures_sha256": _sha256_file(context.output_dir / "v2_failures.csv"),
        "rope": list(ROPE),
        "confirmatory_p_values": False,
        "contrast_table": "bayesian_contrasts.csv",
    }
    _write_json_atomic(output_dir / "analysis_manifest.json", manifest)
    return manifest


def load_analysis_context(root: Path, output_dir: Path | None = None) -> V2ExecutionContext:
    """Load the same frozen execution context used by the full-run guard."""

    return load_execution_context(root, output_dir=output_dir)
