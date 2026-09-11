"""Gate D validation and preregistered analysis for the locked benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.benchmark import (
    FAILURE_COLUMNS,
    LOCKED_FOLDS,
    LOCKED_SEEDS,
    RESULT_COLUMNS,
    _load_context,
    _read_json,
    _validate_completed_dataset,
    _write_json_atomic,
    sha256_file,
)

SCALAR_METRICS = ("macro_f1", "g_mean", "mcc", "balanced_accuracy")
CLASSIFIERS = ("logistic_regression", "random_forest", "xgboost")
EFFICIENCY_METRICS = (
    "fit_predict_seconds",
    "rss_before_mb",
    "rss_after_mb",
    "rss_delta_mb",
    "train_rows_before_sampling",
    "train_rows_after_sampling",
)
FAMILY_DEFINITIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "numeric_core": (
        "numeric",
        (
            "raw",
            "class_weighted",
            "random_over",
            "random_under",
            "smote",
            "adasyn",
            "borderline_smote",
            "smoteenn",
            "smotetomek",
        ),
    ),
    "mixed_core": (
        "mixed",
        (
            "raw",
            "class_weighted",
            "random_over",
            "random_under",
            "smotenc",
            "smoteenn",
            "smotetomek",
        ),
    ),
    "balanced_rf_numeric": ("numeric", ("raw", "class_weighted", "balanced_random_forest")),
    "balanced_rf_mixed": ("mixed", ("raw", "class_weighted", "balanced_random_forest")),
}


def _family_classifiers(family: str) -> tuple[str, ...]:
    """Return the classifiers allowed by the frozen family definition."""

    if family.startswith("balanced_rf"):
        return ("random_forest",)
    return CLASSIFIERS


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _reference(path: Path, root: Path) -> dict[str, str]:
    return {"path": _relative_path(path, root), "sha256": sha256_file(path)}


def _execution_paths(root: Path) -> dict[str, Path]:
    return {
        "scope_lock": root / "artifacts" / "runs" / "scope-lock.json",
        "registry": root / "data" / "dataset_registry.csv",
        "manifest": root / "data" / "acquisition_manifest.csv",
        "overrides": root / "data" / "dataset_feature_overrides.csv",
        "protocol": root / "docs" / "protocol.md",
        "config": root / "config" / "stage0.yaml",
        "environment": root / "artifacts" / "environment" / "metadata.json",
    }


def verify_gate_d(
    root: Path,
    run_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Verify every locked raw output and write a Gate D review payload."""

    paths = _execution_paths(root)
    context = _load_context(root, paths)
    manifest_path = run_dir / "benchmark_manifest.json"
    manifest = _read_json(manifest_path, "benchmark manifest")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"benchmark manifest is not complete: {manifest.get('status')!r}")
    expected_total = len(context.datasets) * 495
    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, dict):
        raise TypeError("benchmark manifest has no aggregate references")
    aggregate_paths = {
        "results": run_dir / "benchmark_results.csv",
        "failures": run_dir / "benchmark_failures.csv",
    }
    aggregate_frames: dict[str, pd.DataFrame] = {}
    aggregate_refs: dict[str, dict[str, str]] = {}
    for name, path in aggregate_paths.items():
        if not path.is_file():
            raise RuntimeError(f"aggregate {name} file is missing: {path}")
        reference = aggregate.get(name)
        actual_reference = _reference(path, root)
        if not isinstance(reference, dict) or reference != actual_reference:
            raise RuntimeError(f"aggregate {name} hash does not match the benchmark manifest")
        aggregate_refs[name] = actual_reference
        aggregate_frames[name] = pd.read_csv(path)
    dataset_counts: dict[str, dict[str, int]] = {}
    marker_refs: dict[str, dict[str, str]] = {}
    for spec in context.datasets:
        dataset_dir = run_dir / spec.dataset_id
        marker_path = dataset_dir / "benchmark_complete.json"
        dataset_counts[spec.dataset_id] = _validate_completed_dataset(
            spec, dataset_dir, marker_path, root
        )
        marker_refs[spec.dataset_id] = _reference(marker_path, root)
    totals = {
        "expected_cells": sum(item["expected_cells"] for item in dataset_counts.values()),
        "valid_cells": sum(item["valid_cells"] for item in dataset_counts.values()),
        "failure_cells": sum(item["failure_cells"] for item in dataset_counts.values()),
    }
    if totals != {
        "expected_cells": expected_total,
        "valid_cells": len(aggregate_frames["results"]),
        "failure_cells": len(aggregate_frames["failures"]),
    }:
        raise RuntimeError(f"aggregate counts do not match dataset markers: {totals}")
    if totals["valid_cells"] + totals["failure_cells"] != totals["expected_cells"]:
        raise RuntimeError("Gate D cell accounting is incomplete")
    missing_results = set(RESULT_COLUMNS).difference(aggregate_frames["results"].columns)
    missing_failures = set(FAILURE_COLUMNS).difference(aggregate_frames["failures"].columns)
    if missing_results or missing_failures:
        raise RuntimeError(
            f"aggregate schema is incomplete: results={sorted(missing_results)}, "
            f"failures={sorted(missing_failures)}"
        )
    payload: dict[str, Any] = {
        "schema_version": "gate-d-review-v1",
        "status": "passed",
        "reviewed_at_utc": _utc_now(),
        "benchmark_manifest": _reference(manifest_path, root),
        "dataset_ids": [spec.dataset_id for spec in context.datasets],
        "dataset_count": len(context.datasets),
        "seeds": list(LOCKED_SEEDS),
        "folds": LOCKED_FOLDS,
        "expected_cells_per_dataset": 495,
        "counts": totals,
        "dataset_counts": dataset_counts,
        "dataset_markers": marker_refs,
        "aggregate": aggregate_refs,
        "frozen_references": context.references,
        "selection_rule": "locked registry eligibility and complete raw cell accounting only",
        "analysis_status": "not_started",
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output_path, payload)
    return payload


def _parse_recall(value: object) -> np.ndarray:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise RuntimeError("per_class_recall contains invalid JSON") from error
    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError("per_class_recall must be a non-empty list")
    values = np.asarray(parsed, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("per_class_recall contains invalid numeric values")
    return values


def load_cell_means(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate valid fold/seed records into complete dataset cells."""

    grouped = results.groupby(
        ["dataset", "classifier", "condition", "feature_type"],
        sort=True,
        dropna=False,
    )
    scalar_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for key, group in grouped:
        if len(group) != len(LOCKED_SEEDS) * LOCKED_FOLDS:
            continue
        if set(group["seed"].astype(int)) != set(LOCKED_SEEDS):
            continue
        if set(group["fold"].astype(int)) != set(range(LOCKED_FOLDS)):
            continue
        dataset, classifier, condition, feature_type = key
        scalar_row: dict[str, Any] = {
            "dataset": dataset,
            "classifier": classifier,
            "condition": condition,
            "feature_type": feature_type,
            "complete_records": len(group),
        }
        scalar_row.update({metric: float(group[metric].mean()) for metric in SCALAR_METRICS})
        recalls = np.vstack([_parse_recall(value) for value in group["per_class_recall"]])
        mean_recalls = recalls.mean(axis=0)
        scalar_row["per_class_recall"] = json.dumps(mean_recalls.tolist())
        scalar_rows.append(scalar_row)
        for class_index, value in enumerate(mean_recalls):
            class_rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "condition": condition,
                    "feature_type": feature_type,
                    "class_index": class_index,
                    "recall": float(value),
                    "complete_records": len(group),
                }
            )
    return pd.DataFrame(scalar_rows), pd.DataFrame(class_rows)


def load_efficiency_means(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate runtime, memory, and resampling fields for complete cells."""

    grouped = results.groupby(
        ["dataset", "classifier", "condition", "feature_type"],
        sort=True,
        dropna=False,
    )
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        if len(group) != len(LOCKED_SEEDS) * LOCKED_FOLDS:
            continue
        if set(group["seed"].astype(int)) != set(LOCKED_SEEDS):
            continue
        if set(group["fold"].astype(int)) != set(range(LOCKED_FOLDS)):
            continue
        dataset, classifier, condition, feature_type = key
        row: dict[str, Any] = {
            "dataset": dataset,
            "classifier": classifier,
            "condition": condition,
            "feature_type": feature_type,
            "complete_records": len(group),
        }
        for metric in EFFICIENCY_METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def _family_matrix(
    cell_means: pd.DataFrame,
    family: str,
    classifier: str,
    metric: str,
    class_index: int | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    feature_type, conditions = FAMILY_DEFINITIONS[family]
    subset = cell_means[
        (cell_means["feature_type"] == feature_type)
        & (cell_means["classifier"] == classifier)
    ]
    if class_index is not None:
        subset = subset[subset["class_index"] == class_index]
    pivot = subset.pivot(index="dataset", columns="condition", values=metric)
    available = tuple(condition for condition in conditions if condition in pivot.columns)
    return pivot, available


def build_baseline_deltas(
    cell_means: pd.DataFrame,
    class_means: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build scalar and class-wise paired deltas from complete cells."""

    scalar_rows: list[dict[str, Any]] = []
    for family, (feature_type, conditions) in FAMILY_DEFINITIONS.items():
        for classifier in _family_classifiers(family):
            subset = cell_means[
                (cell_means["feature_type"] == feature_type)
                & (cell_means["classifier"] == classifier)
            ].set_index(["dataset", "condition"])
            for condition in conditions:
                if condition == "raw":
                    continue
                for dataset in sorted(set(subset.index.get_level_values("dataset"))):
                    if (dataset, "raw") not in subset.index or (dataset, condition) not in subset.index:
                        continue
                    raw = subset.loc[(dataset, "raw")]
                    alternative = subset.loc[(dataset, condition)]
                    for metric in SCALAR_METRICS:
                        scalar_rows.append(
                            {
                                "family": family,
                                "dataset": dataset,
                                "classifier": classifier,
                                "condition": condition,
                                "metric": metric,
                                "raw_mean": float(raw[metric]),
                                "alternative_mean": float(alternative[metric]),
                                "delta": float(alternative[metric] - raw[metric]),
                            }
                        )
    class_rows: list[dict[str, Any]] = []
    for family, (feature_type, conditions) in FAMILY_DEFINITIONS.items():
        for classifier in _family_classifiers(family):
            subset = class_means[
                (class_means["feature_type"] == feature_type)
                & (class_means["classifier"] == classifier)
            ].set_index(["dataset", "condition", "class_index"])
            for condition in conditions:
                if condition == "raw":
                    continue
                for dataset in sorted(set(subset.index.get_level_values("dataset"))):
                    for class_index in sorted(
                        {
                            index[2]
                            for index in subset.index
                            if index[0] == dataset
                        }
                    ):
                        raw_key = (dataset, "raw", class_index)
                        alternative_key = (dataset, condition, class_index)
                        if raw_key not in subset.index or alternative_key not in subset.index:
                            continue
                        raw = float(subset.loc[raw_key, "recall"])
                        alternative = float(subset.loc[alternative_key, "recall"])
                        class_rows.append(
                            {
                                "family": family,
                                "dataset": dataset,
                                "classifier": classifier,
                                "condition": condition,
                                "class_index": class_index,
                                "raw_recall": raw,
                                "alternative_recall": alternative,
                                "delta": alternative - raw,
                            }
                        )
    return pd.DataFrame(scalar_rows), pd.DataFrame(class_rows)


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_rank_biserial(differences: np.ndarray) -> float:
    nonzero = np.asarray(differences, dtype=float)
    nonzero = nonzero[nonzero != 0]
    if len(nonzero) == 0:
        return 0.0
    from scipy.stats import rankdata

    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (len(nonzero) * (len(nonzero) + 1) / 2)


def paired_bootstrap_ci(
    differences: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    if len(differences) == 0:
        raise ValueError("bootstrap requires at least one paired block")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    means = differences[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _comparison_summary(
    family: str,
    classifier: str,
    metric: str,
    condition: str,
    matrix: pd.DataFrame,
) -> dict[str, Any]:
    pair = matrix[["raw", condition]].dropna()
    differences = pair[condition].to_numpy(dtype=float) - pair["raw"].to_numpy(dtype=float)
    wins = int((differences > 1e-12).sum())
    losses = int((differences < -1e-12).sum())
    ties = int(len(differences) - wins - losses)
    row: dict[str, Any] = {
        "family": family,
        "classifier": classifier,
        "metric": metric,
        "condition": condition,
        "n_pairs": len(differences),
        "mean_delta": float(differences.mean()) if len(differences) else None,
        "median_delta": float(np.median(differences)) if len(differences) else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "rank_biserial": paired_rank_biserial(differences) if len(differences) else None,
        "bootstrap_low": None,
        "bootstrap_high": None,
        "wilcoxon_statistic": None,
        "wilcoxon_p_value": None,
        "status": "skipped" if not len(differences) else "computed",
        "reason": "no complete paired blocks" if not len(differences) else None,
    }
    if len(differences):
        row["bootstrap_low"], row["bootstrap_high"] = paired_bootstrap_ci(differences)
        from scipy.stats import wilcoxon

        if np.all(differences == 0):
            row["wilcoxon_statistic"] = 0.0
            row["wilcoxon_p_value"] = 1.0
        else:
            statistic, p_value = wilcoxon(
                pair["raw"].to_numpy(dtype=float),
                pair[condition].to_numpy(dtype=float),
                zero_method="wilcox",
                correction=False,
                method="auto",
                alternative="two-sided",
            )
            row["wilcoxon_statistic"] = float(statistic)
            row["wilcoxon_p_value"] = float(p_value)
    return row


def _run_confirmatory_tests(
    cell_means: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    friedman_rows: list[dict[str, Any]] = []
    nemenyi_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for family, (_feature_type, conditions) in FAMILY_DEFINITIONS.items():
        for classifier in _family_classifiers(family):
            for metric in SCALAR_METRICS:
                matrix, available = _family_matrix(cell_means, family, classifier, metric)
                missing = sorted(set(conditions) - set(available))
                blocks = matrix[list(available)].dropna() if available else pd.DataFrame()
                status = "computed"
                reason = None
                statistic = None
                p_value = None
                if len(available) < 3:
                    status = "skipped"
                    reason = "fewer than three complete conditions"
                elif len(blocks) < 3:
                    status = "skipped"
                    reason = "fewer than three complete dataset blocks"
                elif missing:
                    status = "skipped"
                    reason = "incomplete condition matrix: " + ", ".join(missing)
                else:
                    from scipy.stats import friedmanchisquare

                    statistic, p_value = friedmanchisquare(
                        *[blocks[condition].to_numpy() for condition in conditions]
                    )
                friedman_rows.append(
                    {
                        "family": family,
                        "classifier": classifier,
                        "metric": metric,
                        "class_index": None,
                        "n_blocks": len(blocks),
                        "n_conditions": len(available),
                        "friedman_statistic": float(statistic) if statistic is not None else None,
                        "friedman_p_value": float(p_value) if p_value is not None else None,
                        "status": status,
                        "reason": reason,
                    }
                )
                if status == "computed" and p_value is not None and p_value < 0.05:
                    import scikit_posthocs as sp

                    posthoc = sp.posthoc_nemenyi_friedman(blocks[list(conditions)])
                    for left in conditions:
                        for right in conditions:
                            if left < right:
                                nemenyi_rows.append(
                                    {
                                        "family": family,
                                        "classifier": classifier,
                                        "metric": metric,
                                        "condition_a": left,
                                        "condition_b": right,
                                        "p_value": float(posthoc.loc[left, right]),
                                    }
                                )
                if "raw" in available:
                    for condition in conditions:
                        if condition == "raw" or condition not in available:
                            continue
                        pair_matrix = matrix[["raw", condition]].dropna()
                        row = _comparison_summary(
                            family,
                            classifier,
                            metric,
                            condition,
                            pair_matrix,
                        )
                        comparison_rows.append(row)
    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons["holm_p_value"] = np.nan
        for index in comparisons.groupby(["family", "classifier", "metric"]).groups.values():
            valid = comparisons.loc[index, "wilcoxon_p_value"].notna()
            values = comparisons.loc[index[valid], "wilcoxon_p_value"].astype(float).tolist()
            adjusted = _holm_adjust(values)
            comparisons.loc[index[valid], "holm_p_value"] = adjusted
    return pd.DataFrame(friedman_rows), pd.DataFrame(nemenyi_rows), comparisons


def _classwise_descriptive(class_deltas: pd.DataFrame) -> pd.DataFrame:
    if class_deltas.empty:
        return pd.DataFrame()
    return (
        class_deltas.groupby(
            ["family", "classifier", "condition", "class_index"], as_index=False
        )
        .agg(
            n_pairs=("delta", "size"),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            wins=("delta", lambda values: int((values > 1e-12).sum())),
            ties=("delta", lambda values: int((values.abs() <= 1e-12).sum())),
            losses=("delta", lambda values: int((values < -1e-12).sum())),
        )
    )


def _moderators(root: Path, dataset_ids: list[str]) -> pd.DataFrame:
    registry = pd.read_csv(root / "data" / "dataset_registry.csv", dtype=str, keep_default_na=False)
    selected = registry[registry["dataset_id"].isin(dataset_ids)].copy()
    columns = [
        "dataset_id",
        "n_rows",
        "n_classes",
        "n_min_class",
        "ir_majority_minority",
        "d_raw",
        "d_encoded",
        "feature_type",
    ]
    return selected[columns].sort_values("dataset_id").reset_index(drop=True)


def _class_distribution(root: Path, dataset_ids: list[str]) -> pd.DataFrame:
    """Expand registry class-count strings for the descriptive overview figure."""

    registry = pd.read_csv(root / "data" / "dataset_registry.csv", dtype=str, keep_default_na=False)
    selected = registry[registry["dataset_id"].isin(dataset_ids)]
    rows: list[dict[str, Any]] = []
    for _, record in selected.sort_values("dataset_id").iterrows():
        counts = str(record["class_counts"])
        parsed = [item.split(":", 1) for item in counts.split("|") if ":" in item]
        total = sum(int(value) for _, value in parsed)
        for class_index, (label, value) in enumerate(parsed):
            rows.append(
                {
                    "dataset_id": record["dataset_id"],
                    "class_index": class_index,
                    "class_label": label,
                    "proportion": int(value) / total,
                }
            )
    return pd.DataFrame(rows)


def _generate_figures(
    root: Path,
    cell_means: pd.DataFrame,
    comparisons: pd.DataFrame,
    friedman: pd.DataFrame,
    nemenyi: pd.DataFrame,
    figures_dir: Path,
) -> list[dict[str, str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [{"name": "figures", "status": "skipped: matplotlib is not installed"}]
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, str]] = []

    distribution = _class_distribution(root, sorted(cell_means["dataset"].unique()))
    if not distribution.empty:
        support = distribution.pivot_table(
            index="dataset_id", columns="class_index", values="proportion", fill_value=0.0
        )
        axis = support.plot(
            kind="bar",
            stacked=True,
            figsize=(14, 6),
            ylim=(0, 1),
            colormap="tab20",
            title="Class distribution by retained dataset",
        )
        axis.set_xlabel("Dataset")
        axis.set_ylabel("Class proportion")
        axis.legend(title="Class index", bbox_to_anchor=(1.02, 1), loc="upper left")
        axis.figure.tight_layout()
        path = figures_dir / "class_distribution_overview.png"
        axis.figure.savefig(path, dpi=160)
        plt.close(axis.figure)
        generated.append({"name": path.name, "status": "generated"})

    summary = (
        cell_means.groupby(["feature_type", "classifier", "condition"], as_index=False)["macro_f1"]
        .mean()
    )
    pivot = summary.pivot_table(
        index=["feature_type", "condition"], columns="classifier", values="macro_f1"
    )
    axis = pivot.plot(kind="bar", figsize=(14, 6), ylim=(0, 1), title="Complete-cell macro-F1 means")
    axis.set_xlabel("Feature type and condition")
    axis.set_ylabel("Mean macro-F1")
    axis.figure.tight_layout()
    path = figures_dir / "macro_f1_complete_cell_means.png"
    axis.figure.savefig(path, dpi=160)
    plt.close(axis.figure)
    generated.append({"name": path.name, "status": "generated"})

    heatmap = summary.pivot_table(
        index=["feature_type", "classifier"], columns="condition", values="macro_f1"
    )
    figure, axis = plt.subplots(figsize=(14, 5))
    image = axis.imshow(heatmap.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(heatmap.columns)), labels=list(heatmap.columns), rotation=45, ha="right")
    axis.set_yticks(
        range(len(heatmap.index)),
        labels=[f"{feature}/{classifier}" for feature, classifier in heatmap.index],
    )
    axis.set_title("Complete-cell macro-F1 ranking heatmap")
    axis.set_xlabel("Condition")
    axis.set_ylabel("Feature type / classifier")
    figure.colorbar(image, ax=axis, label="Mean macro-F1")
    figure.tight_layout()
    path = figures_dir / "macro_f1_ranking_heatmap.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append({"name": path.name, "status": "generated"})

    macro_deltas = comparisons[comparisons["metric"] == "macro_f1"]
    if not macro_deltas.empty:
        axis = macro_deltas.boxplot(column="mean_delta", by="condition", figsize=(12, 6), rot=45)
        axis.set_title("Paired macro-F1 deltas from raw")
        axis.set_xlabel("Condition")
        axis.set_ylabel("Mean dataset-level delta")
        axis.figure.suptitle("")
        axis.figure.tight_layout()
        path = figures_dir / "macro_f1_baseline_deltas.png"
        axis.figure.savefig(path, dpi=160)
        plt.close(axis.figure)
        generated.append({"name": path.name, "status": "generated"})

    significant = friedman[
        (friedman["status"] == "computed")
        & (pd.to_numeric(friedman["friedman_p_value"], errors="coerce") < 0.05)
    ]
    if significant.empty:
        generated.append(
            {"name": "critical_difference", "status": "skipped: no significant Friedman results"}
        )
    else:
        try:
            import scikit_posthocs as sp
        except ImportError:
            generated.append(
                {"name": "critical_difference", "status": "skipped: scikit-posthocs is not installed"}
            )
        else:
            for _, test in significant.iterrows():
                family = str(test["family"])
                classifier = str(test["classifier"])
                metric = str(test["metric"])
                matrix, available = _family_matrix(cell_means, family, classifier, metric)
                blocks = matrix[list(available)].dropna()
                if len(available) < 3 or len(blocks) < 3:
                    continue
                p_values = nemenyi[
                    (nemenyi["family"] == family)
                    & (nemenyi["classifier"] == classifier)
                    & (nemenyi["metric"] == metric)
                ]
                significance = pd.DataFrame(1.0, index=available, columns=available)
                for _, pair in p_values.iterrows():
                    left = str(pair["condition_a"])
                    right = str(pair["condition_b"])
                    if left in significance.index and right in significance.columns:
                        value = float(pair["p_value"])
                        significance.loc[left, right] = value
                        significance.loc[right, left] = value
                ranks = blocks.rank(axis=1, ascending=False, method="average").mean()
                figure, axis = plt.subplots(figsize=(13, 5))
                sp.critical_difference_diagram(
                    ranks,
                    significance,
                    alpha=0.05,
                    ax=axis,
                    left_only=True,
                )
                axis.set_title(f"Critical difference: {family} / {classifier} / {metric}")
                figure.tight_layout()
                path = figures_dir / f"critical_difference_{family}_{classifier}_{metric}.png"
                figure.savefig(path, dpi=160)
                plt.close(figure)
                generated.append({"name": path.name, "status": "generated"})
    return generated


def run_analysis(
    root: Path,
    run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run Gate D, aggregation, confirmatory tests, effects, and figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    gate_d_path = output_dir / "gate-d-review.json"
    gate_d = verify_gate_d(root, run_dir, gate_d_path)
    results = pd.read_csv(run_dir / "benchmark_results.csv")
    cell_means, class_means = load_cell_means(results)
    efficiency = load_efficiency_means(results)
    deltas, class_deltas = build_baseline_deltas(cell_means, class_means)
    friedman, nemenyi, comparisons = _run_confirmatory_tests(cell_means)
    moderators = _moderators(root, gate_d["dataset_ids"])
    class_summary = _classwise_descriptive(class_deltas)
    cell_means.to_csv(output_dir / "cell_means.csv", index=False)
    class_means.to_csv(output_dir / "class_means.csv", index=False)
    deltas.to_csv(output_dir / "baseline_deltas.csv", index=False)
    class_deltas.to_csv(output_dir / "baseline_deltas_per_class.csv", index=False)
    class_summary.to_csv(output_dir / "per_class_recall_summary.csv", index=False)
    friedman.to_csv(output_dir / "friedman_tests.csv", index=False)
    nemenyi.to_csv(output_dir / "nemenyi_pvalues.csv", index=False)
    comparisons.to_csv(output_dir / "wilcoxon_effects.csv", index=False)
    moderators.to_csv(output_dir / "moderators.csv", index=False)
    efficiency.to_csv(output_dir / "efficiency_means.csv", index=False)
    figures = _generate_figures(
        root,
        cell_means,
        comparisons,
        friedman,
        nemenyi,
        output_dir / "figures",
    )
    artifact_paths = [
        gate_d_path,
        output_dir / "cell_means.csv",
        output_dir / "class_means.csv",
        output_dir / "baseline_deltas.csv",
        output_dir / "baseline_deltas_per_class.csv",
        output_dir / "per_class_recall_summary.csv",
        output_dir / "friedman_tests.csv",
        output_dir / "nemenyi_pvalues.csv",
        output_dir / "wilcoxon_effects.csv",
        output_dir / "moderators.csv",
        output_dir / "efficiency_means.csv",
    ]
    artifact_refs = {
        _relative_path(path, root): _reference(path, root) for path in artifact_paths
    }
    for figure in figures:
        if figure["status"] == "generated":
            path = output_dir / "figures" / figure["name"]
            artifact_refs[_relative_path(path, root)] = _reference(path, root)
    manifest = {
        "schema_version": "analysis-run-v1",
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "gate_d": gate_d,
        "input_run_manifest": _reference(run_dir / "benchmark_manifest.json", root),
        "dataset_ids": gate_d["dataset_ids"],
        "cell_mean_count": len(cell_means),
        "class_mean_count": len(class_means),
        "baseline_delta_count": len(deltas),
        "per_class_delta_count": len(class_deltas),
        "efficiency_mean_count": len(efficiency),
        "friedman_test_count": len(friedman),
        "nemenyi_pair_count": len(nemenyi),
        "wilcoxon_comparison_count": len(comparisons),
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 0,
        "alpha": 0.05,
        "family_definitions": {
            family: {
                "feature_type": feature_type,
                "conditions": list(conditions),
                "classifiers": list(_family_classifiers(family)),
            }
            for family, (feature_type, conditions) in FAMILY_DEFINITIONS.items()
        },
        "procedures": {
            "friedman": "complete dataset blocks, at least three conditions and three blocks",
            "nemenyi": "all pairs after a significant Friedman test, alpha 0.05",
            "wilcoxon": {
                "comparison": "raw versus every alternative in the same family",
                "alternative": "two-sided",
                "zero_method": "wilcox",
                "correction": False,
                "method": "auto",
            },
            "holm": "within family, classifier, and metric across Wilcoxon p-values",
            "rank_biserial": "paired signed ranks after removing zero differences",
            "bootstrap": "paired percentile over complete dataset blocks",
        },
        "per_class_recall_inference": "descriptive class-wise summaries; inferential matrices skipped because class labels are dataset-specific",
        "figures": figures,
        "artifacts": artifact_refs,
    }
    _write_json_atomic(output_dir / "analysis_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and analyse the locked benchmark")
    parser.add_argument("--run-dir", type=Path, default=Path("results/full-run"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--gate-d-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if args.gate_d_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        verify_gate_d(root, run_dir, output_dir / "gate-d-review.json")
        print(output_dir / "gate-d-review.json")
        return
    run_analysis(root, run_dir, output_dir)
    print(output_dir / "analysis_manifest.json")


if __name__ == "__main__":
    main()
