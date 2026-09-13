"""Gated V2 nested execution and complete-cell accounting.

The runner is deliberately unable to start an outcome-bearing run until the
protocol, environment, and decision-register gates are explicitly ready.  A
dry run can still report the locked registry scope and expected cell matrix.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.v2_engineering import (
    STATUS_FAILED,
    STATUS_VALID,
    V2_CONDITIONS,
    V2_MODELS,
    evaluate_probabilities,
    failure_record,
    fit_threshold_offsets,
    infer_feature_layout,
)
from src.v2_pipeline import build_pipeline_v2
from src.v2_registry import RegistryValidation, validate_registry_file
from src.v2_tuning import TuningContract, load_tuning_contract

OUTER_FOLDS = 5
INNER_FOLDS = 3
OUTER_SEED = 20260913
INNER_SEED = 20260914
THRESHOLD_SEED = 20260916
WORKER_COUNT = 1

CELL_COLUMNS = ("dataset_id", "outer_fold", "outer_seed", "classifier", "condition")
RESULT_COLUMNS = (
    *CELL_COLUMNS,
    "feature_type",
    "selected_trial",
    "selected_params",
    "macro_average_precision",
    "per_class_average_precision",
    "macro_f1_argmax",
    "macro_f1_calibrated",
    "g_mean_argmax",
    "mcc_argmax",
    "balanced_accuracy_argmax",
    "per_class_recall_argmax",
    "multiclass_brier",
    "multiclass_log_loss",
    "threshold_offsets",
    "calibration_status",
    "calibration_reason",
    "rows_before_sampling",
    "rows_after_sampling",
    "wall_seconds",
    "fit_seconds",
    "predict_seconds",
    "peak_rss_mb",
    "worker_count",
    "timeout",
    "config_sha256",
    "status",
)
FAILURE_COLUMNS = (
    *CELL_COLUMNS,
    "feature_type",
    "stage",
    "exception_type",
    "reason",
    "applicability",
    "wall_seconds",
    "resource_status",
    "retry_count",
    "worker_count",
    "timeout",
    "config_sha256",
    "status",
)
TRIAL_COLUMNS = (
    *CELL_COLUMNS,
    "trial",
    "params",
    "status",
    "macro_average_precision",
    "wall_seconds",
    "exception_type",
    "reason",
)


@dataclass(frozen=True)
class V2DatasetSpec:
    dataset_id: str
    raw_path: Path
    raw_sha256: str
    target_column: str
    categorical_columns: tuple[str, ...]
    feature_type: str


@dataclass(frozen=True)
class V2ExecutionContext:
    root: Path
    config_path: Path
    protocol_path: Path
    registry_path: Path
    environment_path: Path
    output_dir: Path
    config: dict[str, Any]
    tuning: TuningContract
    registry_validation: RegistryValidation
    datasets: tuple[V2DatasetSpec, ...]
    config_sha256: str
    registry_sha256: str
    protocol_sha256: str

    @property
    def expected_cells_per_dataset(self) -> int:
        return OUTER_FOLDS * len(V2_MODELS) * len(V2_CONDITIONS)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def expected_cell_keys(dataset_id: str) -> set[tuple[Any, ...]]:
    """Return the complete V2 dataset/fold/model/condition cell matrix."""

    return {
        (dataset_id, fold, OUTER_SEED, classifier, condition)
        for fold, classifier, condition in itertools.product(
            range(OUTER_FOLDS), V2_MODELS, V2_CONDITIONS
        )
    }


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "pass"}


def _read_categories(overrides: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for row in overrides.to_dict(orient="records"):
        values = tuple(
            value.strip()
            for value in str(row.get("categorical_columns", "")).split(";")
            if value.strip()
        )
        result[str(row["dataset_id"])] = values
    return result


def load_execution_context(
    root: Path,
    *,
    config_path: Path | None = None,
    protocol_path: Path | None = None,
    registry_path: Path | None = None,
    overrides_path: Path | None = None,
    output_dir: Path | None = None,
) -> V2ExecutionContext:
    """Load score-free V2 inputs and construct a hashable execution context."""

    root = root.resolve()
    config_path = config_path or root / "config/v2.yaml"
    protocol_path = protocol_path or root / "docs/protocol-v2.md"
    registry_path = registry_path or root / "data/dataset_registry_v2.csv"
    overrides_path = overrides_path or root / "data/dataset_feature_overrides_v2.csv"
    output_dir = output_dir or root / "results/v2-full"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    registry_validation = validate_registry_file(registry_path)
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    overrides = pd.read_csv(overrides_path, dtype=str, keep_default_na=False)
    category_map = _read_categories(overrides)
    raw_dir = root / str(config["paths"]["raw_data"])
    datasets: list[V2DatasetSpec] = []
    for row in registry.to_dict(orient="records"):
        if not _truthy(row.get("eligible")):
            continue
        dataset_id = str(row["dataset_id"])
        raw_path = raw_dir / f"{dataset_id}.csv"
        datasets.append(
            V2DatasetSpec(
                dataset_id=dataset_id,
                raw_path=raw_path,
                raw_sha256=str(row["raw_sha256"]).lower(),
                target_column=str(row["target_column"]),
                categorical_columns=category_map.get(dataset_id, ()),
                feature_type=str(row["feature_type"]),
            )
        )
    return V2ExecutionContext(
        root=root,
        config_path=config_path,
        protocol_path=protocol_path,
        registry_path=registry_path,
        environment_path=root / str(config["paths"]["environment_record"]),
        output_dir=output_dir,
        config=config,
        tuning=load_tuning_contract(config_path),
        registry_validation=registry_validation,
        datasets=tuple(sorted(datasets, key=lambda item: item.dataset_id)),
        config_sha256=_sha256_file(config_path),
        registry_sha256=_sha256_file(registry_path),
        protocol_sha256=_sha256_file(protocol_path),
    )


def execution_blockers(context: V2ExecutionContext) -> tuple[str, ...]:
    """Return blockers that must be empty before an outcome-bearing run."""

    blockers: list[str] = []
    if context.config.get("status") != "full-run-ready":
        blockers.append("config status is not full-run-ready")
    protocol_text = context.protocol_path.read_text(encoding="utf-8").casefold()
    if "must lock before protocol freeze" in protocol_text or "if any item remains unresolved" in protocol_text:
        blockers.append("protocol decision register still contains unresolved must-lock language")
    if not context.environment_path.is_dir():
        blockers.append(f"environment record is missing: {_relative(context.environment_path, context.root)}")
    if context.registry_validation.status != "READY":
        blockers.extend(context.registry_validation.blockers)
    for spec in context.datasets:
        if not spec.raw_path.is_file():
            blockers.append(f"{spec.dataset_id}: raw input is missing")
        elif _sha256_file(spec.raw_path) != spec.raw_sha256:
            blockers.append(f"{spec.dataset_id}: raw input hash does not match registry")
    return tuple(dict.fromkeys(blockers))


def dry_run_summary(context: V2ExecutionContext) -> dict[str, Any]:
    """Return a non-outcome-bearing scope and gate summary."""

    blockers = execution_blockers(context)
    return {
        "status": "BLOCKED" if blockers else "READY",
        "dataset_count": len(context.datasets),
        "dataset_ids": [spec.dataset_id for spec in context.datasets],
        "expected_cells_per_dataset": context.expected_cells_per_dataset,
        "expected_cells": context.expected_cells_per_dataset * len(context.datasets),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "hpo_trials": context.tuning.trials,
        "config_sha256": context.config_sha256,
        "registry_sha256": context.registry_sha256,
        "blockers": list(blockers),
        "outcome_bearing_run_started": False,
    }


def _applicability(layout_feature_type: str, classifier: str, condition: str) -> tuple[bool, str]:
    if condition == "balanced_random_forest" and classifier != "random_forest":
        return False, "Balanced Random Forest is an RF-only comparator"
    if condition == "smotenc" and layout_feature_type != "mixed":
        return False, "SMOTENC requires mixed features"
    if condition in {"smote", "adasyn", "borderline_smote"} and layout_feature_type != "numeric":
        return False, f"{condition} is numeric-only in the frozen matrix"
    return True, "applicable"


def _distribution(spec: Any) -> Any:
    if not isinstance(spec, dict) or "distribution" not in spec:
        return [spec]
    distribution = str(spec["distribution"])
    if distribution == "loguniform":
        return loguniform(float(spec["low"]), float(spec["high"]))
    if distribution == "randint":
        return randint(int(spec["low"]), int(spec["high"]))
    if distribution == "uniform":
        bounds = [float(value) for value in spec["bounds"]]
        return uniform(bounds[0], bounds[1] - bounds[0])
    if distribution == "categorical":
        return list(spec["values"])
    raise ValueError(f"unsupported HPO distribution: {distribution}")


def _trial_parameters(contract: TuningContract, classifier: str, condition: str) -> list[dict[str, Any]]:
    model = "balanced_random_forest" if condition == "balanced_random_forest" else classifier
    space = {key: _distribution(value) for key, value in contract.spaces[model].items()}
    return [dict(params) for params in ParameterSampler(space, n_iter=contract.trials, random_state=contract.random_state)]


def _pipeline(
    layout: Any,
    classifier: str,
    condition: str,
    params: dict[str, Any],
    seed: int,
) -> Any:
    actual_classifier = "random_forest" if condition == "balanced_random_forest" else classifier
    return build_pipeline_v2(
        layout,
        condition,
        actual_classifier,
        random_state=seed,
        model_params=params,
    )


def _inner_score(
    X: pd.DataFrame,
    y: np.ndarray,
    layout: Any,
    classifier: str,
    condition: str,
    params: dict[str, Any],
    seed: int,
) -> float:
    splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=INNER_SEED)
    scores: list[float] = []
    for train_index, validation_index in splitter.split(X, y):
        model = _pipeline(layout, classifier, condition, params, seed)
        model.fit(X.iloc[train_index], y[train_index])
        probabilities = model.predict_proba(X.iloc[validation_index])
        scores.append(
            float(
                evaluate_probabilities(
                    y[validation_index],
                    probabilities,
                    list(range(len(np.unique(y)))),
                )["macro_average_precision"]
            )
        )
    return float(np.mean(scores))


def _inner_oof_probabilities(
    X: pd.DataFrame,
    y: np.ndarray,
    layout: Any,
    classifier: str,
    condition: str,
    params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    class_count = len(np.unique(y))
    probabilities = np.full((len(y), class_count), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=INNER_SEED)
    for train_index, validation_index in splitter.split(X, y):
        model = _pipeline(layout, classifier, condition, params, seed)
        model.fit(X.iloc[train_index], y[train_index])
        probabilities[validation_index] = model.predict_proba(X.iloc[validation_index])
    if not np.isfinite(probabilities).all():
        raise RuntimeError("inner out-of-fold probabilities are incomplete")
    return probabilities


def _cell_failure(
    *,
    dataset_id: str,
    outer_fold: int,
    classifier: str,
    condition: str,
    feature_type: str,
    stage: str,
    error: BaseException,
    applicability: str,
    wall_seconds: float,
    config_sha256: str,
) -> dict[str, Any]:
    record = failure_record(
        dataset_id=dataset_id,
        outer_fold=outer_fold,
        outer_seed=OUTER_SEED,
        classifier=classifier,
        condition=condition,
        stage=stage,
        exception_type=type(error).__name__,
        reason=str(error),
        applicability=applicability,
        wall_seconds=wall_seconds,
        resource_status="not_started" if applicability == "NOT_APPLICABLE" else "failed",
        retry_count=0,
        config_sha256=config_sha256,
    )
    record.update(
        {
            "feature_type": feature_type,
            "worker_count": WORKER_COUNT,
            "timeout": False,
        }
    )
    return record


def _run_cell(
    *,
    context: V2ExecutionContext,
    spec: V2DatasetSpec,
    layout: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    outer_fold: int,
    classifier: str,
    condition: str,
    trial_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    started = time.perf_counter()
    applicable, reason = _applicability(layout.feature_type, classifier, condition)
    if not applicable:
        return None, _cell_failure(
            dataset_id=spec.dataset_id,
            outer_fold=outer_fold,
            classifier=classifier,
            condition=condition,
            feature_type=layout.feature_type,
            stage="applicability",
            error=ValueError(reason),
            applicability="NOT_APPLICABLE",
            wall_seconds=0.0,
            config_sha256=context.config_sha256,
        )

    best_score = -math.inf
    best_trial = 0
    best_params: dict[str, Any] | None = None
    for trial_number, params in enumerate(
        _trial_parameters(context.tuning, classifier, condition), start=1
    ):
        trial_started = time.perf_counter()
        try:
            score = _inner_score(
                X_train,
                y_train,
                layout,
                classifier,
                condition,
                params,
                OUTER_SEED,
            )
            trial_rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "outer_fold": outer_fold,
                    "outer_seed": OUTER_SEED,
                    "classifier": classifier,
                    "condition": condition,
                    "trial": trial_number,
                    "params": json.dumps(params, sort_keys=True, default=str),
                    "status": STATUS_VALID,
                    "macro_average_precision": score,
                    "wall_seconds": time.perf_counter() - trial_started,
                    "exception_type": "",
                    "reason": "",
                }
            )
            if score > best_score:
                best_score = score
                best_trial = trial_number
                best_params = params
        except Exception as error:  # noqa: BLE001 - trial failures remain auditable
            trial_rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "outer_fold": outer_fold,
                    "outer_seed": OUTER_SEED,
                    "classifier": classifier,
                    "condition": condition,
                    "trial": trial_number,
                    "params": json.dumps(params, sort_keys=True, default=str),
                    "status": STATUS_FAILED,
                    "macro_average_precision": None,
                    "wall_seconds": time.perf_counter() - trial_started,
                    "exception_type": type(error).__name__,
                    "reason": str(error),
                }
            )

    if best_params is None:
        return None, _cell_failure(
            dataset_id=spec.dataset_id,
            outer_fold=outer_fold,
            classifier=classifier,
            condition=condition,
            feature_type=layout.feature_type,
            stage="inner_hpo",
            error=RuntimeError("all HPO trials failed"),
            applicability="APPLICABLE",
            wall_seconds=time.perf_counter() - started,
            config_sha256=context.config_sha256,
        )

    calibration_status = STATUS_VALID
    calibration_reason = ""
    offsets = tuple(0.0 for _ in np.unique(y_train))
    try:
        inner_probabilities = _inner_oof_probabilities(
            X_train,
            y_train,
            layout,
            classifier,
            condition,
            best_params,
            OUTER_SEED,
        )
        threshold_fit = fit_threshold_offsets(
            y_train,
            inner_probabilities,
            list(range(len(np.unique(y_train)))),
            seed=THRESHOLD_SEED,
        )
        offsets = threshold_fit.offsets
        calibration_status = threshold_fit.status
        calibration_reason = threshold_fit.reason or ""
    except Exception as error:  # noqa: BLE001 - protocol fallback is recorded in result
        calibration_status = STATUS_FAILED
        calibration_reason = f"threshold fallback: {type(error).__name__}: {error}"

    try:
        model = _pipeline(layout, classifier, condition, best_params, OUTER_SEED)
        fit_started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        probabilities = model.predict_proba(X_test)
        predict_seconds = time.perf_counter() - predict_started
        metrics = evaluate_probabilities(
            y_test,
            probabilities,
            list(range(len(np.unique(y_train)))),
            threshold_offsets=offsets,
        )
        rows_after = len(y_train)
        if hasattr(model, "rows_after_"):
            rows_after = int(model.rows_after_)
        elif hasattr(model, "named_steps") and "sampler" in model.named_steps:
            rows_after = int(model.named_steps["sampler"].rows_after_)
        return (
            {
                "dataset_id": spec.dataset_id,
                "outer_fold": outer_fold,
                "outer_seed": OUTER_SEED,
                "classifier": classifier,
                "condition": condition,
                "feature_type": layout.feature_type,
                "selected_trial": best_trial,
                "selected_params": json.dumps(best_params, sort_keys=True, default=str),
                **metrics,
                "threshold_offsets": json.dumps(offsets),
                "calibration_status": calibration_status,
                "calibration_reason": calibration_reason,
                "rows_before_sampling": len(y_train),
                "rows_after_sampling": rows_after,
                "wall_seconds": time.perf_counter() - started,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "peak_rss_mb": None,
                "worker_count": WORKER_COUNT,
                "timeout": False,
                "config_sha256": context.config_sha256,
                "status": STATUS_VALID,
            },
            None,
        )
    except Exception as error:  # noqa: BLE001 - cell failure is retained explicitly
        return None, _cell_failure(
            dataset_id=spec.dataset_id,
            outer_fold=outer_fold,
            classifier=classifier,
            condition=condition,
            feature_type=layout.feature_type,
            stage="outer_fit_predict",
            error=error,
            applicability="APPLICABLE",
            wall_seconds=time.perf_counter() - started,
            config_sha256=context.config_sha256,
        )


def validate_v2_output_frames(
    dataset_id: str,
    results: pd.DataFrame,
    failures: pd.DataFrame,
) -> dict[str, int]:
    """Require exactly one schema-complete record for every V2 cell."""

    missing_results = sorted(set(RESULT_COLUMNS) - set(results.columns))
    missing_failures = sorted(set(FAILURE_COLUMNS) - set(failures.columns))
    if missing_results:
        raise RuntimeError(f"{dataset_id} results are missing columns: {missing_results}")
    if missing_failures:
        raise RuntimeError(f"{dataset_id} failures are missing columns: {missing_failures}")
    expected = expected_cell_keys(dataset_id)

    def keys(frame: pd.DataFrame, label: str) -> set[tuple[Any, ...]]:
        values = list(frame[list(CELL_COLUMNS)].itertuples(index=False, name=None))
        if len(values) != len(set(values)):
            raise RuntimeError(f"{dataset_id} {label} contains duplicate cell keys")
        return set(values)

    result_keys = keys(results, "results")
    failure_keys = keys(failures, "failures")
    if result_keys & failure_keys:
        raise RuntimeError(f"{dataset_id} has a cell recorded as both valid and failed")
    if result_keys | failure_keys != expected:
        raise RuntimeError(f"{dataset_id} V2 cell accounting does not match the expected matrix")
    if set(results["status"]) != {STATUS_VALID} and len(results):
        raise RuntimeError(f"{dataset_id} valid results contain a non-VALID status")
    if set(failures["status"]) != {STATUS_FAILED} and len(failures):
        raise RuntimeError(f"{dataset_id} failures contain a non-failure status")
    for column in ("macro_average_precision", "multiclass_brier", "multiclass_log_loss"):
        if results[column].isna().any() or not np.isfinite(results[column].astype(float)).all():
            raise RuntimeError(f"{dataset_id} valid results contain a missing/non-finite {column}")
    return {
        "expected_cells": len(expected),
        "valid_cells": len(result_keys),
        "failure_cells": len(failure_keys),
    }


def _load_dataset(spec: V2DatasetSpec) -> tuple[pd.DataFrame, np.ndarray, Any]:
    if not spec.raw_path.is_file():
        raise FileNotFoundError(spec.raw_path)
    frame = pd.read_csv(spec.raw_path)
    if spec.target_column not in frame.columns:
        raise ValueError(f"{spec.dataset_id}: target column is absent")
    frame = frame.dropna(subset=[spec.target_column]).reset_index(drop=True)
    X = frame.drop(columns=[spec.target_column])
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[spec.target_column].astype(str))
    layout = infer_feature_layout(frame, spec.target_column, spec.categorical_columns)
    if layout.feature_type != spec.feature_type:
        raise ValueError(
            f"{spec.dataset_id}: registry feature type {spec.feature_type} disagrees with loaded frame {layout.feature_type}"
        )
    return X, y, layout


def _run_dataset(context: V2ExecutionContext, spec: V2DatasetSpec, dataset_dir: Path) -> dict[str, int]:
    X, y, layout = _load_dataset(spec)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED)
    for outer_fold, (train_index, test_index) in enumerate(splitter.split(X, y)):
        for classifier, condition in itertools.product(V2_MODELS, V2_CONDITIONS):
            result, failure = _run_cell(
                context=context,
                spec=spec,
                layout=layout,
                X_train=X.iloc[train_index],
                y_train=y[train_index],
                X_test=X.iloc[test_index],
                y_test=y[test_index],
                outer_fold=outer_fold,
                classifier=classifier,
                condition=condition,
                trial_rows=trials,
            )
            if result is not None:
                results.append(result)
            if failure is not None:
                failures.append(failure)
    result_frame = pd.DataFrame(results, columns=RESULT_COLUMNS)
    failure_frame = pd.DataFrame(failures, columns=FAILURE_COLUMNS)
    counts = validate_v2_output_frames(spec.dataset_id, result_frame, failure_frame)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(dataset_dir / "v2_results.csv", index=False)
    failure_frame.to_csv(dataset_dir / "v2_failures.csv", index=False)
    pd.DataFrame(trials, columns=TRIAL_COLUMNS).to_csv(dataset_dir / "v2_trials.csv", index=False)
    marker = {
        "schema_version": "v2-dataset-run-v1",
        "status": "complete",
        "dataset_id": spec.dataset_id,
        "raw_sha256": spec.raw_sha256,
        "config_sha256": context.config_sha256,
        "registry_sha256": context.registry_sha256,
        "counts": counts,
        "completed_at_utc": _utc_now(),
    }
    _write_json_atomic(dataset_dir / "v2_complete.json", marker)
    return counts


def run_v2_benchmark(
    context: V2ExecutionContext,
    *,
    selected_dataset_ids: tuple[str, ...] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or plan the V2 benchmark, with a hard pre-outcome gate."""

    selected = selected_dataset_ids or tuple(spec.dataset_id for spec in context.datasets)
    known = {spec.dataset_id for spec in context.datasets}
    unknown = sorted(set(selected) - known)
    if unknown:
        raise ValueError(f"requested dataset(s) are outside the eligible registry: {unknown}")
    summary = dry_run_summary(context)
    if dry_run:
        return summary
    blockers = tuple(summary["blockers"])
    if blockers:
        raise RuntimeError(f"full V2 execution is blocked: {'; '.join(blockers)}")

    context.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = context.output_dir / "v2_manifest.json"
    manifest = {
        "schema_version": "v2-full-run-v1",
        "status": "running",
        "created_at_utc": _utc_now(),
        "config_sha256": context.config_sha256,
        "registry_sha256": context.registry_sha256,
        "protocol_sha256": context.protocol_sha256,
        "dataset_ids": list(selected),
        "expected_cells_per_dataset": context.expected_cells_per_dataset,
        "dataset_runs": {},
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("schema_version", "config_sha256", "registry_sha256", "protocol_sha256", "dataset_ids"):
            if existing.get(field) != manifest[field]:
                raise RuntimeError(f"frozen V2 manifest field changed: {field}")
        manifest = existing
    for spec in context.datasets:
        if spec.dataset_id not in selected:
            continue
        dataset_dir = context.output_dir / spec.dataset_id
        marker_path = dataset_dir / "v2_complete.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("raw_sha256") != spec.raw_sha256 or marker.get("config_sha256") != context.config_sha256:
                raise RuntimeError(f"{spec.dataset_id}: completion marker is stale")
            manifest["dataset_runs"][spec.dataset_id] = marker
            _write_json_atomic(manifest_path, manifest)
            continue
        counts = _run_dataset(context, spec, dataset_dir)
        manifest["dataset_runs"][spec.dataset_id] = {"status": "complete", "counts": counts}
        _write_json_atomic(manifest_path, manifest)
    manifest["status"] = "complete" if len(manifest["dataset_runs"]) == len(selected) else "partial"
    manifest["completed_at_utc"] = _utc_now()
    _write_json_atomic(manifest_path, manifest)
    return manifest
