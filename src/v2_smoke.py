"""Synthetic Gate 4 smoke runner excluded from scientific evidence."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.v2_engineering import (
    STATUS_VALID,
    V2_CONDITIONS,
    V2_MODELS,
    evaluate_probabilities,
    failure_record,
    infer_feature_layout,
    validate_cell_accounting,
)
from src.v2_pipeline import build_pipeline_v2

SMOKE_SEED = 20260913
SMOKE_MODELS = V2_MODELS
SMOKE_CONDITIONS = V2_CONDITIONS
SMOKE_RESULT_COLUMNS = (
    "dataset_id",
    "outer_fold",
    "outer_seed",
    "classifier",
    "condition",
    "selected_trial",
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
    "rows_before_sampling",
    "rows_after_sampling",
    "wall_seconds",
    "fit_seconds",
    "predict_seconds",
    "peak_rss_mb",
    "config_sha256",
    "status",
)
SMOKE_FAILURE_COLUMNS = (
    "dataset_id",
    "outer_fold",
    "outer_seed",
    "classifier",
    "condition",
    "stage",
    "exception_type",
    "reason",
    "applicability",
    "wall_seconds",
    "resource_status",
    "retry_count",
    "config_sha256",
    "status",
)


def _fixtures() -> dict[str, pd.DataFrame]:
    labels = np.repeat([0, 1, 2], [30, 16, 10])
    numeric = pd.DataFrame(
        {
            "x1": np.arange(len(labels), dtype=float),
            "x2": np.sin(np.arange(len(labels))),
            "target": labels,
        }
    )
    mixed = pd.DataFrame(
        {
            "x1": np.arange(len(labels), dtype=float),
            "category": np.tile(["a", "b", "c"], len(labels) // 3 + 1)[: len(labels)],
            "target": labels,
        }
    )
    return {"synthetic_numeric": numeric, "synthetic_mixed": mixed}


def _applicability(feature_type: str, classifier: str, condition: str) -> tuple[bool, str]:
    if condition == "balanced_random_forest" and classifier != "random_forest":
        return False, "Balanced Random Forest is an RF-only comparator"
    if condition == "smotenc" and feature_type != "mixed":
        return False, "SMOTENC requires mixed features"
    if condition in {"smote", "adasyn", "borderline_smote"} and feature_type != "numeric":
        return False, f"{condition} is numeric-only in the frozen matrix"
    return True, "applicable"


def _config_hash(root: Path) -> str:
    return hashlib.sha256((root / "config" / "v2.yaml").read_bytes()).hexdigest()


def run_smoke(root: Path, output_dir: Path) -> dict[str, Any]:
    """Run one outer fold per synthetic fixture and write explicit artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = _config_hash(root)
    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    expected_keys: list[tuple[str, int, int, str, str]] = []
    for dataset_id, frame in _fixtures().items():
        layout = infer_feature_layout(frame, "target")
        X = frame.drop(columns=["target"])
        y = frame["target"]
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SMOKE_SEED)
        train_index, test_index = next(splitter.split(X, y))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        for classifier in SMOKE_MODELS:
            for condition in SMOKE_CONDITIONS:
                key = (dataset_id, 0, SMOKE_SEED, classifier, condition)
                expected_keys.append(key)
                applicable, applicability_reason = _applicability(
                    layout.feature_type, classifier, condition
                )
                started = time.perf_counter()
                if not applicable:
                    failure_rows.append(
                        failure_record(
                            dataset_id=dataset_id,
                            outer_fold=0,
                            outer_seed=SMOKE_SEED,
                            classifier=classifier,
                            condition=condition,
                            stage="applicability",
                            exception_type="NotApplicableError",
                            reason=applicability_reason,
                            applicability="NOT_APPLICABLE",
                            wall_seconds=0.0,
                            resource_status="smoke_within_limit",
                            retry_count=0,
                            config_sha256=config_sha256,
                        )
                    )
                    continue
                try:
                    pipeline = build_pipeline_v2(
                        layout,
                        condition,
                        classifier,
                        random_state=SMOKE_SEED,
                        n_estimators=8,
                    )
                    fit_started = time.perf_counter()
                    pipeline.fit(X_train, y_train)
                    fit_seconds = time.perf_counter() - fit_started
                    predict_started = time.perf_counter()
                    probabilities = pipeline.predict_proba(X_test)
                    predict_seconds = time.perf_counter() - predict_started
                    metrics = evaluate_probabilities(
                        y_test.to_numpy(),
                        probabilities,
                        [0, 1, 2],
                        threshold_offsets=[0.0, 0.0, 0.0],
                    )
                    rows_after_sampling = len(y_train)
                    if hasattr(pipeline, "rows_after_"):
                        rows_after_sampling = pipeline.rows_after_
                    elif hasattr(pipeline, "named_steps") and "sampler" in pipeline.named_steps:
                        rows_after_sampling = pipeline.named_steps["sampler"].rows_after_
                    result_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "outer_fold": 0,
                            "outer_seed": SMOKE_SEED,
                            "classifier": classifier,
                            "condition": condition,
                            "selected_trial": 0,
                            **metrics,
                            "threshold_offsets": json.dumps([0.0, 0.0, 0.0]),
                            "rows_before_sampling": len(y_train),
                            "rows_after_sampling": rows_after_sampling,
                            "wall_seconds": time.perf_counter() - started,
                            "fit_seconds": fit_seconds,
                            "predict_seconds": predict_seconds,
                            "peak_rss_mb": None,
                            "config_sha256": config_sha256,
                            "status": STATUS_VALID,
                        }
                    )
                except Exception as error:  # noqa: BLE001 - smoke records are evidence
                    failure_rows.append(
                        failure_record(
                            dataset_id=dataset_id,
                            outer_fold=0,
                            outer_seed=SMOKE_SEED,
                            classifier=classifier,
                            condition=condition,
                            stage="fit_predict",
                            exception_type=type(error).__name__,
                            reason=str(error),
                            applicability="APPLICABLE",
                            wall_seconds=time.perf_counter() - started,
                            resource_status="smoke_within_limit",
                            retry_count=0,
                            config_sha256=config_sha256,
                        )
                    )

    result_frame = pd.DataFrame(result_rows, columns=SMOKE_RESULT_COLUMNS)
    failure_frame = pd.DataFrame(failure_rows, columns=SMOKE_FAILURE_COLUMNS)
    counts = validate_cell_accounting(
        expected_keys,
        [tuple(row) for row in result_frame[["dataset_id", "outer_fold", "outer_seed", "classifier", "condition"]].itertuples(index=False, name=None)],
        [tuple(row) for row in failure_frame[["dataset_id", "outer_fold", "outer_seed", "classifier", "condition"]].itertuples(index=False, name=None)],
    )
    result_frame.to_csv(output_dir / "smoke_results.csv", index=False)
    failure_frame.to_csv(output_dir / "smoke_failures.csv", index=False)
    manifest = {
        "schema_version": "v2-smoke-v1",
        "status": "complete",
        "excluded_from_scientific_evidence": True,
        "config_sha256": config_sha256,
        "datasets": list(_fixtures()),
        "classifiers": list(SMOKE_MODELS),
        "conditions": list(SMOKE_CONDITIONS),
        "hpo_trials_contract": 20,
        "outer_fold_count": 1,
        "counts": counts,
        "valid_cells": len(result_frame),
        "failure_cells": len(failure_frame),
    }
    (output_dir / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
