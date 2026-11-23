"""Validate and replay the aggregate evidence included with V1 publications.

This route verifies the frozen aggregate records and their derived analysis. It
does not verify raw dataset bytes, dataset completion markers, or model runs, so
it is deliberately separate from Gate D and Gate E.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import (
    UNSUPPORTED_ERROR_MESSAGES,
    run_analysis,
)
from src.benchmark import FAILURE_COLUMNS, RESULT_COLUMNS

EXPECTED_COUNTS = {"expected_cells": 5445, "valid_cells": 4191, "failure_cells": 1254}
ANALYSIS_TABLES = (
    "cell_means.csv",
    "class_means.csv",
    "efficiency_means.csv",
    "baseline_deltas.csv",
    "baseline_deltas_per_class.csv",
    "per_class_recall_summary.csv",
    "friedman_tests.csv",
    "nemenyi_pvalues.csv",
    "wilcoxon_effects.csv",
    "moderators.csv",
    "failure_accounting.csv",
    "failure_reason_counts.csv",
    "macro_f1_figure_coverage.csv",
    "friedman_permutation_sensitivity.csv",
    "exact_sign_flip_sensitivity.csv",
    "adasyn_excluded_friedman.csv",
    "adasyn_excluded_pairwise.csv",
    "adasyn_excluded_imbalance_descriptive.csv",
)
RESULT_SCHEMA = (
    *RESULT_COLUMNS,
    "fit_predict_seconds",
    "rss_before_mb",
    "rss_after_mb",
    "rss_delta_mb",
    "train_rows_before_sampling",
    "train_rows_after_sampling",
)
FAILURE_SCHEMA = (*FAILURE_COLUMNS, "elapsed_seconds")
KEY_COLUMNS = ("dataset", "seed", "fold", "classifier", "condition")
GENERATED_FIGURE_COUNT = 8


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_reference(root: Path, reference: object, label: str) -> Path:
    if not isinstance(reference, dict):
        raise TypeError(f"{label} reference is not an object")
    relative = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise TypeError(f"{label} reference is incomplete")
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise RuntimeError(f"{label} reference has an invalid SHA-256")
    candidate = Path(relative.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"{label} reference escapes the evidence root: {relative}")
    path = (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} reference escapes the evidence root: {relative}") from error
    if not path.is_file():
        raise RuntimeError(f"{label} file is missing: {relative}")
    if _sha256(path) != expected_hash:
        raise RuntimeError(f"{label} SHA-256 does not match its manifest: {relative}")
    return path


def _numeric(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    if column not in frame:
        raise RuntimeError(f"{label} is missing column {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label}.{column} contains a missing or non-finite value")
    frame[column] = values
    return values


def _integer_column(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    values = _numeric(frame, column, label)
    if not np.equal(values, np.floor(values)).all():
        raise RuntimeError(f"{label}.{column} must contain integers")
    frame[column] = values.astype(int)
    return values.astype(int)


def _recall_values(value: object, label: str) -> np.ndarray:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} contains invalid per_class_recall JSON") from error
    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError(f"{label} per_class_recall must be a non-empty list")
    try:
        recalls = np.asarray(parsed, dtype=float)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} per_class_recall must contain numbers") from error
    if recalls.ndim != 1 or not np.isfinite(recalls).all():
        raise RuntimeError(f"{label} per_class_recall contains invalid numeric values")
    if ((recalls < 0) | (recalls > 1)).any():
        raise RuntimeError(f"{label} per_class_recall values must be in [0, 1]")
    return recalls


def _expected_unsupported_error(
    dataset: str,
    classifier: str,
    condition: str,
    feature_types: dict[str, str],
    applicability: dict[str, set[str]],
) -> str | None:
    if condition == "balanced_random_forest" and classifier != "random_forest":
        return "Balanced Random Forest is an RF-only comparator"
    feature_type = feature_types[dataset]
    if condition not in applicability[dataset]:
        return UNSUPPORTED_ERROR_MESSAGES.get((feature_type, condition))
    return None


def validate_aggregate_records(
    results: pd.DataFrame,
    failures: pd.DataFrame,
    *,
    dataset_ids: list[str],
    seeds: list[int],
    folds: int,
    classifiers: list[str],
    conditions: list[str],
    feature_types: dict[str, str],
    class_counts: dict[str, int],
    applicability: dict[str, set[str]],
) -> dict[str, int]:
    """Validate schemas, metrics, supported cells, and exact aggregate keys."""

    if tuple(results.columns) != RESULT_SCHEMA:
        raise RuntimeError("benchmark results have an unexpected column schema")
    if tuple(failures.columns) != FAILURE_SCHEMA:
        raise RuntimeError("benchmark failures have an unexpected column schema")
    if not dataset_ids or len(dataset_ids) != len(set(dataset_ids)):
        raise RuntimeError("dataset IDs must be non-empty and unique")
    for mapping, label in (
        (feature_types, "feature types"),
        (class_counts, "class counts"),
        (applicability, "applicability"),
    ):
        if set(mapping) != set(dataset_ids):
            raise RuntimeError(f"{label} do not cover exactly the retained datasets")

    for frame, label in ((results, "results"), (failures, "failures")):
        for column in KEY_COLUMNS:
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise RuntimeError(f"{label}.{column} contains an empty key value")
        _integer_column(frame, "seed", label)
        _integer_column(frame, "fold", label)
        if frame[list(KEY_COLUMNS)].duplicated().any():
            raise RuntimeError(f"{label} contains duplicate cell keys")
        if not set(frame["dataset"].astype(str)) <= set(dataset_ids):
            raise RuntimeError(f"{label} contains a dataset outside the retained scope")
        for dataset, feature_type in frame[["dataset", "feature_type"]].drop_duplicates().itertuples(
            index=False, name=None
        ):
            if feature_type != feature_types[str(dataset)]:
                raise RuntimeError(f"{label} feature type disagrees with the registry for {dataset}")

    result_keys = set(results[list(KEY_COLUMNS)].itertuples(index=False, name=None))
    failure_keys = set(failures[list(KEY_COLUMNS)].itertuples(index=False, name=None))
    overlap = result_keys & failure_keys
    if overlap:
        raise RuntimeError(f"valid and failure records overlap at {len(overlap)} cell keys")
    expected_keys = set(
        itertools.product(dataset_ids, seeds, range(folds), classifiers, conditions)
    )
    if result_keys | failure_keys != expected_keys:
        missing = expected_keys - (result_keys | failure_keys)
        extra = (result_keys | failure_keys) - expected_keys
        raise RuntimeError(
            f"aggregate key coverage is incomplete: missing={len(missing)}, extra={len(extra)}"
        )

    for column in ("macro_f1", "g_mean", "balanced_accuracy"):
        values = _numeric(results, column, "results")
        if ((values < 0) | (values > 1)).any():
            raise RuntimeError(f"results.{column} must be in [0, 1]")
    mcc = _numeric(results, "mcc", "results")
    if ((mcc < -1) | (mcc > 1)).any():
        raise RuntimeError("results.mcc must be in [-1, 1]")
    for column in ("fit_predict_seconds", "rss_before_mb", "rss_after_mb"):
        if (_numeric(results, column, "results") < 0).any():
            raise RuntimeError(f"results.{column} must be non-negative")
    rss_delta = _numeric(results, "rss_delta_mb", "results")
    if not np.allclose(
        rss_delta,
        results["rss_after_mb"].to_numpy(dtype=float)
        - results["rss_before_mb"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-9,
    ):
        raise RuntimeError("results.rss_delta_mb disagrees with the endpoint RSS values")
    for column in ("train_rows_before_sampling", "train_rows_after_sampling"):
        if (_integer_column(results, column, "results") <= 0).any():
            raise RuntimeError(f"results.{column} must be a positive integer")

    recalls_by_row: list[np.ndarray] = []
    for row in results[["dataset", "per_class_recall"]].itertuples(index=False, name=None):
        dataset, value = str(row[0]), row[1]
        recalls = _recall_values(value, dataset)
        if len(recalls) != class_counts[dataset]:
            raise RuntimeError(f"per_class_recall length disagrees with class count for {dataset}")
        recalls_by_row.append(recalls)
    if recalls_by_row:
        expected_balanced = np.asarray([recall.mean() for recall in recalls_by_row])
        expected_g_mean = np.asarray(
            [0.0 if (recall == 0).any() else math.exp(np.log(recall).mean()) for recall in recalls_by_row]
        )
        if not np.allclose(
            results["balanced_accuracy"].to_numpy(dtype=float), expected_balanced, rtol=0, atol=1e-12
        ):
            raise RuntimeError("balanced_accuracy disagrees with per-class recall")
        if not np.allclose(
            results["g_mean"].to_numpy(dtype=float), expected_g_mean, rtol=0, atol=1e-12
        ):
            raise RuntimeError("g_mean disagrees with per-class recall")

    elapsed = _numeric(failures, "elapsed_seconds", "failures")
    if (elapsed < 0).any():
        raise RuntimeError("failures.elapsed_seconds must be non-negative")
    for column in ("stage", "error_type", "error"):
        if failures[column].isna().any() or failures[column].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"failures.{column} contains an empty classification value")

    failure_by_key = {
        tuple(row[column] for column in KEY_COLUMNS): row
        for _, row in failures.iterrows()
    }
    for dataset, seed, fold, classifier, condition in expected_keys:
        message = _expected_unsupported_error(
            dataset, classifier, condition, feature_types, applicability
        )
        key = (dataset, seed, fold, classifier, condition)
        if message is None:
            continue
        if key in result_keys or key not in failure_by_key:
            raise RuntimeError(f"unsupported cell is not recorded as a failure: {key}")
        failure = failure_by_key[key]
        if failure["error_type"] != "ValueError" or failure["error"] != message:
            raise RuntimeError(f"unsupported failure classification is inconsistent for {key}")

    return {
        "expected_cells": len(expected_keys),
        "valid_cells": len(result_keys),
        "failure_cells": len(failure_keys),
    }


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        return pd.read_csv(path, dtype={"dataset": str, "classifier": str, "condition": str})
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise RuntimeError(f"{label} cannot be read: {path}") from error


def _same_counts(actual: object, expected: dict[str, int], label: str) -> None:
    if not isinstance(actual, dict):
        raise TypeError(f"{label} counts are missing")
    observed = {key: actual.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"{label} counts disagree with the aggregate records")


def validate_publication_evidence(root: Path) -> dict[str, Any]:
    """Validate the packaged aggregate records and frozen analysis artifacts."""

    root = root.resolve()
    run_dir = root / "results" / "full-run"
    analysis_dir = root / "results" / "analysis"
    benchmark_path = run_dir / "benchmark_manifest.json"
    gate_d_path = analysis_dir / "gate-d-review.json"
    analysis_path = analysis_dir / "analysis_manifest.json"
    scope_path = root / "artifacts" / "runs" / "scope-lock.json"
    benchmark = _read_json(benchmark_path, "benchmark manifest")
    gate_d = _read_json(gate_d_path, "stored Gate D receipt")
    analysis = _read_json(analysis_path, "analysis manifest")
    scope = _read_json(scope_path, "scope lock")

    if benchmark.get("schema_version") != "full-benchmark-run-v1" or benchmark.get("status") != "complete":
        raise RuntimeError("benchmark manifest is not a complete V1 full-run manifest")
    if scope.get("status") != "ready" or gate_d.get("status") != "passed":
        raise RuntimeError("scope lock or stored Gate D receipt is not passed")
    dataset_ids = benchmark.get("dataset_ids")
    if not isinstance(dataset_ids, list) or dataset_ids != scope.get("dataset_ids"):
        raise RuntimeError("benchmark and scope-lock dataset IDs disagree")
    if len(dataset_ids) != 11 or len(set(dataset_ids)) != len(dataset_ids):
        raise RuntimeError("the V1 evidence must name eleven unique retained datasets")
    if analysis.get("schema_version") != "analysis-run-v1" or analysis.get("status") != "complete":
        raise RuntimeError("analysis manifest is not a complete V1 analysis manifest")
    if analysis.get("gate_d") != gate_d:
        raise RuntimeError("analysis and stored Gate D receipts disagree")
    if analysis.get("dataset_ids") != dataset_ids:
        raise RuntimeError("analysis dataset IDs disagree with the benchmark manifest")

    registry_path = root / "data" / "dataset_registry.csv"
    acquisition_path = root / "data" / "acquisition_manifest.csv"
    registry = _read_csv(registry_path, "dataset registry")
    acquisition = _read_csv(acquisition_path, "acquisition manifest")
    if registry["dataset_id"].duplicated().any() or acquisition["dataset_id"].duplicated().any():
        raise RuntimeError("registry and acquisition manifest dataset IDs must be unique")
    registry_by_id = registry.set_index("dataset_id")
    acquisition_by_id = acquisition.set_index("dataset_id")
    if not set(dataset_ids) <= set(registry_by_id.index):
        raise RuntimeError("the retained scope contains a dataset absent from the registry")
    retained = registry_by_id.loc[dataset_ids]
    if not retained["eligible"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("the retained V1 dataset scope includes an ineligible registry row")
    feature_types = retained["feature_type"].astype(str).to_dict()
    class_counts = {
        dataset: int(value) for dataset, value in retained["n_classes"].astype(str).items()
    }
    if (
        scope.get("registry_sha256") != _sha256(registry_path)
        or scope.get("acquisition_manifest_sha256") != _sha256(acquisition_path)
    ):
        raise RuntimeError("scope-lock registry or acquisition hash is stale")
    applicability_matrix = scope.get("applicability_matrix")
    if not isinstance(applicability_matrix, dict) or set(applicability_matrix) != set(dataset_ids):
        raise RuntimeError("scope-lock applicability does not cover the retained datasets")
    applicability: dict[str, set[str]] = {}
    for dataset in dataset_ids:
        row = applicability_matrix[dataset]
        if not isinstance(row, list) or not row or row[0] != feature_types[dataset]:
            raise RuntimeError(f"scope-lock feature type disagrees with registry for {dataset}")
        applicability[dataset] = set(row[1:])

    # Verify every packaged, non-raw reference. Raw-data references are compared
    # to the acquisition manifest below, but their excluded bytes are not checked.
    frozen = gate_d.get("frozen_references")
    if not isinstance(frozen, dict):
        raise TypeError("stored Gate D receipt has no frozen references")
    for name, reference in frozen.items():
        if name.startswith("raw/"):
            dataset = name.removeprefix("raw/")
            if dataset not in dataset_ids:
                raise RuntimeError(f"stored raw reference is outside the retained scope: {name}")
            continue
        resolved = _resolve_reference(root, reference, f"frozen input {name}")
        if reference.get("path") != name:
            raise RuntimeError(f"frozen input path key disagrees with its reference: {name}")
        if name == "artifacts/runs/scope-lock.json" and resolved != scope_path.resolve():
            raise RuntimeError("scope-lock reference points to an unexpected file")

    dataset_inputs = benchmark.get("dataset_inputs")
    if not isinstance(dataset_inputs, dict) or set(dataset_inputs) != set(dataset_ids):
        raise RuntimeError("benchmark raw-input references do not cover the retained datasets")
    for dataset in dataset_ids:
        raw_reference = dataset_inputs[dataset]
        acquired = acquisition_by_id.loc[dataset]
        if (
            raw_reference.get("path") != str(acquired["local_path"]).replace("\\", "/")
            or raw_reference.get("sha256") != str(acquired["raw_sha256"])
        ):
            raise RuntimeError(f"raw-input metadata disagrees with acquisition manifest for {dataset}")
        frozen_raw = frozen.get(f"raw/{dataset}")
        if (
            not isinstance(frozen_raw, dict)
            or frozen_raw.get("sha256") != raw_reference.get("sha256")
            or frozen_raw.get("path") != raw_reference.get("path")
        ):
            raise RuntimeError(f"stored Gate D raw-input metadata disagrees for {dataset}")

    for reference_name in ("scope_lock", "protocol"):
        reference = benchmark.get(reference_name)
        if isinstance(reference, dict):
            _resolve_reference(root, reference, f"benchmark {reference_name}")

    aggregate = benchmark.get("aggregate")
    if not isinstance(aggregate, dict) or set(aggregate) != {"results", "failures", "counts"}:
        raise RuntimeError("benchmark aggregate manifest has an unexpected schema")
    result_path = _resolve_reference(root, aggregate["results"], "benchmark results")
    failure_path = _resolve_reference(root, aggregate["failures"], "benchmark failures")
    if result_path.parent != run_dir.resolve() or failure_path.parent != run_dir.resolve():
        raise RuntimeError("aggregate records are outside the frozen full-run directory")

    seeds = benchmark.get("seeds")
    classifiers = benchmark.get("classifiers")
    conditions = benchmark.get("conditions")
    folds = benchmark.get("folds")
    if (
        seeds != [0, 1, 2]
        or folds != 5
        or classifiers != ["random_forest", "xgboost", "logistic_regression"]
        or conditions
        != [
            "raw",
            "class_weighted",
            "random_over",
            "random_under",
            "smote",
            "adasyn",
            "borderline_smote",
            "smotenc",
            "smoteenn",
            "smotetomek",
            "balanced_random_forest",
        ]
    ):
        raise RuntimeError("benchmark matrix differs from the frozen V1 design")
    if benchmark.get("expected_cells_per_dataset") != 495:
        raise RuntimeError("benchmark manifest has an unexpected per-dataset cell count")
    if (
        gate_d.get("dataset_ids") != dataset_ids
        or gate_d.get("dataset_count") != 11
        or gate_d.get("seeds") != seeds
        or gate_d.get("folds") != folds
        or gate_d.get("expected_cells_per_dataset") != 495
        or gate_d.get("benchmark_manifest")
        != {"path": "results/full-run/benchmark_manifest.json", "sha256": _sha256(benchmark_path)}
        or gate_d.get("aggregate") != {
            name: reference for name, reference in aggregate.items() if name in {"results", "failures"}
        }
    ):
        raise RuntimeError("stored Gate D scope metadata disagrees with the benchmark manifest")

    results = _read_csv(result_path, "benchmark results")
    failures = _read_csv(failure_path, "benchmark failures")
    counts = validate_aggregate_records(
        results,
        failures,
        dataset_ids=dataset_ids,
        seeds=seeds,
        folds=folds,
        classifiers=classifiers,
        conditions=conditions,
        feature_types=feature_types,
        class_counts=class_counts,
        applicability=applicability,
    )
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"aggregate record counts differ from frozen V1 totals: {counts}")
    _same_counts(aggregate.get("counts"), counts, "benchmark manifest")
    _same_counts(gate_d.get("counts"), counts, "stored Gate D receipt")

    dataset_runs = benchmark.get("dataset_runs")
    gate_counts = gate_d.get("dataset_counts")
    gate_markers = gate_d.get("dataset_markers")
    if not all(isinstance(value, dict) for value in (dataset_runs, gate_counts, gate_markers)):
        raise RuntimeError("benchmark or Gate D per-dataset accounting is missing")
    if any(set(value) != set(dataset_ids) for value in (dataset_runs, gate_counts, gate_markers)):
        raise RuntimeError("per-dataset accounting does not cover the retained scope")
    for dataset in dataset_ids:
        run = dataset_runs[dataset]
        expected_run = run.get("counts", {})
        if run.get("status") != "complete" or expected_run.get("expected_cells") != 495:
            raise RuntimeError(f"dataset run metadata is incomplete for {dataset}")
        if expected_run.get("valid_cells", 0) + expected_run.get("failure_cells", 0) != 495:
            raise RuntimeError(f"dataset run counts do not reconcile for {dataset}")
        if gate_counts[dataset] != expected_run:
            raise RuntimeError(f"stored Gate D counts disagree for {dataset}")
        if gate_markers[dataset] != run.get("marker"):
            raise RuntimeError(f"stored completion-marker metadata disagrees for {dataset}")
        actual_valid = int(results["dataset"].astype(str).eq(dataset).sum())
        actual_failures = int(failures["dataset"].astype(str).eq(dataset).sum())
        if (actual_valid, actual_failures) != (
            expected_run["valid_cells"],
            expected_run["failure_cells"],
        ):
            raise RuntimeError(f"aggregate records disagree with per-dataset counts for {dataset}")
    if sum(item["counts"]["valid_cells"] for item in dataset_runs.values()) != counts["valid_cells"]:
        raise RuntimeError("per-dataset valid counts do not sum to the aggregate count")
    if sum(item["counts"]["failure_cells"] for item in dataset_runs.values()) != counts["failure_cells"]:
        raise RuntimeError("per-dataset failure counts do not sum to the aggregate count")

    expected_input_reference = {"path": "results/full-run/benchmark_manifest.json", "sha256": _sha256(benchmark_path)}
    if analysis.get("input_run_manifest") != expected_input_reference:
        raise RuntimeError("analysis manifest does not reference this benchmark manifest")
    artifacts = analysis.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("analysis artifact references are missing")
    figure_rows = analysis.get("figures")
    if not isinstance(figure_rows, list):
        raise TypeError("analysis figure manifest is missing")
    figure_names = [
        row.get("name")
        for row in figure_rows
        if isinstance(row, dict) and row.get("status") == "generated"
    ]
    if len(figure_names) != GENERATED_FIGURE_COUNT or len(set(figure_names)) != GENERATED_FIGURE_COUNT:
        raise RuntimeError("analysis manifest must contain eight unique generated figures")
    expected_artifacts = {
        "results/analysis/gate-d-review.json",
        *(f"results/analysis/{name}" for name in ANALYSIS_TABLES),
        *(f"results/analysis/figures/{name}" for name in figure_names),
    }
    if set(artifacts) != expected_artifacts:
        raise RuntimeError("analysis manifest artifact inventory is incomplete or unexpected")
    for name, reference in artifacts.items():
        if not isinstance(reference, dict) or reference.get("path") != name:
            raise RuntimeError(f"analysis artifact key/reference mismatch: {name}")
        _resolve_reference(root, reference, f"analysis artifact {name}")

    from src.robustness import _validate_tables

    _validate_tables(analysis_dir, analysis, dataset_ids)
    return {
        "schema_version": "publication-aggregate-evidence-v1",
        "status": "passed",
        "scope": "aggregate record validation and deterministic analysis replay",
        "counts": counts,
        "dataset_ids": dataset_ids,
        "seeds": seeds,
        "folds": folds,
        "dataset_count": len(dataset_ids),
        "analysis_table_count": len(ANALYSIS_TABLES),
        "figure_count": GENERATED_FIGURE_COUNT,
        "verified_aggregate_hashes": [aggregate["results"], aggregate["failures"]],
        "not_checked": [
            "raw dataset bytes against acquisition SHA-256 values; raw datasets are excluded",
            "per-dataset completion-marker files; they are excluded from the evidence package",
            "model training, predictions, or an independent training reproduction",
            "a fresh Gate D or Gate E review",
        ],
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replay_publication_analysis(
    root: Path,
    *,
    run_dir: Path | None = None,
    analysis_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Replay packaged tables and figures without invoking Gate D or training."""

    root = root.resolve()
    run_dir = run_dir or root / "results" / "full-run"
    analysis_dir = analysis_dir or root / "results" / "analysis"
    output_dir = output_dir or root / "results" / "analysis-portable-replay"
    run_dir = run_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == analysis_dir:
        raise RuntimeError("portable replay output must not overwrite frozen analysis artifacts")
    receipt = validate_publication_evidence(root)
    run_analysis(root, run_dir, output_dir, aggregate_validation=receipt)

    analysis = _read_json(analysis_dir / "analysis_manifest.json", "analysis manifest")
    figure_names = [
        row["name"] for row in analysis["figures"] if row["status"] == "generated"
    ]
    artifact_names = [
        *(f"{name}" for name in ANALYSIS_TABLES),
        *(f"figures/{name}" for name in figure_names),
    ]
    replayed: list[dict[str, str]] = []
    for name in artifact_names:
        original = analysis_dir / name
        regenerated = output_dir / name
        if not original.is_file() or not regenerated.is_file():
            raise RuntimeError(f"portable replay artifact is missing: {name}")
        original_hash = _sha256(original)
        regenerated_hash = _sha256(regenerated)
        if original_hash != regenerated_hash:
            raise RuntimeError(f"portable replay changed frozen analysis artifact: {name}")
        replayed.append({"path": name, "sha256": regenerated_hash})
    if len(replayed) != 26:
        raise RuntimeError("portable replay did not compare all 18 tables and 8 figures")
    receipt["replay"] = {
        "status": "passed",
        "table_count": len(ANALYSIS_TABLES),
        "figure_count": len(figure_names),
        "compared_artifacts": replayed,
    }
    _write_json(output_dir / "publication-evidence-review.json", receipt)
    return receipt
