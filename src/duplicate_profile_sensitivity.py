"""Run the frozen exact-profile grouped-CV sensitivity outside the V1 outputs."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import itertools
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.duplicate_profiles import GROUPED_SENSITIVITY_DATASETS, predictor_profile_codes
from src.pilot import (
    CLASSIFIERS,
    CONDITIONS,
    build_pipeline,
    feature_layout,
    fit_kwargs_for,
    geometric_mean,
    psutil,
    sampler_for,
)

SEEDS = (0, 1, 2)
FOLDS = 5
KEY_COLUMNS = ("dataset", "seed", "fold", "classifier", "condition")
METRICS = ("macro_f1", "g_mean", "mcc", "balanced_accuracy")
SOURCE_FILES = (
    "data/dataset_registry.csv",
    "data/acquisition_manifest.csv",
    "data/dataset_feature_overrides.csv",
    "config/stage0.yaml",
    "docs/protocol.md",
    "docs/duplicate-profile-policy.md",
    "requirements-analysis.lock",
    "results/full-run/benchmark_manifest.json",
    "results/full-run/benchmark_results.csv",
    "results/full-run/benchmark_failures.csv",
    "results/analysis/analysis_manifest.json",
    "scripts/run_duplicate_profile_sensitivity.py",
    "src/duplicate_profile_sensitivity.py",
    "src/duplicate_profiles.py",
    "src/pilot.py",
)
RESULT_COLUMNS = (
    *KEY_COLUMNS,
    "feature_type",
    *METRICS,
    "per_class_recall",
    "fit_predict_seconds",
    "rss_before_mb",
    "rss_after_mb",
    "rss_delta_mb",
    "train_rows",
    "test_rows",
)
FAILURE_COLUMNS = (
    *KEY_COLUMNS,
    "feature_type",
    "failure_category",
    "stage",
    "error_type",
    "error",
    "elapsed_seconds",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _environment(root: Path) -> dict[str, Any]:
    names = (
        "imbalanced-learn",
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "scikit-posthocs",
        "scipy",
        "statsmodels",
        "xgboost",
        "psutil",
    )
    packages: dict[str, str | None] = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "analysis_lock_sha256": sha256_file(root / "requirements-analysis.lock"),
        "training_environment_note": (
            "The July Stage 0 environment record is Python 3.12.10, while the frozen analysis "
            "manifest records Python 3.13.12. The full-run manifest does not identify its "
            "interpreter; this sensitivity records the environment actually used here."
        ),
    }


def _load_dataset_inputs(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    paths = {relative: root / relative for relative in SOURCE_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"sensitivity inputs are missing: {missing}")
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    run_manifest = json.loads(paths["results/full-run/benchmark_manifest.json"].read_text())
    if run_manifest.get("status") != "complete":
        raise ValueError("the frozen full-run manifest is not complete")
    for key, relative in (
        ("results", "results/full-run/benchmark_results.csv"),
        ("failures", "results/full-run/benchmark_failures.csv"),
    ):
        reference = run_manifest.get("aggregate", {}).get(key, {})
        if reference.get("path") != relative or reference.get("sha256") != input_hashes[relative]:
            raise ValueError(f"frozen full-run {key} reference does not match its CSV")
    analysis_manifest = json.loads(paths["results/analysis/analysis_manifest.json"].read_text())
    if analysis_manifest.get("status") != "complete":
        raise ValueError("the frozen analysis manifest is not complete")
    for relative, reference in analysis_manifest.get("artifacts", {}).items():
        artifact_path = (root / relative).resolve()
        if not artifact_path.is_relative_to(root.resolve()) or not artifact_path.is_file():
            raise ValueError(
                f"frozen analysis artifact is missing or outside the repository: {relative}"
            )
        actual = sha256_file(artifact_path)
        if reference.get("path") != relative or reference.get("sha256") != actual:
            raise ValueError(f"frozen analysis artifact hash does not match: {relative}")
        input_hashes[f"analysis:{relative}"] = actual
    registry = pd.read_csv(paths["data/dataset_registry.csv"], dtype=str, keep_default_na=False)
    acquisition = pd.read_csv(
        paths["data/acquisition_manifest.csv"], dtype=str, keep_default_na=False
    )
    overrides = pd.read_csv(
        paths["data/dataset_feature_overrides.csv"], dtype=str, keep_default_na=False
    )
    for frame, label in ((registry, "registry"), (acquisition, "acquisition manifest")):
        if frame["dataset_id"].duplicated().any():
            raise ValueError(f"{label} contains duplicate dataset IDs")
    if overrides["dataset_id"].duplicated().any():
        raise ValueError("feature override table contains duplicate dataset IDs")
    registry_by_id = registry.set_index("dataset_id")
    acquisition_by_id = acquisition.set_index("dataset_id")
    overrides_by_id = overrides.set_index("dataset_id")
    inputs: list[dict[str, Any]] = []
    for dataset_id in GROUPED_SENSITIVITY_DATASETS:
        if dataset_id not in registry_by_id.index or dataset_id not in acquisition_by_id.index:
            raise ValueError(f"{dataset_id}: registry or acquisition record is missing")
        record = registry_by_id.loc[dataset_id]
        source = acquisition_by_id.loc[dataset_id]
        if str(record["eligible"]).strip().casefold() != "true":
            raise ValueError(f"{dataset_id}: grouped sensitivity dataset is not eligible")
        registry_hash = str(record["raw_sha256"]).lower()
        acquisition_hash = str(source["raw_sha256"]).lower()
        if registry_hash != acquisition_hash:
            raise ValueError(f"{dataset_id}: registry and acquisition hashes disagree")
        csv_path = (root / str(source["local_path"])).resolve()
        if not csv_path.is_relative_to(root.resolve()):
            raise ValueError(f"{dataset_id}: raw CSV path escapes the repository")
        actual_hash = sha256_file(csv_path)
        if actual_hash != registry_hash:
            raise ValueError(f"{dataset_id}: raw CSV hash differs from frozen references")
        run_input = run_manifest.get("dataset_inputs", {}).get(dataset_id, {})
        if (
            run_input.get("path") != str(source["local_path"])
            or run_input.get("sha256") != actual_hash
        ):
            raise ValueError(f"{dataset_id}: raw CSV differs from the frozen full-run input")
        frame = pd.read_csv(csv_path)
        target = str(record["target_column"])
        if target not in frame or frame.empty or frame[target].isna().any():
            raise ValueError(f"{dataset_id}: target is missing or incomplete")
        if len(frame) != int(record["n_rows"]):
            raise ValueError(f"{dataset_id}: raw row count differs from the registry")
        categorical_value = (
            str(overrides_by_id.loc[dataset_id, "categorical_columns"])
            if dataset_id in overrides_by_id.index
            else ""
        )
        categorical = tuple(
            value.strip() for value in categorical_value.split(";") if value.strip()
        )
        inputs.append(
            {
                "dataset_id": dataset_id,
                "frame": frame,
                "csv_path": csv_path,
                "raw_sha256": actual_hash,
                "target": target,
                "categorical_columns": categorical,
            }
        )
        input_hashes[f"raw:{dataset_id}"] = actual_hash
    return inputs, input_hashes


def _build_preflight(
    inputs: list[dict[str, Any]],
    seeds: tuple[int, ...] = SEEDS,
    folds: int = FOLDS,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if seeds != SEEDS or folds != FOLDS:
        raise ValueError("the frozen sensitivity requires seeds 0, 1, 2 and five folds")
    groups_by_dataset: dict[str, np.ndarray] = {}
    splits_by_dataset: dict[
        str, dict[tuple[int, int], tuple[np.ndarray, np.ndarray, bool, str]]
    ] = {}
    group_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []
    for item in inputs:
        dataset_id = item["dataset_id"]
        frame = item["frame"]
        target = item["target"]
        features = frame.drop(columns=[target]).reset_index(drop=True)
        labels_raw = frame[target].reset_index(drop=True)
        encoder = LabelEncoder()
        labels = pd.Series(encoder.fit_transform(labels_raw), name=target)
        groups = predictor_profile_codes(features)
        groups_by_dataset[dataset_id] = groups
        splits_by_dataset[dataset_id] = {}
        group_rows.extend(
            {
                "dataset": dataset_id,
                "row_index": int(row_index),
                "profile_group_code": int(group_code),
            }
            for row_index, group_code in enumerate(groups)
        )
        expected_classes = set(labels.unique().tolist())
        layout = feature_layout(frame, target, item["categorical_columns"])
        for seed in seeds:
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            try:
                generated_splits = list(splitter.split(features, labels, groups))
            except ValueError as error:
                empty = np.asarray([], dtype=np.int64)
                for fold in range(folds):
                    splits_by_dataset[dataset_id][(seed, fold)] = (
                        empty,
                        empty,
                        False,
                        f"splitter_error: {type(error).__name__}: {error}",
                    )
                    feasibility_rows.append(
                        {
                            "dataset": dataset_id,
                            "seed": seed,
                            "fold": fold,
                            "train_rows": 0,
                            "test_rows": 0,
                            "train_profile_groups": 0,
                            "test_profile_groups": 0,
                            "overlapping_profile_groups": 0,
                            "missing_train_classes": ";".join(
                                str(value) for value in sorted(expected_classes)
                            ),
                            "missing_test_classes": ";".join(
                                str(value) for value in sorted(expected_classes)
                            ),
                            "feasible": False,
                            "split_error": f"{type(error).__name__}: {error}",
                        }
                    )
                continue
            for fold, (train_idx, test_idx) in enumerate(generated_splits):
                train_groups = set(groups[train_idx].tolist())
                test_groups = set(groups[test_idx].tolist())
                overlap = train_groups.intersection(test_groups)
                train_classes = set(labels.iloc[train_idx].tolist())
                test_classes = set(labels.iloc[test_idx].tolist())
                missing_train = sorted(expected_classes - train_classes)
                missing_test = sorted(expected_classes - test_classes)
                feasible = not overlap and not missing_train and not missing_test
                reasons = []
                if overlap:
                    reasons.append("profile groups overlap across train and test")
                if missing_train:
                    reasons.append(f"training split missing classes {missing_train}")
                if missing_test:
                    reasons.append(f"test split missing classes {missing_test}")
                splits_by_dataset[dataset_id][(seed, fold)] = (
                    train_idx,
                    test_idx,
                    feasible,
                    "; ".join(reasons),
                )
                feasibility_rows.append(
                    {
                        "dataset": dataset_id,
                        "seed": seed,
                        "fold": fold,
                        "train_rows": len(train_idx),
                        "test_rows": len(test_idx),
                        "train_profile_groups": len(train_groups),
                        "test_profile_groups": len(test_groups),
                        "overlapping_profile_groups": len(overlap),
                        "missing_train_classes": ";".join(str(value) for value in missing_train),
                        "missing_test_classes": ";".join(str(value) for value in missing_test),
                        "feasible": feasible,
                        "split_error": "; ".join(reasons),
                    }
                )
                for partition, indices in (("train", train_idx), ("test", test_idx)):
                    split_rows.extend(
                        {
                            "dataset": dataset_id,
                            "row_index": int(row_index),
                            "profile_group_code": int(groups[row_index]),
                            "seed": seed,
                            "fold": fold,
                            "partition": partition,
                        }
                        for row_index in indices
                    )
        # Layout validation is intentionally part of preflight, before any score is generated.
        if not layout.numeric:
            raise ValueError(f"{dataset_id}: no numeric predictors are supported")
    return (
        {"groups": groups_by_dataset, "splits": splits_by_dataset},
        pd.DataFrame(group_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(feasibility_rows),
    )


def _output_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "sensitivity_manifest.json"
    }


def prepare_preflight(root: Path, output_dir: Path) -> dict[str, Any]:
    """Write frozen inputs and all grouped fold assignments before model fitting."""
    root = root.resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("sensitivity output must remain inside the repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"sensitivity output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, input_hashes = _load_dataset_inputs(root)
    _plan, group_assignments, split_assignments, feasibility = _build_preflight(inputs)
    group_assignments.to_csv(output_dir / "profile_groups.csv", index=False)
    split_assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    feasibility.to_csv(output_dir / "split_feasibility.csv", index=False)
    valid_splits = int(feasibility["feasible"].sum())
    manifest: dict[str, Any] = {
        "schema_version": "duplicate-profile-sensitivity-v1",
        "status": "preflight_passed"
        if valid_splits == len(feasibility) == 90
        else "preflight_infeasible",
        "created_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "scope": "post hoc generalization sensitivity to unseen exact predictor profiles",
        "datasets": list(GROUPED_SENSITIVITY_DATASETS),
        "seeds": list(SEEDS),
        "folds": FOLDS,
        "expected_splits": 90,
        "valid_splits": valid_splits,
        "expected_model_keys": len(inputs)
        * len(SEEDS)
        * FOLDS
        * len(CLASSIFIERS)
        * len(CONDITIONS),
        "model_fitting_performed": False,
        "inference": "descriptive only; no hypothesis tests",
        "group_policy": "exact raw predictors excluding target; labels never form group IDs",
        "inputs": input_hashes,
        "environment": _environment(root),
        "dataset_status": {dataset_id: "pending" for dataset_id in GROUPED_SENSITIVITY_DATASETS},
        "dataset_outputs": {},
        "outputs": {},
    }
    manifest["outputs"] = _output_hashes(output_dir)
    _write_json(output_dir / "sensitivity_manifest.json", manifest)
    return manifest


def _applicability_error(layout: Any, classifier: str, condition: str, seed: int) -> str | None:
    if condition == "balanced_random_forest" and classifier != "random_forest":
        return "Balanced Random Forest is an RF-only ensemble comparator"
    try:
        sampler_for(condition, layout, random_state=seed)
    except ValueError as error:
        return str(error)
    return None


def _run_dataset(
    item: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_id = item["dataset_id"]
    frame = item["frame"]
    target = item["target"]
    features = frame.drop(columns=[target]).reset_index(drop=True)
    encoder = LabelEncoder()
    labels = pd.Series(encoder.fit_transform(frame[target]), name=target)
    layout = feature_layout(frame, target, item["categorical_columns"])
    feature_type = "mixed" if layout.is_mixed else "numeric"
    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in range(FOLDS):
            train_idx, test_idx, feasible, split_error = plan["splits"][dataset_id][(seed, fold)]
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
            if not feasible:
                split_error = split_error or "grouped split failed the recorded preflight"
            for classifier in CLASSIFIERS:
                for condition in CONDITIONS:
                    key = {
                        "dataset": dataset_id,
                        "seed": seed,
                        "fold": fold,
                        "classifier": classifier,
                        "condition": condition,
                        "feature_type": feature_type,
                    }
                    started = time.perf_counter()
                    if split_error:
                        failure_rows.append(
                            {
                                **key,
                                "failure_category": "infeasible_split",
                                "stage": "split_preflight",
                                "error_type": "InfeasibleSplit",
                                "error": split_error,
                                "elapsed_seconds": 0.0,
                            }
                        )
                        continue
                    applicability_error = _applicability_error(layout, classifier, condition, seed)
                    if applicability_error:
                        failure_rows.append(
                            {
                                **key,
                                "failure_category": "unsupported",
                                "stage": "applicability",
                                "error_type": "NotApplicable",
                                "error": applicability_error,
                                "elapsed_seconds": 0.0,
                            }
                        )
                        continue
                    rss_before = psutil.Process().memory_info().rss / (1024**2) if psutil else None
                    try:
                        pipeline = build_pipeline(layout, condition, classifier, random_state=seed)
                        pipeline.fit(
                            X_train,
                            y_train,
                            **fit_kwargs_for(condition, classifier, y_train),
                        )
                        predicted = pipeline.predict(X_test)
                        elapsed = time.perf_counter() - started
                        rss_after = (
                            psutil.Process().memory_info().rss / (1024**2) if psutil else None
                        )
                        result_rows.append(
                            {
                                **key,
                                "macro_f1": f1_score(
                                    y_test, predicted, average="macro", zero_division=0
                                ),
                                "g_mean": geometric_mean(
                                    y_test, predicted, sorted(labels.unique().tolist())
                                ),
                                "mcc": matthews_corrcoef(y_test, predicted),
                                "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                                "per_class_recall": json.dumps(
                                    recall_score(
                                        y_test,
                                        predicted,
                                        labels=sorted(labels.unique().tolist()),
                                        average=None,
                                        zero_division=0,
                                    ).tolist()
                                ),
                                "fit_predict_seconds": elapsed,
                                "rss_before_mb": rss_before,
                                "rss_after_mb": rss_after,
                                "rss_delta_mb": (
                                    rss_after - rss_before
                                    if rss_before is not None and rss_after is not None
                                    else None
                                ),
                                "train_rows": len(train_idx),
                                "test_rows": len(test_idx),
                            }
                        )
                    except Exception as error:  # noqa: BLE001 - model failures are evidence
                        failure_rows.append(
                            {
                                **key,
                                "failure_category": "model_failure",
                                "stage": "fit_predict",
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        )
    return (
        pd.DataFrame(result_rows, columns=RESULT_COLUMNS),
        pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS),
    )


def _cell_summary(
    inputs: list[dict[str, Any]],
    result_frames: list[pd.DataFrame],
    failure_frames: list[pd.DataFrame],
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped_results = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else pd.DataFrame(columns=RESULT_COLUMNS)
    )
    grouped_failures = (
        pd.concat(failure_frames, ignore_index=True)
        if failure_frames
        else pd.DataFrame(columns=FAILURE_COLUMNS)
    )
    original_results = pd.read_csv(root / "results/full-run/benchmark_results.csv")
    original_failures = pd.read_csv(root / "results/full-run/benchmark_failures.csv")
    cells: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    ranks: list[dict[str, Any]] = []
    for item in inputs:
        dataset_id = item["dataset_id"]
        layout = feature_layout(item["frame"], item["target"], item["categorical_columns"])
        supported_by_classifier: dict[str, list[str]] = {}
        for classifier in CLASSIFIERS:
            supported_by_classifier[classifier] = [
                condition
                for condition in CONDITIONS
                if _applicability_error(layout, classifier, condition, SEEDS[0]) is None
            ]
        cell_index: dict[tuple[str, str], dict[str, Any]] = {}
        for classifier in CLASSIFIERS:
            for condition in CONDITIONS:
                grouped = grouped_results.loc[
                    (grouped_results["dataset"] == dataset_id)
                    & (grouped_results["classifier"] == classifier)
                    & (grouped_results["condition"] == condition)
                ]
                grouped_failed = grouped_failures.loc[
                    (grouped_failures["dataset"] == dataset_id)
                    & (grouped_failures["classifier"] == classifier)
                    & (grouped_failures["condition"] == condition)
                ]
                applicable = condition in supported_by_classifier[classifier]
                expected = len(SEEDS) * FOLDS if applicable else 0
                complete = applicable and len(grouped) == expected and len(grouped_failed) == 0
                original = original_results.loc[
                    (original_results["dataset"] == dataset_id)
                    & (original_results["classifier"] == classifier)
                    & (original_results["condition"] == condition)
                ]
                original_failed = original_failures.loc[
                    (original_failures["dataset"] == dataset_id)
                    & (original_failures["classifier"] == classifier)
                    & (original_failures["condition"] == condition)
                ]
                original_complete = (
                    applicable and len(original) == len(SEEDS) * FOLDS and len(original_failed) == 0
                )
                row: dict[str, Any] = {
                    "dataset": dataset_id,
                    "classifier": classifier,
                    "condition": condition,
                    "feature_type": "mixed" if layout.is_mixed else "numeric",
                    "applicable": applicable,
                    "expected_fold_records": expected,
                    "valid_fold_records": len(grouped),
                    "failure_records": len(grouped_failed),
                    "unsupported_records": int(
                        grouped_failed.get("failure_category", pd.Series(dtype=str))
                        .astype(str)
                        .eq("unsupported")
                        .sum()
                    ),
                    "complete_cell": complete,
                    "original_valid_fold_records": len(original),
                    "original_failure_records": len(original_failed),
                    "original_complete_cell": original_complete,
                }
                for metric in METRICS:
                    row[f"grouped_mean_{metric}"] = (
                        float(grouped[metric].mean()) if len(grouped) else np.nan
                    )
                    row[f"original_mean_{metric}"] = (
                        float(original[metric].mean()) if len(original) else np.nan
                    )
                cells.append(row)
                cell_index[(classifier, condition)] = row
                for metric in METRICS:
                    paired = complete and original_complete
                    grouped_mean = row[f"grouped_mean_{metric}"]
                    original_mean = row[f"original_mean_{metric}"]
                    delta = float(grouped_mean - original_mean) if paired else np.nan
                    comparisons.append(
                        {
                            "dataset": dataset_id,
                            "classifier": classifier,
                            "condition": condition,
                            "metric": metric,
                            "grouped_mean": grouped_mean,
                            "original_mean": original_mean,
                            "grouped_minus_original": delta,
                            "complete_pair": paired,
                            "interpretation": "descriptive_only_no_tests",
                        }
                    )
        for classifier, supported in supported_by_classifier.items():
            grouped_ready = bool(supported) and all(
                cell_index[(classifier, condition)]["complete_cell"] for condition in supported
            )
            original_ready = bool(supported) and all(
                cell_index[(classifier, condition)]["original_complete_cell"]
                for condition in supported
            )
            status = (
                "complete_both_designs"
                if grouped_ready and original_ready
                else "incomplete_grouped"
                if not grouped_ready
                else "incomplete_original"
            )
            if grouped_ready:
                grouped_order = rankdata(
                    [
                        -cell_index[(classifier, condition)]["grouped_mean_macro_f1"]
                        for condition in supported
                    ],
                    method="average",
                )
            else:
                grouped_order = [np.nan] * len(supported)
            if original_ready:
                original_order = rankdata(
                    [
                        -cell_index[(classifier, condition)]["original_mean_macro_f1"]
                        for condition in supported
                    ],
                    method="average",
                )
            else:
                original_order = [np.nan] * len(supported)
            for index, condition in enumerate(supported):
                ranks.append(
                    {
                        "dataset": dataset_id,
                        "classifier": classifier,
                        "condition": condition,
                        "original_macro_f1_rank": original_order[index],
                        "grouped_macro_f1_rank": grouped_order[index],
                        "grouped_minus_original_rank": (
                            grouped_order[index] - original_order[index]
                            if grouped_ready and original_ready
                            else np.nan
                        ),
                        "rank_status": status,
                        "interpretation": "descriptive_only_no_tests",
                    }
                )
    cell_frame = pd.DataFrame(cells)
    comparison_frame = pd.DataFrame(comparisons)
    rank_frame = pd.DataFrame(ranks)
    summary_rows: list[dict[str, Any]] = []
    for item in inputs:
        subset = cell_frame.loc[cell_frame["dataset"] == item["dataset_id"]]
        failures = grouped_failures.loc[grouped_failures["dataset"] == item["dataset_id"]]
        summary_rows.append(
            {
                "dataset": item["dataset_id"],
                "expected_model_keys": len(SEEDS) * FOLDS * len(CLASSIFIERS) * len(CONDITIONS),
                "valid_model_records": int(subset["valid_fold_records"].sum()),
                "supported_expected_records": int(subset["expected_fold_records"].sum()),
                "unsupported_model_records": int(
                    failures["failure_category"].astype(str).eq("unsupported").sum()
                ),
                "model_failure_records": int(
                    failures["failure_category"].astype(str).eq("model_failure").sum()
                ),
                "infeasible_split_records": int(
                    failures["failure_category"].astype(str).eq("infeasible_split").sum()
                ),
                "complete_supported_cells": int(
                    (subset["complete_cell"] & subset["applicable"]).sum()
                ),
                "supported_cells": int(subset["applicable"].sum()),
                "complete_cell_coverage": float(
                    (subset["complete_cell"] & subset["applicable"]).sum()
                    / max(1, subset["applicable"].sum())
                ),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    return cell_frame, comparison_frame, rank_frame, summary_frame


def _validate_model_key_accounting(
    result_frames: list[pd.DataFrame], failure_frames: list[pd.DataFrame]
) -> int:
    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame()
    actual_rows = pd.concat(
        [results.loc[:, KEY_COLUMNS], failures.loc[:, KEY_COLUMNS]], ignore_index=True
    )
    if actual_rows.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("grouped sensitivity contains duplicate model-grid keys")
    actual = {
        (str(row.dataset), int(row.seed), int(row.fold), str(row.classifier), str(row.condition))
        for row in actual_rows.itertuples(index=False)
    }
    expected = set(
        itertools.product(
            GROUPED_SENSITIVITY_DATASETS,
            SEEDS,
            range(FOLDS),
            CLASSIFIERS,
            CONDITIONS,
        )
    )
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"grouped model-grid accounting mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    return len(actual)


def run_sensitivity(root: Path, output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    """Run the fixed grouped model grid, resuming completed dataset checkpoints."""
    root = root.resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("sensitivity output must remain inside the repository")
    manifest_path = output_dir / "sensitivity_manifest.json"
    if not manifest_path.is_file():
        prepare_preflight(root, output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "complete":
        raise FileExistsError("sensitivity run is already complete; refusing to overwrite it")
    if manifest.get("status") not in {"preflight_passed", "preflight_infeasible", "running"}:
        raise ValueError("sensitivity manifest is not in a resumable state")
    if manifest.get("status") == "running" and not resume:
        raise ValueError("partial sensitivity run exists; pass --resume to continue")
    inputs, input_hashes = _load_dataset_inputs(root)
    if input_hashes != manifest.get("inputs"):
        raise ValueError("sensitivity source inputs changed after the preflight")
    plan, _, _split_assignments, feasibility = _build_preflight(inputs)
    for relative in ("profile_groups.csv", "split_assignments.csv", "split_feasibility.csv"):
        if manifest["outputs"].get(relative) != sha256_file(output_dir / relative):
            raise ValueError(f"preflight artifact changed after it was recorded: {relative}")
    manifest["status"] = "running"
    manifest["model_fitting_performed"] = True
    _write_json(manifest_path, manifest)
    all_results: list[pd.DataFrame] = []
    all_failures: list[pd.DataFrame] = []
    for item in inputs:
        dataset_id = item["dataset_id"]
        dataset_dir = output_dir / "datasets" / dataset_id
        result_path = dataset_dir / "results.csv"
        failure_path = dataset_dir / "failures.csv"
        saved_hashes = manifest.get("dataset_outputs", {}).get(dataset_id, {})
        if manifest.get("dataset_status", {}).get(dataset_id) == "complete":
            if not saved_hashes or any(
                not (output_dir / relative).is_file()
                or sha256_file(output_dir / relative) != digest
                for relative, digest in saved_hashes.items()
            ):
                raise ValueError(f"{dataset_id}: completed checkpoint failed hash verification")
            all_results.append(pd.read_csv(result_path))
            all_failures.append(pd.read_csv(failure_path))
            continue
        manifest["dataset_status"][dataset_id] = "running"
        _write_json(manifest_path, manifest)
        results, failures = _run_dataset(item, plan)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(result_path, index=False)
        failures.to_csv(failure_path, index=False)
        rel_results = result_path.relative_to(output_dir).as_posix()
        rel_failures = failure_path.relative_to(output_dir).as_posix()
        manifest["dataset_status"][dataset_id] = "complete"
        manifest["dataset_outputs"][dataset_id] = {
            rel_results: sha256_file(result_path),
            rel_failures: sha256_file(failure_path),
        }
        _write_json(manifest_path, manifest)
        all_results.append(results)
        all_failures.append(failures)
        valid = len(results)
        failed = len(failures)
        print(f"{dataset_id}: complete; valid={valid}; explicit failures={failed}", flush=True)
    recorded_keys = _validate_model_key_accounting(all_results, all_failures)
    cells, comparisons, ranks, summary = _cell_summary(inputs, all_results, all_failures, root)
    output_frames = {
        "grouped_results.csv": pd.concat(all_results, ignore_index=True),
        "grouped_failures.csv": pd.concat(all_failures, ignore_index=True),
        "cell_means.csv": cells,
        "design_comparisons.csv": comparisons,
        "macro_f1_rank_comparisons.csv": ranks,
        "coverage.csv": summary,
    }
    for name, frame in output_frames.items():
        frame.to_csv(output_dir / name, index=False)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    manifest["outputs"] = _output_hashes(output_dir)
    manifest["counts"] = {
        "expected_model_keys": int(manifest["expected_model_keys"]),
        "verified_model_keys": recorded_keys,
        "recorded_result_keys": len(output_frames["grouped_results.csv"]),
        "recorded_failure_keys": len(output_frames["grouped_failures.csv"]),
        "valid_splits": int(feasibility["feasible"].sum()),
        "infeasible_splits": int((~feasibility["feasible"]).sum()),
        "comparison_rows": len(comparisons),
        "rank_rows": len(ranks),
    }
    manifest["model_fitting_performed"] = True
    _write_json(manifest_path, manifest)
    return manifest
