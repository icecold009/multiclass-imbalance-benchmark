"""Execute the locked benchmark with provenance and dataset-level resumability."""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.pilot import CLASSIFIERS, CONDITIONS, run_pilot
from src.stage0 import sha256_file

LOCKED_SEEDS = (0, 1, 2)
LOCKED_FOLDS = 5
EXPECTED_CELLS_PER_DATASET = (
    len(LOCKED_SEEDS) * LOCKED_FOLDS * len(CLASSIFIERS) * len(CONDITIONS)
)
KEY_COLUMNS = ("dataset", "seed", "fold", "classifier", "condition")
RESULT_COLUMNS = (
    *KEY_COLUMNS,
    "feature_type",
    "macro_f1",
    "g_mean",
    "mcc",
    "balanced_accuracy",
    "per_class_recall",
)
FAILURE_COLUMNS = (*KEY_COLUMNS, "feature_type", "stage", "error_type", "error")
PROTOCOL_HASH_PATHS = (
    "data/dataset_registry.csv",
    "data/acquisition_manifest.csv",
    "config/stage0.yaml",
    "artifacts/environment/metadata.json",
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    csv_path: Path
    target_column: str
    categorical_columns: tuple[str, ...]
    raw_sha256: str


@dataclass(frozen=True)
class ExecutionContext:
    root: Path
    protocol_path: Path
    scope_lock_path: Path
    registry_path: Path
    manifest_path: Path
    config_path: Path
    environment_path: Path
    datasets: tuple[DatasetSpec, ...]
    references: dict[str, dict[str, str]]


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _reference(path: Path, root: Path) -> dict[str, str]:
    return {"path": _relative_path(path, root), "sha256": sha256_file(path)}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is unreadable: {type(error).__name__}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _parse_protocol_hashes(protocol_path: Path) -> dict[str, str]:
    pattern = re.compile(r"\| `([^`]+)` \| `([0-9a-fA-F]{64})` \|")
    text = protocol_path.read_text(encoding="utf-8")
    return {path: digest.lower() for path, digest in pattern.findall(text)}


def _git_head(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def _load_context(root: Path, paths: dict[str, Path]) -> ExecutionContext:
    for label, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"{label} is missing: {path}")

    protocol_hashes = _parse_protocol_hashes(paths["protocol"])
    references: dict[str, dict[str, str]] = {}
    for relative_path in PROTOCOL_HASH_PATHS:
        expected = protocol_hashes.get(relative_path)
        path = root / relative_path
        if expected is None:
            raise RuntimeError(f"protocol does not freeze a SHA-256 for {relative_path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"protocol hash mismatch for {relative_path}: expected {expected}, got {actual}"
            )
        references[relative_path] = {"path": relative_path, "sha256": actual}

    scope_lock = _read_json(paths["scope_lock"], "scope lock")
    if scope_lock.get("schema_version") != "stage0-scope-lock-v2":
        raise RuntimeError("scope lock has an unexpected schema version")
    if scope_lock.get("status") != "ready":
        blockers = scope_lock.get("blockers", [])
        raise RuntimeError(f"scope lock is not ready: {blockers}")
    locked_ids = tuple(sorted(str(value) for value in scope_lock.get("dataset_ids", [])))
    if not locked_ids or scope_lock.get("dataset_count") != len(locked_ids):
        raise RuntimeError("scope lock has an invalid dataset list")
    if scope_lock.get("registry_sha256", "").lower() != references[PROTOCOL_HASH_PATHS[0]]["sha256"]:
        raise RuntimeError("scope lock registry hash is stale")
    if scope_lock.get("acquisition_manifest_sha256", "").lower() != references[PROTOCOL_HASH_PATHS[1]]["sha256"]:
        raise RuntimeError("scope lock acquisition-manifest hash is stale")
    references["docs/protocol.md"] = _reference(paths["protocol"], root)
    references["artifacts/runs/scope-lock.json"] = _reference(paths["scope_lock"], root)

    registry = pd.read_csv(paths["registry"], dtype=str, keep_default_na=False)
    manifest = pd.read_csv(paths["manifest"], dtype=str, keep_default_na=False)
    overrides = pd.read_csv(paths["overrides"], dtype=str, keep_default_na=False)
    if registry["dataset_id"].duplicated().any() or manifest["dataset_id"].duplicated().any():
        raise RuntimeError("registry and acquisition manifest dataset IDs must be unique")
    eligible_ids = {
        str(row.dataset_id)
        for row in registry.itertuples(index=False)
        if _truthy(row.eligible)
    }
    if eligible_ids != set(locked_ids):
        raise RuntimeError(
            "scope lock dataset IDs do not match eligible registry IDs: "
            f"locked={sorted(locked_ids)}, eligible={sorted(eligible_ids)}"
        )

    registry_by_id = registry.set_index("dataset_id")
    manifest_by_id = manifest.set_index("dataset_id")
    override_map = {
        str(row.dataset_id): tuple(
            value.strip()
            for value in str(row.categorical_columns).split(";")
            if value.strip()
        )
        for row in overrides.itertuples(index=False)
    }
    specs: list[DatasetSpec] = []
    for dataset_id in locked_ids:
        registry_row = registry_by_id.loc[dataset_id]
        manifest_row = manifest_by_id.loc[dataset_id]
        csv_path = (root / str(manifest_row.local_path)).resolve()
        raw_sha256 = str(manifest_row.raw_sha256).lower()
        try:
            expected_size = int(manifest_row.file_size_bytes)
        except ValueError as error:
            raise RuntimeError(f"{dataset_id} has an invalid manifest file size") from error
        if not csv_path.is_file():
            raise RuntimeError(f"{dataset_id} raw file is missing: {manifest_row.local_path}")
        if csv_path.stat().st_size != expected_size or sha256_file(csv_path) != raw_sha256:
            raise RuntimeError(f"{dataset_id} raw file does not match the acquisition manifest")
        target_column = str(registry_row.target_column)
        if not target_column:
            raise RuntimeError(f"{dataset_id} has no target column")
        references[f"raw/{dataset_id}"] = {
            "path": _relative_path(csv_path, root),
            "sha256": raw_sha256,
        }
        specs.append(
            DatasetSpec(
                dataset_id=dataset_id,
                csv_path=csv_path,
                target_column=target_column,
                categorical_columns=override_map.get(dataset_id, ()),
                raw_sha256=raw_sha256,
            )
        )
    return ExecutionContext(
        root=root,
        protocol_path=paths["protocol"],
        scope_lock_path=paths["scope_lock"],
        registry_path=paths["registry"],
        manifest_path=paths["manifest"],
        config_path=paths["config"],
        environment_path=paths["environment"],
        datasets=tuple(specs),
        references=references,
    )


def expected_cell_keys(dataset_id: str) -> set[tuple[object, ...]]:
    """Return every locked dataset/seed/fold/classifier/condition key."""

    return {
        (dataset_id, seed, fold, classifier, condition)
        for seed, fold, classifier, condition in itertools.product(
            LOCKED_SEEDS, range(LOCKED_FOLDS), CLASSIFIERS, CONDITIONS
        )
    }


def _normalised_keys(frame: pd.DataFrame, label: str) -> set[tuple[object, ...]]:
    missing = set(KEY_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"{label} is missing key columns: {sorted(missing)}")
    keys: list[tuple[object, ...]] = []
    for row in frame[list(KEY_COLUMNS)].itertuples(index=False, name=None):
        try:
            key = (str(row[0]), int(row[1]), int(row[2]), str(row[3]), str(row[4]))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{label} contains an invalid comparison key: {row}") from error
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"{label} contains duplicate comparison keys")
    return set(keys)


def validate_output_frames(
    dataset_id: str, results: pd.DataFrame, failures: pd.DataFrame
) -> dict[str, int]:
    """Validate complete cell accounting for one locked dataset run."""

    missing_results = set(RESULT_COLUMNS).difference(results.columns)
    missing_failures = set(FAILURE_COLUMNS).difference(failures.columns)
    if missing_results:
        raise RuntimeError(f"{dataset_id} results are missing columns: {sorted(missing_results)}")
    if missing_failures:
        raise RuntimeError(f"{dataset_id} failures are missing columns: {sorted(missing_failures)}")
    result_keys = _normalised_keys(results, f"{dataset_id} results")
    failure_keys = _normalised_keys(failures, f"{dataset_id} failures")
    expected = expected_cell_keys(dataset_id)
    if result_keys & failure_keys:
        raise RuntimeError(f"{dataset_id} has a key recorded as both valid and failed")
    observed = result_keys | failure_keys
    if observed != expected:
        missing = sorted(expected - observed, key=str)
        extra = sorted(observed - expected, key=str)
        raise RuntimeError(f"{dataset_id} cell accounting mismatch: missing={missing[:3]}, extra={extra[:3]}")
    for value in results["per_class_recall"].tolist():
        try:
            recalls = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{dataset_id} has invalid per-class recall JSON") from error
        if not isinstance(recalls, list) or any(
            not isinstance(item, (int, float)) or not math.isfinite(float(item))
            for item in recalls
        ):
            raise RuntimeError(f"{dataset_id} has invalid per-class recall values")
    return {
        "expected_cells": len(expected),
        "valid_cells": len(result_keys),
        "failure_cells": len(failure_keys),
    }


def _dataset_artifacts(dataset_dir: Path) -> dict[str, Path]:
    return {
        name: dataset_dir / f"benchmark_{name}"
        for name in (
            "metadata.json",
            "results.csv",
            "failures.csv",
            "rankings.csv",
            "friedman_summary.csv",
        )
    }


def _validate_completed_dataset(
    spec: DatasetSpec, dataset_dir: Path, marker_path: Path, root: Path
) -> dict[str, int]:
    marker = _read_json(marker_path, f"{spec.dataset_id} completion marker")
    if marker.get("schema_version") != "full-benchmark-dataset-v1":
        raise RuntimeError(f"{spec.dataset_id} completion marker has an unexpected schema")
    if marker.get("status") != "complete" or marker.get("dataset_id") != spec.dataset_id:
        raise RuntimeError(f"{spec.dataset_id} completion marker is not complete for this dataset")
    if marker.get("raw_sha256") != spec.raw_sha256:
        raise RuntimeError(f"{spec.dataset_id} completion marker has a stale raw-file hash")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError(f"{spec.dataset_id} completion marker has no artifact references")
    paths = _dataset_artifacts(dataset_dir)
    for name, path in paths.items():
        reference = artifacts.get(name)
        if not isinstance(reference, dict) or reference.get("path") != _relative_path(path, root):
            raise RuntimeError(f"{spec.dataset_id} completion marker has a stale {name} path")
        if reference.get("sha256") != sha256_file(path):
            raise RuntimeError(f"{spec.dataset_id} completion marker has a stale {name} hash")
    results = pd.read_csv(paths["results.csv"])
    failures = pd.read_csv(paths["failures.csv"])
    counts = validate_output_frames(spec.dataset_id, results, failures)
    if marker.get("counts") != counts:
        raise RuntimeError(f"{spec.dataset_id} completion marker counts do not match its outputs")
    return counts


def _write_dataset_marker(
    spec: DatasetSpec,
    dataset_dir: Path,
    marker_path: Path,
    root: Path,
    elapsed_seconds: float,
    counts: dict[str, int],
) -> None:
    artifacts = _dataset_artifacts(dataset_dir)
    marker = {
        "schema_version": "full-benchmark-dataset-v1",
        "status": "complete",
        "dataset_id": spec.dataset_id,
        "raw_sha256": spec.raw_sha256,
        "target_column": spec.target_column,
        "categorical_columns": list(spec.categorical_columns),
        "seeds": list(LOCKED_SEEDS),
        "folds": LOCKED_FOLDS,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "counts": counts,
        "artifacts": {
            name: _reference(path, root) for name, path in artifacts.items()
        },
        "completed_at_utc": _utc_now(),
    }
    _write_json_atomic(marker_path, marker)


def _manifest_template(context: ExecutionContext, output_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "full-benchmark-run-v1",
        "status": "running",
        "created_at_utc": _utc_now(),
        "git_head": _git_head(context.root),
        "protocol": context.references["docs/protocol.md"],
        "scope_lock": context.references["artifacts/runs/scope-lock.json"],
        "frozen_inputs": {
            path: value for path, value in context.references.items() if path in PROTOCOL_HASH_PATHS
        },
        "dataset_inputs": {
            spec.dataset_id: context.references[f"raw/{spec.dataset_id}"]
            for spec in context.datasets
        },
        "dataset_ids": [spec.dataset_id for spec in context.datasets],
        "seeds": list(LOCKED_SEEDS),
        "folds": LOCKED_FOLDS,
        "classifiers": list(CLASSIFIERS),
        "conditions": list(CONDITIONS),
        "expected_cells_per_dataset": EXPECTED_CELLS_PER_DATASET,
        "output_dir": _relative_path(output_dir, context.root),
        "dataset_runs": {},
    }


def _load_or_create_manifest(
    context: ExecutionContext, output_dir: Path
) -> dict[str, Any]:
    path = output_dir / "benchmark_manifest.json"
    expected = _manifest_template(context, output_dir)
    if not path.exists():
        _write_json_atomic(path, expected)
        return expected
    existing = _read_json(path, "benchmark manifest")
    for key in (
        "schema_version",
        "protocol",
        "scope_lock",
        "frozen_inputs",
        "dataset_inputs",
        "dataset_ids",
        "seeds",
        "folds",
        "classifiers",
        "conditions",
        "expected_cells_per_dataset",
        "output_dir",
    ):
        if existing.get(key) != expected.get(key):
            raise RuntimeError(f"benchmark manifest frozen field changed: {key}")
    if not isinstance(existing.get("dataset_runs"), dict):
        raise TypeError("benchmark manifest dataset_runs must be an object")
    return existing


def _record_dataset_run(
    manifest: dict[str, Any], spec: DatasetSpec, marker_path: Path, counts: dict[str, int], root: Path
) -> None:
    manifest.setdefault("dataset_runs", {})[spec.dataset_id] = {
        "status": "complete",
        "marker": _reference(marker_path, root),
        "counts": counts,
    }


def _aggregate_complete_run(
    context: ExecutionContext, output_dir: Path, manifest: dict[str, Any]
) -> dict[str, int]:
    result_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    for spec in context.datasets:
        dataset_dir = output_dir / spec.dataset_id
        marker_path = dataset_dir / "benchmark_complete.json"
        _validate_completed_dataset(spec, dataset_dir, marker_path, context.root)
        artifacts = _dataset_artifacts(dataset_dir)
        result_frames.append(pd.read_csv(artifacts["results.csv"]))
        failure_frames.append(pd.read_csv(artifacts["failures.csv"]))
    all_results = pd.concat(result_frames, ignore_index=True)
    all_failures = pd.concat(failure_frames, ignore_index=True)
    total_counts = {
        "expected_cells": EXPECTED_CELLS_PER_DATASET * len(context.datasets),
        "valid_cells": len(all_results),
        "failure_cells": len(all_failures),
    }
    if total_counts["valid_cells"] + total_counts["failure_cells"] != total_counts["expected_cells"]:
        raise RuntimeError("aggregate benchmark outputs do not cover the locked cell matrix")
    all_results.to_csv(output_dir / "benchmark_results.csv", index=False)
    all_failures.to_csv(output_dir / "benchmark_failures.csv", index=False)
    manifest["aggregate"] = {
        "results": _reference(output_dir / "benchmark_results.csv", context.root),
        "failures": _reference(output_dir / "benchmark_failures.csv", context.root),
        "counts": total_counts,
    }
    return total_counts


def run_benchmark(
    context: ExecutionContext,
    output_dir: Path,
    selected_dataset_ids: tuple[str, ...] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected = selected_dataset_ids or tuple(spec.dataset_id for spec in context.datasets)
    known_ids = {spec.dataset_id for spec in context.datasets}
    unknown = sorted(set(selected) - known_ids)
    if unknown:
        raise RuntimeError(f"requested dataset(s) are outside the locked scope: {', '.join(unknown)}")
    specs = tuple(spec for spec in context.datasets if spec.dataset_id in selected)
    if dry_run:
        for spec in specs:
            print(f"{spec.dataset_id}: target={spec.target_column}, categorical={list(spec.categorical_columns)}")
        print(
            f"planned datasets={len(specs)}, cells={len(specs) * EXPECTED_CELLS_PER_DATASET}, "
            f"seeds={list(LOCKED_SEEDS)}, folds={LOCKED_FOLDS}"
        )
        return {"status": "dry-run", "dataset_ids": list(selected)}

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(context, output_dir)
    for spec in specs:
        dataset_dir = output_dir / spec.dataset_id
        marker_path = dataset_dir / "benchmark_complete.json"
        if marker_path.exists():
            counts = _validate_completed_dataset(spec, dataset_dir, marker_path, context.root)
            _record_dataset_run(manifest, spec, marker_path, counts, context.root)
            _write_json_atomic(output_dir / "benchmark_manifest.json", manifest)
            print(f"{spec.dataset_id}: already complete; skipped")
            continue
        dataset_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f"{spec.dataset_id}: running {EXPECTED_CELLS_PER_DATASET} cells")
        run_pilot(
            spec.csv_path,
            spec.target_column,
            dataset_dir,
            seeds=LOCKED_SEEDS,
            folds=LOCKED_FOLDS,
            categorical_columns=spec.categorical_columns,
            artifact_prefix="benchmark",
        )
        artifacts = _dataset_artifacts(dataset_dir)
        results = pd.read_csv(artifacts["results.csv"])
        failures = pd.read_csv(artifacts["failures.csv"])
        counts = validate_output_frames(spec.dataset_id, results, failures)
        _write_dataset_marker(
            spec,
            dataset_dir,
            marker_path,
            context.root,
            time.perf_counter() - started,
            counts,
        )
        _record_dataset_run(manifest, spec, marker_path, counts, context.root)
        _write_json_atomic(output_dir / "benchmark_manifest.json", manifest)
        print(f"{spec.dataset_id}: complete; valid={counts['valid_cells']}, failed={counts['failure_cells']}")

    all_complete = all(
        (output_dir / spec.dataset_id / "benchmark_complete.json").exists()
        for spec in context.datasets
    )
    if all_complete:
        counts = _aggregate_complete_run(context, output_dir, manifest)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = _utc_now()
        manifest["aggregate"]["counts"] = counts
        _write_json_atomic(output_dir / "benchmark_manifest.json", manifest)
        print(
            f"benchmark complete; datasets={len(context.datasets)}, valid={counts['valid_cells']}, "
            f"failed={counts['failure_cells']}"
        )
    else:
        manifest["status"] = "partial"
        _write_json_atomic(output_dir / "benchmark_manifest.json", manifest)
        print("selected datasets complete; locked scope remains partial")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the locked full benchmark")
    parser.add_argument("--scope-lock", type=Path, default=Path("artifacts/runs/scope-lock.json"))
    parser.add_argument("--registry", type=Path, default=Path("data/dataset_registry.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/acquisition_manifest.csv"))
    parser.add_argument(
        "--overrides", type=Path, default=Path("data/dataset_feature_overrides.csv")
    )
    parser.add_argument("--protocol", type=Path, default=Path("docs/protocol.md"))
    parser.add_argument("--config", type=Path, default=Path("config/stage0.yaml"))
    parser.add_argument(
        "--environment", type=Path, default=Path("artifacts/environment/metadata.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/full-run"))
    parser.add_argument("--dataset", dest="datasets", action="append", metavar="DATASET_ID")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw_paths = {
        "scope_lock": args.scope_lock,
        "registry": args.registry,
        "manifest": args.manifest,
        "overrides": args.overrides,
        "protocol": args.protocol,
        "config": args.config,
        "environment": args.environment,
    }
    paths = {
        key: value if value.is_absolute() else root / value
        for key, value in raw_paths.items()
    }
    context = _load_context(root, paths)
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    selected = tuple(args.datasets) if args.datasets else None
    run_benchmark(context, output_dir, selected_dataset_ids=selected, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
