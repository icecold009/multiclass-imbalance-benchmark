"""Prepare and merge dataset-sharded V2 execution artifacts.

Sharding is an operational mechanism only: every shard uses the same frozen
configuration, registry, protocol, and dataset-level execution unit.  This
module never starts model fitting.  It validates completed per-dataset output
before copying it into a fresh aggregate directory.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.v2_execution import (
    TRIAL_COLUMNS,
    V2ExecutionContext,
    validate_v2_output_frames,
)

SHARD_PLAN_SCHEMA_VERSION = "v2-shard-plan-v1"
SHARD_RUN_SCHEMA_VERSION = "v2-full-run-v1"
SHARD_ARTIFACTS = (
    "v2_results.csv",
    "v2_failures.csv",
    "v2_trials.csv",
    "v2_complete.json",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_shard_plan(context: V2ExecutionContext, shard_count: int) -> dict[str, Any]:
    """Build a deterministic, complete dataset assignment plan."""

    dataset_ids = sorted(spec.dataset_id for spec in context.datasets)
    if not dataset_ids:
        raise ValueError("cannot build a shard plan without locked datasets")
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_count > len(dataset_ids):
        raise ValueError("shard_count cannot exceed the locked dataset count")

    shards = []
    for index in range(shard_count):
        assigned = dataset_ids[index::shard_count]
        shards.append(
            {
                "shard_id": f"shard-{index + 1:02d}",
                "dataset_ids": assigned,
                "expected_cells": len(assigned) * context.expected_cells_per_dataset,
            }
        )
    return {
        "schema_version": SHARD_PLAN_SCHEMA_VERSION,
        "execution_unit": "dataset",
        "dataset_ids": dataset_ids,
        "expected_cells_per_dataset": context.expected_cells_per_dataset,
        "expected_cells": len(dataset_ids) * context.expected_cells_per_dataset,
        "shard_count": shard_count,
        "config_sha256": context.config_sha256,
        "registry_sha256": context.registry_sha256,
        "protocol_sha256": context.protocol_sha256,
        "shards": shards,
    }


def validate_shard_plan(
    plan: Mapping[str, Any],
    locked_dataset_ids: Sequence[str],
    *,
    expected_cells_per_dataset: int,
) -> None:
    """Validate exact one-time coverage of the frozen dataset scope."""

    if plan.get("schema_version") != SHARD_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported shard-plan schema version")
    if plan.get("execution_unit") != "dataset":
        raise ValueError("shard plan must use dataset execution units")
    locked = sorted(str(dataset_id) for dataset_id in locked_dataset_ids)
    plan_dataset_ids = plan.get("dataset_ids")
    if plan_dataset_ids != locked:
        raise ValueError("shard plan dataset_ids do not exactly match the locked registry")
    if plan.get("expected_cells_per_dataset") != expected_cells_per_dataset:
        raise ValueError("shard plan has a stale expected cell count")
    if plan.get("expected_cells") != len(locked) * expected_cells_per_dataset:
        raise ValueError("shard plan expected cell count is inconsistent")

    shards = plan.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("shard plan must contain at least one shard")
    if plan.get("shard_count") != len(shards):
        raise ValueError("shard plan shard_count is inconsistent")

    assigned: list[str] = []
    expected_shard_ids = [f"shard-{index + 1:02d}" for index in range(len(shards))]
    for shard, expected_shard_id in zip(shards, expected_shard_ids, strict=True):
        if not isinstance(shard, Mapping) or shard.get("shard_id") != expected_shard_id:
            raise ValueError("shard IDs must be consecutive and deterministic")
        dataset_ids = shard.get("dataset_ids")
        if not isinstance(dataset_ids, list) or not dataset_ids:
            raise ValueError(f"{expected_shard_id} must contain at least one dataset")
        if dataset_ids != sorted(dataset_ids):
            raise ValueError(f"{expected_shard_id} dataset_ids must be sorted")
        if shard.get("expected_cells") != len(dataset_ids) * expected_cells_per_dataset:
            raise ValueError(f"{expected_shard_id} expected cell count is inconsistent")
        assigned.extend(str(dataset_id) for dataset_id in dataset_ids)

    if len(assigned) != len(set(assigned)):
        raise ValueError("shard plan contains duplicate dataset assignments")
    if sorted(assigned) != locked:
        raise ValueError("shard plan does not cover the locked registry exactly once")


def load_shard_plan(path: Path) -> dict[str, Any]:
    """Read a JSON shard plan without relaxing its schema."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("shard plan root must be an object")
    return payload


def write_shard_plan(path: Path, plan: Mapping[str, Any]) -> None:
    """Write a shard plan, refusing to overwrite an existing plan."""

    if path.exists():
        raise FileExistsError(f"shard plan already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, plan)


def _validate_shard_output(
    context: V2ExecutionContext,
    shard: Mapping[str, Any],
    shard_output_dir: Path,
    locked_dataset_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    manifest_path = shard_output_dir / "v2_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"{shard_output_dir}: v2_manifest.json is missing")
    manifest = load_shard_plan(manifest_path)
    if manifest.get("schema_version") != SHARD_RUN_SCHEMA_VERSION:
        raise RuntimeError(f"{shard_output_dir}: unsupported V2 run manifest schema")
    for field, expected in (
        ("config_sha256", context.config_sha256),
        ("registry_sha256", context.registry_sha256),
        ("protocol_sha256", context.protocol_sha256),
    ):
        if manifest.get(field) != expected:
            raise RuntimeError(f"{shard_output_dir}: manifest {field} is stale")
    if manifest.get("dataset_ids") != sorted(locked_dataset_ids):
        raise RuntimeError(f"{shard_output_dir}: manifest dataset scope is not frozen")
    if manifest.get("status") not in {"partial", "complete"}:
        raise RuntimeError(f"{shard_output_dir}: manifest is not a finished run")

    assigned_ids = [str(dataset_id) for dataset_id in shard["dataset_ids"]]
    dataset_runs = manifest.get("dataset_runs")
    if not isinstance(dataset_runs, dict) or set(dataset_runs) != set(assigned_ids):
        raise RuntimeError(f"{shard_output_dir}: manifest dataset runs do not match the shard")

    markers: dict[str, dict[str, Any]] = {}
    result_frames: dict[str, pd.DataFrame] = {}
    failure_frames: dict[str, pd.DataFrame] = {}
    specs = {spec.dataset_id: spec for spec in context.datasets}
    for dataset_id in assigned_ids:
        spec = specs[dataset_id]
        dataset_dir = shard_output_dir / dataset_id
        paths = {name: dataset_dir / name for name in SHARD_ARTIFACTS}
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise RuntimeError(f"{shard_output_dir}/{dataset_id}: missing {', '.join(missing)}")

        marker = json.loads(paths["v2_complete.json"].read_text(encoding="utf-8"))
        if marker.get("status") != "complete" or marker.get("dataset_id") != dataset_id:
            raise RuntimeError(f"{shard_output_dir}/{dataset_id}: completion marker is invalid")
        if marker.get("raw_sha256") != spec.raw_sha256:
            raise RuntimeError(f"{shard_output_dir}/{dataset_id}: raw input hash is stale")
        if marker.get("config_sha256") != context.config_sha256:
            raise RuntimeError(f"{shard_output_dir}/{dataset_id}: config hash is stale")
        if marker.get("registry_sha256") != context.registry_sha256:
            raise RuntimeError(f"{shard_output_dir}/{dataset_id}: registry hash is stale")

        results = pd.read_csv(paths["v2_results.csv"])
        failures = pd.read_csv(paths["v2_failures.csv"])
        counts = validate_v2_output_frames(dataset_id, results, failures)
        if marker.get("counts") != counts or dataset_runs[dataset_id].get("counts") != counts:
            raise RuntimeError(f"{shard_output_dir}/{dataset_id}: recorded counts are stale")
        trials = pd.read_csv(paths["v2_trials.csv"])
        missing_trial_columns = sorted(set(TRIAL_COLUMNS) - set(trials.columns))
        if missing_trial_columns:
            raise RuntimeError(
                f"{shard_output_dir}/{dataset_id}: trials missing {missing_trial_columns}"
            )
        if dataset_runs[dataset_id].get("status") != "complete":
            raise RuntimeError(f"{shard_output_dir}/{dataset_id}: manifest run is incomplete")
        markers[dataset_id] = marker
        result_frames[dataset_id] = results
        failure_frames[dataset_id] = failures
    return markers, result_frames, failure_frames


def merge_v2_shards(
    context: V2ExecutionContext,
    plan: Mapping[str, Any],
    shard_outputs: Mapping[str, Path],
    output_dir: Path,
    *,
    plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate all shard outputs and assemble one complete V2 output directory."""

    locked_dataset_ids = [spec.dataset_id for spec in context.datasets]
    validate_shard_plan(
        plan,
        locked_dataset_ids,
        expected_cells_per_dataset=context.expected_cells_per_dataset,
    )
    for field, expected in (
        ("config_sha256", context.config_sha256),
        ("registry_sha256", context.registry_sha256),
        ("protocol_sha256", context.protocol_sha256),
    ):
        if plan.get(field) != expected:
            raise ValueError(f"shard plan {field} is stale")
    expected_shard_ids = {str(shard["shard_id"]) for shard in plan["shards"]}
    if set(shard_outputs) != expected_shard_ids:
        raise ValueError("shard output paths must match the plan exactly")
    resolved_sources = [Path(path).resolve() for path in shard_outputs.values()]
    if len(resolved_sources) != len(set(resolved_sources)):
        raise ValueError("each shard must have a distinct output directory")
    if output_dir.exists():
        raise FileExistsError(f"merge output directory already exists: {output_dir}")

    validated: dict[str, tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]] = {}
    for shard in plan["shards"]:
        shard_id = str(shard["shard_id"])
        source = Path(shard_outputs[shard_id])
        if not source.is_dir():
            raise FileNotFoundError(f"{shard_id} output directory is missing: {source}")
        validated[shard_id] = _validate_shard_output(
            context,
            shard,
            source,
            locked_dataset_ids,
        )

    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    if staging_dir.exists():
        raise FileExistsError(f"staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    try:
        markers: dict[str, dict[str, Any]] = {}
        result_frames: dict[str, pd.DataFrame] = {}
        failure_frames: dict[str, pd.DataFrame] = {}
        for shard in plan["shards"]:
            shard_id = str(shard["shard_id"])
            source = Path(shard_outputs[shard_id])
            shard_markers, shard_results, shard_failures = validated[shard_id]
            for dataset_id in shard["dataset_ids"]:
                dataset_dir = staging_dir / dataset_id
                dataset_dir.mkdir()
                source_dataset_dir = source / dataset_id
                for name in SHARD_ARTIFACTS:
                    shutil.copy2(source_dataset_dir / name, dataset_dir / name)
                markers[dataset_id] = shard_markers[dataset_id]
                result_frames[dataset_id] = shard_results[dataset_id]
                failure_frames[dataset_id] = shard_failures[dataset_id]

        ordered_dataset_ids = sorted(markers)
        ordered_results = [result_frames[dataset_id] for dataset_id in ordered_dataset_ids]
        ordered_failures = [failure_frames[dataset_id] for dataset_id in ordered_dataset_ids]
        aggregate_counts = {
            "expected_cells": 0,
            "valid_cells": 0,
            "failure_cells": 0,
        }
        for dataset_id in sorted(markers):
            counts = markers[dataset_id]["counts"]
            for key in aggregate_counts:
                aggregate_counts[key] += int(counts[key])
        if aggregate_counts["valid_cells"] + aggregate_counts["failure_cells"] != aggregate_counts[
            "expected_cells"
        ]:
            raise RuntimeError("merged V2 outputs do not cover the locked cell matrix")

        results_path = staging_dir / "v2_results.csv"
        failures_path = staging_dir / "v2_failures.csv"
        pd.concat(ordered_results, ignore_index=True).to_csv(results_path, index=False)
        pd.concat(ordered_failures, ignore_index=True).to_csv(failures_path, index=False)
        manifest: dict[str, Any] = {
            "schema_version": SHARD_RUN_SCHEMA_VERSION,
            "status": "complete",
            "execution_mode": "dataset-sharded-merge",
            "created_at_utc": _utc_now(),
            "completed_at_utc": _utc_now(),
            "config_sha256": context.config_sha256,
            "registry_sha256": context.registry_sha256,
            "protocol_sha256": context.protocol_sha256,
            "dataset_ids": sorted(locked_dataset_ids),
            "expected_cells_per_dataset": context.expected_cells_per_dataset,
            "dataset_runs": {dataset_id: markers[dataset_id] for dataset_id in sorted(markers)},
            "shards": list(plan["shards"]),
            "aggregate": {
                "results": "v2_results.csv",
                "failures": "v2_failures.csv",
                "counts": aggregate_counts,
                "results_sha256": _sha256_file(results_path),
                "failures_sha256": _sha256_file(failures_path),
            },
        }
        if plan_sha256 is not None:
            manifest["shard_plan_sha256"] = plan_sha256
        _write_json(staging_dir / "v2_manifest.json", manifest)
        staging_dir.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging_dir)
        raise
