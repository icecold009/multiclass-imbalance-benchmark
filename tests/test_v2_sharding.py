import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.v2_engineering import failure_record
from src.v2_execution import (
    FAILURE_COLUMNS,
    RESULT_COLUMNS,
    TRIAL_COLUMNS,
    V2DatasetSpec,
    expected_cell_keys,
)
from src.v2_sharding import (
    build_shard_plan,
    merge_v2_shards,
    validate_shard_plan,
)


def _context(tmp_path: Path, dataset_ids: tuple[str, ...]) -> SimpleNamespace:
    datasets = tuple(
        V2DatasetSpec(
            dataset_id=dataset_id,
            raw_path=tmp_path / f"{dataset_id}.csv",
            raw_sha256=f"raw-{dataset_id}",
            target_column="target",
            categorical_columns=(),
            feature_type="numeric",
        )
        for dataset_id in dataset_ids
    )
    return SimpleNamespace(
        datasets=datasets,
        expected_cells_per_dataset=275,
        config_sha256="config-hash",
        registry_sha256="registry-hash",
        protocol_sha256="protocol-hash",
    )


def _write_complete_dataset(context: SimpleNamespace, dataset_id: str, output_dir: Path) -> None:
    spec = next(spec for spec in context.datasets if spec.dataset_id == dataset_id)
    dataset_dir = output_dir / dataset_id
    dataset_dir.mkdir(parents=True)
    failures = []
    for key in expected_cell_keys(dataset_id):
        row = failure_record(
            dataset_id=key[0],
            outer_fold=key[1],
            outer_seed=key[2],
            classifier=key[3],
            condition=key[4],
            stage="applicability",
            exception_type="NotApplicableError",
            reason="fixture failure",
            applicability="NOT_APPLICABLE",
            wall_seconds=0.0,
            resource_status="not_started",
            retry_count=0,
            config_sha256=context.config_sha256,
        )
        row.update({"feature_type": "numeric", "worker_count": 1, "timeout": False})
        failures.append(row)
    pd.DataFrame(columns=RESULT_COLUMNS).to_csv(dataset_dir / "v2_results.csv", index=False)
    pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(
        dataset_dir / "v2_failures.csv", index=False
    )
    pd.DataFrame(columns=TRIAL_COLUMNS).to_csv(dataset_dir / "v2_trials.csv", index=False)
    marker = {
        "schema_version": "v2-dataset-run-v1",
        "status": "complete",
        "dataset_id": dataset_id,
        "raw_sha256": spec.raw_sha256,
        "config_sha256": context.config_sha256,
        "registry_sha256": context.registry_sha256,
        "counts": {"expected_cells": 275, "valid_cells": 0, "failure_cells": 275},
    }
    (dataset_dir / "v2_complete.json").write_text(json.dumps(marker), encoding="utf-8")


def _write_shard(context: SimpleNamespace, plan: dict, shard: dict, output_dir: Path) -> None:
    output_dir.mkdir()
    for dataset_id in shard["dataset_ids"]:
        _write_complete_dataset(context, dataset_id, output_dir)
    manifest = {
        "schema_version": "v2-full-run-v1",
        "status": "partial",
        "config_sha256": context.config_sha256,
        "registry_sha256": context.registry_sha256,
        "protocol_sha256": context.protocol_sha256,
        "dataset_ids": sorted(dataset_id for dataset_id in plan["dataset_ids"]),
        "dataset_runs": {
            dataset_id: {"status": "complete", "counts": {"expected_cells": 275, "valid_cells": 0, "failure_cells": 275}}
            for dataset_id in shard["dataset_ids"]
        },
    }
    (output_dir / "v2_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_build_shard_plan_is_deterministic_and_covers_locked_datasets(tmp_path: Path) -> None:
    context = _context(tmp_path, ("dataset-c", "dataset-a", "dataset-b", "dataset-d"))

    plan = build_shard_plan(context, shard_count=2)

    assert [shard["shard_id"] for shard in plan["shards"]] == ["shard-01", "shard-02"]
    assert plan["dataset_ids"] == ["dataset-a", "dataset-b", "dataset-c", "dataset-d"]
    assert plan["shards"][0]["dataset_ids"] == ["dataset-a", "dataset-c"]
    assert plan["shards"][1]["dataset_ids"] == ["dataset-b", "dataset-d"]
    validate_shard_plan(plan, plan["dataset_ids"], expected_cells_per_dataset=275)


def test_validate_shard_plan_rejects_duplicate_dataset_assignment(tmp_path: Path) -> None:
    context = _context(tmp_path, ("dataset-a", "dataset-b"))
    plan = build_shard_plan(context, shard_count=2)
    plan["shards"][1]["dataset_ids"] = ["dataset-a"]

    with pytest.raises(ValueError, match="duplicate"):
        validate_shard_plan(plan, ["dataset-a", "dataset-b"], expected_cells_per_dataset=275)


def test_merge_v2_shards_validates_and_writes_complete_manifest(tmp_path: Path) -> None:
    context = _context(tmp_path, ("dataset-a", "dataset-b"))
    plan = build_shard_plan(context, shard_count=2)
    shard_outputs = {}
    for shard in plan["shards"]:
        shard_dir = tmp_path / shard["shard_id"]
        _write_shard(context, plan, shard, shard_dir)
        shard_outputs[shard["shard_id"]] = shard_dir

    manifest = merge_v2_shards(
        context,
        plan,
        shard_outputs,
        tmp_path / "merged",
        plan_sha256="plan-hash",
    )

    assert manifest["status"] == "complete"
    assert manifest["execution_mode"] == "dataset-sharded-merge"
    assert manifest["shard_plan_sha256"] == "plan-hash"
    assert manifest["aggregate"]["counts"] == {
        "expected_cells": 550,
        "valid_cells": 0,
        "failure_cells": 550,
    }
    assert (tmp_path / "merged" / "dataset-a" / "v2_trials.csv").is_file()
    assert (tmp_path / "merged" / "v2_results.csv").is_file()


def test_merge_v2_shards_refuses_existing_output(tmp_path: Path) -> None:
    context = _context(tmp_path, ("dataset-a",))
    plan = build_shard_plan(context, shard_count=1)
    shard_dir = tmp_path / "shard-01"
    _write_shard(context, plan, plan["shards"][0], shard_dir)
    output_dir = tmp_path / "merged"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        merge_v2_shards(context, plan, {"shard-01": shard_dir}, output_dir)
