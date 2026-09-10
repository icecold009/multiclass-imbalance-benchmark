"""Verify the locked pilot replay and feasibility evidence for Gate C."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pilot_review import (
    EXPECTED_CELLS,
    RUNTIME_BUDGET_SECONDS,
    _replay_matches,
    relative,
    sha256_file,
)
from src.stage0 import build_scope_lock

GATE_C_SCHEMA = "stage0-gate-c-review-v1"
EXPECTED_DATASET_COUNT = 11


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON: {type(error).__name__}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def verify_gate_c(
    review_path: Path,
    registry_path: Path,
    manifest_path: Path,
    scope_lock_path: Path,
) -> dict[str, Any]:
    """Validate Gate A evidence and independently verify every recorded replay."""

    recorded_scope = _read_json(scope_lock_path, "recorded scope lock")
    with tempfile.TemporaryDirectory(prefix="stage0-gate-c-") as temp_dir:
        computed_scope_path = Path(temp_dir) / "scope-lock.json"
        scope = build_scope_lock(
            registry_path,
            computed_scope_path,
            pilot_status="passed",
            manifest_path=manifest_path,
            pilot_evidence_path=review_path,
        )
    if scope["status"] != "ready":
        blockers = "; ".join(scope.get("blockers", []))
        raise ValueError(f"scope lock is not ready: {blockers}")
    for field in (
        "status",
        "pilot_status",
        "registry_sha256",
        "acquisition_manifest_sha256",
        "dataset_count",
        "dataset_ids",
    ):
        if recorded_scope.get(field) != scope.get(field):
            raise ValueError(f"recorded scope lock field is stale: {field}")

    review = _read_json(review_path, "pilot review evidence")
    if review.get("schema_version") != "stage0-pilot-review-v1":
        raise ValueError("pilot review evidence has an unexpected schema")
    dataset_ids = list(scope["dataset_ids"])
    if len(dataset_ids) != EXPECTED_DATASET_COUNT:
        raise ValueError(
            f"Gate C requires the locked {EXPECTED_DATASET_COUNT}-dataset scope; "
            f"found {len(dataset_ids)}"
        )

    runs = review.get("runs")
    runs_by_id = {
        run.get("dataset_id"): run
        for run in runs
        if isinstance(run, dict) and isinstance(run.get("dataset_id"), str)
    } if isinstance(runs, list) else {}
    if set(runs_by_id) != set(dataset_ids):
        raise ValueError("pilot review runs do not exactly match the locked dataset scope")

    run_summaries: list[dict[str, Any]] = []
    total_valid = 0
    total_failures = 0
    max_runtime = 0.0
    max_replay_runtime = 0.0
    for dataset_id in dataset_ids:
        run = runs_by_id[dataset_id]
        if run.get("deterministic_replay") != "passed":
            raise ValueError(f"{dataset_id}: deterministic replay is not passed")
        if run.get("failure_review") != "passed":
            raise ValueError(f"{dataset_id}: failure review is not passed")
        if run.get("output_completeness") != "complete":
            raise ValueError(f"{dataset_id}: output completeness is not complete")
        if run.get("expected_cells") != EXPECTED_CELLS:
            raise ValueError(f"{dataset_id}: expected cell count is not {EXPECTED_CELLS}")
        valid_cells = run.get("valid_cells")
        failure_cells = run.get("failure_cells")
        if not isinstance(valid_cells, int) or not isinstance(failure_cells, int):
            raise TypeError(f"{dataset_id}: cell counts are not integers")
        if valid_cells + failure_cells != EXPECTED_CELLS:
            raise ValueError(f"{dataset_id}: valid and failure cells do not cover the matrix")

        runtime = run.get("runtime_evidence")
        replay_runtime = run.get("replay_runtime_evidence")
        if not isinstance(runtime, dict) or not isinstance(replay_runtime, dict):
            raise TypeError(f"{dataset_id}: runtime evidence is incomplete")
        runtime_seconds = float(runtime["wall_clock_seconds"])
        replay_seconds = float(replay_runtime["wall_clock_seconds"])
        replay_budget = float(replay_runtime["runtime_budget_seconds"])
        if runtime_seconds > RUNTIME_BUDGET_SECONDS:
            raise ValueError(f"{dataset_id}: first-pass runtime exceeds the Gate C budget")
        if replay_seconds > replay_budget or replay_runtime.get("budget_exceeded"):
            raise ValueError(f"{dataset_id}: replay runtime exceeds its recorded budget")

        result_path = ROOT / Path(run["results_path"])
        run_dir = result_path.parent
        replay_dir = run_dir.with_name(run_dir.name.removesuffix("-gatea") + "-replay")
        try:
            replay_matches = _replay_matches(run_dir, replay_dir)
        except (OSError, KeyError, TypeError, ValueError):
            replay_matches = False
        if not replay_matches:
            raise ValueError(f"{dataset_id}: replay results or failures do not match")

        total_valid += valid_cells
        total_failures += failure_cells
        max_runtime = max(max_runtime, runtime_seconds)
        max_replay_runtime = max(max_replay_runtime, replay_seconds)
        run_summaries.append(
            {
                "dataset_id": dataset_id,
                "expected_cells": EXPECTED_CELLS,
                "valid_cells": valid_cells,
                "failure_cells": failure_cells,
                "deterministic_replay": "passed",
                "failure_review": "passed",
                "replay_runtime_seconds": replay_seconds,
                "replay_runtime_budget_seconds": replay_budget,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": GATE_C_SCHEMA,
        "checked_at_utc": dt.datetime.now(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "passed",
        "scope_lock": {"path": relative(scope_lock_path), "sha256": sha256_file(scope_lock_path)},
        "pilot_review": {"path": relative(review_path), "sha256": sha256_file(review_path)},
        "dataset_count": len(dataset_ids),
        "dataset_ids": dataset_ids,
        "expected_cells_per_dataset": EXPECTED_CELLS,
        "total_valid_cells": total_valid,
        "total_failure_cells": total_failures,
        "maximum_first_pass_runtime_seconds": max_runtime,
        "maximum_replay_runtime_seconds": max_replay_runtime,
        "runs": run_summaries,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review", type=Path, default=ROOT / "artifacts" / "runs" / "stage0-pilot-review.json"
    )
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "dataset_registry.csv")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data" / "acquisition_manifest.csv"
    )
    parser.add_argument(
        "--scope-lock", type=Path, default=ROOT / "artifacts" / "runs" / "scope-lock.json"
    )
    parser.add_argument("--output", type=Path, help="Optional path for the generated review payload")
    args = parser.parse_args()
    payload = verify_gate_c(args.review, args.registry, args.manifest, args.scope_lock)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
