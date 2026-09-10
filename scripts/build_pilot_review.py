"""Build the hashed Stage 0 pilot-review manifest from local pilot outputs.

This records output completeness, runtime, and peak RSS evidence without
using pilot scores to select datasets or methods. Deterministic replay and the
human scope decision remain explicit review fields until independently signed
off.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CELLS = 3 * 5 * 3 * 11
RUNTIME_BUDGET_SECONDS = 900
REQUIRED_ARTIFACTS = (
    "pilot_failures.csv",
    "pilot_friedman_summary.csv",
    "pilot_metadata.json",
    "pilot_rankings.csv",
    "pilot_results.csv",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def file_reference(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def _replay_matches(first: Path, replay: Path) -> bool:
    result_columns = [
        "dataset", "seed", "fold", "classifier", "condition", "feature_type",
        "macro_f1", "g_mean", "mcc", "balanced_accuracy",
        "train_rows_before_sampling", "train_rows_after_sampling",
    ]
    failure_columns = [
        "dataset", "seed", "fold", "classifier", "condition", "feature_type",
        "stage", "error_type", "error",
    ]
    first_results = pd.read_csv(first / "pilot_results.csv")
    replay_results = pd.read_csv(replay / "pilot_results.csv")
    first_failures = pd.read_csv(first / "pilot_failures.csv")
    replay_failures = pd.read_csv(replay / "pilot_failures.csv")
    for columns, left, right in (
        (result_columns, first_results, replay_results),
        (failure_columns, first_failures, replay_failures),
    ):
        if any(column not in left.columns or column not in right.columns for column in columns):
            return False
        if len(left) != len(right):
            return False
        if not left[columns].reset_index(drop=True).equals(right[columns].reset_index(drop=True)):
            return False
    return True


def build_run(
    dataset_id: str,
    run_dir: Path,
    replay_root: Path | None,
    failure_review: str,
) -> dict[str, object]:
    paths = {name: run_dir / name for name in REQUIRED_ARTIFACTS}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{dataset_id}: missing pilot artifact(s): {', '.join(missing)}")

    results = pd.read_csv(paths["pilot_results.csv"])
    failures = pd.read_csv(paths["pilot_failures.csv"])
    if len(results) + len(failures) != EXPECTED_CELLS:
        raise ValueError(
            f"{dataset_id}: output cells {len(results) + len(failures)} "
            f"do not equal {EXPECTED_CELLS}"
        )
    key_columns = ["dataset", "seed", "fold", "classifier", "condition"]
    combined = pd.concat([results[key_columns], failures[key_columns]], ignore_index=True)
    if combined.duplicated().any():
        raise ValueError(f"{dataset_id}: duplicate pilot comparison cell")

    started = dt.datetime.fromtimestamp(run_dir.stat().st_ctime, tz=dt.UTC)
    finished = dt.datetime.fromtimestamp(run_dir.stat().st_mtime, tz=dt.UTC)
    wall_clock_seconds = max(0.0, (finished - started).total_seconds())
    rss_values = pd.to_numeric(results.get("rss_after_mb", pd.Series(dtype=float)), errors="coerce")
    peak_rss = float(rss_values.max()) if not rss_values.dropna().empty else 0.0
    artifact_refs = [file_reference(paths[name]) for name in REQUIRED_ARTIFACTS]
    results_ref = file_reference(paths["pilot_results.csv"])
    replay_passed = replay_root is not None and _replay_matches(
        run_dir, replay_root / f"pilot-{dataset_id}-replay"
    )
    replay_dir = replay_root / f"pilot-{dataset_id}-replay" if replay_root is not None else None
    replay_results_path = replay_dir / "pilot_results.csv" if replay_dir is not None else None
    replay_runtime = None
    if replay_dir is not None and replay_dir.is_dir() and replay_results_path.is_file():
        replay_runtime = round(
            max(0.0, (replay_dir.stat().st_mtime - replay_dir.stat().st_ctime)), 1
        )
    run = {
        "dataset_id": dataset_id,
        "artifacts": artifact_refs,
        "results_path": results_ref["path"],
        "failures_path": relative(paths["pilot_failures.csv"]),
        "runtime_evidence": {
            **results_ref,
            "wall_clock_seconds": round(wall_clock_seconds, 1),
        },
        "memory_evidence": {
            **results_ref,
            "method": "psutil process RSS sampled before and after each cell; peak of recorded RSS",
            "peak_mb": round(peak_rss, 3),
        },
        "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
        "deterministic_replay": "passed" if replay_passed else "pending",
        "failure_review": failure_review,
        "output_completeness": "complete",
        "expected_cells": EXPECTED_CELLS,
        "valid_cells": len(results),
        "failure_cells": len(failures),
    }
    if replay_results_path is not None and replay_results_path.is_file() and replay_runtime is not None:
        replay_reference = file_reference(replay_results_path)
        run["replay_runtime_evidence"] = {
            **replay_reference,
            "wall_clock_seconds": replay_runtime,
            "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
            "budget_exceeded": replay_runtime > RUNTIME_BUDGET_SECONDS,
        }
    return run


def build(
    output_path: Path,
    run_root: Path,
    dataset_ids: list[str],
    replay_root: Path | None,
    failure_review: str,
) -> None:
    environment_path = ROOT / "artifacts" / "environment" / "metadata.json"
    configuration_path = ROOT / "config" / "stage0.yaml"
    for path in (environment_path, configuration_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "schema_version": "stage0-pilot-review-v1",
        "reviewed_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "registry_sha256": sha256_file(ROOT / "data" / "dataset_registry.csv"),
        "manifest_sha256": sha256_file(ROOT / "data" / "acquisition_manifest.csv"),
        "environment": file_reference(environment_path),
        "configuration": file_reference(configuration_path),
        "runs": [
            build_run(
                dataset_id,
                run_root / f"pilot-{dataset_id}-gatea",
                replay_root,
                failure_review,
            )
            for dataset_id in dataset_ids
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "runs" / "stage0-pilot-review.json")
    parser.add_argument("--run-root", type=Path, default=ROOT / "artifacts" / "runs")
    parser.add_argument("--replay-root", type=Path, help="Root containing pilot-<dataset>-replay outputs")
    parser.add_argument(
        "--failure-review",
        choices=("pending", "passed"),
        default="pending",
        help="Set to passed only after inspecting every failure row",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "yeast", "cmc", "glass", "balance_scale", "mfeat_factors", "optdigits",
            "dermatology", "iris", "wine", "cnae_9", "seeds", "wine_quality_red",
        ],
    )
    args = parser.parse_args()
    build(args.output, args.run_root, args.datasets, args.replay_root, args.failure_review)
    print(args.output)


if __name__ == "__main__":
    main()
