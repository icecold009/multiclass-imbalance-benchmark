"""Audit duplicate predictor profiles and preflight grouped CV without fitting models."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.duplicate_profiles import (
    GROUPED_SENSITIVITY_DATASETS,
    audit_dataset_profiles,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def run_audit(
    registry_path: Path,
    acquisition_path: Path,
    output_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    folds: int = 5,
) -> dict[str, Any]:
    registry = _read_csv(registry_path, "dataset registry")
    acquisition = _read_csv(acquisition_path, "acquisition manifest")
    for frame, columns, label in (
        (
            registry,
            {"dataset_id", "target_column", "n_rows", "raw_sha256", "eligible", "has_missing"},
            "registry",
        ),
        (acquisition, {"dataset_id", "local_path", "raw_sha256"}, "acquisition manifest"),
    ):
        missing = sorted(columns - set(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing required columns: {missing}")
        if frame["dataset_id"].duplicated().any():
            raise ValueError(f"{label} contains duplicate dataset IDs")

    registry_by_id = registry.set_index("dataset_id")
    acquisition_by_id = acquisition.set_index("dataset_id")
    dataset_ids = tuple(
        registry.loc[registry["eligible"].str.casefold().eq("true"), "dataset_id"].astype(str)
    )
    if len(dataset_ids) != 11:
        raise ValueError(f"expected the locked 11 eligible datasets, found {len(dataset_ids)}")
    if set(dataset_ids) - set(acquisition_by_id.index.astype(str)):
        raise ValueError("acquisition manifest does not cover every eligible dataset")

    summaries: list[dict[str, Any]] = []
    rowwise_splits: list[dict[str, Any]] = []
    grouped_splits: list[dict[str, Any]] = []
    raw_inputs: dict[str, dict[str, str]] = {}
    for dataset_id in dataset_ids:
        registry_row = registry_by_id.loc[dataset_id]
        acquisition_row = acquisition_by_id.loc[dataset_id]
        registry_hash = str(registry_row["raw_sha256"]).lower()
        acquisition_hash = str(acquisition_row["raw_sha256"]).lower()
        if registry_hash != acquisition_hash:
            raise ValueError(f"{dataset_id}: registry and acquisition hashes disagree")
        local_path = ROOT / Path(str(acquisition_row["local_path"]))
        actual_hash = sha256_file(local_path)
        if actual_hash != registry_hash:
            raise ValueError(f"{dataset_id}: raw CSV hash does not match frozen references")
        frame = pd.read_csv(local_path)
        target = str(registry_row["target_column"])
        if len(frame) != int(registry_row["n_rows"]):
            raise ValueError(f"{dataset_id}: raw row count differs from the registry")
        summary, original_splits, grouped_preflight = audit_dataset_profiles(
            dataset_id, frame, target, seeds=seeds, folds=folds
        )
        registry_missing = str(registry_row["has_missing"]).strip().casefold()
        if registry_missing not in {"true", "false"}:
            raise ValueError(f"{dataset_id}: registry has_missing must be True or False")
        summary["registry_has_missing_flag"] = registry_missing == "true"
        summary["registry_missing_flag_matches_raw"] = (
            (registry_missing == "true") == (summary["missing_predictor_values"] > 0)
        )
        summary["raw_sha256"] = actual_hash
        summaries.append(summary)
        rowwise_splits.extend(original_splits)
        grouped_splits.extend(grouped_preflight)
        raw_inputs[dataset_id] = {
            "path": str(acquisition_row["local_path"]),
            "sha256": actual_hash,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = pd.DataFrame(
        {
            **summary,
            "profile_groups_per_class": json.dumps(
                summary["profile_groups_per_class"], sort_keys=True
            ),
            "missing_predictor_columns": json.dumps(
                summary["missing_predictor_columns"], sort_keys=True
            ),
        }
        for summary in summaries
    )
    outputs = {
        "dataset_summary.csv": summary_csv,
        "rowwise_fold_overlap.csv": pd.DataFrame(rowwise_splits),
        "grouped_cv_preflight.csv": pd.DataFrame(grouped_splits),
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    receipt = {
        "schema_version": "duplicate-profile-audit-v1",
        "scope": "model-free exact predictor-profile audit and grouped-CV preflight",
        "seeds": list(seeds),
        "folds": folds,
        "grouped_sensitivity_datasets": list(GROUPED_SENSITIVITY_DATASETS),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "registry_sha256": sha256_file(registry_path),
        "acquisition_manifest_sha256": sha256_file(acquisition_path),
        "raw_inputs": raw_inputs,
        "outputs": {
            name: {
                "path": name,
                "sha256": sha256_file(output_dir / name),
                "rows": len(frame),
            }
            for name, frame in outputs.items()
        },
        "model_fitting_performed": False,
    }
    receipt_path = output_dir / "audit_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("data/dataset_registry.csv"))
    parser.add_argument(
        "--acquisition-manifest", type=Path, default=Path("data/acquisition_manifest.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/duplicate-profile-audit")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    receipt = run_audit(
        args.registry,
        args.acquisition_manifest,
        args.output_dir,
        seeds=tuple(args.seeds),
        folds=args.folds,
    )
    summary = pd.read_csv(args.output_dir / "dataset_summary.csv")
    print(summary.to_string(index=False))
    print(f"\nAudit receipt: {args.output_dir / 'audit_receipt.json'}")
    print(f"Model fitting performed: {receipt['model_fitting_performed']}")


if __name__ == "__main__":
    main()
