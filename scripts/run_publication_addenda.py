"""Reproduce the separate V4 exact Friedman permutation addendum."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANALYSIS_MANIFEST = Path("results/analysis/analysis_manifest.json")
CELL_MEANS = Path("results/analysis/cell_means.csv")
EXPECTED_ANALYSIS_MANIFEST_SHA256 = "f398bc3dcd53eefc05e7ce8b96d2ae5736a8d0e50a7cb6f594e79557b4ecdb3b"
EXPECTED_CELL_MEANS_SHA256 = "868aa61153ff8fa2d7fbae094608eb514333354ea7d0d17700269ce8a4c0195a"
OUTPUT_CSV = "exact_friedman_permutation.csv"
OUTPUT_MANIFEST = "publication_addendum_manifest.json"
EXPECTED_DATASETS = ("cnae_9", "iris", "seeds")
CONDITIONS = (
    "raw",
    "class_weighted",
    "random_over",
    "random_under",
    "smote",
    "adasyn",
    "borderline_smote",
    "smoteenn",
    "smotetomek",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_matrix_sha256(matrix: pd.DataFrame) -> str:
    canonical = matrix.to_csv(
        index=True,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_frozen_matrix(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    manifest_path = root / ANALYSIS_MANIFEST
    cell_means_path = root / CELL_MEANS
    manifest_sha256 = _sha256(manifest_path)
    cell_means_sha256 = _sha256(cell_means_path)
    if manifest_sha256 != EXPECTED_ANALYSIS_MANIFEST_SHA256:
        raise RuntimeError("frozen analysis manifest SHA-256 does not match the V4 source")
    if cell_means_sha256 != EXPECTED_CELL_MEANS_SHA256:
        raise RuntimeError("frozen cell_means.csv SHA-256 does not match the V4 source")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("artifacts", {}).get(CELL_MEANS.as_posix())
    if not isinstance(recorded, dict) or recorded.get("sha256") != cell_means_sha256:
        raise RuntimeError("cell_means.csv does not match its analysis-manifest reference")

    means = pd.read_csv(cell_means_path)
    selection = means.loc[
        means["feature_type"].eq("numeric")
        & means["classifier"].eq("xgboost")
        & means["dataset"].isin(EXPECTED_DATASETS)
        & means["condition"].isin(CONDITIONS),
        ["dataset", "condition", "macro_f1"],
    ].copy()
    if selection.duplicated(["dataset", "condition"]).any():
        raise RuntimeError("frozen cell means contain duplicate selected dataset-condition rows")
    if set(selection["dataset"]) != set(EXPECTED_DATASETS):
        raise RuntimeError("selected frozen matrix does not contain the expected datasets")
    if set(selection["condition"]) != set(CONDITIONS) or len(selection) != len(EXPECTED_DATASETS) * len(CONDITIONS):
        raise RuntimeError("selected frozen matrix does not contain all nine expected conditions")

    matrix = selection.pivot(index="dataset", columns="condition", values="macro_f1")
    matrix = matrix.reindex(index=EXPECTED_DATASETS, columns=CONDITIONS)
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise RuntimeError("selected frozen macro-F1 matrix is incomplete or non-finite")
    return matrix, {
        "analysis_manifest_sha256": manifest_sha256,
        "cell_means_sha256": cell_means_sha256,
        "selected_matrix_sha256": _canonical_matrix_sha256(matrix),
    }


def run_addendum(root: Path, output_dir: Path) -> dict[str, Any]:
    """Validate the frozen matrix, enumerate the exact check, and write outputs."""

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.sensitivity import _friedman_rank_statistic, exact_friedman_permutation_sensitivity

    matrix, input_hashes = _load_frozen_matrix(root)
    ranks = np.vstack([rankdata(row, method="average") for row in matrix.to_numpy(dtype=float)])
    expected_rank_pattern = np.asarray([*([5.0] * 7), 1.0, 9.0])
    if not np.array_equal(ranks, np.tile(expected_rank_pattern, (len(EXPECTED_DATASETS), 1))):
        raise RuntimeError("frozen matrix no longer has the ordered [5 x 7, 1, 9] rank pattern")

    statistic, tie_correction = _friedman_rank_statistic(ranks)
    if not np.isclose(statistic, 24.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("frozen macro-F1 matrix no longer has Friedman statistic 24")
    if not np.isclose(tie_correction, 8.0 / 15.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("frozen macro-F1 matrix no longer has the declared tie correction")

    exact = exact_friedman_permutation_sensitivity(matrix)
    expected_counts = {
        "reduced_total_states": 5_184,
        "reduced_extreme_count": 1,
        "full_total_states": 373_248,
        "full_extreme_count": 72,
    }
    if any(exact[name] != value for name, value in expected_counts.items()):
        raise RuntimeError("exact permutation enumeration no longer matches the frozen V4 result")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / OUTPUT_CSV
    manifest_path = output_dir / OUTPUT_MANIFEST
    exact_p = float(exact["reduced_exact_p_value"])
    output_row = {
        "post_hoc": True,
        "family": "numeric_core",
        "classifier": "xgboost",
        "metric": "macro_f1",
        "dataset_order": json.dumps(list(EXPECTED_DATASETS), separators=(",", ":")),
        "condition_order": json.dumps(list(CONDITIONS), separators=(",", ":")),
        "rank_pattern_per_dataset": json.dumps([5, 5, 5, 5, 5, 5, 5, 1, 9], separators=(",", ":")),
        "friedman_statistic": float(exact["friedman_statistic"]),
        "tie_correction": float(exact["tie_correction"]),
        "unique_rank_arrangements_per_block": 72,
        "reduced_total_states": int(exact["reduced_total_states"]),
        "reduced_extreme_count": int(exact["reduced_extreme_count"]),
        "full_total_states": int(exact["full_total_states"]),
        "full_extreme_count": int(exact["full_extreme_count"]),
        "exact_p_value_fraction": "1/5184",
        "exact_p_value": exact_p,
        "enumeration_method": str(exact["enumeration_method"]),
        "symmetry_reduction": str(exact["symmetry_reduction"]),
        "input_cell_means_sha256": input_hashes["cell_means_sha256"],
        "selected_matrix_sha256": input_hashes["selected_matrix_sha256"],
    }
    pd.DataFrame([output_row]).to_csv(csv_path, index=False, lineterminator="\n")
    manifest: dict[str, Any] = {
        "schema_version": "publication-addendum-exact-friedman-v1",
        "status": "complete",
        "post_hoc": True,
        "interpretation": (
            "Additive exact calibration of the frozen three-block XGBoost macro-F1 omnibus; "
            "the planned asymptotic analysis and prior Monte Carlo record are unchanged."
        ),
        "input": {
            "analysis_manifest": {"path": ANALYSIS_MANIFEST.as_posix(), "sha256": input_hashes["analysis_manifest_sha256"]},
            "cell_means": {"path": CELL_MEANS.as_posix(), "sha256": input_hashes["cell_means_sha256"]},
            "selected_matrix_sha256": input_hashes["selected_matrix_sha256"],
            "family": "numeric_core",
            "feature_type": "numeric",
            "classifier": "xgboost",
            "metric": "macro_f1",
            "dataset_order": list(EXPECTED_DATASETS),
            "condition_order": list(CONDITIONS),
        },
        "result": {
            "n_blocks": int(exact["n_blocks"]),
            "n_conditions": int(exact["n_conditions"]),
            "rank_pattern_per_dataset": [5, 5, 5, 5, 5, 5, 5, 1, 9],
            "friedman_statistic": float(exact["friedman_statistic"]),
            "tie_correction": float(exact["tie_correction"]),
            "unique_rank_arrangements_per_block": exact["unique_rank_arrangements_per_block"],
            "reduced_total_states": int(exact["reduced_total_states"]),
            "reduced_extreme_count": int(exact["reduced_extreme_count"]),
            "full_total_states": int(exact["full_total_states"]),
            "full_extreme_count": int(exact["full_extreme_count"]),
            "exact_p_value_fraction": "1/5184",
            "exact_p_value": exact_p,
            "symmetry_reduction": exact["symmetry_reduction"],
            "enumeration_method": exact["enumeration_method"],
        },
        "outputs": {
            OUTPUT_CSV: {
                "path": csv_path.name,
                "sha256": _sha256(csv_path),
            }
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/publication-addenda/v4"),
        help="separate output directory for this exact post hoc calibration",
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    manifest = run_addendum(ROOT, output_dir)
    print(json.dumps({"status": manifest["status"], "manifest": str(output_dir / OUTPUT_MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
