"""Gate E robustness checks for the locked benchmark analysis."""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import (
    EFFICIENCY_METRICS,
    FAMILY_DEFINITIONS,
    SCALAR_METRICS,
    _family_classifiers,
    run_analysis,
    verify_gate_d,
)
from src.benchmark import _read_json, _write_json_atomic, sha256_file

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
)
PROCEDURES = {
    "friedman": "complete dataset blocks, at least three conditions and three blocks",
    "nemenyi": "all pairs after a significant Friedman test, alpha 0.05",
    "wilcoxon": {
        "comparison": "raw versus every alternative in the same family",
        "alternative": "two-sided",
        "zero_method": "wilcox",
        "correction": False,
        "method": "auto",
    },
    "holm": "within family, classifier, and metric across Wilcoxon p-values",
    "rank_biserial": "paired signed ranks after removing zero differences",
    "bootstrap": "paired percentile over complete dataset blocks",
}
PER_CLASS_NOTE = (
    "descriptive class-wise summaries; inferential matrices skipped because class labels are dataset-specific"
)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reference(path: Path, root: Path) -> dict[str, str]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Gate E artifact is outside the repository: {path}") from error
    return {"path": relative, "sha256": sha256_file(path)}


def _resolve_artifact(root: Path, reference: object, label: str) -> Path:
    if not isinstance(reference, dict):
        raise TypeError(f"{label} reference is not an object")
    relative = reference.get("path")
    digest = reference.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise TypeError(f"{label} reference is incomplete")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} reference escapes the repository: {relative}") from error
    if not path.is_file():
        raise RuntimeError(f"{label} artifact is missing: {path}")
    if sha256_file(path) != digest:
        raise RuntimeError(f"{label} artifact hash does not match its manifest")
    return path


def _assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"Gate E mismatch for {label}: expected {expected!r}, got {actual!r}")


def _finite_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        if column not in frame.columns:
            raise RuntimeError(f"{label} is missing column {column!r}")
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{label}.{column} contains non-finite values")


def _validate_family_rows(frame: pd.DataFrame, label: str) -> None:
    for _, row in frame.iterrows():
        family = str(row["family"])
        classifier = str(row["classifier"])
        condition = str(row["condition"])
        if family not in FAMILY_DEFINITIONS:
            raise RuntimeError(f"{label} contains an unknown family: {family}")
        if classifier not in _family_classifiers(family):
            raise RuntimeError(f"{label} contains a classifier outside its family: {classifier}")
        if condition not in FAMILY_DEFINITIONS[family][1] or condition == "raw":
            raise RuntimeError(f"{label} contains an invalid condition: {condition}")


def _key_set(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> set[tuple[str, ...]]:
    if any(column not in frame.columns for column in columns):
        raise RuntimeError(f"{label} is missing a required key column")
    return {tuple(str(value) for value in row) for row in frame[list(columns)].itertuples(index=False)}


def _validate_tables(
    analysis_dir: Path,
    manifest: dict[str, Any],
    dataset_ids: list[str],
) -> dict[str, int]:
    tables = {name: pd.read_csv(analysis_dir / name) for name in ANALYSIS_TABLES}
    cell_keys = ("dataset", "classifier", "condition", "feature_type")
    cells = tables["cell_means.csv"]
    if len(cells) != int(manifest["cell_mean_count"]):
        raise RuntimeError("cell_means row count does not match the analysis manifest")
    if cells[list(cell_keys)].duplicated().any():
        raise RuntimeError("cell_means contains duplicate complete cells")
    if set(cells["dataset"].astype(str)) - set(dataset_ids):
        raise RuntimeError("cell_means contains a dataset outside the locked scope")
    if set(cells["complete_records"].astype(int)) != {15}:
        raise RuntimeError("cell_means contains a non-15-record cell")
    _finite_columns(cells, SCALAR_METRICS, "cell_means")

    class_means = tables["class_means.csv"]
    if len(class_means) != int(manifest["class_mean_count"]):
        raise RuntimeError("class_means row count does not match the analysis manifest")
    class_keys = cell_keys + ("class_index",)
    if class_means[list(class_keys)].duplicated().any():
        raise RuntimeError("class_means contains duplicate cell/class rows")
    _finite_columns(class_means, ("class_index", "recall", "complete_records"), "class_means")
    efficiency = tables["efficiency_means.csv"]
    if len(efficiency) != int(manifest["efficiency_mean_count"]):
        raise RuntimeError("efficiency_means row count does not match the analysis manifest")
    if set(efficiency["complete_records"].astype(int)) != {15}:
        raise RuntimeError("efficiency_means contains a non-15-record cell")
    if _key_set(efficiency, cell_keys, "efficiency_means") != _key_set(cells, cell_keys, "cell_means"):
        raise RuntimeError("efficiency_means keys do not match complete cells")
    efficiency_columns = tuple(f"{metric}_{statistic}" for metric in EFFICIENCY_METRICS for statistic in ("mean", "median"))
    _finite_columns(efficiency, efficiency_columns, "efficiency_means")

    deltas = tables["baseline_deltas.csv"]
    if len(deltas) != int(manifest["baseline_delta_count"]):
        raise RuntimeError("baseline_deltas row count does not match the analysis manifest")
    if (deltas["condition"] == "raw").any():
        raise RuntimeError("baseline_deltas contains a raw self-comparison")
    _validate_family_rows(deltas, "baseline_deltas")
    _finite_columns(deltas, ("raw_mean", "alternative_mean", "delta"), "baseline_deltas")

    class_deltas = tables["baseline_deltas_per_class.csv"]
    if len(class_deltas) != int(manifest["per_class_delta_count"]):
        raise RuntimeError("baseline_deltas_per_class row count does not match the analysis manifest")
    _finite_columns(
        class_deltas,
        ("class_index", "raw_recall", "alternative_recall", "delta"),
        "baseline_deltas_per_class",
    )
    _validate_family_rows(class_deltas, "baseline_deltas_per_class")
    class_summary = tables["per_class_recall_summary.csv"]
    _finite_columns(
        class_summary,
        ("class_index", "n_pairs", "mean_delta", "median_delta", "wins", "ties", "losses"),
        "per_class_recall_summary",
    )

    friedman = tables["friedman_tests.csv"]
    expected_friedman = sum(
        len(_family_classifiers(family)) * len(SCALAR_METRICS) for family in FAMILY_DEFINITIONS
    )
    if len(friedman) != expected_friedman or len(friedman) != int(manifest["friedman_test_count"]):
        raise RuntimeError("friedman_tests row count does not match the frozen family matrix")
    for _, row in friedman.iterrows():
        family = str(row["family"])
        classifier = str(row["classifier"])
        metric = str(row["metric"])
        if family not in FAMILY_DEFINITIONS:
            raise RuntimeError(f"Friedman output contains an unknown family: {family}")
        if classifier not in _family_classifiers(family):
            raise RuntimeError(f"Friedman output contains a classifier outside its family: {classifier}")
        if metric not in SCALAR_METRICS:
            raise RuntimeError(f"Friedman output contains an unknown metric: {metric}")
        status = str(row["status"])
        if status not in {"computed", "skipped"}:
            raise RuntimeError(f"unknown Friedman status: {status}")
        if status == "computed":
            if not (int(row["n_blocks"]) >= 3 and int(row["n_conditions"]) >= 3):
                raise RuntimeError("computed Friedman test does not meet minimum dimensions")
            if not 0 <= float(row["friedman_p_value"]) <= 1:
                raise RuntimeError("computed Friedman p-value is outside [0, 1]")
        else:
            if not str(row["reason"]):
                raise RuntimeError("skipped Friedman test has no reason")
            if pd.notna(row["friedman_statistic"]) or pd.notna(row["friedman_p_value"]):
                raise RuntimeError("skipped Friedman test contains a statistic or p-value")

    significant: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for _, row in friedman.iterrows():
        if str(row["status"]) == "computed" and float(row["friedman_p_value"]) < 0.05:
            family = str(row["family"])
            significant[(family, str(row["classifier"]), str(row["metric"]))] = FAMILY_DEFINITIONS[family][1]
    nemenyi = tables["nemenyi_pvalues.csv"]
    if len(nemenyi) != int(manifest["nemenyi_pair_count"]):
        raise RuntimeError("nemenyi_pvalues row count does not match the analysis manifest")
    seen_pairs: set[tuple[str, str, str, str, str]] = set()
    for _, row in nemenyi.iterrows():
        key = (str(row["family"]), str(row["classifier"]), str(row["metric"]))
        if key not in significant:
            raise RuntimeError("Nemenyi output exists for a non-significant Friedman block")
        pair = (key[0], key[1], key[2], str(row["condition_a"]), str(row["condition_b"]))
        if pair in seen_pairs or str(row["condition_a"]) == str(row["condition_b"]):
            raise RuntimeError("Nemenyi output contains a duplicate or diagonal pair")
        if pair[3] not in significant[key] or pair[4] not in significant[key]:
            raise RuntimeError("Nemenyi output contains a condition outside its family")
        seen_pairs.add(pair)
        if not 0 <= float(row["p_value"]) <= 1:
            raise RuntimeError("Nemenyi p-value is outside [0, 1]")
    for key, conditions in significant.items():
        count = sum(1 for pair in seen_pairs if pair[:3] == key)
        if count != len(conditions) * (len(conditions) - 1) // 2:
            raise RuntimeError(f"Nemenyi pair count is incomplete for {key}")

    wilcoxon = tables["wilcoxon_effects.csv"]
    if len(wilcoxon) != int(manifest["wilcoxon_comparison_count"]):
        raise RuntimeError("wilcoxon_effects row count does not match the analysis manifest")
    for _, row in wilcoxon.iterrows():
        if str(row["condition"]) == "raw":
            raise RuntimeError("Wilcoxon output contains a raw self-comparison")
        if int(row["n_pairs"]) < 0:
            raise RuntimeError("Wilcoxon output contains a negative pair count")
    _validate_family_rows(wilcoxon, "wilcoxon_effects")
    for _, row in wilcoxon.iterrows():
        status = str(row["status"])
        if status == "computed":
            for column in ("wilcoxon_p_value", "holm_p_value", "bootstrap_low", "bootstrap_high"):
                value = float(row[column])
                if not np.isfinite(value):
                    raise RuntimeError(f"Wilcoxon output has an invalid {column}")
                if column in {"wilcoxon_p_value", "holm_p_value"} and not 0 <= value <= 1:
                    raise RuntimeError(f"Wilcoxon output has an invalid {column}")
            if float(row["bootstrap_low"]) > float(row["bootstrap_high"]):
                raise RuntimeError("Wilcoxon bootstrap interval is reversed")
        elif status != "skipped" or not str(row["reason"]):
            raise RuntimeError("Wilcoxon output has an invalid skipped status")

    moderators = tables["moderators.csv"]
    if set(moderators["dataset_id"].astype(str)) != set(dataset_ids):
        raise RuntimeError("moderators do not cover exactly the locked datasets")
    return {name.removesuffix(".csv"): len(frame) for name, frame in tables.items()}


def _replay_check(
    root: Path,
    run_dir: Path,
    analysis_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stage0-gate-e-") as temporary:
        replay_dir = Path(temporary) / "analysis"
        replay_manifest = run_analysis(root, run_dir, replay_dir)
        for field in (
            "status",
            "dataset_ids",
            "cell_mean_count",
            "class_mean_count",
            "efficiency_mean_count",
            "baseline_delta_count",
            "per_class_delta_count",
            "friedman_test_count",
            "nemenyi_pair_count",
            "wilcoxon_comparison_count",
            "bootstrap_resamples",
            "bootstrap_seed",
            "alpha",
            "family_definitions",
            "procedures",
            "per_class_recall_inference",
            "figures",
        ):
            _assert_equal(replay_manifest.get(field), manifest.get(field), f"replay {field}")
        compared: list[str] = []
        for name in ANALYSIS_TABLES:
            original = analysis_dir / name
            replay = replay_dir / name
            if sha256_file(original) != sha256_file(replay):
                raise RuntimeError(f"deterministic replay changed {name}")
            compared.append(name)
        for figure in manifest["figures"]:
            if figure["status"] != "generated":
                continue
            original = analysis_dir / "figures" / figure["name"]
            replay = replay_dir / "figures" / figure["name"]
            if sha256_file(original) != sha256_file(replay):
                raise RuntimeError(f"deterministic replay changed figure {figure['name']}")
            compared.append(f"figures/{figure['name']}")
    return {"status": "passed", "compared_artifacts": compared}


def verify_gate_e(
    root: Path,
    run_dir: Path,
    analysis_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and replay the locked analysis before paper drafting."""

    analysis_manifest_path = analysis_dir / "analysis_manifest.json"
    manifest = _read_json(analysis_manifest_path, "analysis manifest")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"analysis manifest is not complete: {manifest.get('status')!r}")
    fresh_gate = verify_gate_d(root, run_dir)
    analysis_gate = manifest.get("gate_d")
    if not isinstance(analysis_gate, dict):
        raise TypeError("analysis manifest has no Gate D payload")
    for field in (
        "status",
        "benchmark_manifest",
        "dataset_ids",
        "dataset_count",
        "seeds",
        "folds",
        "expected_cells_per_dataset",
        "counts",
        "aggregate",
        "frozen_references",
        "selection_rule",
    ):
        _assert_equal(analysis_gate.get(field), fresh_gate.get(field), f"Gate D {field}")
    _assert_equal(manifest.get("input_run_manifest"), fresh_gate.get("benchmark_manifest"), "input run manifest")
    _assert_equal(manifest.get("per_class_recall_inference"), PER_CLASS_NOTE, "per-class inference boundary")
    _assert_equal(manifest.get("procedures"), PROCEDURES, "analysis procedures")
    expected_families = {
        family: {
            "feature_type": feature_type,
            "conditions": list(conditions),
            "classifiers": list(_family_classifiers(family)),
        }
        for family, (feature_type, conditions) in FAMILY_DEFINITIONS.items()
    }
    _assert_equal(manifest.get("family_definitions"), expected_families, "family definitions")
    for key, reference in manifest.get("artifacts", {}).items():
        if not isinstance(reference, dict) or reference.get("path") != key:
            raise RuntimeError(f"analysis artifact key/reference mismatch: {key}")
        _resolve_artifact(root, reference, key)
    required = {"gate-d-review.json", *ANALYSIS_TABLES}
    actual = set(manifest.get("artifacts", {}))
    expected_prefix = "results/analysis/"
    if {name.removeprefix(expected_prefix) for name in actual if name.startswith(expected_prefix)} != required | {
        f"figures/{figure['name']}" for figure in manifest.get("figures", []) if figure["status"] == "generated"
    }:
        raise RuntimeError("analysis manifest artifact set is incomplete or unexpected")
    counts = _validate_tables(analysis_dir, manifest, list(fresh_gate["dataset_ids"]))
    replay = _replay_check(root, run_dir, analysis_dir, manifest)
    payload: dict[str, Any] = {
        "schema_version": "gate-e-review-v1",
        "status": "passed",
        "reviewed_at_utc": _utc_now(),
        "analysis_manifest": _reference(analysis_manifest_path, root),
        "gate_d": {
            "status": fresh_gate["status"],
            "counts": fresh_gate["counts"],
            "dataset_ids": fresh_gate["dataset_ids"],
            "benchmark_manifest": fresh_gate["benchmark_manifest"],
        },
        "analysis_counts": counts,
        "deterministic_replay": replay,
        "decision": "analysis frozen for independent robustness review and paper drafting",
        "claim_boundary": "no paper claims, scope changes, or protocol amendments are authorized by this payload",
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output_path, payload)
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate and replay the locked analysis")
    parser.add_argument("--run-dir", type=Path, default=Path("results/full-run"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    analysis_dir = args.analysis_dir if args.analysis_dir.is_absolute() else root / args.analysis_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    payload = verify_gate_e(root, run_dir, analysis_dir, output_dir / "gate-e-review.json")
    print(payload["status"])
    print(output_dir / "gate-e-review.json")


if __name__ == "__main__":
    main()
