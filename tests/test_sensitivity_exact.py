from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from scripts import run_publication_addenda as publication_addenda
from src.reproducibility import _validate_pinned_analysis_manifest
from src.sensitivity import exact_friedman_permutation_sensitivity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_addendum_inputs(root: Path, monkeypatch) -> str:
    """Create deterministic synthetic inputs for exact-calibration unit tests."""

    monkeypatch.setattr(
        publication_addenda,
        "EXPECTED_DATASETS",
        ("synthetic-dataset-a", "synthetic-dataset-b", "synthetic-dataset-c"),
    )
    analysis_dir = root / "results" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    values = [0.5] * 7 + [0.1, 0.9]
    rows = [
        {
            "feature_type": "numeric",
            "classifier": "xgboost",
            "dataset": dataset,
            "condition": condition,
            "macro_f1": value,
        }
        for dataset in publication_addenda.EXPECTED_DATASETS
        for condition, value in zip(publication_addenda.CONDITIONS, values, strict=True)
    ]
    cell_means_path = analysis_dir / "cell_means.csv"
    pd.DataFrame(rows).to_csv(cell_means_path, index=False, lineterminator="\n")
    cell_means_sha256 = _sha256(cell_means_path)
    manifest = {
        "artifacts": {
            publication_addenda.CELL_MEANS.as_posix(): {
                "path": publication_addenda.CELL_MEANS.as_posix(),
                "sha256": cell_means_sha256,
            }
        }
    }
    manifest_path = analysis_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cell_means_sha256


def _independent_friedman_statistic(ranks: np.ndarray) -> tuple[float, float]:
    blocks, conditions = ranks.shape
    correction = 1.0 - sum(
        sum(count**3 - count for count in Counter(row).values())
        for row in ranks
    ) / (blocks * (conditions**3 - conditions))
    rank_sums = ranks.sum(axis=0)
    statistic = (
        12.0 / (blocks * conditions * (conditions + 1)) * np.square(rank_sums).sum()
        - 3.0 * blocks * (conditions + 1)
    ) / correction
    return float(statistic), float(correction)


def _independent_full_enumeration(matrix: pd.DataFrame) -> tuple[int, int, float]:
    ranks = np.vstack([rankdata(row, method="average") for row in matrix.to_numpy(dtype=float)])
    observed, correction = _independent_friedman_statistic(ranks)
    arrangements = [sorted(set(itertools.permutations(row.tolist()))) for row in ranks]
    extreme = 0
    total = 0
    for blocks in itertools.product(*arrangements):
        rank_sums = np.asarray(blocks, dtype=float).sum(axis=0)
        statistic = (
            12.0 / (len(ranks) * ranks.shape[1] * (ranks.shape[1] + 1))
            * np.square(rank_sums).sum()
            - 3.0 * len(ranks) * (ranks.shape[1] + 1)
        ) / correction
        total += 1
        extreme += int(statistic >= observed - 1e-12)
    return total, extreme, extreme / total


def test_exact_friedman_matches_independent_small_full_enumeration() -> None:
    matrix = pd.DataFrame(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.1],
            [0.2, 0.1, 0.3],
        ],
        columns=["a", "b", "c"],
    )

    expected_total, expected_extreme, expected_p = _independent_full_enumeration(matrix)
    actual = exact_friedman_permutation_sensitivity(matrix)

    assert expected_total == 216
    assert actual["full_total_states"] == expected_total
    assert actual["full_extreme_count"] == expected_extreme
    assert actual["full_exact_p_value"] == pytest.approx(expected_p)
    assert actual["reduced_exact_p_value"] == pytest.approx(expected_p)


def test_exact_friedman_counts_unique_tied_rank_arrangements() -> None:
    matrix = pd.DataFrame(
        [
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 2.0],
        ]
    )

    actual = exact_friedman_permutation_sensitivity(matrix)

    assert actual["unique_rank_arrangements_per_block"] == [12, 12, 12]
    assert actual["reduced_total_states"] == 144
    assert actual["full_total_states"] == 1_728
    assert actual["reduced_exact_p_value"] == actual["full_exact_p_value"]


def test_exact_friedman_is_invariant_to_condition_and_dataset_order() -> None:
    matrix = pd.DataFrame(
        [
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 2.0],
        ],
        columns=["a", "b", "c", "d"],
        index=["one", "two", "three"],
    )
    original = exact_friedman_permutation_sensitivity(matrix)
    reordered = exact_friedman_permutation_sensitivity(
        matrix.iloc[[2, 0, 1], [3, 1, 0, 2]]
    )

    for key in (
        "friedman_statistic",
        "tie_correction",
        "reduced_total_states",
        "reduced_extreme_count",
        "full_total_states",
        "full_extreme_count",
        "reduced_exact_p_value",
        "full_exact_p_value",
    ):
        assert reordered[key] == pytest.approx(original[key])


@pytest.mark.parametrize(
    "matrix",
    [
        pd.DataFrame([[0.1, 0.2, 0.3], [0.2, 0.3, 0.1]]),
        pd.DataFrame([[0.1, 0.2], [0.2, 0.3], [0.3, 0.1]]),
        pd.DataFrame([[0.1, 0.2, np.nan], [0.2, 0.3, 0.1], [0.3, 0.1, 0.2]]),
        pd.DataFrame([[0.1, 0.2, np.inf], [0.2, 0.3, 0.1], [0.3, 0.1, 0.2]]),
    ],
)
def test_exact_friedman_rejects_unsupported_or_nonfinite_matrices(matrix: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        exact_friedman_permutation_sensitivity(matrix)


def test_exact_friedman_rejects_state_overflow_before_enumeration() -> None:
    matrix = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 1.0],
            [3.0, 4.0, 5.0, 1.0, 2.0],
        ]
    )

    with pytest.raises(ValueError, match="max_states"):
        exact_friedman_permutation_sensitivity(matrix, max_states=1_000)
    with pytest.raises(ValueError, match="max_states"):
        exact_friedman_permutation_sensitivity(matrix, max_states=0)
    with pytest.raises(ValueError, match="max_states"):
        exact_friedman_permutation_sensitivity(matrix, max_states=1_000_001)


def test_frozen_analysis_manifest_hash_guard_uses_explicit_input(tmp_path: Path) -> None:
    manifest_path = tmp_path / "results" / "analysis" / "analysis_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"synthetic": true}\n', encoding="utf-8")
    expected_sha256 = _sha256(manifest_path)

    assert _validate_pinned_analysis_manifest(
        tmp_path, expected_sha256=expected_sha256
    ) == expected_sha256

    manifest_path.write_text('{"synthetic": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 does not match"):
        _validate_pinned_analysis_manifest(tmp_path, expected_sha256=expected_sha256)


def test_publication_addendum_runner_writes_exact_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    cell_means_sha256 = _write_synthetic_addendum_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        publication_addenda, "EXPECTED_CELL_MEANS_SHA256", cell_means_sha256
    )
    output_dir = tmp_path / "publication-addenda" / "v4"

    publication_addenda.run_addendum(tmp_path, output_dir)

    manifest_path = output_dir / "publication_addendum_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.read_bytes() == (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    result = manifest["result"]
    assert manifest["post_hoc"] is True
    assert result["friedman_statistic"] == pytest.approx(24.0)
    assert result["tie_correction"] == pytest.approx(8.0 / 15.0)
    assert result["reduced_total_states"] == 5_184
    assert result["reduced_extreme_count"] == 1
    assert result["full_total_states"] == 373_248
    assert result["full_extreme_count"] == 72
    assert result["exact_p_value_fraction"] == "1/5184"
    assert result["exact_p_value"] == pytest.approx(1 / 5_184)
    output = manifest["outputs"]["exact_friedman_permutation.csv"]
    assert _sha256(output_dir / output["path"]) == output["sha256"]


def test_publication_addendum_rejects_unpinned_synthetic_cell_means(
    tmp_path: Path, monkeypatch
) -> None:
    _write_synthetic_addendum_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(publication_addenda, "EXPECTED_CELL_MEANS_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="frozen cell_means.csv SHA-256"):
        publication_addenda.run_addendum(
            tmp_path, tmp_path / "results" / "publication-addenda" / "v4"
        )


def test_publication_addendum_replays_with_a_rebound_anonymous_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "anonymous-snapshot"
    cell_means_sha256 = _write_synthetic_addendum_inputs(snapshot, monkeypatch)
    monkeypatch.setattr(
        publication_addenda, "EXPECTED_CELL_MEANS_SHA256", cell_means_sha256
    )
    analysis_dir = snapshot / "results" / "analysis"
    manifest_path = analysis_dir / "analysis_manifest.json"
    original_manifest_sha256 = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["anonymous_snapshot"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_dir = snapshot / "results" / "publication-addenda" / "v4"

    replayed = publication_addenda.run_addendum(snapshot, output_dir)
    replay_manifest_path = output_dir / "publication_addendum_manifest.json"

    assert replayed["input"]["analysis_manifest"]["sha256"] == _sha256(manifest_path)
    assert replayed["input"]["analysis_manifest"]["sha256"] != original_manifest_sha256
    assert replayed["result"]["exact_p_value_fraction"] == "1/5184"
    assert replayed["result"]["reduced_total_states"] == 5_184
    assert replayed["result"]["full_total_states"] == 373_248
    assert replay_manifest_path.read_bytes() == (
        json.dumps(replayed, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
