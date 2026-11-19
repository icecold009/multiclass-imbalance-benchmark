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

from scripts.run_publication_addenda import run_addendum
from src.sensitivity import exact_friedman_permutation_sensitivity

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ANALYSIS_MANIFEST_SHA256 = "f398bc3dcd53eefc05e7ce8b96d2ae5736a8d0e50a7cb6f594e79557b4ecdb3b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_frozen_analysis_manifest_and_original_outputs_keep_their_hashes() -> None:
    manifest_path = ROOT / "results/analysis/analysis_manifest.json"
    assert _sha256(manifest_path) == FROZEN_ANALYSIS_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    frozen_tables_and_figures = {
        name: reference
        for name, reference in artifacts.items()
        if name.startswith("results/analysis/") and name.endswith((".csv", ".png"))
    }

    assert sum(name.endswith(".csv") for name in frozen_tables_and_figures) == 18
    assert sum(name.endswith(".png") for name in frozen_tables_and_figures) == 8
    assert len(frozen_tables_and_figures) == 26
    for relative_path, reference in frozen_tables_and_figures.items():
        assert _sha256(ROOT / relative_path) == reference["sha256"]


def test_publication_addendum_runner_writes_exact_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "publication-addenda" / "v4"

    run_addendum(ROOT, output_dir)

    manifest_path = output_dir / "publication_addendum_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
