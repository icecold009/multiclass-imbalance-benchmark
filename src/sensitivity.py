"""Small-sample sensitivity procedures for frozen dataset-level results."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Sequence
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata

MAX_EXACT_FRIEDMAN_STATES = 1_000_000
_EXACT_FRIEDMAN_BATCH_SIZE = 10_000


def _complete_numeric_matrix(matrix: pd.DataFrame) -> np.ndarray:
    if matrix.empty or matrix.shape[0] < 3 or matrix.shape[1] < 3:
        raise ValueError("sensitivity analysis requires at least three blocks and conditions")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("sensitivity analysis requires a complete finite matrix")
    return values


def _friedman_rank_statistic(ranks: np.ndarray) -> tuple[float, float]:
    n_blocks, n_conditions = ranks.shape
    tie_term = 0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_term += int(sum(int(count) ** 3 - int(count) for count in counts))
    tie_correction = 1.0 - tie_term / (n_blocks * (n_conditions**3 - n_conditions))
    if tie_correction <= 0:
        raise ValueError("Friedman tie correction is not positive")
    rank_sums = ranks.sum(axis=0)
    uncorrected = 12.0 / (n_blocks * n_conditions * (n_conditions + 1)) * float(
        np.square(rank_sums).sum()
    ) - 3.0 * n_blocks * (n_conditions + 1)
    return uncorrected / tie_correction, tie_correction


def _unique_rank_arrangements(row: np.ndarray) -> np.ndarray:
    """Return unique permutations of one tied rank row in stable order."""

    counts = Counter(float(value) for value in row)
    arrangement_count = math.factorial(len(row)) // math.prod(
        math.factorial(count) for count in counts.values()
    )
    arrangements: list[tuple[float, ...]] = []
    current: list[float] = []
    values = sorted(counts)

    def append_remaining() -> None:
        if len(current) == len(row):
            arrangements.append(tuple(current))
            return
        for value in values:
            if counts[value] == 0:
                continue
            counts[value] -= 1
            current.append(value)
            append_remaining()
            current.pop()
            counts[value] += 1

    append_remaining()
    if len(arrangements) != arrangement_count:
        raise RuntimeError("unique rank arrangement count did not match its tie multiplicities")
    return np.asarray(arrangements, dtype=float)


def _count_friedman_extremes(
    base_ranks: np.ndarray,
    varying_arrangements: Sequence[np.ndarray],
    *,
    n_blocks: int,
    n_conditions: int,
    tie_correction: float,
    observed_statistic: float,
) -> int:
    """Count extreme statistics, vectorizing over the last block's arrangements."""

    coefficient = 12.0 / (n_blocks * n_conditions * (n_conditions + 1))
    offset = 3.0 * n_blocks * (n_conditions + 1)
    exceedances = 0
    prefix_ranges = [range(len(arrangements)) for arrangements in varying_arrangements[:-1]]
    prefix_combinations = itertools.product(*prefix_ranges) if prefix_ranges else [()]
    last_arrangements = varying_arrangements[-1]

    for combination in prefix_combinations:
        rank_sums = np.array(base_ranks, dtype=float, copy=True)
        for arrangements, index in zip(varying_arrangements[:-1], combination, strict=True):
            rank_sums += arrangements[index]
        for start in range(0, len(last_arrangements), _EXACT_FRIEDMAN_BATCH_SIZE):
            last_batch = last_arrangements[start : start + _EXACT_FRIEDMAN_BATCH_SIZE]
            sums_by_last_arrangement = rank_sums[None, :] + last_batch
            uncorrected = coefficient * np.square(sums_by_last_arrangement).sum(axis=1) - offset
            statistics = uncorrected / tie_correction
            exceedances += int(np.count_nonzero(statistics >= observed_statistic - 1e-12))
    return exceedances


def exact_friedman_permutation_sensitivity(
    matrix: pd.DataFrame,
    *,
    max_states: int = 1_000_000,
) -> dict[str, object]:
    """Exhaust tied-rank condition-label permutations for a complete Friedman block.

    The full distribution independently permutes condition labels within every
    dataset block. By global label symmetry, the reduced distribution fixes
    the first block's observed rank arrangement and enumerates every remaining
    block. Under the same exchangeability assumption, applying a common label
    permutation maps the remaining arrangement sets bijectively, so every
    possible first-block arrangement has the same conditional tail count.
    Both enumerations are computed and required to agree.
    """

    if (
        not isinstance(max_states, Integral)
        or isinstance(max_states, bool)
        or not 1 <= max_states <= MAX_EXACT_FRIEDMAN_STATES
    ):
        raise ValueError(
            f"max_states must be an integer from 1 through {MAX_EXACT_FRIEDMAN_STATES}"
        )
    values = _complete_numeric_matrix(matrix)
    n_blocks, n_conditions = values.shape
    ranks = np.vstack([rankdata(row, method="average") for row in values])
    observed_statistic, tie_correction = _friedman_rank_statistic(ranks)

    arrangement_counts = []
    for row in ranks:
        counts = Counter(float(value) for value in row)
        arrangements = math.factorial(n_conditions) // math.prod(
            math.factorial(count) for count in counts.values()
        )
        arrangement_counts.append(arrangements)
    full_states = math.prod(arrangement_counts)
    reduced_states = math.prod(arrangement_counts[1:])
    if full_states > max_states:
        raise ValueError(
            f"exact Friedman enumeration requires {full_states} full states, "
            f"exceeding max_states={max_states}"
        )

    arrangements_by_block = [_unique_rank_arrangements(row) for row in ranks]
    reduced_extreme = _count_friedman_extremes(
        ranks[0],
        arrangements_by_block[1:],
        n_blocks=n_blocks,
        n_conditions=n_conditions,
        tie_correction=tie_correction,
        observed_statistic=observed_statistic,
    )
    full_extreme = _count_friedman_extremes(
        np.zeros(n_conditions, dtype=float),
        arrangements_by_block,
        n_blocks=n_blocks,
        n_conditions=n_conditions,
        tie_correction=tie_correction,
        observed_statistic=observed_statistic,
    )
    reduced_p = reduced_extreme / reduced_states
    full_p = full_extreme / full_states
    if not math.isclose(reduced_p, full_p, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("reduced and full exact Friedman enumerations disagree")

    return {
        "n_blocks": int(n_blocks),
        "n_conditions": int(n_conditions),
        "friedman_statistic": float(observed_statistic),
        "tie_correction": float(tie_correction),
        "unique_rank_arrangements_per_block": [int(count) for count in arrangement_counts],
        "reduced_total_states": int(reduced_states),
        "reduced_extreme_count": int(reduced_extreme),
        "reduced_exact_p_value": float(reduced_p),
        "full_total_states": int(full_states),
        "full_extreme_count": int(full_extreme),
        "full_exact_p_value": float(full_p),
        "symmetry_reduction": (
            "fix the first block under global condition-label symmetry; a common label "
            "permutation maps the remaining tied-rank arrangements bijectively"
        ),
        "enumeration_method": "exhaustive unique tied-rank arrangements within each block",
    }


def friedman_permutation_sensitivity(
    matrix: pd.DataFrame,
    *,
    n_resamples: int = 200_000,
    seed: int = 0,
    batch_size: int = 10_000,
) -> dict[str, float | int]:
    """Monte Carlo label-permutation sensitivity for a complete Friedman block.

    Within each dataset block, the observed condition ranks are permuted across
    condition labels. The resulting plus-one p-value is conditional on the
    selected blocks and assumes condition-label exchangeability under the
    omnibus null. It is a post hoc sensitivity, not a replacement for the
    protocol-planned asymptotic Friedman/Nemenyi analysis.
    """

    if n_resamples < 1 or batch_size < 1:
        raise ValueError("n_resamples and batch_size must be positive")
    values = _complete_numeric_matrix(matrix)
    ranks = np.vstack([rankdata(row, method="average") for row in values])
    observed, tie_correction = _friedman_rank_statistic(ranks)
    asymptotic = friedmanchisquare(*[values[:, column] for column in range(values.shape[1])])

    rng = np.random.default_rng(seed)
    n_blocks, n_conditions = ranks.shape
    exceedances = 0
    for offset in range(0, n_resamples, batch_size):
        count = min(batch_size, n_resamples - offset)
        order = np.argsort(
            rng.random((count, n_blocks, n_conditions)),
            axis=2,
            kind="stable",
        )
        permuted = np.take_along_axis(
            np.broadcast_to(ranks, (count, n_blocks, n_conditions)),
            order,
            axis=2,
        )
        rank_sums = permuted.sum(axis=1)
        uncorrected = 12.0 / (n_blocks * n_conditions * (n_conditions + 1)) * np.square(
            rank_sums
        ).sum(axis=1) - 3.0 * n_blocks * (n_conditions + 1)
        statistics = uncorrected / tie_correction
        exceedances += int(np.count_nonzero(statistics >= observed - 1e-12))

    p_value = (exceedances + 1) / (n_resamples + 1)
    return {
        "n_blocks": int(n_blocks),
        "n_conditions": int(n_conditions),
        "friedman_statistic": float(observed),
        "asymptotic_p_value": float(asymptotic.pvalue),
        "tie_correction": float(tie_correction),
        "permutation_resamples": int(n_resamples),
        "permutation_seed": int(seed),
        "permutation_exceedances": int(exceedances),
        "permutation_p_value": float(p_value),
        "permutation_monte_carlo_se": float(
            math.sqrt(p_value * (1.0 - p_value) / (n_resamples + 1))
        ),
    }


def exact_sign_flip_sensitivity(matrix: pd.DataFrame) -> pd.DataFrame:
    """Enumerate two-sided paired sign flips for every condition pair.

    The exact test conditions on the observed magnitudes and assumes
    independent sign exchangeability of paired dataset-level differences.
    """

    values = _complete_numeric_matrix(matrix)
    n_blocks = values.shape[0]
    if n_blocks > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 paired blocks")
    rows: list[dict[str, object]] = []
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=n_blocks)),
        dtype=float,
    )
    for left, right in itertools.combinations(matrix.columns, 2):
        differences = matrix[left].to_numpy(dtype=float) - matrix[right].to_numpy(dtype=float)
        observed = abs(float(differences.mean()))
        permuted = np.abs(signs @ differences / n_blocks)
        p_value = float(np.count_nonzero(permuted >= observed - 1e-12) / len(signs))
        rows.append(
            {
                "condition_a": str(left),
                "condition_b": str(right),
                "mean_delta_a_minus_b": float(differences.mean()),
                "n_blocks": int(n_blocks),
                "sign_patterns": len(signs),
                "exact_p_value": p_value,
            }
        )
    return pd.DataFrame(rows)
