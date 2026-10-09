"""Small-sample sensitivity procedures for frozen dataset-level results."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata


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
