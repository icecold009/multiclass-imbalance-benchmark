"""Post hoc macro-F1 sensitivity excluding ADASYN from the numeric family."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.sensitivity import friedman_permutation_sensitivity

POSTHOC_CONDITIONS = (
    "raw",
    "class_weighted",
    "random_over",
    "random_under",
    "smote",
    "borderline_smote",
    "smoteenn",
    "smotetomek",
)
POSTHOC_CLASSIFIERS = ("logistic_regression", "random_forest", "xgboost")
POSTHOC_RESAMPLES = 200_000
POSTHOC_SEED = 20261009
POSTHOC_ALPHA = 0.05
TIE_TOLERANCE = 1e-12

POSTHOC_PROCEDURE = {
    "status": "post hoc sensitivity; not a protocol amendment or a model-training rerun",
    "outcome": "macro_f1",
    "condition_set": list(POSTHOC_CONDITIONS),
    "condition_change": "numeric core with ADASYN excluded",
    "complete_case_rule": (
        "the same nine numeric datasets must have complete cells for all eight conditions "
        "for each classifier"
    ),
    "omnibus": {
        "statistic": "Friedman rank statistic",
        "reported_calibration": "asymptotic chi-square p-value and seeded Monte Carlo permutation p-value",
        "permutation_resamples": POSTHOC_RESAMPLES,
        "permutation_seed": POSTHOC_SEED,
        "p_value": "plus-one Monte Carlo estimate",
        "assumption": "condition-label exchangeability within each dataset block under the omnibus null",
    },
    "follow_up": {
        "contrasts": "raw versus each of the seven remaining alternative conditions",
        "statistic": "two-sided Wilcoxon signed-rank statistic with midranks for tied absolute differences",
        "calibration": "exact enumeration of sign flips over nonzero paired dataset differences",
        "zero_rule": "discard exact-zero differences; the separate descriptive W/T/L rule uses tolerance 1e-12",
        "assumption": "independent sign exchangeability of paired dataset-level differences within each contrast",
        "multiplicity": "Holm correction within each classifier across its seven raw-baseline contrasts",
        "classifier_boundary": "the three classifier-specific families are reported separately; no across-classifier adjustment",
    },
    "imbalance_stratification": {
        "rule": "exactly balanced when registry majority/minority ratio equals 1.0; otherwise ratio greater than 1.0",
        "status": "descriptive only; no tests or inferential claims",
    },
}


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _require_complete_matrices(
    cell_means: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    required = {"dataset", "classifier", "condition", "feature_type", "complete_records", "macro_f1"}
    missing = required.difference(cell_means.columns)
    if missing:
        raise ValueError(f"cell means are missing required columns: {sorted(missing)}")
    numeric = cell_means[cell_means["feature_type"].astype(str) == "numeric"]
    dataset_ids = tuple(sorted(numeric["dataset"].astype(str).unique()))
    if len(dataset_ids) != 9:
        raise ValueError(f"expected nine retained numeric datasets, found {len(dataset_ids)}")
    matrices: dict[str, pd.DataFrame] = {}
    for classifier in POSTHOC_CLASSIFIERS:
        subset = numeric[numeric["classifier"].astype(str) == classifier]
        if subset.duplicated(["dataset", "condition"]).any():
            raise ValueError(f"duplicate numeric cell mean for {classifier}")
        matrix = subset.pivot(index="dataset", columns="condition", values="macro_f1")
        missing_conditions = set(POSTHOC_CONDITIONS).difference(matrix.columns)
        if missing_conditions:
            raise ValueError(
                f"{classifier} is missing conditions: {sorted(missing_conditions)}"
            )
        matrix = matrix.reindex(index=dataset_ids, columns=POSTHOC_CONDITIONS)
        if matrix.isna().any().any():
            raise ValueError(f"{classifier} does not have nine complete post hoc blocks")
        if not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise ValueError(f"{classifier} has non-finite post hoc macro-F1 values")
        if not subset["complete_records"].astype(int).eq(15).all():
            raise ValueError(f"{classifier} has a post hoc cell without all 15 seed-fold records")
        matrices[classifier] = matrix
    return matrices, dataset_ids


def _signed_rank_sign_flip(differences: np.ndarray) -> dict[str, float | int]:
    """Exhaustively test a two-sided signed-rank statistic under sign exchangeability."""

    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0.0]
    n_nonzero = len(nonzero)
    if n_nonzero > 20:
        raise ValueError("exact signed-rank sign-flip enumeration is limited to 20 nonzero pairs")
    if not n_nonzero:
        return {
            "wilcoxon_statistic": 0.0,
            "exact_p_value": 1.0,
            "n_nonzero_pairs": 0,
            "sign_patterns": 1,
            "minimum_attainable_p": 1.0,
        }

    ranks = rankdata(np.abs(nonzero), method="average")
    positive_rank_sum = float(ranks[nonzero > 0].sum())
    negative_rank_sum = float(ranks[nonzero < 0].sum())
    observed = abs(positive_rank_sum - negative_rank_sum)
    signs = np.asarray(tuple(itertools.product((-1.0, 1.0), repeat=n_nonzero)), dtype=float)
    statistics = np.abs(signs @ ranks)
    p_value = float(np.count_nonzero(statistics >= observed - 1e-12) / len(statistics))
    return {
        "wilcoxon_statistic": min(positive_rank_sum, negative_rank_sum),
        "exact_p_value": p_value,
        "n_nonzero_pairs": int(n_nonzero),
        "sign_patterns": len(signs),
        "minimum_attainable_p": float(2 / len(signs)),
    }


def _pairwise_rows(matrices: dict[str, pd.DataFrame], dataset_ids: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for classifier, matrix in matrices.items():
        family_rows: list[dict[str, object]] = []
        for condition in POSTHOC_CONDITIONS[1:]:
            differences = (
                matrix[condition].to_numpy(dtype=float) - matrix["raw"].to_numpy(dtype=float)
            )
            signed_rank = _signed_rank_sign_flip(differences)
            row: dict[str, object] = {
                "analysis_type": "post_hoc_adasyn_excluded_macro_f1",
                "family": "numeric_core_without_adasyn",
                "classifier": classifier,
                "condition": condition,
                "dataset_ids": ";".join(dataset_ids),
                "n_pairs": len(differences),
                "mean_delta": float(differences.mean()),
                "median_delta": float(np.median(differences)),
                "wins": int(np.count_nonzero(differences > TIE_TOLERANCE)),
                "ties": int(np.count_nonzero(np.abs(differences) <= TIE_TOLERANCE)),
                "losses": int(np.count_nonzero(differences < -TIE_TOLERANCE)),
                **signed_rank,
                "holm_family_size": len(POSTHOC_CONDITIONS) - 1,
                "tie_tolerance": TIE_TOLERANCE,
                "assumption": POSTHOC_PROCEDURE["follow_up"]["assumption"],
            }
            family_rows.append(row)
        adjusted = _holm_adjust([float(row["exact_p_value"]) for row in family_rows])
        for row, adjusted_p in zip(family_rows, adjusted, strict=True):
            row["holm_p_value"] = adjusted_p
            rows.append(row)
    return pd.DataFrame(rows)


def _stratified_rows(
    matrices: dict[str, pd.DataFrame],
    dataset_ids: tuple[str, ...],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    required = {"dataset_id", "ir_majority_minority"}
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"dataset registry is missing stratification columns: {sorted(missing)}")
    selected = registry[registry["dataset_id"].astype(str).isin(dataset_ids)].copy()
    if selected["dataset_id"].duplicated().any() or set(selected["dataset_id"].astype(str)) != set(
        dataset_ids
    ):
        raise ValueError("registry does not identify exactly the nine post hoc datasets")
    ratios = pd.to_numeric(selected.set_index("dataset_id")["ir_majority_minority"], errors="raise")
    balanced = tuple(sorted(ratios.index[np.isclose(ratios.to_numpy(), 1.0, rtol=0, atol=1e-12)]))
    imbalanced = tuple(sorted(ratios.index[ratios.to_numpy() > 1.0 + 1e-12]))
    if len(balanced) != 3 or len(imbalanced) != 6 or set(balanced) | set(imbalanced) != set(
        dataset_ids
    ):
        raise ValueError("post hoc IR strata must contain three IR=1.0 and six IR>1.0 datasets")

    rows: list[dict[str, object]] = []
    strata = (
        ("exactly_balanced_ir_1.0", balanced, "IR equals 1.0"),
        ("imbalanced_ir_gt_1.0", imbalanced, "IR greater than 1.0"),
    )
    for classifier, matrix in matrices.items():
        for condition in POSTHOC_CONDITIONS[1:]:
            for stratum, stratum_ids, threshold in strata:
                differences = (
                    matrix.loc[list(stratum_ids), condition].to_numpy(dtype=float)
                    - matrix.loc[list(stratum_ids), "raw"].to_numpy(dtype=float)
                )
                rows.append(
                    {
                        "analysis_type": "post_hoc_descriptive_imbalance_stratification",
                        "metric": "macro_f1",
                        "classifier": classifier,
                        "condition": condition,
                        "stratum": stratum,
                        "threshold": threshold,
                        "dataset_ids": ";".join(stratum_ids),
                        "n_datasets": len(stratum_ids),
                        "mean_paired_delta": float(differences.mean()),
                        "median_paired_delta": float(np.median(differences)),
                        "wins": int(np.count_nonzero(differences > TIE_TOLERANCE)),
                        "ties": int(np.count_nonzero(np.abs(differences) <= TIE_TOLERANCE)),
                        "losses": int(np.count_nonzero(differences < -TIE_TOLERANCE)),
                        "inferential_status": "descriptive_only_no_tests",
                        "tie_tolerance": TIE_TOLERANCE,
                    }
                )
    return pd.DataFrame(rows)


def build_adasyn_excluded_sensitivity(
    cell_means: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    n_resamples: int = POSTHOC_RESAMPLES,
    seed: int = POSTHOC_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize a deterministic post hoc eight-condition analysis from frozen cells."""

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    matrices, dataset_ids = _require_complete_matrices(cell_means)
    omnibus_rows: list[dict[str, object]] = []
    for classifier, matrix in matrices.items():
        permutation = friedman_permutation_sensitivity(
            matrix,
            n_resamples=n_resamples,
            seed=seed,
        )
        omnibus_rows.append(
            {
                "analysis_type": "post_hoc_adasyn_excluded_macro_f1",
                "family": "numeric_core_without_adasyn",
                "classifier": classifier,
                "metric": "macro_f1",
                "dataset_ids": ";".join(dataset_ids),
                **permutation,
                "alpha": POSTHOC_ALPHA,
                "classifier_multiplicity_boundary": "reported separately; no across-classifier adjustment",
                "assumption": POSTHOC_PROCEDURE["omnibus"]["assumption"],
            }
        )
    omnibus = pd.DataFrame(omnibus_rows)
    pairwise = _pairwise_rows(matrices, dataset_ids)
    stratified = _stratified_rows(matrices, dataset_ids, registry)
    return omnibus, pairwise, stratified
