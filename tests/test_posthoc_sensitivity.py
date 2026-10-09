import pandas as pd
import pytest

from src.analysis import _comparison_summary, _run_confirmatory_tests
from src.posthoc_sensitivity import (
    POSTHOC_CLASSIFIERS,
    POSTHOC_CONDITIONS,
    _signed_rank_sign_flip,
    build_adasyn_excluded_sensitivity,
)


def _fixture_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_rows = []
    ratios = [1.0, 1.0, 1.0, 1.03, 1.48, 2.0, 5.0, 20.0, 90.0]
    for dataset_index in range(9):
        dataset = f"d{dataset_index}"
        for classifier_index, classifier in enumerate(POSTHOC_CLASSIFIERS):
            baseline = 0.35 + 0.03 * dataset_index + 0.01 * classifier_index
            for condition_index, condition in enumerate(POSTHOC_CONDITIONS):
                cell_rows.append(
                    {
                        "dataset": dataset,
                        "classifier": classifier,
                        "condition": condition,
                        "feature_type": "numeric",
                        "complete_records": 15,
                        "macro_f1": baseline + 0.002 * condition_index * ((dataset_index % 3) - 1),
                    }
                )
    registry = pd.DataFrame(
        {
            "dataset_id": [f"d{index}" for index in range(9)],
            "ir_majority_minority": ratios,
        }
    )
    return pd.DataFrame(cell_rows), registry


def test_exact_signed_rank_sign_flip_reports_effective_resolution() -> None:
    result = _signed_rank_sign_flip([0.1, 0.2, 0.3])

    assert result["n_nonzero_pairs"] == 3
    assert result["sign_patterns"] == 8
    assert result["exact_p_value"] == pytest.approx(0.25)
    assert result["minimum_attainable_p"] == pytest.approx(0.25)


def test_adasyn_excluded_sensitivity_is_complete_and_deterministic() -> None:
    cells, registry = _fixture_inputs()
    first = build_adasyn_excluded_sensitivity(cells, registry, n_resamples=128, seed=4)
    second = build_adasyn_excluded_sensitivity(cells, registry, n_resamples=128, seed=4)

    assert [len(table) for table in first] == [3, 21, 42]
    for left, right in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(left, right)
    omnibus, pairwise, stratified = first
    assert set(omnibus["n_blocks"]) == {9}
    assert set(omnibus["n_conditions"]) == {8}
    assert set(pairwise["n_pairs"]) == {9}
    assert set(pairwise["holm_family_size"]) == {7}
    assert set(stratified["n_datasets"]) == {3, 6}
    assert set(stratified["inferential_status"]) == {"descriptive_only_no_tests"}


def test_posthoc_sensitivity_rejects_incomplete_reduced_condition_matrix() -> None:
    cells, registry = _fixture_inputs()
    cells = cells[~((cells["dataset"] == "d0") & (cells["condition"] == "smotetomek"))]

    with pytest.raises(ValueError, match="nine complete post hoc blocks"):
        build_adasyn_excluded_sensitivity(cells, registry, n_resamples=8)


def test_tolerance_ties_and_exact_zero_wilcoxon_counts_stay_distinct() -> None:
    matrix = pd.DataFrame(
        {
            "raw": [0.5, 0.7],
            "smotenc": [0.6, 0.7 - 2.220446049250313e-16],
        }
    )

    row = _comparison_summary("mixed_core", "xgboost", "balanced_accuracy", "smotenc", matrix)

    assert row["n_pairs"] == 2
    assert row["wins"] == 1
    assert row["ties"] == 1
    assert row["wilcoxon_effective_n"] == 2
    assert row["wilcoxon_p_value"] is None
    assert row["wilcoxon_min_attainable_p"] is None
    assert row["wilcoxon_calibration"] == "not_run_below_three_pairs"


def test_all_zero_eligible_comparison_stays_in_holm_family() -> None:
    values = {
        "raw": [0.5, 0.6, 0.7],
        "class_weighted": [0.5, 0.6, 0.7],
        "random_over": [0.6, 0.6, 0.8],
        "random_under": [0.4, 0.7, 0.6],
        "smote": [0.6, 0.5, 0.8],
        "adasyn": [0.5, 0.7, 0.6],
        "borderline_smote": [0.7, 0.5, 0.6],
        "smoteenn": [0.4, 0.5, 0.8],
        "smotetomek": [0.8, 0.6, 0.5],
    }
    rows = [
        {
            "dataset": f"d{dataset_index}",
            "classifier": "logistic_regression",
            "condition": condition,
            "feature_type": "numeric",
            **{metric: scores[dataset_index] for metric in ("macro_f1", "g_mean", "mcc", "balanced_accuracy")},
        }
        for condition, scores in values.items()
        for dataset_index in range(3)
    ]

    _, _, comparisons = _run_confirmatory_tests(pd.DataFrame(rows))
    all_tie = comparisons[
        (comparisons["metric"] == "macro_f1")
        & (comparisons["condition"] == "class_weighted")
    ].iloc[0]

    assert all_tie["wilcoxon_p_value"] == pytest.approx(1.0)
    assert all_tie["wilcoxon_min_attainable_p"] == pytest.approx(1.0)
    assert all_tie["wilcoxon_effective_n"] == 0
    assert all_tie["holm_family_size"] == 8
    assert all_tie["holm_p_value"] == pytest.approx(1.0)
