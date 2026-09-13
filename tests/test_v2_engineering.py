import numpy as np
import pandas as pd
import pytest

from src.v2_engineering import (
    STATUS_FAILED,
    STATUS_VALID,
    FeatureLayout,
    apply_threshold_offsets,
    assert_nested_split_independence,
    categorical_path,
    evaluate_probabilities,
    failure_record,
    fit_threshold_offsets,
    fit_time_sample_weight,
    fold_local_neighbours,
    infer_feature_layout,
    macro_average_precision,
    multiclass_brier_score,
    prepare_native_categorical_frame,
    preprocessing_for,
    sampler_for_v2,
    splitters,
    validate_cell_accounting,
)


def test_dynamic_neighbour_rule_is_fold_local_and_explicitly_fails_below_two() -> None:
    spec = fold_local_neighbours([0] * 10 + [1] * 3 + [2] * 7, "smote")
    assert spec.k_neighbors == 2
    assert spec.minority_support == 3
    with pytest.raises(ValueError, match="fewer than two"):
        fold_local_neighbours([0] * 10 + [1], "smote")


def test_borderline_neighbours_are_bounded_by_actual_training_rows() -> None:
    spec = fold_local_neighbours([0] * 5 + [1] * 2 + [2] * 2, "borderline_smote")
    assert spec.k_neighbors == 1
    assert spec.m_neighbors <= 8
    assert spec.m_neighbors >= 2


def test_sampler_is_built_from_current_fit_labels_and_preserves_mixed_categories() -> None:
    layout = FeatureLayout(numeric=("x",), categorical=("category",))
    sampler = sampler_for_v2("smotenc", layout, [0] * 4 + [1] * 2 + [2] * 2, 7)
    assert sampler.categorical_features == [1]
    assert sampler.k_neighbors == 1


def test_fit_time_weights_cover_all_three_native_gbdts_only() -> None:
    y = [0, 0, 0, 1, 2, 2]
    for model in ("xgboost", "lightgbm", "catboost"):
        weights = fit_time_sample_weight("class_weighted", model, y)
        assert weights is not None
        assert len(weights) == len(y)
    assert fit_time_sample_weight("class_weighted", "random_forest", y) is None
    assert fit_time_sample_weight("raw", "xgboost", y) is None


def test_model_specific_paths_are_explicit_and_unknown_categories_are_safe() -> None:
    frame = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0], "category": ["a", "b", "a"], "target": [0, 1, 2]}
    )
    layout = infer_feature_layout(frame, "target")
    assert categorical_path("logistic_regression", layout, "raw") == "one_hot"
    assert categorical_path("random_forest", layout, "raw") == "ordinal_encoded_tree_features"
    assert categorical_path("xgboost", layout, "raw") == "native_categorical"
    transformer = preprocessing_for("logistic_regression", layout, "raw")
    transformed = transformer.fit_transform(frame.drop(columns=["target"]))
    assert transformed.shape[0] == 3
    native, categories = prepare_native_categorical_frame(frame.drop(columns=["target"]), layout)
    assert str(native["category"].dtype) == "category"
    unseen = pd.DataFrame({"x": [4.0], "category": ["unseen"]})
    native_unseen, _ = prepare_native_categorical_frame(unseen, layout, categories)
    assert native_unseen["category"].isna().sum() == 0
    assert "__MISSING__" in native_unseen["category"].cat.categories


def test_nested_splitters_have_independent_seeds_and_no_cross_fold_leakage() -> None:
    X = np.zeros((30, 1))
    y = np.repeat([0, 1, 2], 10)
    outer, inner = splitters()
    outer_train, outer_test = next(outer.split(X, y))
    inner_train, inner_test = next(inner.split(X[outer_train], y[outer_train]))
    mapped_inner_train = outer_train[inner_train]
    mapped_inner_test = outer_train[inner_test]
    assert outer.random_state != inner.random_state
    assert_nested_split_independence(outer_train, outer_test, mapped_inner_train, mapped_inner_test)
    assert set(mapped_inner_test).isdisjoint(set(outer_test))


def test_thresholds_use_validation_probabilities_and_are_deterministic() -> None:
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.70, 0.20],
            [0.20, 0.10, 0.70],
            [0.65, 0.25, 0.10],
            [0.10, 0.65, 0.25],
            [0.25, 0.10, 0.65],
            [0.60, 0.30, 0.10],
            [0.10, 0.60, 0.30],
            [0.30, 0.10, 0.60],
        ]
    )
    first = fit_threshold_offsets(y, probabilities, [0, 1, 2])
    second = fit_threshold_offsets(y, probabilities, [0, 1, 2])
    assert first.status == STATUS_VALID
    assert first.offsets == second.offsets
    assert np.array_equal(apply_threshold_offsets(probabilities, first.offsets), np.argmax(probabilities, axis=1))


def test_threshold_fallback_is_explicit_for_invalid_validation_support() -> None:
    result = fit_threshold_offsets([0, 0, 1], np.full((3, 3), 1 / 3), [0, 1, 2])
    assert result.status == STATUS_FAILED
    assert "every class" in (result.reason or "")
    assert result.offsets == (0.0, 0.0, 0.0)


def test_probability_metrics_are_normalized_and_use_original_probabilities() -> None:
    y = [0, 1, 2, 0, 1, 2]
    probabilities = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] * 2,
        dtype=float,
    )
    macro_ap, per_class = macro_average_precision(y, probabilities, [0, 1, 2])
    assert macro_ap == pytest.approx(1.0)
    assert per_class == pytest.approx([1.0, 1.0, 1.0])
    assert multiclass_brier_score(y, probabilities, [0, 1, 2]) == pytest.approx(0.02)
    metrics = evaluate_probabilities(y, probabilities, [0, 1, 2], [0.0, 0.0, 0.0])
    assert metrics["macro_average_precision"] == pytest.approx(1.0)
    assert metrics["multiclass_brier"] == pytest.approx(0.02)
    assert sum(probabilities[0]) == pytest.approx(1.0)


def test_cell_accounting_requires_exactly_one_outcome_per_key() -> None:
    expected = [("d", 0), ("d", 1), ("d", 2)]
    assert validate_cell_accounting(expected, expected[:1], expected[1:]) == {
        "expected_cells": 3,
        "valid_cells": 1,
        "failure_cells": 2,
    }
    with pytest.raises(ValueError, match="both valid and failed"):
        validate_cell_accounting(expected, expected[:1], expected[:1])
    with pytest.raises(ValueError, match="mismatch"):
        validate_cell_accounting(expected, expected[:1], [])


def test_failure_record_never_uses_zero_filled_metrics() -> None:
    record = failure_record(
        dataset_id="fixture",
        outer_fold=0,
        outer_seed=1,
        classifier="random_forest",
        condition="smote",
        stage="sampler",
        exception_type="NotApplicableError",
        reason="fewer than two training examples",
        applicability="NOT_APPLICABLE",
        wall_seconds=0.2,
        resource_status="within_limit",
        retry_count=0,
        config_sha256="a" * 64,
    )
    assert record["status"] == STATUS_FAILED
    assert "macro_average_precision" not in record
