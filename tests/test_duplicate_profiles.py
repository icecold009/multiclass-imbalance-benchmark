import pandas as pd

from src.duplicate_profiles import audit_dataset_profiles, predictor_profile_codes


def test_profile_codes_use_only_the_supplied_predictors() -> None:
    features = pd.DataFrame({"numeric": [1, 1, 2], "category": ["a", "a", "b"]})

    groups = predictor_profile_codes(features)

    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_profile_summary_distinguishes_exact_rows_and_label_conflicts() -> None:
    frame = pd.DataFrame(
        {
            "feature": [0, 0, 0, *range(1, 18), 99, 99],
            "label": ["a", "a", "b", *(["a"] * 8), *(["b"] * 9), "b", "b"],
        }
    )

    summary, rowwise_splits, grouped_splits = audit_dataset_profiles(
        "cmc", frame, "label"
    )

    assert summary["exact_duplicate_excess_rows"] == 2
    assert summary["repeated_profile_excess_rows"] == 3
    assert summary["repeated_profile_groups"] == 2
    assert summary["conflicting_profile_groups"] == 1
    assert summary["rows_in_conflicting_profile_groups"] == 3
    assert summary["rowwise_cv_test_rows_with_training_profile"] > 0
    assert len(rowwise_splits) == 15
    assert len(grouped_splits) == 15
    assert summary["grouped_cv_zero_profile_overlap"] is True
    assert summary["grouped_cv_all_classes_in_train_test"] is True


def test_missingness_pattern_is_part_of_the_exact_profile() -> None:
    features = pd.DataFrame({"value": [1.0, None, None], "other": [0, 1, 1]})

    groups = predictor_profile_codes(features)

    assert groups[0] != groups[1]
    assert groups[1] == groups[2]
