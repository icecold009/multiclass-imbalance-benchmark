import itertools

import pandas as pd
import pytest

from src.duplicate_profile_sensitivity import (
    FAILURE_COLUMNS,
    KEY_COLUMNS,
    RESULT_COLUMNS,
    _applicability_error,
    _build_preflight,
    _run_dataset,
    _validate_model_key_accounting,
)
from src.duplicate_profiles import GROUPED_SENSITIVITY_DATASETS
from src.pilot import CLASSIFIERS, CONDITIONS, FeatureLayout


def _input(dataset_id: str, values: list[int], labels: list[str]) -> dict[str, object]:
    frame = pd.DataFrame({"feature": values, "target": labels})
    return {
        "dataset_id": dataset_id,
        "frame": frame,
        "target": "target",
        "categorical_columns": (),
    }


def test_grouped_preflight_records_all_fixed_splits_without_profile_overlap() -> None:
    values = list(range(15))
    labels = [label for label in ("a", "b", "c") for _ in range(5)]

    plan, groups, assignments, feasibility = _build_preflight([_input("fixture", values, labels)])

    assert len(groups) == 15
    assert len(feasibility) == 15
    assert len(assignments) == 15 * 15
    assert feasibility["feasible"].all()
    assert feasibility["overlapping_profile_groups"].eq(0).all()
    assert len(plan["splits"]["fixture"]) == 15


def test_infeasible_group_splits_are_recorded_for_every_model_key_without_replacement() -> None:
    values = [0] * 5 + [1] * 5 + [2] * 5
    labels = [label for label in ("a", "b", "c") for _ in range(5)]
    item = _input("fixture", values, labels)
    plan, _, _, feasibility = _build_preflight([item])

    results, failures = _run_dataset(item, plan)

    assert not feasibility["feasible"].any()
    assert results.empty
    assert len(failures) == 15 * len(CLASSIFIERS) * len(CONDITIONS)
    assert failures["failure_category"].eq("infeasible_split").all()
    assert failures["stage"].eq("split_preflight").all()
    assert failures[["seed", "fold"]].drop_duplicates().shape[0] == 15


def test_applicability_keeps_mixed_and_numeric_sampler_rules_and_rf_only_comparator() -> None:
    numeric = FeatureLayout(numeric=("x",), categorical=())
    mixed = FeatureLayout(numeric=("x",), categorical=("category",))

    assert _applicability_error(numeric, "random_forest", "smote", 0) is None
    assert _applicability_error(numeric, "random_forest", "smotenc", 0)
    assert _applicability_error(mixed, "random_forest", "smotenc", 0) is None
    assert _applicability_error(mixed, "random_forest", "smote", 0)
    assert _applicability_error(mixed, "xgboost", "balanced_random_forest", 0)


def test_model_grid_accounting_requires_each_expected_key_exactly_once() -> None:
    keys = list(
        itertools.product(
            GROUPED_SENSITIVITY_DATASETS,
            (0, 1, 2),
            range(5),
            CLASSIFIERS,
            CONDITIONS,
        )
    )
    results = pd.DataFrame(
        [dict(zip(KEY_COLUMNS, key, strict=True)) for key in keys], columns=RESULT_COLUMNS
    )
    failures = pd.DataFrame(columns=FAILURE_COLUMNS)

    assert _validate_model_key_accounting([results], [failures]) == 2_970
    with pytest.raises(ValueError, match="duplicate model-grid keys"):
        _validate_model_key_accounting([pd.concat([results, results.iloc[[0]]])], [failures])
