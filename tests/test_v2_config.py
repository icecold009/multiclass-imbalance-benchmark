import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load((ROOT / "config" / "v2.yaml").read_text(encoding="utf-8"))


def test_v2_configuration_freezes_protocol_and_analysis_contract() -> None:
    config = _config()

    assert config["protocol_version"] == "expanded-v2"
    assert config["status"] == "gate-6-analysis-ready"
    assert config["readiness_blocker"] == "compute_instance_approval"
    assert config["scope"]["minimum_classes"] == 3
    assert config["scope"]["minimum_minority_support"] == 10
    assert config["hyperparameter_search"]["trials_per_dataset_outer_fold_classifier_condition"] == 20
    assert config["hyperparameter_search"]["sampler_parameters_tuned"] is False
    assert config["cross_validation"]["outer"]["n_splits"] == 5
    assert config["cross_validation"]["inner"]["n_splits"] == 3
    assert config["sampling"]["dynamic_neighbour_formula"] == (
        "k = min(5, N_minority_target - 1)"
    )
    assert config["bayesian_analysis"]["model"] == (
        "exchangeable_fold_correlated_hierarchical_normal"
    )
    assert config["bayesian_analysis"]["rope_macro_average_precision"] == [-0.01, 0.01]
    assert config["record_schemas"]["status_values"] == ["VALID", "FAILED / NOT APPLICABLE"]


def test_v2_condition_matrix_contains_ten_conditions_and_rf_comparator() -> None:
    config = _config()
    condition_ids = [condition["id"] for condition in config["conditions"]]

    assert condition_ids == [
        "raw",
        "class_weighted",
        "random_over",
        "random_under",
        "smote",
        "adasyn",
        "borderline_smote",
        "smotenc",
        "smoteenn",
        "smotetomek",
        "balanced_random_forest",
    ]
    assert config["conditions"][-1]["applicable_models"] == ["random_forest"]
    assert config["conditions"][-1]["is_universal_sampler_condition"] is False


def test_v2_registry_schema_is_provenance_ready() -> None:
    expected = [
        "dataset_id",
        "display_name",
        "source",
        "source_id",
        "source_url",
        "license_or_terms",
        "source_version",
        "raw_sha256",
        "task_type",
        "target_column",
        "n_rows",
        "n_classes",
        "class_counts",
        "ir_majority_minority",
        "n_min_class",
        "d_raw",
        "d_encoded",
        "feature_type",
        "has_missing",
        "has_groups",
        "has_time_order",
        "has_duplicates",
        "leakage_status",
        "imbalance_band",
        "eligible",
        "exclusion_reason",
        "applicable_conditions",
        "notes",
    ]
    with (ROOT / "data" / "dataset_registry_v2.csv").open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))

    assert header == expected
