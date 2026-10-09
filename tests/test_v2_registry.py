from pathlib import Path

import pandas as pd

from src.v2_registry import (
    REGISTRY_COLUMNS,
    coverage_report,
    imbalance_band,
    validate_registry_file,
    validate_registry_frame,
)


def _row(dataset_id: str = "fixture", *, feature_type: str = "numeric", ratio: float = 1.5) -> dict[str, str]:
    conditions = (
        "raw;class_weighted;random_over;random_under;smote;adasyn;borderline_smote;smoteenn;"
        "smotetomek;balanced_random_forest"
        if feature_type == "numeric"
        else "raw;class_weighted;random_over;random_under;smotenc;smoteenn;smotetomek;balanced_random_forest"
    )
    row = {column: "" for column in REGISTRY_COLUMNS}
    row.update(
        {
            "dataset_id": dataset_id,
            "display_name": dataset_id,
            "source": "fixture",
            "source_id": f"fixture:{dataset_id}",
            "source_url": "https://example.invalid/fixture",
            "license_or_terms": "fixture",
            "source_version": "1",
            "raw_sha256": "a" * 64,
            "task_type": "single-label-classification",
            "target_column": "target",
            "n_rows": "100",
            "n_classes": "3",
            "class_counts": "0:50|1:30|2:20",
            "ir_majority_minority": str(ratio),
            "n_min_class": "20",
            "d_raw": "4",
            "d_encoded": "4",
            "feature_type": feature_type,
            "has_missing": "False",
            "has_groups": "False",
            "has_time_order": "False",
            "has_duplicates": "False",
            "leakage_status": "pass; fixture audit",
            "imbalance_band": imbalance_band(ratio),
            "eligible": "True",
            "exclusion_reason": "",
            "applicable_conditions": conditions,
            "notes": "provenance-only fixture",
        }
    )
    return row


def test_band_boundaries_are_frozen() -> None:
    assert imbalance_band(1.99) == "near_balanced_ir_lt_2"
    assert imbalance_band(2.0) == "mild_ir_2_to_5"
    assert imbalance_band(5.0) == "moderate_ir_5_to_20"
    assert imbalance_band(20.0) == "moderate_ir_5_to_20"
    assert imbalance_band(20.01) == "severe_ir_gt_20"


def test_registry_validator_is_score_free_and_checks_applicability() -> None:
    frame = pd.DataFrame([_row()])
    validation = validate_registry_frame(frame, require_coverage=False)
    assert validation.status == "READY"
    assert validation.eligible_count == 1
    assert coverage_report(validation)["selection_basis"].startswith("provenance_eligibility_only")

    with_score = frame.assign(macro_average_precision="0.5")
    blocked = validate_registry_frame(with_score, require_coverage=False)
    assert blocked.status == "BLOCKED"
    assert any("score/performance" in blocker for blocker in blocked.blockers)

    bad_conditions = frame.copy()
    bad_conditions.loc[0, "applicable_conditions"] = "raw"
    blocked = validate_registry_frame(bad_conditions, require_coverage=False)
    assert blocked.status == "BLOCKED"
    assert any("applicability mismatch" in blocker for blocker in blocked.blockers)


def test_coverage_gate_reports_all_missing_quotas() -> None:
    validation = validate_registry_frame(pd.DataFrame([_row()]), require_coverage=True)
    assert validation.status == "BLOCKED"
    assert any("eligible dataset count" in blocker for blocker in validation.blockers)
    assert any("feature coverage mixed" in blocker for blocker in validation.blockers)
    assert any("imbalance coverage severe_ir_gt_20" in blocker for blocker in validation.blockers)


def test_ineligible_rows_require_explicit_reasons() -> None:
    row = _row()
    row["eligible"] = "False"
    row["exclusion_reason"] = "exclude: declared fixture control"
    validation = validate_registry_frame(pd.DataFrame([row]), require_coverage=False)
    assert validation.status == "BLOCKED"
    assert any("disagrees with rule-derived value" in blocker for blocker in validation.blockers)


def test_frozen_v2_registry_meets_score_free_coverage_gate() -> None:
    validation = validate_registry_file(Path("data/dataset_registry_v2.csv"))
    assert validation.status == "READY"
    assert validation.eligible_count == 50
    assert validation.coverage["feature_type"] == {"numeric": 30, "mixed": 20}
    assert validation.coverage["imbalance_band"] == {
        "near_balanced_ir_lt_2": 8,
        "mild_ir_2_to_5": 19,
        "moderate_ir_5_to_20": 15,
        "severe_ir_gt_20": 8,
    }
