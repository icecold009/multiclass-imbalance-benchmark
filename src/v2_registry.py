"""Score-free V2 dataset-registry validation and coverage accounting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REGISTRY_COLUMNS = (
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
)
NUMERIC_FEATURE_TYPES = frozenset({"numeric", "mixed"})
FEATURE_QUOTAS = {"numeric": 20, "mixed": 10}
BAND_QUOTAS = {
    "near_balanced_ir_lt_2": 8,
    "mild_ir_2_to_5": 10,
    "moderate_ir_5_to_20": 15,
    "severe_ir_gt_20": 8,
}
NUMERIC_CONDITIONS = {
    "raw",
    "class_weighted",
    "random_over",
    "random_under",
    "smote",
    "adasyn",
    "borderline_smote",
    "smoteenn",
    "smotetomek",
    "balanced_random_forest",
}
MIXED_CONDITIONS = {
    "raw",
    "class_weighted",
    "random_over",
    "random_under",
    "smotenc",
    "smoteenn",
    "smotetomek",
    "balanced_random_forest",
}
SCORE_COLUMN_MARKERS = (
    "accuracy",
    "auc",
    "auprc",
    "average_precision",
    "f1",
    "macro_ap",
    "macro_average_precision",
    "score",
    "metric",
    "performance",
)


@dataclass(frozen=True)
class RegistryValidation:
    """Machine-readable registry validation outcome."""

    status: str
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    eligible_count: int
    coverage: dict[str, dict[str, int]]


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _bool(value: object) -> bool | None:
    normalized = _text(value).casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def imbalance_band(ir: float) -> str:
    """Return the frozen imbalance band from the majority/minority ratio."""

    if ir < 2:
        return "near_balanced_ir_lt_2"
    if ir < 5:
        return "mild_ir_2_to_5"
    if ir <= 20:
        return "moderate_ir_5_to_20"
    return "severe_ir_gt_20"


def expected_conditions(feature_type: str) -> set[str]:
    if feature_type == "numeric":
        return set(NUMERIC_CONDITIONS)
    if feature_type == "mixed":
        return set(MIXED_CONDITIONS)
    return set()


def _coverage(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    eligible = frame[frame["eligible"] == "True"]
    return {
        "feature_type": {
            feature_type: int((eligible["feature_type"] == feature_type).sum())
            for feature_type in FEATURE_QUOTAS
        },
        "imbalance_band": {
            band: int((eligible["imbalance_band"] == band).sum())
            for band in BAND_QUOTAS
        },
    }


def validate_registry_frame(
    frame: pd.DataFrame,
    *,
    require_coverage: bool = True,
    min_datasets: int = 40,
    max_datasets: int = 60,
) -> RegistryValidation:
    """Validate registry provenance, eligibility, applicability, and quotas.

    This function intentionally does not inspect any model score or result
    artifact.  It validates only pre-outcome metadata and documented rules.
    """

    blockers: list[str] = []
    warnings: list[str] = []
    missing = sorted(set(REGISTRY_COLUMNS) - set(frame.columns))
    if missing:
        blockers.append(f"registry missing required columns: {', '.join(missing)}")
        return RegistryValidation("BLOCKED", tuple(blockers), tuple(warnings), 0, {})
    score_columns = [
        column for column in frame.columns if any(marker in column.casefold() for marker in SCORE_COLUMN_MARKERS)
    ]
    if score_columns:
        blockers.append(f"registry contains score/performance columns: {', '.join(score_columns)}")
    if frame["dataset_id"].map(_text).eq("").any():
        blockers.append("dataset_id contains an empty value")
    if frame["dataset_id"].duplicated().any():
        blockers.append("dataset_id contains duplicates")
    for column in ("source_id", "raw_sha256"):
        if frame[column].map(_text).eq("").any():
            blockers.append(f"{column} contains an empty value")
    if frame["source_id"].duplicated().any():
        blockers.append("source_id contains duplicates; one underlying task must count once")
    if frame["raw_sha256"].map(lambda value: len(_text(value)) == 64).eq(False).any():
        blockers.append("raw_sha256 contains a non-SHA-256 value")

    eligible_ids: list[str] = []
    for row in frame.to_dict(orient="records"):
        dataset_id = _text(row["dataset_id"]) or "<empty>"
        try:
            n_rows = int(row["n_rows"])
            n_classes = int(row["n_classes"])
            n_min_class = int(row["n_min_class"])
            ratio = float(row["ir_majority_minority"])
        except (TypeError, ValueError):
            blockers.append(f"{dataset_id}: invalid numeric eligibility metadata")
            continue
        feature_type = _text(row["feature_type"])
        recorded_eligible = _bool(row["eligible"])
        required = expected_conditions(feature_type)
        recorded_conditions = {
            condition.strip()
            for condition in _text(row["applicable_conditions"]).split(";")
            if condition.strip()
        }
        reasons: list[str] = []
        if _text(row["task_type"]) != "single-label-classification":
            reasons.append("task type is not single-label classification")
        if n_rows < 1 or n_classes < 3 or n_min_class < 10:
            reasons.append("minimum row/class/minority-support rule is not met")
        if feature_type not in NUMERIC_FEATURE_TYPES:
            reasons.append("feature type is outside the numeric/mixed core benchmark")
        if _bool(row["has_groups"]) is not False:
            reasons.append("grouped data is not explicitly supported")
        if _bool(row["has_time_order"]) is not False:
            reasons.append("time-ordered data is excluded from the core benchmark")
        if not _text(row["leakage_status"]).casefold().startswith("pass"):
            reasons.append("leakage audit is not marked pass")
        if feature_type not in NUMERIC_FEATURE_TYPES:
            reasons.append("unsupported feature representation")
        if recorded_conditions != required:
            reasons.append(
                f"applicability mismatch: expected {sorted(required)}, got {sorted(recorded_conditions)}"
            )
        if _text(row["imbalance_band"]) != imbalance_band(ratio):
            reasons.append("imbalance_band does not match ir_majority_minority")
        expected_eligible = not reasons
        if reasons and recorded_eligible:
            blockers.append(f"{dataset_id}: eligibility rule failures: {'; '.join(reasons)}")
        if recorded_eligible is None:
            blockers.append(f"{dataset_id}: eligible is not boolean")
        elif recorded_eligible != expected_eligible:
            blockers.append(
                f"{dataset_id}: eligible={recorded_eligible} disagrees with rule-derived value {expected_eligible}"
            )
        if expected_eligible:
            eligible_ids.append(dataset_id)
        elif not _text(row["exclusion_reason"]):
            blockers.append(f"{dataset_id}: ineligible row has no exclusion_reason")
        elif recorded_eligible:
            blockers.append(f"{dataset_id}: eligible row has exclusion_reason")

    eligible_count = len(eligible_ids)
    coverage = _coverage(frame)
    if require_coverage:
        if not min_datasets <= eligible_count <= max_datasets:
            blockers.append(
                f"eligible dataset count {eligible_count} is outside [{min_datasets}, {max_datasets}]"
            )
        for feature_type, minimum in FEATURE_QUOTAS.items():
            observed = coverage["feature_type"][feature_type]
            if observed < minimum:
                blockers.append(f"feature coverage {feature_type}: {observed} < {minimum}")
        for band, minimum in BAND_QUOTAS.items():
            observed = coverage["imbalance_band"][band]
            if observed < minimum:
                blockers.append(f"imbalance coverage {band}: {observed} < {minimum}")
    elif eligible_count == 0:
        warnings.append("registry has no eligible rows while coverage checks are disabled")

    status = "READY" if not blockers else "BLOCKED"
    return RegistryValidation(status, tuple(blockers), tuple(warnings), eligible_count, coverage)


def validate_registry_file(
    path: Path,
    *,
    require_coverage: bool = True,
    min_datasets: int = 40,
    max_datasets: int = 60,
) -> RegistryValidation:
    """Load and validate a V2 registry CSV."""

    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        return RegistryValidation("BLOCKED", (f"registry is unreadable: {error}",), (), 0, {})
    return validate_registry_frame(
        frame,
        require_coverage=require_coverage,
        min_datasets=min_datasets,
        max_datasets=max_datasets,
    )


def coverage_report(validation: RegistryValidation) -> dict[str, Any]:
    """Serialize a validation result for Gate 3 evidence."""

    return {
        "status": validation.status,
        "eligible_count": validation.eligible_count,
        "coverage": validation.coverage,
        "blockers": list(validation.blockers),
        "warnings": list(validation.warnings),
        "selection_basis": "provenance_eligibility_only; no benchmark scores inspected",
    }
