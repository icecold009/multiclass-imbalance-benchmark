"""Stage 0 validation and pilot utilities.

This module deliberately keeps data validation separate from model fitting. It
can be used to audit candidate CSV files before adding them to the committed
dataset registry. Full model execution will be added only after the registry
and applicability decisions are locked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_type(frame: pd.DataFrame, target: str) -> str:
    features = frame.drop(columns=[target])
    numeric = features.select_dtypes(include="number").shape[1]
    categorical = features.shape[1] - numeric
    if numeric and categorical:
        return "mixed"
    if numeric:
        return "numeric"
    return "categorical"


def audit_csv(path: Path, target: str) -> dict[str, Any]:
    """Audit a candidate CSV and return JSON-serialisable registry evidence."""

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not present in {path}")

    labels = frame[target].dropna()
    counts = Counter(labels.tolist())
    ordered_counts = dict(sorted(counts.items(), key=lambda item: str(item[0])))
    n_classes = len(counts)
    n_rows = len(frame)
    min_count = min(counts.values(), default=0)
    max_count = max(counts.values(), default=0)

    evidence: dict[str, Any] = {
        "dataset_id": path.stem,
        "display_name": path.stem,
        "source_version": "local-unverified",
        "raw_sha256": sha256_file(path),
        "task_type": "single-label-classification",
        "target_column": target,
        "n_rows": n_rows,
        "n_classes": n_classes,
        "class_counts": ordered_counts,
        "ir_majority_minority": (
            max_count / min_count if min_count else None
        ),
        "n_min_class": min_count,
        "d_raw": frame.shape[1] - 1,
        "feature_type": _feature_type(frame, target),
        "has_missing": bool(frame.isna().any().any()),
        "has_groups": "unknown",
        "has_time_order": "unknown",
        "has_duplicates": bool(frame.duplicated().any()),
        "leakage_status": "not-audited",
        "eligible": False,
        "exclusion_reason": "registry metadata and split audit required",
        "applicable_families": [],
    }
    return evidence


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


REGISTRY_REQUIRED_COLUMNS = frozenset(
    {
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
        "eligible",
        "exclusion_reason",
        "applicable_families",
        "notes",
    }
)
MANIFEST_REQUIRED_COLUMNS = frozenset(
    {
        "dataset_id",
        "source",
        "source_id",
        "source_url",
        "source_version",
        "accessed_at_utc",
        "license_or_terms",
        "download_method",
        "source_file_name",
        "local_path",
        "raw_sha256",
        "file_size_bytes",
        "download_status",
        "notes",
    }
)
APPLICABILITY_VALUES = frozenset(
    {
        "numeric",
        "mixed",
        "categorical-only",
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
    }
)
APPLICABILITY_ORDER = (
    "numeric",
    "mixed",
    "categorical-only",
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
)
HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
PILOT_REVIEW_SCHEMA = "stage0-pilot-review-v1"
EXPECTED_PILOT_CELLS = 3 * 5 * 3 * 11


def _text(value: object) -> str:
    """Return a trimmed CSV/JSON scalar without turning missing values into text."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _parse_bool(value: object) -> bool | None:
    normalized = _text(value).lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return None


def _valid_hash(value: object) -> bool:
    return bool(HASH_PATTERN.fullmatch(_text(value)))


def _canonical_path(value: object) -> str:
    return _text(value).replace("\\", "/").casefold()


def _project_root(registry_path: Path) -> Path:
    resolved = registry_path.resolve()
    return resolved.parent.parent if resolved.parent.name.casefold() == "data" else resolved.parent


def _resolve_path(value: object, root: Path) -> Path:
    path = Path(_text(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parse_applicability(value: object) -> tuple[list[str], list[str]]:
    """Parse the registry's semicolon-delimited canonical applicability values."""

    raw = _text(value)
    if not raw:
        return [], ["applicable_families is empty"]
    parts = [part.strip() for part in raw.split(";")]
    if any(not part for part in parts):
        return [], ["applicable_families contains an empty semicolon-delimited value"]
    unknown = sorted({part for part in parts if part not in APPLICABILITY_VALUES})
    if unknown:
        return [], [f"unknown applicability condition(s): {', '.join(unknown)}"]
    if len(set(parts)) != len(parts):
        return [], ["applicable_families contains duplicate conditions"]
    order = {value: index for index, value in enumerate(APPLICABILITY_ORDER)}
    return sorted(parts, key=order.__getitem__), []


def _read_csv(path: Path, label: str, blockers: list[str]) -> pd.DataFrame:
    if not path.is_file():
        blockers.append(f"{label} is missing: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        blockers.append(f"{label} is malformed: {type(error).__name__}: {error}")
        return pd.DataFrame()


def _missing_columns(
    frame: pd.DataFrame, required: frozenset[str], label: str, blockers: list[str]
) -> bool:
    missing = sorted(required.difference(frame.columns))
    if missing:
        blockers.append(f"{label} missing required columns: {', '.join(missing)}")
        return True
    return False


def _check_unique_nonempty(
    frame: pd.DataFrame, column: str, label: str, blockers: list[str]
) -> None:
    if column not in frame.columns:
        return
    values = frame[column].map(_text)
    if values.eq("").any():
        blockers.append(f"{label}.{column} contains an empty value")
    duplicates = sorted(values[values.duplicated(keep=False)].unique())
    if duplicates:
        blockers.append(f"{label}.{column} contains duplicate value(s): {', '.join(duplicates)}")


def _validate_registry_and_manifest(
    registry_path: Path, manifest_path: Path
) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Validate both provenance tables and return blockers plus lock candidates."""

    blockers: list[str] = []
    registry = _read_csv(registry_path, "dataset registry", blockers)
    manifest = _read_csv(manifest_path, "acquisition manifest", blockers)
    registry_missing = _missing_columns(registry, REGISTRY_REQUIRED_COLUMNS, "dataset registry", blockers)
    manifest_missing = _missing_columns(manifest, MANIFEST_REQUIRED_COLUMNS, "acquisition manifest", blockers)
    if registry_missing or manifest_missing:
        return {"registry": blockers, "pilot": []}, registry, manifest, {}

    for column in ("dataset_id", "source_id"):
        _check_unique_nonempty(registry, column, "dataset registry", blockers)
        _check_unique_nonempty(manifest, column, "acquisition manifest", blockers)
    _check_unique_nonempty(manifest, "local_path", "acquisition manifest", blockers)

    registry_ids = set(registry["dataset_id"].map(_text))
    manifest_ids = set(manifest["dataset_id"].map(_text))
    if registry_ids != manifest_ids:
        missing = sorted(registry_ids - manifest_ids)
        extra = sorted(manifest_ids - registry_ids)
        if missing:
            blockers.append(f"acquisition manifest is missing dataset(s): {', '.join(missing)}")
        if extra:
            blockers.append(f"acquisition manifest has unregistered dataset(s): {', '.join(extra)}")

    manifest_paths = manifest["local_path"].map(_canonical_path)
    duplicate_paths = sorted(manifest_paths[manifest_paths.duplicated(keep=False)].unique())
    if duplicate_paths:
        blockers.append(f"acquisition manifest.local_path contains duplicate path(s): {', '.join(duplicate_paths)}")

    root = _project_root(registry_path)
    manifest_by_id = manifest.set_index("dataset_id", drop=False)
    registry_by_id = registry.set_index("dataset_id", drop=False)
    applicability: dict[str, list[str]] = {}
    comparable_fields = (
        "source",
        "source_id",
        "source_url",
        "source_version",
        "license_or_terms",
        "raw_sha256",
    )
    for dataset_id in sorted(registry_ids & manifest_ids):
        registry_row = registry_by_id.loc[dataset_id]
        manifest_row = manifest_by_id.loc[dataset_id]
        for field in comparable_fields:
            registry_value = _text(registry_row[field])
            manifest_value = _text(manifest_row[field])
            if not registry_value:
                blockers.append(f"{dataset_id}: registry {field} is empty")
            if not manifest_value:
                blockers.append(f"{dataset_id}: acquisition manifest {field} is empty")
            versions_match = field == "source_version" and (
                manifest_value in registry_value or registry_value in manifest_value
            )
            hashes_match = field == "raw_sha256" and registry_value.lower() == manifest_value.lower()
            if registry_value != manifest_value and not versions_match and not hashes_match:
                blockers.append(f"{dataset_id}: registry/manifest {field} mismatch")
        for field in ("local_path", "file_size_bytes"):
            if field in registry.columns and _text(registry_row[field]) != _text(manifest_row[field]):
                blockers.append(f"{dataset_id}: registry/manifest {field} mismatch")

        raw_hash = _text(manifest_row["raw_sha256"]).lower()
        if not _valid_hash(raw_hash):
            blockers.append(f"{dataset_id}: acquisition manifest has an invalid SHA-256")
        manifest_size = _text(manifest_row["file_size_bytes"])
        try:
            expected_size = int(manifest_size)
            if expected_size < 1:
                raise ValueError
        except ValueError:
            blockers.append(f"{dataset_id}: acquisition manifest has an invalid byte size")
            expected_size = None

        local_path = _resolve_path(manifest_row["local_path"], root)
        if not local_path.is_file():
            blockers.append(f"{dataset_id}: raw file is missing at {manifest_row['local_path']}")
        else:
            actual_size = local_path.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                blockers.append(f"{dataset_id}: raw file byte size does not match acquisition manifest")
            if _valid_hash(raw_hash) and sha256_file(local_path) != raw_hash:
                blockers.append(f"{dataset_id}: raw file SHA-256 does not match acquisition manifest")

        eligible = _parse_bool(registry_row["eligible"])
        if eligible is None:
            blockers.append(f"{dataset_id}: eligible must be true or false")
        elif eligible and _text(registry_row["exclusion_reason"]):
            blockers.append(f"{dataset_id}: eligible row has an exclusion reason")
        elif eligible is False and not _text(registry_row["exclusion_reason"]):
            blockers.append(f"{dataset_id}: excluded row is missing an exclusion reason")
        if eligible:
            for field in (
                "source",
                "source_id",
                "source_url",
                "source_version",
                "license_or_terms",
                "raw_sha256",
                "target_column",
                "task_type",
            ):
                if not _text(registry_row[field]):
                    blockers.append(f"{dataset_id}: eligible row has incomplete {field} provenance")

        leakage_status = _text(registry_row["leakage_status"]).lower()
        if not leakage_status:
            blockers.append(f"{dataset_id}: leakage_status is empty")
        elif eligible and not leakage_status.startswith("pass"):
            blockers.append(f"{dataset_id}: eligible row does not have passing leakage status")

        conditions, condition_errors = _parse_applicability(registry_row["applicable_families"])
        for error in condition_errors:
            blockers.append(f"{dataset_id}: {error}")
        if conditions:
            feature_type = _text(registry_row["feature_type"])
            if feature_type == "categorical":
                feature_type = "categorical-only"
            feature_types = [value for value in conditions if value in {"numeric", "mixed", "categorical-only"}]
            if feature_types != [feature_type]:
                blockers.append(f"{dataset_id}: applicability feature type does not match feature_type")
            applicability[dataset_id] = conditions

        for field in (
            "accessed_at_utc",
            "download_method",
            "source_file_name",
            "local_path",
            "file_size_bytes",
            "download_status",
        ):
            if not _text(manifest_row[field]):
                blockers.append(f"{dataset_id}: acquisition manifest {field} is empty")

    eligible_ids = sorted(
        _text(row["dataset_id"])
        for _, row in registry.iterrows()
        if _parse_bool(row["eligible"]) is True and _text(row["dataset_id"])
    )
    return {"registry": blockers, "pilot": []}, registry, manifest, {
        "eligible_ids": eligible_ids,
        "applicability": applicability,
    }


def _parse_timestamp(value: object) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat(_text(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def _verify_file_reference(
    reference: object,
    label: str,
    root: Path,
    blockers: list[str],
) -> Path | None:
    if not isinstance(reference, dict):
        blockers.append(f"{label} must contain a path and SHA-256")
        return None
    path_text = _text(reference.get("path"))
    digest = _text(reference.get("sha256")).lower()
    if not path_text or not _valid_hash(digest):
        blockers.append(f"{label} has a missing or invalid path/hash")
        return None
    path = _resolve_path(path_text, root)
    if not path.is_file():
        blockers.append(f"{label} path is missing: {path_text}")
        return None
    if sha256_file(path) != digest:
        blockers.append(f"{label} hash is stale: {path_text}")
        return None
    return path


def _as_finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_pilot_outputs(
    run: dict[str, Any],
    dataset_id: str,
    root: Path,
    blockers: list[str],
) -> None:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        blockers.append(f"{dataset_id}: pilot evidence has no artifact manifest")
        artifact_paths: set[str] = set()
    else:
        artifact_paths = set()
        for index, artifact in enumerate(artifacts):
            path = _verify_file_reference(artifact, f"{dataset_id} artifact {index}", root, blockers)
            if path is not None:
                artifact_paths.add(_canonical_path(path))

    required_names = {
        "pilot_metadata.json",
        "pilot_results.csv",
        "pilot_failures.csv",
        "pilot_rankings.csv",
        "pilot_friedman_summary.csv",
    }
    present_names = {Path(path).name for path in artifact_paths}
    missing_names = sorted(required_names - present_names)
    if missing_names:
        blockers.append(f"{dataset_id}: pilot output artifact(s) missing: {', '.join(missing_names)}")

    runtime = run.get("runtime_evidence")
    memory = run.get("memory_evidence")
    runtime_path = _verify_file_reference(runtime, f"{dataset_id} runtime evidence", root, blockers)
    memory_path = _verify_file_reference(memory, f"{dataset_id} memory evidence", root, blockers)
    if runtime_path is not None and _canonical_path(runtime_path) not in artifact_paths:
        blockers.append(f"{dataset_id}: runtime evidence is not in the artifact manifest")
    if memory_path is not None and _canonical_path(memory_path) not in artifact_paths:
        blockers.append(f"{dataset_id}: memory evidence is not in the artifact manifest")

    runtime_seconds = _as_finite_number(
        runtime.get("wall_clock_seconds") if isinstance(runtime, dict) else None
    )
    memory_mb = _as_finite_number(memory.get("peak_mb") if isinstance(memory, dict) else None)
    runtime_budget = _as_finite_number(run.get("runtime_budget_seconds"))
    if runtime_seconds is None or runtime_seconds < 0:
        blockers.append(f"{dataset_id}: runtime evidence has no finite wall-clock value")
    if runtime_budget is None or runtime_budget <= 0:
        blockers.append(f"{dataset_id}: runtime budget is missing or invalid")
    elif runtime_seconds is not None and runtime_seconds > runtime_budget:
        blockers.append(f"{dataset_id}: pilot runtime exceeds the recorded budget")
    if not isinstance(memory, dict) or not _text(memory.get("method")):
        blockers.append(f"{dataset_id}: peak-memory measurement method is missing")
    if memory_mb is None or memory_mb < 0:
        blockers.append(f"{dataset_id}: peak-memory evidence has no finite value")

    if _text(run.get("deterministic_replay")).lower() != "passed":
        blockers.append(f"{dataset_id}: deterministic replay is not passed")
    if _text(run.get("failure_review")).lower() != "passed":
        blockers.append(f"{dataset_id}: failure review is not passed")
    if _text(run.get("output_completeness")).lower() != "complete":
        blockers.append(f"{dataset_id}: output completeness is not complete")

    expected = run.get("expected_cells")
    valid = run.get("valid_cells")
    failures = run.get("failure_cells")
    integer_values: list[int] = []
    for field, value in (("expected_cells", expected), ("valid_cells", valid), ("failure_cells", failures)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            blockers.append(f"{dataset_id}: {field} must be a non-negative integer")
        else:
            integer_values.append(value)
    if len(integer_values) == 3 and valid + failures != expected:
        blockers.append(f"{dataset_id}: valid and failure cells do not cover expected cells")
    if isinstance(expected, int) and expected != EXPECTED_PILOT_CELLS:
        blockers.append(
            f"{dataset_id}: expected_cells must equal the locked {EXPECTED_PILOT_CELLS}-cell pilot matrix"
        )

    results_path = _resolve_path(run.get("results_path"), root)
    failures_path = _resolve_path(run.get("failures_path"), root)
    if _canonical_path(results_path) not in artifact_paths:
        blockers.append(f"{dataset_id}: results_path is not a hashed artifact")
    if _canonical_path(failures_path) not in artifact_paths:
        blockers.append(f"{dataset_id}: failures_path is not a hashed artifact")
    try:
        results = pd.read_csv(results_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        results = pd.DataFrame()
    try:
        failure_rows = pd.read_csv(failures_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        failure_rows = pd.DataFrame()
    result_columns = {"dataset", "seed", "fold", "classifier", "condition", "feature_type"}
    failure_columns = result_columns | {"stage", "error_type", "error"}
    if not result_columns.issubset(results.columns):
        blockers.append(f"{dataset_id}: pilot_results.csv has an incomplete schema")
    if not failure_columns.issubset(failure_rows.columns):
        blockers.append(f"{dataset_id}: pilot_failures.csv has an incomplete schema")
    if isinstance(valid, int) and len(results) != valid:
        blockers.append(f"{dataset_id}: valid_cells does not match pilot_results.csv")
    if isinstance(failures, int) and len(failure_rows) != failures:
        blockers.append(f"{dataset_id}: failure_cells does not match pilot_failures.csv")
    if result_columns.issubset(results.columns) and failure_columns.issubset(failure_rows.columns):
        key_columns = ["dataset", "seed", "fold", "classifier", "condition"]
        combined = pd.concat([results[key_columns], failure_rows[key_columns]], ignore_index=True)
        if combined.duplicated().any():
            blockers.append(f"{dataset_id}: pilot output contains duplicate comparison cells")
        if not results.empty and set(results["dataset"].astype(str)) != {dataset_id}:
            blockers.append(f"{dataset_id}: pilot_results.csv contains another dataset")
        if not failure_rows.empty and set(failure_rows["dataset"].astype(str)) != {dataset_id}:
            blockers.append(f"{dataset_id}: pilot_failures.csv contains another dataset")


def _validate_pilot_evidence(
    evidence_path: Path,
    root: Path,
    registry_sha256: str,
    manifest_sha256: str,
    eligible_ids: list[str],
) -> list[str]:
    blockers: list[str] = []
    if not evidence_path.is_file():
        return [f"pilot review evidence is missing: {evidence_path}"]
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return [f"pilot review evidence is malformed: {type(error).__name__}: {error}"]
    if not isinstance(evidence, dict):
        return ["pilot review evidence must be a JSON object"]
    if evidence.get("schema_version") != PILOT_REVIEW_SCHEMA:
        blockers.append(f"pilot review evidence schema must be {PILOT_REVIEW_SCHEMA}")
    reviewed_at = _parse_timestamp(evidence.get("reviewed_at_utc"))
    if reviewed_at is None:
        blockers.append("pilot review evidence has an invalid reviewed_at_utc")
    elif reviewed_at > dt.datetime.now(dt.UTC) + dt.timedelta(minutes=5):
        blockers.append("pilot review evidence is dated in the future")
    if _text(evidence.get("registry_sha256")).lower() != registry_sha256:
        blockers.append("pilot review evidence has a stale registry hash")
    if _text(evidence.get("manifest_sha256")).lower() != manifest_sha256:
        blockers.append("pilot review evidence has a stale acquisition-manifest hash")
    _verify_file_reference(evidence.get("environment"), "environment evidence", root, blockers)
    _verify_file_reference(evidence.get("configuration"), "configuration evidence", root, blockers)

    runs = evidence.get("runs")
    if not isinstance(runs, list):
        return blockers + ["pilot review evidence runs must be a list"]
    runs_by_id: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            blockers.append("pilot review evidence contains a non-object run")
            continue
        dataset_id = _text(run.get("dataset_id"))
        if not dataset_id:
            blockers.append("pilot review evidence contains a run without dataset_id")
        elif dataset_id in runs_by_id:
            blockers.append(f"pilot review evidence contains duplicate run for {dataset_id}")
        else:
            runs_by_id[dataset_id] = run
    missing_runs = sorted(set(eligible_ids) - set(runs_by_id))
    extra_runs = sorted(set(runs_by_id) - set(eligible_ids))
    if missing_runs:
        blockers.append(f"pilot review evidence is missing eligible dataset(s): {', '.join(missing_runs)}")
    if extra_runs:
        blockers.append(f"pilot review evidence has non-eligible dataset(s): {', '.join(extra_runs)}")
    for dataset_id in sorted(set(eligible_ids) & set(runs_by_id)):
        _validate_pilot_outputs(runs_by_id[dataset_id], dataset_id, root, blockers)
    return blockers


def build_scope_lock(
    registry_path: Path,
    output_path: Path,
    *,
    pilot_status: str = "pending",
    manifest_path: Path | None = None,
    pilot_evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Build deterministic, evidence-backed Gate A status.

    ``pilot_status`` records a human review decision, but it is never enough
    to pass the gate. A current, hashed pilot-review manifest must verify the
    runtime, memory, deterministic replay, failures, and complete output cells.
    """

    if pilot_status not in {"pending", "passed"}:
        raise ValueError("pilot_status must be either 'pending' or 'passed'")
    manifest_path = manifest_path or registry_path.parent / "acquisition_manifest.csv"
    root = _project_root(registry_path)
    pilot_evidence_path = pilot_evidence_path or root / "artifacts" / "runs" / "stage0-pilot-review.json"
    registry_sha256 = sha256_file(registry_path) if registry_path.is_file() else ""
    manifest_sha256 = sha256_file(manifest_path) if manifest_path.is_file() else ""
    validation, _registry, _manifest, candidates = _validate_registry_and_manifest(
        registry_path, manifest_path
    )
    blockers = list(validation["registry"])
    eligible_ids = candidates.get("eligible_ids", [])
    applicability = candidates.get("applicability", {})
    pilot_blockers = _validate_pilot_evidence(
        pilot_evidence_path,
        root,
        registry_sha256,
        manifest_sha256,
        eligible_ids,
    )
    blockers.extend(pilot_blockers)
    if pilot_status != "passed":
        blockers.append("Gate A human pilot status is not passed")
    if not eligible_ids:
        blockers.append("no eligible registry rows have complete provenance")
    blockers = sorted(set(blockers))
    dataset_ids = sorted(eligible_ids)
    applicable_families = sorted({condition for values in applicability.values() for condition in values})
    payload: dict[str, Any] = {
        "schema_version": "stage0-scope-lock-v2",
        "registry_path": str(registry_path),
        "registry_sha256": registry_sha256,
        "acquisition_manifest_path": str(manifest_path),
        "acquisition_manifest_sha256": manifest_sha256,
        "pilot_evidence_path": str(pilot_evidence_path),
        "pilot_status": pilot_status,
        "status": "ready" if not blockers else "pending",
        "dataset_ids": dataset_ids,
        "dataset_count": len(dataset_ids),
        "applicable_families": applicable_families,
        "applicability_matrix": {dataset_id: applicability[dataset_id] for dataset_id in dataset_ids if dataset_id in applicability},
        "blockers": blockers,
        "selection_rule": "registry eligibility and provenance only; pilot scores are not used",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a Stage 0 CSV candidate")
    parser.add_argument("path_or_command", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("data/dataset_registry.csv"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pilot-evidence", type=Path)
    parser.add_argument("--pilot-status", choices=("pending", "passed"), default="pending")
    args = parser.parse_args()

    if str(args.path_or_command).lower() == "lock":
        output_path = args.json_out or Path("artifacts/runs/scope-lock.json")
        build_scope_lock(
            args.registry,
            output_path,
            pilot_status=args.pilot_status,
            manifest_path=args.manifest,
            pilot_evidence_path=args.pilot_evidence,
        )
        print(output_path)
        return
    if not args.target:
        parser.error("audit requires a CSV path and --target")
    evidence = audit_csv(args.path_or_command, args.target)
    rendered = json.dumps(evidence, indent=2, default=str)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
