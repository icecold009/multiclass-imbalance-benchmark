import datetime as dt
import json
from pathlib import Path

import pandas as pd

from src.stage0 import audit_csv, build_scope_lock, sha256_file

REGISTRY_COLUMNS = [
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
]


def _provenance_fixture(tmp_path: Path, dataset_id: str = "cmc") -> tuple[Path, Path, Path]:
    root = tmp_path
    raw_path = root / "data" / "raw" / f"{dataset_id}.csv"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("feature,target\n1,a\n2,b\n", encoding="utf-8")
    digest = sha256_file(raw_path)
    registry_path = root / "data" / "dataset_registry.csv"
    manifest_path = root / "data" / "acquisition_manifest.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_row = {
        "dataset_id": dataset_id,
        "display_name": "Fixture",
        "source": "Fixture source",
        "source_id": f"fixture:{dataset_id}",
        "source_url": "https://example.test/dataset",
        "license_or_terms": "CC BY 4.0",
        "source_version": "1",
        "raw_sha256": digest,
        "task_type": "single-label-classification",
        "target_column": "target",
        "n_rows": "2",
        "n_classes": "2",
        "class_counts": "a:1|b:1",
        "ir_majority_minority": "1",
        "n_min_class": "1",
        "d_raw": "1",
        "d_encoded": "1",
        "feature_type": "numeric",
        "has_missing": "False",
        "has_groups": "False",
        "has_time_order": "False",
        "has_duplicates": "False",
        "leakage_status": "pass",
        "eligible": "True",
        "exclusion_reason": "",
        "applicable_families": "numeric;raw;class_weighted;random_over;random_under;smote;adasyn;borderline_smote;smoteenn;smotetomek;balanced_random_forest",
        "notes": "fixture",
    }
    pd.DataFrame([registry_row], columns=REGISTRY_COLUMNS).to_csv(registry_path, index=False)
    manifest_row = {
        "dataset_id": dataset_id,
        "source": "Fixture source",
        "source_id": f"fixture:{dataset_id}",
        "source_url": "https://example.test/dataset",
        "source_version": "1",
        "accessed_at_utc": "2026-08-21T00:00:00Z",
        "license_or_terms": "CC BY 4.0",
        "download_method": "fixture",
        "source_file_name": raw_path.name,
        "local_path": f"data/raw/{dataset_id}.csv",
        "raw_sha256": digest,
        "file_size_bytes": str(raw_path.stat().st_size),
        "download_status": "downloaded",
        "notes": "fixture",
    }
    pd.DataFrame([manifest_row]).to_csv(manifest_path, index=False)
    return registry_path, manifest_path, raw_path


def _reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _pilot_fixture(
    tmp_path: Path,
    registry_path: Path,
    manifest_path: Path,
    *,
    deterministic_replay: str = "passed",
) -> Path:
    run_dir = tmp_path / "artifacts" / "runs" / "pilot-cmc"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pilot_metadata.json").write_text("{}\n", encoding="utf-8")
    conditions = [
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
    cells = [
        {"dataset": "cmc", "seed": seed, "fold": fold, "classifier": classifier, "condition": condition, "feature_type": "numeric"}
        for seed in range(3)
        for fold in range(5)
        for classifier in ("random_forest", "xgboost", "logistic_regression")
        for condition in conditions
    ]
    pd.DataFrame(cells[:1]).to_csv(run_dir / "pilot_results.csv", index=False)
    pd.DataFrame(
        [
            {**cell, "stage": "fit", "error_type": "ValueError", "error": "not applicable"}
            for cell in cells[1:]
        ]
    ).to_csv(run_dir / "pilot_failures.csv", index=False)
    (run_dir / "pilot_rankings.csv").write_text("condition\nraw\n", encoding="utf-8")
    (run_dir / "pilot_friedman_summary.csv").write_text("status\nsmoke\n", encoding="utf-8")
    config_path = tmp_path / "config" / "stage0.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("protocol_version: fixture\n", encoding="utf-8")
    environment_path = tmp_path / "artifacts" / "environment.json"
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.write_text('{"python":"fixture"}\n', encoding="utf-8")
    artifacts = [_reference(path) for path in sorted(run_dir.iterdir())]
    evidence = {
        "schema_version": "stage0-pilot-review-v1",
        "reviewed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "registry_sha256": sha256_file(registry_path),
        "manifest_sha256": sha256_file(manifest_path),
        "environment": _reference(environment_path),
        "configuration": _reference(config_path),
        "runs": [
            {
                "dataset_id": "cmc",
                "artifacts": artifacts,
                "results_path": str(run_dir / "pilot_results.csv"),
                "failures_path": str(run_dir / "pilot_failures.csv"),
                "runtime_evidence": _reference(run_dir / "pilot_results.csv") | {"wall_clock_seconds": 1.0},
                "memory_evidence": _reference(run_dir / "pilot_results.csv") | {"method": "fixture RSS", "peak_mb": 1.0},
                "runtime_budget_seconds": 10,
                "deterministic_replay": deterministic_replay,
                "failure_review": "passed",
                "output_completeness": "complete",
                "expected_cells": 495,
                "valid_cells": 1,
                "failure_cells": 494,
            }
        ],
    }
    evidence_path = tmp_path / "artifacts" / "runs" / "stage0-pilot-review.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence_path


def test_audit_csv_records_multiclass_distribution(tmp_path: Path) -> None:
    path = tmp_path / "candidate.csv"
    pd.DataFrame(
        {
            "amount": [1, 2, 3, 4, 5, 6],
            "segment": ["a", "a", "b", "b", "c", "c"],
            "target": ["x", "x", "y", "y", "z", "z"],
        }
    ).to_csv(path, index=False)

    evidence = audit_csv(path, "target")

    assert evidence["n_rows"] == 6
    assert evidence["n_classes"] == 3
    assert evidence["class_counts"] == {"x": 2, "y": 2, "z": 2}
    assert evidence["feature_type"] == "mixed"
    assert len(evidence["raw_sha256"]) == 64


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("stage0", encoding="utf-8")

    assert sha256_file(path) == sha256_file(path)


def test_scope_lock_requires_all_registry_and_manifest_columns(tmp_path: Path) -> None:
    registry = tmp_path / "data" / "dataset_registry.csv"
    registry.parent.mkdir(parents=True)
    pd.DataFrame([{"dataset_id": "cmc", "eligible": "True"}]).to_csv(registry, index=False)

    payload = build_scope_lock(registry, tmp_path / "lock.json")

    assert payload["status"] == "pending"
    assert any("dataset registry missing required columns" in blocker for blocker in payload["blockers"])


def test_scope_lock_rejects_invalid_pilot_status(tmp_path: Path) -> None:
    registry, _, _ = _provenance_fixture(tmp_path)

    try:
        build_scope_lock(registry, tmp_path / "lock.json", pilot_status="approved")
    except ValueError as error:
        assert "pilot_status" in str(error)
    else:
        raise AssertionError("invalid pilot status was accepted")


def test_scope_lock_stays_pending_for_missing_or_stale_pilot_evidence(tmp_path: Path) -> None:
    registry, manifest, _ = _provenance_fixture(tmp_path)
    missing = build_scope_lock(registry, tmp_path / "missing.json", pilot_status="passed", manifest_path=manifest)
    assert missing["status"] == "pending"
    assert any("pilot review evidence is missing" in blocker for blocker in missing["blockers"])

    evidence = _pilot_fixture(tmp_path, registry, manifest)
    evidence.write_text("{malformed", encoding="utf-8")
    malformed = build_scope_lock(
        registry,
        tmp_path / "malformed.json",
        pilot_status="passed",
        manifest_path=manifest,
        pilot_evidence_path=evidence,
    )
    assert malformed["status"] == "pending"
    assert any("pilot review evidence is malformed" in blocker for blocker in malformed["blockers"])

    evidence = _pilot_fixture(tmp_path, registry, manifest)
    evidence.write_text(evidence.read_text(encoding="utf-8").replace(sha256_file(registry), "0" * 64), encoding="utf-8")
    stale = build_scope_lock(
        registry,
        tmp_path / "stale.json",
        pilot_status="passed",
        manifest_path=manifest,
        pilot_evidence_path=evidence,
    )
    assert stale["status"] == "pending"
    assert any("stale registry hash" in blocker for blocker in stale["blockers"])


def test_scope_lock_rejects_malformed_applicability_conditions(tmp_path: Path) -> None:
    registry, manifest, _ = _provenance_fixture(tmp_path)
    frame = pd.read_csv(registry)
    frame.loc[0, "applicable_families"] = "numeric;raw class_weighted"
    frame.to_csv(registry, index=False)

    payload = build_scope_lock(registry, tmp_path / "lock.json", manifest_path=manifest)

    assert payload["status"] == "pending"
    assert any("unknown applicability condition" in blocker for blocker in payload["blockers"])


def test_scope_lock_rejects_duplicate_dataset_ids(tmp_path: Path) -> None:
    registry, manifest, _ = _provenance_fixture(tmp_path)
    frame = pd.read_csv(registry)
    frame.to_csv(registry, index=False, mode="w")
    frame = pd.concat([frame, frame], ignore_index=True)
    frame.to_csv(registry, index=False)
    manifest_frame = pd.read_csv(manifest)
    pd.concat([manifest_frame, manifest_frame], ignore_index=True).to_csv(manifest, index=False)

    payload = build_scope_lock(registry, tmp_path / "lock.json", manifest_path=manifest)

    assert payload["status"] == "pending"
    assert any("dataset_id contains duplicate" in blocker for blocker in payload["blockers"])


def test_scope_lock_rejects_eligible_rows_with_incomplete_provenance(tmp_path: Path) -> None:
    registry, manifest, _ = _provenance_fixture(tmp_path)
    frame = pd.read_csv(registry)
    frame.loc[0, "license_or_terms"] = ""
    frame.to_csv(registry, index=False)

    payload = build_scope_lock(registry, tmp_path / "lock.json", manifest_path=manifest)

    assert payload["status"] == "pending"
    assert any("license_or_terms mismatch" in blocker for blocker in payload["blockers"])


def test_scope_lock_output_is_deterministic(tmp_path: Path) -> None:
    registry, manifest, _ = _provenance_fixture(tmp_path)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = build_scope_lock(registry, first_path, manifest_path=manifest)
    second = build_scope_lock(registry, second_path, manifest_path=manifest)

    assert first == second
    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")


def test_scope_lock_never_reports_ready_with_unresolved_gate(tmp_path: Path) -> None:
    registry, manifest, _ = _provenance_fixture(tmp_path)
    evidence = _pilot_fixture(tmp_path, registry, manifest, deterministic_replay="pending")

    payload = build_scope_lock(
        registry,
        tmp_path / "lock.json",
        pilot_status="passed",
        manifest_path=manifest,
        pilot_evidence_path=evidence,
    )

    assert payload["status"] == "pending"
    assert any("deterministic replay is not passed" in blocker for blocker in payload["blockers"])


def test_scope_lock_accepts_human_approval_only_when_evidence_is_complete(tmp_path: Path) -> None:
    registry, manifest, _ = _provenance_fixture(tmp_path)
    evidence = _pilot_fixture(tmp_path, registry, manifest)
    pending = build_scope_lock(registry, tmp_path / "pending.json", manifest_path=manifest)
    ready = build_scope_lock(
        registry,
        tmp_path / "ready.json",
        pilot_status="passed",
        manifest_path=manifest,
        pilot_evidence_path=evidence,
    )

    assert pending["status"] == "pending"
    assert ready["status"] == "ready"
    assert ready["applicability_matrix"]["cmc"] == [
        "numeric",
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
    ]
