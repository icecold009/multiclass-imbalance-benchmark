from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v2_engineering import (
    STATUS_VALID,
    V2_CONDITIONS,
    V2_MODELS,
    infer_feature_layout,
)
from src.v2_execution import (
    CELL_COLUMNS,
    FAILURE_COLUMNS,
    RESULT_COLUMNS,
    V2DatasetSpec,
    _run_cell,
    dry_run_summary,
    execution_blockers,
    expected_cell_keys,
    load_execution_context,
    run_v2_benchmark,
    validate_v2_output_frames,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v2_expected_matrix_has_five_by_five_by_eleven_cells() -> None:
    keys = expected_cell_keys("fixture")
    assert len(keys) == 5 * len(V2_MODELS) * len(V2_CONDITIONS)
    assert all(len(key) == len(CELL_COLUMNS) for key in keys)


def test_v2_execution_is_blocked_until_full_run_and_compute_are_ready() -> None:
    context = load_execution_context(ROOT)
    blockers = execution_blockers(context)
    assert any("full-run-ready" in blocker for blocker in blockers)
    assert any("compute_instance_approval" in blocker for blocker in blockers)
    assert not any("environment record" in blocker for blocker in blockers)


def test_v2_dry_run_reports_scope_without_creating_outcome_artifacts(tmp_path: Path) -> None:
    context = load_execution_context(ROOT, output_dir=tmp_path / "v2-full")
    summary = dry_run_summary(context)
    assert summary["status"] == "BLOCKED"
    assert summary["outcome_bearing_run_started"] is False
    assert summary["expected_cells"] == summary["dataset_count"] * 275
    assert not (tmp_path / "v2-full").exists()
    assert run_v2_benchmark(context, dry_run=True) == summary


def test_v2_output_validation_requires_one_record_per_cell() -> None:
    key = ("fixture", 0, 20260913, "logistic_regression", "raw")
    result = {
        column: None
        for column in RESULT_COLUMNS
    }
    result.update(
        {
            "dataset_id": key[0],
            "outer_fold": key[1],
            "outer_seed": key[2],
            "classifier": key[3],
            "condition": key[4],
            "status": STATUS_VALID,
            "macro_average_precision": 0.5,
            "multiclass_brier": 0.1,
            "multiclass_log_loss": 0.2,
        }
    )
    # The complete matrix is intentionally not fabricated in this unit test;
    # the validator must reject a partial result rather than score it.
    with pytest.raises(RuntimeError, match="does not match"):
        validate_v2_output_frames(
            "fixture",
            pd.DataFrame([result]),
            pd.DataFrame(columns=FAILURE_COLUMNS),
        )


def test_one_real_nested_cell_runs_hpo_oof_calibration_and_outer_test() -> None:
    context = load_execution_context(ROOT)
    labels = np.tile([0, 1, 2], 10)
    frame = pd.DataFrame(
        {
            "x1": np.arange(len(labels), dtype=float),
            "x2": np.sin(np.arange(len(labels))),
            "target": labels,
        }
    )
    layout = infer_feature_layout(frame, "target")
    trial_rows: list[dict[str, object]] = []
    result, failure = _run_cell(
        context=context,
        spec=V2DatasetSpec(
            dataset_id="synthetic_nested",
            raw_path=ROOT / "synthetic_nested.csv",
            raw_sha256="",
            target_column="target",
            categorical_columns=(),
            feature_type="numeric",
        ),
        layout=layout,
        X_train=frame.drop(columns=["target"]).iloc[:21],
        y_train=labels[:21],
        X_test=frame.drop(columns=["target"]).iloc[21:],
        y_test=labels[21:],
        outer_fold=0,
        classifier="logistic_regression",
        condition="raw",
        trial_rows=trial_rows,
    )
    assert failure is None
    assert result is not None
    assert result["status"] == STATUS_VALID
    assert result["selected_trial"] >= 1
    assert len(trial_rows) == 20
    assert result["calibration_status"] in {STATUS_VALID, "FAILED"}
