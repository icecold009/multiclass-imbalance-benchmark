import pandas as pd
import pytest

from src.benchmark import (
    EXPECTED_CELLS_PER_DATASET,
    expected_cell_keys,
    validate_output_frames,
)


def _frames(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = sorted(expected_cell_keys(dataset_id), key=str)
    result_rows = [
        {
            "dataset": key[0],
            "seed": key[1],
            "fold": key[2],
            "classifier": key[3],
            "condition": key[4],
            "feature_type": "numeric",
            "macro_f1": 0.5,
            "g_mean": 0.5,
            "mcc": 0.5,
            "balanced_accuracy": 0.5,
            "per_class_recall": "[0.5, 0.5, 0.5]",
        }
        for key in keys[:2]
    ]
    failure_rows = [
        {
            "dataset": key[0],
            "seed": key[1],
            "fold": key[2],
            "classifier": key[3],
            "condition": key[4],
            "feature_type": "numeric",
            "stage": "fit_predict_or_accounting",
            "error_type": "ValueError",
            "error": "fixture failure",
        }
        for key in keys[2:]
    ]
    return pd.DataFrame(result_rows), pd.DataFrame(failure_rows)


def test_expected_cell_count_matches_locked_protocol() -> None:
    assert EXPECTED_CELLS_PER_DATASET == 495
    assert len(expected_cell_keys("fixture")) == EXPECTED_CELLS_PER_DATASET


def test_validate_output_frames_requires_complete_disjoint_accounting() -> None:
    results, failures = _frames("fixture")

    assert validate_output_frames("fixture", results, failures) == {
        "expected_cells": 495,
        "valid_cells": 2,
        "failure_cells": 493,
    }

    incomplete = failures.iloc[:-1].copy()
    with pytest.raises(RuntimeError, match="cell accounting mismatch"):
        validate_output_frames("fixture", results, incomplete)


def test_validate_output_frames_rejects_duplicate_keys() -> None:
    results, failures = _frames("fixture")
    duplicate = pd.concat([results, results.iloc[[0]]], ignore_index=True)

    with pytest.raises(RuntimeError, match="duplicate comparison keys"):
        validate_output_frames("fixture", duplicate, failures)
