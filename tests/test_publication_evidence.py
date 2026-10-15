import hashlib
import json

import pandas as pd
import pytest

from src.publication_evidence import (
    FAILURE_SCHEMA,
    RESULT_SCHEMA,
    _resolve_reference,
    _same_counts,
    validate_aggregate_records,
)


def _result(**overrides):
    row = {
        "dataset": "sample",
        "seed": 0,
        "fold": 0,
        "classifier": "random_forest",
        "condition": "raw",
        "feature_type": "numeric",
        "macro_f1": 0.5,
        "g_mean": 0.5,
        "mcc": 0.0,
        "balanced_accuracy": 0.5,
        "per_class_recall": json.dumps([0.5, 0.5, 0.5]),
        "fit_predict_seconds": 0.25,
        "rss_before_mb": 100.0,
        "rss_after_mb": 102.0,
        "rss_delta_mb": 2.0,
        "train_rows_before_sampling": 10,
        "train_rows_after_sampling": 10,
    }
    row.update(overrides)
    return row


def _failure(**overrides):
    row = {
        "dataset": "sample",
        "seed": 0,
        "fold": 0,
        "classifier": "random_forest",
        "condition": "smotenc",
        "feature_type": "numeric",
        "stage": "fit_predict_or_accounting",
        "error_type": "ValueError",
        "error": "SMOTENC is only applicable to mixed datasets",
        "elapsed_seconds": 0.01,
    }
    row.update(overrides)
    return row


def _validate(results, failures, *, conditions=("raw",)):
    return validate_aggregate_records(
        results,
        failures,
        dataset_ids=["sample"],
        seeds=[0],
        folds=1,
        classifiers=["random_forest"],
        conditions=list(conditions),
        feature_types={"sample": "numeric"},
        class_counts={"sample": 3},
        applicability={"sample": {"raw"}},
    )


def test_aggregate_records_validate_complete_schema_and_unsupported_classification():
    results = pd.DataFrame([_result()], columns=RESULT_SCHEMA)
    failures = pd.DataFrame([_failure()], columns=FAILURE_SCHEMA)

    counts = _validate(results, failures, conditions=("raw", "smotenc"))

    assert counts == {"expected_cells": 2, "valid_cells": 1, "failure_cells": 1}


def test_aggregate_records_reject_duplicate_and_missing_keys():
    result = pd.DataFrame([_result()], columns=RESULT_SCHEMA)
    failures = pd.DataFrame(columns=FAILURE_SCHEMA)

    with pytest.raises(RuntimeError, match="duplicate cell keys"):
        _validate(pd.concat([result, result], ignore_index=True), failures)
    with pytest.raises(RuntimeError, match="key coverage is incomplete"):
        _validate(result.iloc[0:0].copy(), failures)


def test_aggregate_records_reject_valid_failure_overlap():
    results = pd.DataFrame([_result()], columns=RESULT_SCHEMA)
    failures = pd.DataFrame([_failure(condition="raw")], columns=FAILURE_SCHEMA)

    with pytest.raises(RuntimeError, match="overlap"):
        _validate(results, failures)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"macro_f1": 1.01}, r"macro_f1 must be in \[0, 1\]"),
        ({"g_mean": 0.4}, "g_mean disagrees with per-class recall"),
        ({"per_class_recall": "[0.5, 0.5]"}, "length disagrees with class count"),
        ({"rss_delta_mb": 3.0}, "rss_delta_mb disagrees"),
    ],
)
def test_aggregate_records_reject_invalid_metrics_and_recall_schema(overrides, message):
    results = pd.DataFrame([_result(**overrides)], columns=RESULT_SCHEMA)
    failures = pd.DataFrame(columns=FAILURE_SCHEMA)

    with pytest.raises(RuntimeError, match=message):
        _validate(results, failures)


def test_aggregate_records_reject_incorrect_unsupported_failure_message():
    results = pd.DataFrame([_result()], columns=RESULT_SCHEMA)
    failures = pd.DataFrame(
        [_failure(error="unexpected error")], columns=FAILURE_SCHEMA
    )

    with pytest.raises(RuntimeError, match="failure classification is inconsistent"):
        _validate(results, failures, conditions=("raw", "smotenc"))


def test_manifest_reference_detects_tampering_and_path_escape(tmp_path):
    artifact = tmp_path / "aggregate.csv"
    artifact.write_text("frozen\n", encoding="utf-8")
    reference = {
        "path": "aggregate.csv",
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    assert _resolve_reference(tmp_path, reference, "aggregate") == artifact

    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        _resolve_reference(tmp_path, reference, "aggregate")
    with pytest.raises(RuntimeError, match="escapes the evidence root"):
        _resolve_reference(tmp_path, {**reference, "path": "../outside.csv"}, "aggregate")


def test_manifest_counts_must_match_reconciled_aggregate_counts():
    expected = {"expected_cells": 2, "valid_cells": 1, "failure_cells": 1}

    with pytest.raises(RuntimeError, match="counts disagree"):
        _same_counts(
            {"expected_cells": 2, "valid_cells": 2, "failure_cells": 0},
            expected,
            "benchmark manifest",
        )
