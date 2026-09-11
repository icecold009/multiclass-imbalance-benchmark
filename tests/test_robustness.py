import pandas as pd
import pytest

from src.robustness import _resolve_artifact, _validate_family_rows


def test_artifact_reference_cannot_escape_repository(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="escapes the repository"):
        _resolve_artifact(
            tmp_path,
            {"path": "../outside.csv", "sha256": "0" * 64},
            "fixture",
        )


def test_balanced_rf_rows_are_restricted_to_random_forest() -> None:
    rows = pd.DataFrame(
        [
            {
                "family": "balanced_rf_numeric",
                "classifier": "logistic_regression",
                "condition": "balanced_random_forest",
            }
        ]
    )
    with pytest.raises(RuntimeError, match="outside its family"):
        _validate_family_rows(rows, "fixture")
