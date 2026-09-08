import pandas as pd
import pytest

from src.pilot import build_pipeline, feature_layout, sampler_for


def test_feature_layout_honours_semantic_categorical_override() -> None:
    frame = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "encoded_category": [1, 2, 1],
            "target": ["a", "b", "c"],
        }
    )

    layout = feature_layout(frame, "target", categorical_columns=("encoded_category",))

    assert layout.numeric == ("numeric",)
    assert layout.categorical == ("encoded_category",)
    assert sampler_for("smotenc", layout, random_state=0)[1] is True
    with pytest.raises(ValueError, match="not valid for mixed data"):
        sampler_for("smote", layout, random_state=0)


def test_feature_layout_rejects_unknown_categorical_override() -> None:
    frame = pd.DataFrame({"numeric": [1, 2, 3], "target": ["a", "b", "c"]})

    with pytest.raises(ValueError, match="not present"):
        feature_layout(frame, "target", categorical_columns=("missing",))


def test_numeric_pipeline_fits_through_imbalanced_pipeline() -> None:
    frame = pd.DataFrame(
        {
            "x1": range(9),
            "x2": [value * 2 for value in range(9)],
            "target": ["a", "b", "c"] * 3,
        }
    )
    layout = feature_layout(frame, "target")
    pipeline = build_pipeline(layout, "raw", "random_forest", random_state=0)

    pipeline.fit(frame.drop(columns=["target"]), frame["target"])
