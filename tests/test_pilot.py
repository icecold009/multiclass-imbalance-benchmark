import numpy as np
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_sample_weight

from src import pilot
from src.pilot import (
    build_pipeline,
    feature_layout,
    fit_kwargs_for,
    geometric_mean,
    sampler_for,
)


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


def test_preprocessing_statistics_use_training_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, np.nan, 4.0, 1000.0, 2000.0],
            "x2": [0.0, 1.0, 2.0, 3.0, 4000.0, 5000.0],
            "target": ["a", "b", "c", "a", "b", "c"],
        }
    )
    layout = feature_layout(frame, "target")
    pipeline = build_pipeline(layout, "raw", "random_forest", random_state=0)

    X = frame.drop(columns=["target"])
    pipeline.fit(X.iloc[:4], frame["target"].iloc[:4])

    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps[
        "impute"
    ]
    assert imputer.statistics_.tolist() == [2.0, 1.5]


def test_sampler_only_receives_training_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "x1": range(15),
            "x2": [value * 2 for value in range(15)],
            "target": ["a", "b", "c"] * 5,
        }
    )
    layout = feature_layout(frame, "target")
    observed_rows: list[int] = []
    original_fit_resample = pilot.SMOTE.fit_resample

    def recording_fit_resample(self, X, y, *args, **kwargs):
        observed_rows.append(len(X))
        return original_fit_resample(self, X, y, *args, **kwargs)

    monkeypatch.setattr(pilot.SMOTE, "fit_resample", recording_fit_resample)
    pipeline = build_pipeline(layout, "smote", "random_forest", random_state=0)
    X = frame.drop(columns=["target"])
    X_train, X_test = X.iloc[:12], X.iloc[12:]
    y_train, y_test = frame["target"].iloc[:12], frame["target"].iloc[12:]

    pipeline.fit(X_train, y_train)
    assert observed_rows == [len(X_train)]
    pipeline.predict(X_test)
    assert observed_rows == [len(X_train)]
    assert len(y_test) == 3


def test_target_column_is_excluded_from_feature_layout() -> None:
    frame = pd.DataFrame(
        {
            "signal": [1.0, 2.0, 3.0],
            "target": ["a", "b", "c"],
        }
    )

    layout = feature_layout(frame, "target")

    assert "target" not in layout.numeric + layout.categorical
    assert layout.numeric == ("signal",)


def test_mixed_pipeline_encodes_categories_after_resampling() -> None:
    frame = pd.DataFrame(
        {
            "numeric": range(12),
            "category": ["low", "high", "mid"] * 4,
            "target": ["a", "b", "c"] * 4,
        }
    )
    layout = feature_layout(frame, "target")
    pipeline = build_pipeline(layout, "smotenc", "logistic_regression", random_state=0)

    pipeline.fit(frame.drop(columns=["target"]), frame["target"])

    assert list(pipeline.named_steps) == ["pre_sampler", "sampler", "post_sampler", "classifier"]
    encoder = pipeline.named_steps["post_sampler"].named_transformers_["categorical"]
    assert encoder.categories_[0].tolist() == [0.0, 1.0, 2.0]


def test_xgboost_weight_routing_matches_balanced_training_weights() -> None:
    y_train = pd.Series([0, 0, 0, 1, 2, 2], name="target")

    fit_kwargs = fit_kwargs_for("class_weighted", "xgboost", y_train)

    assert set(fit_kwargs) == {"classifier__sample_weight"}
    np.testing.assert_array_equal(
        fit_kwargs["classifier__sample_weight"],
        compute_sample_weight(class_weight="balanced", y=y_train),
    )
    assert fit_kwargs_for("class_weighted", "random_forest", y_train) == {}


def test_geometric_mean_is_zero_when_any_declared_class_has_zero_recall() -> None:
    y_true = pd.Series([0, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1])

    assert geometric_mean(y_true, y_pred, labels=[0, 1, 2]) == 0.0


def test_geometric_mean_uses_all_declared_class_recalls() -> None:
    y_true = pd.Series([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 0])

    expected = (1.0 * 1.0 * 0.5) ** (1 / 3)
    assert geometric_mean(y_true, y_pred, labels=[0, 1, 2]) == pytest.approx(expected)
