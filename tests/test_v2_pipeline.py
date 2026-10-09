import numpy as np
import pandas as pd
import pytest

from src.v2_engineering import FeatureLayout, categorical_path, infer_feature_layout
from src.v2_pipeline import AdaptiveSampler, NativeCategoricalEstimator, build_pipeline_v2


def _numeric_frame() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(
        {
            "x1": np.arange(30, dtype=float),
            "x2": np.tile([0.0, 1.0, 2.0], 10),
            "target": np.repeat([0, 1, 2], 10),
        }
    )
    return frame.drop(columns=["target"]), frame["target"]


def _mixed_frame() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(
        {
            "x1": np.arange(30, dtype=float),
            "category": np.tile(["a", "b", "c"], 10),
            "target": np.repeat([0, 1, 2], [12, 10, 8]),
        }
    )
    return frame.drop(columns=["target"]), frame["target"]


def test_numeric_lr_and_rf_pipelines_fit_with_adaptive_sampler() -> None:
    X, y = _numeric_frame()
    layout = infer_feature_layout(X.assign(target=y), "target")
    for classifier in ("logistic_regression", "random_forest"):
        pipeline = build_pipeline_v2(layout, "smote", classifier, random_state=1, n_estimators=8)
        assert isinstance(pipeline.named_steps["sampler"], AdaptiveSampler)
        pipeline.fit(X, y)
        assert pipeline.predict_proba(X).shape == (len(X), 3)


def test_mixed_lr_uses_ordinal_sampling_then_one_hot() -> None:
    X, y = _mixed_frame()
    layout = infer_feature_layout(X.assign(target=y), "target")
    pipeline = build_pipeline_v2(layout, "smotenc", "logistic_regression", 2, n_estimators=8)
    pipeline.fit(X, y)
    assert pipeline.named_steps["sampler"].sampler_.categorical_features == [1]
    assert pipeline.predict_proba(X).shape == (len(X), 3)


@pytest.mark.parametrize("classifier", ["xgboost", "lightgbm", "catboost"])
def test_native_gbdt_mixed_raw_path_preserves_categories(classifier: str) -> None:
    X, y = _mixed_frame()
    layout = infer_feature_layout(X.assign(target=y), "target")
    estimator = build_pipeline_v2(layout, "raw", classifier, 3, n_estimators=8)
    assert isinstance(estimator, NativeCategoricalEstimator)
    estimator.fit(X, y)
    assert estimator.predict_proba(X).shape == (len(X), 3)
    assert categorical_path(classifier, layout, "raw") == "native_categorical"


def test_native_gbdt_mixed_smote_path_is_explicitly_ordinal_then_recast() -> None:
    layout = FeatureLayout(numeric=("x1",), categorical=("category",))
    assert categorical_path("xgboost", layout, "smotenc") == "fold_local_ordinal_sampler_then_native_recast"
