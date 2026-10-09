"""Fold-safe V2 model pipelines for engineering smoke validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.v2_engineering import (
    NATIVE_CATEGORICAL_MODELS,
    SAMPLER_CONDITIONS,
    FeatureLayout,
    categorical_path,
    fit_time_sample_weight,
    prepare_native_categorical_frame,
    sampler_for_v2,
)


def _one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility with old sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _mixed_sampler_preprocessor(layout: FeatureLayout) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), list(layout.numeric)),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                list(layout.categorical),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _numeric_preprocessor(model: str) -> ColumnTransformer:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if model == "logistic_regression":
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("numeric", Pipeline(steps), slice(0, None))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _mixed_postprocessor(layout: FeatureLayout, model: str) -> ColumnTransformer:
    numeric_count = len(layout.numeric)
    categorical_indices = list(range(numeric_count, numeric_count + len(layout.categorical)))
    numeric_steps: list[tuple[str, Any]] = []
    if model == "logistic_regression":
        numeric_steps.append(("scale", StandardScaler()))
    numeric_transformer: Any = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    if model == "logistic_regression":
        categorical_transformer: Any = _one_hot()
    else:
        categorical_transformer = "passthrough"
    return ColumnTransformer(
        [
            ("numeric", numeric_transformer, list(range(numeric_count))),
            ("categorical", categorical_transformer, categorical_indices),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class AdaptiveSampler(BaseEstimator):
    """Construct a sampler from the labels of the current fit split."""

    def __init__(self, condition: str, layout: FeatureLayout, random_state: int):
        self.condition = condition
        self.layout = layout
        self.random_state = random_state

    def fit_resample(self, X: Any, y: Any) -> tuple[Any, Any]:
        self.rows_before_ = len(y)
        self.sampler_ = sampler_for_v2(
            self.condition,
            self.layout,
            y,
            random_state=self.random_state,
        )
        if self.sampler_ is None:
            self.rows_after_ = len(y)
            return X, y
        X_resampled, y_resampled = self.sampler_.fit_resample(X, y)
        self.rows_after_ = len(y_resampled)
        return X_resampled, y_resampled


def _classifier(
    name: str,
    random_state: int,
    *,
    weighted: bool = False,
    n_estimators: int = 40,
) -> Any:
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced" if weighted else None,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
            class_weight="balanced" if weighted else None,
        )
    raise ValueError(f"{name} is not a sklearn numeric baseline")


def _native_classifier(
    name: str,
    random_state: int,
    n_classes: int,
    n_estimators: int,
    model_params: dict[str, Any] | None = None,
) -> Any:
    params = dict(model_params or {})
    if name == "xgboost":
        from xgboost import XGBClassifier

        base_params = {
            "n_estimators": n_estimators,
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "enable_categorical": True,
            "n_jobs": 1,
            "random_state": random_state,
            "verbosity": 0,
        }
        base_params.update(params)
        return XGBClassifier(**base_params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        base_params = {
            "n_estimators": n_estimators,
            "objective": "multiclass",
            "num_class": n_classes,
            "n_jobs": 1,
            "verbosity": -1,
            "random_state": random_state,
        }
        base_params.update(params)
        return LGBMClassifier(**base_params)
    if name == "catboost":
        from catboost import CatBoostClassifier

        base_params = {
            "iterations": params.pop("iterations", n_estimators),
            "loss_function": "MultiClass",
            "thread_count": 1,
            "random_seed": random_state,
            "allow_writing_files": False,
            "verbose": False,
        }
        base_params.update(params)
        return CatBoostClassifier(**base_params)
    raise ValueError(f"{name} is not a native categorical model")


class NativeCategoricalEstimator(BaseEstimator, ClassifierMixin):
    """Adapt fold-local sampling to native categorical GBDTs.

    Raw and weighted paths preserve pandas categorical dtypes.  Resampling
    occurs on fold-local ordinal codes and is then recast to categorical
    columns before the native model sees the data; one-hot features are never
    passed to the native categorical models.
    """

    def __init__(
        self,
        classifier: str,
        layout: FeatureLayout,
        condition: str,
        random_state: int,
        n_estimators: int = 40,
        model_params: dict[str, Any] | None = None,
    ):
        self.classifier = classifier
        self.layout = layout
        self.condition = condition
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model_params = model_params

    def _numeric_frame(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit:
            self.numeric_imputer_ = SimpleImputer(strategy="median")
            values = self.numeric_imputer_.fit_transform(X.loc[:, list(self.layout.numeric)])
        else:
            values = self.numeric_imputer_.transform(X.loc[:, list(self.layout.numeric)])
        return pd.DataFrame(values, columns=list(self.layout.numeric), index=X.index)

    def _resampled_native_frame(self, values: np.ndarray) -> pd.DataFrame:
        columns = [*self.layout.numeric, *self.layout.categorical]
        frame = pd.DataFrame(values, columns=columns)
        for column, categories in zip(self.layout.categorical, self.ordinal_categories_):
            codes = pd.to_numeric(frame[column], errors="coerce").round()
            valid = codes.between(0, len(categories) - 1)
            labels = codes.where(valid, -1).astype(int).map(
                lambda code, categories=categories: categories[code]
                if code >= 0
                else "__MISSING__"
            )
            category_values = [*categories, "__MISSING__"]
            frame[column] = pd.Series(
                pd.Categorical(labels, categories=category_values), index=frame.index
            )
        return frame

    def _prepare_for_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.sampling_used_:
            transformed = self.pre_sampler_.transform(X)
            if not self.layout.categorical:
                return pd.DataFrame(transformed, columns=list(self.layout.numeric))
            return self._resampled_native_frame(transformed)
        if self.layout.categorical:
            prepared, _ = prepare_native_categorical_frame(
                X,
                self.layout,
                fitted_categories=self.native_categories_,
            )
            return prepared
        return self._numeric_frame(X, fit=False)

    def fit(self, X: pd.DataFrame, y: Any, sample_weight: Any = None) -> NativeCategoricalEstimator:
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)
        self.sampling_used_ = self.condition in SAMPLER_CONDITIONS
        fit_weight = fit_time_sample_weight(self.condition, self.classifier, y_array)
        if sample_weight is not None:
            fit_weight = sample_weight
        if self.sampling_used_:
            if not self.layout.categorical:
                self.pre_sampler_ = _numeric_preprocessor(self.classifier)
            else:
                self.pre_sampler_ = _mixed_sampler_preprocessor(self.layout)
            transformed = self.pre_sampler_.fit_transform(X, y_array)
            sampler = sampler_for_v2(
                self.condition,
                self.layout,
                y_array,
                random_state=self.random_state,
            )
            if sampler is None:
                resampled, y_resampled = transformed, y_array
            else:
                resampled, y_resampled = sampler.fit_resample(transformed, y_array)
            self.rows_before_ = len(y_array)
            self.rows_after_ = len(y_resampled)
            if self.layout.categorical:
                categorical_transformer = self.pre_sampler_.named_transformers_["categorical"]
                self.ordinal_categories_ = list(
                    categorical_transformer.named_steps["ordinal"].categories_
                )
                model_frame = self._resampled_native_frame(resampled)
            else:
                model_frame = pd.DataFrame(resampled, columns=list(self.layout.numeric))
            fit_weight = fit_time_sample_weight(self.condition, self.classifier, y_resampled)
            y_array = np.asarray(y_resampled)
        elif self.layout.categorical:
            self.rows_before_ = len(y_array)
            self.rows_after_ = len(y_array)
            model_frame, self.native_categories_ = prepare_native_categorical_frame(X, self.layout)
        else:
            self.rows_before_ = len(y_array)
            self.rows_after_ = len(y_array)
            model_frame = self._numeric_frame(X, fit=True)
        self.model_ = _native_classifier(
            self.classifier,
            self.random_state,
            n_classes=len(self.classes_),
            n_estimators=self.n_estimators,
            model_params=self.model_params,
        )
        fit_kwargs: dict[str, Any] = {}
        if fit_weight is not None:
            fit_kwargs["sample_weight"] = fit_weight
        if self.classifier == "catboost" and self.layout.categorical:
            fit_kwargs["cat_features"] = list(range(len(self.layout.numeric), model_frame.shape[1]))
        self.model_.fit(model_frame, y_array, **fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(
            self.model_.predict_proba(self._prepare_for_predict(X)), dtype=float
        )
        totals = probabilities.sum(axis=1, keepdims=True)
        return probabilities / totals

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_pipeline_v2(
    layout: FeatureLayout,
    condition: str,
    classifier: str,
    random_state: int,
    *,
    n_estimators: int = 40,
    model_params: dict[str, Any] | None = None,
) -> Any:
    """Build one frozen V2 model/condition pipeline for a fold-local fit."""

    if classifier in NATIVE_CATEGORICAL_MODELS:
        return NativeCategoricalEstimator(
            classifier,
            layout,
            condition,
            random_state,
            n_estimators=n_estimators,
            model_params=model_params,
        )
    if classifier not in {"logistic_regression", "random_forest"}:
        raise ValueError(f"Unknown V2 classifier: {classifier}")
    if condition == "balanced_random_forest":
        if classifier != "random_forest":
            raise ValueError("Balanced Random Forest is an RF-only comparator")
        estimator: Any = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            sampling_strategy="all",
            replacement=True,
            random_state=random_state,
            n_jobs=1,
        )
    else:
        estimator = _classifier(
            classifier,
            random_state,
            weighted=condition == "class_weighted",
            n_estimators=n_estimators,
        )
        if model_params:
            estimator.set_params(**model_params)
    steps: list[tuple[str, Any]] = []
    if layout.categorical:
        steps.append(("pre_sampler", _mixed_sampler_preprocessor(layout)))
        if condition in SAMPLER_CONDITIONS:
            steps.append(("sampler", AdaptiveSampler(condition, layout, random_state)))
        steps.extend(
            [
                ("post_sampler", _mixed_postprocessor(layout, classifier)),
                ("classifier", estimator),
            ]
        )
    else:
        steps.append(("preprocess", _numeric_preprocessor(classifier)))
        if condition in SAMPLER_CONDITIONS:
            steps.append(("sampler", AdaptiveSampler(condition, layout, random_state)))
        steps.append(("classifier", estimator))
    return ImblearnPipeline(steps)


def declared_path(layout: FeatureLayout, classifier: str, condition: str) -> str:
    """Expose the path used by smoke assertions and manifests."""

    return categorical_path(classifier, layout, condition)
