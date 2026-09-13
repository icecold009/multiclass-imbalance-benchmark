"""Engineering primitives for the frozen V2 benchmark protocol.

This module deliberately contains only fold-local, deterministic building
blocks.  Dataset selection and full benchmark execution are implemented in
later gates; these helpers make the safety-critical rules executable and
testable before any outcome-bearing run is allowed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.optimize import differential_evolution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

V2_MODELS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
)
V2_CONDITIONS = (
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
NATIVE_CATEGORICAL_MODELS = ("xgboost", "lightgbm", "catboost")
WEIGHTED_FIT_MODELS = NATIVE_CATEGORICAL_MODELS
SAMPLER_CONDITIONS = frozenset(V2_CONDITIONS) - {
    "raw",
    "class_weighted",
    "balanced_random_forest",
}
STATUS_VALID = "VALID"
STATUS_FAILED = "FAILED / NOT APPLICABLE"


class NotApplicableError(ValueError):
    """A protocol-defined cell failure, not an implementation crash."""


@dataclass(frozen=True)
class FeatureLayout:
    """Semantic feature columns for a single dataset."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def feature_type(self) -> str:
        if self.numeric and self.categorical:
            return "mixed"
        if self.numeric:
            return "numeric"
        return "categorical"


@dataclass(frozen=True)
class NeighbourSpec:
    """Fold-local sampler neighbour parameters and their support basis."""

    k_neighbors: int
    m_neighbors: int | None
    minority_support: int


@dataclass(frozen=True)
class ThresholdFit:
    """Calibration result with explicit fallback provenance."""

    offsets: tuple[float, ...]
    status: str
    reason: str | None
    objective: float


def infer_feature_layout(
    frame: pd.DataFrame,
    target: str,
    categorical_columns: Iterable[str] = (),
) -> FeatureLayout:
    """Infer numeric/mixed layout without inspecting the target as a feature."""

    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not present")
    features = frame.drop(columns=[target])
    declared = tuple(categorical_columns)
    missing = set(declared) - set(features.columns)
    if missing:
        raise ValueError(f"Categorical columns are not present: {sorted(missing)}")
    numeric = tuple(
        column
        for column in features.select_dtypes(include="number").columns
        if column not in declared
    )
    categorical = tuple(column for column in features.columns if column not in numeric)
    if not numeric:
        raise NotApplicableError("categorical-only datasets are outside the V2 core path")
    return FeatureLayout(numeric=numeric, categorical=categorical)


def fold_local_neighbours(y: Iterable[Any], condition: str) -> NeighbourSpec:
    """Apply the locked dynamic-neighbour rule to one actual training split.

    ``not majority`` samplers operate on each non-majority class.  The
    smallest class they may synthesize from determines the safe neighbour
    count, so the rule is conservative across all targeted classes.
    """

    if condition not in SAMPLER_CONDITIONS:
        raise ValueError(f"{condition!r} does not use a fold-local sampler")
    labels = list(y)
    counts = pd.Series(labels, dtype="object").value_counts()
    if counts.empty:
        raise NotApplicableError("no training labels are available")
    majority = counts.max()
    targeted = counts[counts < majority]
    if targeted.empty:
        raise NotApplicableError("no non-majority class is available for synthesis")
    minority_support = int(targeted.min())
    if minority_support < 2:
        raise NotApplicableError(
            "fewer than two training examples in the smallest targeted class"
        )
    k_neighbors = min(5, minority_support - 1)
    m_neighbors: int | None = None
    if condition == "borderline_smote":
        # BorderlineSMOTE's danger-neighbourhood is bounded by the available
        # fold rows and remains at least k+1 for a valid neighbourhood.
        m_neighbors = min(10, max(k_neighbors + 1, len(counts) + 1))
        m_neighbors = min(m_neighbors, len(labels) - 1)
        if m_neighbors < 2:
            raise NotApplicableError("insufficient fold rows for borderline m-neighbours")
    return NeighbourSpec(k_neighbors, m_neighbors, minority_support)


def _categorical_indices(layout: FeatureLayout) -> list[int]:
    start = len(layout.numeric)
    return list(range(start, start + len(layout.categorical)))


def sampler_for_v2(
    condition: str,
    layout: FeatureLayout,
    y_train: Iterable[Any],
    random_state: int,
) -> Any | None:
    """Build the frozen sampler from labels in the current fit split only."""

    if condition in {"raw", "class_weighted", "balanced_random_forest"}:
        return None
    if condition not in SAMPLER_CONDITIONS:
        raise ValueError(f"Unknown V2 condition: {condition}")
    spec = fold_local_neighbours(y_train, condition)
    if condition == "random_over":
        return RandomOverSampler(sampling_strategy="not majority", random_state=random_state)
    if condition == "random_under":
        return RandomUnderSampler(sampling_strategy="not minority", random_state=random_state)

    if condition == "smotenc" and not layout.categorical:
        raise NotApplicableError("SMOTENC requires a mixed feature layout")
    if condition in {"smote", "adasyn", "borderline_smote"} and layout.categorical:
        raise NotApplicableError(f"{condition} is numeric-only; use SMOTENC for mixed data")

    if layout.categorical:
        base: Any = SMOTENC(
            categorical_features=_categorical_indices(layout),
            sampling_strategy="not majority",
            random_state=random_state,
            k_neighbors=spec.k_neighbors,
        )
    elif condition == "smote":
        base = SMOTE(
            sampling_strategy="not majority",
            random_state=random_state,
            k_neighbors=spec.k_neighbors,
        )
    elif condition == "adasyn":
        base = ADASYN(
            sampling_strategy="not majority",
            random_state=random_state,
            n_neighbors=spec.k_neighbors,
        )
    elif condition == "borderline_smote":
        base = BorderlineSMOTE(
            sampling_strategy="not majority",
            random_state=random_state,
            k_neighbors=spec.k_neighbors,
            m_neighbors=spec.m_neighbors,
            kind="borderline-1",
        )
    else:
        base = SMOTE(
            sampling_strategy="not majority",
            random_state=random_state,
            k_neighbors=spec.k_neighbors,
        )

    if condition == "smoteenn":
        return SMOTEENN(smote=base, sampling_strategy="not majority")
    if condition == "smotetomek":
        return SMOTETomek(smote=base, sampling_strategy="not majority")
    return base


def fit_time_sample_weight(condition: str, model: str, y_fit: Iterable[Any]) -> np.ndarray | None:
    """Return weights computed from the labels in the current model fit."""

    if condition != "class_weighted" or model not in WEIGHTED_FIT_MODELS:
        return None
    return compute_sample_weight(class_weight="balanced", y=np.asarray(list(y_fit)))


def categorical_path(model: str, layout: FeatureLayout, condition: str) -> str:
    """Declare the non-leaking representation used by a model/condition cell."""

    if model not in V2_MODELS:
        raise ValueError(f"Unknown V2 model: {model}")
    if not layout.categorical:
        return "numeric"
    if model == "logistic_regression":
        return "one_hot_after_fold_local_sampling" if condition in SAMPLER_CONDITIONS else "one_hot"
    if model == "random_forest":
        return "ordinal_encoded_tree_features"
    if model in NATIVE_CATEGORICAL_MODELS and condition in {"raw", "class_weighted"}:
        return "native_categorical"
    if model in NATIVE_CATEGORICAL_MODELS and condition in {"smotenc", "smoteenn", "smotetomek"}:
        return "fold_local_ordinal_sampler_then_native_recast"
    raise NotApplicableError(
        f"{model} has no declared categorical path for condition {condition}"
    )


def _one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility with old sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocessing_for(
    model: str,
    layout: FeatureLayout,
    condition: str,
) -> ColumnTransformer | Pipeline:
    """Return a fold-fittable transformer for the declared model path."""

    path = categorical_path(model, layout, condition)
    if not layout.categorical:
        steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
        if model == "logistic_regression":
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)
    if path in {"native_categorical", "fold_local_ordinal_sampler_then_native_recast"}:
        # Native models consume a DataFrame with categorical dtypes.  The
        # caller must apply fold-local missing-value handling and recast after
        # any sampler; no one-hot or global encoder is permitted here.
        raise NotApplicableError("native categorical paths are DataFrame adapters, not transformers")
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if model == "logistic_regression":
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(layout.numeric)),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", _one_hot()),
                    ]
                ),
                list(layout.categorical),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_native_categorical_frame(
    frame: pd.DataFrame,
    layout: FeatureLayout,
    fitted_categories: dict[str, list[Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[Any]]]:
    """Impute and cast a mixed frame for native categorical GBDTs.

    The category vocabulary is learned from the fit frame and can be reused
    for validation/test frames, so unseen categories become an explicit
    missing category rather than leaking their labels into fitting.
    """

    result = frame.loc[:, [*layout.numeric, *layout.categorical]].copy()
    for column in layout.numeric:
        median = result[column].median()
        result[column] = result[column].fillna(median)
    category_map = fitted_categories or {}
    for column in layout.categorical:
        if column not in category_map:
            values = result[column].dropna().drop_duplicates().tolist()
            category_map[column] = values
        dtype = pd.api.types.CategoricalDtype(categories=category_map[column])
        result[column] = result[column].astype(dtype)
        result[column] = result[column].cat.add_categories(["__MISSING__"]).fillna("__MISSING__")
    return result, category_map


def splitters() -> tuple[StratifiedKFold, StratifiedKFold]:
    """Return the frozen outer and inner splitters with independent seeds."""

    return (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=20260913),
        StratifiedKFold(n_splits=3, shuffle=True, random_state=20260914),
    )


def assert_nested_split_independence(
    outer_train: Iterable[int], outer_test: Iterable[int], inner_train: Iterable[int], inner_test: Iterable[int]
) -> None:
    """Check that inner indices are relative to the outer training partition."""

    outer_train_set = set(outer_train)
    outer_test_set = set(outer_test)
    inner_train_set = set(inner_train)
    inner_test_set = set(inner_test)
    if inner_train_set & inner_test_set:
        raise AssertionError("inner train and validation indices overlap")
    if not inner_train_set | inner_test_set <= outer_train_set:
        raise AssertionError("inner split contains an outer test index")
    if outer_train_set & outer_test_set:
        raise AssertionError("outer train and test indices overlap")


def _validate_probabilities(probabilities: np.ndarray, class_count: int) -> np.ndarray:
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != class_count:
        raise ValueError("probabilities have the wrong shape")
    if not np.isfinite(proba).all() or (proba < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    totals = proba.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-8):
        raise ValueError("probabilities must sum to one")
    return proba


def apply_threshold_offsets(probabilities: np.ndarray, offsets: Iterable[float]) -> np.ndarray:
    """Apply ``argmax(p_c / t_c)`` with class zero as the reference."""

    offset_array = np.asarray(list(offsets), dtype=float)
    proba = _validate_probabilities(probabilities, len(offset_array))
    if not np.isfinite(offset_array).all():
        raise ValueError("threshold offsets must be finite")
    return np.argmax(proba / np.exp(offset_array), axis=1)


def fit_threshold_offsets(
    y_validation: Iterable[Any],
    probabilities: np.ndarray,
    class_labels: Iterable[Any],
    *,
    seed: int = 20260916,
) -> ThresholdFit:
    """Fit thresholds from inner-CV validation predictions only."""

    labels = list(class_labels)
    try:
        proba = _validate_probabilities(probabilities, len(labels))
        y_array = np.asarray(list(y_validation))
        if len(y_array) != len(proba) or len(np.unique(y_array)) != len(labels):
            raise ValueError("validation labels do not provide every class")
        bounds = [(-2.0, 2.0)] * (len(labels) - 1)
        if not bounds:
            return ThresholdFit((0.0,), STATUS_VALID, None, 1.0)

        def objective(free_offsets: np.ndarray) -> float:
            offsets = np.concatenate(([0.0], np.asarray(free_offsets, dtype=float)))
            predicted = apply_threshold_offsets(proba, offsets)
            score = f1_score(y_array, predicted, labels=list(range(len(labels))), average="macro", zero_division=0)
            # The tiny deterministic term implements the frozen tie rule
            # without changing any practically distinct macro-F1 score.
            tie_term = 1e-12 * float(np.dot(np.asarray(free_offsets), np.arange(1, len(labels))))
            return -float(score) + tie_term

        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=seed,
            maxiter=40,
            popsize=8,
            polish=False,
            workers=1,
            updating="immediate",
        )
        offsets = tuple(float(value) for value in np.concatenate(([0.0], result.x)))
        return ThresholdFit(offsets, STATUS_VALID, None, -float(result.fun))
    except (ValueError, FloatingPointError, RuntimeError) as error:
        return ThresholdFit(
            tuple(0.0 for _ in labels),
            STATUS_FAILED,
            f"threshold fallback: {type(error).__name__}: {error}",
            float("nan"),
        )


def macro_average_precision(
    y_true: Iterable[Any], probabilities: np.ndarray, class_labels: Iterable[Any]
) -> tuple[float, list[float]]:
    """Compute mean one-vs-rest average precision for declared classes."""

    labels = list(class_labels)
    proba = _validate_probabilities(probabilities, len(labels))
    y_array = np.asarray(list(y_true))
    per_class: list[float] = []
    for index, label in enumerate(labels):
        binary = (y_array == label).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            raise NotApplicableError(f"class {label!r} lacks both positive and negative examples")
        per_class.append(float(average_precision_score(binary, proba[:, index])))
    return float(np.mean(per_class)), per_class


def multiclass_brier_score(y_true: Iterable[Any], probabilities: np.ndarray, class_labels: Iterable[Any]) -> float:
    """Return the frozen mean-over-rows-and-classes multiclass Brier score."""

    labels = list(class_labels)
    proba = _validate_probabilities(probabilities, len(labels))
    y_array = np.asarray(list(y_true))
    one_hot = np.equal(y_array[:, None], np.asarray(labels)[None, :]).astype(float)
    return float(np.mean((proba - one_hot) ** 2))


def evaluate_probabilities(
    y_true: Iterable[Any],
    probabilities: np.ndarray,
    class_labels: Iterable[Any],
    threshold_offsets: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Compute all frozen cell metrics without altering probability metrics."""

    labels = list(class_labels)
    y_array = np.asarray(list(y_true))
    proba = _validate_probabilities(probabilities, len(labels))
    argmax = np.argmax(proba, axis=1)
    calibrated = (
        apply_threshold_offsets(proba, threshold_offsets)
        if threshold_offsets is not None
        else argmax
    )
    macro_ap, per_class_ap = macro_average_precision(y_array, proba, labels)
    per_class_recall = recall_score(
        y_array, argmax, labels=list(range(len(labels))), average=None, zero_division=0
    ).tolist()
    return {
        "macro_average_precision": macro_ap,
        "per_class_average_precision": json.dumps(per_class_ap),
        "macro_f1_argmax": float(f1_score(y_array, argmax, average="macro", zero_division=0)),
        "macro_f1_calibrated": float(f1_score(y_array, calibrated, average="macro", zero_division=0)),
        "g_mean_argmax": float(np.prod(per_class_recall) ** (1 / len(per_class_recall))),
        "mcc_argmax": float(matthews_corrcoef(y_array, argmax)),
        "balanced_accuracy_argmax": float(balanced_accuracy_score(y_array, argmax)),
        "per_class_recall_argmax": json.dumps(per_class_recall),
        "multiclass_brier": multiclass_brier_score(y_array, proba, labels),
        "multiclass_log_loss": float(log_loss(y_array, proba, labels=list(range(len(labels))))),
    }


def validate_cell_accounting(
    expected_keys: Iterable[tuple[Any, ...]],
    valid_keys: Iterable[tuple[Any, ...]],
    failure_keys: Iterable[tuple[Any, ...]],
) -> dict[str, int]:
    """Require exactly one valid or explicit failure record per expected cell."""

    expected = set(expected_keys)
    valid = list(valid_keys)
    failures = list(failure_keys)
    valid_set = set(valid)
    failure_set = set(failures)
    if len(valid) != len(valid_set) or len(failures) != len(failure_set):
        raise ValueError("duplicate cell keys are not permitted")
    if valid_set & failure_set:
        raise ValueError("a cell cannot be both valid and failed")
    observed = valid_set | failure_set
    if observed != expected:
        raise ValueError(
            f"cell accounting mismatch: missing={sorted(expected - observed, key=str)}, "
            f"extra={sorted(observed - expected, key=str)}"
        )
    return {
        "expected_cells": len(expected),
        "valid_cells": len(valid_set),
        "failure_cells": len(failure_set),
    }


def failure_record(
    *,
    dataset_id: str,
    outer_fold: int,
    outer_seed: int,
    classifier: str,
    condition: str,
    stage: str,
    exception_type: str,
    reason: str,
    applicability: str,
    wall_seconds: float,
    resource_status: str,
    retry_count: int,
    config_sha256: str,
) -> dict[str, Any]:
    """Create a schema-complete explicit failure record."""

    return {
        "dataset_id": dataset_id,
        "outer_fold": outer_fold,
        "outer_seed": outer_seed,
        "classifier": classifier,
        "condition": condition,
        "stage": stage,
        "exception_type": exception_type,
        "reason": reason,
        "applicability": applicability,
        "wall_seconds": wall_seconds,
        "resource_status": resource_status,
        "retry_count": retry_count,
        "config_sha256": config_sha256,
        "status": STATUS_FAILED,
    }
