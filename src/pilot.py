"""Run the Stage 0 end-to-end pilot on one local CSV dataset.

The pilot is intentionally explicit rather than highly configurable. It
produces long-form fold results and failure records so unsupported data/method
combinations are visible before the full benchmark is designed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTENC,
    BorderlineSMOTE,
    RandomOverSampler,
)
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.stats import analyse_results

try:
    import psutil
except ImportError:  # pragma: no cover - optional measurement dependency
    psutil = None


CONDITIONS = (
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
CLASSIFIERS = ("random_forest", "xgboost", "logistic_regression")


@dataclass(frozen=True)
class FeatureLayout:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def is_mixed(self) -> bool:
        return bool(self.numeric and self.categorical)

    @property
    def is_numeric(self) -> bool:
        return bool(self.numeric) and not self.categorical


def feature_layout(
    frame: pd.DataFrame,
    target: str,
    categorical_columns: tuple[str, ...] = (),
) -> FeatureLayout:
    features = frame.drop(columns=[target])
    missing_columns = set(categorical_columns) - set(features.columns)
    if missing_columns:
        raise ValueError(f"Categorical columns are not present: {sorted(missing_columns)}")
    numeric = tuple(
        column
        for column in features.select_dtypes(include="number").columns
        if column not in categorical_columns
    )
    categorical = tuple(column for column in features.columns if column not in numeric)
    if not numeric:
        raise ValueError("Categorical-only datasets are not supported by this pilot")
    return FeatureLayout(numeric=numeric, categorical=categorical)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2 compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def numeric_preprocessor(layout: FeatureLayout) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                SklearnPipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(layout.numeric),
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def mixed_pre_sampler(layout: FeatureLayout) -> ColumnTransformer:
    categorical = SklearnPipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), list(layout.numeric)),
            ("categorical", categorical, list(layout.categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def mixed_post_sampler(layout: FeatureLayout) -> ColumnTransformer:
    numeric_count = len(layout.numeric)
    categorical_indices = list(range(numeric_count, numeric_count + len(layout.categorical)))
    return ColumnTransformer(
        [
            (
                "numeric",
                SklearnPipeline(
                    [
                        ("scale", StandardScaler()),
                    ]
                ),
                list(range(numeric_count)),
            ),
            ("categorical", _one_hot_encoder(), categorical_indices),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def sampler_for(
    condition: str,
    layout: FeatureLayout,
    random_state: int,
) -> tuple[Any | None, bool]:
    """Return sampler and whether it requires mixed-data handling."""

    if condition in {"raw", "class_weighted", "balanced_random_forest"}:
        return None, False

    if layout.is_mixed:
        if condition == "smotenc":
            categorical_indices = list(
                range(len(layout.numeric), len(layout.numeric) + len(layout.categorical))
            )
            return SMOTENC(
                categorical_features=categorical_indices,
                sampling_strategy="not majority",
                random_state=random_state,
                k_neighbors=3,
            ), True
        if condition in {"smoteenn", "smotetomek"}:
            categorical_indices = list(
                range(len(layout.numeric), len(layout.numeric) + len(layout.categorical))
            )
            mixed_smote = SMOTENC(
                categorical_features=categorical_indices,
                sampling_strategy="not majority",
                random_state=random_state,
                k_neighbors=3,
            )
            sampler = (
                SMOTEENN(smote=mixed_smote)
                if condition == "smoteenn"
                else SMOTETomek(smote=mixed_smote)
            )
            return sampler, True
        if condition in {"random_over", "random_under"}:
            return (
                RandomOverSampler(
                    sampling_strategy="not majority", random_state=random_state
                )
                if condition == "random_over"
                else RandomUnderSampler(
                    sampling_strategy="not minority", random_state=random_state
                ),
                True,
            )
        raise ValueError(f"{condition} is not valid for mixed data in Stage 0")

    samplers = {
        "random_over": RandomOverSampler(
            sampling_strategy="not majority", random_state=random_state
        ),
        "random_under": RandomUnderSampler(
            sampling_strategy="not minority", random_state=random_state
        ),
        "smote": SMOTE(
            sampling_strategy="not majority", random_state=random_state, k_neighbors=3
        ),
        "adasyn": ADASYN(
            sampling_strategy="not majority", random_state=random_state, n_neighbors=3
        ),
        "borderline_smote": BorderlineSMOTE(
            sampling_strategy="not majority",
            random_state=random_state,
            k_neighbors=3,
            kind="borderline-1",
        ),
        "smoteenn": SMOTEENN(
            smote=SMOTE(
                sampling_strategy="not majority", random_state=random_state, k_neighbors=3
            ),
            sampling_strategy="not majority",
        ),
        "smotetomek": SMOTETomek(
            smote=SMOTE(
                sampling_strategy="not majority", random_state=random_state, k_neighbors=3
            ),
            sampling_strategy="not majority",
        ),
    }
    if condition == "smotenc":
        raise ValueError("SMOTENC is only applicable to mixed datasets")
    if condition not in samplers:
        raise ValueError(f"Unknown condition: {condition}")
    return samplers[condition], False


def classifier_for(name: str, random_state: int, weighted: bool = False) -> Any:
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=1,
            class_weight="balanced" if weighted else None,
        )
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced" if weighted else None,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("xgboost is not installed") from error
        return XGBClassifier(
            n_estimators=200,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown classifier: {name}")


def fit_kwargs_for(
    condition: str,
    classifier_name: str,
    y_train: pd.Series,
) -> dict[str, Any]:
    """Return classifier fit parameters that are derived from training labels."""

    if condition == "class_weighted" and classifier_name == "xgboost":
        return {
            "classifier__sample_weight": compute_sample_weight(
                class_weight="balanced", y=y_train
            )
        }
    return {}


def build_pipeline(
    layout: FeatureLayout,
    condition: str,
    classifier_name: str,
    random_state: int,
) -> ImblearnPipeline:
    weighted = condition == "class_weighted"
    classifier = classifier_for(classifier_name, random_state, weighted=weighted)
    if condition == "balanced_random_forest":
        if classifier_name != "random_forest":
            raise ValueError("Balanced Random Forest is an RF-only comparator")
        classifier = BalancedRandomForestClassifier(
            n_estimators=200,
            sampling_strategy="all",
            replacement=True,
            random_state=random_state,
            n_jobs=1,
        )

    sampler, _ = sampler_for(condition, layout, random_state)
    if layout.is_mixed:
        steps: list[tuple[str, Any]] = [("pre_sampler", mixed_pre_sampler(layout))]
        if sampler is not None:
            steps.append(("sampler", sampler))
        steps.extend(
            [("post_sampler", mixed_post_sampler(layout)), ("classifier", classifier)]
        )
    else:
        steps = [("preprocess", numeric_preprocessor(layout))]
        if sampler is not None:
            steps.append(("sampler", sampler))
        steps.append(("classifier", classifier))
    return ImblearnPipeline(steps)


def geometric_mean(y_true: pd.Series, y_pred: np.ndarray, labels: list[Any]) -> float:
    recalls = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return float(np.prod(recalls) ** (1 / len(recalls))) if len(recalls) else 0.0


def sampled_row_count(
    pipeline: ImblearnPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> int:
    """Reproduce the fitted sampler output size for pilot accounting.

    Samplers do not expose a common transformed-row-count API. Re-fitting a
    cloned sampler with the same seed keeps this accounting separate from the
    fitted classifier and avoids reporting the pre-sampling count as evidence.
    """

    sampler = pipeline.named_steps.get("sampler")
    if sampler is None:
        return len(y_train)
    preprocessor_name = (
        "pre_sampler" if "pre_sampler" in pipeline.named_steps else "preprocess"
    )
    transformed = pipeline.named_steps[preprocessor_name].transform(X_train)
    _, y_resampled = clone(sampler).fit_resample(transformed, y_train)
    return len(y_resampled)


def run_pilot(
    csv_path: Path,
    target: str,
    output_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    folds: int = 5,
    categorical_columns: tuple[str, ...] = (),
) -> None:
    frame = pd.read_csv(csv_path)
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not present")
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    layout = feature_layout(frame, target, categorical_columns=categorical_columns)
    X = frame.drop(columns=[target])
    target_encoder = LabelEncoder()
    y = pd.Series(target_encoder.fit_transform(frame[target]), name=target)
    labels = sorted(y.unique().tolist(), key=str)
    if len(labels) < 3:
        raise ValueError("Stage 0 requires at least three target classes")
    if y.value_counts().min() < folds:
        raise ValueError("Every class must have at least one example per fold")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pilot_metadata.json").write_text(
        json.dumps(
            {
                "target_column": target,
                "target_label_classes": [str(value) for value in target_encoder.classes_],
                "categorical_columns": list(categorical_columns),
                "folds": folds,
                "seeds": list(seeds),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for seed in seeds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            for classifier_name in CLASSIFIERS:
                for condition in CONDITIONS:
                    started = time.perf_counter()
                    rss_before = (
                        psutil.Process().memory_info().rss / (1024**2)
                        if psutil is not None
                        else None
                    )
                    record = {
                        "dataset": csv_path.stem,
                        "seed": seed,
                        "fold": fold,
                        "classifier": classifier_name,
                        "condition": condition,
                        "feature_type": "mixed" if layout.is_mixed else "numeric",
                    }
                    try:
                        pipeline = build_pipeline(
                            layout, condition, classifier_name, random_state=seed
                        )
                        fit_kwargs = fit_kwargs_for(condition, classifier_name, y_train)
                        pipeline.fit(X_train, y_train, **fit_kwargs)
                        predicted = pipeline.predict(X_test)
                        elapsed = time.perf_counter() - started
                        rss_after = (
                            psutil.Process().memory_info().rss / (1024**2)
                            if psutil is not None
                            else None
                        )
                        record.update(
                            {
                                "macro_f1": f1_score(
                                    y_test, predicted, average="macro", zero_division=0
                                ),
                                "g_mean": geometric_mean(y_test, predicted, labels),
                                "mcc": matthews_corrcoef(y_test, predicted),
                                "balanced_accuracy": balanced_accuracy_score(
                                    y_test, predicted
                                ),
                                "fit_predict_seconds": elapsed,
                                "rss_before_mb": rss_before,
                                "rss_after_mb": rss_after,
                                "rss_delta_mb": (
                                    rss_after - rss_before
                                    if rss_before is not None and rss_after is not None
                                    else None
                                ),
                                "train_rows_before_sampling": len(y_train),
                                "train_rows_after_sampling": sampled_row_count(
                                    pipeline, X_train, y_train
                                ),
                            }
                        )
                        results.append(record)
                    except Exception as error:  # noqa: BLE001 - failures are pilot evidence
                        failures.append(
                            {
                                **record,
                                "stage": "fit_predict_or_accounting",
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        )

    pd.DataFrame(results).to_csv(output_dir / "pilot_results.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "pilot_failures.csv", index=False)
    if results:
        analyse_results(output_dir / "pilot_results.csv", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 0 pilot")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage0"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--categorical-columns",
        nargs="+",
        default=[],
        help="Semantic categorical columns that are numerically encoded in the CSV",
    )
    args = parser.parse_args()
    run_pilot(
        args.csv_path,
        args.target,
        args.output_dir,
        seeds=tuple(args.seeds),
        folds=args.folds,
        categorical_columns=tuple(args.categorical_columns),
    )


if __name__ == "__main__":
    main()
