"""Model-free audit and split preflight for exact predictor profiles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

GROUPED_SENSITIVITY_DATASETS = (
    "cmc",
    "cnae_9",
    "glass",
    "iris",
    "wine_quality_red",
    "yeast",
)


def predictor_profile_codes(features: pd.DataFrame) -> np.ndarray:
    """Assign equal codes to rows with identical predictor values only.

    Codes depend on row order; their equality relation does not. Missing values
    are treated as equal only when they occur in the same predictor columns,
    so the missingness pattern is part of the profile.
    """
    if not isinstance(features, pd.DataFrame) or features.shape[0] == 0 or features.shape[1] == 0:
        raise TypeError("features must be a non-empty pandas DataFrame")
    if features.columns.has_duplicates:
        raise ValueError("feature columns must be unique")
    codes = (
        features.groupby(
            list(features.columns), sort=False, dropna=False, observed=True
        )
        .ngroup()
        .to_numpy()
    )
    if (codes < 0).any():
        raise RuntimeError("profile factorization produced an unassigned row")
    return codes.astype(np.int64, copy=False)


def audit_dataset_profiles(
    dataset_id: str,
    frame: pd.DataFrame,
    target_column: str,
    seeds: tuple[int, ...] = (0, 1, 2),
    folds: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return profile counts, original split overlap, and grouped-CV preflight.

    No classifier or sampler is invoked. Group labels use predictor columns;
    target labels are used only to stratify splits and count conflicting
    profiles.
    """
    if target_column not in frame.columns:
        raise ValueError(f"{dataset_id}: target column {target_column!r} is missing")
    if frame.empty or frame[target_column].isna().any():
        raise ValueError(f"{dataset_id}: audit input must be non-empty with complete targets")
    if folds < 2 or not seeds:
        raise ValueError("at least two folds and one seed are required")

    features = frame.drop(columns=[target_column]).reset_index(drop=True)
    labels = frame[target_column].reset_index(drop=True)
    if labels.nunique() < 2:
        raise ValueError(f"{dataset_id}: at least two target classes are required")
    groups = predictor_profile_codes(features)
    profile_counts = pd.Series(groups).value_counts()
    repeated_ids = set(profile_counts[profile_counts > 1].index.astype(int))
    profile_labels = pd.DataFrame({"profile": groups, "label": labels.astype(str)})
    label_counts_by_profile = profile_labels.groupby("profile", sort=False)["label"].nunique()
    conflicting_ids = set(label_counts_by_profile[label_counts_by_profile > 1].index.astype(int))
    class_group_counts = (
        profile_labels.groupby("label", sort=True)["profile"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    repeated_rows = int(profile_counts[profile_counts > 1].sum())
    conflicting_rows = int(profile_labels["profile"].isin(conflicting_ids).sum())

    rowwise_splits: list[dict[str, Any]] = []
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(features, labels)):
            seen = np.isin(groups[test_idx], groups[train_idx])
            rowwise_splits.append(
                {
                    "dataset": dataset_id,
                    "seed": int(seed),
                    "fold": int(fold),
                    "train_rows": len(train_idx),
                    "test_rows": len(test_idx),
                    "test_rows_with_training_profile": int(seen.sum()),
                    "overlapping_profile_groups": int(
                        np.intersect1d(groups[train_idx], groups[test_idx]).size
                    ),
                }
            )

    grouped_splits: list[dict[str, Any]] = []
    if dataset_id in GROUPED_SENSITIVITY_DATASETS:
        expected_classes = set(labels.astype(str))
        for seed in seeds:
            splitter = StratifiedGroupKFold(
                n_splits=folds, shuffle=True, random_state=seed
            )
            for fold, (train_idx, test_idx) in enumerate(
                splitter.split(features, labels, groups)
            ):
                train_groups = set(groups[train_idx].tolist())
                test_groups = set(groups[test_idx].tolist())
                train_labels = set(labels.iloc[train_idx].astype(str))
                test_labels = set(labels.iloc[test_idx].astype(str))
                train_counts = labels.iloc[train_idx].astype(str).value_counts()
                test_counts = labels.iloc[test_idx].astype(str).value_counts()
                grouped_splits.append(
                    {
                        "dataset": dataset_id,
                        "seed": int(seed),
                        "fold": int(fold),
                        "train_rows": len(train_idx),
                        "test_rows": len(test_idx),
                        "train_min_class_rows": int(
                            train_counts.reindex(sorted(expected_classes), fill_value=0).min()
                        ),
                        "test_min_class_rows": int(
                            test_counts.reindex(sorted(expected_classes), fill_value=0).min()
                        ),
                        "missing_train_classes": ";".join(
                            sorted(expected_classes - train_labels)
                        ),
                        "missing_test_classes": ";".join(
                            sorted(expected_classes - test_labels)
                        ),
                        "overlapping_profile_groups": len(
                            train_groups.intersection(test_groups)
                        ),
                    }
                )

    total_test_rows = len(frame) * len(seeds)
    overlapping_test_rows = sum(
        row["test_rows_with_training_profile"] for row in rowwise_splits
    )
    summary = {
        "dataset": dataset_id,
        "rows": len(frame),
        "classes": int(labels.nunique()),
        "missing_predictor_values": int(features.isna().to_numpy().sum()),
        "missing_predictor_columns": {
            str(column): int(count)
            for column, count in features.isna().sum().items()
            if count
        },
        "unique_predictor_profiles": len(profile_counts),
        "exact_duplicate_excess_rows": int(frame.duplicated().sum()),
        "repeated_profile_groups": len(repeated_ids),
        "repeated_profile_excess_rows": int(len(frame) - len(profile_counts)),
        "rows_in_repeated_profile_groups": repeated_rows,
        "conflicting_profile_groups": len(conflicting_ids),
        "rows_in_conflicting_profile_groups": conflicting_rows,
        "minimum_profile_groups_per_class": int(min(class_group_counts.values())),
        "profile_groups_per_class": class_group_counts,
        "rowwise_cv_test_rows_with_training_profile": int(overlapping_test_rows),
        "rowwise_cv_test_row_occurrences": int(total_test_rows),
        "rowwise_cv_overlap_percent": (
            100.0 * overlapping_test_rows / total_test_rows if total_test_rows else 0.0
        ),
        "grouped_cv_preflight_folds": len(grouped_splits),
        "grouped_cv_zero_profile_overlap": (
            all(row["overlapping_profile_groups"] == 0 for row in grouped_splits)
            if grouped_splits
            else None
        ),
        "grouped_cv_all_classes_in_train_test": (
            all(
                not row["missing_train_classes"] and not row["missing_test_classes"]
                for row in grouped_splits
            )
            if grouped_splits
            else None
        ),
    }
    return summary, rowwise_splits, grouped_splits
