"""Guarded Stage 0 aggregation and rank summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

METRICS = ("macro_f1", "g_mean", "mcc", "balanced_accuracy")


def write_rankings(results: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Write per-dataset/classifier ranks without treating folds as blocks."""

    grouped = (
        results.groupby(["dataset", "feature_type", "classifier", "condition"], as_index=False)[
            list(METRICS)
        ]
        .mean()
    )
    rows: list[pd.DataFrame] = []
    for metric in METRICS:
        ranked = grouped[
            ["dataset", "feature_type", "classifier", "condition", metric]
        ].copy()
        ranked["metric"] = metric
        ranked["score"] = ranked[metric]
        ranked["rank"] = ranked.groupby(
            ["dataset", "feature_type", "classifier"]
        )[metric].rank(ascending=False, method="average")
        rows.append(
            ranked[
                [
                    "dataset",
                    "feature_type",
                    "classifier",
                    "condition",
                    "metric",
                    "score",
                    "rank",
                ]
            ]
        )
    rankings = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    rankings.to_csv(output_dir / "pilot_rankings.csv", index=False)
    return rankings


def write_friedman_summary(results: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Run Friedman only on complete, multi-dataset blocks.

    This is a feasibility diagnostic. It does not replace the preregistered
    final analysis, and it deliberately reports why a comparison was skipped.
    """

    records: list[dict[str, object]] = []
    for feature_type in results["feature_type"].dropna().unique():
        for classifier in results["classifier"].dropna().unique():
            subset = results[
                (results["feature_type"] == feature_type)
                & (results["classifier"] == classifier)
            ]
            for metric in METRICS:
                means = subset.groupby(["dataset", "condition"])[metric].mean().unstack()
                reason = None
                statistic = None
                p_value = None
                if len(means) < 3:
                    reason = "fewer than three dataset blocks"
                elif means.shape[1] < 3:
                    reason = "fewer than three conditions for Friedman test"
                elif means.isna().any().any():
                    reason = "incomplete dataset-by-condition block"
                else:
                    try:
                        from scipy.stats import friedmanchisquare

                        statistic, p_value = friedmanchisquare(
                            *[means[column].to_numpy() for column in means.columns]
                        )
                    except ImportError:
                        reason = "scipy is not installed"
                records.append(
                    {
                        "feature_type": feature_type,
                        "classifier": classifier,
                        "metric": metric,
                        "n_blocks": len(means),
                        "n_conditions": means.shape[1],
                        "friedman_statistic": statistic,
                        "friedman_p_value": p_value,
                        "status": "computed" if reason is None else "skipped",
                        "reason": reason,
                    }
                )
    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "pilot_friedman_summary.csv", index=False)
    return summary


def analyse_results(results_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(results_path)
    if results.empty:
        raise ValueError("Cannot analyse an empty pilot result file")
    write_rankings(results, output_dir)
    write_friedman_summary(results, output_dir)
