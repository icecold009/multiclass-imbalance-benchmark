from pathlib import Path

import pandas as pd

from src.stats import write_friedman_summary, write_rankings


def test_rankings_aggregate_folds_and_seeds(tmp_path: Path) -> None:
    rows = []
    for dataset in ["a", "b", "c"]:
        for condition, score in [("raw", 0.4), ("smote", 0.6)]:
            for seed in [0, 1]:
                rows.append(
                    {
                        "dataset": dataset,
                        "feature_type": "numeric",
                        "classifier": "random_forest",
                        "condition": condition,
                        "seed": seed,
                        "fold": 0,
                        "macro_f1": score,
                        "g_mean": score,
                        "mcc": score,
                        "balanced_accuracy": score,
                    }
                )
    results = pd.DataFrame(rows)

    rankings = write_rankings(results, tmp_path)
    summary = write_friedman_summary(results, tmp_path)

    assert set(rankings["condition"]) == {"raw", "smote"}
    assert (rankings.loc[rankings["condition"] == "smote", "rank"] == 1).all()
    assert (summary["status"] == "computed").any()
