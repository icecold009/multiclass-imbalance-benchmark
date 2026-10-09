from pathlib import Path

import pandas as pd

from src.v2_analysis import (
    analysis_dry_run,
    declared_contrasts,
    load_analysis_context,
    paired_contrast_blocks,
    rope_category,
)
from src.v2_engineering import STATUS_VALID

ROOT = Path(__file__).resolve().parents[1]


def _results() -> pd.DataFrame:
    rows = []
    for dataset_id in ("d1", "d2", "d3"):
        for fold in range(5):
            for condition, value in (("raw", 0.4), ("smote", 0.42)):
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "outer_fold": fold,
                        "outer_seed": 20260913,
                        "classifier": "logistic_regression",
                        "condition": condition,
                        "feature_type": "numeric",
                        "macro_average_precision": value + fold * 0.001,
                        "status": STATUS_VALID,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_contrast_blocks_require_complete_five_fold_datasets() -> None:
    blocks = paired_contrast_blocks(_results(), "smote", "raw", "logistic_regression")
    assert len(blocks) == 15
    assert blocks["delta"].min() > 0


def test_rope_category_is_not_a_significance_label() -> None:
    assert rope_category(0.2, 0.7, 0.1) == "practically_equivalent"


def test_declared_contrasts_are_score_independent() -> None:
    context = load_analysis_context(ROOT)
    contrasts = declared_contrasts(context.config)
    assert ("smote", "raw") in contrasts
    assert ("balanced_random_forest", "raw") in contrasts


def test_analysis_dry_run_is_blocked_without_complete_full_run() -> None:
    context = load_analysis_context(ROOT)
    summary = analysis_dry_run(context)
    assert summary["status"] == "BLOCKED"
    assert summary["analysis_started"] is False
    assert summary["confirmatory_p_values"] is False
