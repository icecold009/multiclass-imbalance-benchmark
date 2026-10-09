import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    _analysis_runtime,
    _comparison_summary,
    _failure_accounting,
    _family_classifiers,
    _holm_adjust,
    _macro_f1_figure_coverage,
    _run_confirmatory_tests,
    load_cell_means,
    load_efficiency_means,
    paired_bootstrap_ci,
    paired_rank_biserial,
)
from src.sensitivity import (
    _friedman_rank_statistic,
    exact_sign_flip_sensitivity,
    friedman_permutation_sensitivity,
)


def test_critical_difference_diagram_is_stable_across_hash_seeds(tmp_path) -> None:
    script = textwrap.dedent(
        """\
        import hashlib
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import scikit_posthocs as sp

        from src.analysis import _draw_critical_difference_diagram

        labels = ["raw", "random_over", "smote", "smoteenn", "smotetomek"]
        ranks = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=labels)
        p_values = pd.DataFrame(0.001, index=labels, columns=labels)
        for label in labels:
            p_values.loc[label, label] = 1.0
        for left, right in (
            ("raw", "random_over"),
            ("raw", "smote"),
            ("random_over", "smoteenn"),
            ("smote", "smotetomek"),
        ):
            p_values.loc[left, right] = p_values.loc[right, left] = 0.5

        figure, axis = plt.subplots(figsize=(8, 3))
        _draw_critical_difference_diagram(sp, ranks, p_values, axis)
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=160)
        print(hashlib.sha256(buffer.getvalue()).hexdigest())
        plt.close(figure)
        """
    )
    digests = []
    for hash_seed in ("1", "8675309"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        environment["MPLCONFIGDIR"] = str(tmp_path / f"matplotlib-{hash_seed}")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        digests.append(completed.stdout.strip())

    assert digests[0] == digests[1]


def test_analysis_runtime_records_plot_and_statistics_dependencies() -> None:
    runtime = _analysis_runtime()

    assert runtime["python"]
    assert {
        "numpy",
        "pandas",
        "scipy",
        "scikit-posthocs",
        "statsmodels",
        "matplotlib",
    } <= set(runtime["packages"])
    assert all(runtime["packages"].values())


def _valid_records(condition: str, scalar: float) -> pd.DataFrame:
    rows = []
    for seed in (0, 1, 2):
        for fold in range(5):
            rows.append(
                {
                    "dataset": "fixture",
                    "seed": seed,
                    "fold": fold,
                    "classifier": "random_forest",
                    "condition": condition,
                    "feature_type": "numeric",
                    "macro_f1": scalar,
                    "g_mean": scalar,
                    "mcc": scalar,
                    "balanced_accuracy": scalar,
                    "per_class_recall": json.dumps([scalar, scalar / 2, 1.0]),
                    "fit_predict_seconds": 1.0,
                    "rss_before_mb": 100.0,
                    "rss_after_mb": 102.0,
                    "rss_delta_mb": 2.0,
                    "train_rows_before_sampling": 10,
                    "train_rows_after_sampling": 12,
                }
            )
    return pd.DataFrame(rows)


def test_load_cell_means_requires_all_fifteen_seed_fold_records() -> None:
    means, class_means = load_cell_means(_valid_records("raw", 0.6))

    assert len(means) == 1
    assert means.iloc[0]["macro_f1"] == pytest.approx(0.6)
    assert len(class_means) == 3
    assert class_means.loc[class_means["class_index"] == 1, "recall"].iloc[0] == pytest.approx(0.3)
    efficiency = load_efficiency_means(_valid_records("raw", 0.6))
    assert len(efficiency) == 1
    assert efficiency.iloc[0]["train_rows_after_sampling_mean"] == pytest.approx(12.0)
    incomplete = _valid_records("raw", 0.6).iloc[:-1]
    incomplete_means, _ = load_cell_means(incomplete)
    assert incomplete_means.empty


def test_rank_biserial_and_holm_adjustment_are_paired() -> None:
    assert paired_rank_biserial(np.array([1.0, -1.0, 0.0])) == pytest.approx(0.0)
    assert _holm_adjust([0.01, 0.04, 0.2]) == pytest.approx([0.03, 0.08, 0.2])


def test_bootstrap_is_deterministic_for_fixed_seed() -> None:
    differences = np.array([0.1, 0.2, -0.1, 0.0])
    first = paired_bootstrap_ci(differences, n_resamples=1000, seed=0)
    second = paired_bootstrap_ci(differences, n_resamples=1000, seed=0)

    assert first == second
    assert first[0] <= differences.mean() <= first[1]


def test_two_block_baseline_comparison_is_descriptive_without_inference() -> None:
    matrix = pd.DataFrame({"raw": [0.50, 0.60], "smote": [0.60, 0.55]})

    row = _comparison_summary("mixed_core", "xgboost", "macro_f1", "smote", matrix)

    assert row["n_pairs"] == 2
    assert row["status"] == "descriptive"
    assert row["wilcoxon_statistic"] is None
    assert row["wilcoxon_p_value"] is None
    assert row["bootstrap_low"] is not None
    assert row["bootstrap_high"] is not None
    assert row["bootstrap_interpretation"] == "descriptive_only_below_minimum_three_blocks"


def test_three_block_baseline_comparison_receives_inference() -> None:
    matrix = pd.DataFrame({"raw": [0.5, 0.6, 0.7], "smote": [0.6, 0.7, 0.8]})

    row = _comparison_summary("numeric_core", "xgboost", "macro_f1", "smote", matrix)

    assert row["n_pairs"] == 3
    assert row["status"] == "computed"
    assert row["wilcoxon_p_value"] is not None
    assert row["bootstrap_interpretation"] == "paired_dataset_block_percentile_interval"


@pytest.mark.parametrize(
    ("n_blocks", "expected_status", "expected_family_size"),
    [(2, "descriptive", 0), (3, "computed", 1)],
)
def test_confirmatory_holm_family_excludes_fewer_than_three_blocks(
    n_blocks: int,
    expected_status: str,
    expected_family_size: int,
) -> None:
    rows = []
    for index in range(n_blocks):
        for condition, offset in (("raw", 0.0), ("random_over", 0.1)):
            value = 0.4 + 0.1 * index + offset
            rows.append(
                {
                    "dataset": f"d{index}",
                    "classifier": "logistic_regression",
                    "condition": condition,
                    "feature_type": "numeric",
                    "macro_f1": value,
                    "g_mean": value,
                    "mcc": value,
                    "balanced_accuracy": value,
                }
            )

    _, _, comparisons = _run_confirmatory_tests(pd.DataFrame(rows))
    row = comparisons[
        (comparisons["family"] == "numeric_core")
        & (comparisons["classifier"] == "logistic_regression")
        & (comparisons["metric"] == "macro_f1")
        & (comparisons["condition"] == "random_over")
    ].iloc[0]

    assert row["status"] == expected_status
    assert row["holm_family_size"] == expected_family_size
    assert (
        pd.isna(row["holm_p_value"])
        if expected_status == "descriptive"
        else pd.notna(row["holm_p_value"])
    )


def test_failure_accounting_separates_unsupported_and_sampler_failures() -> None:
    valid = pd.DataFrame(
        [
            {
                "dataset": "numeric-a",
                "seed": 0,
                "fold": 0,
                "classifier": "random_forest",
                "condition": "raw",
                "feature_type": "numeric",
            }
        ]
    )
    failures = pd.DataFrame(
        [
            {
                "dataset": "mixed-a",
                "seed": 0,
                "fold": 0,
                "classifier": "random_forest",
                "condition": "smote",
                "feature_type": "mixed",
                "error_type": "ValueError",
                "error": "smote is not valid for mixed data in Stage 0",
            },
            {
                "dataset": "numeric-a",
                "seed": 0,
                "fold": 0,
                "classifier": "random_forest",
                "condition": "adasyn",
                "feature_type": "numeric",
                "error_type": "ValueError",
                "error": "No samples will be generated with the provided ratio settings.",
            },
        ]
    )

    summary, reasons = _failure_accounting(valid, failures, expected_records=3)
    overall = summary.iloc[0]

    assert overall["enumerated_records"] == 3
    assert overall["unsupported_records"] == 1
    assert overall["applicable_records"] == overall["attempted_evaluations"] == 2
    assert overall["valid_records"] == 1
    assert overall["sampler_specific_failures"] == 1
    assert overall["other_applicable_failures"] == 0
    assert overall["sampler_failure_rate"] == pytest.approx(0.5)
    assert reasons["records"].sum() == 2
    assert set(reasons["record_category"]) == {"unsupported", "sampler_specific_failure"}


def test_figure_coverage_records_empty_and_observed_tiles() -> None:
    cells = pd.DataFrame(
        [
            {
                "feature_type": "numeric",
                "classifier": "logistic_regression",
                "condition": "raw",
                "dataset": "a",
                "macro_f1": 0.5,
            },
            {
                "feature_type": "numeric",
                "classifier": "logistic_regression",
                "condition": "smote",
                "dataset": "a",
                "macro_f1": 0.7,
            },
            {
                "feature_type": "numeric",
                "classifier": "logistic_regression",
                "condition": "smote",
                "dataset": "b",
                "macro_f1": 0.9,
            },
            {
                "feature_type": "mixed",
                "classifier": "random_forest",
                "condition": "raw",
                "dataset": "c",
                "macro_f1": 0.6,
            },
        ]
    )

    coverage = _macro_f1_figure_coverage(cells)
    observed = coverage.set_index(["feature_type", "classifier", "condition"])

    assert observed.loc[("numeric", "logistic_regression", "raw"), "n_datasets"] == 1
    assert observed.loc[("numeric", "logistic_regression", "smote"), "n_datasets"] == 2
    assert observed.loc[
        ("numeric", "logistic_regression", "smote"), "mean_macro_f1"
    ] == pytest.approx(0.8)
    assert observed.loc[("mixed", "logistic_regression", "smote"), "n_datasets"] == 0
    assert pd.isna(observed.loc[("mixed", "logistic_regression", "smote"), "mean_macro_f1"])


def test_small_sample_sensitivities_are_reproducible_and_exact_pairs_have_eight_patterns() -> None:
    matrix = pd.DataFrame(
        {
            "a": [0.1, 0.3, 0.2],
            "b": [0.2, 0.1, 0.3],
            "c": [0.3, 0.2, 0.1],
        }
    )
    first = friedman_permutation_sensitivity(matrix, n_resamples=1_000, seed=7)
    second = friedman_permutation_sensitivity(matrix, n_resamples=1_000, seed=7)
    exact = exact_sign_flip_sensitivity(matrix)
    ranks = np.vstack(
        [pd.Series(row).rank(method="average").to_numpy() for row in matrix.to_numpy()]
    )
    expected_statistic, _ = _friedman_rank_statistic(ranks)

    assert first == second
    assert first["friedman_statistic"] == pytest.approx(expected_statistic)
    assert first["permutation_resamples"] == 1_000
    assert len(exact) == 3
    assert set(exact["sign_patterns"]) == {8}
    assert (exact["exact_p_value"] >= 0.25).all()


def test_balanced_rf_family_is_random_forest_only() -> None:
    assert _family_classifiers("numeric_core") == (
        "logistic_regression",
        "random_forest",
        "xgboost",
    )
    assert _family_classifiers("balanced_rf_numeric") == ("random_forest",)
