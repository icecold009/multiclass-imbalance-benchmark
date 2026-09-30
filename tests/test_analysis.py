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
    _family_classifiers,
    _holm_adjust,
    load_cell_means,
    load_efficiency_means,
    paired_bootstrap_ci,
    paired_rank_biserial,
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


def test_balanced_rf_family_is_random_forest_only() -> None:
    assert _family_classifiers("numeric_core") == (
        "logistic_regression",
        "random_forest",
        "xgboost",
    )
    assert _family_classifiers("balanced_rf_numeric") == ("random_forest",)
