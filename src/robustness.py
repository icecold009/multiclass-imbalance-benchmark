"""Gate E robustness checks for the locked benchmark analysis."""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import (
    EFFICIENCY_METRICS,
    FAMILY_DEFINITIONS,
    SCALAR_METRICS,
    _family_classifiers,
    _holm_adjust,
    run_analysis,
    verify_gate_d,
)
from src.benchmark import _read_json, _write_json_atomic, sha256_file
from src.posthoc_sensitivity import (
    POSTHOC_CLASSIFIERS,
    POSTHOC_CONDITIONS,
    POSTHOC_PROCEDURE,
    POSTHOC_RESAMPLES,
    POSTHOC_SEED,
)

ANALYSIS_TABLES = (
    "cell_means.csv",
    "class_means.csv",
    "efficiency_means.csv",
    "baseline_deltas.csv",
    "baseline_deltas_per_class.csv",
    "per_class_recall_summary.csv",
    "friedman_tests.csv",
    "nemenyi_pvalues.csv",
    "wilcoxon_effects.csv",
    "moderators.csv",
    "failure_accounting.csv",
    "failure_reason_counts.csv",
    "macro_f1_figure_coverage.csv",
    "friedman_permutation_sensitivity.csv",
    "exact_sign_flip_sensitivity.csv",
    "adasyn_excluded_friedman.csv",
    "adasyn_excluded_pairwise.csv",
    "adasyn_excluded_imbalance_descriptive.csv",
)
PROCEDURES = {
    "friedman": (
        "complete dataset blocks, at least three conditions and three blocks; "
        "asymptotic chi-square p-values retained as planned"
    ),
    "nemenyi": "all pairs after a significant Friedman test, alpha 0.05",
    "wilcoxon": {
        "comparison": "raw versus every alternative in the same family",
        "alternative": "two-sided",
        "zero_method": "wilcox",
        "correction": False,
        "method": "auto",
    },
    "holm": (
        "within family, classifier, and metric across Wilcoxon p-values for "
        "comparisons with at least three complete dataset blocks"
    ),
    "rank_biserial": "paired signed ranks after removing zero differences",
    "bootstrap": (
        "paired dataset-block percentile interval: 10,000 resamples, seed 0; "
        "intervals with fewer than three blocks are descriptive and not calibrated "
        "confidence intervals"
    ),
    "failure_accounting": (
        "classify unsupported records from frozen feature/classifier/condition "
        "applicability; count valid and applicable failures as attempted evaluations"
    ),
    "macro_f1_figure": (
        "feature type/classifier by condition; tile mean over complete dataset cells; "
        "coverage counts and empty tiles are recorded separately"
    ),
    "primary_small_sample_sensitivity": {
        "outcome": "macro_f1",
        "family": "numeric_core",
        "classifier": "xgboost",
        "post_hoc": True,
        "friedman_permutation": (
            "200,000 seeded within-block condition-label rank permutations, seed 0, "
            "plus-one Monte Carlo p-value; assumes condition-label exchangeability"
        ),
        "exact_pairwise": (
            "two-sided exact sign flips across all 36 condition pairs; assumes "
            "independent sign exchangeability of dataset-level paired differences; "
            "Holm-adjusted within the 36-pair sensitivity family"
        ),
        "status": "sensitivity only; planned asymptotic Friedman/Nemenyi results retained",
    },
    "adasyn_excluded_sensitivity": POSTHOC_PROCEDURE,
}
PER_CLASS_NOTE = "descriptive class-wise summaries; inferential matrices skipped because class labels are dataset-specific"


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reference(path: Path, root: Path) -> dict[str, str]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Gate E artifact is outside the repository: {path}") from error
    return {"path": relative, "sha256": sha256_file(path)}


def _resolve_artifact(root: Path, reference: object, label: str) -> Path:
    if not isinstance(reference, dict):
        raise TypeError(f"{label} reference is not an object")
    relative = reference.get("path")
    digest = reference.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise TypeError(f"{label} reference is incomplete")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} reference escapes the repository: {relative}") from error
    if not path.is_file():
        raise RuntimeError(f"{label} artifact is missing: {path}")
    if sha256_file(path) != digest:
        raise RuntimeError(f"{label} artifact hash does not match its manifest")
    return path


def _assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"Gate E mismatch for {label}: expected {expected!r}, got {actual!r}")


def _finite_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        if column not in frame.columns:
            raise RuntimeError(f"{label} is missing column {column!r}")
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{label}.{column} contains non-finite values")


def _validate_family_rows(frame: pd.DataFrame, label: str) -> None:
    for _, row in frame.iterrows():
        family = str(row["family"])
        classifier = str(row["classifier"])
        condition = str(row["condition"])
        if family not in FAMILY_DEFINITIONS:
            raise RuntimeError(f"{label} contains an unknown family: {family}")
        if classifier not in _family_classifiers(family):
            raise RuntimeError(f"{label} contains a classifier outside its family: {classifier}")
        if condition not in FAMILY_DEFINITIONS[family][1] or condition == "raw":
            raise RuntimeError(f"{label} contains an invalid condition: {condition}")


def _key_set(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> set[tuple[str, ...]]:
    if any(column not in frame.columns for column in columns):
        raise RuntimeError(f"{label} is missing a required key column")
    return {
        tuple(str(value) for value in row) for row in frame[list(columns)].itertuples(index=False)
    }


def _validate_tables(
    analysis_dir: Path,
    manifest: dict[str, Any],
    dataset_ids: list[str],
) -> dict[str, int]:
    tables = {name: pd.read_csv(analysis_dir / name) for name in ANALYSIS_TABLES}
    cell_keys = ("dataset", "classifier", "condition", "feature_type")
    cells = tables["cell_means.csv"]
    if len(cells) != int(manifest["cell_mean_count"]):
        raise RuntimeError("cell_means row count does not match the analysis manifest")
    if cells[list(cell_keys)].duplicated().any():
        raise RuntimeError("cell_means contains duplicate complete cells")
    if set(cells["dataset"].astype(str)) - set(dataset_ids):
        raise RuntimeError("cell_means contains a dataset outside the locked scope")
    if set(cells["complete_records"].astype(int)) != {15}:
        raise RuntimeError("cell_means contains a non-15-record cell")
    _finite_columns(cells, SCALAR_METRICS, "cell_means")

    class_means = tables["class_means.csv"]
    if len(class_means) != int(manifest["class_mean_count"]):
        raise RuntimeError("class_means row count does not match the analysis manifest")
    class_keys = cell_keys + ("class_index",)
    if class_means[list(class_keys)].duplicated().any():
        raise RuntimeError("class_means contains duplicate cell/class rows")
    _finite_columns(class_means, ("class_index", "recall", "complete_records"), "class_means")
    efficiency = tables["efficiency_means.csv"]
    if len(efficiency) != int(manifest["efficiency_mean_count"]):
        raise RuntimeError("efficiency_means row count does not match the analysis manifest")
    if set(efficiency["complete_records"].astype(int)) != {15}:
        raise RuntimeError("efficiency_means contains a non-15-record cell")
    if _key_set(efficiency, cell_keys, "efficiency_means") != _key_set(
        cells, cell_keys, "cell_means"
    ):
        raise RuntimeError("efficiency_means keys do not match complete cells")
    efficiency_columns = tuple(
        f"{metric}_{statistic}" for metric in EFFICIENCY_METRICS for statistic in ("mean", "median")
    )
    _finite_columns(efficiency, efficiency_columns, "efficiency_means")

    deltas = tables["baseline_deltas.csv"]
    if len(deltas) != int(manifest["baseline_delta_count"]):
        raise RuntimeError("baseline_deltas row count does not match the analysis manifest")
    if (deltas["condition"] == "raw").any():
        raise RuntimeError("baseline_deltas contains a raw self-comparison")
    _validate_family_rows(deltas, "baseline_deltas")
    _finite_columns(deltas, ("raw_mean", "alternative_mean", "delta"), "baseline_deltas")
    comparisons = tables["wilcoxon_effects.csv"]
    _finite_columns(comparisons, ("wilcoxon_effective_n",), "wilcoxon_effects")
    if (comparisons["wilcoxon_effective_n"] < 0).any() or (
        comparisons["wilcoxon_effective_n"] > comparisons["n_pairs"]
    ).any():
        raise RuntimeError("Wilcoxon effective nonzero counts are invalid")
    exact_calibrations = {
        "exact_no_ties_or_zeros",
        "exhaustive_sign_permutation",
        "degenerate_all_zero_p_one",
    }
    exact_rows = comparisons["wilcoxon_calibration"].isin(exact_calibrations)
    _finite_columns(
        comparisons.loc[exact_rows],
        ("wilcoxon_min_attainable_p",),
        "exact Wilcoxon minimum p-values",
    )
    if (
        comparisons.loc[exact_rows, "wilcoxon_min_attainable_p"].le(0).any()
        or comparisons.loc[exact_rows, "wilcoxon_min_attainable_p"].gt(1).any()
    ):
        raise RuntimeError("Wilcoxon minimum attainable p-values are outside (0, 1]")
    if comparisons.loc[~exact_rows, "wilcoxon_min_attainable_p"].notna().any():
        raise RuntimeError("a minimum p-value is reported where the test is not exactly calibrated")

    class_deltas = tables["baseline_deltas_per_class.csv"]
    if len(class_deltas) != int(manifest["per_class_delta_count"]):
        raise RuntimeError(
            "baseline_deltas_per_class row count does not match the analysis manifest"
        )
    _finite_columns(
        class_deltas,
        ("class_index", "raw_recall", "alternative_recall", "delta"),
        "baseline_deltas_per_class",
    )
    _validate_family_rows(class_deltas, "baseline_deltas_per_class")
    class_summary = tables["per_class_recall_summary.csv"]
    _finite_columns(
        class_summary,
        ("class_index", "n_pairs", "mean_delta", "median_delta", "wins", "ties", "losses"),
        "per_class_recall_summary",
    )

    friedman = tables["friedman_tests.csv"]
    expected_friedman = sum(
        len(_family_classifiers(family)) * len(SCALAR_METRICS) for family in FAMILY_DEFINITIONS
    )
    if len(friedman) != expected_friedman or len(friedman) != int(manifest["friedman_test_count"]):
        raise RuntimeError("friedman_tests row count does not match the frozen family matrix")
    for _, row in friedman.iterrows():
        family = str(row["family"])
        classifier = str(row["classifier"])
        metric = str(row["metric"])
        if family not in FAMILY_DEFINITIONS:
            raise RuntimeError(f"Friedman output contains an unknown family: {family}")
        if classifier not in _family_classifiers(family):
            raise RuntimeError(
                f"Friedman output contains a classifier outside its family: {classifier}"
            )
        if metric not in SCALAR_METRICS:
            raise RuntimeError(f"Friedman output contains an unknown metric: {metric}")
        status = str(row["status"])
        if status not in {"computed", "skipped"}:
            raise RuntimeError(f"unknown Friedman status: {status}")
        if status == "computed":
            if not (int(row["n_blocks"]) >= 3 and int(row["n_conditions"]) >= 3):
                raise RuntimeError("computed Friedman test does not meet minimum dimensions")
            if not 0 <= float(row["friedman_p_value"]) <= 1:
                raise RuntimeError("computed Friedman p-value is outside [0, 1]")
        else:
            if not str(row["reason"]):
                raise RuntimeError("skipped Friedman test has no reason")
            if pd.notna(row["friedman_statistic"]) or pd.notna(row["friedman_p_value"]):
                raise RuntimeError("skipped Friedman test contains a statistic or p-value")

    significant: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for _, row in friedman.iterrows():
        if str(row["status"]) == "computed" and float(row["friedman_p_value"]) < 0.05:
            family = str(row["family"])
            significant[(family, str(row["classifier"]), str(row["metric"]))] = FAMILY_DEFINITIONS[
                family
            ][1]
    nemenyi = tables["nemenyi_pvalues.csv"]
    if len(nemenyi) != int(manifest["nemenyi_pair_count"]):
        raise RuntimeError("nemenyi_pvalues row count does not match the analysis manifest")
    seen_pairs: set[tuple[str, str, str, str, str]] = set()
    for _, row in nemenyi.iterrows():
        key = (str(row["family"]), str(row["classifier"]), str(row["metric"]))
        if key not in significant:
            raise RuntimeError("Nemenyi output exists for a non-significant Friedman block")
        pair = (key[0], key[1], key[2], str(row["condition_a"]), str(row["condition_b"]))
        if pair in seen_pairs or str(row["condition_a"]) == str(row["condition_b"]):
            raise RuntimeError("Nemenyi output contains a duplicate or diagonal pair")
        if pair[3] not in significant[key] or pair[4] not in significant[key]:
            raise RuntimeError("Nemenyi output contains a condition outside its family")
        seen_pairs.add(pair)
        if not 0 <= float(row["p_value"]) <= 1:
            raise RuntimeError("Nemenyi p-value is outside [0, 1]")
    for key, conditions in significant.items():
        count = sum(1 for pair in seen_pairs if pair[:3] == key)
        if count != len(conditions) * (len(conditions) - 1) // 2:
            raise RuntimeError(f"Nemenyi pair count is incomplete for {key}")

    wilcoxon = tables["wilcoxon_effects.csv"]
    if len(wilcoxon) != int(manifest["wilcoxon_comparison_count"]):
        raise RuntimeError("wilcoxon_effects row count does not match the analysis manifest")
    for _, row in wilcoxon.iterrows():
        if str(row["condition"]) == "raw":
            raise RuntimeError("Wilcoxon output contains a raw self-comparison")
        if int(row["n_pairs"]) < 0:
            raise RuntimeError("Wilcoxon output contains a negative pair count")
    _validate_family_rows(wilcoxon, "wilcoxon_effects")
    for _, row in wilcoxon.iterrows():
        status = str(row["status"])
        if status == "computed":
            if int(row["n_pairs"]) < 3:
                raise RuntimeError("computed Wilcoxon test has fewer than three dataset blocks")
            for column in ("wilcoxon_p_value", "holm_p_value", "bootstrap_low", "bootstrap_high"):
                value = float(row[column])
                if not np.isfinite(value):
                    raise RuntimeError(f"Wilcoxon output has an invalid {column}")
                if column in {"wilcoxon_p_value", "holm_p_value"} and not 0 <= value <= 1:
                    raise RuntimeError(f"Wilcoxon output has an invalid {column}")
            if float(row["bootstrap_low"]) > float(row["bootstrap_high"]):
                raise RuntimeError("Wilcoxon bootstrap interval is reversed")
            if str(row["bootstrap_interpretation"]) != "paired_dataset_block_percentile_interval":
                raise RuntimeError("computed Wilcoxon row has an invalid bootstrap interpretation")
        elif status == "descriptive":
            if int(row["n_pairs"]) not in {1, 2}:
                raise RuntimeError("descriptive Wilcoxon row does not have one or two blocks")
            for column in ("wilcoxon_statistic", "wilcoxon_p_value", "holm_p_value"):
                if pd.notna(row[column]):
                    raise RuntimeError(f"descriptive comparison contains inferential {column}")
            for column in ("bootstrap_low", "bootstrap_high"):
                if not np.isfinite(float(row[column])):
                    raise RuntimeError("descriptive comparison is missing its descriptive interval")
            if str(
                row["bootstrap_interpretation"]
            ) != "descriptive_only_below_minimum_three_blocks" or not str(row["reason"]):
                raise RuntimeError("descriptive comparison has an invalid reason or interval note")
        elif status == "skipped":
            if int(row["n_pairs"]) != 0 or not str(row["reason"]):
                raise RuntimeError("skipped Wilcoxon row has an invalid count or reason")
            for column in (
                "wilcoxon_statistic",
                "wilcoxon_p_value",
                "holm_p_value",
                "bootstrap_low",
                "bootstrap_high",
            ):
                if pd.notna(row[column]):
                    raise RuntimeError(f"skipped comparison contains {column}")
            if str(row["bootstrap_interpretation"]) != "not_available":
                raise RuntimeError("skipped comparison has an invalid bootstrap interpretation")
        else:
            raise RuntimeError(f"Wilcoxon output has an invalid status: {status}")
    if int((wilcoxon["status"] == "computed").sum()) != int(manifest["wilcoxon_test_count"]):
        raise RuntimeError("Wilcoxon inferential-test count does not match the manifest")
    if int((wilcoxon["status"] == "descriptive").sum()) != int(
        manifest["wilcoxon_descriptive_count"]
    ):
        raise RuntimeError("Wilcoxon descriptive-row count does not match the manifest")
    if int((wilcoxon["status"] == "skipped").sum()) != int(manifest["wilcoxon_skipped_count"]):
        raise RuntimeError("Wilcoxon skipped-row count does not match the manifest")
    for _, group in wilcoxon.groupby(["family", "classifier", "metric"], sort=True):
        expected_family_size = int(group["wilcoxon_p_value"].notna().sum())
        if set(group["holm_family_size"].astype(int)) != {expected_family_size}:
            raise RuntimeError("Wilcoxon Holm-family size is inconsistent with tested rows")
        tested = group["wilcoxon_p_value"].notna()
        if not np.all(
            group.loc[tested, "holm_p_value"].to_numpy(dtype=float)
            >= group.loc[tested, "wilcoxon_p_value"].to_numpy(dtype=float) - 1e-12
        ):
            raise RuntimeError("Holm-adjusted p-value is smaller than its raw p-value")
    if len(wilcoxon) != 184 or int(wilcoxon["status"].eq("computed").sum()) != 104:
        raise RuntimeError("Wilcoxon reporting does not apply the frozen three-block rule")
    if int(wilcoxon["holm_p_value"].lt(0.05).sum()) != 1:
        raise RuntimeError("the frozen Wilcoxon/Holm result no longer has one significant row")

    failure_accounting = tables["failure_accounting.csv"]
    required_failure_columns = {
        "aggregation_level",
        "feature_type",
        "classifier",
        "condition",
        "enumerated_records",
        "unsupported_records",
        "applicable_records",
        "attempted_evaluations",
        "valid_records",
        "sampler_specific_failures",
        "other_applicable_failures",
        "sampler_failure_rate",
    }
    if required_failure_columns - set(failure_accounting.columns):
        raise RuntimeError("failure_accounting is missing required fields")
    overall = failure_accounting[failure_accounting["aggregation_level"] == "overall"]
    if len(overall) != 1:
        raise RuntimeError("failure_accounting must contain one overall row")
    overall = overall.iloc[0]
    accounting_counts = manifest["failure_accounting_counts"]
    for column in (
        "enumerated_records",
        "unsupported_records",
        "applicable_records",
        "attempted_evaluations",
        "valid_records",
        "sampler_specific_failures",
        "other_applicable_failures",
    ):
        value = int(overall[column])
        if value != int(accounting_counts[column]):
            raise RuntimeError(f"failure accounting differs from manifest for {column}")
    gate_counts = manifest["gate_d"]["counts"]
    if (
        int(overall["enumerated_records"]) != int(gate_counts["expected_cells"])
        or int(overall["valid_records"]) != int(gate_counts["valid_cells"])
        or int(overall["sampler_specific_failures"])
        + int(overall["other_applicable_failures"])
        + int(overall["unsupported_records"])
        != int(gate_counts["failure_cells"])
    ):
        raise RuntimeError("failure accounting does not reconcile with Gate D counts")
    if (
        int(overall["enumerated_records"]) != 5445
        or int(overall["unsupported_records"]) != 1005
        or int(overall["applicable_records"]) != 4440
        or int(overall["valid_records"]) != 4191
        or int(overall["sampler_specific_failures"]) != 249
        or int(overall["other_applicable_failures"]) != 0
    ):
        raise RuntimeError("failure categories differ from the verified frozen-run totals")
    grouped_accounting = failure_accounting[
        failure_accounting["aggregation_level"] == "feature_classifier_condition"
    ]
    for column in (
        "enumerated_records",
        "unsupported_records",
        "applicable_records",
        "attempted_evaluations",
        "valid_records",
        "sampler_specific_failures",
        "other_applicable_failures",
    ):
        if int(grouped_accounting[column].sum()) != int(overall[column]):
            raise RuntimeError(f"grouped failure accounting does not sum for {column}")
    if not np.isclose(
        float(overall["sampler_failure_rate"]),
        int(overall["sampler_specific_failures"]) / int(overall["applicable_records"]),
    ):
        raise RuntimeError("overall sampler failure rate has the wrong applicable denominator")
    reason_counts = tables["failure_reason_counts.csv"]
    if int(reason_counts["records"].sum()) != int(gate_counts["failure_cells"]):
        raise RuntimeError("failure reason counts do not cover the raw failure log")
    if set(reason_counts["record_category"]) - {
        "unsupported",
        "sampler_specific_failure",
        "other_applicable_failure",
    }:
        raise RuntimeError("failure reason counts contain an unknown record category")
    if len(failure_accounting) != int(manifest["failure_accounting_row_count"]):
        raise RuntimeError("failure_accounting row count does not match the manifest")
    if len(reason_counts) != int(manifest["failure_reason_row_count"]):
        raise RuntimeError("failure_reason_counts row count does not match the manifest")

    figure_coverage = tables["macro_f1_figure_coverage.csv"]
    coverage_keys = ["feature_type", "classifier", "condition"]
    if figure_coverage[coverage_keys].duplicated().any():
        raise RuntimeError("macro-F1 figure coverage contains duplicate tiles")
    expected_coverage = cells.groupby(coverage_keys, as_index=False, sort=True).agg(
        expected_mean=("macro_f1", "mean"),
        expected_n=("dataset", "nunique"),
    )
    coverage_joined = figure_coverage.merge(
        expected_coverage,
        on=coverage_keys,
        how="left",
        validate="one_to_one",
    )
    expected_n = coverage_joined["expected_n"].fillna(0).astype(int).to_numpy()
    if not np.array_equal(figure_coverage["n_datasets"].to_numpy(dtype=int), expected_n):
        raise RuntimeError("macro-F1 figure dataset counts differ from complete cells")
    nonempty = expected_n > 0
    actual_means = pd.to_numeric(
        coverage_joined.loc[nonempty, "mean_macro_f1"], errors="coerce"
    ).to_numpy(dtype=float)
    expected_means = coverage_joined.loc[nonempty, "expected_mean"].to_numpy(dtype=float)
    if not np.allclose(actual_means, expected_means, rtol=0, atol=1e-12):
        raise RuntimeError("macro-F1 figure means differ from complete-cell means")
    if figure_coverage.loc[~nonempty, "mean_macro_f1"].notna().any():
        raise RuntimeError("empty macro-F1 figure tile contains a mean")
    if len(figure_coverage) != int(manifest["figure_coverage_row_count"]):
        raise RuntimeError("macro-F1 figure coverage count does not match the manifest")

    permutation = tables["friedman_permutation_sensitivity.csv"]
    if len(permutation) != 1:
        raise RuntimeError("primary Friedman permutation sensitivity must have one row")
    permutation_row = permutation.iloc[0]
    if (
        str(permutation_row["family"]) != "numeric_core"
        or str(permutation_row["classifier"]) != "xgboost"
        or str(permutation_row["metric"]) != "macro_f1"
        or int(permutation_row["n_blocks"]) != 3
        or int(permutation_row["n_conditions"]) != 9
        or int(permutation_row["permutation_resamples"]) != 200_000
        or int(permutation_row["permutation_seed"]) != 0
    ):
        raise RuntimeError("primary Friedman permutation sensitivity has unexpected scope")
    for column in (
        "friedman_statistic",
        "asymptotic_p_value",
        "tie_correction",
        "permutation_p_value",
        "permutation_monte_carlo_se",
    ):
        value = float(permutation_row[column])
        if not np.isfinite(value):
            raise RuntimeError(f"permutation sensitivity has non-finite {column}")
    if not 0 <= float(permutation_row["permutation_p_value"]) <= 1:
        raise RuntimeError("permutation sensitivity p-value is outside [0, 1]")
    if len(permutation) != int(manifest["primary_permutation_sensitivity_count"]):
        raise RuntimeError("permutation sensitivity count does not match the manifest")
    exact = tables["exact_sign_flip_sensitivity.csv"]
    if (
        len(exact) != 36
        or len(exact) != int(manifest["primary_exact_pairwise_count"])
        or exact[["condition_a", "condition_b"]].duplicated().any()
        or set(exact["n_blocks"].astype(int)) != {3}
        or set(exact["sign_patterns"].astype(int)) != {8}
        or set(exact["holm_family_size"].astype(int)) != {36}
    ):
        raise RuntimeError("exact sign-flip sensitivity does not contain 36 three-block pairs")
    for column in ("exact_p_value", "holm_p_value", "mean_delta_a_minus_b"):
        _finite_columns(exact, (column,), "exact_sign_flip_sensitivity")
    if (
        (exact["exact_p_value"] < 0).any()
        or (exact["exact_p_value"] > 1).any()
        or (exact["holm_p_value"] < exact["exact_p_value"] - 1e-12).any()
    ):
        raise RuntimeError("exact sign-flip p-values are invalid")
    if not np.allclose(
        exact["holm_p_value"].to_numpy(dtype=float),
        np.asarray(_holm_adjust(exact["exact_p_value"].astype(float).tolist())),
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("exact sign-flip Holm adjustment is incorrect")

    posthoc_omnibus = tables["adasyn_excluded_friedman.csv"]
    if (
        len(posthoc_omnibus) != 3
        or len(posthoc_omnibus) != int(manifest["adasyn_excluded_omnibus_count"])
        or set(posthoc_omnibus["classifier"].astype(str)) != set(POSTHOC_CLASSIFIERS)
        or set(posthoc_omnibus["n_blocks"].astype(int)) != {9}
        or set(posthoc_omnibus["n_conditions"].astype(int)) != {8}
        or set(posthoc_omnibus["permutation_resamples"].astype(int)) != {POSTHOC_RESAMPLES}
        or set(posthoc_omnibus["permutation_seed"].astype(int)) != {POSTHOC_SEED}
    ):
        raise RuntimeError("ADASYN-excluded omnibus sensitivity has unexpected scope")
    _finite_columns(
        posthoc_omnibus,
        ("friedman_statistic", "asymptotic_p_value", "permutation_p_value"),
        "adasyn_excluded_friedman",
    )
    if posthoc_omnibus["permutation_p_value"].lt(0).any() or posthoc_omnibus[
        "permutation_p_value"
    ].gt(1).any():
        raise RuntimeError("ADASYN-excluded permutation p-values are invalid")

    posthoc_pairs = tables["adasyn_excluded_pairwise.csv"]
    if (
        len(posthoc_pairs) != 21
        or len(posthoc_pairs) != int(manifest["adasyn_excluded_pairwise_count"])
        or posthoc_pairs[["classifier", "condition"]].duplicated().any()
        or set(posthoc_pairs["classifier"].astype(str)) != set(POSTHOC_CLASSIFIERS)
        or set(posthoc_pairs["condition"].astype(str)) != set(POSTHOC_CONDITIONS[1:])
        or set(posthoc_pairs["n_pairs"].astype(int)) != {9}
        or set(posthoc_pairs["holm_family_size"].astype(int)) != {7}
    ):
        raise RuntimeError("ADASYN-excluded exact follow-ups have unexpected scope")
    _finite_columns(
        posthoc_pairs,
        (
            "mean_delta",
            "median_delta",
            "n_nonzero_pairs",
            "sign_patterns",
            "wilcoxon_statistic",
            "exact_p_value",
            "minimum_attainable_p",
            "holm_p_value",
        ),
        "adasyn_excluded_pairwise",
    )
    if (
        posthoc_pairs["n_nonzero_pairs"].lt(0).any()
        or posthoc_pairs["n_nonzero_pairs"].gt(9).any()
        or posthoc_pairs["exact_p_value"].lt(0).any()
        or posthoc_pairs["exact_p_value"].gt(1).any()
        or posthoc_pairs["holm_p_value"].lt(posthoc_pairs["exact_p_value"] - 1e-12).any()
        or posthoc_pairs["minimum_attainable_p"].lt(0).any()
        or posthoc_pairs["minimum_attainable_p"].gt(1).any()
    ):
        raise RuntimeError("ADASYN-excluded exact follow-up values are invalid")
    for classifier, family in posthoc_pairs.groupby("classifier", sort=True):
        expected_adjustment = _holm_adjust(family["exact_p_value"].astype(float).tolist())
        if not np.allclose(
            family["holm_p_value"].to_numpy(dtype=float),
            np.asarray(expected_adjustment),
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(f"ADASYN-excluded Holm adjustment is incorrect for {classifier}")

    stratified = tables["adasyn_excluded_imbalance_descriptive.csv"]
    if (
        len(stratified) != 42
        or len(stratified) != int(manifest["adasyn_excluded_stratified_count"])
        or stratified[["classifier", "condition", "stratum"]].duplicated().any()
        or set(stratified["classifier"].astype(str)) != set(POSTHOC_CLASSIFIERS)
        or set(stratified["condition"].astype(str)) != set(POSTHOC_CONDITIONS[1:])
        or set(stratified["n_datasets"].astype(int)) != {3, 6}
        or set(stratified["inferential_status"].astype(str)) != {"descriptive_only_no_tests"}
    ):
        raise RuntimeError("post hoc imbalance-stratified summaries have unexpected scope")
    _finite_columns(
        stratified,
        ("mean_paired_delta", "median_paired_delta", "wins", "ties", "losses"),
        "adasyn_excluded_imbalance_descriptive",
    )

    moderators = tables["moderators.csv"]
    if set(moderators["dataset_id"].astype(str)) != set(dataset_ids):
        raise RuntimeError("moderators do not cover exactly the locked datasets")
    return {name.removesuffix(".csv"): len(frame) for name, frame in tables.items()}


def _replay_check(
    root: Path,
    run_dir: Path,
    analysis_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    temporary_root = root / "output" / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="stage0-gate-e-",
        dir=temporary_root,
    ) as temporary:
        replay_dir = Path(temporary) / "analysis"
        replay_manifest = run_analysis(root, run_dir, replay_dir)
        for field in (
            "status",
            "analysis_runtime",
            "dataset_ids",
            "cell_mean_count",
            "class_mean_count",
            "efficiency_mean_count",
            "baseline_delta_count",
            "per_class_delta_count",
            "friedman_test_count",
            "nemenyi_pair_count",
            "wilcoxon_comparison_count",
            "wilcoxon_test_count",
            "wilcoxon_descriptive_count",
            "wilcoxon_skipped_count",
            "failure_accounting_row_count",
            "failure_reason_row_count",
            "figure_coverage_row_count",
            "primary_permutation_sensitivity_count",
            "primary_exact_pairwise_count",
            "adasyn_excluded_omnibus_count",
            "adasyn_excluded_pairwise_count",
            "adasyn_excluded_stratified_count",
            "failure_accounting_counts",
            "bootstrap_resamples",
            "bootstrap_seed",
            "alpha",
            "family_definitions",
            "procedures",
            "per_class_recall_inference",
            "figures",
        ):
            _assert_equal(replay_manifest.get(field), manifest.get(field), f"replay {field}")
        compared: list[str] = []
        for name in ANALYSIS_TABLES:
            original = analysis_dir / name
            replay = replay_dir / name
            if sha256_file(original) != sha256_file(replay):
                raise RuntimeError(f"deterministic replay changed {name}")
            compared.append(name)
        for figure in manifest["figures"]:
            if figure["status"] != "generated":
                continue
            original = analysis_dir / "figures" / figure["name"]
            replay = replay_dir / "figures" / figure["name"]
            if sha256_file(original) != sha256_file(replay):
                raise RuntimeError(f"deterministic replay changed figure {figure['name']}")
            compared.append(f"figures/{figure['name']}")
    return {"status": "passed", "compared_artifacts": compared}


def verify_gate_e(
    root: Path,
    run_dir: Path,
    analysis_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and replay the locked analysis before paper drafting."""

    analysis_manifest_path = analysis_dir / "analysis_manifest.json"
    manifest = _read_json(analysis_manifest_path, "analysis manifest")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"analysis manifest is not complete: {manifest.get('status')!r}")
    fresh_gate = verify_gate_d(root, run_dir)
    analysis_gate = manifest.get("gate_d")
    if not isinstance(analysis_gate, dict):
        raise TypeError("analysis manifest has no Gate D payload")
    for field in (
        "status",
        "benchmark_manifest",
        "dataset_ids",
        "dataset_count",
        "seeds",
        "folds",
        "expected_cells_per_dataset",
        "counts",
        "aggregate",
        "frozen_references",
        "selection_rule",
    ):
        _assert_equal(analysis_gate.get(field), fresh_gate.get(field), f"Gate D {field}")
    _assert_equal(
        manifest.get("input_run_manifest"),
        fresh_gate.get("benchmark_manifest"),
        "input run manifest",
    )
    _assert_equal(
        manifest.get("per_class_recall_inference"), PER_CLASS_NOTE, "per-class inference boundary"
    )
    _assert_equal(manifest.get("procedures"), PROCEDURES, "analysis procedures")
    expected_families = {
        family: {
            "feature_type": feature_type,
            "conditions": list(conditions),
            "classifiers": list(_family_classifiers(family)),
        }
        for family, (feature_type, conditions) in FAMILY_DEFINITIONS.items()
    }
    _assert_equal(manifest.get("family_definitions"), expected_families, "family definitions")
    for key, reference in manifest.get("artifacts", {}).items():
        if not isinstance(reference, dict) or reference.get("path") != key:
            raise RuntimeError(f"analysis artifact key/reference mismatch: {key}")
        _resolve_artifact(root, reference, key)
    required = {"gate-d-review.json", *ANALYSIS_TABLES}
    actual = set(manifest.get("artifacts", {}))
    expected_prefix = "results/analysis/"
    if {
        name.removeprefix(expected_prefix) for name in actual if name.startswith(expected_prefix)
    } != required | {
        f"figures/{figure['name']}"
        for figure in manifest.get("figures", [])
        if figure["status"] == "generated"
    }:
        raise RuntimeError("analysis manifest artifact set is incomplete or unexpected")
    counts = _validate_tables(analysis_dir, manifest, list(fresh_gate["dataset_ids"]))
    replay = _replay_check(root, run_dir, analysis_dir, manifest)
    payload: dict[str, Any] = {
        "schema_version": "gate-e-review-v1",
        "status": "passed",
        "reviewed_at_utc": _utc_now(),
        "analysis_manifest": _reference(analysis_manifest_path, root),
        "gate_d": {
            "status": fresh_gate["status"],
            "counts": fresh_gate["counts"],
            "dataset_ids": fresh_gate["dataset_ids"],
            "benchmark_manifest": fresh_gate["benchmark_manifest"],
        },
        "analysis_counts": counts,
        "deterministic_replay": replay,
        "decision": "analysis frozen for independent robustness review and paper drafting",
        "claim_boundary": "no paper claims, scope changes, or protocol amendments are authorized by this payload",
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output_path, payload)
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate and replay the locked analysis")
    parser.add_argument("--run-dir", type=Path, default=Path("results/full-run"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    analysis_dir = (
        args.analysis_dir if args.analysis_dir.is_absolute() else root / args.analysis_dir
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    payload = verify_gate_e(root, run_dir, analysis_dir, output_dir / "gate-e-review.json")
    print(payload["status"])
    print(output_dir / "gate-e-review.json")


if __name__ == "__main__":
    main()
