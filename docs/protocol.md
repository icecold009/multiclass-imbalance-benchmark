# Locked protocol template

This document is the pre-registration checklist for the first benchmark. It
must be completed and committed before the full dataset run. Stage 0 may update
the values marked as provisional; full-run results must not update them.

## Research question

Which imbalance-handling strategies improve multiclass imbalanced tabular
classification, and how do their effects vary with imbalance ratio, number of
classes, feature dimensionality, and feature type?

The benchmark compares raw training, class weighting, classical resampling,
and an RF-only balanced ensemble. It does not pool the portfolio case studies
into the statistical analysis.

## Experimental factors

Classifiers:

- Random Forest;
- Logistic Regression;
- XGBoost.

Conditions:

- raw baseline;
- class-weighted classifier;
- applicable resamplers;
- Balanced Random Forest as an RF-only comparator.

All classifier and sampler parameters are recorded in
[`config/stage0.yaml`](../config/stage0.yaml). Any change after scope lock
requires a dated protocol amendment and a timeline impact note.

## Data and folds

Use the audited registry as the dataset source of truth. For each dataset,
record source version, hash, license/terms, label counts, feature types, raw
and encoded dimensionality, missingness, duplicate status, leakage status, and
split policy.

Use five-fold stratified CV with seeds 0, 1, and 2 only for datasets whose rows
are independent. A dataset with groups, repeated entities, or temporal order
must use an approved group/time split or be excluded from the core benchmark.

For every fold, fit imputation, encoding, scaling, resampling, and training-fold
weights using training data only. Never resample or fit training-derived
transforms on the test fold.

## Applicability and complete cells

Ordinary SMOTE-family methods are not applied to one-hot categorical features.
Mixed data uses SMOTENC before final one-hot encoding. Categorical-only data is
excluded until SMOTEN is implemented. Unsupported combinations remain visible
in the failure log.

Friedman/Nemenyi comparisons use complete dataset-by-condition blocks. Numeric
and mixed-data analyses are separated when their applicable condition sets
differ. Balanced Random Forest is analysed separately from the three-classifier
resampling matrix.

## Outcomes

Macro-F1 is the primary outcome. Secondary outcomes are multiclass G-mean,
MCC, balanced accuracy, and per-class recall. The exact G-mean definition,
zero-recall handling, and `zero_division` behavior must be implemented before
the full run.

Report per-dataset means over folds and seeds, paired deltas from the raw
baseline, rank summaries, win/tie/loss counts, effect sizes, and confidence
intervals. Report runtime, memory measurement method, post-sampling row counts,
and all failures as reproducibility evidence.

## Statistical analysis

Datasets are the statistical blocks. Do not treat folds or seeds as independent
datasets. Run Friedman only on complete blocks, then use the pre-specified
post-hoc procedure. Preselect limited Wilcoxon comparisons and correct for
multiple testing across each declared family.

Moderator analyses for IR, class count, dimensionality, and feature type are
exploratory unless the final dataset count supports stronger inference.

## Claim boundary

The result is an empirical benchmark, not a new algorithm. Do not claim that a
method is universally best, that synthetic pilot data proves real-world
performance, or that an observed moderator is a validated selection rule.
