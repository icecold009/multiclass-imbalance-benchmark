---
name: benchmark-statistics
description: Aggregate and statistically analyse multiclass imbalance benchmark results. Use when creating rankings, Friedman/Nemenyi analyses, Wilcoxon comparisons, effect sizes, confidence intervals, metric summaries, or claims from experiment outputs.
---

# Benchmark statistics

## Choose the statistical unit

The dataset is the block. Aggregate folds and seeds within each dataset and
condition before ranking or hypothesis testing. Never treat folds or seeds as
independent datasets.

Keep classifier effects separate unless a pre-specified model explicitly
handles the classifier-by-condition interaction.

## Rank and test safely

1. Compute per-dataset mean metrics across the configured folds and seeds.
2. Rank higher-is-better metrics within each dataset and classifier.
3. Run Friedman only on complete dataset-by-condition blocks.
4. Use Nemenyi post-hoc comparisons only when the planned global test and
   method family justify them.
5. Report missing or inapplicable cells; never silently complete them.
6. Preselect the limited Wilcoxon comparisons and correct for multiple testing.
7. Report paired effect sizes and confidence intervals, not p-values alone.

## Metrics and claims

Use macro-F1 as the primary metric unless the approved protocol changes it.
Define multiclass G-mean, zero-recall behavior, MCC, balanced accuracy, and
per-class recall before analysing results. Distinguish statistical significance
from practically meaningful improvement.

With 12–19 datasets, moderator analyses involving IR, class count, or
dimensionality are exploratory unless the design and power justify stronger
claims. Do not select the most favorable metric, classifier, or subset after
viewing results.

## Reproducibility

Save the aggregation input, configuration version, dataset subset, complete
case rule, test settings, correction method, and generated tables. A pilot with
too few blocks may produce rankings for debugging but must report that the
Friedman analysis was skipped.
