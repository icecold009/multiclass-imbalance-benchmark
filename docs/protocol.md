# Final protocol and statistical analysis plan

**Protocol version:** `stage0-v1`<br>
**Analysis-plan version:** `final-analysis-v1`<br>
**Frozen:** 2026-09-10<br>
**Status:** committed before the full benchmark

This document is the pre-registered protocol for the first benchmark. Stage 0
feasibility outputs are not full-run evidence and cannot change this plan.
Any material change requires a dated amendment, a reason, and a timeline note
before the affected run or analysis.

## Scope lock

The approved scope contains these eleven independent, public datasets:

`balance_scale`, `cmc`, `cnae_9`, `dermatology`, `glass`, `iris`, `optdigits`,
`seeds`, `wine`, `wine_quality_red`, and `yeast`.

The committed registry, acquisition manifest, configuration, and environment
references at protocol freeze are:

| Evidence | SHA-256 |
|---|---|
| `data/dataset_registry.csv` | `f623b2bec5881431e6f69840626a1cfa0e527e446d452d30d4aea702f45916d4` |
| `data/acquisition_manifest.csv` | `261cbc9141080d3d57bf2b5a71943a0ebd291c93c71e90eb24670be7eff0bcd0` |
| `config/stage0.yaml` | `18f8fd5f41ee47f2b3b104b608cf36b6a99b7a405601480860e31e9c5c041491` |
| `artifacts/environment/metadata.json` | `4cb76bfb8e0418b7920088edf75175a8aee52d021af93bf1d3fcd44ab7c1b56a` |

Multiple Features (Factors) remains deferred because its corrected replay
exceeded the 900-second Stage 0 budget. It is not part of this benchmark
unless a protocol-preserving resource amendment is approved before the run.

## Research question and factors

The benchmark asks which locked imbalance-handling strategies improve
multiclass imbalanced tabular classification, and whether their effects vary
with imbalance ratio, class count, dimensionality, and feature type. The
benchmark is an empirical comparison; it does not introduce a new algorithm
and does not pool the portfolio case studies into the analysis.

The classifiers are Random Forest, Logistic Regression, and XGBoost. The
conditions and all parameters are exactly those in `config/stage0.yaml`:

- raw baseline;
- class-weighted classifier using the balanced class-weight rule;
- random over-sampling and random under-sampling;
- SMOTE, ADASYN, and Borderline-SMOTE for numeric data;
- SMOTENC, SMOTEENN, and SMOTETomek for mixed data, with SMOTENC inside the
  combination samplers;
- Balanced Random Forest as an RF-only comparator.

No classifier, sampler, metric, tuning step, or parameter is added after this
freeze without a documented amendment.

## Data preparation and folds

The registry and acquisition manifest are the data source of truth. The raw
file hash, source metadata, target semantics, class counts, feature type,
missingness, duplicate status, leakage status, and split policy are retained
with every dataset. Semantic categorical overrides are read from
`data/dataset_feature_overrides.csv`.

Rows with a missing target are removed before splitting. Predictor missingness
is handled inside the fold pipeline. No domain-specific cleaning, row
deduplication, feature selection, or tuning is performed after scope lock.
The target is label-encoded for the classifier; this encoding uses labels only
and is not a predictor transform.

Use `StratifiedKFold(n_splits=5, shuffle=True)` with seeds `0`, `1`, and `2`.
Every dataset row is assigned to one test fold per seed. Datasets with groups
or time order are excluded from the core benchmark under the locked registry
decisions.

For every fold, fit imputation, ordinal encoding, one-hot encoding, scaling,
resampling, and training-fold weights on the training rows only. The test fold
is passed through fitted transforms for prediction and is never resampled or
used to fit any transform.

Numeric pipelines impute with the training median and scale after imputation.
Mixed pipelines impute numeric values with the training median, impute
categorical values with the training most-frequent value, ordinal-encode
before SMOTENC, and one-hot encode after resampling with unknown categories
ignored. Categorical-only data remains excluded until SMOTEN is implemented.

## Applicability and complete cells

Ordinary SMOTE-family samplers are never applied to one-hot categorical
features. Unsupported dataset-condition-classifier combinations are written
to the failure log with their stage and error class; they are not converted to
zero scores or silently omitted.

A required cell is one dataset, classifier, condition, seed, and fold. A
dataset-condition-classifier cell is complete only when all 15 seed-fold
records are valid. A failure in any one of those records makes the cell
incomplete for inferential analysis; its valid records remain in the raw audit
and the failure is reported.

The analysis uses dataset-level means of complete cells. Folds and seeds are
never treated as independent datasets. For each metric and analysis family,
include only dataset blocks that have a complete cell for every condition in
that family and the selected classifier. This rule is applied mechanically
from validity and applicability, never from scores or rankings. If fewer than
three dataset blocks remain, report the comparison as skipped with its reason.

The pre-specified analysis families are:

| Family | Conditions | Classifiers / blocks |
|---|---|---|
| Numeric core | `raw`, `class_weighted`, `random_over`, `random_under`, `smote`, `adasyn`, `borderline_smote`, `smoteenn`, `smotetomek` | Each of the three classifiers; numeric datasets only |
| Mixed core | `raw`, `class_weighted`, `random_over`, `random_under`, `smotenc`, `smoteenn`, `smotetomek` | Each of the three classifiers; mixed datasets only |
| Balanced RF | `raw`, `class_weighted`, `balanced_random_forest` | Random Forest only; numeric and mixed datasets analysed separately |

Classifier-to-classifier comparisons are descriptive only. They do not pool
model families into a single Friedman block.

## Outcomes and exact metric rules

Macro-F1 is the primary outcome. For each test fold, compute
`f1_score(average="macro", zero_division=0)` over all encoded target classes.

The secondary outcomes are multiclass G-mean, MCC, balanced accuracy, and
per-class recall. Let `R_c` be recall for declared class `c`, computed with
`zero_division=0`. G-mean is:

\[
G = \left(\prod_{c=1}^{C} R_c\right)^{1/C}.
\]

All classes in the dataset's encoded label set are included. A class with
zero recall therefore makes G-mean exactly zero; no epsilon, omission, or
imputation is used. MCC is `matthews_corrcoef`, balanced accuracy is the
unweighted mean class recall, and per-class recall uses the same declared
labels and zero behavior.

For every valid record, retain the metric values, dataset, feature type,
classifier, condition, seed, fold, training rows before and after sampling,
fit/predict runtime, and memory measurement fields. A failed record retains
the same cell key plus failure stage, exception type, and message.

## Aggregation and descriptive comparisons

First average each metric over the 15 valid seed-fold records within a complete
dataset-condition-classifier cell. Then compare conditions within the same
dataset and classifier.

For every non-baseline condition, report its paired delta from `raw`, defined
as the condition mean minus the raw mean for the same dataset, classifier, and
metric. Report the mean and median paired delta, the number of complete pairs,
and win/tie/loss counts. A win is a delta greater than `1e-12`, a loss is less
than `-1e-12`, and all other deltas are ties.

Runtime, peak RSS, and post-sampling row counts are descriptive efficiency
evidence. They do not enter the primary performance tests.

## Confirmatory statistical analysis

Use two-sided tests with `alpha = 0.05`. The statistical block is the dataset,
and the observations supplied to every test are dataset-level means.

1. For each feature family, classifier, and metric, run the Friedman test over
   the complete dataset-by-condition matrix when there are at least three
   blocks and three conditions.
2. If the Friedman result is significant, run the pre-specified Nemenyi
   post-hoc test over the same conditions and blocks. Do not replace it with a
   score-selected subset.
3. For the limited baseline comparisons, run two-sided paired Wilcoxon
   signed-rank tests for `raw` versus each alternative condition in the same
   family. Use `zero_method="wilcox"`, no continuity correction, and
   `method="auto"`; discard zero differences as defined by that method.
4. Apply Holm correction across all raw-versus-alternative Wilcoxon p-values
   within each feature-family, classifier, and metric family. Nemenyi's own
   all-pairs correction is retained. Do not make a cross-metric familywise
   claim.

Macro-F1 is the sole primary confirmatory outcome. Secondary metrics are
reported with the same declared procedure but are interpreted as secondary
evidence. A skipped test records the exact reason, such as too few blocks,
too few conditions, or an incomplete matrix.

## Effect sizes and intervals

For each paired condition-versus-raw comparison, report the mean paired delta
and the paired rank-biserial correlation. The latter is computed after zero
difference pairs are removed as:

\[
r_{rb} = \frac{W^+ - W^-}{n(n+1)/2},
\]

where `W+` and `W-` are the sums of positive and negative signed ranks and
`n` is the number of non-zero pairs. If every pair is zero, report `r_rb = 0`.

Use a paired percentile bootstrap over complete dataset blocks for a 95%
confidence interval around the mean delta: 10,000 resamples, sampling blocks
with replacement, NumPy generator seed `0`, and the 2.5th and 97.5th
percentiles. Do not bootstrap folds or seeds independently. Report the number
of dataset blocks used for each interval.

## Exploratory moderators and claims

Imbalance ratio, class count, raw and encoded dimensionality, and feature type
are exploratory moderators. Report their distributions and effect patterns
without using them to select datasets, methods, or post-hoc tests. With eleven
datasets, moderator findings are descriptive and do not establish a general
selection rule.

The paper may claim only what the complete dataset-level evidence supports.
It must report negative results, failed cells, skipped tests, uncertainty,
runtime, memory, and resampling consequences. It must not claim a universally
best method, causal superiority, or real-world performance from Stage 0
pilot outputs. No full-run result exists until the locked benchmark is
executed after this plan is committed.

## Reproducibility and release boundary

The full run must preserve the locked configuration, registry and manifest
hashes, environment record, raw-file hashes, seed/fold keys, output schema,
failure logs, and resource evidence. Generated full results remain ignored
until their completeness and hashes are reviewed. This protocol package does
not execute the benchmark, select a method from pilot scores, or alter the
Stage 0 scope.
