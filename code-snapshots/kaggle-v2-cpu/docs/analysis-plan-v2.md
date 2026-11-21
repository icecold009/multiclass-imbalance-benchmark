# V2 Bayesian Analysis Plan

**Analysis-plan version:** `expanded-v2-analysis-1`
**Protocol:** [`docs/protocol-v2.md`](protocol-v2.md)
**Status:** Gate 1 freeze; run only after the V2 configuration and registry are locked
**Primary statistical unit:** Dataset

## 1. Analysis boundary

V2 is analysed as a new benchmark. V1 outputs, V1 hypotheses, and V1
confirmatory tests remain historical evidence and are not pooled with V2.
The V2 analysis package must be generated only from records produced under the
hash-locked V2 configuration, environment, and dataset registry.

No pilot, smoke, or engineering result may select a dataset, model, sampler,
hyperparameter, contrast, or manuscript claim. Engineering results are
reported as validation evidence and excluded from inferential tables.

## 2. Estimands and records

For dataset `d`, outer fold `f`, classifier `m`, condition `c`, and metric `q`,
the raw evaluation record contains the untouched outer-test prediction and all
fold-local provenance. Inner-CV trial records are nested under the outer
training record and are not treated as independent observations.

The primary estimand for a declared pair `(A, B)` is the population mean
macro-average-precision difference:

\[
\Delta_{A-B} = \mathrm{E}_{d}[\mathrm{MacroAP}_{d,A} - \mathrm{MacroAP}_{d,B}].
\]

The analysis unit is the dataset. Outer folds provide repeated observations
within a dataset and are correlated; they are not independent datasets.

For a contrast and a classifier/feature-family analysis block, include a
dataset only when both conditions have valid records for all five outer folds.
If fewer than three complete dataset blocks remain, mark the contrast
`SKIPPED` with an explicit denominator reason.

## 3. Metric definitions

### Primary metric

For every class `c`, compute one-vs-rest average precision from the model's
untouched probability for class `c`. The primary metric is:

\[
\mathrm{MacroAP} = \frac{1}{C}\sum_{c=1}^{C} AP_c.
\]

Use `macro_average_precision` consistently in code, schemas, tables, and the
manuscript. Do not call this numerically integrated AUPRC unless a separate
curve integral is explicitly reported.

### Secondary metrics

Report:

- per-class average precision;
- macro-F1 under default argmax;
- macro-F1 under validation-calibrated decisions;
- G-mean, MCC, balanced accuracy, and per-class recall under default argmax;
- multiclass Brier score; and
- multiclass log loss.

The Brier score is:

\[
BS = \frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}(p_{ic}-y_{ic})^2.
\]

It is computed from original probabilities before threshold calibration and is
described as a joint probabilistic-accuracy and calibration measure, not as a
pure calibration statistic.

## 4. Decision calibration

The probability track uses untouched probabilities for average precision,
Brier score, and log loss. The decision track reports ordinary argmax and the
following calibrated rule:

\[
\hat y = \arg\max_c \frac{p_c}{t_c}.
\]

Threshold offsets are fitted only from inner-CV out-of-fold validation
predictions. The lowest encoded class is the reference with offset zero; the
remaining log offsets are bounded to `[-2, 2]` and optimized with
`scipy.optimize.differential_evolution` using the frozen seed and settings in
`config/v2.yaml`.

The objective is macro-F1. Ties select the lexicographically smallest offset
vector. Invalid class support, non-finite probabilities, optimizer failure,
or a missing class causes deterministic fallback to ordinary argmax and an
explicit calibration-status record. The outer test fold never participates in
threshold fitting.

## 5. Bayesian hierarchical correlated model

For each declared pairwise contrast, form the five outer-fold paired
differences for every complete dataset block:

\[
\delta_{d,f} = q_{d,f,A} - q_{d,f,B}.
\]

Let `F = 5`. Model the fold vector for dataset `d` as:

\[
\boldsymbol{\delta}_d
\sim
\mathrm{MVNormal}\left((\mu + u_d)\mathbf{1}_F, \Sigma_{fold}\right),
\]

where:

\[
u_d \sim \mathrm{Normal}(0,\tau_{dataset}),
\]

and:

\[
\Sigma_{fold}
=
\sigma_{fold}^2\left((1-\rho)I_F + \rho\mathbf{1}_F\mathbf{1}_F^T\right).
\]

This exchangeable covariance treats the outer folds as correlated repeated
measurements within a dataset while the dataset random effect represents
dataset-to-dataset variation. The primary estimand is `mu`, the population
mean method difference.

Use these priors:

| Parameter | Prior |
|---|---|
| `mu` | `Normal(0, 0.5)` |
| `tau_dataset` | `HalfNormal(0.5)` |
| `sigma_fold` | `HalfNormal(0.25)` |
| `rho` | `Beta(2, 2)` |

Fit each declared contrast with PyMC NUTS using four chains, 2,000 tuning
draws, 2,000 posterior draws per chain, `target_accept=0.90`, and random seed
`20260917`. Run one model per contrast and analysis block so that missing
applicability is never silently imputed.

### Diagnostics

A contrast is diagnostically acceptable only when:

- maximum `r_hat <= 1.01`;
- bulk effective sample size is at least 400 for reported parameters;
- tail effective sample size is at least 400; and
- there are zero divergent transitions.

Diagnostic failure is reported as `ANALYSIS_FAILED` and does not become a
scientific conclusion. The posterior sample and sampler metadata remain in
the analysis evidence directory for review.

## 6. Declared contrasts

Generate the following before inspecting full-run scores:

1. every applicable imbalance condition versus `raw`;
2. `class_weighted` versus each applicable sampling condition;
3. Balanced Random Forest versus each applicable ordinary RF intervention; and
4. the fixed sampler pairs in `config/v2.yaml`.

A contrast is evaluated only where both conditions are applicable to the same
dataset, classifier, and feature family. There is no cross-family pooling of
numeric-only and mixed-only samplers.

For each contrast report:

- complete dataset-block count and excluded-block count with reasons;
- posterior mean and median of `mu`;
- 95% credible interval;
- `P(mu < -0.01)`;
- `P(-0.01 <= mu <= 0.01)`; and
- `P(mu > 0.01)`.

## 7. ROPE interpretation

The primary practical-equivalence region is:

\[
ROPE_{MacroAP} = [-0.01, +0.01].
\]

Use these interpretations:

- practically worse: posterior difference below `-0.01`;
- practically equivalent: posterior difference within the ROPE; and
- practically better: posterior difference above `+0.01`.

Use calibrated language such as “the posterior probability that method A was
practically better than method B was 0.87.” Do not use “statistically
significant,” “failed to reject,” or “proved identical.”

## 8. Descriptive moderators and resource outcomes

Report descriptive performance patterns by:

- imbalance-ratio band;
- numeric versus mixed feature representation;
- class count;
- row count;
- raw dimensionality;
- missingness; and
- categorical proportion.

Moderators are not used to select datasets or contrasts after the freeze.
Runtime, peak RSS, worker count, timeout status, and rows before/after
sampling are descriptive resource outcomes and do not enter the primary
performance estimand.

## 9. Failure and missing-data rules

Every expected dataset/fold/classifier/condition key must have exactly one
`VALID` or `FAILED / NOT APPLICABLE` record. A failed record preserves its
stage, exception type, reason, applicability, runtime, resource status, retry
count, and configuration hash.

No failed record is converted to a zero metric. No method is ranked on a
success-only denominator without its coverage being shown. Analysis tables
must expose the expected, valid, failed, and not-applicable counts.

Infrastructure-only retries may occur once under the same frozen inputs.
Protocol failures are not retried with modified parameters. Any change to
data, code, dependencies, search spaces, resource policy, or analysis requires
a dated amendment and a new reproducibility checkpoint.

## 10. Reproducible analysis outputs

The analysis runner must produce:

- `analysis_manifest.json` with protocol/config/environment/input hashes;
- complete-cell and per-class metric tables;
- baseline deltas and coverage tables;
- posterior samples or an auditable posterior summary per contrast;
- diagnostics and skipped/failed-analysis records;
- ROPE probability tables;
- figures generated only from the locked analysis tables; and
- a machine-readable list of manuscript-ready claims and their evidence rows.

The final manuscript must distinguish protocol decisions, observed results,
interpretations, and limitations. No manuscript section may report smoke,
pilot, or incomplete-run output as V2 benchmark evidence.

## 11. Amendment and release rule

This plan is frozen with `config/v2.yaml` and the environment lock. A material
change requires a dated amendment that records the reason, affected artifacts,
whether previous results remain comparable, and the expected schedule impact.
No full-run execution may begin while the decision register, configuration,
environment, registry schema, or analysis plan is unresolved.
