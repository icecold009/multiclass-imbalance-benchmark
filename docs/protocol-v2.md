# V2 Expanded Multiclass Imbalance Benchmark Protocol

**Protocol ID:** `expanded-v2`
**Status:** Proposed; all entries marked *must lock* are prerequisites for outcome-bearing runs
**V1 boundary:** The completed V1 benchmark, paper, configuration, registry, and evidence package remain frozen
**V2 development branch:** `experiment/v2-expanded-benchmark`
**Statistical unit:** Dataset, with outer-fold dependence retained within dataset

## 1. Purpose and scope

V2 is a new, expanded benchmark. It is not a rewrite of the completed V1
experiment and must not overwrite, reinterpret, or pool the V1 evidence.

The benchmark will test which optimized imbalance-handling pipelines perform
best for multiclass imbalanced tabular classification under equal tuning
budgets. It will also measure how results vary with imbalance severity,
feature representation, problem complexity, probability quality, and resource
cost.

The current V1 repository and paper remain the reproducible V1 result. The
V2 branch may introduce a new dataset registry, model pool, preprocessing
paths, tuning policy, metrics, and analysis framework, but every such change
must be locked before the full run. No V2 result may be used to amend V1
retroactively.

### Research questions

1. Which optimized imbalance-handling pipeline performs best under equal
   hyperparameter-search budgets?
2. Do the effects of weighting, sampling, and specialized ensembles vary by
   imbalance severity, feature type, class count, dimensionality, missingness,
   or dataset size?
3. Do imbalance interventions change probability quality or the difference
   between default and validation-calibrated decisions?
4. What performance, failure, runtime, and memory trade-offs accompany each
   method?

### Non-negotiable invariants

- Keep V1 unchanged and retain its provenance, hashes, failure accounting, and
  reproducibility gates.
- Freeze the V2 protocol, analysis plan, dataset registry, environment, and
  search spaces before inspecting outcome-bearing full-run results.
- Fit imputation, encoding, scaling, resampling, threshold rules, tuning, and
  training-derived weights only within the applicable training data.
- Never use an outer test fold to select a model, threshold, parameter, or
  dataset.
- Record every expected result as `VALID` or `FAILED / NOT APPLICABLE`; never
  silently omit a failed cell or replace it with a zero metric.
- Do not select datasets, methods, or claims from pilot scores.
- Do not change the frozen dataset or protocol after scope lock except through
  a dated, documented amendment that records its effect on the timeline and
  evidence boundary.
- Do not use confirmatory p-value language for V2 Bayesian inference.

## 2. V2 design at a glance

| Component | Locked V2 direction |
|---|---|
| Dataset corpus | 40–60 public tabular datasets; target approximately 50 |
| Minimum classes | 3 |
| Minimum minority support | 10 observations in the full dataset |
| Feature coverage | Numeric and mixed numeric/categorical datasets |
| Classifiers | Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost |
| Specialized comparator | Balanced Random Forest, treated as an RF-only comparator |
| Conditions | Raw, class weighting, random over-sampling, random under-sampling, SMOTE, ADASYN, Borderline-SMOTE, SMOTENC, SMOTEENN, SMOTETomek |
| Synthetic-neighbour rule | Fold-local `k = min(5, N_minority,target - 1)` |
| Outer evaluation | One frozen five-fold stratified split |
| Inner optimization | Three-fold stratified split within each outer training fold |
| HPO budget | 20 randomized configurations for every comparable pipeline |
| HPO score | Macro average precision |
| Primary inferential metric | Macro average precision, operationalized as one-vs-rest per-class average precision averaged across classes |
| Decision metrics | Macro-F1 with default argmax and validation-calibrated decisions reported separately |
| Probability quality | Multiclass Brier score; log loss only if locked before the full run |
| Primary inference | Bayesian hierarchical correlated comparison across datasets and outer-CV observations |
| Practical-equivalence region | `[-0.01, +0.01]` macro average precision |
| Confirmatory p-values | Removed from V2 |
| Runtime exclusion | Removed after registry freeze; timeouts remain explicit failures |
| V1 | Preserved unchanged |

The ten ordinary conditions are `raw`, `class_weighted`, `random_over`,
`random_under`, `smote`, `adasyn`, `borderline_smote`, `smotenc`, `smoteenn`,
and `smotetomek`. Balanced Random Forest is a separate RF-specific comparator,
not a universal condition available to every classifier.

## 3. Protocol artifacts and lock sequence

Create and version the following artifacts before the V2 full run:

| Artifact | Role | Lock point |
|---|---|---|
| `docs/protocol-v2.md` | Complete experimental protocol | Protocol freeze |
| `config/v2.yaml` | Models, conditions, folds, metrics, search spaces, seeds, and resource policy | Protocol freeze |
| `data/dataset_registry_v2.csv` | Final eligible dataset corpus and provenance | Dataset freeze |
| `docs/analysis-plan-v2.md` | Bayesian model, contrasts, ROPE, summaries, and reporting language | Analysis freeze |
| `requirements-v2.txt` and lockfile | Exact runtime environment | Environment freeze |
| `results/v2-smoke/` | Engineering validation outputs only; excluded from scientific evidence | Pipeline freeze |
| `results/v2-full/` | Raw full-run records, manifests, and failure logs | Full-run freeze |
| `results/v2-analysis/` | Aggregates, posterior summaries, figures, and review manifests | Analysis freeze |

Generated results remain separate from source and are not used to modify the
protocol. Each major checkpoint may be tagged as:

`v2-protocol-freeze` → `v2-dataset-freeze` → `v2-pipeline-freeze` →
`v2-full-run` → `v2-analysis-freeze`

## 4. Gated workplan

| Stage | Required work | Exit evidence | Blocker rule |
|---|---|---|---|
| 1. Protocol and analysis freeze | Complete this protocol, `config/v2.yaml`, analysis plan, environment specification, and decision register | All must-lock choices are resolved and hashed before outcome-bearing execution | No full run while any protocol, search-space, or analysis choice is unresolved |
| 2. Engineering validation | Run synthetic datasets and unit/integration tests for leakage, samplers, weights, categorical paths, thresholds, metrics, determinism, and cell accounting | All required tests pass; failures are classified and reproducible | Stop and repair the harness; do not use the V2 registry as a debugging fixture |
| 3. Dataset registry and coverage freeze | Acquire, audit, classify, deduplicate, and freeze 40–60 eligible datasets | Registry, source metadata, hashes, licenses, coverage quotas, and exclusions are complete | Do not replace a dataset because its pilot score is inconvenient |
| 4. Pipeline, tuning, and smoke-run freeze | Implement fold-local pipelines, equal HPO budgets, resumability, resource logging, and a data-free/engineering smoke run | Complete output schema, deterministic replay, and resource policy pass | Do not begin the full run with schema or replay defects |
| 5. Full execution and failure accounting | Run every frozen dataset/fold/classifier/condition cell with bounded resources | Every expected cell has a valid or explicit failure/applicability record | Re-run only documented infrastructure failures; do not silently change methods |
| 6. Bayesian analysis and manuscript revision | Freeze aggregates, fit the declared Bayesian model, generate figures, and rewrite claims from V2 evidence | Analysis manifest, posterior outputs, figures, limitations, and manuscript review pass | Do not rewrite V1 Results or make claims from smoke/pilot outputs |

Every stage must record its inputs, outputs, validation commands, unresolved
issues, and next safe action. A stage is not complete because a command ran;
its exit evidence must pass the stated gate.

## 5. Dataset registry and coverage

### 5.1 Eligibility rules

An included dataset must be:

- tabular and supervised;
- multiclass with at least three classes;
- publicly obtainable with documented license or redistribution terms;
- documented at the target and label-semantics level;
- non-time-series for the core benchmark;
- non-grouped unless groups can be handled explicitly;
- free from unresolved target leakage, duplicate/repackaged overlap, or
  unexplained identifier features; and
- supported by at least 10 observations in the full minority class.

The support floor is a V2 scope rule, not a claim that every retained dataset
has equal statistical power. Registry review must also examine whether each
class remains usable under the declared five-by-three nested splits and the
sampler applicability rules.

Candidate sources may include OpenML-CC18, suitable multiclass OpenML tasks,
CLIMB datasets, and other documented public tabular benchmark suites. Source
membership alone is never sufficient for inclusion.

Each registry record must preserve, at minimum:

- source identifier, URL, version, license, retrieval metadata, and file hash;
- target field, target semantics, class counts, number of classes, and minority
  support;
- raw feature count, dimensionality, missingness, categorical proportion, and
  feature-type classification;
- imbalance ratio and its predeclared severity band;
- duplicate, group, time-order, and leakage review outcomes;
- inclusion/exclusion decision, rationale, and reviewer evidence.

### 5.2 Coverage quotas

Do not select approximately 50 convenient datasets from one source. Before
the dataset freeze, define and report coverage across:

| Dimension | Required coverage |
|---|---|
| Feature representation | Numeric and mixed numeric/categorical |
| Imbalance ratio | Near balanced `<2`, mild `2–5`, moderate `5–20`, severe `>20` |
| Class structure | Multiple class counts, always at least three |
| Dataset size | Small, medium, and larger tasks where eligible |
| Dimensionality | Low and high dimensionality |
| Missingness | Meaningful variation, including missing-free and missing-feature tasks where available |
| Categorical proportion | Multiple mixed-data compositions |

Versions of the same underlying dataset do not count as independent tasks.
The registry must identify and consolidate repackaged duplicates before the
scope lock.

### 5.3 Compute policy

Runtime is not a dataset eligibility criterion after the registry is frozen.
If an execution exceeds the declared limit, terminate the affected fit, retain
the dataset, and record an explicit timeout in the failure denominator.

The execution manifest must record:

- cloud provider/instance specification and available Student Developer Pack
  credit boundary;
- CPU count, RAM, operating system, Python and library versions;
- worker count, timeout value, wall-clock runtime, and peak memory;
- dataset/job completion markers and resumability state.

Use CPU execution initially. XGBoost, LightGBM, and CatBoost must use
comparable CPU budgets unless GPU use is explicitly added through a protocol
amendment.

## 6. Pipeline and algorithm specification

### 6.1 Dynamic synthetic-sampler neighbours

For every actual training split, including every inner-CV training split,
compute:

\[
k = \min\left(5, N_{\mathrm{minority,target}} - 1\right),
\]

where `N_minority,target` is the smallest class count among the classes the
sampler is attempting to synthesize in that training split.

Apply the fold-local calculation to:

- SMOTE;
- ADASYN;
- Borderline-SMOTE;
- SMOTENC;
- the SMOTE component of SMOTEENN; and
- the SMOTE component of SMOTETomek.

If the relevant class has fewer than two training examples, record the
sampler-fold as `FAILED / NOT APPLICABLE` with an explicit reason. Never crash,
silently substitute a different method, or calculate `k` globally. Bound
Borderline-SMOTE's additional neighbourhood parameter against the available
fold size using the same fold-local policy.

### 6.2 Fit-time class weighting

V1 already computes balanced XGBoost weights from the training labels and
routes them as `classifier__sample_weight`. V2 must retain that correctness
while moving the behavior into a reusable estimator or fit wrapper so that
weights are recomputed for every individual inner-CV fit:

\[
w_i = \texttt{compute\_sample\_weight("balanced", y_{\mathrm{current\ fit}})}.
\]

Under `class_weighted`, use fit-time instance weights for XGBoost, LightGBM,
and CatBoost. Retain the balanced class-weight behavior for Logistic
Regression and Random Forest. Tests must prove that no outer-test labels reach
weight computation.

### 6.3 Model pool

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Standard bagged-tree baseline |
| XGBoost | Established gradient-boosted-tree baseline |
| LightGBM | Modern gradient-boosted-tree baseline |
| CatBoost | Modern categorical-aware gradient-boosted-tree baseline |
| Balanced Random Forest | RF-specific imbalance comparator |

Balanced Random Forest is not a general sampler condition and is not fitted as
an intervention for Logistic Regression, XGBoost, LightGBM, or CatBoost.

### 6.4 Model-specific preprocessing

Do not force every classifier through one representation.

| Model/data path | Required representation |
|---|---|
| Logistic Regression, numeric | Training-fold imputation → scaling → classifier |
| Logistic Regression, mixed | Training-fold imputation → ordinal coding → SMOTENC where applicable → one-hot encoding → numeric scaling → classifier |
| Random Forest, numeric | Training-fold imputation → classifier; no scaling |
| Random Forest, mixed | Training-fold imputation → declared ordinal/one-hot representation → applicable sampler → classifier; no scaling |
| XGBoost, LightGBM, CatBoost, numeric | Training-fold imputation → native numeric features → GBDT |
| XGBoost, LightGBM, CatBoost, mixed raw/weighted/ROS/RUS | Training-fold imputation → preserve categorical identity → native categorical GBDT |
| XGBoost, LightGBM, CatBoost, mixed synthetic | Training-fold imputation → categorical integer representation → SMOTENC or combination sampler → restore categorical columns → native categorical GBDT |

Do not one-hot encode mixed data before CatBoost or LightGBM. V2 also uses
native categorical handling for XGBoost so that the three GBDTs are evaluated
with model-appropriate representations rather than giving one model an
OHE-specific compatibility path.

Ordinary SMOTE-family samplers must never be applied to one-hot categorical
features. Mixed synthetic paths must preserve categorical integrity after
resampling, and unknown validation categories must be handled explicitly by
the locked encoder policy.

## 7. Nested evaluation and tuning

### 7.1 Outer and inner splits

Use one frozen outer split per dataset:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=<frozen-seed>)
```

Within every outer training fold, use:

```python
StratifiedKFold(n_splits=3, shuffle=True, random_state=<frozen-seed>)
```

V2 deliberately uses one five-fold outer split rather than V1's three-seed
by-five-fold structure. The expanded dataset corpus supplies more independent
statistical blocks; the trade-off must be documented in the protocol.

### 7.2 Equal hyperparameter budgets

For every complete combination of:

`dataset × outer fold × classifier × imbalance condition`

run 20 randomized hyperparameter configurations inside the outer training
data. Score configurations with macro average precision. Use the same search
budget for every comparable model and condition, with search spaces frozen in
`config/v2.yaml` before the full run.

Tune classifier parameters only. Do not tune sampler parameters from observed
benchmark results; the fold-safe dynamic-neighbour rule is predetermined.

Search-space families are:

- GBDTs: estimators, learning rate, depth/leaves, minimum child size,
  subsampling, feature subsampling, L1, and L2 regularization;
- Random Forest: number of trees, depth, minimum leaf size, and max features;
- Logistic Regression: regularization strength and supported L1/L2 choices.

The exact ranges, distributions, seeds, and compatibility constraints are
*must lock* entries in the decision register. Do not optimize a classifier on
raw data and reuse it for every sampling condition; resampling changes the
training distribution and may change the optimum.

## 8. Probability and decision evaluation

Maintain separate probability/ranking and decision tracks.

### 8.1 Probability/ranking track

Use untouched model probabilities for:

- macro average precision;
- per-class average precision;
- multiclass Brier score; and
- optional multiclass log loss, only if included in the locked analysis plan.

No threshold tuning may affect these metrics.

### 8.2 Decision track

Report both:

1. ordinary multiclass `argmax`; and
2. a validation-calibrated decision rule.

The calibrated rule uses class-specific thresholds or offsets:

\[
\hat{y} = \arg\max_c \frac{p_c}{t_c},
\]

where `t_c` is learned exclusively from validation predictions and optimized
for macro-F1, not average precision.

For every outer fold, follow this order:

```text
outer training data
  → inner CV
  → out-of-fold validation probabilities
  → optimize thresholds for macro-F1
  → refit the selected pipeline on complete outer training data
  → evaluate once on the untouched outer test fold
```

The outer test fold must never influence threshold selection. Exact threshold
parameterization, optimizer constraints, tie handling, and failure behavior
are *must lock* entries in the decision register.

## 9. Metrics and reporting

### 9.1 Primary outcome

The primary V2 inferential outcome is macro average precision, operationally
computed as one-vs-rest average precision for each class followed by an
unweighted mean:

\[
\mathrm{Macro\text{-}AP}
= \frac{1}{C}\sum_{c=1}^{C} AP_c.
\]

Use the term **macro average precision** in implementation and manuscript
text. Reserve numerical **AUPRC** for a separately integrated precision-recall
curve; do not use the labels interchangeably. Retain every per-class AP value.

### 9.2 Secondary outcomes

Retain:

- macro-F1 under default argmax;
- macro-F1 under the calibrated decision rule;
- G-mean;
- Matthews correlation coefficient;
- balanced accuracy; and
- per-class recall.

Add multiclass Brier score and per-class average precision. Brier score is
computed from original probabilities before threshold calibration:

\[
BS = \frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}(p_{ic}-y_{ic})^2.
\]

Brier score measures probabilistic accuracy as well as calibration; do not
describe it as a pure calibration metric. Every reported metric must state
its probability source, decision rule, class averaging, and failure behavior.

## 10. Bayesian analysis plan

### 10.1 Inference framework

Remove Friedman, Nemenyi, Wilcoxon, Holm-adjusted confirmatory testing, and
confirmatory p-values from the V2 framework. Descriptive ranks may be shown,
but they are not the inferential conclusion.

Use a Bayesian hierarchical correlated model over dataset-level comparisons
and outer-CV observations. For each declared comparison, estimate the
posterior of:

\[
\Delta = \mathrm{method\ A} - \mathrm{method\ B}.
\]

The hierarchy must account for dataset-to-dataset variation and dependence
among outer folds within a dataset. Folds are never treated as independent
datasets.

Declare pairwise comparisons before the full run, including:

- every imbalance method versus raw;
- weighting versus sampling;
- Balanced Random Forest versus ordinary RF interventions; and
- selected sampler-versus-sampler comparisons defined in the analysis plan.

The exact likelihood, priors, correlation structure, posterior computation,
diagnostic thresholds, and missing/failure handling are *must lock* entries in
the decision register.

### 10.2 ROPE and reporting language

Pre-register the primary practical-equivalence region:

\[
ROPE_{\mathrm{macro\text{-}AP}} = [-0.01, +0.01].
\]

For every primary contrast report:

\[
P(\Delta < -0.01),\qquad
P(-0.01 \leq \Delta \leq 0.01),\qquad
P(\Delta > 0.01),
\]

alongside posterior mean, posterior median, and 95% credible interval.

Use the labels practically worse, practically equivalent, and practically
better. Do not use “statistically significant,” “failed to reject,” or “proved
identical.”

## 11. Failure, applicability, and completeness accounting

Each expected execution must resolve to exactly one of:

- `VALID`; or
- `FAILED / NOT APPLICABLE`.

At minimum, each record must identify:

| Field group | Required content |
|---|---|
| Cell identity | Dataset, outer fold, classifier, condition, and inner configuration where relevant |
| Outcome state | `VALID` or `FAILED / NOT APPLICABLE` |
| Failure detail | Stage, exception type, reason, and whether the failure is protocol-defined or infrastructure-related |
| Resource evidence | Runtime, timeout status, worker count, memory/resource status |
| Provenance | Configuration/environment/input references and deterministic seed information |
| Valid result | Metrics, class/probability metadata, training rows before/after sampling, and prediction status |

Do not substitute zero metric values for failures. Do not rank methods using
only the easiest datasets on which they succeeded. Report method coverage,
valid-cell counts, failure counts, applicability counts, and the denominators
alongside performance.

Resumability must be dataset-granular. Re-run only documented infrastructure
failures under the same frozen configuration. Any code, dependency, dataset,
resource-policy, or search-space change requires a new checkpoint or formal
amendment rather than an unrecorded retry.

## 12. Engineering validation before the full run

Use synthetic fixtures, excluded from scientific evidence, covering:

- balanced numeric data;
- severe imbalance;
- a minority class with two observations;
- mixed numeric/categorical data;
- unseen validation categories;
- missing categorical and numeric values;
- many classes;
- high dimensionality; and
- a sampler that produces no new samples.

Tests must verify:

- no test-fold leakage;
- fold-local dynamic `k` and explicit fewer-than-two behavior;
- fit-time weights for XGBoost, LightGBM, and CatBoost;
- correct model-specific categorical paths;
- SMOTENC category integrity;
- independence of inner and outer splits;
- threshold fitting only from validation predictions;
- probabilities summing to one;
- macro average precision and Brier calculations;
- complete-cell and failure accounting; and
- deterministic replay under frozen seeds.

Only after these tests pass may the frozen V2 registry be run.

## 13. Compute, execution, and provenance

The research separation comes from Git; GitKraken is optional as a local
branch-management interface. The recommended structure is:

```text
main
└── frozen V1 evidence package

experiment/v2-expanded-benchmark
└── V2 protocol, implementation, and controlled evidence
```

Parallelize across independent dataset/fold jobs while keeping each model at
a controlled worker count so timing and memory remain interpretable. Every
dataset must be independently resumable. Every output must be traceable to:

- the frozen protocol/configuration and its hash;
- the dataset registry and raw-input hashes;
- the environment lock and hardware record;
- dataset, fold, condition, classifier, and seed keys; and
- the execution status and resource log.

## 14. Manuscript revision boundary

Do not rewrite the V1 Results section before the V2 run finishes. The final
paper must be rewritten from the V2 evidence package, while preserving the
V1 result as a separate historical and reproducibility boundary.

### Related Work

Cover, as relevant:

- TabArena for modern tabular benchmark methodology and model diversity;
- CLIMB for class-imbalanced tabular learning;
- modern class-imbalance benchmark literature;
- Balanced Random Forest, CatBoost, and LightGBM;
- synthetic resampling versus ensemble interventions;
- probability-calibration effects of resampling; and
- Bayesian classifier-comparison methodology.

### Results order

Organize the V2 Results around:

1. dataset coverage and failures;
2. primary macro average precision;
3. posterior method comparisons;
4. practical-equivalence probabilities;
5. classifier-by-intervention interactions;
6. performance by imbalance severity;
7. numeric versus mixed datasets;
8. probability-quality effects;
9. default versus calibrated decisions; and
10. runtime and memory cost.

Do not make a single global leaderboard the main conclusion.

### SMOTEENN and ENN interpretation

The V1 observation that SMOTEENN reduced Logistic Regression MCC is a
hypothesis-generating result, not proof of a causal mechanism. If V2
reproduces the pattern across substantially more datasets, discuss the
possibility that ENN changes local class density and decision-boundary
geometry in a way that is poorly matched to a single global linear surface.
Use qualified, observational language and do not claim that ENN “destroys
global linear separability.”

## 15. Decision register: must lock before protocol freeze

The following items are intentionally surfaced rather than silently chosen.
Resolve each in `config/v2.yaml` or `docs/analysis-plan-v2.md`, record the
rationale, and hash the result before the dataset or full-run freeze.

| Decision | Required resolution |
|---|---|
| Bayesian likelihood and priors | Specify the response model, priors, dataset-level and fold-level correlation structure, posterior sampler, diagnostics, and convergence thresholds |
| Primary contrast matrix | List every confirmatory/primary pairwise contrast and the exact compatible dataset/model family for each |
| Threshold optimizer | Fix parameterization, bounds, optimizer, initialization, ties, class-support failures, and deterministic seed |
| Native categorical compatibility | Pin XGBoost, LightGBM, and CatBoost versions and prove that each declared mixed-data path is supported by tests |
| HPO search spaces | Freeze exact parameter names, ranges, distributions, trial seeds, invalid-combination handling, and 20-trial budget |
| Sampler applicability | Define the exact class-targeting behavior and failure policy for every sampler and combination sampler |
| Runtime/resource policy | Freeze instance, timeout, worker count, peak-memory method, retry rule, and infrastructure-failure classification |
| Optional metrics | Decide whether log loss is included and freeze its exact multiclass definition if included |
| Dataset coverage quotas | Convert the representation, imbalance, size, dimensionality, missingness, and categorical-proportion goals into an auditable registry rule |
| Output schema | Freeze raw-record, aggregate, failure, posterior, figure, and manifest schemas before the smoke run |

If any item remains unresolved, the V2 run is blocked. A later change must be
handled as a dated protocol amendment and must state whether prior evidence is
still comparable.

## 16. Final acceptance checklist

The V2 protocol package is ready for a full benchmark only when:

- V1 files and evidence remain unchanged;
- the V2 protocol, analysis plan, configuration, environment, registry, and
  decision register are versioned;
- all must-lock decisions are resolved before outcome-bearing execution;
- the model/condition matrix agrees with the final-design summary;
- every fold-sensitive operation is proven training-only;
- synthetic engineering tests pass;
- the dataset registry satisfies eligibility and coverage rules without
  score-based selection;
- the smoke run proves deterministic schemas, resumability, and failure
  accounting;
- every full-run expected cell can be classified as valid or explicit failure;
- Bayesian outputs include the ROPE probabilities and uncertainty summaries;
- manuscript language distinguishes protocol, observation, interpretation, and
  limitation; and
- the final evidence package is reproducible from the locked environment and
  provenance manifests.
