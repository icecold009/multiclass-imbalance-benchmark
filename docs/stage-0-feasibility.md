# Stage 0: Feasibility and Dataset Audit

Stage 0 decides the scope of the first benchmark before the full experiment is
run. It is a feasibility gate, not a results-generating phase: pilot scores
must not be used to select datasets, methods, or claims.

## Decisions to lock

Stage 0 must produce:

- a versioned dataset registry with source identifiers, hashes, licenses, and
  inclusion/exclusion decisions;
- an applicability matrix for numeric, mixed, and categorical-only datasets;
- explicit sampler parameters and expected post-resampling distributions;
- a hardware and environment record;
- pilot runtime, memory, determinism, and failure logs;
- the final dataset count and the complete statistical comparison cells.

The target is 12–19 public datasets, but that range is provisional until the
registry audit is complete.

## Dataset eligibility

Every candidate must satisfy the following before entering the benchmark:

1. It is a single-label classification task with at least three classes.
2. It is tabular and has a documented public source and redistribution status.
3. Its target column and label semantics are documented.
4. It has enough observations per class for the chosen five-fold split and
   sampler neighbor requirements.
5. It has no unresolved group, time-order, duplicate, or target-leakage issue.
6. Its cleaning decisions can be applied without domain-specific guesswork.

For the current Stage 0 feasibility audit, the provisional structural floor is
`n_min_class >= 5`: this is the minimum needed to place one observation in each
of five stratified test folds while retaining four training observations for
the configured `k_neighbors=3` samplers. This is a feasibility floor, not a
claim of adequate statistical power; the final threshold must be frozen at the
scope-lock gate.

The registry must record both the global majority/minority ratio and the full
per-class distribution. `d_raw` means the number of source features before
encoding; `d_encoded` is measured after fitting the fold-specific encoder.

## Applicability policy

The benchmark must not compare invalid pipelines merely to complete a table.

- Numeric datasets support the ordinary numeric sampler family.
- Mixed datasets use SMOTENC for synthetic mixed-type oversampling.
- Categorical-only datasets are excluded unless SMOTEN is implemented.
- Ordinary SMOTE is not applied to one-hot categorical features.
- Balanced Random Forest is analysed as an RF-only ensemble comparator, not as
  a condition available to Logistic Regression and XGBoost.
- A method failure is logged with its stage and error class; it is never
  silently converted into a missing result.

The statistical analysis uses complete cells. The main Friedman/Nemenyi tables
must therefore be defined separately for compatible numeric and mixed-data
populations when method availability differs.

## Pilot matrix

Choose representative candidates by data properties, not by preliminary
performance:

- one numeric dataset;
- one mixed-type dataset;
- one dataset near the minimum minority-support boundary;
- one high-dimensional or larger dataset, if available.

Run one end-to-end pass with the planned five-fold splitter, three seeds, all
three classifiers, and every applicable condition. A smaller smoke test may be
used first to debug the harness, but it is not evidence for feasibility.

Record:

- wall-clock fit and prediction time;
- resampled row counts;
- peak-memory measurement method and value, where available;
- dependency and hardware versions;
- deterministic replay status;
- pipeline and sampler failures;
- output schema completeness.

## Scope-lock gate

Proceed to the full benchmark only when:

- every retained dataset has a complete registry record;
- every main comparison cell is either valid or explicitly excluded;
- no leakage test fails;
- the pilot can replay deterministically for fixed seeds;
- memory and runtime fit the available hardware budget;
- the final dataset list and analysis families are committed before inspecting
  full-run performance.
