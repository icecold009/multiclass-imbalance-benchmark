---
name: dataset-audit
description: Audit, curate, and register datasets for the multiclass imbalance benchmark. Use when adding candidate datasets, checking eligibility, resolving labels or feature types, investigating leakage, deduplicating sources, or updating dataset_registry.csv.
---

# Dataset audit

Use `data/dataset_registry.csv` as the structured evidence record and
`docs/stage-0-feasibility.md` as the eligibility contract.

## Audit sequence

1. Record the public source URL/ID, license or terms, version/date, and local
   SHA-256 hash.
2. Identify the target column, label semantics, task type, and all class
   counts.
3. Record `n_rows`, `n_classes`, global majority/minority IR, `n_min_class`,
   `d_raw`, missingness, and feature types.
4. Check duplicates, identifiers, target leakage, repeated entities, groups,
   temporal order, and derived copies of other candidate datasets.
5. Decide the valid split policy: stratified, grouped, or temporal. Do not use
   ordinary StratifiedKFold when row independence is false.
6. Determine sampler applicability. Numeric, mixed, and categorical-only data
   must not be treated as interchangeable.
7. Record an eligibility decision and a concrete exclusion reason.

## Eligibility safeguards

- Require at least three classes and enough minority support for five-fold CV
  and the configured neighbor count.
- Define minimum support before looking at model performance.
- Treat semantic categorical columns as categorical even if encoded as numbers;
  document manual overrides.
- Measure `d_raw` before encoding and `d_encoded` only within fitted folds.
- Do not discard difficult datasets after seeing poor scores. Exclusions must
  be protocol-based and visible.

## Provenance rules

Never commit raw data without confirming redistribution rights. Prefer download
scripts, source identifiers, hashes, and a small legal test fixture. Do not
replace empty or unavailable real data with fabricated benchmark data.

Use `python -m src.stage0 <csv> --target <column>` for the initial local audit,
then manually complete provenance, leakage, split, and applicability fields.
