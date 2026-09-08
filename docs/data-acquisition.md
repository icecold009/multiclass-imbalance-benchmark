# Data acquisition and provenance

No benchmark data is committed yet. Real datasets enter the project only after
the audit below is complete.

## Candidate sources

Use OpenML, UCI, KEEL, or another public source only when the exact dataset
identity, version, access method, and license/terms can be recorded. Prefer
source IDs and download scripts over manually renamed files.

The current Stage 0 candidate pool contains 25 OpenML/UCI records. The fixed
acquisition list and target checks are in
[`scripts/acquire_openml_candidates.py`](../scripts/acquire_openml_candidates.py).
Run it only when refreshing the deliberately fixed pool; it does not decide
eligibility and it never runs a model.

## Required evidence

For every candidate, add a registry record containing:

- stable dataset ID and display name;
- source, source ID, URL, license/terms, and version/date;
- local SHA-256 hash;
- file format and target column;
- row count, class count, complete class counts, IR, and minimum support;
- raw feature count, semantic feature types, missingness, and encoded count;
- duplicate, group, time-order, and leakage audit results;
- applicable condition families;
- inclusion decision and exclusion reason where relevant.

Semantic categorical overrides for numerically encoded source columns are
recorded in [`data/dataset_feature_overrides.csv`](../data/dataset_feature_overrides.csv).
The semicolon-delimited values are passed to the pilot as categorical-column
overrides; they are never inferred from preliminary scores.

## Cleaning rules

Apply only protocol-defined, domain-neutral cleaning. Do not drop a dataset
because it produces weak scores. Remove identifiers or obvious target leakage
only when the decision is documented and reproducible.

Do not merge duplicate OpenML/UCI/KEEL versions as independent evidence. Mark
related or derived datasets so they cannot silently inflate the effective number
of blocks.

The current audit retains 12 provisional candidates: CMC, Glass, Yeast, Balance
Scale, Multiple Features (Factors), Optdigits, Dermatology, Iris, Wine, CNAE-9,
Seeds, and Wine Quality Red. This is a structural Stage 0 decision only. The
minimum support threshold remains provisional, and the retained set still needs
the representative pilot, deterministic replay, and scope-lock review. The
candidate pool keeps explicit exclusions for runtime, support, group, duplicate
label, related-source, and categorical-only reasons.

## Storage policy

Raw and processed data directories are ignored by Git by default. Keep source
manifests, hashes, small legal test fixtures, and generated registry metadata in
version control. Do not commit restricted or personally identifying data.
