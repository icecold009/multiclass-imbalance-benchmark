# Data acquisition and provenance

No benchmark data is committed yet. Real datasets enter the project only after
the audit below is complete.

## Candidate sources

Use OpenML, UCI, KEEL, or another public source only when the exact dataset
identity, version, access method, and license/terms can be recorded. Prefer
source IDs and download scripts over manually renamed files.

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

## Cleaning rules

Apply only protocol-defined, domain-neutral cleaning. Do not drop a dataset
because it produces weak scores. Remove identifiers or obvious target leakage
only when the decision is documented and reproducible.

Do not merge duplicate OpenML/UCI/KEEL versions as independent evidence. Mark
related or derived datasets so they cannot silently inflate the effective number
of blocks.

## Storage policy

Raw and processed data directories are ignored by Git by default. Keep source
manifests, hashes, small legal test fixtures, and generated registry metadata in
version control. Do not commit restricted or personally identifying data.
