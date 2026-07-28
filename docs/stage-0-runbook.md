# Stage 0 execution runbook

This runbook is the operational checklist for the feasibility and dataset-audit
stage. It does not authorize a full benchmark run. The scope gate remains
[`docs/stage-0-feasibility.md`](stage-0-feasibility.md), and the live schedule
is [`docs/timeline.md`](timeline.md).

## 1. Bootstrap and record the environment

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
powershell -ExecutionPolicy Bypass -File scripts/record_environment.ps1
powershell -ExecutionPolicy Bypass -File scripts/verify_setup.ps1
```

The first command creates `.venv` and installs the provisional Stage 0
dependencies. The second records hardware, package, repository, and input
configuration evidence under `artifacts/environment/`. The third runs the
required tests, lint check, and module smoke checks. Do not acquire research
datasets until all three commands succeed.

## 2. Acquire candidates with provenance

For every candidate, add one row to:

- [`data/acquisition_manifest.csv`](../data/acquisition_manifest.csv) for the
  retrieval event and local file hash;
- [`data/dataset_registry.csv`](../data/dataset_registry.csv) for the audit
  evidence and inclusion decision.

Use stable source IDs and preserve the original downloaded file name. Never
rename a file in a way that loses its source identity. Keep raw files under
`data/raw/`; they are intentionally ignored by Git. Commit the manifests,
registry evidence, and scripts, not unreviewed or restricted raw data.

Minimum acquisition-manifest evidence:

- exact source and source ID;
- source URL, version/date, and license or terms;
- UTC retrieval time and download method;
- local path, byte size, and SHA-256 hash;
- an explicit download status and notes for redirects, archives, or conversions.

## 3. Audit candidates

Use the CSV auditor as an initial evidence generator:

```powershell
.venv\Scripts\python.exe -m src.stage0 data\raw\candidate.csv `
  --target target_column `
  --json-out artifacts\runs\candidate-audit.json
```

Review the generated evidence manually and update the registry with source
metadata, target semantics, feature types, duplicate/group/time checks,
leakage status, applicability, and the inclusion or exclusion decision. The
initial auditor output is not an eligibility decision by itself.

## 4. Run the representative pilot

Select candidates by data properties, not preliminary scores:

- one numeric dataset;
- one mixed-type dataset;
- one near the minimum minority-support boundary;
- one larger or high-dimensional dataset, if available.

Run each pilot candidate with the planned five folds, three seeds, all three
classifiers, and all applicable conditions:

```powershell
.venv\Scripts\python.exe -m src.pilot data\raw\candidate.csv `
  --target target_column `
  --output-dir artifacts\runs\candidate-stage0
```

Review `pilot_results.csv`, `pilot_failures.csv`, rankings, and the guarded
Friedman summary. Record runtime, memory, sampler failures, output completeness,
and deterministic replay results. Pilot scores are feasibility evidence only;
they must not select datasets or methods.

## 5. Scope-lock review

Before the full benchmark, commit evidence for every Stage 0 gate:

- complete registry and acquisition manifests;
- valid numeric/mixed applicability matrix;
- explicit sampler and classifier settings;
- no unresolved leakage-test failures;
- deterministic replay for fixed seeds;
- runtime and memory within the available budget;
- final dataset count and complete comparison cells;
- dated protocol amendment for any material change.

Only after this review may the timeline advance from Stage 0 to the locked
benchmark phase.
