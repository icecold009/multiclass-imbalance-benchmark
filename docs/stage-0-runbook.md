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

After the first pass and an identical replay pass are complete, build the
hashed review manifest from the ignored run directories:

```powershell
.venv\Scripts\python.exe scripts\build_pilot_review.py `
  --run-root artifacts\runs `
  --replay-root artifacts\runs `
  --failure-review passed `
  --datasets balance_scale cmc cnae_9 dermatology glass iris optdigits `
    seeds wine wine_quality_red yeast
```

The generator compares replay metrics, sampled-row counts, and failure cells;
timing and RSS fields are recorded separately because they can vary between
runs. When replay runtime evidence is present, the scope-lock validator also
requires that replay to remain within its recorded runtime budget. Set
`--failure-review passed` only after reviewing every failure row.
List only currently retained dataset IDs in `--datasets`; deferred pilot
outputs remain ignored but must not enter the retained-scope evidence manifest.

## 5. Scope-lock review

The repository provides a deterministic evidence writer for the scope lock.
`--pilot-status passed` records a human review decision only; it cannot replace
the hashed pilot-review evidence. Missing, stale, malformed, or incomplete
evidence keeps Gate A pending. Use `pending` while review is outstanding and
`passed` only after the independent scope decision is recorded:

```powershell
.venv\Scripts\python.exe -m src.stage0 lock `
  --registry data\dataset_registry.csv `
  --manifest data\acquisition_manifest.csv `
  --pilot-evidence artifacts\runs\stage0-pilot-review.json `
  --pilot-status passed `
  --json-out artifacts\runs\scope-lock.json
```

The pilot-review manifest must use `stage0-pilot-review-v1` and record the
current registry and acquisition-manifest hashes, hashed environment and
configuration references, and one run for every retained dataset. Each run
must hash `pilot_metadata.json`, `pilot_results.csv`, `pilot_failures.csv`,
`pilot_rankings.csv`, and `pilot_friedman_summary.csv`; reference hashed
runtime and peak-memory evidence; record the memory measurement method and
value, runtime and budget, deterministic replay, failure review, output
completeness, and expected/valid/failure cell counts. The validator checks the
actual raw-file path, byte size, and SHA-256 from the acquisition manifest and
checks that valid plus explicit failure rows cover every expected comparison
cell. It never selects datasets or methods by pilot score.

Applicability values in `dataset_registry.csv` are canonical, semicolon-
delimited tokens such as `numeric;raw;class_weighted;smote`; whitespace is not
a delimiter, and empty or unknown tokens block the lock. The registry and
acquisition manifest must agree on dataset/source IDs, source metadata,
licenses/terms, hashes, and the local file evidence. Raw data and generated run
artifacts remain ignored and uncommitted.

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

## 6. Verify Gates B and C

Run the focused pipeline tests from a writable temporary pytest location. They
cover train-only preprocessing, untouched test folds, target exclusion, mixed
categorical ordering, and training-fold weight routing:

```powershell
$validationRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("stage0-gates-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $validationRoot -Force | Out-Null
.venv\Scripts\python.exe -m pytest tests\test_pilot.py `
  --override-ini "addopts=" `
  --basetemp "$validationRoot\pytest" `
  -o "cache_dir=$validationRoot\cache"
```

After Gate A is approved, verify Gate C from the retained ignored pilot and
replay directories:

```powershell
$gateCReview = Join-Path ([System.IO.Path]::GetTempPath()) "stage0-gate-c-review.json"
.venv\Scripts\python.exe scripts\verify_gate_c.py --output $gateCReview
```

The Gate C command revalidates the eleven-dataset scope lock, exact 495-cell
output and failure schemas, deterministic replay equality, reviewed failures,
and first-pass/replay runtime budgets. It does not execute the full benchmark.
The repository's `artifacts\runs` directory may be read-only on OneDrive, so
the optional review payload is written to the system temporary directory and
the canonical ignored scope-lock and pilot-review artifacts are preserved.

## 7. Execute the locked benchmark

After the final protocol is committed and Gates A-C are approved, run the
locked benchmark from the repository root:

```powershell
.venv\Scripts\python.exe scripts\run_benchmark.py
```

The executor validates the protocol's frozen hashes, the ready scope lock,
every retained raw-file hash and byte size, and the semantic categorical
overrides before fitting a model. It writes ignored outputs under
`results\full-run\`, including a provenance manifest, one directory per
dataset, explicit valid and failure rows, per-cell runtime/RSS fields, and a
completion marker for each dataset. A dataset is skipped on later invocations
only when its marker and every referenced artifact still match their hashes;
an incomplete or tampered marker stops the run instead of silently replacing
evidence.

For staged execution, repeat the command with one or more locked IDs. The
root manifest retains the same eleven-dataset scope and changes to complete
only after every dataset has a valid 495-cell accounting record:

```powershell
.venv\Scripts\python.exe scripts\run_benchmark.py --dataset balance_scale
```

Use `--dry-run` to validate the frozen inputs and print the planned cells
without fitting models. Do not pass datasets outside the approved scope, add
seeds or folds, or inspect scores to change the locked conditions. Aggregation,
confirmatory statistics, and figures are separate Gate D/E work after the
full raw run is reviewed.

## 8. Review Gate D raw-output integrity

After the locked run is complete, verify the manifest, every dataset completion
marker, aggregate schemas, cell accounting, and all frozen input hashes:

```powershell
.venv\Scripts\python.exe scripts\verify_gate_d.py `
  --run-dir results\full-run `
  --output-dir $env:TEMP\stage0-gate-d-review
```

The command must report a `passed` review payload. It does not rerun models or
change the locked protocol. Keep the generated review payload as the audit
record; the canonical result files remain ignored under `results\full-run\`.

## 9. Run the frozen analysis package

Only after Gate D passes, run the pre-registered aggregation and analysis:

```powershell
.venv\Scripts\python.exe scripts\run_analysis.py
```

This validates Gate D again, aggregates only complete dataset cells, runs the
frozen scalar-metric Friedman/Nemenyi and raw-versus-alternative Wilcoxon tests
with Holm correction, computes paired rank-biserial effects and deterministic
10,000-resample bootstrap intervals, summarizes registry moderators, writes
complete-cell efficiency means for runtime, RSS, and post-sampling rows, and
generates the class-distribution overview, ranking heatmap, baseline-delta plot,
and critical-difference diagrams where the frozen Friedman/Nemenyi results are
valid. Outputs are ignored under `results\analysis\` and include an analysis
manifest with hashes, counts, procedure settings, and skipped-test reasons.
Per-class recall is retained as a descriptive summary because class labels are
dataset-specific; no cross-dataset inferential class matrix is created. Do not
use these commands to change the dataset scope, conditions, seeds, folds, or
claim boundary, and do not draft paper claims until the independent Gate E
robustness review is complete.

## 10. Freeze the analysis at Gate E

Before drafting the paper, validate the analysis manifest and replay the full
analysis package into an isolated temporary directory:

```powershell
.venv\Scripts\python.exe scripts\verify_gate_e.py `
  --run-dir results\full-run `
  --analysis-dir results\analysis `
  --output-dir results\analysis
```

The command must report `passed`. It rechecks the Gate D payload, frozen scope
and protocol references, artifact hashes and schemas, complete-cell and family
rules, skipped-test reasons, p-value and interval bounds, and descriptive
efficiency coverage. It then reruns aggregation and figure generation in a
fresh temporary directory and requires every analysis table and generated
figure to match byte-for-byte. The resulting `gate-e-review.json` is the Gate E
freeze record. This step does not amend the protocol, change the dataset or
condition scope, select a method from scores, or authorize paper claims beyond
what the evidence supports.

## 11. Verify Gate F from a clean source checkout

After the analysis freeze, validate the committed source snapshot and the
frozen ignored evidence package:

```powershell
.venv\Scripts\python.exe scripts\verify_gate_f.py
```

The command creates a temporary `git archive` checkout, confirms that generated
data and environment artifacts are absent from the source snapshot, runs the
full tests, Ruff, and module smoke checks there, and revalidates the Gate E
manifest and byte-for-byte analysis replay in the original evidence checkout.
It writes the ignored `results\analysis\gate-f-review.json` payload. This is an
independent-style source and manifest check; it does not rerun the benchmark,
expand the dataset scope, amend the protocol, or authorize paper claims.
