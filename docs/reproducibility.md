# V1 reproducibility and evidence boundary

## Frozen benchmark evidence

The V1 full run finished on 2026-09-10. Its manifest identifies the immutable
benchmark source revision; that revision is disclosed in the de-anonymized
author copy and withheld from the anonymous supplement. The manifest
enumerates 5,445 records: 4,191 valid results and 1,254 rows in
the failure log. The audited applicability split is 1,005 unsupported rows
and 4,440 applicable attempts; of those attempts, 249 are sampler-specific
failures and none are other applicable failures.

The frozen full-run CSV hashes are:

| File | SHA-256 |
| --- | --- |
| `results/full-run/benchmark_results.csv` | `eda651d6d64f17c87e3c3135312549a0b2db82442143d1ab872f060663d34f91` |
| `results/full-run/benchmark_failures.csv` | `558115822c342351b2aa6408f33cca10c54130ef8bb5b5bc2a55d0c97fe71c49` |
| `results/full-run/benchmark_manifest.json` | `e689eb0f3a8f01faf7eb5e899e93492b0f23718ab09d85247b5da2d8b1875cda` |

The full-run manifest also records SHA-256 references for the dataset markers,
the protocol, registry, acquisition manifest, environment record, and scope
lock. Raw datasets are not redistributed. The acquisition manifest records
their source identifiers and terms; a fresh model rerun requires obtaining
those datasets under their stated terms.

## Reproduce the analysis from frozen records

From the repository root, install the declared analysis and release
dependencies, then use the existing `results/full-run/` evidence:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
.venv\Scripts\python.exe -m pip install -r requirements-release.txt
.venv\Scripts\python.exe scripts/run_analysis.py
.venv\Scripts\python.exe scripts/verify_gate_e.py
```

This path validates and re-analyzes the frozen result and failure records; it
does not retrain models. The analysis manifest records the current analysis
Python and package versions, all analysis-table hashes, and all generated
figure hashes. Gate E independently replays the analysis and compares those
tables and figures. The critical-difference renderer is also checked in fresh
processes with distinct `PYTHONHASHSEED` values.

The corrected analysis retains all 184 baseline-comparison rows but computes
Wilcoxon and Holm values only for the 104 comparisons with at least three
complete dataset blocks. The other 80 rows are descriptive. The supplementary
package includes the failure-accounting tables, figure coverage table,
200,000-resample primary permutation sensitivity, exact sign-flip sensitivity,
manifests, and generated figures.

## What this evidence does and does not establish

The frozen result commit identifies the code used for the original model run.
It is not the immutable revision for the revised manuscript and analysis in
this working branch. Those revisions must be committed, rebuilt, and archived
before a DOI-backed public release can identify them. No DOI-backed archive
is cited in this draft.

Validating the stored records and regenerating the analysis is not an
independent reproduction of model training. No claim of independent
replication is made. A fresh full benchmark is a separate, expensive run and
is not part of this procedure.
