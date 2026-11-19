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

## Full-checkout analysis and Gate E

From the repository root, install the declared analysis and release
dependencies, then use the existing `results/full-run/` evidence:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
.venv\Scripts\python.exe -m pip install -r requirements-release.txt
.venv\Scripts\python.exe scripts/run_analysis.py
.venv\Scripts\python.exe scripts/verify_gate_e.py
```

This full-checkout path validates the frozen result and failure records and
re-analyzes them; it does not retrain models. Gate D also requires the raw CSVs
and per-dataset completion markers, and Gate E calls Gate D before replaying
the analysis. Those inputs are not included in the publication evidence ZIPs.
The analysis manifest records the analysis Python and package versions,
analysis-table hashes, and generated-figure hashes. The critical-difference
renderer is also checked in fresh processes with distinct `PYTHONHASHSEED`
values.

## Portable replay from a publication evidence package

The rebuilt package should include a separate aggregate-only replay route.
Create Python 3.13.12 environment and install the pinned replay/test lock. From
the unpacked evidence directory, run:

```powershell
python -c "import sys; assert sys.version_info[:3] == (3, 13, 12), sys.version"
python -m venv .venv-analysis
.venv-analysis\Scripts\python.exe -m pip install -r requirements-analysis.lock
.venv-analysis\Scripts\python.exe scripts/replay_publication_analysis.py
```

This route checks the included manifests and hashes, the exact 5,445 unique
result/failure keys, record schemas and metrics, and consistency of the stored
analysis artifacts. It regenerates the 18 analysis tables and 8 figures under
`results/analysis-portable-replay/` and compares them byte-for-byte with the
frozen versions. The receipt lists the checks and their limits. It does not
check raw dataset bytes or excluded completion markers, rerun training, or
constitute Gate D, Gate E, or an independent training reproduction. Keep the
full-checkout Gate D/E route above when those provenance inputs are available.

The corrected analysis retains all 184 baseline-comparison rows but computes
Wilcoxon and Holm values only for the 104 comparisons with at least three
complete dataset blocks. The other 80 rows are descriptive. The supplementary
package includes the failure-accounting tables, figure coverage table,
200,000-resample primary permutation sensitivity, exact sign-flip sensitivity,
manifests, and generated figures.

Publication review found that those baseline rows use pair-specific complete
cases, whereas the frozen protocol states a family-wide complete-case rule.
The 104 computed Wilcoxon/Holm results therefore remain exploratory and are
not planned confirmatory evidence. The family-wide complete-case rule is used
for the planned Friedman/Nemenyi analyses.

The current analysis adds a separately labelled post hoc macro-F1 sensitivity
that omits ADASYN and compares the other eight numeric conditions on the nine
datasets complete for every classifier. It uses 200,000 seeded within-block
rank permutations for each classifier and exact signed-rank sign flips for
the seven raw-baseline contrasts, with Holm correction within each
classifier's seven-test family. The dataset coverage changes from the
original three-block XGBoost omnibus to nine blocks in this sensitivity, so
it does not isolate a causal explanation. The three generated tables are
`adasyn_excluded_friedman.csv`, `adasyn_excluded_pairwise.csv`, and
`adasyn_excluded_imbalance_descriptive.csv`; the IR-stratified rows are
descriptive only.

The three-block numeric-core XGBoost macro-F1 sensitivity also has an additive
exact tied-rank calibration. From the locked analysis environment, run
`.venv-analysis\Scripts\python.exe scripts/run_publication_addenda.py`.
The command verifies the frozen analysis manifest and `cell_means.csv` hashes,
then writes a separately manifested result to
`results/publication-addenda/v4/`. It enumerates the 5,184 configurations after
fixing one block under global condition-label symmetry and independently
checks all 373,248 configurations. It neither regenerates the original
analysis tables and figures nor changes the planned asymptotic or Monte Carlo
results; the exact calibration is post hoc.

## Post hoc duplicate-profile sensitivity

The separate `results/duplicate-profile-sensitivity/` directory records the
grouped-CV run on the six datasets with observed exact predictor-profile
overlap. Its manifest hashes the inputs, split assignments, result and failure
tables, descriptive comparisons, and coverage summaries. The run completed 90
feasible splits and accounts for all 2,970 model-grid keys. It is a post hoc
model run under a different split design; it does not overwrite or amend the
row-wise V1 results or frozen analysis outputs. The publication evidence
package includes its manifest and hash-verified output files, but does not
redistribute raw datasets.

The July Stage 0 environment record is Python 3.12.10 and the frozen analysis
manifest records Python 3.13.12. The full-run manifest omits the original
training interpreter, which therefore remains unverified.

Gate E denotes local validation and independent replay of analysis tables
and figures from frozen records after the full-checkout Gate D checks pass. Its
replay now writes temporary files under the repository's writable `output/tmp`
directory. Passing Gate E does not establish an exact clean-commit reproduction
(Gate F) or a release snapshot (Gate G); those checks remain separate and
require their stated source state.

## What this evidence does and does not establish

The frozen result commit identifies the code used for the original model run.
It is not the immutable revision for the revised manuscript and analysis in
this working branch. A DOI-backed public release must identify the immutable
manuscript and analysis revision recorded in the rebuilt release manifest.
No DOI-backed archive is cited in this draft.

Validating the stored records and regenerating the analysis is not an
independent reproduction of model training. No claim of independent
replication is made. A fresh full benchmark is a separate, expensive run and
is not part of this procedure.
