# Research project timeline

This timeline covers the first paper only: the multiclass imbalanced tabular
benchmark. It assumes one researcher working approximately 8–10 hours per
week. The portfolio case studies are staged future work and are not part of the
critical path.

The dates below assume a start week of 2026-08-03. Shift the dates if the Python
environment or dataset access is delayed; do not compress the validation gates
to preserve the target date.

## Live progress

Last updated: 2026-09-11

The critical-path table below is the forecast. The task-status table is the
current execution source of truth and must be updated whenever a task is
completed, blocked, or materially re-scoped.

Status markers: `[x]` complete, `[>]` current, `[!]` blocked, `[ ]` pending.

| ID | Task | Status | Evidence / next action |
|---|---|---|---|
| SETUP-01 | Prepare the pre-research repository foundation, governance, skills, Stage 0 protocol, and pilot utilities. | [x] Complete | `c6143a5` plus the Stage 0 runbook, acquisition manifest, and setup verification script; PowerShell syntax and manifest-header checks passed. |
| ENV-01 | Install supported Python, create `.venv`, install dependencies, and run the local test/import smoke checks. | [x] Complete | Repository-local Python 3.12.10 and `.venv` are usable without elevation; 3 tests passed; Ruff and both Stage 0 module smoke checks passed. |
| ENV-02 | Record the reproducible OS, hardware, Python, and package environment. | [x] Complete | `artifacts/environment/metadata.json`, `pip-freeze.txt`, and clean `pip-check.txt` generated. |
| REG-01 | Build and version the candidate dataset registry and acquisition manifests. | [x] Complete | Twenty-five OpenML/UCI candidates are recorded with exact source IDs/URLs, versioned retrieval metadata, local SHA-256/file-size evidence, class distributions, and canonical semicolon-delimited applicability. The reproducible acquisition script and semantic categorical override manifest are committed; raw files remain ignored. |
| AUD-01 | Audit eligibility, leakage, duplicates, identifiers, missingness, labels, support, and feature types. | [x] Complete | Eleven candidates remain provisionally retained: Yeast, CMC, Glass, Balance Scale, Optdigits, Dermatology, Iris, Wine, CNAE-9, Seeds, and Wine Quality Red. Multiple Features (Factors) is deferred for its documented 1,043.5-second replay against the 900-second budget. The provisional floor remains `n_min_class >= 5`; it is not final statistical-power policy. |
| PILOT-01 | Run the representative Stage 0 pilot across numeric, mixed, low-support, and larger/high-dimensional datasets. | [x] Complete | The twelve-candidate feasibility pilot completed the locked 495-cell matrix with explicit valid/failure rows; after the runtime review, eleven remain provisionally retained. Outputs, runtime/RSS evidence, and categorical overrides are recorded under ignored `artifacts/runs/`. No scores were used for selection. |
| LOCK-01 | Resolve failures and freeze dataset count, applicability, samplers, classifiers, and settings. | [x] Complete | Independent review approved the eleven-dataset scope. The hashed pilot-review manifest verifies complete cells, reviewed failures, deterministic replay, and runtime/memory evidence; Multiple Features (Factors) remains deferred for exceeding the 900-second replay budget. The approved scope lock is `artifacts/runs/scope-lock.json`; do not start the full benchmark. |
| PROTO-01 | Freeze the statistical analysis plan and complete-case rules. | [x] Complete | `docs/protocol.md` records `final-analysis-v1`: exact metric and zero-recall rules, dataset-level complete cells, numeric/mixed/BRF analysis families, Friedman/Nemenyi and Wilcoxon procedures, Holm correction, paired effect sizes, bootstrap intervals, and claim boundaries. Registry, acquisition, configuration, and environment hashes are frozen in the document. |
| TEST-01 | Pass leakage, fold-boundary, categorical, and weight-routing tests. | [x] Complete | Gate B tests are covered by the focused `tests/test_pilot.py` suite: training-only preprocessing, untouched test folds, target exclusion, mixed categorical ordering, and XGBoost training-weight routing. |
| REPLAY-01 | Confirm the locked replay is reproducible and feasible within the recorded resource budget. | [x] Complete | `scripts/verify_gate_c.py` independently revalidated the eleven retained datasets, 495 cells per dataset, exact replay result/failure rows, reviewed failures, and replay runtimes below 900 seconds. The payload was emitted to the system temp directory because `artifacts/runs` is read-only in this checkout. |
| RUN-01 | Execute the locked benchmark with resumability, provenance, failure logs, and resource logging. | [x] Complete | `results/full-run/benchmark_manifest.json` records the frozen hashes, all 11 datasets, 5,445 expected cells, 4,191 valid cells, and 1,254 explicit failure cells; aggregate result and failure hashes are recorded in the ignored manifest. |
| ANALYSIS-01 | Aggregate results, run pre-specified statistics/effect sizes, and generate figures. | [x] Complete | `results/analysis/analysis_manifest.json` records Gate D status, complete-cell aggregates and efficiency means, frozen scalar tests/effects, descriptive per-class summaries, moderator data, and the planned figures. |
| ROBUST-01 | Review analysis outputs for protocol, scope, failure, and locked-sensitivity consistency. | [x] Complete | `results/analysis/gate-e-review.json` passed the frozen-reference, table-invariant, skipped-test, artifact-hash, and deterministic replay checks. |
| PAPER-01 | Write and review the paper against generated evidence. | [>] Current | Draft Methods and Results only from the frozen analysis package; report failures, skipped tests, uncertainty, and efficiency evidence. |
| REPRO-01 | Reproduce the locked workflow from a clean checkout and archive manifests. | [x] Complete | `scripts/verify_gate_f.py` archives the exact tracked source, runs the full checks in that temporary checkout, and revalidates the frozen Gate E evidence; the ignored payload is `results/analysis/gate-f-review.json`. |
| SUBMIT-01 | Finalize release materials and preserve the submitted commit snapshot. | [x] Complete | Gate G technical release snapshot: `docs/release.md`, `scripts/verify_gate_g.py`, and ignored `results/analysis/gate-g-review.json`; no external submission is claimed. |

### Update rule

When work lands, update this table and the relevant schedule row in the same
logical commit. Add the commit or artifact that supports `[x]`, record the
blocker for `[!]`, move `[>]` to the next actionable task, and update the
`Last updated` date. A pilot is a feasibility gate, not evidence for the final
paper's claims.

### Stage 0 evidence boundary (2026-09-09)

The retained local pilot artifacts under ignored `artifacts/runs/` are the
evidence source for the counts above; they are not committed results. Their
failure CSVs remain part of the evidence and include the expected protocol
boundaries: ordinary SMOTE-family conditions are not applicable to mixed CMC,
SMOTENC is not applicable to numeric Glass or Yeast, and Balanced Random Forest
is an RF-only comparator. ADASYN and other sampler failures remain explicit
rows, not silently missing cells.

The corrected Letter run is explicitly deferred for feasibility reasons: it
exceeded the 900-second Stage 0 budget before complete output artifacts were
written. This is a runtime/provenance decision, not a pilot-score decision;
reconsideration requires protocol-preserving optimization or a dated,
approved resource-budget amendment. The registry and acquisition manifest
continue to preserve Letter's source ID, path, byte size, hash, license, and
exclusion rationale.

Gate A is frozen for the eleven retained datasets after independent review. The
expanded twelve-candidate pilot and deterministic replay provide mechanical
evidence, while Multiple Features (Factors) remains deferred because its replay
took 1,043.5 seconds against the 900-second budget. The approved scope lock
records the eleven-dataset scope and applicability matrix; no full benchmark
work has begun.

The expanded audit records a fixed 25-candidate OpenML/UCI pool and eleven
provisional retained datasets. The locked pilot ran all 12 candidates before
the computational deferral, and a second identical pass matched every metric,
sampled-row count, and failure cell. Model scores remain feasibility evidence
only: no full benchmark was run, no dataset or method was selected from scores,
and the next controlled step is the final-protocol package.

### Gate B/C evidence boundary (2026-09-10)

Gate B is complete on `research/final-protocol`. The focused pipeline tests
prove that training-derived preprocessing and samplers see training rows only,
the declared target is excluded from features, mixed data is ordinal-encoded
before SMOTENC and one-hot encoded afterward, and balanced XGBoost receives
training-fold sample weights.

Gate C is complete for the locked eleven-dataset scope. The executable replay
validator rechecks the current scope lock and pilot-review hashes, exact output
schemas and cell accounting, every first-pass/replay result and failure table,
failure-review status, and the 900-second runtime budget. It emits a complete
review payload without running the full benchmark or changing any generated
research artifacts.

### Final protocol evidence boundary (2026-09-10)

The final analysis plan is frozen in `docs/protocol.md` as
`final-analysis-v1`. It fixes the primary and secondary metric definitions,
zero-recall behavior, complete-case rules, analysis families, confirmatory and
exploratory procedures, multiplicity correction, paired effect sizes,
bootstrap intervals, and the claim boundary before the full benchmark run.
The G-mean edge cases are covered by focused tests in `tests/test_pilot.py`.

### Full-run evidence boundary (2026-09-10)

RUN-01 completed against the frozen protocol on `codex/run-01-benchmark`.
The ignored manifest records all eleven retained datasets, 5,445 expected
cells, 4,191 valid records, 1,254 explicit failure records, raw-file hashes,
the environment and configuration hashes, per-cell runtime/RSS fields, and
dataset completion markers. Aggregate raw tables are hashed in the manifest;
no statistical analysis, figures, or paper claims have been generated.

### Gate D and analysis evidence boundary (2026-09-11)

Gate D passed against the complete ignored `results/full-run/` run. The review
rechecked all eleven dataset markers, 5,445 expected cells, 4,191 valid records,
1,254 explicit failure records, aggregate schemas, and frozen input hashes.
ANALYSIS-01 then generated the ignored `results/analysis/` package from complete
dataset cells using the frozen scalar-metric procedures, paired effects, and
10,000-resample bootstrap intervals. The package includes complete-cell
runtime/RSS/resampling means, descriptive per-class-recall summaries because
class labels are dataset-specific, registry-derived moderators, a class
distribution overview, a ranking heatmap, a baseline-delta plot, and valid
critical-difference diagrams. These are evidence artifacts reviewed by Gate E;
at that stage they did not approve paper claims, amend the protocol, or
complete Gates F-G.

### Gate E analysis-freeze boundary (2026-09-11)

Gate E passed for the locked analysis package. The validator rechecked the Gate
D counts and frozen references, all analysis artifact hashes and schemas,
complete-cell and family restrictions, statistical output invariants, and the
descriptive efficiency tables. A fresh replay regenerated every analysis table
and generated figure byte-for-byte. The resulting ignored
`results/analysis/gate-e-review.json` freezes the evidence for paper drafting;
it does not authorize scope changes, protocol amendments, or claims beyond the
complete dataset-level evidence.

### Gate G technical release boundary (2026-09-11)

Gate G preserves the exact feature-branch source snapshot and the hashes of the
environment, scope-lock, full-run, Gate D, analysis, and Gate E evidence
manifests. The validator requires a clean dedicated branch and a current
passed Gate F review, then records the `git archive` hash in the ignored
`results/analysis/gate-g-review.json` payload. This is a technical release
snapshot only: no paper artifact, venue submission, acceptance, deployment,
or publication claim is made.

## Completed before Week 1

- [x] Created the Stage 0 feasibility protocol and registry schema.
- [x] Added the fold-safe pilot, audit, failure, and guarded statistics utilities.
- [x] Added repository governance and project-specific skills.
- [x] Established the feature branch and pushed the current implementation.

## Critical path

| Weeks | Dates | Work | Deliverable / gate |
|---|---|---|---|
| 1 | Aug 3–9 | Unblock the environment: install a supported Python version, create `.venv`, install provisional dependencies, record OS/CPU/RAM/Python/package versions. | Reproducible environment record; import smoke test. |
| 2 | Aug 10–16 | Inventory OpenML/UCI/KEEL candidates. Record source IDs, licenses, versions, hashes, target semantics, and candidate class distributions. | Registry populated with candidates; no performance-based selection. |
| 3 | Aug 17–23 | Audit duplicates, identifiers, missingness, label quality, group/time structure, leakage, minority support, and raw feature types. | Eligibility and exclusion decisions documented. |
| 4 | Aug 24–30 | Run the representative Stage 0 pilot: numeric, mixed, low-support, and larger/high-dimensional candidates. | Pilot results, failure log, runtime/RSS report, deterministic replay check. |
| 5 | Aug 31–Sep 6 | Resolve pilot failures and confirm the complete comparison families. Freeze dataset count, applicability matrix, sampling settings, and classifier settings. | **Gate A: scope lock.** |
| 6 | Sep 7–13 | Write the final protocol and statistical analysis plan. Define the primary metric, complete-case rules, pairwise comparisons, effect sizes, and correction policy. | Versioned analysis plan before full results. |
| 7 | Sep 14–20 | Harden preprocessing and fold boundaries. Add tests for train-only fitting, untouched test folds, target leakage, categorical handling, and weight routing. | **Gate B: leakage and pipeline tests pass.** |
| 8 | Sep 21–27 | Add dataset adapters and provenance checks. Validate every retained dataset through the same audit path. | Final registry and reproducible input manifests. |
| 9 | Sep 28–Oct 4 | Run a second pilot/replay on the locked configuration. Confirm output schema, failure handling, and resource budget. | **Gate C: reproducibility and feasibility pass.** |
| 10 | Oct 5–11 | Prepare full-run scripts, logging, resumability, and result directories. Do not commit generated full results yet. | One-command or documented staged execution path. |
| 11 | Oct 12–18 | Run the first third of datasets across all seeds, folds, classifiers, and applicable conditions. | Partial raw results and failure review. |
| 12 | Oct 19–25 | Run the second third of datasets. Investigate only protocol-defined failures; do not tune methods from observed scores. | Partial raw results and resource review. |
| 13 | Oct 26–Nov 1 | Run the final third of datasets. Re-run only documented infrastructure failures. | Full raw result set. |
| 14 | Nov 2–8 | Verify completeness, seed/fold coverage, sampler applicability, class counts, and output hashes. | **Gate D: full-run integrity.** |
| 15 | Nov 9–15 | Aggregate fold/seed results at dataset level. Produce rankings, baseline deltas, win/tie/loss tables, and per-class summaries. | Auditable aggregated results. |
| 16 | Nov 16–22 | Run the pre-specified Friedman/Nemenyi analyses on complete blocks and limited corrected Wilcoxon comparisons. | Statistical tables with skipped-test reasons. |
| 17 | Nov 23–29 | Compute paired effect sizes and confidence intervals. Analyse runtime and resampled-size consequences. | Effect-size and efficiency appendix. |
| 18 | Nov 30–Dec 6 | Create figures: class-distribution overview, ranking heatmaps, critical-difference diagrams where valid, and baseline-delta plots. | Figure set generated from scripts. |
| 19 | Dec 7–13 | Robustness review: confirm no post-hoc dataset/method changes, inspect failures, run locked sensitivity checks only. | **Gate E: analysis freeze.** |
| 20 | Dec 14–20 | Draft Methods and Experimental Protocol sections from the locked configuration. | Methods draft with no result-dependent claims. |
| 21 | Dec 21–27 | Draft Results and figures. Report negative, failed, and non-significant findings. | Results draft tied to generated artifacts. |
| 22 | Dec 28–Jan 3 | Draft Related Work, Introduction, Contributions, and Limitations. Position against larger benchmarks without claiming size-based novelty. | Complete paper draft v1. |
| 23 | Jan 4–10 | Draft Discussion and Conclusion. Separate empirical findings, exploratory moderators, and future portfolio studies. | Complete paper draft v2. |
| 24 | Jan 11–17 | Reproducibility pass: clean checkout, environment install, registry rebuild, pilot replay, and result-generation instructions. | **Gate F: independent-style reproduction pass.** |
| 25 | Jan 18–24 | Technical review: check citations, formulas, tables, claims, labels, confidence intervals, and source provenance. | Review issue list resolved. |
| 26 | Jan 25–31 | Finalize README, model/data cards, limitations, license notes, and release manifest. | Submission/release candidate. |
| 27 | Feb 1–7 | Proofread and format for the selected venue. Run final smoke tests and verify branch/commit history. | Submission package ready. |
| 28 | Feb 8–14 | Submit or circulate for review. Preserve the exact submitted commit and archive the environment/results manifest. | **Gate G: submission snapshot archived.** |

## Weekly operating rhythm

Each week should end with:

1. one reviewable commit or a documented reason to defer committing;
2. updated task status and evidence;
3. a short record of blockers and decisions;
4. a clean or intentionally documented working tree;
5. a push when the coherent slice materially benefits from remote review or
   backup.

## Scope-control rules

- Do not add F1, StadiumPulse, Past Paper AI, or Token Router experiments to
  this critical path.
- Do not increase the dataset count after Gate A because a method performs
  poorly.
- Do not add classifiers, samplers, metrics, or tuning after Gate A without a
  documented protocol change and timeline impact.
- If the environment, runtime, or complete-cell requirements fail, reduce the
  dataset count before reducing leakage or reproducibility controls.

## Definition of done

The first paper is complete when the repository contains:

- a frozen dataset registry and provenance record;
- reproducible code and environment instructions;
- raw, aggregated, ranking, statistical, and figure-generation artifacts;
- documented failures and exclusions;
- a paper whose claims match the evidence;
- a final commit snapshot that can be reproduced from a clean checkout.
