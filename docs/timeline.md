# Research project timeline

This timeline covers the first paper only: the multiclass imbalanced tabular
benchmark. It assumes one researcher working approximately 8–10 hours per
week. The portfolio case studies are staged future work and are not part of the
critical path.

The dates below assume a start week of 2026-08-03. Shift the dates if the Python
environment or dataset access is delayed; do not compress the validation gates
to preserve the target date.

## Completed before Week 1

- Created the Stage 0 feasibility protocol and registry schema.
- Added the fold-safe pilot, audit, failure, and guarded statistics utilities.
- Added repository governance and project-specific skills.
- Established the feature branch and pushed the current implementation.

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

