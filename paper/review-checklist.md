# Paper review checklist

## Complete in this V1 release candidate

- [x] Official TMLR source format and current JMLR-maintained style assets.
- [x] De-anonymized title, abstract, introduction, related work, methods,
      results, discussion, limitations, reproducibility, and conclusion.
- [x] Locked eleven-dataset scope and registry-derived dataset table.
- [x] Fold-safe preprocessing, classifier settings, condition applicability,
      metric definitions, complete-cell aggregation, and statistical plan.
- [x] Exact full-run counts: 5,445 expected, 4,191 valid, and 1,254 failures.
- [x] Explicit failure categories, skipped-test rationale, complete-cell
      coverage, uncertainty, negative result, and efficiency evidence.
- [x] References for the imbalance methods, models, metrics, statistics, and
      adjacent tabular benchmark work.
- [x] Figure paths linked to the frozen analysis outputs with compile-safe
      placeholders when ignored result artifacts are unavailable.
- [x] V1/V2 boundary recorded; no V2 execution result is used in this paper.

## Previously verified technical snapshot

The corrected percentage sentence and the four-artifact Gate G snapshot were
verified on clean commit `9bfed74` before the ethics/broader-impact revision
below. Those checks are historical evidence for that commit; they do not cover
the current manuscript edits and must be repeated for the final submission
candidate.

- [x] The de-anonymized and anonymous PDFs were freshly compiled from commit
      `9bfed74` with BibTeX and two final LaTeX passes for each.
- [x] Gates E and F passed twice, and Gate G passed twice, on the clean release
      candidate before this revision.
- [x] Gate G validated the current-commit de-anonymized PDF and source ZIP,
      plus the anonymous PDF and supplement ZIP; it checked package hashes,
      anonymization, unchanged TMLR style files, and the 100 MB supplement cap.
- [x] Add the author name, contact detail, and acknowledgments to this
      de-anonymized local draft.
- [ ] Recompile both manuscript variants after the current ethics revision;
      inspect the corrected PDFs and approve the final anonymous PDF.
- [ ] Rebuild the anonymized supplement with the final PDF, supporting source,
      tables, figures, provenance, and per-file integrity manifest.
- [ ] Repeat Gates E/F/G on the final clean commit and verify the final hashes.
- [ ] Recheck anonymous PDF and supplement metadata/content after rebuilding;
      keep the supplement under TMLR's 100 MB limit.
- [ ] Complete an active OpenReview profile with accurate affiliation,
      conflicts, and publication history. Confirm sole authorship and complete
      funding, competing-interest, and human-subjects/IRB disclosures from
      actual records; do not infer or invent values. See the
      [TMLR Author Guidelines](https://www.jmlr.org/tmlr/author-guide.html).
- [ ] Verify the source documentation and applicable oversight for the CMC and
      Dermatology records; the new ethics discussion does not assert consent,
      IRB approval, or exemption.
- [ ] Confirm the ethics/broader-impact discussion is accurate and adequate for
      the actual data use and likely downstream risks.
- [ ] Obtain an independent human technical and claim review. Jev's AI review
      is advisory and does not satisfy this human review item.
- [ ] Confirm acceptance of TMLR's CC BY 4.0 submission terms before
      submission; authors retain copyright, while the license permits reuse
      with attribution.
- [ ] After submission, respond to TMLR's request for appropriate,
      non-conflicted action-editor suggestions.
- [ ] Submit only the anonymous PDF and anonymous supplement to TMLR. Keep the
      named PDF and public source release private until the TMLR decision.

No external submission, upload, DOI release, or public source release has been
performed by preparing this checklist or the local package.

## Current local revision (2026-10-09)

- [x] Added the post hoc ADASYN-excluded macro-F1 analysis from frozen
      dataset-level records, with deterministic rank permutations, exact
      signed-rank follow-ups, classifier-specific Holm families, and a
      descriptive-only imbalance summary.
- [x] Reconciled the balanced XGBoost omnibus interpretation, ADASYN complete
      versus incomplete records, small-sample resolution, bootstrap/Holm
      distinction, all-zero comparisons, and the mixed XGBoost near-zero
      SMOTENC example with the generated tables.
- [x] Gate E passed twice in fresh processes after the analysis changes; the
      replay compared the new sensitivity tables and generated figures.
- [x] Focused analysis, robustness, sensitivity, and package tests passed;
      Ruff and Python compilation passed.
- [x] Named and anonymous manuscripts compiled with BibTeX and two final
      LaTeX passes and were visually checked page by page. Both are 12 pages;
      unresolved citation and cross-reference warnings were absent.
- [x] Rebuilt and validated local named source/evidence and anonymous
      supplementary packages against their exact input entries, PDF metadata,
      anonymization, style hashes, and per-file SHA-256 manifests. The check
      report is `output/review-preview-20261009-final/review/artifact-validation.json`.
- [ ] Author review of these freshly rebuilt PDFs remains pending.
The preview above identifies itself as dirty and its Gate F entry says
`not_run`. Final clean-commit validation is recorded separately in
`results/analysis/gate-f-review.json` and `results/analysis/gate-g-review.json`;
their recorded source commit must match the release manifest before the
package can be treated as an exact clean-commit snapshot.
