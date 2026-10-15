# Paper review checklist

## Historical V1 release candidate checklist

The completed items in this V1 section and the pending items in its historical
TMLR snapshot below describe earlier candidates. They do not establish current
package or publication readiness. Current status is recorded in the dated
sections at the end of this checklist.

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

## Historical V1 TMLR snapshot and pending items

The checklist below records the state of the earlier TMLR-oriented V1
candidate. Its TMLR-specific pending items are historical and are not current
submission instructions. The controlling status for the October 14 package
revision is in [Current local revision (2026-10-14)](#current-local-revision-2026-10-14)
and the final publication gate must be completed for the venue actually
selected. No venue or submission route is currently selected for the named
non-arXiv candidate.

### Previously verified technical snapshot

The corrected percentage sentence and the four-artifact Gate G snapshot were
verified on clean commit `9bfed74` before the ethics/broader-impact revision
below. Those checks are historical evidence for that commit; they do not cover
later manuscript edits or the October 14 package candidates.

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

## Current local revision (2026-10-14)

- [x] Reconciled grouped-profile sensitivity results and limits with the
      hash-verified run record; retained the post hoc, descriptive scope.
- [x] Corrected the memory and elapsed-time descriptions to match endpoint RSS
      sampling and the implemented timer boundary; recorded the frozen-protocol
      departure in `docs/analysis-implementation-note.md`.
- [x] Clarified that the original full-run training interpreter is unverified;
      the Stage 0 and analysis environments are separate records.
- [x] Distinguished this paper's per-class-recall G-mean from TILBench's
      macro-sensitivity/macro-specificity formula.
- [x] Audited cited bibliography records, corrected Branco et al.'s issue and
      article pages, and removed the unused misattributed Gorishniy entry.
- [x] Added named PDF title/author source metadata and preserved anonymous
      metadata substitution in the release builder.
- [x] Added the grouped-sensitivity script, results note, tests, and verified
      outputs to the anonymous evidence package inventory.
- [x] Rebuilt separate arXiv and named non-arXiv V2 candidates. Both 13-page
      PDFs have the expected title/author metadata and embedded fonts; source
      ZIPs rebuild to matching PDF text with no undefined references or
      overfull boxes. All rendered pages were inspected with no visual defects.
- [x] Both evidence ZIPs passed CRC, per-file manifest hashes, file-inventory
      and exclusion checks. Each fresh extraction passed the 114-test suite and
      two independent 18-table/8-figure replays; the 41 installed dependency
      versions match `requirements-analysis.lock`.
- [!] A clean Python environment was created for each package, but installing
      the lock was blocked because this shell cannot resolve the PyPI host.
      Tests and replays therefore used the repository Python 3.13.12 environment
      matching every locked version; fresh installation is not claimed.
- [x] This pre-commit checkpoint was recorded before local commits were
      authorized. Its dirty-branch status is superseded by the final status
      reconciliation below.
- [ ] Author and independent human technical review, author/disclosure
      confirmation, arXiv preview/account checks, and non-arXiv venue selection
      remain pending before any submission or public release.

## Final status reconciliation (2026-10-15)

- [x] Checked the official [CMC source description](https://archive.ics.uci.edu/dataset/30/contraceptive%26), which identifies the data as a subset of the 1987 National Indonesia Contraceptive Prevalence Survey, and the [Dermatology source description](https://archive.ics.uci.edu/dataset/33/dermatology), which says patient names and ID numbers were removed.
- [ ] Author/institution confirmation of original data authority, consent, and applicable oversight remains required. The UCI descriptions do not establish those facts for this secondary analysis; no approval, consent, or exemption is claimed.
- [!] Fresh dependency installation remains blocked by this shell's inability to resolve PyPI. The extracted-package tests and replays used the repository Python 3.13.12 environment with all locked versions installed.
- [x] Gate E and Gate F passed twice and Gate G passed twice on the final local
      feature-branch snapshot. Gate G validated all four current-commit
      artifacts and records `external_submission: false`; the ignored receipts
      bind them to the same source head.
- [x] The gate commands used a temporary, process-only Git exclude for the
      pre-existing untracked `docs/professor-feedback-emails.md`. The draft was
      not staged or changed and remains in the checkout.
- [!] Fresh lock installation is still blocked by PyPI DNS, and the earlier
      online CMC/Glass/Yeast acquisition attempt was incomplete. No fresh
      installation or complete live acquisition is claimed.
- [ ] Obtain independent human technical and claim review, and confirm authorship, contact, affiliation, AI disclosure, funding, conflicts, and applicable ethics statements from the author's records.
- [ ] For arXiv, verify account/endorsement, categories, metadata/license, and arXiv's compiled preview. For the non-arXiv candidate, select a venue and verify its current formatting, anonymity, supplementary-material, and licensing requirements.
- [ ] Present the exact PDFs, ZIPs, metadata, and verification receipts for final approval before either submission.
