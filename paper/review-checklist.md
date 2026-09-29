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

## Required before an external submission

- [x] Compile the current de-anonymized and anonymous PDFs from the source at
      the exact release commit; run BibTeX and two final LaTeX passes for each.
- [x] Inspect the current anonymous PDF for layout, overfull boxes, citation
      order, page readability, and author metadata.
- [x] Rebuild the Gate E/F analysis evidence twice in fresh processes and the
      Gate G snapshot twice after the source and artifacts are final.
- [x] Add the author name, contact detail, and acknowledgments to this
      de-anonymized local draft.
- [x] Choose an anonymized supplementary ZIP rather than an external
      repository link, and include the directly supporting source, tables,
      figures, provenance, and a per-file integrity manifest.
- [x] Confirm the anonymous PDF and supplementary ZIP contain no author,
      contact, repository, or document metadata that could identify the authors;
      keep the ZIP under TMLR's 100 MB limit.
- [ ] Authors must complete active OpenReview profiles with accurate
      affiliation, conflicts, and publication history, and supply appropriate
      action-editor suggestions, human-subjects/IRB reporting, funding,
      competing interests, and any conflicts not covered by institutional
      history. See the [TMLR Author Guidelines](https://www.jmlr.org/tmlr/author-guide.html);
      authors must confirm actual values rather than infer or invent them.
- [ ] Obtain an independent human technical and claim review. Jev's AI review
      is advisory and does not satisfy this human review item.
- [ ] Authors confirm that all authorship declarations, funding/conflict
      statements, and TMLR's CC BY 4.0 submission terms are accurate and
      acceptable before submission.

No external submission, upload, DOI release, or public source release has been
performed by preparing this checklist or the local package.
