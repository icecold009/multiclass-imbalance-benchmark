# Gate G: technical release snapshot

Gate G binds one clean feature-branch commit to the frozen benchmark evidence
and four current artifacts: the de-anonymized local release PDF and source ZIP,
plus the anonymous TMLR submission PDF and supplementary ZIP. A passing Gate G
is a technical release check; it does not mean that the paper was submitted or
published.

## Scope

The payload records the current commit and branch, the `git archive` hash and
tracked-file set, the passed Gate F review, and hashes for the environment,
scope-lock, full-run, Gate D, analysis, and Gate E evidence. It also records the
SHA-256 and size of each current PDF and ZIP artifact.

The public source ZIP name contains the current short commit ID. The anonymous
submission files use neutral names so their filenames do not disclose the
source revision:

- `paper/main.pdf` — compiled de-anonymized V1 release candidate;
- `output/release/multiclass-imbalance-benchmark-paper-<commit>.zip` — source,
  paper, and frozen evidence package;
- `output/submission/tmlr-anonymous-submission.pdf` — anonymous TMLR review
  manuscript;
- `output/submission/tmlr-anonymous-supplement.zip` — anonymous supplementary
  material. The supplementary ZIP includes the exact anonymous PDF submitted
  separately, so Gate G can compare both copies byte-for-byte.

The package builder writes a `package-manifest.json` with a SHA-256 for every
member. The public source ZIP records the full source commit; the anonymous
supplement withholds it. Gate G checks ZIP integrity, safe member paths,
per-file hashes, commit identity, PDF framing and blind metadata, the
anonymous manuscript source, and TMLR's 100 MB supplementary limit. TMLR requires an
anonymized manuscript and supplement; its guide also warns against style-file
changes that alter formatting, fonts, or layout. See the [TMLR author
guide](https://www.jmlr.org/tmlr/author-guide.html).

Generated data, environment records, results, PDFs, and ZIPs remain ignored by
Git. Gate F and Gate G record their hashes without adding these artifacts to the
source commit.

## Package

The tracked implementation includes `src/submission.py`,
`src/release_artifacts.py`, `scripts/build_release_artifacts.py`, the Gate G
validator, `requirements-release.txt` for its PDF parser, focused tests, the
de-anonymized paper source, and the review checklist. Install both
`requirements.txt` and `requirements-release.txt` before running Gate G or the
release artifact tests. The package builder creates a separate anonymous TeX source in the
ignored output tree. It retains the original TMLR style assets byte-for-byte,
removes the author block and acknowledgments, uses TMLR's default double-blind
header, clears PDF author metadata, checks the author name/email against the
PDF and supplement, and withholds the Git commit from the anonymous ZIP. It
does not add a public repository link to the anonymous supplement.

The local Gate G payload is written to the ignored
`results/analysis/gate-g-review.json`. Preserve it with the commit and the
environment/results archive. Any later source or artifact change requires new
Gate E, Gate F, package, and Gate G evidence.

## Acceptance criteria

Gate G passes only when:

- the feature branch is dedicated and the tracked/untracked working tree is clean;
- Gate F passed for the exact current commit;
- all frozen evidence exists and has the expected status;
- the `git archive` file set matches `git ls-files` and contains no generated
  results;
- all four PDF/ZIP artifacts exist and pass structural checks; public artifacts
  identify the current commit, while the Gate G payload binds the anonymous
  artifacts by hash without disclosing the commit;
- the anonymous PDF and supplementary ZIP pass author, contact, and local-origin
  link scans and retain the official TMLR style files unchanged;
- the payload records `external_submission: false`.

## Verification

After the source changes have been committed to the feature branch, reproduce
the analysis from the existing frozen run, run Gate E twice, and run Gate F
twice in fresh processes. Compile `paper/main.tex`, stage the anonymous TeX,
compile it with BibTeX and two final `pdflatex` passes, copy that PDF to its
neutral submission path, then build the ZIPs and run Gate G twice:

```powershell
python scripts\run_analysis.py
python scripts\verify_gate_e.py
python scripts\verify_gate_e.py
python scripts\verify_gate_f.py
python scripts\verify_gate_f.py
Set-Location paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
Set-Location ..
python scripts\build_release_artifacts.py --stage-only
Set-Location output\staging\tmlr-anonymous\paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
Copy-Item main.pdf ..\..\..\submission\tmlr-anonymous-submission.pdf
Set-Location ..\..\..\..
python scripts\build_release_artifacts.py
python scripts\verify_gate_g.py
python scripts\verify_gate_g.py
```

## Non-goals

This package does not submit the manuscript, upload data, merge to `main`,
publish a release, deploy an application, or claim a human technical review.
Authors must complete active OpenReview profiles, including accurate
affiliations, conflicts, and publication histories, and provide appropriate
action-editor suggestions, human-subjects/IRB reporting, funding, competing
interests, and any conflicts not covered by institutional history. TMLR says
this information is not shown to reviewers before a decision. Authors must
provide and confirm the actual values; do not infer or invent them. An
independent human claim review remains required before submission. Jev's AI
review is advisory and does not replace it.
