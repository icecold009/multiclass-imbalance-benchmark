# TMLR paper release candidate

This directory contains the de-anonymized, TMLR-formatted V1 manuscript source
for the locked multiclass imbalance benchmark. It is a local release candidate
and has not been submitted to TMLR or any other venue.

The manuscript contains only the frozen V1 benchmark evidence. The expanded V2
study is a separate follow-up and must not be used to amend these results.

## Compile locally

The source uses the official TMLR style files (`tmlr.sty`, `tmlr.bst`, and the
supporting files copied from the JMLR-maintained
[`tmlr-style-file`](https://github.com/JmlrOrg/tmlr-style-file) repository).
From this directory, run:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript references generated figures under
`../results/analysis/figures/`. A clean checkout must first reproduce the
ignored full-run and analysis artifacts, or the source will show a clear
placeholder box in place of a missing figure. Generated PDFs and auxiliary
files are intentionally not tracked.

## Review checklist

The companion [`review-checklist.md`](review-checklist.md) records the V1 paper
completion checks, evidence boundary, and remaining venue-preparation items.
Before external submission, confirm the author affiliation, funding statement,
acknowledgments, and target-venue de-anonymization requirements.
