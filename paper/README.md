# TMLR paper draft

This directory contains the anonymous TMLR-formatted manuscript source for
the locked multiclass imbalance benchmark. The manuscript is a technical draft
and has not been submitted to TMLR or any other venue.

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

The companion [`review-checklist.md`](review-checklist.md) records the paper
completion checks, evidence boundary, and remaining venue-preparation items.
Before de-anonymization, replace the anonymous author block only after the
authors, affiliations, acknowledgments, and target submission workflow are
known.
