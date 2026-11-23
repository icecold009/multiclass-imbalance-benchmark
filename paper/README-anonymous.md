# Anonymous TMLR source

This source accompanies the blinded V1 benchmark manuscript. The expanded V2
study is a separate follow-up and must not be used to amend the V1 results.

The manuscript uses the TMLR style files supplied in this package. Compile the
source in this directory with `pdflatex main.tex`, `bibtex main`, and two more
`pdflatex main.tex` passes. The paper figures are staged under
`../results/analysis/figures/` by the release builder. A clean checkout needs
the frozen analysis evidence before those figures can be rendered.

The anonymous supplement is generated from the blinded source and the frozen
evidence inventory. It withholds source revisions and scans its contents for
identifying information. See `docs/reproducibility.md` for the replay boundary
and evidence limits.
