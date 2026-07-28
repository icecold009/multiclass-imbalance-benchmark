# Agent instructions

## Project mission

This repository is the core implementation for a reproducible benchmark of
resampling and other imbalance-handling strategies for multiclass tabular
classification. The first paper is deliberately narrower than the broader
portfolio research program.

## Current phase

The project is in **Stage 0: feasibility and dataset audit**. The authoritative
gate is [`docs/stage-0-feasibility.md`](docs/stage-0-feasibility.md). Do not
start or describe a full benchmark run until the dataset registry, applicability
matrix, environment record, and representative pilot have passed that gate.

## Git and change control

- Always work on a dedicated feature branch.
- Never commit or merge directly to `main` without explicit user approval.
- Keep commits coherent and reviewable; do not commit generated results with
  implementation changes.
- Preserve unrelated user changes and inspect `git status` before editing.
- Use `apply_patch` for source and documentation edits.

## Research invariants

- Keep the core question focused on multiclass imbalanced tabular
  classification. F1, StadiumPulse, Past Paper AI, and Token Router are
  motivation or staged future work, not pooled observations in this paper.
- Never resample, fit preprocessing, encode, scale, impute, tune, or compute
  training-derived weights using a test fold.
- Treat Balanced Random Forest as an RF-only ensemble comparator, not as a
  universal sampler condition.
- Use SMOTENC for valid mixed-type synthetic sampling; never apply ordinary
  SMOTE to one-hot categorical features.
- Log unsupported sampler/dataset combinations and failures explicitly.
- Never fabricate benchmark outputs, use pilot scores to select datasets, or
  describe simulated data as real-world evidence.
- Keep claims proportional to the number of datasets, complete comparison
  cells, and the statistical unit of analysis.

## Sources of truth

- [`docs/stage-0-feasibility.md`](docs/stage-0-feasibility.md): scope gates
  and eligibility rules.
- [`data/dataset_registry.csv`](data/dataset_registry.csv): dataset evidence
  and inclusion decisions.
- [`src/stage0.py`](src/stage0.py): candidate CSV audit behavior.
- [`src/pilot.py`](src/pilot.py): fold-safe pilot pipeline.
- [`src/stats.py`](src/stats.py): guarded aggregation and rank summaries.
- `.codex/skills/`: task-specific operating instructions.

## Validation and handoff

At minimum, run `git diff --check` and the relevant focused checks. When a
Python environment exists, use the project environment and run the pilot or
tests from the documented module entry points. State clearly which checks ran,
which were blocked by environment or data availability, and whether results
are local pilot evidence or full benchmark evidence.

Do not push, open, or merge a pull request unless the user asks for that release
step. Before handoff, report the branch, commit(s), changed files, checks,
remaining blockers, and the next safe action.
