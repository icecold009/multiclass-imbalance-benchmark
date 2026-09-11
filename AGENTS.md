# Agent instructions

## Project mission

This repository is the core implementation for a reproducible benchmark of
resampling and other imbalance-handling strategies for multiclass tabular
classification. The first paper is deliberately narrower than the broader
portfolio research program.

## Current phase

Stage 0 Gates A-D and Gate E are complete. The locked full-run raw evidence and
the frozen analysis outputs are recorded under the ignored `results/full-run/`
and `results/analysis/` directories. The authoritative scope gate is
[`docs/stage-0-feasibility.md`](docs/stage-0-feasibility.md); the next work is
paper drafting against the frozen evidence.

## Git and change control

- Always work on a dedicated feature branch.
- Never commit or merge directly to `main` without explicit user approval.
- Use professional, coherent commits with Conventional Commit-style prefixes:
  `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `data`,
  `research`, or `chore`.
- Write an imperative subject line, keep it concise, and use an optional scope
  when it improves reviewability, for example:
  `research(stage0): record mixed-data applicability rules`.
- Keep each commit focused on one logical change. Put the reason, evidence,
  validation, and material risks in the body when the subject is insufficient.
- Commit and push coherent work whenever doing so is necessary and relevant to
  the task or enables review. Do not push unrelated work or use pushes as a
  substitute for review.
- A push never authorizes merging. Preserve explicit user approval for merge or
  direct `main` work.
- Do not commit generated results with implementation changes.
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
- [`docs/protocol.md`](docs/protocol.md): pre-research protocol template.
- [`docs/timeline.md`](docs/timeline.md): staged delivery timeline and gates.
- [`docs/environment.md`](docs/environment.md): environment bootstrap and
  recording procedure.
- [`docs/data-acquisition.md`](docs/data-acquisition.md): provenance and data
  storage policy.
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

## Codex package workflow

- Work only on a feature branch. Never edit or merge `main` directly.
- One bounded package uses one feature branch and one pull request.
- Before editing, inspect the current checkout, branch, remote, working tree, backlog, and existing PRs.
- Use the repository’s existing canonical backlog. Do not create duplicate TODO plans.
- Every package must define its goal, scope, non-goals, files, tests, acceptance criteria, and evidence.
- Preserve unrelated work and never discard dirty changes without explicit approval.
- Run the repository’s real verification commands and report exact results and commit SHA.
- Treat local checks, GitHub checks, reviews, hosted testing, deployment, and user-reported evidence as separate.
- Do not fabricate external evidence, live-provider results, deployment results, or review approval.
- Do not merge, deploy, publish, submit, or expose secrets without explicit user approval.
- After implementation, stop for independent ChatGPT validation before the next package.
