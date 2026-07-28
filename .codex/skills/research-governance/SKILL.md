---
name: research-governance
description: Govern scope, claims, evidence, and staged delivery for this multiclass imbalance research project. Use when planning experiments, revising the research question, connecting portfolio projects, interpreting results, or deciding whether work is ready to advance beyond Stage 0.
---

# Research governance

Use this skill for research decisions, not routine code edits.

## Establish the phase

1. Read `AGENTS.md` and `docs/stage-0-feasibility.md`.
2. Check the current branch and working tree.
3. Identify whether the request concerns Stage 0, the locked benchmark, or
   staged future work.
4. Do not widen the first paper merely because an adjacent portfolio project
   is interesting.

## Keep the evidence boundary explicit

- The core paper evaluates public multiclass tabular datasets.
- F1, StadiumPulse, Past Paper AI, and Token Router remain motivation,
  discussion, or future studies unless separately approved and methodologically
  specified.
- Synthetic data supports controlled stress testing only; it does not support
  real-world claims.
- A pilot establishes pipeline feasibility, not scientific conclusions.
- Exploratory moderator findings must not be presented as validated decision
  rules.

## Advance only through gates

Before the full run, require a complete registry, valid applicability matrix,
explicit sampler settings, leakage checks, deterministic replay, environment
record, and representative runtime/failure evidence. If a gate fails, narrow
the scope or repair the protocol before adding datasets or methods.

## Review changes by risk

For each proposed change, state its effect on:

- comparison completeness;
- leakage or selection bias;
- statistical independence and power;
- reproducibility;
- runtime and memory;
- the claims the paper can support.

Prefer a smaller complete comparison over a larger matrix with invalid or
missing cells. Record material decisions in the relevant document or commit
message.
