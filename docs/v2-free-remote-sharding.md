# V2 free remote sharding runbook

This runbook describes an operational way to use multiple no-cost remote
sessions when one 32-GiB VM is unavailable. It does not change the V2
scientific protocol, dataset registry, model matrix, tuning budget, metrics,
failure policy, or Bayesian analysis plan.

The benchmark remains blocked until the normal compute-approval decision is
locked. No command in this runbook should be run on the local Windows machine.
The commands are templates for a remote Linux environment only.

## What is being split

The V2 runner's resumability unit is one dataset. Each shard receives a
disjoint set of locked datasets, while every dataset keeps its complete
five-by-three nested-CV, classifier, condition, HPO, threshold-calibration,
resource-logging, and failure-accounting contract.

The current runner already supports repeated dataset selection with
`--dataset DATASET_ID`. The new shard plan and merge validator make the
multi-session workflow auditable rather than relying on manual copying.

Sharding does not combine the RAM of multiple machines. Every remote session
must have enough memory for the datasets it runs. A free session with less
memory is acceptable only if a remote preflight establishes that it can run
its assigned datasets without changing the locked configuration. Otherwise,
the resulting failure is a blocker, not a reason to remove the dataset.

## Required remote preflight

Before any outcome-bearing work, record the remote Linux image, Python version,
CPU, RAM, storage, provider or host, and exact source commit. Verify that:

- the checkout contains the frozen V2 config, protocol, registry, and feature
  overrides;
- every assigned raw file matches the frozen SHA-256 recorded in the registry;
- the environment record is complete and agrees with the config hash;
- the remote process is CPU-only with one worker per model;
- results and logs will persist if the session is interrupted; and
- the host will not be allowed to change the registry or configuration.

Do not start a shard if any of these checks is unresolved.

## Prepare the immutable plan

On one clean remote checkout, select the number of available remote sessions:

```text
python scripts/prepare_v2_shards.py --shards 4 \
  --output artifacts/runs/v2-shard-plan.json
```

The plan is deterministic and records the config, registry, and protocol
hashes. Preserve it with the final evidence. Do not edit it by hand or create
a second plan after a shard has started.

## Run one shard per remote session

Each remote session must use the same locked commit and environment. Give each
session the dataset IDs from its plan entry and a distinct output directory:

```text
python scripts/run_v2_full.py --dataset DATASET_ID_1 \
  --dataset DATASET_ID_2 \
  --output-dir results/v2-shards/shard-01
```

Repeat for each shard. The runner remains responsible for the existing hard
execution gate, nested-CV procedure, equal HPO budget, threshold fitting,
resource fields, retry policy, and explicit `VALID` or `FAILED / NOT
APPLICABLE` records.

If a session stops, resume that shard in the same output directory only after
checking that the config, registry, protocol, raw-data hashes, and environment
record are unchanged. Do not merge partial outputs.

## Validate and merge

After every shard has a complete per-dataset marker, collect the shard output
directories on one remote Linux session and run:

```text
python scripts/merge_v2_shards.py \
  --plan artifacts/runs/v2-shard-plan.json \
  --shard shard-01=results/v2-shards/shard-01 \
  --shard shard-02=results/v2-shards/shard-02 \
  --shard shard-03=results/v2-shards/shard-03 \
  --shard shard-04=results/v2-shards/shard-04 \
  --output-dir results/v2-full
```

The merge refuses to overwrite an existing destination. Before copying, it
checks every shard's frozen hashes, exact assigned dataset set, completion
marker, result/failure cell matrix, trial schema, and recorded counts. It
creates a final `v2_manifest.json` with `execution_mode` set to
`dataset-sharded-merge`, the shard-plan hash, shard assignments, complete
aggregate tables, and aggregate hashes.

If any check fails, stop and preserve the failing shard as evidence. Never
drop a dataset, replace a failure with a zero, or manually edit a result table.

## Analysis boundary

Run the existing V2 analysis only after the merge manifest is complete and the
normal analysis gate is satisfied. The analysis must consume the merged
aggregate and its manifest; shard-level summaries are operational evidence,
not separate scientific results.

## No-cost boundary

This runbook does not select a provider, claim that a free service offers
32 GiB RAM, provision resources, or authorize billing. Those details belong in
the decision register before the full-run gate is opened. A provider or host
that cannot preserve the locked environment and evidence is not acceptable,
even if it is free.
