# V2 Kaggle CPU package

This package contains the locked V2 code, configuration, registry metadata, and
an inert Kaggle notebook template. It contains no raw datasets, credentials,
results, or local machine artifacts.

## Required order

1. Keep the Kaggle notebook private and set Accelerator to `None / CPU`.
2. Run only the notebook's preflight cell first.
3. Preserve the preflight environment record with the shard output.
4. Do not start a shard until the compute-approval gate is explicitly cleared.

The notebook does not run `Run All`; benchmark and data-acquisition commands
are commented templates. Every shard must use the same package manifest,
source commit, locked config, registry, and raw-file hashes.

## Package layout

- `v2_cpu_shard.ipynb` — private CPU notebook template.
- `../scripts/record_kaggle_environment.py` — remote Linux preflight record.
- `../scripts/run_v2_full.py` — existing gated runner.
- `../scripts/prepare_v2_shards.py` and `../scripts/merge_v2_shards.py` — shard plan and merge validation.

The package is prepared for remote use only. Do not upload local `data/raw`,
`results`, `output`, or `paper` directories.
