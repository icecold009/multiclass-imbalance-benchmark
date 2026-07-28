# multiclass-imbalance-benchmark
Benchmark of resampling strategies for multi-class imbalanced tabular classification

## Current phase

The repository is in **Stage 0: feasibility and dataset audit**. The full
dataset list and comparison matrix will be locked only after the pilot passes
the gates in [`docs/stage-0-feasibility.md`](docs/stage-0-feasibility.md).

The registry schema is in [`data/dataset_registry.csv`](data/dataset_registry.csv).
Candidate CSVs can be audited with:

```powershell
python -m src.stage0 path\to\candidate.csv --target target_column
```

The pilot is evidence-gathering only; its scores must not be used to select
datasets or methods.
