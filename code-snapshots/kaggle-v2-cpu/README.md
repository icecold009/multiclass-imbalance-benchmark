# multiclass-imbalance-benchmark
Benchmark of resampling strategies for multi-class imbalanced tabular classification

## Current phase

Stage 0 Gates A-F are complete. The locked full benchmark raw run and the
pre-registered analysis are recorded under the ignored `results/full-run/` and
`results/analysis/` directories; clean-checkout reproducibility is recorded by
Gate F before release review.

The staged work plan is documented in
[`docs/timeline.md`](docs/timeline.md).

The operational Stage 0 sequence is documented in
[`docs/stage-0-runbook.md`](docs/stage-0-runbook.md).

The registry schema is in [`data/dataset_registry.csv`](data/dataset_registry.csv).
Retrieval provenance is tracked separately in
[`data/acquisition_manifest.csv`](data/acquisition_manifest.csv).
Candidate CSVs can be audited with:

```powershell
python -m src.stage0 path\to\candidate.csv --target target_column
```

The pilot is evidence-gathering only; its scores must not be used to select
datasets or methods.

The locked benchmark executor is:

```powershell
.venv\Scripts\python.exe scripts\run_benchmark.py
```

It validates the frozen protocol and scope lock, resumes only from hashed
dataset markers, and writes raw result and failure tables without running the
statistical analysis. See [`docs/stage-0-runbook.md`](docs/stage-0-runbook.md)
for staged execution and review boundaries.

Before acquiring research data, follow [`docs/environment.md`](docs/environment.md)
and [`docs/data-acquisition.md`](docs/data-acquisition.md). The pre-registered
protocol template is [`docs/protocol.md`](docs/protocol.md).

## Gate F clean-checkout reproducibility

After the analysis freeze, run:

```powershell
.venv\Scripts\python.exe scripts\verify_gate_f.py
```

This creates a temporary `git archive` checkout, runs the full tests, Ruff, and
module smoke checks there, and revalidates the frozen Gate E evidence. The
ignored `results\analysis\gate-f-review.json` payload records the source
snapshot and manifest hashes. It does not rerun the benchmark, expand the
dataset scope, amend the protocol, or authorize paper claims.
