# multiclass-imbalance-benchmark
Benchmark of resampling strategies for multi-class imbalanced tabular classification

## Current phase

Stage 0 Gates A-F are complete, and the anonymous TMLR paper draft is tracked
under `paper/`. The locked full benchmark raw run and the pre-registered
analysis are recorded under the ignored `results/full-run/` and
`results/analysis/` directories; clean-checkout reproducibility is recorded by
Gate F before the Gate G technical release snapshot.

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

## Gate G technical release snapshot

After the exact release-candidate commit is created, run:

```powershell
.venv\Scripts\python.exe scripts/verify_gate_g.py
```

This validates the dedicated feature branch, clean working tree, passed Gate F
review, frozen environment/results manifests, and the exact `git archive`
source snapshot. It writes the ignored `results\analysis\gate-g-review.json`.
See [`docs/release.md`](docs/release.md) for the archive boundary and explicit
non-goals. Gate G does not claim a paper submission, venue acceptance,
deployment, or publication.

## Paper draft

The anonymous TMLR-formatted source is
[`paper/main.tex`](paper/main.tex). It reports only the frozen full-run and
analysis evidence, including explicit failures, skipped tests, uncertainty,
and efficiency outcomes. See [`paper/README.md`](paper/README.md) for local
compilation instructions and
[`paper/review-checklist.md`](paper/review-checklist.md) for the pre-submission
review boundary. The draft has not been submitted to a venue.
