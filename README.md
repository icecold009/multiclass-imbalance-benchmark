# Multiclass Imbalance Benchmark

A benchmark of imbalance-handling methods for multiclass tabular classification,
with explicit accounting for applicability, sampler failures, and the dataset
coverage behind statistical comparisons.

**[Read the latest paper](paper/main4.pdf)** ·
[Manuscript source](paper/main.tex) ·
[Reproducibility guide](docs/reproducibility.md)

## The study

The V1 study compares fixed imbalance-handling conditions across **11 public
multiclass datasets**: nine numeric and two mixed-type. It evaluates logistic
regression, random forest, and XGBoost using fold-local preprocessing,
stratified five-fold evaluation under three seeds, and macro-F1 as the sole
primary metric. Statistical comparisons use complete dataset-level blocks;
their coverage can be smaller than the full dataset collection.

The frozen run accounts for all 5,445 enumerated records:

| Outcome | Records | Meaning |
| --- | ---: | --- |
| Unsupported combination | 1,005 | Excluded by the applicability rules; no model run was attempted. |
| Valid result | 4,191 | An applicable attempt produced a valid result. |
| Sampler failure | 249 | An applicable attempt failed in the sampler. |

The 4,440 applicable attempts comprise the valid results and sampler failures.
The paper reports statistical coverage, uncertainty, exploratory analyses,
runtime, and resampled row counts alongside predictive performance. The
[evidence guide](docs/reproducibility.md) explains which analyses are planned
and which are exploratory.

V1's protocol was frozen internally before the full run; it was not publicly
preregistered. The original protocol remains unchanged for provenance, with
its terminology clarified in [the protocol status note](docs/protocol-status.md).

## Paper and project status

The current manuscript snapshot is [`paper/main4.pdf`](paper/main4.pdf),
received from the author on **9 October 2026**. Its 12 pages are stored unchanged
from the supplied file; the [paper guide](paper/README.md) records its SHA-256
and provides compilation and anonymous-package instructions.

The V1 model-run records are frozen. Revised analysis and manuscript sources
are included in this repository, while raw data and generated results remain
local. Analysis replay from stored records and a fresh model-training rerun
are separate procedures. This repository does not establish an independent
replication or a DOI-backed manuscript release.

The **V2 expanded study** has a separate protocol, registry, and implementation.
Its full execution remains subject to the locked compute and remote-preflight
gates. Start with [the V2 protocol](docs/protocol-v2.md) and
[remote execution guide](docs/v2-free-remote-sharding.md). V2 does not amend the
frozen V1 results.

## Repository guide

| Location | Contents |
| --- | --- |
| [`paper/`](paper/) | Latest supplied PDF, LaTeX source, style files, and paper review guide. |
| [`src/`](src/) | Benchmark, preprocessing, statistical analysis, and validation components. |
| [`scripts/`](scripts/) | Acquisition, execution, analysis, reproducibility, and packaging commands. |
| [`tests/`](tests/) | Tests for analysis, sampling, execution, and evidence boundaries. |
| [`data/`](data/) | Dataset registries, acquisition provenance, and feature overrides. |
| [`docs/`](docs/) | Protocols, environment setup, runbooks, and release requirements. |

The [V1 registry](data/dataset_registry.csv) defines the dataset collection;
the [acquisition manifest](data/acquisition_manifest.csv) records retrieval
provenance and terms. Raw datasets are not redistributed in this checkout.

## Set up locally

Use **Python 3.12**, the recorded benchmark runtime. These commands assume
Windows PowerShell with Python 3.12 available through the `py` launcher and
are run from the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-release.txt
.venv\Scripts\python.exe -m src.stage0 --help
.venv\Scripts\python.exe -m src.pilot --help
```

The [environment guide](docs/environment.md) describes the Windows bootstrap
alternative and environment recording. Setup installs dependencies; it does
not acquire datasets or generate benchmark results. V2 uses its own declared
environment in [`requirements-v2.txt`](requirements-v2.txt), with an exact
record in [`requirements-v2.lock`](requirements-v2.lock).

### Validate the checkout

Run the source-boundary and statistical helper tests as a setup check:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_source_snapshot.py tests/test_stats.py
.venv\Scripts\python.exe -m ruff check src tests
```

[GitHub Actions](https://github.com/icecold009/multiclass-imbalance-benchmark/actions/workflows/ci.yml)
runs the V1 suite on Python 3.11 and 3.12 and the combined suite in a separate
Python 3.12 environment with V2, analysis, and PDF dependencies. Follow
[the workflow](.github/workflows/ci.yml) for the full suite selections;
the V1 setup above does not install the optional V2 dependencies.

## Reproduce or extend the work

| Task | Starting point |
| --- | --- |
| Replay V1 analysis from frozen result records | [Reproducibility guide](docs/reproducibility.md) |
| Acquire datasets and run the locked V1 benchmark | [Acquisition guide](docs/data-acquisition.md) and [V1 runbook](docs/stage-0-runbook.md) |
| Prepare the expanded V2 study for remote execution | [V2 protocol](docs/protocol-v2.md) and [sharding runbook](docs/v2-free-remote-sharding.md) |
| Compile the paper or stage anonymous review materials | [Paper guide](paper/README.md) and [review checklist](paper/review-checklist.md) |
| Validate a clean source revision and release snapshot | [Release requirements](docs/release.md) |

Analysis replay requires the frozen records under `results/full-run/`, which
are ignored by Git. It regenerates tables and figures without retraining
models. A fresh benchmark run also requires the acquired datasets and their
verified hashes. Pilot scores are for feasibility checks and must not be used
to select datasets or methods.

Before an external release, rerun Gates F and G against the exact final source
revision. They validate clean-checkout reproducibility and the technical
release snapshot; venue submission and publication remain separate steps.

## License

Repository code is released under the [MIT License](LICENSE). Dataset use is
governed by the source terms recorded in the acquisition manifest.
