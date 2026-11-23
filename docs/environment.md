# Environment setup

The project requires a supported Python installation before Stage 0 can run.
The current repository does not bundle Python or a virtual environment.

## Windows bootstrap

From PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
```

The script creates `.venv`, upgrades packaging tools, installs
`requirements.txt`, and runs an import smoke test. It does not download
benchmark data or create research results.

The bootstrap script prefers an installed Python 3.12 from the Windows Python
Launcher, then another available launcher version, then a working `python`
command, and finally a repository-local `.python312\python.exe`. This fallback
also handles a present but empty or unusable `py` launcher. The repository-local
runtime itself is ignored and must never be committed. Python 3.12 remains the
repository bootstrap preference and matches the recorded July Stage 0
environment. Other supported environments should use bounded dependencies and
record their own versions. The full-run manifest does not record its
interpreter, so this preference does not establish the historical benchmark
training runtime.

## Frozen analysis runtime and portable replay

`results/analysis/analysis_manifest.json` records Python 3.13.12 and the exact
direct analysis package versions used for the frozen analysis: imbalanced-learn
0.14.2, matplotlib 3.11.2, NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.0,
scikit-posthocs 0.14.0, SciPy 1.18.1, statsmodels 0.14.6, and XGBoost 3.2.0.
The July 28, 2026 environment record is Python 3.12.10 and describes the earlier
Stage 0 setup. It is not the frozen analysis environment. The full-run manifest
does not identify the original training interpreter, so that runtime remains
unverified. The analysis manifest also does not preserve the complete
historical transitive `pip freeze`; the original training environment cannot
be fully reconstructed from the available manifests.

`requirements-analysis.in` pins those recorded direct analysis versions and
specifies the replay and test tools. `requirements-analysis.lock` pins the full
package set captured from the repository `.venv` on October 14, 2026. Its nine
direct analysis versions match the analysis manifest, but its transitive
versions are not evidence of the historical training environment. Use a
separate environment:

```powershell
python -c "import sys; assert sys.version_info[:3] == (3, 13, 12), sys.version"
python -m venv .venv-analysis
.venv-analysis\Scripts\python.exe -m pip install -r requirements-analysis.lock
.venv-analysis\Scripts\python.exe scripts\replay_publication_analysis.py
```

If `python` does not point to Python 3.13.12, invoke that installation's
`python.exe` by its full path for the first two commands.

The replay regenerates aggregate tables and figures from the included frozen
result and analysis records. It does not download datasets, rerun model
training, or validate the author-side Gates F or G. To verify the locally
available CMC, Glass, and Yeast materializations without network access, run
`python scripts/acquire_openml_candidates.py --verify-local` from the repository
root. Live OpenML acquisition and its source-version checks are a separate
operation.

## Record the environment

After bootstrap:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/record_environment.ps1
```

This writes local machine and package evidence under `artifacts/environment/`.
The generated files are intentionally ignored until the exact environment is
reviewed and a lock snapshot is approved.

Pytest uses repository-local temporary and cache directories under
`artifacts/test-tmp/` and `artifacts/test-cache/` so test execution does not
depend on permissions in a stale system temporary directory.

## Required pre-research checks

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m src.stage0 --help
.venv\Scripts\python.exe -m src.pilot --help
```

Do not begin dataset acquisition until imports, tests, and the environment
record succeed.
