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
runtime itself is ignored and must never be committed. Python 3.12 is the
project's recorded benchmark runtime; other supported environments should use
the bounded dependencies and record their own versions.

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
