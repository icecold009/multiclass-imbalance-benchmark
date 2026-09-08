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

If the system Python installation is unavailable but a repository-local
`.python312\python.exe` exists, the bootstrap script uses that ignored runtime
to recover or recreate `.venv`. The runtime itself must never be committed.

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
