$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'The project virtual environment is missing. Run scripts/bootstrap.ps1 first.'
}

Write-Host 'Running Python tests...'
& $python -m pytest
if ($LASTEXITCODE -ne 0) {
    throw 'Python tests failed.'
}

Write-Host 'Running Ruff...'
& $python -m ruff check (Join-Path $repoRoot 'src') (Join-Path $repoRoot 'tests')
if ($LASTEXITCODE -ne 0) {
    throw 'Ruff checks failed.'
}

Write-Host 'Checking Stage 0 module entry points...'
& $python -m src.stage0 --help | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'The stage0 module smoke check failed.'
}
& $python -m src.pilot --help | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'The pilot module smoke check failed.'
}

Write-Host 'Stage 0 repository setup checks passed.'
