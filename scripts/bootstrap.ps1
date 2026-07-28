$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$venvPath = Join-Path $repoRoot '.venv'

if (-not (Get-Command py -ErrorAction SilentlyContinue) -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python launcher not found. Install Python 3.12, then rerun this script.'
}

if (-not (Test-Path (Join-Path $venvPath 'Scripts\python.exe'))) {
    if ((Get-Command py -ErrorAction SilentlyContinue) -and (& py -0p | Select-String '3\.12')) {
        & py -3.12 -m venv $venvPath
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $venvPath
    } else {
        & python -m venv $venvPath
    }
}

$python = Join-Path $venvPath 'Scripts\python.exe'
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $repoRoot 'requirements.txt')
& $python -c "import imblearn, pandas, sklearn, xgboost; print('Stage 0 imports: OK')"
