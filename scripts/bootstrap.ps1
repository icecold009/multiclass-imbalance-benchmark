$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$venvPath = Join-Path $repoRoot '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
$repoPython = Join-Path $repoRoot '.python312\python.exe'

$venvUsable = $false
if (Test-Path -LiteralPath $venvPython) {
    try {
        & $venvPython --version *> $null
        $venvUsable = ($LASTEXITCODE -eq 0)
    } catch {
        $venvUsable = $false
    }
}

if (-not $venvUsable) {
    $pythonLauncher = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        if (& py -0p | Select-String '3\.12') {
            $pythonLauncher = 'py'
            & $pythonLauncher -3.12 -m venv --clear $venvPath
        } else {
            $pythonLauncher = 'py'
            & $pythonLauncher -3 -m venv --clear $venvPath
        }
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonLauncher = 'python'
        & $pythonLauncher -m venv --clear $venvPath
    } elseif (Test-Path -LiteralPath $repoPython) {
        $pythonLauncher = $repoPython
        & $pythonLauncher -m venv --clear $venvPath
    } else {
        throw 'Python launcher not found. Install Python 3.12, then rerun this script.'
    }
}

$python = $venvPython
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $repoRoot 'requirements.txt')
& $python -c "import imblearn, pandas, sklearn, xgboost; print('Stage 0 imports: OK')"
