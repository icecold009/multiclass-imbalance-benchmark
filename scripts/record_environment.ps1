$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$outputDir = Join-Path $repoRoot 'artifacts\environment'
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    throw 'The project virtual environment is missing. Run scripts/bootstrap.ps1 first.'
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors
$memory = Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
$pythonVersion = & $python --version 2>&1
$pipFreeze = & $python -m pip freeze

$metadata = [ordered]@{
    recorded_at_utc = [DateTime]::UtcNow.ToString('o')
    os = [System.Environment]::OSVersion.VersionString
    python = $pythonVersion
    cpu = $cpu
    memory = $memory
    git_commit = (git -C $repoRoot rev-parse HEAD)
    git_branch = (git -C $repoRoot branch --show-current)
}

$metadata | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $outputDir 'metadata.json') -Encoding utf8
$pipFreeze | Set-Content (Join-Path $outputDir 'pip-freeze.txt') -Encoding utf8
Write-Host "Environment record written to $outputDir"
