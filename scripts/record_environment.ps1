$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$outputDir = Join-Path $repoRoot 'artifacts\environment'
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$requirements = Join-Path $repoRoot 'requirements.txt'
$stage0Config = Join-Path $repoRoot 'config\stage0.yaml'

if (-not (Test-Path $python)) {
    throw 'The project virtual environment is missing. Run scripts/bootstrap.ps1 first.'
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors
$memory = Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
$pythonVersion = & $python --version 2>&1
$pipFreeze = & $python -m pip freeze
$pipCheck = & $python -m pip check 2>&1
$pipCheckExit = $LASTEXITCODE
$requirementsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $requirements).Hash.ToLowerInvariant()
$stage0ConfigHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $stage0Config).Hash.ToLowerInvariant()
$workingTree = @(git -C $repoRoot status --porcelain)

$metadata = [ordered]@{
    recorded_at_utc = [DateTime]::UtcNow.ToString('o')
    os = [System.Environment]::OSVersion.VersionString
    python = $pythonVersion
    cpu = $cpu
    memory = $memory
    git_commit = (git -C $repoRoot rev-parse HEAD)
    git_branch = (git -C $repoRoot branch --show-current)
    requirements_sha256 = $requirementsHash
    stage0_config_sha256 = $stage0ConfigHash
    working_tree_status = $workingTree
}

$metadata | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $outputDir 'metadata.json') -Encoding utf8
$pipFreeze | Set-Content (Join-Path $outputDir 'pip-freeze.txt') -Encoding utf8
$pipCheck | Set-Content (Join-Path $outputDir 'pip-check.txt') -Encoding utf8
if ($pipCheckExit -ne 0) {
    throw 'pip check reported dependency problems; review artifacts/environment/pip-check.txt.'
}
Write-Host "Environment record written to $outputDir"
