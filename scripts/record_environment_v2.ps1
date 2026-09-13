$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$outputDir = Join-Path $repoRoot 'artifacts\environment\v2'
$python = Join-Path $repoRoot '.venv-v2\Scripts\python.exe'
$requirements = Join-Path $repoRoot 'requirements-v2.txt'
$config = Join-Path $repoRoot 'config\v2.yaml'

if (-not (Test-Path $python)) {
    throw 'The isolated V2 environment is missing: .venv-v2\Scripts\python.exe'
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors
$memory = Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
$pythonVersion = & $python --version 2>&1
$pipFreeze = & $python -m pip freeze
$pipCheck = & $python -m pip check 2>&1
$pipCheckExit = $LASTEXITCODE
$requirementsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $requirements).Hash.ToLowerInvariant()
$configHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $config).Hash.ToLowerInvariant()
$workingTree = @(git -C $repoRoot status --porcelain)

$metadata = [ordered]@{
    schema_version = 'v2-environment-v1'
    recorded_at_utc = [DateTime]::UtcNow.ToString('o')
    purpose = 'V2 development, validation, and explicitly approved full-run host record'
    os = [System.Environment]::OSVersion.VersionString
    python = $pythonVersion
    cpu = $cpu
    memory = $memory
    git_commit = (git -C $repoRoot rev-parse HEAD)
    git_branch = (git -C $repoRoot branch --show-current)
    requirements_v2_sha256 = $requirementsHash
    config_v2_sha256 = $configHash
    execution_policy = 'cpu_only; workers_per_model=1; cloud_instance=not_selected'
    working_tree_status = $workingTree
}

$metadata | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $outputDir 'metadata.json') -Encoding utf8
$pipFreeze | Set-Content (Join-Path $outputDir 'pip-freeze.txt') -Encoding utf8
$pipCheck | Set-Content (Join-Path $outputDir 'pip-check.txt') -Encoding utf8
if ($pipCheckExit -ne 0) {
    throw 'pip check reported dependency problems; review artifacts/environment/v2/pip-check.txt.'
}
Write-Host "V2 environment record written to $outputDir"
