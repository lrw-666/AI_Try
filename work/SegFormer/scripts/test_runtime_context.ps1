<#
.SYNOPSIS
    Regression checks for process-only runtime setup (no installs or downloads).
.NOTES
    Run with pwsh or Windows PowerShell 5.1. Uses the existing project prefix.
#>
[CmdletBinding()]
param([string]$ResourceRoot = '')

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ResourceRoot)) {
    $workspace = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $project))
    $ResourceRoot = Join-Path $workspace 'segformer_resources'
}
$ResourceRoot = [IO.Path]::GetFullPath($ResourceRoot)
$testResourceRoot = $ResourceRoot
$enter = Join-Path $PSScriptRoot 'enter_project.ps1'
$runner = Join-Path $PSScriptRoot 'run_project.ps1'
$failures = @()

function Assert-Check {
    param([bool]$Condition, [string]$Message)
    if ($Condition) { Write-Output "PASS: $Message" }
    else {
        Write-Output "FAIL: $Message"
        $script:failures += $Message
    }
}

function Get-ProcessSnapshot {
    $snapshot = @{}
    foreach ($key in [Environment]::GetEnvironmentVariables('Process').Keys) {
        $snapshot[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
    }
    return $snapshot
}

function Test-SnapshotEqual {
    param([hashtable]$Expected)
    $actual = Get-ProcessSnapshot
    $equal = $true
    foreach ($key in $actual.Keys) {
        if (-not $Expected.ContainsKey($key)) { Write-Host "Unexpected environment key: $key"; $equal = $false }
    }
    foreach ($key in $Expected.Keys) {
        if (-not $actual.ContainsKey($key) -or $actual[$key] -cne $Expected[$key]) {
            Write-Host "Missing or changed environment key: $key"
            $equal = $false
        }
    }
    return $equal
}

$initialEnv = Get-ProcessSnapshot
$initialLocation = Get-Location
$initialConsoleEncoding = [Console]::OutputEncoding
$initialOutputEncoding = $OutputEncoding
try {
    # Inherited values must be absent, rather than present with an empty value.
    $clearNames = @('PYTHONPATH', 'PYTHONHOME', 'TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE', 'PYTORCH_PRETRAINED_BERT_CACHE', 'PIP_TARGET')
    foreach ($name in $clearNames) { [Environment]::SetEnvironmentVariable($name, 'inherited-value', 'Process') }
    $env:SEGFORMER_RESOURCE = 'stale-resource-root'
    Remove-Item -LiteralPath Env:HF_MODULES_CACHE -ErrorAction SilentlyContinue
    $beforeEntry = Get-ProcessSnapshot
    $customRoot = Join-Path $testResourceRoot 'audit\context-override-check'
    . $enter -ResourceRoot $customRoot -NoDirectoryCreation -Quiet
    $current = Get-ProcessSnapshot
    foreach ($name in $clearNames) {
        Assert-Check (-not $current.ContainsKey($name)) "entry removes inherited $name key"
    }
    Assert-Check ($env:SEGFORMER_RESOURCE -eq $customRoot) 'entry exports the selected resource root to Python'
    Restore-SegProcessEnvironment
    Assert-Check (Test-SnapshotEqual $beforeEntry) 'entry restoration recreates the exact environment key set and values'
    Set-Location -LiteralPath $initialLocation

    # Use real conda run and Python to check native argument forwarding.
    $beforeRunner = Get-ProcessSnapshot
    $beforeLocation = Get-Location
    [Console]::OutputEncoding = New-Object System.Text.ASCIIEncoding
    $OutputEncoding = New-Object System.Text.ASCIIEncoding
    $consoleCodePage = [Console]::OutputEncoding.CodePage
    $pipelineCodePage = $OutputEncoding.CodePage
    $code = 'import json,sys; from segformer_lab import paths; print(json.dumps(dict(argv=sys.argv[1:],root=str(paths.RESOURCE_ROOT))))'
    $result = & $runner -ResourceRoot $testResourceRoot -Quiet -PythonArgs @('-c', $code, 'space value', '--flag')
    $runnerExit = $LASTEXITCODE
    $facts = ($result -join "`n") | ConvertFrom-Json
    Assert-Check ($runnerExit -eq 0) 'runner returns a successful child exit code'
    Assert-Check (($facts.argv.Count -eq 2) -and ($facts.argv[0] -ceq 'space value') -and ($facts.argv[1] -ceq '--flag')) 'runner preserves argument boundaries'
    Assert-Check ($facts.root -eq $testResourceRoot) 'Python paths uses the resource root selected by the runner'
    Assert-Check (Test-SnapshotEqual $beforeRunner) 'runner restores the exact caller environment'
    Assert-Check ((Get-Location).Path -eq $beforeLocation.Path) 'runner restores the caller location'
    Assert-Check ([Console]::OutputEncoding.CodePage -eq $consoleCodePage) 'runner restores console encoding'
    Assert-Check ($OutputEncoding.CodePage -eq $pipelineCodePage) 'runner preserves caller pipeline encoding'
    & $runner -ResourceRoot $testResourceRoot -Quiet -PythonArgs @('-c', 'import sys; sys.exit(17)')
    Assert-Check ($LASTEXITCODE -eq 17) 'runner returns the real nonzero Python exit code'
    $PSNativeCommandUseErrorActionPreference = $true
    $nativePreferenceOk = $false
    try {
        & $runner -ResourceRoot $testResourceRoot -Quiet -PythonArgs @('-c', 'import sys; sys.exit(17)')
        $nativePreferenceOk = ($LASTEXITCODE -eq 17)
    }
    catch { Write-Output ('Native-preference error: ' + $_.Exception.Message) }
    Assert-Check $nativePreferenceOk 'runner returns nonzero exit codes when caller enables native error actions'
    $PSNativeCommandUseErrorActionPreference = $false

    $emptyArgsOk = $false
    try {
        $emptyResult = & $runner -ResourceRoot $testResourceRoot -Quiet -PythonArgs @('-c', 'import sys; print(len(sys.argv)); print(repr(sys.argv[1]))', '')
        $emptyArgsOk = (($emptyResult[0] -eq '2') -and ($emptyResult[1] -eq "''"))
    }
    catch { Write-Output ('Empty-argument error: ' + $_.Exception.Message) }
    Assert-Check $emptyArgsOk 'runner preserves an empty Python argument'

    # An existing file cannot act as a resource directory. Compare the direct
    # entry error with the wrapper error to catch a masking finally failure.
    $invalidRoot = Join-Path $testResourceRoot 'env\python.exe'
    $directError = ''
    try { . $enter -ResourceRoot $invalidRoot -Quiet }
    catch { $directError = $_.Exception.Message }
    $beforeFailure = Get-ProcessSnapshot
    $wrappedError = ''
    try { & $runner -ResourceRoot $invalidRoot -Quiet -PythonArgs @('-V') }
    catch { $wrappedError = $_.Exception.Message }
    Assert-Check (($directError -ne '') -and ($wrappedError -eq $directError)) 'runner preserves an entry failure instead of masking it in finally'
    Assert-Check (Test-SnapshotEqual $beforeFailure) 'runner preserves caller environment when entry fails'
}
finally {
    foreach ($key in @([Environment]::GetEnvironmentVariables('Process').Keys)) {
        if (-not $initialEnv.ContainsKey($key)) { Remove-Item -LiteralPath ('Env:' + $key) -ErrorAction SilentlyContinue }
    }
    foreach ($key in $initialEnv.Keys) { [Environment]::SetEnvironmentVariable($key, $initialEnv[$key], 'Process') }
    Set-Location -LiteralPath $initialLocation
    [Console]::OutputEncoding = $initialConsoleEncoding
    $OutputEncoding = $initialOutputEncoding
}

if ($failures.Count -gt 0) { throw "$($failures.Count) runtime context checks failed." }
Write-Output 'All runtime context checks passed.'
