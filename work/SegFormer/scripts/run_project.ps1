<#
.SYNOPSIS
    Run a command with the SegFormer project context, from any new process.

.DESCRIPTION
    Every new tool/terminal process starts without the project context. This
    wrapper loads scripts\enter_project.ps1, runs the requested Python arguments
    with "conda run --prefix <env>", returns the real exit code, and restores
    the caller's environment variables on the way out.

    Use it from automation or a fresh shell:
        & '.\scripts\run_project.ps1' -PythonArgs @('-m','segformer_lab.env_check','--out','reports/environment.json')

    Do not use it to install packages into base or another environment; it only
    knows the project prefix.

.PARAMETER PythonArgs
    Arguments forwarded verbatim to the project interpreter.

.PARAMETER ResourceRoot
    Optional resource-root override forwarded to enter_project.ps1.

.PARAMETER Quiet
    Suppress the context banner printed by enter_project.ps1.

.NOTES
    ASCII-only source on purpose: Windows PowerShell 5.1 misreads UTF-8 files
    without a byte-order mark, and this repository lives under a non-ASCII path.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [AllowEmptyString()]
    [string[]]$PythonArgs,

    [string]$ResourceRoot = '',

    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
# Return native failures as exit codes even if a PowerShell 7 caller enables
# error actions for native programs. This variable is local to this script.
$PSNativeCommandUseErrorActionPreference = $false

$enterScript = Join-Path $PSScriptRoot 'enter_project.ps1'
if (-not (Test-Path -LiteralPath $enterScript)) {
    throw "Missing project entry script: $enterScript"
}

$originalLocation = Get-Location
$originalConsoleEncoding = [Console]::OutputEncoding
$originalOutputEncoding = $OutputEncoding
# Own this snapshot instead of relying on a restore function from the entry
# script: setup can throw before that function is defined.
$originalEnvironment = @{}
foreach ($key in [System.Environment]::GetEnvironmentVariables('Process').Keys) {
    $originalEnvironment[$key] = [System.Environment]::GetEnvironmentVariable($key, 'Process')
}
$exitCode = 1

try {
    if ([string]::IsNullOrWhiteSpace($ResourceRoot)) {
        . $enterScript -Quiet:$Quiet
    }
    else {
        . $enterScript -ResourceRoot $ResourceRoot -Quiet:$Quiet
    }

    # Windows PowerShell's legacy native binder drops empty array elements.
    # A quoted empty argument survives that binder; PowerShell 7 preserves the
    # original empty string itself.
    if ($PSVersionTable.PSVersion.Major -lt 7) {
        $PythonArgs = @($PythonArgs | ForEach-Object { if ($_ -eq '') { '""' } else { $_ } })
    }
    & $SegConda run --no-capture-output --prefix $SegEnv python @PythonArgs
    $exitCode = $LASTEXITCODE
}
finally {
    $currentKeys = @([System.Environment]::GetEnvironmentVariables('Process').Keys)
    foreach ($key in $currentKeys) {
        if (-not $originalEnvironment.ContainsKey($key)) {
            Remove-Item -LiteralPath ('Env:' + $key) -ErrorAction SilentlyContinue
        }
    }
    foreach ($key in $originalEnvironment.Keys) {
        # Do not rewrite unchanged empty values: .NET Framework (PowerShell
        # 5.1) interprets setting an empty value as deleting that key.
        if (-not ($currentKeys -contains $key) -or
            [string][System.Environment]::GetEnvironmentVariable($key, 'Process') -cne [string]$originalEnvironment[$key]) {
            [System.Environment]::SetEnvironmentVariable($key, $originalEnvironment[$key], 'Process')
        }
    }
    Set-Location -LiteralPath $originalLocation
    [Console]::OutputEncoding = $originalConsoleEncoding
    $OutputEncoding = $originalOutputEncoding
}

exit $exitCode
