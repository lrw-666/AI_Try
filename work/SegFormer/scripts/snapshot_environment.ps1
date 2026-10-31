<#
.SYNOPSIS
    Read-only environment snapshot for the SegFormer project.

.DESCRIPTION
    Collects Conda environments, base package list, condarc hashes, Conda
    effective configuration, pip configuration, system/GPU facts and the
    relevant inherited process variables, then writes JSON/text files into
    <resource-root>\audit\<label>-*.json.

    The script is read-only with respect to environments and global config:
    it never installs, removes or edits packages, and never writes outside the
    audit directory.

    Run it BEFORE any environment creation (-Label before) and AFTER it
    (-Label after), then compare the two sets.

.PARAMETER Label
    Snapshot label used as the file name prefix. Default: before.

.PARAMETER ResourceRoot
    Project resource root. Default: computed from this repo layout as
    <workspace>\segformer_resources.

.PARAMETER CondaExe
    Conda executable path. Default: E:\Anaconda\Scripts\conda.exe

.PARAMETER ExtraConfigPath
    Additional condarc files to hash. Defaults cover base and user condarc.

.NOTES
    ASCII-only source on purpose: Windows PowerShell 5.1 misreads UTF-8 files
    without a byte-order mark, and this repository lives under a non-ASCII path.
#>

[CmdletBinding()]
param(
    [string]$Label = 'before',
    [string]$ResourceRoot = '',
    [string]$CondaExe = 'E:\Anaconda\Scripts\conda.exe',
    [string]$BasePrefix = 'E:\Anaconda',
    [string[]]$ExtraConfigPath = @()
)

$ErrorActionPreference = 'Stop'

$scriptDir = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ResourceRoot)) {
    $project = Split-Path -Parent $scriptDir
    $repo = Split-Path -Parent (Split-Path -Parent $project)
    $workspace = Split-Path -Parent $repo
    $ResourceRoot = Join-Path $workspace 'segformer_resources'
}
$auditDir = Join-Path $ResourceRoot 'audit'
New-Item -ItemType Directory -Force -Path $auditDir | Out-Null

function Write-JsonFile {
    param([string]$Path, $Object)
    $json = $Object | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText($Path, $json, (New-Object System.Text.UTF8Encoding($false)))
}

function Write-TextFile {
    param([string]$Path, [string]$Text)
    [IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

$prefix = Join-Path $auditDir $Label

# 1. Registered Conda environments.
$envListJson = & $CondaExe env list --json 2>&1
if ($LASTEXITCODE -ne 0) { throw "conda env list failed: $envListJson" }
Write-TextFile -Path "$prefix-conda-envs.json" -Text $envListJson
$envList = $envListJson | ConvertFrom-Json

# 2. Base environment package list (name/version/build/channel only).
$basePkgsJson = & $CondaExe list --prefix $BasePrefix --json 2>&1
if ($LASTEXITCODE -ne 0) { throw "conda list --prefix $BasePrefix failed: $basePkgsJson" }
$basePkgs = $basePkgsJson | ConvertFrom-Json
$baseFlat = $basePkgs | ForEach-Object { "$($_.name)|$($_.version)|$($_.build_string)|$($_.channel)" } | Sort-Object
Write-TextFile -Path "$prefix-base-packages.txt" -Text ($baseFlat -join "`r`n")
Write-TextFile -Path "$prefix-base-packages.json" -Text $basePkgsJson

# 3. Hash of every condarc file that can influence this machine.
$condarcCandidates = @(
    (Join-Path $BasePrefix '.condarc'),
    (Join-Path $env:USERPROFILE '.condarc'),
    (Join-Path $ResourceRoot 'config\condarc.yml')
) + $ExtraConfigPath
$condarcInfo = foreach ($p in ($condarcCandidates | Select-Object -Unique)) {
    if (Test-Path -LiteralPath $p) {
        $h = Get-FileHash -LiteralPath $p -Algorithm SHA256
        [pscustomobject]@{ path = $p; sha256 = $h.Hash; length = (Get-Item -LiteralPath $p).Length }
    }
    else {
        [pscustomobject]@{ path = $p; sha256 = $null; length = $null; state = 'missing' }
    }
}
Write-JsonFile -Path "$prefix-condarc-hashes.json" -Object @($condarcInfo)

# 4. Effective Conda configuration (channels/paths/priority keys only).
$configKeys = @('channels', 'channel_priority', 'pkgs_dirs', 'envs_dirs', 'auto_activate_base', 'create_default_packages', 'register_envs', 'number_channel_notices')
$configJson = & $CondaExe config --show @configKeys --json 2>&1
if ($LASTEXITCODE -ne 0) { throw "conda config --show failed: $configJson" }
Write-TextFile -Path "$prefix-conda-config.json" -Text $configJson

# 5. Hash of the pip configuration files that would apply outside the project.
$pipFiles = @(
    (Join-Path $env:APPDATA 'pip\pip.ini'),
    (Join-Path $env:USERPROFILE 'pip\pip.ini'),
    'C:\ProgramData\pip\pip.ini'
)
$pipInfo = foreach ($p in $pipFiles) {
    if (Test-Path -LiteralPath $p) {
        [pscustomobject]@{ path = $p; sha256 = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash }
    }
}
Write-JsonFile -Path "$prefix-pip-config.json" -Object @($pipInfo)

# 6. System, GPU and interpreter facts.
$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$gpuRows = @()
$smi = & nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>&1
if ($LASTEXITCODE -eq 0) { $gpuRows = @($smi) }

$pathPython = Get-Command python -ErrorAction SilentlyContinue
$pathPip = Get-Command pip -ErrorAction SilentlyContinue
$system = [pscustomobject]@{
    captured_at_local = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz')
    label             = $Label
    os_caption        = $os.Caption
    os_version        = $os.Version
    os_build          = $os.BuildNumber
    os_arch           = $os.OSArchitecture
    ram_total_gib     = [math]::Round($cs.TotalPhysicalMemory / 1GB, 2)
    gpu               = $gpuRows
    conda_exe         = $CondaExe
    conda_version     = (& $CondaExe --version 2>&1)
    base_prefix       = $BasePrefix
    base_python       = (& (Join-Path $BasePrefix 'python.exe') -c "import sys;print(sys.version)" 2>&1)
    path_python       = if ($pathPython) { $pathPython.Source } else { $null }
    path_pip          = if ($pathPip) { $pathPip.Source } else { $null }
    registered_envs   = @($envList.envs)
    disks = @(Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free -ne $null } | ForEach-Object {
            [pscustomobject]@{ name = $_.Name; free_gib = [math]::Round($_.Free / 1GB, 2) }
        })
}
Write-JsonFile -Path "$prefix-system.json" -Object $system

# 7. Inherited process variables that can redirect caches, packages or proxies.
#    Values recorded as-is except for anything that looks like a secret.
$relevant = '^(PYTHON|PIP_|CONDA|HF_|HUGGINGFACE|TORCH|TRANSFORMERS|CUDA_|XDG|MPL|TEMP$|TMP$|TMPDIR|HTTP_PROXY|HTTPS_PROXY|NO_PROXY|ALL_PROXY)'
$procVars = [System.Environment]::GetEnvironmentVariables('Process')
$recorded = [ordered]@{}
foreach ($k in ($procVars.Keys | Sort-Object)) {
    if ($k -match $relevant) {
        $v = [string]$procVars[$k]
        if ($k -match 'TOKEN|KEY|SECRET|PASSWORD') { $v = '<redacted>' }
        $recorded[$k] = $v
    }
}
if ($recorded.Contains('HTTP_PROXY')) { $recorded['HTTP_PROXY'] = '<set: local proxy>' }
if ($recorded.Contains('HTTPS_PROXY')) { $recorded['HTTPS_PROXY'] = '<set: local proxy>' }
Write-JsonFile -Path "$prefix-process-env.json" -Object $recorded

Write-Output ("snapshot label : {0}" -f $Label)
Write-Output ("audit dir      : {0}" -f $auditDir)
Write-Output ("registered envs: {0}" -f (($envList.envs) -join '; '))
Write-Output ("base packages  : {0}" -f $baseFlat.Count)
Write-Output ("files written  : {0}-*.json / {0}-*.txt" -f $prefix)
