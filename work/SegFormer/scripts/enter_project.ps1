<#
.SYNOPSIS
    Load the SegFormer project runtime context into the current PowerShell process.

.DESCRIPTION
    Establishes, for THIS process only:
      * project and resource-root variables ($SegProject, $SegResource, $SegEnv,
        $SegPython, $SegOutput, cache and temp directories);
      * isolated cache/temp/package directories for Conda, pip, Hugging Face,
        PyTorch, CUDA JIT, Matplotlib and Python bytecode;
      * Conda settings that stop environment registration and channel notices;
      * wrapper commands Invoke-SegPython / Invoke-SegPip that always run inside
        the project prefix through "conda run --prefix".

    It never installs, upgrades or removes packages, never edits base or other
    environments, and never writes persistent (user/machine) environment
    variables, PATH entries or shell profiles.

    Safe to run repeatedly: directories are created when missing, and the
    project condarc is only created when absent (an existing file is compared,
    never overwritten).

.EXAMPLE
    Set-Location -LiteralPath '<project>'
    . '.\scripts\enter_project.ps1'
    Invoke-SegPython -c "import sys; print(sys.executable)"

.EXAMPLE
    . '.\scripts\enter_project.ps1' -ResourceRoot 'D:\segformer_resources' -Quiet

.NOTES
    ASCII-only source on purpose: Windows PowerShell 5.1 misreads UTF-8 files
    without a byte-order mark, and this repository lives under a non-ASCII path.
#>

[CmdletBinding()]
param(
    [string]$ResourceRoot = '',
    [string]$CondaExe = 'E:\Anaconda\Scripts\conda.exe',
    [switch]$NoDirectoryCreation,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------- paths ----
$SegProject = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $SegProject 'AGENTS.md'))) {
    throw "Not a SegFormer project directory (AGENTS.md missing): $SegProject"
}

if ([string]::IsNullOrWhiteSpace($ResourceRoot)) {
    $repo = Split-Path -Parent (Split-Path -Parent $SegProject)   # work -> AI_Try
    $workspace = Split-Path -Parent $repo                        # AI_Try -> AI
    $ResourceRoot = Join-Path $workspace 'segformer_resources'
}

$SegConda = $CondaExe
$SegResource = [IO.Path]::GetFullPath($ResourceRoot)
$SegEnv = Join-Path $SegResource 'env'
$SegPython = Join-Path $SegEnv 'python.exe'
$SegOutput = Join-Path $SegResource 'runs'
$SegCache = Join-Path $SegResource 'cache'
$SegTmp = Join-Path $SegResource 'tmp'
$SegData = Join-Path $SegResource 'data'
$SegModels = Join-Path $SegResource 'models'
$SegAudit = Join-Path $SegResource 'audit'
$SegConfigDir = Join-Path $SegResource 'config'

if (-not (Test-Path -LiteralPath $SegConda)) {
    throw "Conda executable not found: $SegConda"
}

# ------------------------------------------------------------ directories ----
$SegSubdirs = @(
    'cache\conda_pkgs',
    'cache\conda_envs',
    'cache\pip',
    'cache\huggingface\hub',
    'cache\huggingface\xet',
    'cache\huggingface\assets',
    'cache\huggingface\datasets',
    'cache\huggingface\modules',
    'cache\torch',
    'cache\torch_extensions',
    'cache\cuda',
    'cache\xdg',
    'cache\matplotlib',
    'cache\pycache',
    'cache\wheels',
    'config',
    'tmp',
    'data\visa',
    'models',
    'runs',
    'audit'
)
$SegCreatedDirs = @()
if (-not $NoDirectoryCreation) {
    foreach ($sub in $SegSubdirs) {
        $full = Join-Path $SegResource $sub
        if (-not (Test-Path -LiteralPath $full)) {
            New-Item -ItemType Directory -Force -Path $full | Out-Null
            $SegCreatedDirs += $sub
        }
    }
}

# ------------------------------------------------- project condarc creation ----
$SegCondarcPath = Join-Path $SegConfigDir 'condarc.yml'
$SegCondarcExpected = @(
    'channels:',
    '  - https://repo.anaconda.com/pkgs/main',
    'channel_priority: strict',
    'auto_activate_base: false',
    'create_default_packages: []'
)
$SegCondarcState = 'existing'
if (-not (Test-Path -LiteralPath $SegCondarcPath)) {
    if (-not $NoDirectoryCreation) {
        New-Item -ItemType Directory -Force -Path $SegConfigDir | Out-Null
        [IO.File]::WriteAllText($SegCondarcPath, (($SegCondarcExpected -join "`r`n") + "`r`n"), (New-Object System.Text.UTF8Encoding($false)))
        $SegCondarcState = 'created'
    }
    else {
        $SegCondarcState = 'missing'
    }
}
else {
    $SegCondarcActual = @(Get-Content -LiteralPath $SegCondarcPath | Where-Object { $_.Trim() -ne '' } | ForEach-Object { $_.TrimEnd() })
    if (($SegCondarcActual -join "`n") -ne ($SegCondarcExpected -join "`n")) {
        $SegCondarcState = 'differs'
        Write-Warning "Project condarc differs from the documented content; it was NOT overwritten: $SegCondarcPath"
    }
}

# --------------------------------------------- process environment snapshot ----
# Snapshot first so Restore-SegProcessEnvironment can undo every change.
$SegEnvSnapshot = @{}
foreach ($k in [System.Environment]::GetEnvironmentVariables('Process').Keys) {
    $SegEnvSnapshot[$k] = [string][System.Environment]::GetEnvironmentVariable($k, 'Process')
}

$SegClearEnvNames = @(
    'PYTHONPATH',
    'PYTHONHOME',
    'PYTHONDONTWRITEBYTECODE',
    'TRANSFORMERS_CACHE',
    'PYTORCH_TRANSFORMERS_CACHE',
    'PYTORCH_PRETRAINED_BERT_CACHE',
    'HF_HOME',
    'HF_HUB_CACHE',
    'HF_XET_CACHE',
    'HF_ASSETS_CACHE',
    'HF_DATASETS_CACHE',
    'HF_MODULES_CACHE',
    'CONDARC',
    'CONDA_PKGS_DIRS',
    'CONDA_ENVS_PATH',
    'CONDA_REGISTER_ENVS',
    'CONDA_NUMBER_CHANNEL_NOTICES',
    'PIP_CONFIG_FILE',
    'PIP_CACHE_DIR',
    'TMPDIR'
)
foreach ($name in $SegClearEnvNames) {
    # On current .NET, SetEnvironmentVariable(name, $null, 'Process') can
    # leave an empty value. Libraries distinguish an empty key from no key.
    # The existence guard is required: this machine rejects Remove-Item on a
    # path that does not exist instead of honouring -ErrorAction.
    if ($null -ne [System.Environment]::GetEnvironmentVariable($name, 'Process')) {
        Remove-Item -LiteralPath ('Env:' + $name) -ErrorAction SilentlyContinue
    }
}
foreach ($k in @($SegEnvSnapshot.Keys)) {
    if ($k -like 'PIP_*') {
        if ($null -ne [System.Environment]::GetEnvironmentVariable($k, 'Process')) {
            Remove-Item -LiteralPath ('Env:' + $k) -ErrorAction SilentlyContinue
        }
    }
}

$env:SEGFORMER_RESOURCE = $SegResource
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $SegCache 'pycache'

# Windows pip disables its configuration file when PIP_CONFIG_FILE resolves to
# the null device. The documented spelling is the lowercase 'nul'.
$env:PIP_CONFIG_FILE = 'nul'
$env:PIP_CACHE_DIR = Join-Path $SegCache 'pip'

$env:CONDA_PKGS_DIRS = Join-Path $SegCache 'conda_pkgs'
$env:CONDA_ENVS_PATH = Join-Path $SegCache 'conda_envs'
$env:CONDA_REGISTER_ENVS = 'false'
$env:CONDA_NUMBER_CHANNEL_NOTICES = '0'
$env:CONDARC = $SegCondarcPath

$env:HF_HOME = Join-Path $SegCache 'huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_XET_CACHE = Join-Path $env:HF_HOME 'xet'
$env:HF_ASSETS_CACHE = Join-Path $env:HF_HOME 'assets'
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME 'datasets'
$env:HF_MODULES_CACHE = Join-Path $env:HF_HOME 'modules'

$env:TORCH_HOME = Join-Path $SegCache 'torch'
$env:TORCH_EXTENSIONS_DIR = Join-Path $SegCache 'torch_extensions'
$env:CUDA_CACHE_PATH = Join-Path $SegCache 'cuda'
$env:XDG_CACHE_HOME = Join-Path $SegCache 'xdg'
$env:MPLCONFIGDIR = Join-Path $SegCache 'matplotlib'
$env:MPLBACKEND = 'Agg'
$env:TEMP = $SegTmp
$env:TMP = $SegTmp
# Python's tempfile checks TMPDIR before TEMP/TMP on every platform, so an
# inherited TMPDIR would silently send project temp files to the system disk.
$env:TMPDIR = $SegTmp

# Console decoding only (this process's console handle, not a system setting).
# Python is asked to emit UTF-8 and Conda already emits UTF-8, so the default
# GBK console would otherwise print the Chinese path segments as garbage.
try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $OutputEncoding = [Console]::OutputEncoding
}
catch {
    Write-Verbose 'Could not switch console output encoding to UTF-8.'
}

Set-Location -LiteralPath $SegProject

# ------------------------------------------------------------- functions ----
function Invoke-SegPython {
    <#
        Run the project interpreter inside its Conda prefix context.
        All arguments are forwarded verbatim, e.g.
            Invoke-SegPython -c "import sys; print(sys.executable)"
            Invoke-SegPython -m segformer_lab.env_check --out 'reports\environment.json'
    #>
    & $script:SegConda run --no-capture-output --prefix $script:SegEnv python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Project python failed with exit code $LASTEXITCODE"
    }
}

function Invoke-SegPip {
    <#
        Run pip for the project interpreter only, never the user site.
        Example: Invoke-SegPip install --no-user <packages> --index-url <url>
    #>
    & $script:SegConda run --no-capture-output --prefix $script:SegEnv python -m pip --cache-dir (Join-Path $script:SegCache 'pip') @args
    if ($LASTEXITCODE -ne 0) {
        throw "Project pip failed with exit code $LASTEXITCODE"
    }
}

function Assert-SegCondaIsolation {
    <#
        Fail fast when Conda would still merge shared package caches, register
        the project environment in user metadata, or print channel notices.
        Must pass before "conda create".
    #>
    $json = & $script:SegConda config --show pkgs_dirs register_envs number_channel_notices --json 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot read effective Conda configuration: $json"
    }
    $effective = $json | ConvertFrom-Json

    $effectivePkgs = @($effective.pkgs_dirs)
    $expectedPkgs = [IO.Path]::GetFullPath((Join-Path $script:SegCache 'conda_pkgs'))
    if ($effectivePkgs.Count -ne 1) {
        throw "Conda still merges $($effectivePkgs.Count) package caches ($($effectivePkgs -join '; ')). Fix the configuration source before creating an environment."
    }
    if (-not [string]::Equals([IO.Path]::GetFullPath([string]$effectivePkgs[0]), $expectedPkgs, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Conda package cache does not point at the project directory: $($effectivePkgs[0])"
    }
    if ($effective.register_envs -ne $false) {
        throw "CONDA_REGISTER_ENVS is not effective (register_envs=$($effective.register_envs))."
    }
    if ([int]$effective.number_channel_notices -ne 0) {
        throw "Channel notices are not suppressed (number_channel_notices=$($effective.number_channel_notices))."
    }

    Write-Output ("pkgs_dirs              : {0}" -f $effectivePkgs[0])
    Write-Output ("register_envs          : {0}" -f $effective.register_envs)
    Write-Output ("number_channel_notices : {0}" -f $effective.number_channel_notices)
}

function Restore-SegProcessEnvironment {
    <#
        Undo the process-scope changes made above. Used by run_project.ps1 so a
        wrapper invocation leaves the caller's environment untouched.
    #>
    $current = @([System.Environment]::GetEnvironmentVariables('Process').Keys)
    foreach ($k in $current) {
        if (-not $script:SegEnvSnapshot.ContainsKey($k)) {
            Remove-Item -LiteralPath ('Env:' + $k) -ErrorAction SilentlyContinue
        }
        elseif ([string][System.Environment]::GetEnvironmentVariable($k, 'Process') -ne [string]$script:SegEnvSnapshot[$k]) {
            [System.Environment]::SetEnvironmentVariable($k, [string]$script:SegEnvSnapshot[$k], 'Process')
        }
    }
    foreach ($k in $script:SegEnvSnapshot.Keys) {
        if (-not ($current -contains $k)) {
            [System.Environment]::SetEnvironmentVariable($k, [string]$script:SegEnvSnapshot[$k], 'Process')
        }
    }
}

# ---------------------------------------------------------------- report ----
if (-not $Quiet) {
    Write-Output ''
    Write-Output 'SegFormer project context loaded (this process only)'
    Write-Output ("  project  : {0}" -f $SegProject)
    Write-Output ("  resource : {0}" -f $SegResource)
    Write-Output ("  env      : {0}" -f $SegEnv)
    Write-Output ("  python   : {0} (exists={1})" -f $SegPython, (Test-Path -LiteralPath $SegPython))
    Write-Output ("  output   : {0}" -f $SegOutput)
    Write-Output ("  temp     : {0}" -f $SegTmp)
    Write-Output ("  condarc  : {0} [{1}]" -f $SegCondarcPath, $SegCondarcState)
    if ($SegCreatedDirs.Count -gt 0) {
        Write-Output ("  created  : {0} directorie(s)" -f $SegCreatedDirs.Count)
    }
    Write-Output '  commands : Invoke-SegPython, Invoke-SegPip, Assert-SegCondaIsolation, Restore-SegProcessEnvironment'
    Write-Output ''
}
