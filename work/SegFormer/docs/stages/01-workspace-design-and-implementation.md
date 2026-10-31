# 阶段 01：工作区准备与项目启动入口——设计与实现学习文档

| 字段 | 填写内容 |
| --- | --- |
| 阶段编号与名称 | 步骤 1：统一目录、解释器与 Agent 工作约定 |
| 状态 | 已验证（工作区、启动入口、快照三部分均已实际运行；未创建数据或模型产物） |
| 创建/更新时间 | 2026-10-31 12:50 +08:00 |
| 对应代码身份 | 仓库 `E:\桌面\AI\AI_Try`，分支 `master`，HEAD `dfe8fa1`（2026-10-30 08:08:35 +0800，提交信息「方向修改」）；本阶段新增文件均未提交，哈希见第 4 节 |
| 项目根目录 / 资源根目录 | `E:\桌面\AI\AI_Try\work\SegFormer` / `E:\桌面\AI\segformer_resources` |
| Python / 环境prefix | 本阶段创建入口时环境尚未建立；阶段末环境为 `E:\桌面\AI\segformer_resources\env\python.exe`（Python 3.12.16），详见阶段 02 |
| 依赖的上一阶段 | 无。输入为三份指南文档与 `work/SegFormer/AGENTS.md` |

**独立验收更新：** 本阶段已完成复核与修正，详见[阶段01—02验收记录](../../reports/2026-10-31-stage01-02-acceptance.md)。原首次运行输出作为历史证据保留；当前状态以修正后的脚本、最新 `reports/environment.json` 与验收记录为准。

## 1. 本阶段解决什么问题

后续十几个阶段都要反复执行同一句话：**「用项目自己的 Python、自己的缓存、自己的输出目录，跑一段代码」**。如果这件事每次靠记忆敲一遍路径和缓存变量，必然出现三类事故：

1. 用了 `PATH` 上的另一个 Python，装完包发现「导入不到」；
2. 缓存、临时文件悄悄写到 `C:\Users\...`，磁盘被占满却找不到是谁写的；
3. 某次实验的产物落在仓库里，被 Git 一起提交。

所以本阶段的产出不是模型，而是**一个可重复加载的运行时上下文**，以及一套**可复查的资源边界**。

输入：已存在的仓库 `AI_Try`、`work\SegFormer`（当时只有 `AGENTS.md`）、尚不存在的资源根目录。
输出：

- 资源根目录骨架 `E:\桌面\AI\segformer_resources\`（env / cache / config / tmp / data / models / runs / audit）；
- 三个 PowerShell 脚本：`enter_project.ps1`（加载上下文）、`run_project.ps1`（新进程包装）、`snapshot_environment.ps1`（只读快照）；
- `segformer_lab.paths` 模块：Python 侧唯一的路径真源；
- 搭建前快照 `audit\before-*`，供阶段 02 做前后比对。

边界：本阶段**不创建** Conda 环境、**不安装**任何包、**不下载**模型或数据。用户明确要求「本次只做步骤 1—2」，且「已有文件或环境先核对，不覆盖、不删除」。

## 2. 技术栈与版本

| 技术/库 | 实际版本 | 本阶段作用 | 状态与证据 |
| --- | --- | --- | --- |
| Windows PowerShell | 5.1（`powershell.exe`） | 运行入口脚本、快照脚本 | 已运行 |
| Conda | 24.11.3（`E:\Anaconda\Scripts\conda.exe`） | 提供 `conda run --prefix` 的环境上下文 | 已运行（`before-system.json`） |
| Git | 仓库 HEAD `dfe8fa1` | 记录代码身份 | 已读取（`git rev-parse`） |
| Python（base） | 3.12.7（Anaconda 打包） | 仅作为 Conda 自身的运行宿主，不作为项目解释器 | 已读取 |
| Python（PATH 上的独立安装） | 3.12.0rc3 | **不作为**项目解释器；仅登记为既有安装 | 已读取 |
| NVIDIA 驱动 | 617.42 | 决定可用的 CUDA 运行时上限 | 已读取（`nvidia-smi`） |

版本查询命令（都在项目上下文之外执行，只读）：

```powershell
& 'E:\Anaconda\Scripts\conda.exe' --version          # conda 24.11.3
& 'E:\Anaconda\python.exe' -c "import sys;print(sys.version)"   # 3.12.7
python -c "import sys;print(sys.executable, sys.version)"        # 3.12.0rc3（PATH 顺序在前）
```

## 3. 设计与取舍

### 3.1 目录为什么分成「仓库内」和「仓库外」

```text
E:\桌面\AI\
├─ segformer_resources\          <- 仓库外：体积大、可重建、不进 Git
│  ├─ env\                       独立 Conda 前缀（4.3 GB）
│  ├─ cache\                     conda_pkgs / pip / huggingface / torch / cuda / xdg / matplotlib / pycache / wheels
│  ├─ config\condarc.yml         项目级 Conda 配置（只影响本项目进程）
│  ├─ tmp\                       项目临时文件（含 conda run 的 .bat、Python tempfile）
│  ├─ data\visa\                 后续提取的 PCB1
│  ├─ models\mit-b0\             后续固定 revision 的权重
│  ├─ runs\                      训练、评估、批量预测
│  └─ audit\                     快照、比对、安装日志
└─ AI_Try\work\SegFormer\        <- 仓库内：代码、配置、清单、阶段文档
   ├─ scripts\                   enter_project.ps1 / run_project.ps1 / snapshot_environment.ps1
   ├─ segformer_lab\             __init__.py / paths.py / env_check.py
   ├─ reports\environment.json
   ├─ requirements-lock.txt
   └─ docs\stages\               本文件所在目录
```

判断依据：**能否由代码和文档重建**。代码、配置、清单、文档可以重建环境，进 Git；4 GB 的 env、2.5 GB 的 wheel、后续的权重和预测图不能进 Git。这条线一旦划错，后期要么仓库爆炸，要么换机器无法复现。

### 3.2 为什么用 `--prefix` 而不是命名环境

命名环境（`conda create -n segformer`）依赖 `envs_dirs` 的顺序去猜位置，而本机 `envs_dirs` 有三个候选（`E:\Anaconda\envs`、`C:\Users\Luorenwei\.conda\envs`、`C:\Users\Luorenwei\AppData\Local\conda\conda\envs`），还叠加了用户 `.condarc` 的覆盖。用绝对 `--prefix` 只有一个答案，且可以直接写进脚本和文档。

代价：环境不出现在 `conda env list` 里（因为本阶段同时关闭了环境登记），需要靠 `conda list --prefix` 或本项目脚本查询。这是**有意的取舍**：不让项目环境污染用户的全局环境登记表。

### 3.3 为什么脚本要纯 ASCII

本仓库路径含中文（`E:\桌面\AI`）。Windows PowerShell 5.1 读取**无 BOM 的 UTF-8 脚本**时按系统 ANSI 代码页（GBK）解码，源码里的中文字符串会变成乱码，进而拼出错误路径——这种错误表现为「文件明明存在却找不到」，极难排查。

两种解法：给脚本加 UTF-8 BOM，或脚本内不写非 ASCII 字符。本阶段选后者：脚本注释与输出全英文，中文只出现在 Markdown 文档里（Markdown 由编辑器/浏览器按 UTF-8 读取，不受影响）。已验证三个脚本的非 ASCII 字节数为 0。

### 3.4 为什么要有「快照脚本」而不是几条一次性命令

隔离的结论必须是**可复查的**：「我没有动 base」这句话需要证据。`snapshot_environment.ps1 -Label before|after` 把同一组只读检查跑两次，落到 `audit\` 下，再用 `Compare-Object` 比对。它只读：不装包、不删包、不写环境，唯一写入目标是 `audit\`。

### 3.5 未选方案

| 方案 | 为什么本阶段不选 |
| --- | --- |
| 在 `work\SegFormer` 内建 `venv` | 依赖 base 解释器，且 Conda 包缓存仍会落到全局；与指南的独立前缀方案冲突 |
| 直接 `conda activate` 后手工操作 | 每个新工具进程都要重来，且 activate 会改 PATH，容易与系统 Python 混用 |
| 把环境变量写进用户级 `setx` | 会污染其他项目和终端，属于本任务明令禁止的「修改全局配置」 |
| 用 `pip install --user` 借用用户 site | 用户 site 与项目隔离直接冲突，且 `--user` 会绕过前缀 |

## 4. 文件、接口与数据约定

| 文件/模块 | 新增或修改 | 职责 | 关键入口与输入输出 |
| --- | --- | --- | --- |
| `scripts\enter_project.ps1` | 新增 | 在当前进程建立项目上下文；创建缺失目录；必要时创建 `config\condarc.yml`；定义包装命令 | `. .\scripts\enter_project.ps1 [-ResourceRoot <path>] [-Quiet]`；导出 `$SegProject/$SegResource/$SegEnv/$SegPython/$SegOutput/$SegCache/$SegTmp/$SegData/$SegModels/$SegAudit` |
| `scripts\run_project.ps1` | 新增 | 供新进程一次性调用；结束时恢复调用者环境变量 | `& .\scripts\run_project.ps1 -PythonArgs @('-m','segformer_lab.env_check','--out','reports/environment.json')`；返回真实退出码 |
| `scripts\snapshot_environment.ps1` | 新增 | 只读环境快照，落盘到 `audit\<label>-*` | `-Label before|after`、`-ResourceRoot`、`-CondaExe`、`-BasePrefix` |
| `segformer_lab\__init__.py` | 新增 | 包标识与版本号 | `__version__ = "0.1.0"` |
| `segformer_lab\paths.py` | 新增 | Python 侧唯一路径真源 | `PROJECT_ROOT`、`RESOURCE_ROOT`、`ENV_DIR`、`RUNS_DIR`、`resolve()`、`ensure_parent()` |
| `config\condarc.yml`（资源根内） | 新增（原不存在） | 项目级 Conda 配置 | `channels` / `channel_priority: strict` / `auto_activate_base: false` / `create_default_packages: []` |
| `.gitignore`（仓库根） | 修改 | 忽略 Python 字节码 | 追加 `__pycache__/`、`*.pyc` |

文件哈希（SHA-256，供后续核对「文档描述的代码」与「实际运行的代码」是否同一份）：

```text
19d5ef5924568181669f03049bf0d8b2204036db19ee3ad78adaa5e3490b92d8  scripts/enter_project.ps1   (325 行；阶段 03 修复后)
91b7b211b8edf159e61e8e9215f63486b907a051b0f564cb8cf045c175d498b1  scripts/run_project.ps1     ( 66 行)
0e6b140ebc50e34d4f3cc375386187d382805e0bc1a191b076f75d8f529783cd  scripts/snapshot_environment.ps1 (171 行)
2ec79c144804c61e26f957553f5835f9c42f3aa0010724f48c029c669c576a98  segformer_lab/__init__.py
a5d8814ff3adb4b162b96202d2c42d7fd68a1f26442e114f0cfa7f479c80c8b9  segformer_lab/paths.py      ( 69 行)
```

### 4.1 一次会话里到底有哪些变量

| 变量 | 值（本机实测） | 用途 |
| --- | --- | --- |
| `$SegProject` | `E:\桌面\AI\AI_Try\work\SegFormer` | 代码、配置、清单、文档 |
| `$SegResource` | `E:\桌面\AI\segformer_resources` | 一切大资源 |
| `$SegEnv` | `...\segformer_resources\env` | Conda 前缀 |
| `$SegPython` | `...\env\python.exe` | 项目解释器 |
| `$SegOutput` | `...\segformer_resources\runs` | 运行产物根 |
| `$SegCache` | `...\segformer_resources\cache` | 全部缓存 |
| `$SegTmp` | `...\segformer_resources\tmp` | 临时文件 |

## 5. 关键实现讲解

### 5.1 路径不写死，从脚本位置推出来

`scripts\enter_project.ps1`：

```powershell
$SegProject = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $SegProject 'AGENTS.md'))) {
    throw "Not a SegFormer project directory (AGENTS.md missing): $SegProject"
}

if ([string]::IsNullOrWhiteSpace($ResourceRoot)) {
    $repo = Split-Path -Parent (Split-Path -Parent $SegProject)   # work -> AI_Try
    $workspace = Split-Path -Parent $repo                        # AI_Try -> AI
    $ResourceRoot = Join-Path $workspace 'segformer_resources'
}
```

为什么要这样：脚本里一旦写死 `E:\桌面\AI\...`，仓库换位置就得改所有脚本，而中文路径改错一次就很难发现。用 `$PSScriptRoot` 反推，配合 `AGENTS.md` 存在性检查，比「猜目录名」可靠：如果脚本被复制到别处运行，会立刻报错而不是默默操作错误目录。`-ResourceRoot` 参数保留了覆盖能力（例如以后换到 ASCII 盘排错）。

### 5.2 先快照、再改环境变量

同一文件，环境变量处理的前半段：

```powershell
$SegEnvSnapshot = @{}
foreach ($k in [System.Environment]::GetEnvironmentVariables('Process').Keys) {
    $SegEnvSnapshot[$k] = [string][System.Environment]::GetEnvironmentVariable($k, 'Process')
}
```

要点：**改之前先把整个进程环境存一份**。交互入口提供 `Restore-SegProcessEnvironment`，用于恢复进程环境；当前包装器还独立保存自己的快照，在 `finally` 中恢复环境、工作目录和编码，避免入口尚未定义恢复函数就失败时掩盖原始报错。PowerShell 7 中需使用 Env provider 真正删除变量，不能把设为 `$null` 当作键必然消失。修正后的两种 PowerShell 回归见验收记录。

### 5.3 为什么必须清 `TMPDIR`，而不只是设 `TEMP`

第一版只设了 `TEMP`/`TMP`，自检时报出 `temp dir : C:\Users\LUOREN~1\AppData\Local\Temp`。原因是 Python 的 `tempfile` 在 Windows 上也优先读 `TMPDIR`：

```python
# Python 标准库 tempfile._candidate_tempdir_list() 的查找顺序
# 1) TMPDIR  2) TEMP  3) TMP  4) 系统默认
```

本机从上层继承了 `TMPDIR=C:\Users\LUOREN~1\AppData\Local\Temp`，于是「设了 TEMP 也没用」。修正为同时设置三个变量：

```powershell
$env:TEMP = $SegTmp
$env:TMP = $SegTmp
$env:TMPDIR = $SegTmp
```

这条经验的价值在于：**变量设了不等于库会读它**。验证必须问库本身（`tempfile.gettempdir()`、`torch.hub.get_dir()`、`huggingface_hub.constants.HF_HUB_CACHE`），不能只看 `os.environ`。阶段 02 的 `env_check.py` 就是按这个原则写的。

### 5.4 包装命令为什么用 `conda run --prefix`

```powershell
function Invoke-SegPython {
    & $script:SegConda run --no-capture-output --prefix $script:SegEnv python @args
    if ($LASTEXITCODE -ne 0) { throw "Project python failed with exit code $LASTEXITCODE" }
}
```

直接执行 `...\env\python.exe` 能跑起来，但缺 Conda 为 Windows 准备的 DLL 搜索路径与运行上下文。`conda run --prefix` 提供同样的上下文，又不需要 `activate`（不改进程 PATH，不影响其他项目）。函数不带 `param()` 块是刻意的：PowerShell 会把 `-c`、`--out` 这类参数原样放进 `$args`，不会被当成函数自身的命名参数吞掉。

### 5.5 `Assert-SegCondaIsolation` 为什么必须在 `create` 之前

Conda 会**合并多层配置**（系统、用户、`CONDARC`、环境变量），设了 `CONDARC` 不等于其他层不生效。本机实测：设了项目 `condarc.yml` 之后，`channels` 仍然混入用户 `.condarc` 的清华镜像与 `linux-64` 频道。

```powershell
$json = & $script:SegConda config --show pkgs_dirs register_envs number_channel_notices --json
# 断言：pkgs_dirs 只有一项，且等于项目缓存目录
# 断言：register_envs = false，number_channel_notices = 0
```

因此创建环境时必须显式写 `--override-channels --channel <url> --strict-channel-priority`，不能依赖「配置里已经写了」。这个函数就是把「以为生效」变成「确认生效」的那道闸。

## 6. 如何操作与实际执行了什么

所有命令的时间为 2026-10-31（本地时间）。执行前系统**不存在** `segformer_resources`，`work\SegFormer` 下**只有** `AGENTS.md`。

### 6.1 盘点（只读）

| 命令/检查 | 实际结果 | 证据路径 | 是否通过 |
| --- | --- | --- | --- |
| `conda --version` | `conda 24.11.3` | `audit\before-system.json` | 通过 |
| `conda env list` | 仅 `base  E:\Anaconda` | `audit\before-conda-envs.json` | 通过 |
| `E:\Anaconda\envs` 子项数 | 0 | 控制台输出 | 通过 |
| `C:\Users\Luorenwei\.conda\envs` | 只有 `.conda_envs_dir_test` | 控制台输出 | 通过 |
| `C:\Users\Luorenwei\AppData\Local\conda\conda\envs` | 不存在 | 控制台输出 | 通过 |
| `Get-Command python` | `C:\Users\Luorenwei\AppData\Local\Programs\Python\Python312\python.exe`（3.12.0rc3） | `before-system.json` `path_python` | 通过 |
| base 关键包 | 有 torch 2.7.1+cu126 / torchvision 0.22.1+cu126 / numpy 1.26.4；无 transformers / tensorboard | 第 8.1 节 | 通过 |
| `nvidia-smi` | RTX 4070 Laptop，8188 MiB，驱动 617.42，cc 8.9 | `before-system.json` `gpu` | 通过 |
| 磁盘 | C 204.66 GiB / D 40.54 / E 163.52 / F 80.99 / G 429.94 可用 | `before-system.json` `disks` | 通过 |
| 旧 venv | `E:\桌面\AI\.venvs`、`AI_Try\.venv` 均不存在 | 控制台输出 | 通过 |

### 6.2 保存搭建前快照

```powershell
& 'E:\桌面\AI\AI_Try\work\SegFormer\scripts\snapshot_environment.ps1' -Label before
```

实际输出：

```text
snapshot label : before
audit dir      : E:\桌面\AI\segformer_resources\audit
registered envs: E:\Anaconda
base packages  : 510
files written  : E:\桌面\AI\segformer_resources\audit\before-*.json / before-*.txt
```

产物：`before-conda-envs.json`、`before-base-packages.json`（142 KB）、`before-base-packages.txt`（510 行）、`before-condarc-hashes.json`、`before-conda-config.json`、`before-pip-config.json`、`before-system.json`、`before-process-env.json`。

### 6.3 建立工作区与启动入口

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'      # 仅本进程，不改系统/用户策略
. 'E:\桌面\AI\AI_Try\work\SegFormer\scripts\enter_project.ps1'
```

实际输出（首次运行，创建 19 个子目录与 `config\condarc.yml`）：

```text
project  : E:\桌面\AI\AI_Try\work\SegFormer
resource : E:\桌面\AI\segformer_resources
env      : E:\桌面\AI\segformer_resources\env
python   : E:\桌面\AI\segformer_resources\env\python.exe (exists=False)
output   : E:\桌面\AI\segformer_resources\runs
temp     : E:\桌面\AI\segformer_resources\tmp
condarc  : E:\桌面\AI\segformer_resources\config\condarc.yml [created]
created  : 19 directorie(s)
commands : Invoke-SegPython, Invoke-SegPip, Assert-SegCondaIsolation, Restore-SegProcessEnvironment
```

随后 `Assert-SegCondaIsolation`：

```text
pkgs_dirs              : E:\桌面\AI\segformer_resources\cache\conda_pkgs
register_envs          : False
number_channel_notices : 0
```

### 6.4 入口脚本的三项实测

| 检查 | 命令 | 实际结果 | 是否通过 |
| --- | --- | --- | --- |
| 从其他目录加载 | `Set-Location C:\` 后 dot-source | 正确解析到项目根并 `Set-Location` 回去 | 通过 |
| 可重复执行 | 同一进程 dot-source 两次 | 无报错，`condarc` 状态变为 `[existing]`，未覆盖 | 通过 |
| 新进程包装 + 环境复原 | `run_project.ps1 -PythonArgs @('-c', ...)` | 子进程 `exe=...\env\python.exe`、`temp=...\segformer_resources\tmp`、`cwd=...\SegFormer`；返回后调用方 `TEMP=C:\Users\LUOREN~1\AppData\Local\Temp` 已复原 | 通过 |

资源隔离检查：本阶段未安装任何包。写入位置仅 `segformer_resources\` 下 19 个空目录、`config\condarc.yml`、`audit\before-*`，以及仓库内 3 个脚本、3 个 Python 文件、`.gitignore` 追加两行。`cache\pycache\Anaconda\` 出现了约 14 MB 的 `.pyc`，原因是 `conda.exe` 本身由 base Python 运行，继承了 `PYTHONPYCACHEPREFIX`；这是写入**项目缓存**而非 base 目录，属于预期副作用，见 8.2。

## 7. 验收与当前状态

| 验收项（来自主指南/隔离规范） | 结果 | 证据 |
| --- | --- | --- |
| 资源根目录按规范建立 | 满足：19 个子目录 | `enter_project.ps1` 输出 |
| 代码与脚本位于 `work\SegFormer` | 满足 | 第 4 节文件表 |
| 已有文件未被覆盖/删除 | 满足：`work\SegFormer` 原有 `AGENTS.md` 未改；`segformer_resources` 原本不存在 | `git status`、目录清单 |
| 启动入口可在新工具进程复用 | 满足 | 6.4 第三行 |
| 入口可重复执行、不破坏已有环境 | 满足 | 6.4 第二行 |
| 保存搭建前快照 | 满足：8 个 before 文件 | `audit\before-*` |
| 不修改 base / 其他环境 / 全局配置 | 满足（阶段 02 前后比对确认） | `after-*` 与 7.1 |
| 脚本在中文路径下可用 | 满足：三个脚本非 ASCII 字节数为 0 | 编码检查 |
| 阶段文档与索引 | 满足 | 本文件 + `docs/stages/README.md` |

7.1 base 与配置的前后比对（阶段 02 完成后回填，此处为结论）：

```text
base package list: IDENTICAL (name|version|build|channel, 510 entries)
E:\Anaconda\.condarc              same=True
C:\Users\Luorenwei\.condarc       same=True
E:\Anaconda\pkgs 中 11:55 之后被修改的条目数 = 0
E:\Anaconda\Lib\site-packages 中 11:55 之后被修改的条目数 = 0
```

**当前状态说明**：本阶段交付的是「工作区 + 入口 + 快照」，不是「环境已可用」。环境可用性属于阶段 02，已在该文档中验证。未完成项：`manifests\`、`configs\`、`reports\` 之外的计划目录（`configs\baseline.yaml` 等）留给对应阶段，不预先创建空壳。

## 8. 遇到的问题、修复与未解决项

### 8.1 本机状态与指南记录不一致（不是错误，但必须记录）

指南第 1 节写「默认 Python 为 `E:\Anaconda\python.exe` 3.12.7」，本次实测 **`PATH` 上排在前面的并不是它**：

```text
Get-Command python -> C:\Users\Luorenwei\AppData\Local\Programs\Python\Python312\python.exe
                     3.12.0rc3 (tags/v3.12.0rc3)
```

同时 base 里**已经装了** torch 2.7.1+cu126 / torchvision 0.22.1+cu126（指南记录为「尚无 torch」）。

影响与处置：这正是「不沿用历史状态」的意义。裸 `pip install` 会装进 3.12.0rc3 那个用户安装，裸 `python` 会用它跑代码——两者都不是本项目要用的解释器。因此入口脚本**不依赖 PATH**，一切走 `$SegConda run --prefix`。base 里已有的 torch 一律不动、不升级、不卸载，也不借用。

### 8.2 `cache\pycache\Anaconda\` 出现 base 标准库的 `.pyc`

现象：首次加载上下文并调用 `Assert-SegCondaIsolation` 后，`cache\pycache\` 从 0 涨到 14 MB，目录结构形如 `cache\pycache\Anaconda\Lib\*.pyc`。
原因：`conda.exe` 是用 base Python 打包的 Python 程序，它继承了我们设置的 `PYTHONPYCACHEPREFIX=E:\...\cache\pycache`，于是把 base 标准库的字节码缓存写到了我们的目录（镜像出 `Anaconda` 前缀路径）。
判断：写的是**项目缓存目录**，不是 base 安装目录；不影响 base 的可用性，也不会与系统 Python 互相干扰。
处置：保留，不清理（`PYTHONPYCACHEPREFIX` 是隔离设计的一部分，去掉反而会让字节码散落到 base）。代价是约 14 MB 冗余，可接受；装完依赖后该目录增长到 238 MB（含 transformers 等库），见阶段 02 的 6.8。
未解决：如果以后要彻底避免，只能在启动 `conda.exe` 时临时取消该变量——但这会破坏「conda run 子进程继承项目缓存」的一致性，暂不采用。

### 8.3 PowerShell 执行策略为 `Restricted`，本地 `.ps1` 无法直接运行

现象：第一次调用快照脚本报 `UnauthorizedAccess`，`Get-ExecutionPolicy -List` 全部 `Undefined`（即默认限制）。

```text
MachinePolicy  UserPolicy  Process  CurrentUser  LocalMachine
Undefined      Undefined   Undefined  Undefined   Undefined      Effective: Restricted
```

处置：用**进程级**覆盖，不碰系统策略：

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
```

验证有效（脚本随即正常执行）。这条已写入「每次开工命令」，因为每个新工具进程都要重新设置一次。不采用 `Set-ExecutionPolicy`（哪怕 `-Scope CurrentUser`）也不改注册表，符合「不修改全局配置」的要求。

### 8.4 `Get-ChildItem Env:` 直接抛异常

现象：`Get-ChildItem Env: | Where-Object {...}` 报「已添加了具有相同键的项」。
原因：本机进程环境中同时存在 `HTTP_PROXY` 与 `http_proxy`（以及 `HTTPS_PROXY`/`https_proxy`），PowerShell 的 `Env:` 驱动器按不区分大小写建哈希表，遇到仅大小写不同的重名键就崩。
处置：快照脚本改用 `[System.Environment]::GetEnvironmentVariables('Process')`，绕开 `Env:` 驱动器。指南里 `Get-ChildItem Env:PIP_*` 的写法在本机会失败，已替换。

### 8.5 控制台把中文路径显示成乱码

现象：Python 打印 `E:\妗岄潰\AI\...`，conda 创建成功的提示里也是乱码。
原因：Python 侧继承 `PYTHONIOENCODING=utf-8` 输出 UTF-8 字节，而 PowerShell 控制台按 GBK 解码。**文件内容本身是正确的**——`reports\environment.json` 里的路径是对的。
处置：在 `enter_project.ps1` 末尾把当前进程的控制台输出编码切到 UTF-8：

```powershell
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
```

这是控制台句柄设置，不是系统区域设置，关闭窗口即失效。修复后同一命令输出恢复正常中文。

## 9. 给用户的学习要点

### 9.1 三个概念

**（1）解释器身份比版本号重要。** 本机同时存在 3.12.7（Anaconda base）和 3.12.0rc3（用户安装）。版本号几乎一样，但 `site-packages` 完全不同。判断「用的是谁」，看 `sys.executable`，不看 `python --version`。

**（2）`--prefix` 是「位置」，`activate` 是「当前终端的 PATH」。** 前者决定环境在哪，后者只影响这个终端窗口里 `python` 指向谁。本项目的所有命令都用 `conda run --prefix`，等于「每次临时借一下那个位置」，不改终端状态。

**（3）隔离的边界是「包、缓存、临时文件、输出」，不是「整台电脑」。** Conda 管理器本身、Windows、NVIDIA 驱动、GPU 都是共享的。说「隔离完成」时，指的是前四项有明确归属，而不是承诺系统零写入。

### 9.2 三个自测问题

1. **如果我把 `work\SegFormer` 整个复制到 `D:\tmp\SegFormer`，`enter_project.ps1` 会去哪里找资源目录？**
   线索：5.1 的三行 `Split-Path`。答案：会算出 `D:\tmp\..\..\..\segformer_resources`（即 `D:\segformer_resources`），因为资源根是**相对仓库位置**推出来的，不是写死的绝对路径。想改就得传 `-ResourceRoot`。

2. **为什么 `Assert-SegCondaIsolation` 要在 `conda create` 之前调用，而不是之后？**
   线索：5.5。答案：Conda 会合并多层配置，`pkgs_dirs` 一旦是多项，包缓存就会被写到别的盘；这时候再检查已经晚了——包已经落盘。检查的目的就是「在产生副作用之前拦住」。

3. **`reports\environment.json` 里 `cache leak: 0` 说明了什么？又没说明什么？**
   线索：5.3 与 6.4。说明：`tempfile`、`torch.hub`、`matplotlib`、`huggingface_hub` 四个库**实际解析**到的路径都在 `segformer_resources` 内。没说明：Conda 自己的元数据、Windows 临时目录里的其他程序、以及未来某次不经包装命令的直接调用，都不在这个结论范围内。

## 10. 下一阶段与修订记录

阶段 02 依赖本阶段的产物：

- `scripts\enter_project.ps1` 提供的 `$SegConda/$SegEnv/$SegCache/$SegAudit` 与 `Assert-SegCondaIsolation`（创建环境前的闸门）；
- `scripts\snapshot_environment.ps1`（生成 `after-*` 做前后比对）；
- `audit\before-*`（比对基准）；
- `segformer_lab\paths.py`（`env_check.py` 用它定位 `reports\`）。

必须保持一致、后续不得随意改动的东西：资源根的绝对位置、`condarc.yml` 的内容、`Invoke-SegPython`/`Invoke-SegPip` 的语义（都必须经 `conda run --prefix`）。

下一阶段（步骤 3）会读取：三份 CSV 清单的路径规则、`data\visa` 目录，以及本文件第 3.1 节的目录布局。

| 日期 | 修改原因 | 改动与影响 | 修正后重新验证的证据 |
| --- | --- | --- | --- |
| 2026-10-31 | 首版建立 | 新增三个脚本、两个 Python 模块、资源根骨架与 before 快照 | 6.2—6.4 的命令输出 |
| 2026-10-31 | 自检发现 `tempfile` 未落入项目目录 | `enter_project.ps1` 增加 `TMPDIR` 设置；快照/入口逻辑不变 | 阶段 02 的 `effective_paths.temp_dir` 变为 `...\segformer_resources\tmp` |
| 2026-10-31 | 中文路径在控制台显示为乱码 | 增加控制台输出编码切换（仅进程内） | 重跑 `env_check` 输出正常中文 |
| 2026-10-31 | 脚本改为纯 ASCII | 注释与输出改英文，避免 PS 5.1 无 BOM 解码问题 | 非 ASCII 字节数检查为 0 |
| 2026-10-31 | 阶段 03 执行时发现入口在本机会被安全删除策略拦截（`Remove-Item Env:PYTHONHOME` 目标不存在即 fail-closed） | 清理继承变量前增加存在性判断，`PIP_*` 循环同样处理；脚本哈希变为 `19d5ef59…b92d8` | 详见阶段 03 文档 8.1；修复后 `run_project.ps1` 与 `Invoke-SegPython` 恢复正常 |

### 验收修订记录补充

| 日期 | 修改原因 | 改动与影响 | 修正后验证 |
| --- | --- | --- | --- |
| 2026-10-31 | 独立复跑发现PS7保留空缓存变量及资源根覆盖未贯通 | 使用Env provider真正删除键，导出SEGFORMER_RESOURCE；包装器恢复环境/编码/目录，保留入口原错与真实退出码，兼容空参数 | PS7与PS5.1各20项入口回归通过；详见验收记录 |
