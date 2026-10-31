# SegFormer 项目环境隔离与阶段学习文档规范

日期：2026-10-31。配套主指南：[SegFormer 单基线全程指南](./2026-10-31-segformer-baseline-end-to-end-guide.md)。本文件细化“环境如何独立、以后如何找回设置、每阶段如何留下可学习的设计文档”。

**本次只交付完善后的方案、命令与文档规则。尚未创建实验 Conda 环境、安装依赖或训练模型。**下列创建/安装命令供未来阶段执行；盘点命令已作只读核查。若此前按旧指南实际建过 venv，先登记路径与已装包，再选择迁移或保留，不执行自动删除。

## 1. 这台电脑原来有什么，能确认到哪里

本次实际读取结果：

| 项目 | 确认结果 |
| --- | --- |
| Anaconda 根目录 | `E:\Anaconda` |
| Conda 程序 | `E:\Anaconda\Scripts\conda.exe`，版本 24.11.3 |
| 默认 Python | `E:\Anaconda\python.exe`，3.12.7 |
| 当前 CLI 登记环境 | 仅 `E:\Anaconda`，即 base |
| `E:\Anaconda\envs` | 本次检查为空 |
| 旧指南的 `E:\桌面\AI\.venvs` | 本次检查不存在 |
| 新资源目录 `E:\桌面\AI\segformer_resources` | 本次检查不存在 |
| base 的 ML 包 | 未发现 torch、torchvision、transformers；已有 NumPy 1.26.4 |
| 已读到的 Conda 配置文件 | `E:\Anaconda\.condarc`、用户目录下的 `.condarc` |

“登记列表仅有 base”不等于磁盘上没有其他环境。`venv` 不一定登记到 Conda，其他硬盘也可能有旧的 prefix 环境。当前没有完整历史证据可以判断某个旧项目究竟做过哪些设置，不按印象补写。

### 1.1 你需要重新建立的几个概念

| 名称 | 含义与操作 |
| --- | --- |
| Anaconda/Conda | 管理 Python 与环境的工具，不是某个项目本身 |
| base | Anaconda 自带的根环境；本项目将它作为管理工具来源，包安装目标为新的项目环境 |
| `--prefix` / `-p` | 按绝对目录确定环境，比只记环境名字更容易找回 |
| activate | 为当前终端配置环境相关 PATH 等；不等于创建环境 |
| `conda run -p ...` | 在指定环境上下文中运行命令，适合 Agent 和不想修改终端配置的场景 |
| `python -m pip` | 使用这个 Python 对应的 pip，避免裸 `pip` 指向别的解释器 |
| 模型/包缓存 | 不属于 Python site-packages，必须另设目录才能把大资源集中管理 |

### 1.2 以后忘记环境时，先运行这些只读命令

在新 PowerShell 中：

```powershell
$SegConda = 'E:\Anaconda\Scripts\conda.exe'
& $SegConda --version
& $SegConda env list
Get-Command python,pip,conda -ErrorAction SilentlyContinue | Select-Object Name,Source
& $SegConda config --show envs_dirs pkgs_dirs create_default_packages
```

对一个**已确认存在**的环境检查，不要把示例路径原样套到未知目录：

```powershell
$KnownPrefix = 'E:\Anaconda'
& $SegConda list --prefix $KnownPrefix
& $SegConda run --no-capture-output --prefix $KnownPrefix python -c "import sys,site; print(sys.executable); print(sys.version); print(site.getsitepackages()); print(site.ENABLE_USER_SITE)"
Get-Content -LiteralPath (Join-Path $KnownPrefix 'conda-meta\history') -Tail 60
```

`conda-meta/history` 可追踪部分 Conda 包操作，不包含全部 pip 操作、模型缓存路径、编辑器设置或历史环境变量。用 `conda list` 配合该环境中的 `python -m pip list` 查看当前状态。发布环境报告时只摘录必要版本、路径与去敏后的配置；不要复制代理密码、令牌或整份环境变量。

## 2. 新方案：Conda 前缀环境与项目资源目录

本方案使用**新建 Conda prefix 环境**，由 Conda 提供独立 Python；深度学习依赖在该环境内用 pip 安装。不再把“基于 base Python 的 venv”作为主路径，也不在 Conda 环境内再套一层 venv。

```text
E:\桌面\AI\segformer_resources\          # 项目资源根目录，Git仓库外
├─ env\                                # 独立 Conda Python 环境
├─ cache\conda_pkgs\                   # Conda 包缓存
├─ cache\conda_envs\                   # 项目专用环境搜索目录
├─ cache\pip\                          # pip 缓存
├─ cache\huggingface\hub\              # Hub 缓存
├─ cache\torch\                        # torch hub/扩展等缓存
├─ cache\xdg\                          # 遵循 XDG 的其他缓存
├─ cache\matplotlib\                   # Matplotlib 配置/字体缓存
├─ cache\pycache\                      # Python 字节码缓存
├─ config\condarc.yml                   # 项目级 Conda 配置
├─ tmp\                                # 项目进程的临时文件
├─ data\visa\pcb1\                    # 独立提取的 PCB1 数据
├─ models\mit-b0\                      # 固定 revision 的模型快照
├─ runs\                               # 训练、预测、TensorBoard输出
└─ audit\                              # 环境快照与隔离核查

E:\桌面\AI\AI_Try\work\SegFormer\
├─ scripts\enter_project.ps1            # 后续阶段1创建，加载当前终端上下文
├─ scripts\run_project.ps1              # 后续阶段1创建，供Agent新进程使用
├─ docs\stages\README.md               # 后续阶段1建立的阶段索引
├─ docs\stages\01-workspace-design-and-implementation.md
├─ ...                                  # 每个阶段各一份学习文档
├─ configs\ / manifests\ / reports\
└─ segformer_lab\                       # Python工程代码
```

已有数据 TAR、代码 ZIP 继续在原 `docs/dataset/VisA` 中作为只读来源，无须为“独立”额外复制一份 1.9GB 归档。解压结果、后续下载和运行产物进入新资源根目录。源码、配置和阶段学习文档仍由 Git 管理。

### 2.1 隔离的具体边界

- 新环境内的 Python 包、模型、数据派生物、训练输出与主要应用缓存均有明确项目目录。
- 不向 base/旧环境安装、卸载或升级包；不修改用户/系统 `.condarc`、全局 pip 配置或系统 PATH；不运行 `conda init`、`setx` 或全局缓存清理。
- 使用新开的专用 PowerShell 会话；环境变量只在当前进程及其子进程中生效，关闭该窗口后不影响其他已打开终端。
- Conda 管理程序、Windows、NVIDIA 驱动和 GPU 仍共享；本方案在当前进程关闭环境登记和 channel notices，减少共享元数据写入，但不能把 Python/项目资源隔离等同于独立虚拟机，也不承诺系统任何文件都零写入。
- 不迁移、清空或复用其他项目的缓存。即使主缓存都重定向，也需在实际运行时核验具体库的输出路径。

目录含中文，本机工程本就位于中文路径下。创建/导入时如遇某个依赖的真实路径兼容错误，先记录报错，再评估另选 ASCII 资源根目录并同步更新全部配置；不临时把包装到 base 来绕过。

## 3. 创建前的环境快照

未来执行阶段1时，先只创建资源目录与审计目录，保存环境清单、base 包清单、已知 `.condarc` 的 SHA-256。无需为了快照导出所有秘密配置。

```powershell
$SegConda = 'E:\Anaconda\Scripts\conda.exe'
$SegResource = 'E:\桌面\AI\segformer_resources'
$SegAudit = Join-Path $SegResource 'audit'
New-Item -ItemType Directory -Force -Path $SegAudit | Out-Null
& $SegConda env list --json | Set-Content -LiteralPath (Join-Path $SegAudit 'environments-before.json') -Encoding utf8
& $SegConda list --prefix 'E:\Anaconda' --json | Set-Content -LiteralPath (Join-Path $SegAudit 'base-packages-before.json') -Encoding utf8
$SegKnownConfigs = @('E:\Anaconda\.condarc', (Join-Path $env:USERPROFILE '.condarc'))
$SegKnownConfigs | Where-Object { Test-Path -LiteralPath $_ } | Get-FileHash -Algorithm SHA256 | Select-Object Path,Hash | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $SegAudit 'condarc-hashes-before.json') -Encoding utf8
```

若存在已知旧项目环境，额外对那些明确路径执行 `conda list --prefix ... --json`，并保存单独快照；不扫描所有硬盘、不修改旧环境。快照后再进入下节的项目进程配置。

## 4. 每次开工的项目上下文：完整可复制命令

**请在新开的项目专用 PowerShell 中运行。**先读完本节；这段会建立项目目录和当前进程变量，但不会安装包或创建 Conda 环境。第一次可以粘贴；后续由阶段1把相同逻辑保存为 `scripts/enter_project.ps1`，避免靠记忆重新配置。

```powershell
$ResearchRepo = 'E:\桌面\AI\AI_Try'
$SegProject = Join-Path $ResearchRepo 'work\SegFormer'
$SegConda = 'E:\Anaconda\Scripts\conda.exe'
$SegResource = 'E:\桌面\AI\segformer_resources'
$SegEnv = Join-Path $SegResource 'env'
$SegPython = Join-Path $SegEnv 'python.exe'
$SegOutput = Join-Path $SegResource 'runs'
$SegCache = Join-Path $SegResource 'cache'

$SegSubdirs = @('cache\conda_pkgs','cache\conda_envs','cache\pip','cache\huggingface\hub','cache\torch','cache\cuda','cache\xdg','cache\matplotlib','cache\pycache','config','tmp','data\visa','models','runs','audit')
foreach ($SegSubdir in $SegSubdirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $SegResource $SegSubdir) | Out-Null
}

# 仅清理本进程继承的Python/pip定向设置，不修改持久配置或其他终端。
Remove-Item Env:PYTHONPATH,Env:PYTHONHOME -ErrorAction SilentlyContinue
Get-ChildItem Env:PIP_* | ForEach-Object { Remove-Item -LiteralPath ('Env:' + $_.Name) }
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $SegCache 'pycache'
$env:PIP_CONFIG_FILE = 'nul'
$env:PIP_CACHE_DIR = Join-Path $SegCache 'pip'
$env:CONDA_PKGS_DIRS = Join-Path $SegCache 'conda_pkgs'
$env:CONDA_ENVS_PATH = Join-Path $SegCache 'conda_envs'
$env:CONDA_REGISTER_ENVS = 'false'
$env:CONDA_NUMBER_CHANNEL_NOTICES = '0'
$env:CONDARC = Join-Path $SegResource 'config\condarc.yml'
Remove-Item Env:TRANSFORMERS_CACHE,Env:PYTORCH_TRANSFORMERS_CACHE,Env:PYTORCH_PRETRAINED_BERT_CACHE -ErrorAction SilentlyContinue
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
$env:TEMP = Join-Path $SegResource 'tmp'
$env:TMP = $env:TEMP

Set-Location -LiteralPath $SegProject
function Invoke-SegPython {
    & $SegConda run --no-capture-output --prefix $SegEnv python @args
    if ($LASTEXITCODE -ne 0) { throw "项目Python命令失败，退出码：$LASTEXITCODE" }
}
function Invoke-SegPip {
    Invoke-SegPython -m pip --cache-dir (Join-Path $SegCache 'pip') @args
}
function Assert-SegCondaIsolation {
    $SegEffectiveJson = & $SegConda config --show pkgs_dirs register_envs number_channel_notices --json
    if ($LASTEXITCODE -ne 0) { throw '无法核对Conda生效配置。' }
    $SegEffective = $SegEffectiveJson | ConvertFrom-Json
    $SegEffectivePkgs = @($SegEffective.pkgs_dirs)
    $SegExpectedPkgs = [IO.Path]::GetFullPath((Join-Path $SegCache 'conda_pkgs'))
    if ($SegEffectivePkgs.Count -ne 1) { throw 'Conda仍合并了其他包缓存，先检查配置来源，不创建环境。' }
    if (-not [string]::Equals([IO.Path]::GetFullPath($SegEffectivePkgs[0]), $SegExpectedPkgs, [StringComparison]::OrdinalIgnoreCase)) { throw 'Conda包缓存未指向项目目录。' }
    if ($SegEffective.register_envs -ne $false -or $SegEffective.number_channel_notices -ne 0) { throw 'Conda登记/notices隔离设置未生效。' }
    $SegEffective | Format-List
}
```

`PIP_CONFIG_FILE='nul'` 使用 Windows Python 的 `os.devnull` 拼写禁用 pip 配置文件，继承的 `PIP_*` 在本进程重置，安装源在命令里明确给出。本机 pip 24.2 的判断存在大小写敏感比较，因此这里明确使用小写 `nul`。本方案不依赖 `--isolated`，而使用清理继承变量、禁用配置文件、显式安装源和缓存的组合。安装始终使用 `--no-user`，并在创建后核验 Python 用户 site 已禁用。[pip 配置说明](https://pip.pypa.io/en/stable/topics/configuration/)

Conda 会合并不同层级的配置；设置 `CONDARC` 并不代表所有其他来源都不再读取，`CONDA_PKGS_DIRS` 也不能脱离生效结果就宣称排除了其他缓存。因此，创建前执行 `Assert-SegCondaIsolation`；若发现共享缓存仍被合并，先定位配置来源并解决，不能继续创建。创建时另外显式指定 `--prefix`、`--override-channels`、`--no-default-packages`。[Conda 配置说明](https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/use-condarc.html)

`CONDA_ENVS_PATH` 在本机 Conda 中仍可能附加默认的环境搜索目录；环境写入目标由明确的 `--prefix` 保证。`CONDA_REGISTER_ENVS=false` 关闭向用户级 `environments.txt` 登记本项目，所以新环境可能不出现在一般 `conda env list` 中；以后用本文件记录的 prefix 查询即可。`CONDA_NUMBER_CHANNEL_NOTICES=0` 关闭 channel notices，`--no-shortcuts` 避免创建系统快捷方式。旧的 Transformers 缓存变量也在当前进程清除，以免覆盖 HF 缓存路径；动态模块缓存显式设为 `HF_MODULES_CACHE`。[Transformers 4.57.6 缓存实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/utils/hub.py)

本项目的 `config/condarc.yml` 由阶段1写入以下内容，**不覆盖已有文件**；若文件已有不同值，应先检查差异：

```yaml
channels:
  - https://repo.anaconda.com/pkgs/main
channel_priority: strict
auto_activate_base: false
create_default_packages: []
```

`envs_dirs/pkgs_dirs` 以实际生效检查为准。上面的登记和通知选项已按本机 Conda 24.11.3 检查；迁移到其他版本时先查询其帮助与生效配置。以后若需要公司代理或证书，针对当前项目配置并记录，不把关闭 TLS 验证当作下载修复方法。

### 4.1 Agent 每次新工具进程也要加载上下文

很多 Agent 的每次终端调用都是新进程，前一次设置的变量与函数不会自动保留。阶段1应建立：

- `scripts/enter_project.ps1`：上述进程设置与函数，供专用交互终端 dot-source；须可重复执行，不覆盖已有环境或配置。已有项目根目录不同则报错或显式参数化。
- `scripts/run_project.ps1`：接受 `-PythonArgs` 字符串数组；在子进程范围加载上下文，调用 `conda run`，将真实退出码返回调用者。若在调用者进程设置变量，必须用 try/finally 恢复原值。

未来脚本创建之后的使用方式：

```powershell
# 人工专用终端：加载后使用指南中的 Invoke-SegPython / Invoke-SegPip。
. 'E:\桌面\AI\AI_Try\work\SegFormer\scripts\enter_project.ps1'
Invoke-SegPython -c "import sys; print(sys.executable)"

# Agent新工具调用：每次显式使用包装入口，不依赖上一条命令的会话。
& 'E:\桌面\AI\AI_Try\work\SegFormer\scripts\run_project.ps1' -PythonArgs @('-m','segformer_lab.env_check','--out','reports/environment.json')
```

首次脚本还不存在时，先由阶段1实现并检查，或手动粘贴本节完整块。若执行策略阻止本地脚本，可先粘贴等价命令操作；不为此修改系统级/用户级执行策略。使用 VS Code 时在当前工作区选择 `$SegEnv\python.exe`，终端仍需加载项目上下文；其他项目的编辑器全局解释器设置保持原样。

## 5. 创建独立 Conda 环境并安装依赖

加载第4节后，确认 `$SegEnv` 不存在；若已存在，盘点并复用符合条件的环境，不能重新覆盖。

```powershell
if (Test-Path -LiteralPath $SegEnv) { throw '目标环境已存在；请先核对现有环境，勿覆盖创建。' }
Assert-SegCondaIsolation
& $SegConda create --prefix $SegEnv --copy --no-shortcuts --no-default-packages --override-channels --channel 'https://repo.anaconda.com/pkgs/main' --strict-channel-priority 'python=3.12' pip -y
if ($LASTEXITCODE -ne 0) { throw 'Conda环境创建失败，请先查看完整报错。' }
Invoke-SegPython -c "import sys,site; print(sys.executable); print(sys.prefix); print(sys.version); print('user site enabled:',site.ENABLE_USER_SITE)"
Invoke-SegPip --version
```

`--copy` 使用副本而非链接安装环境文件；`--no-default-packages` 避免旧配置自动注入包；只让 Conda 创建 Python/pip，之后深度学习包统一在该 prefix 中由 pip 管理。[conda create](https://docs.conda.io/projects/conda/en/stable/commands/create.html)

`python=3.12` 允许选择当时可取得的 3.12 修订版；不能预设新环境仍是 base 的 3.12.7。安装后保存确切版本。首次包下载的网络时间与空间另计。

```powershell
Invoke-SegPip install --no-user --upgrade pip --index-url 'https://pypi.org/simple'
Invoke-SegPip install --no-user 'torch==2.10.0' 'torchvision==0.25.0' --index-url 'https://download.pytorch.org/whl/cu126'
Invoke-SegPip install --no-user 'transformers==4.57.6' 'huggingface-hub>=0.34,<1' 'numpy>=1.26,<3' 'Pillow>=10,<13' 'matplotlib>=3.9,<4' 'PyYAML>=6,<7' 'tqdm>=4.66,<5' 'tensorboard>=2.18,<3' --index-url 'https://pypi.org/simple'
Invoke-SegPip check
```

核心依赖与主指南一致，组合仍须实际导入验证。不要把旧环境目录加入 PYTHONPATH，也不要通过 `pip install --user` 借用用户 site。[conda run](https://docs.conda.io/projects/conda/en/stable/commands/run.html) 用于获得 Windows 环境 DLL 等运行上下文；仅双击某个 `python.exe` 不等于完成相同的环境设置。

## 6. 怎么证明环境和资源用对了

### 6.1 解释器、用户 site 与缓存

```powershell
Invoke-SegPython -c "import sys,site,os,tempfile; print('python:',sys.executable); print('prefix:',sys.prefix); print('user site enabled:',site.ENABLE_USER_SITE); print('temp:',tempfile.gettempdir()); print('HF_HOME:',os.environ.get('HF_HOME')); print('TORCH_HOME:',os.environ.get('TORCH_HOME'))"
Invoke-SegPip cache dir
Invoke-SegPython -c "import torch,transformers,matplotlib; print('torch:',torch.__version__,torch.__file__); print('transformers:',transformers.__version__,transformers.__file__); print('mpl cache:',matplotlib.get_cachedir()); print('CUDA:',torch.cuda.is_available(),torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
Invoke-SegPython -c "import os,torch; from huggingface_hub import constants as hc; from transformers.utils import hub as th; print('HF actual cache:',hc.HF_HUB_CACHE); print('Transformers actual cache:',th.TRANSFORMERS_CACHE); print('torch hub actual cache:',torch.hub.get_dir()); print('CUDA JIT cache:',os.environ.get('CUDA_CACHE_PATH'))"
```

应看到：Python/prefix 和第三方库都位于 `segformer_resources/env`，用户 site 为 False，pip/HF/torch/Matplotlib/temp 指向项目资源根目录。`sys.path` 中当前项目源码目录正常；旧环境的 site-packages 不应混入。库导入前就应设置缓存变量，不只检查变量字符串，还要检查上述库实际读取的常量。`CUDA_CACHE_PATH` 用于项目进程的 CUDA JIT 缓存，不要求另装 Toolkit。[NVIDIA 环境变量说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-cache-path) 随后执行主指南的 CUDA 前后向运算和 SegFormer smoke。

### 6.2 base 与已有环境没有被改包

安装后保存同类快照：

```powershell
& $SegConda list --prefix 'E:\Anaconda' --json | Set-Content -LiteralPath (Join-Path $SegResource 'audit\base-packages-after.json') -Encoding utf8
$SegBefore = Get-Content -LiteralPath (Join-Path $SegResource 'audit\base-packages-before.json') -Raw | ConvertFrom-Json
$SegAfter = Get-Content -LiteralPath (Join-Path $SegResource 'audit\base-packages-after.json') -Raw | ConvertFrom-Json
Compare-Object ($SegBefore | ForEach-Object { "$($_.name)|$($_.version)|$($_.build_string)" } | Sort-Object) ($SegAfter | ForEach-Object { "$($_.name)|$($_.version)|$($_.build_string)" } | Sort-Object)
```

无输出表示所比较的包名/版本/构建列表一致；这不是磁盘所有文件完全未变的证明。阶段2文档另对第3节记录的 Conda 配置文件重新计算哈希并比对，对已盘点的旧环境执行同样包清单比对。以准确 prefix 查询新环境；关闭登记后，不将它是否出现在一般环境列表作为创建成功的唯一判断。

### 6.3 保存“下次不用猜”的记录

```powershell
Invoke-SegPip freeze | Set-Content -LiteralPath 'requirements-lock.txt' -Encoding utf8
& $SegConda env export --prefix $SegEnv --from-history | Set-Content -LiteralPath 'environment-history.yml' -Encoding utf8
& $SegConda list --prefix $SegEnv --explicit | Set-Content -LiteralPath (Join-Path $SegResource 'audit\conda-explicit-win64.txt') -Encoding utf8
Invoke-SegPython -m segformer_lab.env_check --out 'reports\environment.json'
```

`--from-history` 主要记录 Conda 请求的 Python/pip，不会完整描述随后 pip 安装的训练栈；必须与 `requirements-lock.txt`、CUDA wheel 来源和启动入口配套。explicit 文件用于同平台恢复 Conda 部分，不能当跨 Windows/Linux 通用文件；发布前检查源地址是否含凭证。

以后只记住三个入口即可：**资源根目录、`scripts/enter_project.ps1`、`docs/stages/README.md`**。不需要凭记忆回想一长串 activate 和缓存变量。

## 7. 退出、复查与日后迁移

- 关闭项目专用 PowerShell 即退出这些进程设置；使用 `conda run` 不要求先 activate/deactivate。
- 若习惯在 Anaconda Prompt 手动使用 `conda activate 完整prefix` / `conda deactivate`，应先加载项目缓存变量，并核对实际解释器；不要再建同名新环境。这是可选人工入口，主指南统一用包装命令。
- 迁移电脑时传递代码、清单、锁定文件和阶段文档；新建环境并重新取得数据/模型，不直接复制整个 Conda env。
- 不主动清理全局缓存、旧环境或本项目结果。日后明确要移除环境时，先确认准确 prefix、备份记录并核实只指向本项目；本方案不提供一键递归删除资源根目录的命令。

## 8. 每阶段必须交付一份设计与实现学习文档

**这是本项目后续 Agent 的明确交付要求：每完成一个编号阶段，必须生成或更新一份中文 Markdown，说明本阶段目标、设计、实现细节、技术栈、代码结构、命令、实际验证和学习要点。聊天中的几句总结不能替代文件。**

统一保存到 `work/SegFormer/docs/stages/`，英文文件名、中文正文。索引 `README.md` 记录阶段状态、文档链接、关键证据路径、下一步。未执行的阶段不预先填写为完成；受阻时也记录已做工作与真实阻塞。

| 步骤 | 阶段文档文件名 | 特别要解释的内容 |
| --- | --- | --- |
| 1 | `01-workspace-design-and-implementation.md` | 目录、隔离边界、启动脚本、当前环境盘点、Agent调用入口 |
| 2 | `02-environment-design-and-implementation.md` | Conda/Python/pip、缓存变量、实际版本、base前后比对、CUDA验证 |
| 3 | `03-data-preparation-design-and-implementation.md` | 归档读取、官方CSV、固定划分算法、清单哈希、0/1标签 |
| 4 | `04-data-audit-design-and-implementation.md` | 解码审计、重复检查、缩放、叠加图、缺陷面积 |
| 5 | `05-model-design-and-implementation.md` | MiT/SegFormer、权重revision、二类头、加载报告与梯度 |
| 6 | `06-dataset-design-and-implementation.md` | Dataset/DataLoader、processor、同步增强、0/1/255、几何元数据 |
| 7 | `07-loss-metrics-design-and-implementation.md` | CE/Dice、空前景、原图混淆量、正常误报、手算验证 |
| 8 | `08-smoke-design-and-implementation.md` | 20步通路、AMP、优化更新、保存重载、显存 |
| 9 | `09-overfit-design-and-implementation.md` | 8张训练图、梯度与学习证据、排错、泛化边界 |
| 10 | `10-pilot-design-and-implementation.md` | 全量3轮、数据计数、实测耗时与预算 |
| 11 | `11-training-design-and-implementation.md` | 正式训练循环、scheduler、checkpoint、恢复、best选择 |
| 12 | `12-evaluation-design-and-implementation.md` | 新进程加载、去填充与原图恢复、val/test隔离、无GT预测 |
| 13 | `13-results-design-and-implementation.md` | 曲线、失败案例、实验卡、局限与复现归档 |

一次任务即使合并做步骤1—2，也分别写两份文档并更新同一索引。同一阶段多次修正，更新原文件并增加修订记录，标明哪些验证是修正后重新运行的，避免多份“最终版”互相矛盾。

### 8.1 每份文档的最低质量标准

1. **问题与边界**：为什么有本阶段，输入是什么，完成后能做什么。
2. **技术栈与版本**：列本阶段真正使用的库及作用；拟用与已装/已运行分开。
3. **设计与取舍**：所选方案、关键替代方案和理由；用户要求与 Agent 自行选择的实现细节分开写。
4. **文件与接口**：改动路径、函数/模块职责、配置字段、张量形状/dtype或CSV结构。
5. **关键实现讲解**：用实际代码片段解释，不贴整份源文件；特别说明容易错的地方。
6. **操作与运行证据**：完整命令、解释器/运行上下文、真实输出摘录、日志/图表/权重路径和时间。
7. **验收与状态**：逐项满足/未满足；区分“已写代码”“已运行”“已验证结果”。
8. **问题与修复**：发生过的错误、根因证据、修复和重跑结果；没有错误就写本次未遇到，不编造排错经历。
9. **学习检查**：3—5个本阶段关键概念与自测问题，让用户能定位到相应代码和图。
10. **后续依赖与修订**：下一阶段读取什么、哪些假设仍未证实、此次修改记录。

配套可复制模板：[阶段设计与实现文档模板](./templates/segformer-stage-design-and-implementation-template.md)。阶段文档内容以本阶段实际复杂度为准，一般无需超过几页，不要求固定字数；图像和原始日志放资源目录，文档保存解释与少量摘要。

### 8.2 阶段结束前，Agent 必须自查

- 阶段文档已保存，并在索引中登记，状态与实际一致。
- 代码、配置、命令、版本、清单/权重身份与文档一致。
- 至少一条真实运行证据可定位；未运行时明确写“未验证”及原因。
- 所有新下载/缓存/输出进入项目资源目录，未向 base/其他环境安装包。
- 用户能依据文档重做本阶段，而不是只能看懂聊天中的结论。

本规则也写入 [实验目录 Agent 规则](../../work/SegFormer/AGENTS.md)，供未来在该目录工作的 Agent 自动读取。当前只有规则与模板，不生成假的“已完成阶段”报告。

## 9. 可以粘贴给下一次 Agent 的启动指令

```text
先读 SegFormer 主指南、环境隔离与阶段文档规范，以及 work/SegFormer/AGENTS.md。
本次只做步骤1—2。先盘点当前Conda和环境，不依赖旧聊天记忆。
新环境使用 E:\桌面\AI\segformer_resources\env，资源集中在同一根目录。
使用Conda指定prefix，保留base/其他环境和全局配置；每个新工具进程加载项目上下文。
先保存base/已知旧环境与配置快照，再建立项目启动入口、创建独立环境与安装依赖。
实际安装/运行之前核对本机路径、现有状态和命令，已存在时不要覆盖重建。
完成后分别生成01和02两份中文设计与实现MD，按模板讲解实现、技术栈、真实结果和学习要点，
更新docs/stages/README.md。若失败，记录真实完成范围和阻塞；不要进入模型训练。
返回文档链接、可复用命令、隔离核查和我需要亲自确认的内容。
```
