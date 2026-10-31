# SegFormer-B0 PCB1 有监督二值分割基线

本目录是 VisA PCB1 high-shot 协议下 SegFormer-B0 单基线实验的工程根目录：代码、配置、清单、阶段文档。
环境、缓存、数据、模型与运行产物**不在本目录**，统一放在仓库外的 `E:\桌面\AI\segformer_resources`。

- 当前进度：**步骤 1—2 已完成并通过独立验收；步骤 3（PCB1 数据准备与固定划分）已通过本轮独立验收**；步骤 4 尚未开始，未进入数据可视化与训练。
- 验收记录：[阶段01—02验收与完善](./reports/2026-10-31-stage01-02-acceptance.md)
- 阶段索引：[docs/stages/README.md](./docs/stages/README.md)
- 主指南：`../../docs/research/2026-10-31-segformer-baseline-end-to-end-guide.md`
- 环境隔离规范：`../../docs/research/2026-10-31-segformer-environment-isolation-and-stage-docs.md`

## 1. 每次开工：加载项目上下文

在执行策略为 `Restricted` 的本机上，本地 `.ps1` 默认不能运行。用**进程级**覆盖打开，只影响当前窗口，不改系统/用户策略：

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
$SegProject = 'E:\桌面\AI\AI_Try\work\SegFormer'
Set-Location -LiteralPath $SegProject
. '.\scripts\enter_project.ps1'
```

加载后会得到：

| 变量 | 值 |
| --- | --- |
| `$SegProject` | `E:\桌面\AI\AI_Try\work\SegFormer` |
| `$SegResource` | `E:\桌面\AI\segformer_resources` |
| `$SegEnv` | `...\segformer_resources\env` |
| `$SegPython` | `...\env\python.exe` |
| `$SegOutput` | `...\segformer_resources\runs` |
| `$SegCache` / `$SegTmp` | `...\cache` / `...\tmp` |

以及两个命令：`Invoke-SegPython`、`Invoke-SegPip`（都通过 `conda run --prefix` 运行，不改当前终端 PATH）。

**新的工具进程/脚本调用**（不需要先 dot-source）：

```powershell
& 'E:\桌面\AI\AI_Try\work\SegFormer\scripts\run_project.ps1' -PythonArgs @('-m','segformer_lab.env_check','--out','reports/environment.json')
```

自检两条命令：

```powershell
Invoke-SegPython -c "import sys;print(sys.executable)"              # 必须是 ...\segformer_resources\env\python.exe
Invoke-SegPython -m segformer_lab.env_check --out 'reports\environment.json'
```

## 2. 目录职责

```text
work\SegFormer\                       仓库内（Git）
├─ scripts\enter_project.ps1          当前进程的项目上下文
├─ scripts\run_project.ps1            新进程一次性包装（自动复原环境变量）
├─ scripts\snapshot_environment.ps1   只读环境快照 -> audit\
├─ segformer_lab\paths.py             Python 侧唯一路径真源
├─ segformer_lab\env_check.py         环境/GPU/前后向检查
├─ segformer_lab\prepare_data.py      PCB1 提取、固定划分、清单与元数据
├─ manifests\                         官方 CSV 副本 + train/val/test 清单 + split_meta.json
├─ tests\                             环境检查器与划分/路径守卫的单元测试
├─ reports\environment.json           最新环境报告
├─ requirements-lock.txt              pip freeze 原始结果
├─ requirements-cu126.txt             仅两个CUDA包的辅助配方与来源记录
├─ requirements-runtime.txt           完整44项可移植版本约束（已核对本机版本）
├─ environment-history.yml            conda env export --from-history
└─ docs\stages\                       阶段设计与实现文档 + 索引

E:\桌面\AI\segformer_resources\       仓库外（不进 Git）
├─ env\                               独立 Conda 前缀（Python 3.12.16）
├─ cache\                             conda_pkgs / pip / huggingface / torch / cuda / xdg / matplotlib / pycache / wheels
├─ config\condarc.yml                 项目级 Conda 配置
├─ tmp\                               项目临时目录（含 conda run 的 .bat）
├─ data\visa\                         已提取的 PCB1（pcb1\Data\{Images,Masks} + extraction_manifest.json）
├─ models\mit-b0\                     后续固定 revision 的权重
├─ runs\                              训练 / 评估 / 预测产物
└─ audit\                             快照、日志、比对材料、stage03 独立校验脚本
```

## 3. 当前环境（实测，2026-10-31）

| 项目 | 值 |
| --- | --- |
| 解释器 | `E:\桌面\AI\segformer_resources\env\python.exe` |
| Python | 3.12.16（Anaconda 打包，MSC v.1942） |
| pip | 26.2.2 |
| torch / torchvision | 2.10.0+cu126 / 0.25.0+cu126 |
| transformers | 4.57.6 |
| NumPy / Pillow / Matplotlib / PyYAML | 2.5.4 / 12.3.0 / 3.11.2 / 6.0.3 |
| TensorBoard | 2.21.0 |
| GPU | RTX 4070 Laptop，8188 MiB，cc 8.9，驱动 617.42 |
| 用户 site | 已禁用（`ENABLE_USER_SITE=False`） |
| 资源占用 | env 4.37 GB + wheels 2.48 GB + 其他约 0.6 GB ≈ 7.4 GB |

详细证据：`reports/environment.json`、`segformer_resources/audit/{before,after}-*`。

### 本机几处与旧文档不一致的事实（以实测为准）

1. `PATH` 上的 `python` 是 `C:\Users\Luorenwei\AppData\Local\Programs\Python\Python312\python.exe`（**3.12.0rc3**），不是 Anaconda 的 3.12.7。**不要裸跑 `python`/`pip`**。
2. base（`E:\Anaconda`）已装 torch 2.7.1+cu126、torchvision 0.22.1+cu126。本项目**不使用、不修改**它们。
3. PowerShell 执行策略为 `Restricted`，需按第 1 节做进程级覆盖。
4. `Get-ChildItem Env:` 在本机会因 `HTTP_PROXY`/`http_proxy` 重名而报错，脚本内一律改用 `[System.Environment]::GetEnvironmentVariables('Process')`。
5. 本次 `conda create` 首次因元数据下载失败，重试成功；`pip` 直连 PyPI 约 50 KB/s，已改用清华镜像（仅命令行指定，未改全局配置）。

## 4. 隔离边界

已做：

- 环境、包、缓存、临时文件、输出全部落在 `segformer_resources`；
- `CONDA_REGISTER_ENVS=false`，项目环境不写入 `%USERPROFILE%\.conda\environments.txt`（所以 `conda env list` 看不到它，用 `conda list --prefix ...` 查）；
- `CONDA_PKGS_DIRS` 只指向项目缓存（`Assert-SegCondaIsolation` 强制校验）；
- 不修改 base、`E:\Anaconda\.condarc`、`C:\Users\Luorenwei\.condarc`、`%APPDATA%\pip\pip.ini`（前后哈希一致）；
- 不运行 `conda init`、`setx`，不改系统 PATH。

**没有**承诺的：Conda 管理器、Windows、NVIDIA 驱动、GPU 仍是共享的；这不是虚拟机级隔离。`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6` 存在于本机，但本项目用 wheel 自带运行时，不需要它。

## 5. 阶段进度

| 阶段 | 状态 | 文档 |
| --- | --- | --- |
| 01 工作区与启动入口 | 已验证 | [01-workspace-design-and-implementation.md](./docs/stages/01-workspace-design-and-implementation.md) |
| 02 独立环境与 GPU 验证 | 已验证 | [02-environment-design-and-implementation.md](./docs/stages/02-environment-design-and-implementation.md) |
| 03 PCB1 数据准备与固定划分 | 已通过本轮独立验收 | [03-data-preparation-design-and-implementation.md](./docs/stages/03-data-preparation-design-and-implementation.md) |
| 04—13 | 未开始 | 见 [阶段索引](./docs/stages/README.md) |

## 5.1 数据现状（2026-10-31）

| 部分 | 正常 | 异常 | 合计 | 来源 |
| --- | ---: | ---: | ---: | --- |
| train | 482 | 48 | 530 | 官方 train 662 减去 val |
| val | 120 | 12 | 132 | 从官方 train 固定配额抽出（seed 42） |
| test | 402 | 40 | 442 | 官方 test 原样 |

- 解压位置：`segformer_resources\data\visa\pcb1\Data\{Images,Masks}`（1205 个文件，287 MB），保留官方相对路径；原 TAR/ZIP 未改动。
- 清单：`manifests\train.csv|val.csv|test.csv|split_meta.json|official_2cls_highshot.csv`。
- 文件头检查：1104 张图与 100 张掩码的文件头可解析、尺寸可读取；这项检查未全量解码像素，完整解码审计留阶段 04。
- 稳定身份（CSV 的 SHA-256）：`train b2607726…0d53`、`val 7335d8c8…bb9c`、`test 4bdc5941…1284`；官方 CSV `c3331eed…2f09`。
- 约定：正常图 `mask` 为空 → 运行时生成全零掩码；异常掩码存 1..4 多类 ID，**非零即缺陷（1）**，转换在 Dataset 里做，不改原始文件。

## 6. 常见操作

```powershell
# 重新生成环境报告（含 CUDA 与随机权重 SegFormer 前后向）
Invoke-SegPython -m segformer_lab.env_check --out 'reports\environment.json' --probe full

# 只采集事实，不做 GPU 运算
Invoke-SegPython -m segformer_lab.env_check --out 'reports\environment.json' --probe none

# 重新保存隔离快照并比对（只读，写入 audit\）
& '.\scripts\snapshot_environment.ps1' -Label after
Compare-Object (Get-Content "$SegResource\audit\before-base-packages.txt") `
               (Get-Content "$SegResource\audit\after-base-packages.txt")

# 项目回归检查（另有跨PowerShell入口回归脚本）
Invoke-SegPython -m unittest discover -v
& '.\scripts\test_runtime_context.ps1'

# 数据准备：重新生成三份清单（已存在且内容相同的图不会被覆盖）
Invoke-SegPython -m segformer_lab.prepare_data `
  --archive '..\..\docs\dataset\VisA\VisA_20220922.tar' `
  --source-zip '..\..\docs\dataset\VisA\spot-diff-main.zip' `
  --data-root 'E:\桌面\AI\segformer_resources\data\visa' `
  --category pcb1 --seed 42 --val-normal 120 --val-anomaly 12 --out 'manifests'

# 查环境里装了什么
Invoke-SegPip list
& $SegConda list --prefix $SegEnv
```

清理建议（**不要**直接删资源根）：`cache\wheels`（2.48 GB）在确认环境可用后可删，重装时按 `requirements-cu126.txt` 重新下载并校验哈希。

## 7. 维护约定

- 代码与脚本保持纯 ASCII，中文只写在 Markdown 里（PowerShell 5.1 读无 BOM 的 UTF-8 脚本会按 GBK 解码，中文路径会拼错）。
- 一切 Python 调用走 `Invoke-SegPython` / `run_project.ps1`，不用 PATH 上的 `python`。
- 每个编号阶段完成后更新 `docs/stages/` 对应文档与本 README 的第 5 节。
- 不伪造日志、指标或完成状态；未运行的命令标注「待执行」。

## 第三阶段验收

[第三阶段验收与完善](./reports/2026-10-31-stage03-acceptance.md)：固定划分530/132/442及清单身份均已复核，1205个数据文件与归档逐一匹配；23项项目回归和20项入口回归通过。下一步为阶段04数据完整解码、掩码审计与叠加图。
