# SegFormer-B0 PCB1 有监督二值分割基线

本目录是 VisA PCB1 high-shot 协议下 SegFormer-B0 单基线实验的工程根目录：代码、配置、清单、阶段文档。
环境、缓存、数据、模型与运行产物**不在本目录**，统一放在仓库外的 `E:\桌面\AI\segformer_resources`。

- 当前进度：**步骤 1—5 已通过验收（数据为 v2 分组协议，模型快照固定 commit）；步骤 6（Dataset/同步增广/DataLoader）已通过验收**；未实现损失、指标与训练器。
- 验收记录：[阶段01—02验收与完善](./reports/2026-10-31-stage01-02-acceptance.md)、[第三阶段验收](./reports/2026-10-31-stage03-acceptance.md)、[第四阶段验收](./reports/2026-10-31-stage04-acceptance.md)、[第五阶段模型加载与核验](./reports/2026-10-31-stage05-model-loading-and-check.md)、[第六阶段验收](./reports/2026-10-31-stage06-acceptance.md)
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
├─ segformer_lab\config.py            配置加载/校验/路径解析
├─ segformer_lab\geometry.py          letterbox、二值化、连通域、dHash（阶段 06 复用）
├─ segformer_lab\audit_data.py        全量解码审计与统计
├─ segformer_lab\preview_data.py      固定样例五联图
├─ segformer_lab\download_model.py    固定 revision 下载与快照校验
├─ segformer_lab\modeling.py          二类 SegFormer 加载与编码器逐键核对
├─ segformer_lab\model_check.py       人工张量前向/反向/一步更新/显存
├─ segformer_lab\data.py              Dataset、同步翻转、collate、DataLoader
├─ segformer_lab\dataset_check.py     39 项数据契约检查 + 五联图
├─ configs\baseline.yaml              实验配置（512 letterbox、ignore 255、二类）
├─ manifests\                         官方 CSV 副本 + train/val/test 清单 + split_meta.json + reviewed_groups.json + val_preview_ids.txt
├─ tests\                             环境、分组划分、几何/审计/冻结预览/模型/数据集的单元测试（134 项）
├─ reports\environment.json           最新环境报告
├─ reports\data_audit\                审计摘要、逐图统计、重复与近重复候选
├─ reports\model_*.json               快照下载、加载与运行时核验记录
├─ reports\dataset_check.json         Dataset 契约检查结果
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
├─ models\mit-b0\                     已下载的 MiT-B0 快照（3 文件 + source.json，14.45 MB）
├─ runs\data_preview\                 阶段 04 的 18 张五联图 + 总览图 + near_dup 对比
├─ runs\dataset_check\                阶段 06 的 9 张五联图（原图/GT/512 输入/512 标签/叠加）
├─ runs\                              其余训练 / 评估 / 预测产物
└─ audit\                             快照、日志、比对材料、独立校验脚本
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
| 03 PCB1 数据准备与固定划分 | 已通过独立验收 | [03-data-preparation-design-and-implementation.md](./docs/stages/03-data-preparation-design-and-implementation.md) |
| 04 数据审计与可视化 | 已通过验收（v2 分组） | [04-data-audit-design-and-implementation.md](./docs/stages/04-data-audit-design-and-implementation.md) |
| 05 预训练权重与模型搭建 | 已验证 | [05-model-design-and-implementation.md](./docs/stages/05-model-design-and-implementation.md) |
| 06 Dataset、同步增广与 DataLoader | 已验证 | [06-dataset-design-and-implementation.md](./docs/stages/06-dataset-design-and-implementation.md) |
| 07—13 | 未开始 | 见 [阶段索引](./docs/stages/README.md) |

## 5.1 数据现状（2026-10-31）

| 部分 | 正常 | 异常 | 合计 | 来源 |
| --- | ---: | ---: | ---: | --- |
| train | 482 | 48 | 530 | 官方 train 662 减去 val |
| val | 120 | 12 | 132 | 官方 train 按风险组/单例抽出（v2，seed 42） |
| test | 402 | 40 | 442 | 官方 test 原样 |

- 解压位置：`segformer_resources\data\visa\pcb1\Data\{Images,Masks}`（1205 个 PCB1 文件，296,989,825 字节，含类别元数据），保留官方相对路径；原 TAR/ZIP 未改动。
- 清单：`manifests\train.csv|val.csv|test.csv|split_meta.json|official_2cls_highshot.csv|reviewed_groups.json`。
- 稳定身份（CSV 的 SHA-256）：`train f1522eb9…65b0`、`val 60a6c6cb…6191`、`test 4bdc5941…1284`；官方 CSV `c3331eed…2f09`。
- 约定：正常图 `mask` 为空 → 运行时生成全零掩码；异常掩码存 1..4 多类 ID，**非零即缺陷（1）**，转换在 Dataset 里做，不改原始文件。

## 5.2 数据审计结论（阶段 04）

- 解码：1104/1104 图完整解码，全部 `1404×1070`、`RGB`；100/100 掩码完整解码，全部单通道 `L`，尺寸一致；坏图 0。
- 掩码取值：并集 `{0,1,2,3,4}`，共 9 种组合（最常见 `{0,2}` 46 张）→ 必须二值化后才能进二类损失。
- 缺陷面积：中位数 train 0.188% / val 0.316% / test 0.194%，最大 12.73%。**极小目标**，均值会被少数大缺陷拉高。
- 512 letterbox：`512×390`，上下各 61 px（23.83% 画布）；无异常图的前景全部变空，但 328 个原始 4 连通域中 19 个微小域消失（12 张图、共 39 原图像素）；面积保持率中位 99.94% 不能证明区域无损。原图尺度评价不能恢复已丢训练细节，首轮仍用 512，后续按验证集失败分析决定对照实验。
- 重复与分组：0042/0043 按高可信同实物风险组完整重出 v2，两张都在 train；官方 test 字节未变。当前跨集 362,564 对在本规则下候选 0，不能据此保证绝对实物独立。旧版清单/ID 在 `manifests/archive/pcb1_v1`，旧图片在资源目录 `runs/data_preview/archive/pcb1_v1`。
- 目视复核：原 18 张逐张看过；新版 15 张与旧图字节相同，另 3 张新图实际查看；未见整体错位、镜像或宽高颠倒。正常图 0196/0819/0983 的细痕/网格疑点已登记，证据不足以改标。
- 证据：`reports\data_audit\summary.json`（含 `per_image.csv` 1104 行）、`duplicates_exact.json`、`near_duplicates.json`。
- 预览：`segformer_resources\runs\data_preview\{train,val}`，12 + 6 张五联图（原图/掩码/红色叠加+放大/512 输入/512 标签）与 2 张总览图；固定 ID 见 `manifests\val_preview_ids.txt`。

## 5.3 模型快照与加载结论（阶段 05）

- 快照：`nvidia/mit-b0@80983a413c30d36a39c20203974ae7807835e2b4`，3 个文件共 14 450 344 B，位于 `segformer_resources\models\mit-b0`；来源与逐文件 SHA-256 见该目录 `source.json` 及仓库 `reports/model_source.json`。
- 编码器核验：模型的 192 个 `segformer.encoder.*` 张量与检查点**数值完全相等**（max|diff| = 0.0）；检查点另有 2 个 `classifier.*`（ImageNet 1000 类头，257 000 参数）未被使用。
- 键分类：missing 16（全部 `decode_head.*`）、unexpected 2、mismatched 0；`ignore_mismatched_sizes=True` 本次并未起作用，能加载正确靠的是逐键比对。
- 参数：总 3 714 658，全部可训练（编码器 3 319 392 + 解码头 395 266）。
- 前向：`[2,3,512,512]` → logits `[2,2,128,128]`（stride 4）；两路 CE 均为 0.6495407223701477，差 0.0（阈值≤1e-6）；255 标签未直接计入 CE 归约。
- 反向与一步更新：编码器与解码头梯度均非零有限，AdamW 一步后两者 `max|Δ|`=1.0e-4；快照 SHA-256 前后一致，未被覆盖。
- 显存（隔离 FP32 前反向，AMP 关闭，probe 在创建优化器前）：B1 504 MiB、B2 1008 MiB reserved。接受 batch=2、累积=1起步，训练预算由阶段 08 的 20 步实测。

## 5.4 Dataset 数据契约（阶段 06）

- 入口：`segformer_lab\data.py` 的 `PCB1Dataset` / `collate_pcb1` / `build_dataloader`；`python -m segformer_lab.dataset_check` 一条命令自检 39 项。
- 单样本：`pixel_values float32[3,512,512]`（processor 一次 rescale+normalize，`do_resize=False`、`do_reduce_labels=False`）、`labels int64[512,512]`（0/1/255）、`image_id`（清单相对路径）、`geometry`、`flipped`、`original_mask uint8[1070,1404]`。
- B2：`pixel_values [2,3,512,512]`、`labels [2,512,512]`；有效区 `[2,390,512]`，上下各 61 行全 255（124 928 个 255 像素）。
- 语义：正常图原尺寸构造零掩码 → 有效区全 0；异常掩码 `>0→1` 在 padding **之前**完成；`original_mask` 由 collate 保留为 list，不堆叠；train时为增强后方向的原尺寸GT，val/test无增强时对应文件原图。B2数组payload为3,004,560B，可在训练时关闭保留。
- 增强：训练只用同步水平翻转（`flip="random"`，按 `(seed, epoch, index)` 确定性决策；`flip="force"` 供验收）；val/test 仅允许 none，random/force 都拒绝；每轮由调用方 set_epoch(epoch)。
- 实测证据：强制翻转后标签 `fliplr` 完全相等、图像 uint8 差 0；归一化与 `(x/255−mean)/std` 差 **0.0**；`__getitem__` 输出在 CPU 且源码不含 CUDA/模型加载；真实 B2 eval+no_grad 前向得 `[2,2,128,128]`。
- 证据与小图：`reports\dataset_check.json`（仓库）；`segformer_resources\runs\dataset_check\`（3张train+6张固定val五联图）。

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

# 先离线核验；换机下载模板及有效代理设置见第五阶段验收报告
Invoke-SegPython -m segformer_lab.download_model --out "$SegResource\models\mit-b0" `
  --revision '80983a413c30d36a39c20203974ae7807835e2b4' `
  --source-record 'reports\model_source.json' --verify-only `
  --report 'reports\model_snapshot_check.json'

# 加载核验 + 人工张量前向/反向/一步更新 + batch 显存
Invoke-SegPython -m segformer_lab.model_check --config 'configs\baseline.yaml' `
  --loading-report 'reports\model_loading.json' --check-report 'reports\model_check.json'

# Dataset 契约自检（39 项 + 9 张图 + JSON）
Invoke-SegPython -m segformer_lab.dataset_check --config 'configs\baseline.yaml' `
  --report 'reports\dataset_check.json' --batch-size 2 --seed 42

# 数据准备：重新生成三份清单（已存在且内容相同的图不会被覆盖）
Invoke-SegPython -m segformer_lab.prepare_data `
  --archive '..\..\docs\dataset\VisA\VisA_20220922.tar' `
  --source-zip '..\..\docs\dataset\VisA\spot-diff-main.zip' `
  --data-root 'E:\桌面\AI\segformer_resources\data\visa' `
  --category pcb1 --seed 42 --val-normal 120 --val-anomaly 12 `
  --reviewed-groups 'manifests\reviewed_groups.json' --out 'manifests'

# 数据审计：完整解码 1104 图 + 100 掩码，输出 reports\data_audit
Invoke-SegPython -m segformer_lab.audit_data --config 'configs\baseline.yaml' --out 'reports\data_audit'

# 固定预览：train 6 正常 + 6 异常；val 3 + 3（复用 manifests\val_preview_ids.txt）
Invoke-SegPython -m segformer_lab.preview_data --config 'configs\baseline.yaml' `
  --split train --normal 6 --anomaly 6 --out "$SegResource\runs\data_preview\train"
Invoke-SegPython -m segformer_lab.preview_data --config 'configs\baseline.yaml' `
  --split val --normal 3 --anomaly 3 --out "$SegResource\runs\data_preview\val" --write-ids

# 查环境里装了什么
Invoke-SegPip list
& $SegConda list --prefix $SegEnv
```

清理建议（**不要**直接删资源根）：`cache\wheels`（2.48 GB）在确认环境可用后可删，重装时按 `requirements-cu126.txt` 重新下载并校验哈希。

## 7. 维护约定

- PowerShell 入口脚本保持纯 ASCII（PowerShell 5.1 读无 BOM 的 UTF-8 脚本可能按本机代码页解码）；Python 源文件采用 UTF-8，中文学习说明写在 Markdown 中。
- 一切 Python 调用走 `Invoke-SegPython` / `run_project.ps1`，不用 PATH 上的 `python`。
- 每个编号阶段完成后更新 `docs/stages/` 对应文档与本 README 的第 5 节。
- 不伪造日志、指标或完成状态；未运行的命令标注「待执行」。

## 第三阶段验收

[第三阶段验收与完善](./reports/2026-10-31-stage03-acceptance.md)：固定划分530/132/442及清单身份均已复核，1205个数据文件与归档逐一匹配；23项项目回归和20项入口回归通过。该记录为 v1 历史验收；第四阶段现已通过，并将完整划分升级为 v2 分组协议，旧版清单已归档。

## 第四阶段验收与下一步

[第四阶段验收与完善](./reports/2026-10-31-stage04-acceptance.md)已记录用户五项检查、逐图结论、正常图疑点、v2 协议、真实损失统计和可复制的阶段 05 指令。完整数据审计 13 项通过，项目回归 62 项通过；数据、环境和批量预览不进 Git。当前尚无训练性能指标。

以上是历史阶段依赖；当前阶段06已经验收，下一步为阶段07。继续使用v2清单与同一几何实现。

## 第五阶段验收与下一步

[第五阶段验收报告](./reports/2026-10-31-stage05-model-loading-and-check.md)记录了来源/既有快照保护、加载门槛、显存测量顺序和代理日志的修复。重新执行：101 项回归、5 项加载及9项运行验收通过；encoder 192/192精确相等，快照全部文件及source未变。旧1058MiB记录只供追溯，当前隔离B2 reserved为1008MiB。

以上为阶段05结束时的指令，阶段06现已验收。来源记录进入Git，权重留资源目录，跨机按固定commit与完整文件hash恢复。

## 第六阶段验收与下一步

[第六阶段验收报告](./reports/2026-10-31-stage06-acceptance.md)：134项回归、39项数据契约通过；原图/增强GT坐标展示、skip-model状态、训练子集配置、seed/epoch与processor验证均已完善。9张PNG逐看，真实正常0011+异常087的B2 eval/no_grad前向为[2,2,128,128]，没有训练性能结论。

继续使用batch2、workers0、按(seed,epoch,index)翻转；原尺寸GT保持list，每轮调用set_epoch。仅数据诊断的exit0不代表全阶段通过，需scope=full_stage且full_stage_verified=true。当前代码/源文件/图哈希在dataset_check.json与stage06-acceptance.json，整批图留资源目录。

下一步只做阶段07的精确定义损失、原图logits恢复与累计指标，用人工数组和几何夹具验证。完整Agent指令见第六阶段验收报告第7节；暂不编写训练器或20步smoke。
