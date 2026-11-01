# SegFormer-B0 PCB1 有监督二值分割基线

本目录是 VisA PCB1 high-shot 协议下 SegFormer-B0 单基线实验的工程根目录：代码、配置、清单、阶段文档。
环境、缓存、数据、模型与运行产物**不在本目录**，统一放在仓库外的 `E:\桌面\AI\segformer_resources`。

- 当前进度：**步骤1—10已验收；第11阶段正式训练与保存模型通过独立验收，恢复数值复现保留限制，可进入阶段12。**50轮完成：13250尝试 / 13243成功 / 7次AMP跳步；best=epoch39，val全局IoU **0.7775**、异常macro IoU **0.5396**。当前**263项回归通过**，新进程完整val与预览复验一致。真正epoch1恢复状态与下一轮数据顺序通过，但最终last概率差约1.503e-4、XOR33，未通过1e-5/XOR0门槛。正式run连续完成，未使用resume；**本轮未执行test或阶段12**。
- 验收记录：[阶段01—02验收与完善](./reports/2026-10-31-stage01-02-acceptance.md)、[第三阶段验收](./reports/2026-10-31-stage03-acceptance.md)、[第四阶段验收](./reports/2026-10-31-stage04-acceptance.md)、[第五阶段模型加载与核验](./reports/2026-10-31-stage05-model-loading-and-check.md)、[第六阶段验收](./reports/2026-10-31-stage06-acceptance.md)
- 阶段索引：[docs/stages/README.md](./docs/stages/README.md)
- 最新验收：[第十一阶段](./reports/2026-11-01-stage11-acceptance.md)、[机器摘要](./reports/stage11-acceptance.json)、[阶段学习文档](./docs/stages/11-training-design-and-implementation.md)。下一步按[阶段12执行交接](./reports/2026-11-01-stage12-execution-handoff.md)进行独立推理与冻结协议测试。
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
├─ segformer_lab\losses.py            CE + 0.5×逐图前景 Soft Dice
├─ segformer_lab\metrics.py           原图恢复、int64 混淆量累计、全局指标
├─ segformer_lab\evaluation.py        IEEE FP32 评价精度上下文
├─ segformer_lab\metrics_check.py     42 项损失/指标检查 + 真实数据 probe
├─ segformer_lab\training.py          共享训练基座（AMP、评价、保存、严格重载、AMP 跳步、调度器）
├─ segformer_lab\smoke.py             阶段 08 冒烟入口与重载对照
├─ segformer_lab\train.py             阶段 09 8 图过拟合入口
├─ segformer_lab\pilot.py             阶段 10 完整数据 epoch 入口与新进程复验
├─ segformer_lab\train_full.py        阶段 11 正式 50 轮入口、checkpoint 与 epoch 边界恢复
├─ segformer_lab\full_check.py        完整保存状态、全量val与预览只读复验
├─ segformer_lab\resume_check.py      冻结恢复边界及真正last数值对照
├─ segformer_lab\subset.py            固定 8 图子集生成与校验
├─ configs\baseline.yaml              阶段 11 正式协议（显式 50 轮、13250/662、ieee_fp32）
├─ configs\smoke.yaml                 阶段 08 冒烟配置
├─ configs\overfit.yaml               阶段 09 过拟合配置（200 更新、LR 1e-4 无衰减）
├─ configs\pilot.yaml                 阶段 10 全量三轮配置（530/132、flip random、warmup+cosine）
├─ configs\resume_fixture.yaml        阶段 11 恢复 fixture（12/6 图、3 轮、仅用于续训对照）
├─ manifests\                         官方 CSV 副本 + train/val/test 清单 + split_meta.json + reviewed_groups.json + val_preview_ids.txt + overfit_train_ids.txt + resume_fixture_train_ids.txt
├─ tests\                             环境/分组/几何/审计/模型/数据集/损失指标/训练入口的单元测试（263 项）
├─ reports\environment.json           最新环境报告
├─ reports\data_audit\                审计摘要、逐图统计、重复与近重复候选
├─ reports\model_*.json               快照下载、加载与运行时核验记录
├─ reports\dataset_check.json         Dataset 契约检查结果
├─ reports\overfit_subset.json        8 图子集明细
├─ reports\stage08-gates.json         阶段 08 逐项门禁
├─ reports\stage09-summary.json       阶段 09 小型汇总（曲线/逐图/跳步/局限）
├─ reports\stage10-summary.json       阶段 10 小型汇总（三轮覆盖/损失/指标/耗时/复验）
├─ reports\stage10-unit-tests.txt     阶段 10 最终回归日志（231 项）
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
| 07 损失与原图尺度指标 | 已通过独立验收 | [07-loss-metrics-design-and-implementation.md](./docs/stages/07-loss-metrics-design-and-implementation.md) |
| 08 20 步冒烟检查 | 已通过独立验收 | [08-smoke-design-and-implementation.md](./docs/stages/08-smoke-design-and-implementation.md) |
| 09 8 张训练图过拟合检查 | 已通过独立验收 | [09-overfit-design-and-implementation.md](./docs/stages/09-overfit-design-and-implementation.md) |
| 10 完整训练集三轮 pilot | 已通过独立验收 | [10-pilot-design-and-implementation.md](./docs/stages/10-pilot-design-and-implementation.md) |
| 11 50 轮正式单基线 | 正式训练与保存模型已验收；恢复数值复现未通过 | [11-training-design-and-implementation.md](./docs/stages/11-training-design-and-implementation.md) |
| 12—13 | 未开始 | 见 [阶段索引](./docs/stages/README.md) |

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

## 5.5 损失与指标定义（阶段 07）

- 损失：`L = 1.0 × CE(ignore_index=255) + 0.5 × 逐图前景 Soft Dice`，入口 `segformer_lab\losses.py::segformer_loss`。
  - CE 在**上采样到 512 并转 FP32 的 logits** 上算（bilinear, `align_corners=False`），padding 与当初绘制时同分辨率地被忽略。
  - Dice 分子是逐像素交集 `2Σ(p·g)`；分母 `Σp + Σg + eps`，eps `1e-6`。
  - 只对 `Σg > 0` 的图求平均；全正常 batch 返回 `prob.sum()*0.0`（可反向的精确 0）。
  - 全忽略 batch 抛 `ValueError`；标签出现 0/1/255 之外的值同样拒绝。
- 指标：`segformer_lab\metrics.py`，`int64` 累计 TP/FP/FN/TN 后只除一次；分母为 0 返回 `null` + 原因，不加平滑项。
- 原图恢复（唯一入口 `restore_defect_probability`）：`128→512 双线性 → 按 geometry 裁 padding → 原 H/W 双线性 → softmax 取通道 1 → ≥0.5`。
- GT 用 `original_mask`，**不**把 512 标签放大回去；默认拒绝 `flipped=True`，以遵守 baseline 评价无增强协议。
- 输出：全局 IoU/Dice/precision/recall、正常图 FP 像素比与逐图面积比中位/P95、异常图宏平均、逐图分数（空 GT 且预测空写 NA）。
- 评价：B1、FP32模型/输入，`eval.precision: ieee_fp32`，由统一上下文关闭 TF32/autocast。
- 证据：`reports\loss_metrics_check.json`（42 项）、`reports\figures\stage07-restore-asymmetric.png`；随机头 probe 不算性能。

## 5.6 训练基座与 20 步冒烟（阶段 08）

- `segformer_lab\training.py`：seed、AdamW 分组（bias/norm 不衰减）、AMP+GradScaler 训练循环、跳步判定、B1 IEEE FP32 评价、自包含保存与 `strict=True` 重载。
- 固定子集：`manifests/overfit_train_ids.txt`（4 正常 + 4 异常，SHA `2ac81b81…`），由 `segformer_lab.subset` 生成；阶段 09 复用同一文件。明细 `reports/overfit_subset.json`。
- 冒烟配置：`configs/smoke.yaml`；run `segformer_resources\runs\smoke_001`（checkpoint、日志与批量图不进 Git）。
- 实测：20 成功 / 20 尝试 / 0 跳步；编码器梯度 0.0931、头 0.0253，参数 `max|Δ|` 6.014e-05 / 6.004e-05；稳态 58.5 ms/update，峰值 reserved 934.0 MiB。
- 重载：新进程 `strict=True`，missing/unexpected 0；6 张图最大概率差 0.0、XOR 0，混淆量与指标一致。
- 门禁：`reports/stage08-gates.json`（5/5 技术门禁）。冒烟分数只证通路，不是基线。

## 5.7 八图拟合与独立验收（阶段 09）

- `overfit_002` 200 成功/201 尝试/1 跳步，八图训练全局 IoU 0.008250→0.945710、异常图 macro 0.014414→0.760745；正常 FP 873,511→85。大目标占 GT 前景约96.43%，这些是训练子集排错结果。
- 新进程使用保存的结构、权重与 processor，208 个模型状态张量一致且有限；205 个优化器参数的 step 全为200。重新原图计分与历史全部数值一致；旧 per-image split=val 是元数据错误，真实ID均属于train。
- 共享更新步已区分真实 AMP 溢出和有限元素的范数溢出，以 optimizer post-hook 计真实执行，跳步核对参数/计数未变；评价恢复模式，局部图真正放大。完整项目207项回归通过。
- 修正后另跑 `smoke_003_stage09_acceptance`：20/20/0，峰值806.5/934MiB，新进程概率差0.0、XOR0。旧 overfit 未存完整概率，不能声称概率差0；原分数可以严格复算。
- [联合验收报告](./reports/2026-11-01-stage08-09-acceptance.md)、[阶段09学习文档](./docs/stages/09-overfit-design-and-implementation.md)、[机器复验](./reports/stage09-reload-check.json)、[原件及局限](./reports/stage08_09_evidence/README.md)。三次中断和有效run共429/500次尝试，原文件保留。

## 5.8 完整数据三轮 pilot（阶段 10）

- 入口 `segformer_lab/pilot.py`（新增，824 行，`--config/--out/--verify-run`），**不改阶段 09 的八图边界**；`train_step`/`step_optimizer`/`WarmupCosineLR` 放在 `training.py` 共享，`train.py` 未改。
- 配置 `configs/pilot.yaml`（SHA `74f4c8f1…`）：`train_ids_file: null`、530 train / 132 val、3 epoch、B2、累积 1、workers 0、seed 42、AMP FP16、`flip: random` + `horizontal_flip_p: 0.5`、LR 6e-5、wd 0.01、clip 1.0、warmup 0.05、minLR 1e-6。
- run：`segformer_resources\runs\pilot_001`（约 134 MiB：best/last 各 44.8 MiB、`model_state.pt` 14.9 MiB、预览概率 30 MiB、逐轮/逐次日志）。
- 调度：`total_steps = 3×265 = 795`，`warmup = round(0.05×795) = 40`；第 1 次更新 1.5e-6，第 40 次到 6e-5，第 795 次到 1e-6。**只在 optimizer 真执行后 `step()`**，跳步不推进。
- 实测：**795 尝试 / 795 成功 / 0 跳步**，`scheduler.completed` 795；每轮覆盖 530/530、重复 0、末批 2 张；翻转 248/265/265 张且指纹逐轮不同。
- 损失（加权：CE 按有效像素、Dice 按正 GT 图）：0.610957 → 0.499010 → 0.467857。
- val（132 张、B1 IEEE FP32、原图、阈值 0.5）：IoU 0.0103→0.2370→**0.6081**，Dice 0.0204→0.3832→0.7563，P/R（第 3 轮）0.8610/0.6743，异常宏平均 IoU 0.0031→0.0656→**0.2207**，正常图 FP 三轮全 0，NA 为空。
- **逐图关键事实**：第 3 轮 12 张异常图里 7 张预测前景为 0；045/046/059 占 val GT 前景 86 %，全局 IoU 由它们撑起。"正常图 0 误报"是欠报而非可靠。
- 复验：新进程严格加载 `best.pt`（SHA `d1698b28…`，epoch 3，step 795，strict 0/0），全量 val 混淆量与全局指标逐位相同；6 张预览概率最大差 **0.0**、XOR **0 / 9 013 680 px**。
- 资源：初始化 3.593 s、总计 **208.17 s**；三轮训练 61.06/45.20/45.63 s（读图等待 23.59/16.28/16.39 s）、全量 val 14.30/9.51/9.71 s；单步中位 124.3/105.4/105.4 ms；峰值 allocated **806.3** / reserved **934.0 MiB**。
- 身份：git HEAD `0c5f0bd` + 未提交改动；起末代码/配置/清单与 `git diff` SHA 一致（`source_and_config_unchanged: true`）。**未实现 resume**，`workers>0` 未验证。
- 证据：`reports/stage10-summary.json`（小型汇总）、`reports/stage10-unit-tests.txt`（231 项）、`reports/figures/stage10-*.png`（4 张精选）。

## 5.9 50 轮正式单基线（阶段 11，已验收并声明恢复限制）

- 协议：`stage.kind: formal`，530 train / 132 val，50轮、B2、累积1、workers0、FP16 AMP、同步随机水平翻转；评价B1 IEEE FP32、原图GT、阈值0.5。损失与数据划分不变。
- 预算：13250次计划更新、warmup662（`round(662.5)` ties-to-even）、cosine跨度12588；首步LR9.06344411e-8。实际13250尝试 / **13243成功** / 7跳步，逐轮530唯一图、共26500访次。最后实际更新LR为1.000045016763374e-6；更新后保存LR为1.0000330735426696e-6，二者含义不同。
- best=**epoch39**：完整val全局IoU **0.7775342969**、Dice **0.8748459012**、P0.9397792286、R0.8183056947；异常12图macro IoU **0.5395555278**、Dice0.6203118681；120张正常图1495个FP像素，11张有任意FP。这是验证选模结果，不是test成绩。
- 失败例：051预测空；042虽预测47像素但TP0；081命中67/6373。大缺陷主导全局分数，必须同时报告macro。warmup已在第3轮达到base LR，不能用“第4轮才达到”解释前3轮全背景；best与last差约2e-4，未据此计算噪声或置信区间。
- **263项回归通过**；新进程138次实际CUDA B1 FP32前向，132图逐图/macro/global/normal/NA全部对齐，6张预览概率差0.0、XOR0；best/last/mirror、205个优化器状态及计数核准，原run21个文件字节/SHA不变。
- 恢复修复：实际路径与配置/源码/快照/processor/数据身份严格绑定；保存processor实际使用；日志只接受与last匹配的完整epoch边界，破损/未提交尾日志保留并拒绝；优化器缺slot拒绝；修复重复resume事件与manifest自引用。旧正式checkpoint缺新身份字段且源码不同，当前代码不允许直接续训；只读推理不受影响。
- 恢复证据纠正：旧fixture拿epoch1 best概率及被覆盖的epoch3边界比较，旧“全状态相等”范围不充分，原件已归档。新A/B共36次更新，先冻结真正epoch1状态，16项状态和下一轮6批顺序/翻转通过；真正epoch3 last概率差**0.00015032291412353516**、XOR**33**，**未通过原定1e-5/XOR0门槛**。独立前缀也不同，未确定唯一原因，不提供逐位或该容差内续训保证。
- 正式run未用resume，故可用已复验best进入阶段12。训练2609.179s（计算1711.191s、loader等待897.991s）、验证533.355s、保存8.053s；GPU峰值allocated/reserved806.3/934MiB。workers>0未验证，不改环境或包。
- [验收报告](./reports/2026-11-01-stage11-acceptance.md)、[完整复验](./reports/stage11-reload-check.json)、[纠正后恢复汇总](./reports/stage11-resume-fixture.json)、[原件归档](./reports/stage11_evidence/README.md)、[阶段学习文档](./docs/stages/11-training-design-and-implementation.md)。正式日志无损gzip与原运行源码可追溯；权重/概率/批量输出留仓库外。

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

# 损失与指标自检（42 项；含 6 张固定 val 图的 GPU 前向）
Invoke-SegPython -m segformer_lab.metrics_check --config 'configs\baseline.yaml' `
  --report 'reports\loss_metrics_check.json'

# 只跑夹具（仍核验快照身份；不执行 GPU 模型前向；不代表全阶段通过）
Invoke-SegPython -m segformer_lab.metrics_check --skip-model --device cpu `
  --report "$SegResource\audit\stage07-dryrun.json"

# 固定 8 图子集（阶段 08/09 共用；--verify-only 只复核）
Invoke-SegPython -m segformer_lab.subset --config 'configs\baseline.yaml'

# 阶段 08：20 步 AMP 冒烟 + 保存
Invoke-SegPython -m segformer_lab.smoke --config 'configs\smoke.yaml' `
  --out "$SegResource\runs\smoke_001"

# 阶段 08：新进程严格重载并复算
Invoke-SegPython -m segformer_lab.smoke --verify-run "$SegResource\runs\smoke_001" `
  --config 'configs\smoke.yaml'

# 阶段 09：8 图过拟合检查（200 次成功更新；输出目录必须为空或不存在）
Invoke-SegPython -m segformer_lab.train --config 'configs\overfit.yaml' `
  --out "$SegResource\runs\overfit_002"

# 阶段 10：完整数据三轮 pilot（530 train / 132 val；输出目录必须为空或不存在）
Invoke-SegPython -u -m segformer_lab.pilot --config 'configs\pilot.yaml' `
  --out "$SegResource\runs\pilot_001"

# 阶段 10：新进程严格加载 best.pt，重算全量 val 与 6 张预览
Invoke-SegPython -u -m segformer_lab.pilot --verify-run "$SegResource\runs\pilot_001" `
  --config 'configs\pilot.yaml'

# 阶段 11：已完成；当前只需按需进行只读复验，不要重跑正式50轮
# 输出报告在原训练run之外；默认输出到同级pcb1_b0_v2_seed42-reload-check.json
Invoke-SegPython -u -m segformer_lab.train_full --verify-run "$SegResource\runs\pcb1_b0_v2_seed42" `
  --config 'configs\baseline.yaml' --report 'reports\stage11-reload-check.json'

# 新实验的训练/恢复命令见阶段11学习文档；必须选择新目录。
# 历史resume_fixture_A/B及正式run均保留，不覆盖。
# 当前正式run已跑满50轮；旧checkpoint不满足新增身份字段，不能直接resume。

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

以上是历史阶段依赖；当前进度见文首。继续使用v2清单与同一几何实现。

## 第五阶段验收与下一步

[第五阶段验收报告](./reports/2026-10-31-stage05-model-loading-and-check.md)记录了来源/既有快照保护、加载门槛、显存测量顺序和代理日志的修复。重新执行：101 项回归、5 项加载及9项运行验收通过；encoder 192/192精确相等，快照全部文件及source未变。旧1058MiB记录只供追溯，当前隔离B2 reserved为1008MiB。

以上为阶段05结束时的指令，阶段06现已验收。来源记录进入Git，权重留资源目录，跨机按固定commit与完整文件hash恢复。

## 第六阶段验收与下一步

[第六阶段验收报告](./reports/2026-10-31-stage06-acceptance.md)：134项回归、39项数据契约通过；原图/增强GT坐标展示、skip-model状态、训练子集配置、seed/epoch与processor验证均已完善。9张PNG逐看，真实正常0011+异常087的B2 eval/no_grad前向为[2,2,128,128]，没有训练性能结论。

继续使用batch2、workers0、按(seed,epoch,index)翻转；原尺寸GT保持list，每轮调用set_epoch。仅数据诊断的exit0不代表全阶段通过，需scope=full_stage且full_stage_verified=true。当前代码/源文件/图哈希在dataset_check.json与stage06-acceptance.json，整批图留资源目录。

以上为阶段 06 结束时的依赖；阶段 07 已完成，当前下一步见下节。

## 第七阶段验收与下一步

[第七阶段验收](./reports/2026-11-01-stage07-acceptance.md)记录四项损失约定、NA、原图恢复、FP16 CE 边界与数值精度修订。42 项检查和 186 项回归通过；旧“12 px”算法已撤回，现用逐像素 XOR、概率容差及 B1 重复前向验证。

评价统一写 `eval.precision: ieee_fp32`，用 `segformer_lab.evaluation.evaluation_precision` 上下文禁用 TF32/autocast并在退出后恢复，配合 FP32 模型/输入与 B1。训练 AMP 独立保持。原图 GT 不缩小，正常空预测/零 precision 分母报告 NA；probe 不是基线成绩。

下一轮按[08—09 执行交接](./reports/2026-11-01-stage08-09-execution-handoff.md)串行处理：先 20 次成功更新的 smoke 与新进程重载，通过门槛后另建过拟合实验。两个配置、run 和阶段文档独立，完成后一起验收；不自动进入 pilot 或正式训练。环境、数据、模型和批量输出仍留在仓库外。

## 第八、九阶段验收与下一步

[联合验收](./reports/2026-11-01-stage08-09-acceptance.md)已完成，原记录、当前207项回归、三个新进程复验与精选图均可追溯。环境、数据、权重和完整概率留仓库外，小型原件在 `reports/stage08_09_evidence`。旧阶段08执行源码SHA与提交不同的历史局限明确保留，不能宣称其准确源码已恢复。

以上为阶段09结束时的安排；阶段10现已验收，当前下一步见下节。

## 第十阶段验收与下一步

[第十阶段验收](./reports/2026-11-01-stage10-acceptance.md)已通过：231项回归，新进程完整132张val的逐图/macro/global/NA一致，6张预览概率差0、XOR0；best/last的205个优化器参数step均795，30项产物SHA通过。原三轮日志及源码在 `reports/stage10_evidence`，新复验输出独立JSON，原run保持不变。

验收补强了NaN/shape/XOR、完整保存状态和起止数据身份门槛，修正了配置构造器原地修改、NA解释及产物数。原模型来源未单独冻结开始时snapshot哈希、增强图无逐样本视觉记录等局限如实保留。小目标漏检仍在，不以全局0.608作为论文成绩。

以上为第十阶段验收时的历史安排，正式训练现已完成；当前下一步见下节。

## 第十一阶段验收与下一步

[正式训练与保存模型已验收](./reports/2026-11-01-stage11-acceptance.md)：263项回归，完整val与固定预览新进程复验通过。冻结best39，SHA `192c827bef00607f4258e5fadf0f1d97970ca5cc727a0add749be0fba5997485`；原训练文件保持不变。恢复状态与顺序已确认，最终last数值对照未通过原容差，限制详见验收记录；不影响连续完成的正式模型评价。

下一步只做[阶段12](./reports/2026-11-01-stage12-execution-handoff.md)：从保存best、模型结构及processor独立推理，先通过完整132图val门禁，再在冻结协议下评价442图test（402正常+40异常）。固定阈值0.5、B1 IEEE FP32、原图尺度还原；不能依据test选参。必要技术复核可按同一协议重跑并保留原因与所有结果，不按分数挑选。本轮未执行阶段12或test。

跨机器需另外复制已训练best、结构和processor并核SHA；Git不包含大权重，重下MiT-B0不能替代已训练best。
