# 复现说明

## 1. 需要从仓库外单独搬运的东西

Git 里保存源代码、配置、清单和便携摘要，**没有**模型权重、完整数据与 Conda 环境。当前冻结加载器要求原样保留整个训练 run 的 21 个文件（154 043 135 bytes），下表只按用途列出内容，不是可删减的最小包：

| 内容 | 位置 | 校验方式 |
| --- | --- | --- |
| 已训练的 best 与本次镜像 | `checkpoints/best.pt`、`checkpoints/model_state.pt` | best SHA-256 `192c827bef00607f4258e5fadf0f1d97970ca5cc727a0add749be0fba5997485`；其余文件按 run `summary.json.artifacts.files` 核验 |
| 结构与预处理 | `model_config.json`、`processor/preprocessor_config.json`、`resolved_config.yaml` | 同上；processor 摘要来自保存的工件清单，不是预训练快照 |
| 曲线、选模与工件完整性记录 | `summary.json`、`epochs.jsonl`、`artifact_manifest.json` | 当前冻结加载器核验完整 run 的文件集合与 SHA；不要删去表中未逐项列出的历史日志/预览等工件，推荐整目录保留 |
| val 门禁参考与历史身份 | `pre_save_probabilities.npz`、`code_identity_{start,end}.json` | val 预览复核需要 NPZ；身份文件用于追溯训练时源码（推荐整个 run 保留） |
| 阶段 12 评价目录 | 完整的 `val/` 与 `test/`，包括 `evaluation_meta.json`、`artifact_manifest.json`、`metrics.json`、`per_image.{json,csv}`、`selection.json`、`display_rules.json`、`predictions/*.png`，以及 `val/gate.json` | 原评价与验收技术复核分目录保存；summarize 校验清单 SHA、身份与选择一致性后才绘图 |
| PCB1 原图、GT、解压索引与官方划分副本 | `data/visa/pcb1/Data/{Images,Masks}`、`data/visa/extraction_manifest.json`，以及 `split_meta.sources.official_csv.copy` 指向的文件 | 从原审计数据恢复；split_meta 固定解压索引与官方 CSV SHA，评价/绘图前对照索引核验图像/GT 字节；清单 CSV SHA 不能代替数据字节校验 |
| 环境重建资料 | `requirements-runtime.txt`、`requirements-lock.txt`、`requirements-cu126.txt`、阶段 02 文档；资源区 `audit/conda-explicit-win64.txt` | 在新前缀重建并运行 env_check；不要直接复制/搬移原 Conda prefix |

**重新下载 `nvidia/mit-b0` 不能替代 best**：快照只是初始化起点，训练后的权重只存在于 best.pt。

## 2. 环境与上下文

下列 `<repo>` / `<resource-root>` 是原记录路径的占位说明，不代表任意位置可直接替换。当前加载器还逐项比较 `resolved_config.yaml` 的绝对路径，因此仓库、数据/模型/输出根及官方 CSV 副本必须还原至原记录的同一绝对布局（本机仓库 `E:\桌面\AI\AI_Try`、资源 `E:\桌面\AI\segformer_resources`）。相对 manifest 从项目目录解析，但克隆至另一位置也会改变其 resolved 路径。修改 YAML 中路径又会改变 config SHA，当前身份检查会拒绝。不同根路径需要后续明确审计的迁移协议，本轮未实现，不能改写原 run 绕过检查。跨平台/不同 GPU 的逐位一致性未保证，先复核 val。

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
Set-Location -LiteralPath '<repo>\work\SegFormer'
. '.\scripts\enter_project.ps1' -ResourceRoot '<resource-root>' -CondaExe '<conda.exe>'
$ErrorActionPreference = 'Continue'
```

只在全新的、尚不存在的 `$SegEnv` 前缀重建环境（现有环境先核对，勿覆盖）：

```powershell
if (Test-Path -LiteralPath $SegEnv) { throw 'Use a fresh prefix or verify the existing environment first' }
Assert-SegCondaIsolation
& $SegConda create --prefix $SegEnv --copy --no-shortcuts --no-default-packages `
  --override-channels --channel 'https://repo.anaconda.com/pkgs/main' `
  python=3.12.16 pip -y
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
Invoke-SegPip install --no-user -r requirements-runtime.txt --index-url https://pypi.org/simple
Invoke-SegPython -m pip check
Invoke-SegPython -m segformer_lab.env_check --help
```

按阶段 02 的 env_check 参数记录解释器、包版本、CUDA 实际运算和缓存路径。`requirements-runtime.txt` 才是完整可移植包清单；原始 `requirements-lock.txt` 含本机 wheel URL，不能原样在另一台机器 replay。Conda 管理器、驱动和 OS 仍是共享层，环境重建不是 VM 级隔离，也不保证不同硬件逐位复现。

训练时 git HEAD 加未提交源码的身份保存在 `code_identity_start.json`；`evaluation_meta.json.code` 保留评价源码 SHA；本次生成器 SHA 和输入文件 SHA 保存在 `results-summary.json`。仅 checkout 某个 HEAD 不足以复原当时未提交源码。`reports/stage12_13_evidence/pre_acceptance_source/` 保存验收前已提交版本；它与原评价元数据内 inference/predict 的 SHA 不同，原打分源码字节不可恢复，不能称其为完整原始源码快照。原评价身份与验收修复后技术复核身份分别留档，新 val/test 证明当前通道而非倒推原源码。

## 3. 复算 val（应先与 `reports/stage12_evidence/val_metrics.json` 对齐）

```powershell
Invoke-SegPython -u -m segformer_lab.evaluate --config 'configs\baseline.yaml' `
  --run '<resource-root>\runs\pcb1_b0_v2_seed42' --split val --out '<new-eval-dir>\val'
```

门禁 `gate.json` 应显示 `passed=true`；任何一项不一致先停下定位（eval 模式、归一化、还原顺序、阈值）。

## 4. 复现图表

```powershell
Invoke-SegPython -u -m segformer_lab.summarize --config 'configs\baseline.yaml' `
  --run '<resource-root>\runs\pcb1_b0_v2_seed42' `
  --eval-run 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42_eval_acceptance_20261101' `
  --out '<new-empty-results-dir>'
```

`--out` 必须是尚不存在的新目录，并且不得位于 run、评价来源、源码、配置或数据目录内；程序拒绝复用已有目录。先保留旧摘要，再在新目录生成，验证后才更新仓库内便携交付。`summarize` 不训练、不选 best；它只读上述 run 与评价工件，并用同一冻结通道为 23 张展示图重算概率（二值结果必须与来源评价 PNG 完全一致，否则退出）。原始评价的图表与技术复核图表要记录各自 eval 路径，不能混写来源。

## 5. 约束

- test 已经评价过一次；再跑属于技术复核，需保留原因与全部结果，不得按分数择优。
- 阈值 0.5、B1、IEEE FP32、无 TTA / 后处理，不得为提升分数而修改。
