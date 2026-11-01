# 复现说明

## 1. 需要从仓库外单独搬运的东西

Git 里**没有**模型权重、数据与环境。换一台机器，至少需要：

| 内容 | 位置 | 校验方式 |
| --- | --- | --- |
| 已训练的 best | `runs/pcb1_b0_v2_seed42/checkpoints/best.pt` | SHA-256 `192c827b…`（完整值见 `reports/stage12-summary.json`） |
| 结构与 processor | 同 run 的 `model_config.json`、`processor/` | 与 `code_identity_start.json` 中的摘要一致 |
| PCB1 数据 | `data/visa/pcb1/Data/{Images,Masks}` | 清单 SHA：train/val/test 见实验卡 |
| Conda 环境 | `env/`（Python 3.12.16 + torch 2.10.0+cu126） | `requirements-lock.txt` |

**重新下载 `nvidia/mit-b0` 不能替代 best**：快照只是初始化起点，训练后的权重只存在于 best.pt。

## 2. 环境与上下文

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
Set-Location -LiteralPath '<repo>\work\SegFormer'
. '.\scripts\enter_project.ps1'
$ErrorActionPreference = 'Continue'
```

## 3. 复算 val（应先与 `reports/stage12_evidence/val_metrics.json` 对齐）

```powershell
Invoke-SegPython -u -m segformer_lab.evaluate --config 'configs\baseline.yaml' `
  --run '<...>\runs\pcb1_b0_v2_seed42' --split val --out '<new-dir>\val'
```

门禁 `gate.json` 应显示 `passed=true`；任何一项不一致先停下定位（eval 模式、归一化、还原顺序、阈值）。

## 4. 复现图表

```powershell
Invoke-SegPython -u -m segformer_lab.summarize --config 'configs\baseline.yaml' `
  --run '<...>\runs\pcb1_b0_v2_seed42' `
  --eval-run '<...>\runs\pcb1_b0_v2_seed42_eval_001' `
  --out '<repo>\work\SegFormer\reports\pcb1_b0_v2_seed42'
```

`summarize` 不训练、不选 best；它只读 `epochs.jsonl` 与阶段 12 的评价目录，并用同一冻结通道为展示图重算概率（二值结果必须与阶段 12 的 PNG 完全一致，否则脚本报错退出）。

## 5. 约束

- test 已经评价过一次；再跑属于技术复核，需保留原因与全部结果，不得按分数择优。
- 阈值 0.5、B1、IEEE FP32、无 TTA / 后处理，不得为提升分数而修改。
