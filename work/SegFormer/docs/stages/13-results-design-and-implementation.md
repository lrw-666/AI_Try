# 阶段 13：结果可视化、失败分析与实验卡

| 项目 | 值 |
| --- | --- |
| 创建/更新时间 | 2026-11-01 16:00 +08:00 |
| 对应步骤 | 主指南第 15 节「步骤 13：结果可视化、失败分析与实验卡」 |
| 前置状态 | 原始 val 门禁 29/29；本轮独立验收完整重跑 132 val / 442 test，v2 门禁 **74/74**，全部逐图记录/二值 PNG 与原评价一致 |
| 本阶段范围 | 只从既有 run 与评价目录生成曲线、样本图、最差案例、失败分析、实验卡与复现说明 |
| 明确不做 | 不训练、不 re-select best、不重算另一套指标、不改阈值/后处理、不实施切块/多尺度/DINOv3 等新实验 |
| 状态 | **已通过阶段 12—13 联合独立验收；5 张验收版图已逐张读图检查通过** |
| 对应代码身份 | 初版 `summarize.py` 与 4 项单测；本轮独立验收新增 3 项 CPU 回归（共 7 项），生成器 SHA 随新摘要记录；训练/原评价身份与验收修复身份分别留档 |
| 交付目录 | `reports/pcb1_b0_v2_seed42/{results.md, experiment-card.md, reproduction.md, curves.csv, failure-cases.csv, results-summary.json, figures/}` |

## 1. 本阶段解决什么问题

前 12 个阶段产出了一堆数字和文件：50 轮日志、132 张 val 的逐图统计、442 张 test 的逐图统计、best checkpoint。但这些还不是"能给人看的结果"。本阶段只做三件事：

1. **把数字变成图**：训练曲线（loss / LR / val 指标 / 正常误报）与三类样本图（固定异常、固定正常、最差案例），色标、配色、图例统一，任何人打开就能读。
2. **把失败说清楚**：哪些图漏了、漏在什么地方、误报长什么样，用"观察—证据—可能原因—下一项验证"写下来，而不是只报一个 IoU。
3. **把实验变成可复现的材料**：实验卡（数据/模型/超参/指标/资源/局限）、复现说明（换机要搬什么、怎么核 SHA）、失败案例表。

边界很硬：**不训练、不重新选 best、不算第二套指标**。所有数字要么来自阶段 11 的 `epochs.jsonl`，要么来自阶段 12 的 `metrics.json`/`per_image.json`；本阶段新增的只有"绘图用的中间量"和"可视化用的概率图"。

## 2. 技术栈与版本

| 组件 | 版本 | 本阶段作用 | 状态 |
| --- | --- | --- | --- |
| Python | 3.12.16 | 项目 runner 内 | 已运行 |
| matplotlib | 3.11.2 | 曲线与 5 类面板图（`Agg` 后端） | 已运行 |
| NumPy / Pillow | 2.5.4 / 12.3.0 | 概率数组、原图读取、PNG 保存 | 已运行 |
| PyTorch / transformers | 2.10.0+cu126 / 4.57.6 | 仅用于为展示图重算概率（同一冻结通道） | 已运行 |
| 新增依赖 | **无** | 未安装、未升级任何包 | — |

## 3. 设计与取舍

### 3.1 单一数据源：只读，不重算指标

```text
run/epochs.jsonl        -> curves.csv -> training-curves.png
run/summary.json        -> best epoch / 选模规则 / 资源
eval/test/per_image.json-> failure-cases.csv、各类选择
eval/test/metrics.json  -> results.md、experiment-card.md 的 test 行
eval/val/metrics.json   -> 同上 val 行
eval/test/display_rules.json -> 选择规则的来源与 SHA
```

`summarize` 没有自己的混淆量累计、没有自己的 IoU 公式；阶段 12 的 `metrics.json` 是唯一来源。

### 3.2 展示图需要浮点概率，但阶段 12 故意没存

442 张原尺寸 float32 概率约 2.6 GB，阶段 12 只写了 0/1 PNG。本阶段对被选中的 **23 张**（6 固定异常 + 6 固定正常 + 最差 IoU/recall 并集 + 正常 FP 前 5，去重后 23）用**同一条** `inference` 通道重算概率：

- 同一 `best.pt`、同一 `processor`、同一 `evaluation_precision` 上下文；
- 重算出的二值掩码必须与阶段 12 的 `predictions/*.png` **XOR = 0**，否则脚本直接退出（本次 23/23 全部为 0）；
- 每张图的源文件 SHA 记进 `failure-cases.csv`。

这条"重算 + 校验"是允许的（交接文档明确允许），但它只用于**画图**，不改变 `metrics.json` 里的任何数。

### 3.3 颜色与色标：不留解释空间

- 概率图统一 `cmap=magma, vmin=0, vmax=1`，**不按图单独拉伸**——否则低置信度看起来会像高置信度；
- 叠加图只涂三种颜色：TP 绿 `#00A666`、FP 红 `#E24B4A`、FN 蓝 `#378ADD`，图例固定在底部；
- 正常图也保留原尺寸全零 GT 面板：原图 / 全零 GT / 0—1 概率 / 误报（红）。这既满足交接要求，也让读者直接核对正常样本的标签语义。

### 3.4 曲线要写清"哪个 LR"

训练日志里每轮记了 `lr_first`/`lr_last`，它们是**该轮第一/最后一次尝试前记录的学习率**（`train_step` 之前取的，尝试可能因 AMP 溢出跳步）。epoch 结束后优化器里的 LR 则是 `lr_at(scheduler_completed)`。`curves.csv` 两者都存：

- `lr_last_used`：日志实测值，画主曲线；
- `lr_after_epoch_computed`：用同一公式按 `scheduler_completed` 反算，画虚线。

这样既不用"反算值"冒充实测值，也不会让人误以为两者是同一个数。

### 3.5 Dice 曲线的聚合口径

损失里的 Dice 项是**正 GT 图上的逐图均值**。epoch 聚合把各 batch 的 Dice 均值按该 batch 正 GT 图数加权；全正常 batch 的精确 0（图相连的零）权重为 0，不进分母。本次每轮遍历 48 张正 GT 训练图，包括发生 AMP 跳步的尝试所记录的前向损失。曲线从 0.9836 到 0.1538 描述训练过程中的正图 Soft Dice 损失；它不是原图空间硬阈值评价 Dice，也不能证明背景分割完美。详情见 `results-summary.json.notes`。

### 3.6 选择规则不是本阶段定的

`display_rules.json` 是阶段 12 在**解码任何 test 图之前**写的，本阶段只是执行它，并把它的 SHA 记进 `results-summary.json`。这样"挑图"这件事的时间顺序可查：规则先于结果。

## 4. 文件、接口与产物

| 文件 | 新增 | 作用 |
| --- | --- | --- |
| `segformer_lab/summarize.py` | 新增 | 曲线/图表/失败表/实验卡/复现说明生成器 |
| `tests/test_summarize.py` | 初版 4 项 + 验收新增 3 项 | LR/字段/配色；输出证据保护、评价工件篡改拒绝、processor 身份不一致拒绝 |
| `reports/pcb1_b0_v2_seed42/curves.csv` | 新增 | 50 行逐轮数值（12 KB） |
| `reports/pcb1_b0_v2_seed42/failure-cases.csv` | 新增 | 23 行入选图、理由、计数、源 SHA、重算 XOR |
| `reports/pcb1_b0_v2_seed42/results.md` | 新增 | val/test 指标、读法、图表索引 |
| `reports/pcb1_b0_v2_seed42/experiment-card.md` | 新增 | 实验卡（按主指南模板填真实值） |
| `reports/pcb1_b0_v2_seed42/reproduction.md` | 新增 | 换机复现步骤与必须单独搬运的内容 |
| `reports/pcb1_b0_v2_seed42/results-summary.json` | 新增 | 机器可读摘要，含输入验证 SHA、best step 与生成器 SHA |
| `reports/pcb1_b0_v2_seed42/figures/*.png` | 新增 5 张 | 训练曲线 + 4 张样本/最差案例表 |

图表清单：

| 图 | 内容 | 尺寸/体积 |
| --- | --- | --- |
| `training-curves.png` | 2×2：loss（total/CE/Dice）、LR（记录值 + 反算）、val IoU/Dice 与异常 macro IoU、正常 FP（symlog 保留零）；标 best 39 | 验收版 143 536 bytes |
| `anomaly-fixed-6.png` | 6 行 × 5 列：原图 / GT / 概率 / 二值 / TP-FP-FN | 1 252 032 bytes |
| `anomaly-worst-cases.png` | IoU 最低 ∪ recall 最低（6 张去重），同 5 列 | 1 217 151 bytes |
| `normal-fixed-6.png` | 6 行 × 4 列：原图 / 全零 GT / 概率 / 误报（红） | 验收版 1 191 982 bytes，已补 GT 列 |
| `normal-worst-fp.png` | 正常图 FP 最多 5 张，同 4 列 | 验收版 967 583 bytes |

## 5. 关键实现讲解

### 5.1 LR 反算必须和训练用同一个公式

```python
def lr_from_schedule(completed, plan):
    base, floor = plan["base_lr"], plan["min_lr"]
    if completed < plan["warmup_steps"]:
        return base * (completed + 1) / plan["warmup_steps"]
    progress = min(1.0, (completed - plan["warmup_steps"] + 1) / (plan["total_steps"] - plan["warmup_steps"]))
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * progress))
```

`tests/test_summarize.py` 在 `completed ∈ {0, 1, 661, 662, 663, 5000, 13249}` 七个点上把它与 `training.WarmupCosineLR.lr_at` 逐位对比（`places=15`），防止画图用的公式悄悄跑偏。

### 5.2 配色只在一处定义

```python
TP_COLOR = (0, 166, 102); FP_COLOR = (226, 75, 74); FN_COLOR = (55, 138, 221)

def _overlay(source, gt, pred):
    canvas = (source * 0.55 + 255 * 0.45).astype(np.uint8)   # 淡化原图，保留结构感
    canvas[(pred == 1) & (gt > 0)] = TP_COLOR
    canvas[(pred == 1) & (gt == 0)] = FP_COLOR
    canvas[(pred == 0) & (gt > 0)] = FN_COLOR
    return canvas
```

单测用 2×3 的小数组逐像素断言三种颜色与背景值，避免"图看着对、颜色定义反了"。

### 5.3 重算 + 二值校验

```python
probability = inference.probability_map(frozen.model, item, device, cfg)
pred = (probability >= threshold).astype(np.uint8)
xor = int((pred != stage12_mask).sum())
if xor: mismatch.append(...)
```

`mismatch` 非空即 `SystemExit`，不生成图、不留半成品。

### 5.4 在读模型与写图之前验证来源

验收修订为生成器增加了三层保护：

1. `--out` 必须是尚不存在的新目录，且不能覆盖或落在 run、评价来源、源码、配置、数据等受保护输入目录中。输出目录检查先于模型加载，避免原记录被无意覆盖。
2. 校验 val/test `artifact_manifest.json` 中全部工件的 bytes/SHA，再比较 best SHA、epoch/step、processor、model_config、summary、resolved_config、原始配置、清单与 split_meta 身份；核验 val 门禁通过，指标与元数据协议一致。
3. 以阶段 12 同一 `display_selection` 检查保存名单与既有逐图记录/冻结规则相符（不计算新指标、不改名单），加载后的冻结模型身份仍须一致；23 张概率二值 XOR 全零后才开始写图。保存这些来源 SHA，便于区分原始评价和验收技术复核。

CPU 回归先观察到 3 项失败，再修复至 7/7 通过：原版会继续读取配置并写入非空输出目录，也没有评价清单/processor 绑定；修订版在这些条件下拒绝操作，保留旧证据。

## 6. 如何操作与实际执行了什么

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
. '.\scripts\enter_project.ps1'
$ErrorActionPreference = 'Continue'

Invoke-SegPython -m segformer_lab.summarize --help

& $SegConda run --no-capture-output --prefix $SegEnv python -m segformer_lab.summarize `
  --config 'configs\baseline.yaml' `
  --run 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42' `
  --eval-run 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42_eval_001' `
  --out 'E:\桌面\AI\AI_Try\work\SegFormer\reports\pcb1_b0_v2_seed42'

Invoke-SegPython -m unittest discover -s tests -t . -v
```

| 次序 | 操作 | 结果 | 证据 |
| --- | --- | --- | --- |
| 1 | `--help` | 5 个参数（含 `--device`），退出 0 | 终端 |
| 2 | 首轮 summarize | 4 图 + 3 文档生成，23 张重算 XOR 全 0 | `results-summary.json` |
| 3 | 发现并修正 `tp_zero` 误标（正常图被写 `tp_zero`） | 重跑一次，正常图只保留 `normal_false_positive` | `failure-cases.csv` |
| 4 | 单测（新增 4 项） | 4/4 | `tests/test_summarize.py` |
| 5 | 完整回归 | **277 项，14.142 s，OK**（273 → 277） | `reports/stage13-unit-tests.txt` |
| 6 | 人工看图 | 曲线 1 张 + 样本/最差图 4 张，逐张检查色标、图例、尺寸、覆盖位置 | 见 7.6 |

以上表格记录初次执行；初版的 4 张样本图加 1 张曲线合计 **5 张图**。联合验收已先保存原始摘要、验收前源码和 eval_001 原工件身份，再在 `..._eval_acceptance_20261101` 技术复核目录评价，并在新输出目录生成 5 张验收版图表；最终命令、SHA、测试数与读图证据单独记录，不把重生成写成原先只执行一次。

本轮已实际完成的独立复核：

| 验证 | 实际结果 | 仓库内证据 |
| --- | --- | --- |
| 全量 val / test 新进程技术复核 | CUDA FP32/B1 实际前向 **132 / 442 次**；全部旧指标字段、逐图记录与二值 PNG 数组一致；从原图 GT 和实际输出 PNG 独立累计的计数也一致 | `reports/stage12-13-{val,test}-acceptance-check.json` |
| val v2 门禁 | **74/74 通过**，6 张预览概率差 0、XOR 0；身份/源码/数据/协议/运行时绑定冻结 | `..._eval_acceptance_20261101/val/gate.json` |
| 完整 CPU 回归 | **291 项，11.593 s，OK**，含本阶段 7 项 | `reports/stage12-13-unit-tests.txt` |
| 原证据只读性 | 原训练 run 的 21 个文件与原 `eval_001` 保持不变；新评价单独留档 | 上述 val/test acceptance JSON 中的只读性字段 |
| 图表重生成与逐张读图 | 新目录生成成功，**23 张二值 XOR 全零**；5 张图逐张检查通过，正常全零 GT、val Dice 与 FP=0 的轴显示均已修复 | `..._results_acceptance_20261101/results-summary.json` 与 `figures/`；仓库交付见 `reports/pcb1_b0_v2_seed42/` |

这些是固定 best/数据/阈值/数学协议的技术复核；没有根据重跑成绩择优，也没有训练或测试驱动调参。原评价源码字节不可恢复的限制仍保留，当前复核证明的是修复后通道。

本次图表重生成命令（路径为已执行记录，重跑必须另选新输出路径）：

```powershell
Invoke-SegPython -u -m segformer_lab.summarize --config 'configs\baseline.yaml' `
  --run 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42' `
  --eval-run 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42_eval_acceptance_20261101' `
  --out 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42_results_acceptance_20261101'
```

## 7. 结果与分析

### 7.1 训练曲线（`figures/training-curves.png`，50 轮）

| 项目 | 起点 | 终点 | 观察 |
| --- | --- | --- | --- |
| 总损失（加权） | 0.8030（ep1） | 0.0773（ep50） | 前 4 轮急降（0.803→0.473），之后缓慢下降 |
| CE | 0.3113 | **0.0004002（ep50）** | 背景像素多；低 CE 不能替代前景指标 |
| Dice 项 | 0.9836 | **0.1538301（ep50）** | 正 GT 训练图按图加权；不是 val 硬阈值 Dice |
| val 全局 IoU | 0.0（ep1—3） | 0.7775（ep39）/ 0.7773（ep50） | 第 4 轮跳出全背景解 |
| val 异常 macro IoU | 0.0 | 0.5396（ep39）/ 0.5372（ep50） | 比全局 IoU 更低；后期仍有波动 |
| 正常图 FP | 0（ep1—3） | **1 448 px（ep50）** | 全程最大 3 187；ep4 为 60；symlog 轴显示真实零 |
| LR（每轮末次尝试前记录） | **2.4018127e-5（ep1）** | **1.0000450e-6（ep50）** | 662 次成功更新 warmup；总成功 13 243/计划 13 250，7 次跳步导致末值略高于 floor |

前 3 轮 val IoU/FP 都为 0，逐图诊断显示 12 张异常也均为空预测，说明当时阈值输出全背景。warmup 在第 3 轮内已完成（该轮累计成功 793 次 > 662），因此不能从时间重合推断 warmup 导致全背景，更不能称为必然结果。需要独立 warmup 对照才能讨论因果。每轮首/末 LR 也不能混读：9.0634e-8 是整个训练第一次尝试的 LR，不是 ep1 末值。正常 FP=0 表示没有正常像素超过阈值，并不单独说明模型训练良好；batch 的 Dice=0 与 epoch 0.1538 的口径见 §3.5。

### 7.2 固定样本（按 image_id 升序前 6）

异常（`anomaly-fixed-6.png`）：

| 图 | GT 像素 | 预测像素 | IoU | 观察 |
| --- | ---: | ---: | ---: | --- |
| 000 | 1 524 | 534 | 0.298 | 引脚处细长 GT，TP 473 / GT 1 524（recall 31.04 %）；未统计概率峰值，不能从图推断其刚过阈值 |
| 001 | 1 687 | 986 | 0.497 | 命中主体，两端漏 |
| 002 | 1 702 | 1 776 | 0.618 | 面积接近但形状不完全重合，边缘外扩 |
| 003 | 1 618 | 1 599 | 0.828 | 面积几乎相同、边界贴合，是 6 张里最好的一张 |
| 008 | 3 739 | 3 136 | 0.650 | TP 2 708、FN 1 031、FP 428；净面积差 603 不等于漏检像素数 |
| 012 | 3 712 | 2 499 | 0.394 | TP 1 754、FN 1 958、FP 745，实际 recall **47.25 %**；预测/GT 面积比不能当覆盖率 |

正常（`normal-fixed-6.png`）：这固定 6 张全部 0 误报；概率在统一色标下较暗，不等于浮点概率精确为零。它们是预定 ID 示例，不能代表全部正常图；全量仍有 32/402 张出现任意 FP。

### 7.3 最差案例（`anomaly-worst-cases.png`、`normal-worst-fp.png`）

| 图 | 类别 | GT | 预测 | IoU | 事实 |
| --- | --- | ---: | ---: | ---: | --- |
| 049 | 异常 | 10 459 | **0** | 0.000 | 完全没报，GT 标出底部中央四根引脚的细长区域 |
| 057 | 异常 | 3 573 | **0** | 0.000 | 完全没报，同样细长结构 |
| 077 | 异常 | 7 762 | 351 | 0.000 | 报了 351 px，全在别处（FP），GT 弧形区域未命中 |
| 048 | 异常 | 1 631 | 161 | 0.078 | 只命中极小一部分，其余漏检 |
| 097 | 异常 | 3 045 | 692 | 0.084 | TP 289，recall **9.49 %**；预测 692 中 FP 403 |
| 069 | 异常 | 5 806 | 589 | 0.086 | 命中约十分之一 |
| 0333 | **正常** | 0 | 575 | — | 少量局部 FP，位置以红色叠加为准 |
| 0694 | 正常 | 0 | 540 | — | FP 可见于底部引脚右侧的邻近背景 |
| 0261 | 正常 | 0 | 379 | — | FP 可见于上方板缘外侧背景 |
| 0230 | 正常 | 0 | 367 | — | 上方引脚附近的局部 FP |
| 0669 | 正常 | 0 | 229 | — | 板缘/引脚连接附近的局部 FP |

全量计数（test）：**异常 TP=0 共 3 张（049/057/077）**，其中**预测完全为空 2 张（049/057）**——这两个口径不同，本阶段分开记；**正常有任意 FP 共 32/402 张**。

### 7.4 失败分析（观察—证据—可能原因—下一项验证）

**A. 入选引脚细长区域漏检（049、057；val 的 042/051 也有漏检）**

- **观察**：049/057 的 GT 标在底部中央引脚的若干细长区域，模型预测 0 像素；统一色标下概率较暗。尚未统计 GT 内概率分位数，不能仅凭黑色面板判断距阈值多远。
- **证据**：`failure-cases.csv`中049 GT10459像素、预测0；057也预测为空。val的042 GT15523像素、预测47像素但TP=0；051 GT4335像素、预测0。两张的IoU都为0，预测是否为空的口径不同。
- **可能原因（假设）**：512 letterbox 与主干下采样可能削弱细结构信息，模型也可能将引脚相关外观学习为背景；尚未测量缩放后 GT 宽度，也未比较类别内样本，因此不能把分辨率或正常引脚外观当作已证实原因。
- **下一项验证**：把这些 ID 的 512 输入与 GT 叠出来看是否已不可分辨；再用**同一 best** 在 768/1024 或切块推理下对照（只推理，不训练），看漏检是否随分辨率下降。本轮不实施。

**B. 正常图局部误报（全量 32/402；图示为 FP 最多 5 张）**

- **观察**：`normal-worst-fp.png` 的 5 张中出现少量局部点/片状 FP；部分接近引脚/板缘，0694、0261 也包含邻近背景位置。它们是最多 FP 的入选示例，不能据此断言全部 32 张的误报位置或纹理原因。
- **证据**：0333 575 px、0694 540 px、0261 379 px、0230 367 px、0669 229 px；全部 402 张正常图合计 3 689 px（比例 6.11e-06），每图中位 0、P95 2.19e-05。
- **可能原因（假设）**：局部边缘、阴影或外观变化可能与模型的缺陷响应相关；尚未对全部 FP 做位置/亮度/梯度统计，不能归因于高光或 CE 背景占比。
- **下一项验证**：统计 FP 像素是否落在高梯度/高亮区（用同一概率图与 Sobel 幅度做交叉表）；再用"负样本加权"或"边缘感知损失"做对照实验（新配置、新 run，不改本轮结论）。

**C. 全局 IoU 与 macro 的差距（test 0.683 vs 0.525）**

- **观察**：两数相差约 0.1585；全局对全部图先累加 TP/FP/FN，异常 macro 对 40 张异常图等权。权重（全局可解释为按并集像素数加权）与纳入正常 FP 的范围均不同，提示样本表现不均。
- **证据**：test 40 张里 3 张 TP=0、**7 张 IoU<0.1**（048/049/057/069/077/085/097）；当前未按 GT 面积分层，且完全漏检的 049 有 10 459 个 GT 像素，不能从汇总差值直接归纳为小面积目标失败。
- **可能原因（待验证）**：面积、形态/宽度、外观、FP 分布等均可能与逐图差异相关，当前没有因果证据。
- **下一项验证**：按 GT 面积分层报告 IoU（`per_image.json` 已含面积，可直接分层，不需重训）；若小面积层显著低，再考虑切块/高分辨率。

**D. 边界粗糙（003/008 等）**

- **观察**：003/008 的 TP 命中主体，仍有 FP/FN；图示支持边界不完全贴合。尚未测量边界距离/带宽，不能声称统一外扩 1—3 px 或多数 FP 一定来自轮廓带。
- **证据**：003 IoU 0.828 但 FP 142 px；008 IoU 0.650、FP 428 px。
- **可能原因（假设）**：512 输入 → 128 logits → 双线性还原可能影响细边界；没有分辨率对照或边界度量，不能把边界误差视为上采样的必然结果。当前协议固定无后处理。
- **下一项验证**：比较 128 网格与 512 恢复的边界宽度；若要改善，属"更高输出分辨率"的新实验，不在本轮。

> 以上 A—D 分别列出已观察的计数/图示与未验证的原因。计数是事实，机制/面积效应仍是假设；只改汇总口径的探索也要标明已看 test。新的推理或训练对照本阶段不实施；后续任何方法调整都必须披露该 test 已被查看、不能再声称其完全未参与开发。

### 7.5 实验卡与复现材料

- `experiment-card.md`：数据/清单/快照 SHA、参数量、损失、超参、软硬件版本、50 轮与 best 39、val 与 test **分列**、正常误报、训练耗时/显存、失败与局限、独立预测与复算命令、路径。推理延迟如实写"未测"。
- `reproduction.md`：当前冻结加载器要求保存并核验**整个训练 run 的工件集合**（21 文件，154 043 135 bytes；包括权重/镜像/last、配置、processor、summary、日志、预览和 NPZ），完整评价目录、数据与解压索引也需单独保留；环境须在新前缀重建，不能搬移原 Conda 前缀。重下 `mit-b0` 不能替代 best。原始评价源码 SHA 与验收前已提交源码不同且原字节不可恢复，原评价身份与本次技术复核身份分别披露。
- `results.md`：val/test 两组指标 + 读法 + 图表索引。

### 7.6 人工看图记录

独立验收实际打开新目录 `E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42_results_acceptance_20261101\figures` 下全部 5 张 PNG，逐张检查如下；它们来自验收技术复核目录，与原始 eval_001 分开标记：

| 图 | 检查点 | 结果 |
| --- | --- | --- |
| `training-curves.png` | 2×2 含 CE/Dice/total、两条 LR、val 全局 IoU/Dice 与异常 macro IoU；best 线在 39；FP 轴显示前三轮真实 0 | **通过**；补充的 val Dice 曲线清晰可见，FP=0 已保留；末 LR 略高于 1e-6，数值以 CSV 为准 |
| `anomaly-fixed-6.png` | 6 行 × 5 列完整、概率统一 0—1 色标、TP 绿/FP 红/FN 蓝、底部图例清晰 | **通过**；引脚处 GT 与概率/二值/叠加的位置可对应，逐图 IoU/FP 标签与表一致 |
| `anomaly-worst-cases.png` | 6 个并集 ID/顺序不变；049/057 二值为空并标蓝 FN，077 有远离 GT 的 FP 点 | **通过**；展示区分 TP=0 与空预测；黑色概率图不代替概率数值统计 |
| `normal-fixed-6.png` | 6 行 × 4 列包含原图、全零 GT、概率、红色 FP；全零 GT 清晰可见，6 行 FP 均 0，IoU 为 NA | **通过**；不能由固定 6 图外推全量无 FP，暗概率图也不等于精确零概率 |
| `normal-worst-fp.png` | 5 行 × 4 列均有全零 GT；红色局部 FP 与概率亮点对应；标签 575/540/379/367/229 | **通过**；FP 也出现于邻近背景，图示不证明全量 FP 的空间分布 |

### 7.7 一致性

| 检查 | 结果 |
| --- | --- |
| 重算概率的二值与阶段 12 PNG | **23/23 XOR = 0** |
| 曲线数值来源 | 全部来自 `epochs.jsonl`，未重算指标 |
| run / best 指向 | 曲线、CSV、实验卡、results 均指向 `pcb1_b0_v2_seed42` 与 best epoch 39 |
| 选择规则 | 引用阶段 12 的 `display_rules.json`，SHA 记入 `results-summary.json` |

### 7.8 验收清单

| 条件 | 结果 |
| --- | --- |
| 不训练、不重选 best、不换指标实现 | 通过 |
| 曲线含 CE/Dice/总 loss、val IoU/Dice、macro、正常 FP、LR | 通过（`training-curves.png`、`curves.csv`） |
| 标 best 39 | 通过 |
| 异常 5 面板 + 正常原图/全零 GT/概率/误报 + 三色图例 | **通过**；5 张新图已逐张目视核验 |
| 展示 ID / 排序值 / 理由 / TP0 / 空预测 / 正常 FP 全列表 | 通过（`failure-cases.csv`、`results-summary.json`） |
| 重算概率仅用于可视化并校验二值 | 通过（23/23 XOR 0） |
| 失败分析按"观察—证据—原因—下一步" | 通过（A—D） |
| 只提候选对照、不实施 | 通过 |
| 实验卡 / results / reproduction / failure CSV | 通过 |
| 延迟未测如实标注 | 通过 |
| 图/表/卡指向同一 run 与 best | 通过 |

## 8. 遇到的问题、修复与未解决项

| # | 现象 | 原因 | 处理 |
| --- | --- | --- | --- |
| 1 | 正常图被标成 `tp_zero` | 判断只看 `tp == 0`，没判类别；正常图 TP 天然为 0 | 改为仅异常图记 `tp_zero`，重跑 summarize；`failure-cases.csv` 复验 |
| 2 | 单测里背景色期望写成 255 | `_overlay` 是 `source*0.55+255*0.45`，全黑源图得 114 | 修正断言为 114，4 项单测通过 |
| 3 | 4 张样本表各约 1.2 MB | 每行 1404×1070 的原图被画进面板 | 仍在便携目标内（本阶段 4.6 MB < 20 MB），不额外压缩以免影响目视检查 |

未解决/未验证：

- **推理延迟未测**：没有做预热 + 多次计时的稳定基准，实验卡写"未测"。
- **图像级判定规则未定义**：只有像素级统计，所以没有图像级检出率/误检率。
- **失败原因仍是假设**：A—D 没有做因果对照，也未按面积、宽度或位置分层；需要后续独立分析/实验。未计算 macro 置信区间，不能声称区间宽度已知。
- **test 已被查看**：后续任何基于这些失败模式的改动，都不能再声称"test 完全未参与开发"。

## 9. 给用户的学习要点

1. **图表是结论的一部分，不是装饰。** 同一组数字，概率图统一 0—1 色标还是按图拉伸，会让人得出完全不同的"模型有多自信"的印象。
2. **"漏检"和"误报"要用不同的面板讲。** 异常图要能看出"命中在哪、漏在哪"（TP/FP/FN 三色），正常图要能看出"假在哪"（红），两者的对照组不同。
3. **全局指标与宏平均的差值是诊断信号。** 0.683 vs 0.525 体现权重/样本范围不同和逐图表现不均；它本身不能证明大目标优于小目标。需要面积/宽度分层验证。
4. **失败分析的价值在下一步可验证。** 对 049 先量化原图/512 输入中 GT 的宽度及概率，再预先固定分辨率对照，才能区分不同假设；本轮没有执行这些对照。
5. **诚实标注未测与假设。** 延迟没测就写"未测"；原因是猜的就写"假设"。这两条决定了这份材料能不能被别人接着用。

自测问题：

1. 为什么本阶段的"Dice 项曲线"不能说明模型把背景分得很好？
   *线索*：§3.5、§7.1；要点是 Dice 只在正 GT 图上平均，全正常 batch 的 0 不进分母。
2. 049 与 077 都被记为"TP = 0"，为什么 `failure-cases.csv` 给它们的 `na_notes` 不同？
   *线索*：§7.3；要点是 049 预测为空，077 预测了 351 px 但全是 FP——两个不同的失败模式。
3. 如果要把"分辨率不足导致细结构漏检"从假设变成结论，最少需要哪一步，且为什么不能只改阈值？
   *线索*：§7.4 A；先量化原图和 512 缩放后 GT 宽度、GT 内概率分布，再做预先固定的分辨率/切块对照；单改阈值同时改变 FP/FN，不能证明分辨率机制，并且要披露 test 已被查看。

## 10. 下一阶段与修订记录

本阶段是本次实现范围的终点。以下是候选后续方向，均未在本轮执行；正式实验须预先固定配置、数据用途与选模规则，使用独立 run，并与本次基线分开报告：

- **切块/高分辨率推理对照**：验证 §7.4 A 的假设，只推理不训练，仍用同一 best；需披露 test 已参与开发。
- **采样或损失侧对照**：针对 §7.4 B 的正常图误报，做负样本加权/边缘感知损失的独立实验。
- **DINOv3 等新主干**：先固定可实现的分割头/特征接口、训练预算、冻结或微调方式，再复用同一清单、原图还原与指标实现并创建独立 run。用 val 决定配置；当前 test 已被查看，后续 test 比较须披露开发暴露。本轮只提供交接方向，不声称新主干已实现或一定更好，详细排程由统一后续指南承接。
- **部署侧**：延迟基准（预热 + 多次计时 + 中位/P95）、量化/导出，属工程任务。

| 日期 | 修改原因 | 改动与影响 | 修正后重新验证的证据 |
| --- | --- | --- | --- |
| 2026-11-01 | 初版 | 新增 `summarize.py` 与 4 项单测；生成 5 张图、`curves.csv`、`failure-cases.csv`、`results.md`、实验卡、复现说明；修正正常图 `tp_zero` 误标后重跑，23 张重算二值与阶段 12 全一致 | `reports/pcb1_b0_v2_seed42/`、`reports/stage13-unit-tests.txt`（277 项） |
| 2026-11-01 | 阶段 12—13 联合独立验收修订 | 修正 best step、epoch50 CE/Dice 与首末 LR、7 张低 IoU 计数和面积差/recall 混淆；撤销 warmup 因果、面积效应及未测位置/宽度结论；补正常全零 GT、val Dice 和保留零的 FP 轴；新增来源/输出守卫与完整 run/同绝对路径复现限制 | 新增 3 项守卫测试先红后绿，stage13 CPU **7/7**；完整回归 **291/291（11.593 s）**，val v2 门禁 **74/74**，全量 val/test 与原输出一致；23 张 XOR 全零，5 张验收版图逐张 QA **通过** |
