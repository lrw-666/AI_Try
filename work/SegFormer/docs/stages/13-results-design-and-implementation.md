# 阶段 13：结果可视化、失败分析与实验卡

| 项目 | 值 |
| --- | --- |
| 创建/更新时间 | 2026-11-01 16:00 +08:00 |
| 对应步骤 | 主指南第 15 节「步骤 13：结果可视化、失败分析与实验卡」 |
| 前置状态 | 阶段 12 已完成自检（val 门禁 29/29；test 442 张一次性评价），待独立验收 |
| 本阶段范围 | 只从既有 run 与评价目录生成曲线、样本图、最差案例、失败分析、实验卡与复现说明 |
| 明确不做 | 不训练、不 re-select best、不重算另一套指标、不改阈值/后处理、不实施切块/多尺度/DINOv3 等新实验 |
| 状态 | **已完成，自检通过，待独立验收** |
| 对应代码身份 | 新增 `segformer_lab/summarize.py`、`tests/test_summarize.py`（4 项）；run 时 git HEAD `a80a2da` + 未提交新文件 |
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
- 正常图不画"没有缺陷"的空 GT 面板，而是直接画**误报**（红），因为没有正样本可对比时，空的 GT 面板没有信息量。

### 3.4 曲线要写清"哪个 LR"

训练日志里每轮记了 `lr_first`/`lr_last`，它们是**该轮第一/最后一次更新实际使用的学习率**（`train_step` 之前取的）。而"更新之后优化器里的 LR"是另一个值（`lr_at(scheduler_completed)`）。`curves.csv` 两者都存：

- `lr_last_used`：日志实测值，画主曲线；
- `lr_after_epoch_computed`：用同一公式按 `scheduler_completed` 反算，画虚线。

这样既不用"反算值"冒充实测值，也不会让人误以为两者是同一个数。

### 3.5 Dice 曲线不能读成"分割完美"

损失里的 Dice 项是**正 GT 图上的逐图均值**。全正常 batch 的 Dice 项是精确 0（图相连的零），它不进 epoch 聚合的分母。所以曲线里 Dice 项从 0.98 降到 0.17 表示"有缺陷的图越分越准"，**不是**"背景分得很完美"。这一点写进了 `results-summary.json.notes` 与实验卡。

### 3.6 选择规则不是本阶段定的

`display_rules.json` 是阶段 12 在**解码任何 test 图之前**写的，本阶段只是执行它，并把它的 SHA 记进 `results-summary.json`。这样"挑图"这件事的时间顺序可查：规则先于结果。

## 4. 文件、接口与产物

| 文件 | 新增 | 作用 |
| --- | --- | --- |
| `segformer_lab/summarize.py` | 新增 | 曲线/图表/失败表/实验卡/复现说明生成器 |
| `tests/test_summarize.py` | 新增 4 项 | LR 镜像公式、曲线字段映射、TP/FP/FN 配色、背景淡化 |
| `reports/pcb1_b0_v2_seed42/curves.csv` | 新增 | 50 行逐轮数值（12 KB） |
| `reports/pcb1_b0_v2_seed42/failure-cases.csv` | 新增 | 23 行入选图、理由、计数、源 SHA、重算 XOR |
| `reports/pcb1_b0_v2_seed42/results.md` | 新增 | val/test 指标、读法、图表索引 |
| `reports/pcb1_b0_v2_seed42/experiment-card.md` | 新增 | 实验卡（按主指南模板填真实值） |
| `reports/pcb1_b0_v2_seed42/reproduction.md` | 新增 | 换机复现步骤与必须单独搬运的内容 |
| `reports/pcb1_b0_v2_seed42/results-summary.json` | 新增 | 机器可读摘要（7 KB） |
| `reports/pcb1_b0_v2_seed42/figures/*.png` | 新增 5 张 | 训练曲线 + 4 张样本/最差案例表 |

图表清单：

| 图 | 内容 | 尺寸/体积 |
| --- | --- | --- |
| `training-curves.png` | 2×2：loss（total/CE/Dice）、LR（实测 + 反算）、val IoU 与 macro、正常 FP（log）；标 best 39 | 129 KB |
| `anomaly-fixed-6.png` | 6 行 × 5 列：原图 / GT / 概率 / 二值 / TP-FP-FN | 1.2 MB |
| `anomaly-worst-cases.png` | IoU 最低 ∪ recall 最低（6 张去重），同 5 列 | 1.2 MB |
| `normal-fixed-6.png` | 6 行 × 3 列：原图 / 概率 / 误报（红） | 1.1 MB |
| `normal-worst-fp.png` | 正常图 FP 最多 5 张，同 3 列 | 0.9 MB |

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
| 1 | `--help` | 4 个参数，退出 0 | 终端 |
| 2 | 首轮 summarize | 4 图 + 3 文档生成，23 张重算 XOR 全 0 | `results-summary.json` |
| 3 | 发现并修正 `tp_zero` 误标（正常图被写 `tp_zero`） | 重跑一次，正常图只保留 `normal_false_positive` | `failure-cases.csv` |
| 4 | 单测（新增 4 项） | 4/4 | `tests/test_summarize.py` |
| 5 | 完整回归 | **277 项，14.142 s，OK**（273 → 277） | `reports/stage13-unit-tests.txt` |
| 6 | 人工看图 | 曲线 1 张 + 样本/最差图 4 张，逐张检查色标、图例、尺寸、覆盖位置 | 见 7.6 |

## 7. 结果与分析

### 7.1 训练曲线（`figures/training-curves.png`，50 轮）

| 项目 | 起点 | 终点 | 观察 |
| --- | --- | --- | --- |
| 总损失（加权） | 0.8030（ep1） | 0.0773（ep50） | 前 4 轮急降（0.803→0.473），之后缓慢下降 |
| CE | 0.3113 | 0.0123 附近 | 背景占 99.8 %，低 CE 不等于分割好 |
| Dice 项 | 0.9836 | 0.1710 附近 | 只在 48 张正 GT 图上平均 |
| val 全局 IoU | 0.0（ep1—3） | 0.7775（ep39）/ 0.7773（ep50） | 第 4 轮跳出全背景解 |
| val 异常 macro IoU | 0.0 | 0.5396（ep39） | 平台期 0.49—0.55，抖动明显 |
| 正常图 FP | 0（ep1—3） | 1 000—3 000 px 区间 | 第 4 轮起非零，始终极小 |
| LR（实测末步） | 9.06e-8 | 1.000e-6 | warmup 662 步后 cosine 到 floor |

曲线的三个坑都标在图/CSV 里：前 3 轮 IoU 为 0 是 warmup 太长的必然结果；正常 FP 前 3 轮的 0 是"什么都没报"；Dice 项的 0 是"该 batch 无正 GT"，不是"分得完美"。

### 7.2 固定样本（按 image_id 升序前 6）

异常（`anomaly-fixed-6.png`）：

| 图 | GT 像素 | 预测像素 | IoU | 观察 |
| --- | ---: | ---: | ---: | --- |
| 000 | 1 524 | 534 | 0.298 | 细划痕，概率峰值刚过 0.5，预测面积只有 GT 的三分之一 |
| 001 | 1 687 | 986 | 0.497 | 命中主体，两端漏 |
| 002 | 1 702 | 1 776 | 0.618 | 面积接近但形状不完全重合，边缘外扩 |
| 003 | 1 618 | 1 599 | 0.828 | 面积几乎相同、边界贴合，是 6 张里最好的一张 |
| 008 | 3 739 | 3 136 | 0.650 | 命中主体，欠报约 600 px |
| 012 | 3 712 | 2 499 | 0.394 | 只覆盖约三分之二，剩下三分之一漏检 |

正常（`normal-fixed-6.png`）：6 张全部 0 误报（概率图纯黑），说明模型对"标准板"确实保守。

### 7.3 最差案例（`anomaly-worst-cases.png`、`normal-worst-fp.png`）

| 图 | 类别 | GT | 预测 | IoU | 事实 |
| --- | --- | ---: | ---: | ---: | --- |
| 049 | 异常 | 10 459 | **0** | 0.000 | 完全没报，GT 是左侧三根细金属柱 |
| 057 | 异常 | 3 573 | **0** | 0.000 | 完全没报，同样细长结构 |
| 077 | 异常 | 7 762 | 351 | 0.000 | 报了 351 px，全在别处（FP），GT 弧形区域未命中 |
| 048 | 异常 | 1 631 | 161 | 0.078 | 只命中极小一部分，其余漏检 |
| 097 | 异常 | 3 045 | 692 | 0.084 | 命中约五分之一 |
| 069 | 异常 | 5 806 | 589 | 0.086 | 命中约十分之一 |
| 0333 | **正常** | 0 | 575 | — | 误报集中在连接器引脚端与板边 |
| 0694 | 正常 | 0 | 540 | — | 同上，引脚/边缘高光 |
| 0261 | 正常 | 0 | 379 | — | 板边与丝印附近 |
| 0230 | 正常 | 0 | 367 | — | 上方金属针脚 |
| 0669 | 正常 | 0 | 229 | — | 边缘 |

全量计数（test）：**异常 TP=0 共 3 张（049/057/077）**，其中**预测完全为空 2 张（049/057）**——这两个口径不同，本阶段分开记；**正常有任意 FP 共 32/402 张**。

### 7.4 失败分析（观察—证据—可能原因—下一项验证）

**A. 细长金属结构系统性漏检（049、057；val 的 042/051 同类）**

- **观察**：GT 是左侧 3—4 根细金属柱/划痕，模型预测 0 像素；概率图几乎全黑，不是"差一点过阈值"。
- **证据**：`failure-cases.csv` 中 049 GT 10 459 px / 预测 0；057 同；val 的 042（15 523 px）与 051（4 335 px）在阶段 11/12 均为 0。
- **可能原因（假设）**：缺陷宽度在 512 letterbox 后小于 2 px，主干 stride 4 的下采样后几乎不留响应；同时这类"金属柱+高光"在正常图中也存在（见 B），模型更倾向把它当背景。
- **下一项验证**：把这些 ID 的 512 输入与 GT 叠出来看是否已不可分辨；再用**同一 best** 在 768/1024 或切块推理下对照（只推理，不训练），看漏检是否随分辨率下降。本轮不实施。

**B. 正常图误报集中在引脚端与板边（32/402）**

- **观察**：`normal-worst-fp.png` 里红色区域集中在连接器引脚尖端、板边与丝印附近，是**成小片**的结构，不是随机噪点。
- **证据**：0333 575 px、0694 540 px、0261 379 px、0230 367 px、0669 229 px；全部 402 张正常图合计 3 689 px（比例 6.11e-06），每图中位 0、P95 2.19e-05。
- **可能原因（假设）**：引脚/边缘的高梯度与缺陷纹理相似；训练集中"正常引脚"样本虽多，但损失里 CE 由背景主导，模型对这类小面积结构没有强约束。
- **下一项验证**：统计 FP 像素是否落在高梯度/高亮区（用同一概率图与 Sobel 幅度做交叉表）；再用"负样本加权"或"边缘感知损失"做对照实验（新配置、新 run，不改本轮结论）。

**C. 全局 IoU 与 macro 的差距（test 0.683 vs 0.525）**

- **观察**：全局高、macro 低 0.16，说明大面积缺陷分得好、小缺陷分得差。
- **证据**：test 40 张里 3 张 TP=0、5 张 IoU<0.1；而少数上千像素的缺陷贡献了大部分 TP。
- **可能原因**：面积加权 + 交叉熵被背景主导 + 小目标在 512 下信息损失。
- **下一项验证**：按 GT 面积分层报告 IoU（`per_image.json` 已含面积，可直接分层，不需重训）；若小面积层显著低，再考虑切块/高分辨率。

**D. 边界粗糙（003/008 等）**

- **观察**：命中主体但边界外扩 1—3 px，FP 多来自轮廓带。
- **证据**：003 IoU 0.828 但 FP 142 px；008 IoU 0.650、FP 428 px。
- **可能原因（假设）**：512 输入 → 128 logits → 双线性上采样的固有模糊；无 CRF/后处理（协议也不允许）。
- **下一项验证**：比较 128 网格与 512 恢复的边界宽度；若要改善，属"更高输出分辨率"的新实验，不在本轮。

> 以上 A—D 都是**假设 + 下一步**，不是结论。验证它们需要新的推理或训练配置，本阶段不实施；而且 test 已经被查看，后续任何方法调整都必须披露"该 test 已参与开发"。

### 7.5 实验卡与复现材料

- `experiment-card.md`：数据/清单/快照 SHA、参数量、损失、超参、软硬件版本、50 轮与 best 39、val 与 test **分列**、正常误报、训练耗时/显存、失败与局限、独立预测与复算命令、路径。推理延迟如实写"未测"。
- `reproduction.md`：Git 里没有权重/数据/环境；换机必须单独搬 `best.pt`、`model_config.json`、`processor/` 并核 SHA；重下 `mit-b0` **不能**替代 best。
- `results.md`：val/test 两组指标 + 读法 + 图表索引。

### 7.6 人工看图记录

本阶段实际打开了 5 张图逐张检查：

| 图 | 检查点 | 结果 |
| --- | --- | --- |
| `training-curves.png` | 四个面板都有数据、best 线在 39、LR 用对数轴且末值 1e-6、图例可读 | 通过；前 3 轮 val IoU 为 0 与 §7.1 解释一致 |
| `anomaly-fixed-6.png` | 6 行 × 5 列完整、概率统一色标、三色可见、图例在底部 | 通过；GT 面板的细划痕与概率图亮点位置一致 |
| `anomaly-worst-cases.png` | 049/057 的 GT 是细金属柱、概率近全黑、二值全黑、叠加为蓝（FN） | 通过；与 CSV 的"预测为空"一致 |
| `normal-fixed-6.png` | 6 张正常图概率纯黑、无误报 | 通过 |
| `normal-worst-fp.png` | 红色误报集中在引脚端/板边，成小片而非随机噪点 | 通过；支撑假设 B |

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
| 异常 5 面板 + 正常误报面板 + 三色图例 | 通过 |
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
- **失败原因仍是假设**：A—D 都没有做对照实验验证；需要新的推理（高分辨率/切块）或新的训练配置，属后续任务。
- **test 已被查看**：后续任何基于这些失败模式的改动，都不能再声称"test 完全未参与开发"。

## 9. 给用户的学习要点

1. **图表是结论的一部分，不是装饰。** 同一组数字，概率图统一 0—1 色标还是按图拉伸，会让人得出完全不同的"模型有多自信"的印象。
2. **"漏检"和"误报"要用不同的面板讲。** 异常图要能看出"命中在哪、漏在哪"（TP/FP/FN 三色），正常图要能看出"假在哪"（红），两者的对照组不同。
3. **全局指标与宏平均的差值是诊断信号。** 0.683 vs 0.525 告诉你"大目标可以、小目标不行"，比单看任何一个数都有用。
4. **失败分析的价值在"下一步可验证"。** 写"小目标难"没用；写"把 049 的 512 输入与 GT 叠出来看是否已不可分辨，再用同一 best 在 768 下推理对照"才是可执行的一步。
5. **诚实标注未测与假设。** 延迟没测就写"未测"；原因是猜的就写"假设"。这两条决定了这份材料能不能被别人接着用。

自测问题：

1. 为什么本阶段的"Dice 项曲线"不能说明模型把背景分得很好？
   *线索*：§3.5、§7.1；要点是 Dice 只在正 GT 图上平均，全正常 batch 的 0 不进分母。
2. 049 与 077 都被记为"TP = 0"，为什么 `failure-cases.csv` 给它们的 `na_notes` 不同？
   *线索*：§7.3；要点是 049 预测为空，077 预测了 351 px 但全是 FP——两个不同的失败模式。
3. 如果要把"分辨率不足导致细结构漏检"从假设变成结论，最少需要哪一步，且为什么不能只改阈值？
   *线索*：§7.4 A；要点是阈值不改变"概率图几乎没有响应"这一事实，需要高分辨率/切块推理的对照，并且要披露 test 已被查看。

## 10. 下一阶段与修订记录

本阶段是当前任务的终点。后续可能的方向（**均需新的授权与新配置/新 run**）：

- **切块/高分辨率推理对照**：验证 §7.4 A 的假设，只推理不训练，仍用同一 best；需披露 test 已参与开发。
- **采样或损失侧对照**：针对 §7.4 B 的正常图误报，做负样本加权/边缘感知损失的独立实验。
- **DINOv3 等新主干**：必须复用同一清单、同一还原与评价代码，并重建独立 run；不能与本基线混报。
- **部署侧**：延迟基准（预热 + 多次计时 + 中位/P95）、量化/导出，属工程任务。

| 日期 | 修改原因 | 改动与影响 | 修正后重新验证的证据 |
| --- | --- | --- | --- |
| 2026-11-01 | 初版 | 新增 `summarize.py` 与 4 项单测；生成 5 张图、`curves.csv`、`failure-cases.csv`、`results.md`、实验卡、复现说明；修正正常图 `tp_zero` 误标后重跑，23 张重算二值与阶段 12 全一致 | `reports/pcb1_b0_v2_seed42/`、`reports/stage13-unit-tests.txt`（277 项） |
