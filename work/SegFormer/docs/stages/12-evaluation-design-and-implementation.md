# 阶段 12：独立推理、原图还原与最终 test 评价

| 项目 | 值 |
| --- | --- |
| 创建/更新时间 | 2026-11-01 15:40 +08:00 |
| 对应步骤 | 主指南第 14 节「步骤 12：独立推理、原图还原与最终测试」 |
| 前置状态 | 阶段 11 已通过独立验收（263 项回归；best epoch 39 / step 10 330） |
| 本阶段范围 | 新进程从保存结构 + best 严格重建；完整 132 张 val 对齐门禁；冻结协议下 **442 张 test 首次评价及必要技术复核**；无 GT 单图预测 |
| 明确不做 | 不训练、不 resume、不调阈值/轮次/后处理、不使用 TTA、不据 test 选参、不进入阶段 13 之外的实验 |
| 状态 | **已通过2026-11-01独立验收；历史首次执行与验收复核分别留档** |
| 对应代码身份 | 新增 `inference.py`（共享通道）、`evaluate.py`、`predict.py`、`tests/test_inference_eval.py`；run 时 git HEAD `6682014` + 4 个未跟踪新文件 |
| 输出目录 | `runs\pcb1_b0_v2_seed42_eval_001\{val,test,single_val_045_v2}`（仓库外） |

## 1. 本阶段解决什么问题

独立推理要检验保存物能否重建相同模型、预处理与评价协议。阶段11已做过新进程复验，本阶段进一步提供专用evaluate/predict入口与test门禁。应区分输入或代码身份错误造成的偏差和已声明容差内的浮点差异，不能把任何非零差异都解释为模型无效。

所以本阶段要按顺序证明三件事：

1. **可重建**：只用 run 内 `model_config.json`、`best.pt` 的模型张量、`processor/`，在新进程里 `strict=True` 重建，missing/unexpected 必须为 0；不许用预训练快照重新搭一个"看起来一样"的模型。
2. **可复算**：完整 132 张 val 的逐图计数、尺寸、面积、TP/FP/FN/TN、各指标与 NA 结构，必须与训练时保存的 epoch 39 记录逐项一致（整数精确、浮点 1e-9）。**不一致就停下，不许进 test**。
3. **可冻结**：test 442张在协议冻结后评价，结果与val分开报告，不依据test调参。必要技术复核可使用同一冻结协议重跑，保留各次结果和原因，不按成绩择优。

## 2. 技术栈与版本

| 组件 | 版本 | 本阶段作用 | 状态 |
| --- | --- | --- | --- |
| Python | 3.12.16 | `segformer_resources\env`（项目 runner 内） | 已运行 |
| PyTorch | 2.10.0+cu126 | 重建模型、B1 前向、`inference_mode`、RNG 无关的只读推理 | 已运行 |
| transformers | 4.57.6 | `SegformerForSemanticSegmentation`、`SegformerImageProcessor`（`local_files_only`） | 已运行 |
| NumPy / Pillow | 2.5.4 / 12.3.0 | 概率数组、0/1 PNG 写盘 | 已运行 |
| matplotlib | 3.11.2 | 单图热力图与叠加图 | 已运行 |
| GPU | RTX 4070 Laptop 8 187.5 MiB | B1 前向 | 已运行 |
| 新增依赖 | **无** | 未安装、未升级任何包 | — |

## 3. 设计与取舍

### 3.1 一个共享推理通道，三个入口

`segformer_lab/inference.py` 是唯一实现"从图到概率"的地方：

```text
prepare_image / dataset[index]
  -> processor（do_resize=False, do_reduce_labels=False）
  -> model(pixel_values)  B1、eval、inference_mode、IEEE FP32
  -> restore_defect_probability（512 双线性 → 裁 padding → 原 H/W → softmax 通道 1）
  -> >= 0.5 二值
```

`evaluate.py`、`predict.py` 都调它；阶段 13 复算展示图也调它。指标累计仍用阶段 07 验收过的 `OriginalSpaceEvaluator`（`int64` 累加、只除一次），没有第二套公式。

### 3.2 为什么 val 是"门禁"而不是"顺便看看"

`gate.json` 在 test 之前产生，包含 29 项检查：图像计数、混淆量、全局四指标、NA 结构、132 行逐图字段（整数精确、浮点 1e-9）、异常 macro 的 mean 与 `defined_images`、正常图 FP 像素/比例/有 FP 图数，以及 6 张固定预览的概率最大差与 XOR。

test 入口**强制**要求 `--val-gate <gate.json>`，并核对 gate 里的 `best_sha256`、`config_sha256`、三份清单 SHA、`split_meta`、`model_config` 与当前 run 完全一致；任何一项不符直接退出。这样"用不同 checkpoint 的 gate 去放行 test"在代码层面不成立。

### 3.3 冻结test协议与展示规则，保留技术复核记录

`display_rules.json` 在任何 test 图像被解码之前写入并计算 SHA（写进 `evaluation_meta.json`），规则是：test 清单按 `image_id` 升序取前 6 正常 / 前 6 异常；异常 IoU 最低 5、recall 最低 5；正常 FP 像素最多 5；并列按 `image_id`。评价结束后才按这套规则算 `selection.json`，每张入选图都记录自己的入选理由。

这样做是为了防止"看完结果再挑图"：规则在前、挑选在后，两者的时间顺序和哈希都留了痕。

### 3.4 流式：442 张不能全留在内存

本机运行中可用内存只有 2.5 GiB 量级，而一张原尺寸 `float32` 概率是 1404×1070×4 ≈ 6.0 MB，442 张全留就是 2.6 GB。所以：

- 逐张前向 → 立即 `>=0.5` 二值化 → 写 0/1 PNG → 释放概率；
- 只把 `int64` 混淆量和逐图标量留下（`OriginalSpaceEvaluator`）；
- 全量浮点概率**默认不保存**（`predict` 单图可存，因为它只有一张）；
- 需要保留概率的只有 6 张固定预览（val 阶段，用于与 run 内 `pre_save_probabilities.npz` 比对）。

### 3.5 0/1 与 0/255 分开

计分约定是 **0 背景 / 1 缺陷**，`predictions/` 目录写的就是 0/1 的 `uint8` PNG（`save_binary_mask` 会校验取值只能为 0/1）。便于人眼查看的 0/255 版本只在 `predictions_display/`（`--display-masks`）出现，且明确标注"仅为显示约定，读取计分时非零转 1"。

## 4. 文件、接口与数据约定

| 文件/模块 | 新增 | 职责 | 关键接口 |
| --- | --- | --- | --- |
| `segformer_lab/inference.py` | 新增 | 冻结 run 加载、单图前向、流式评价、0/1 PNG、无 GT 预处理 | `load_frozen_run`、`stream_split`、`probability_map`、`prepare_image` |
| `segformer_lab/evaluate.py` | 新增 | val 门禁 + test 全量评价 + 元数据/产物清单 | `--config --run --split --out [--val-gate] [--display-masks]` |
| `segformer_lab/predict.py` | 新增 | 无 GT 单图推理（二值/显示/概率/热力图/叠加） | `--config --run --image --out` |
| `tests/test_inference_eval.py` | 新增 10 项 | 选择规则、门禁比较、掩码写盘、预处理等价、run 缺失拒绝 | `python -m unittest tests.test_inference_eval` |
| `reports/stage12-summary.json` | 新增 15 KB | val 门禁 + test 指标 + 一致性 + 身份 + 局限 | 进 Git |
| `reports/stage12_evidence/` | 新增 56 KB | val/test 逐图 CSV（gzip）、门禁、指标、选择、一致性 | 进 Git |

输出目录（`runs\pcb1_b0_v2_seed42_eval_001`）：

```text
val/   metrics.json  per_image.json  per_image.csv  gate.json  selection.json
       display_rules.json  evaluation_meta.json  artifact_manifest.json
       predictions/pcb1/Data/Images/{Normal,Anomaly}/*.png   (132 张 0/1)
test/  同结构（442 张 0/1 PNG，metrics 为最终 test 成绩）
single_val_045_v2/  mask.png  mask_display.png  probability.npz  heatmap.png  overlay.png
```

数据约定：val 132 = 120 正常 + 12 异常，共 198 300 960 像素；test 442 = 402 正常 + 40 异常，共 664 007 760 像素；原图 1404×1070；GT 用原尺寸二值掩码（正常图为全零），**不把 512 标签放大回去**，也不把训练 padding 的 255 当成原图 ignore。

## 5. 关键实现讲解

### 5.1 重建不是"重新搭一个模型"

```python
payload = torch.load(best_path, map_location="cpu", weights_only=False)
if payload.get("kind") != "best": ...
model = SegformerForSemanticSegmentation(SegformerConfig.from_dict(payload["model_config"]))
missing, unexpected = model.load_state_dict(payload["model"], strict=True)
```

同时核对：run 内 `model_config.json` 与 checkpoint 里的 `model_config` 一致、`model_state.pt` 与 best 逐张量相等、`processor/` 四个开关（`do_resize`/`do_reduce_labels`/`do_rescale`/`do_normalize`）符合预期、调用方传入的 config SHA 与 run 记录一致、三份清单与 `split_meta` 未被替换。任一不符抛 `FrozenRunError`，不会"先跑起来再说"。

### 5.2 精度上下文必须包住单图推理

最初的 `predict.py` 直接调 `probability_map`，没有套 `evaluation_precision`。本机 `cudnn.allow_tf32` 默认是 `True`，于是单图推理走的是 TF32，而评价走的是 IEEE FP32：

- 045 的预测像素 **127 755**（predict）vs **127 757**（val），XOR **4**；
- 补上 `with evaluation_precision(config, device=device):` 后，predict **127 757**，XOR **0**，概率最大差 **0.0**。

修复方式不是放宽容差，而是让 `probability_map(..., config)` **强制**要求传 config（少传就报错），从接口上避免"另走一条精度路径"。

### 5.3 门禁比较：整数精确、浮点 1e-9、NA 结构精确

```python
exact("confusion.tp", fresh_tp, saved_tp)                 # int
close("global.foreground_iou", fresh, saved, tol=1e-9)    # float
exact("global.na_reasons", fresh_na, saved_na)            # 结构
# 132 行逐图：7 个整数域、5 个浮点域、4 个 NA 文本域
```

浮点容差 1e-9 是**先声明再比较**的（写进 `gate.json.tolerance`），不是看到差异后回调的。本次 29 项全部为 0 差异。

### 5.4 无 GT 的路径要和有 GT 的路径逐位一致

`prepare_image` 复制了 dataset 的预处理（letterbox → processor → 同一归一化），并由 `PreprocessEquivalenceTests` 断言它与 `PCB1Dataset[0]` 的 `pixel_values` **`torch.equal`**、几何字段逐个相等。这样"没有 mask 所以走了另一条预处理"这种错误在测试层就被挡住。

## 6. 如何操作与实际执行了什么

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
. '.\scripts\enter_project.ps1'
$ErrorActionPreference = 'Continue'
$run  = 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42'
$eval = 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42_eval_001'

Invoke-SegPython -m segformer_lab.evaluate --help
Invoke-SegPython -m segformer_lab.predict  --help

# 1) 训练 run 文件清单（before），用于事后证明未被改动
Invoke-SegPython 'E:\桌面\AI\tmp\stage12_inventory.py'

# 2) 完整 132 张 val：复算 + 门禁 + 6 张预览概率比对
Invoke-SegPython -u -m segformer_lab.evaluate --config 'configs\baseline.yaml' --run $run `
  --split val --out "$eval\val"

# 3) 仅在 gate.json 明确 passed 后：442 张 test（一次性）
Invoke-SegPython -u -m segformer_lab.evaluate --config 'configs\baseline.yaml' --run $run `
  --split test --val-gate "$eval\val\gate.json" --out "$eval\test"

# 4) 无 GT 单图预测（val 里的 045，不是 test 成绩）
Invoke-SegPython -u -m segformer_lab.predict --config 'configs\baseline.yaml' --run $run `
  --image '...\data\visa\pcb1\Data\Images\Anomaly\045.JPG' --out "$eval\single_val_045_v2"

# 5) 完整回归
Invoke-SegPython -m unittest discover -s tests -t . -v
```

| 次序 | 操作 | 实际结果 | 证据 |
| --- | --- | --- | --- |
| 1 | `--help`（两个入口） | 参数与预期一致，退出 0 | 终端 |
| 2 | 单测（新增 10 项） | 10/10 | `tests/test_inference_eval.py` |
| 3 | 完整回归 | **273 项，14.709 s，OK**（263 → 273） | `reports/stage12-unit-tests.txt` |
| 4 | val 评价 | 132 张，门禁 **29/29 通过**，19.839 s | `val/gate.json` |
| 5 | test 评价 | 442 张，60.894 s，退出 0 | `test/metrics.json` |
| 6 | 单图预测 | 127 757 px，与 val 的 045 完全一致 | `single_val_045_v2/` |
| 7 | 一致性核对 | 预处理位级相等、概率差 0.0、XOR 0、run 21 文件未变 | `reports/stage12_evidence/consistency.json` |

**计划与已执行的差异**：阶段 12 交接给出的示例命令（`--checkpoint`、`--split val --out ...\val_independent`）是"待实现接口"，实际实现为 `--run`（强制从 run 目录取 best）并新增 `--val-gate` 与 `--display-masks`；文档按真实 `--help` 记录，不照抄示例。

## 7. 首次执行与自检记录（最新独立验收见第10.1节）

### 7.1 冻结身份

| 项目 | 实测 |
| --- | --- |
| best | `checkpoints/best.pt`，epoch **39**，`global_step` **10 330** |
| best SHA-256 | `192c827bef00607f4258e5fadf0f1d97970ca5cc727a0add749be0fba5997485` |
| 严格加载 | missing 0 / unexpected 0；`model_state.pt` 与 best 逐张量相等 |
| 结构 / processor | `model_config.json` 与 checkpoint 内结构一致；processor 四开关符合预期 |
| 协议 | config SHA 与 run 记录一致；三份清单 + `split_meta` + 快照 `80983a41…` 未变 |
| 只读性 | 无 backward、无 optimizer.step；模型参数 `requires_grad=False` |

### 7.2 val 门禁（132 张，B1 / IEEE FP32 / 阈值 0.5）

| 项目 | 本次复算 | 训练时 epoch 39 | 差异 |
| --- | --- | --- | --- |
| TP / FP / FN / TN | 279 246 / 17 894 / 62 003 / 197 941 817 | 同 | 0 |
| 全局 IoU / Dice | 0.7775342969 / 0.8748459012 | 同 | 0 |
| Precision / Recall | 0.9397792286 / 0.8183056947 | 同 | 0 |
| 异常 macro IoU（12 张） | 0.5395555278 | 同 | 0 |
| 正常图 FP | 1 495 px（120 张中 11 张有 FP） | 同 | 0 |
| 6 张预览概率 | max\|Δ\| = **0.0**，XOR **0** / 9 013 680 px | — | 0 |

`gate.json` 共 **29 项**检查，`failed = []`。其中逐图部分覆盖 132 行的 7 个整数域、5 个浮点域、4 个 NA 文本域。

### 7.3 test 全量（442 = 402 + 40，协议与 val 完全相同）

| 指标 | 值 |
| --- | --- |
| 像素总数 | 664 007 760（异常 40 张 / 正常 402 张） |
| TP / FP / FN / TN | 252 227 / 27 779 / 89 101 / 663 638 653 |
| 全局前景 IoU | **0.6833438542** |
| 全局 Dice / P / R | 0.8118886139 / 0.9007914116 / 0.7389578353 |
| 异常 40 张 macro IoU | **0.5248580728**（40/40 有定义） |
| macro Dice / P / R | 0.6350735444 / 0.8070891646（38/40 有定义，2 张因"无任何预测像素"记 NA）/ 0.5831748857 |
| 正常 402 张 FP | **3 689 px**，占正常像素 **6.108459751459705e-06**，**32/402** 张出现任意 FP |
| 每图预测前景比（正常） | 中位 0，P95 2.19e-05，最大 3.83e-04 |
| NA 原因 | 全局无 NA；macro precision 2 张 NA |

与 val 的对照要在**同一列**里读：val IoU 0.7775 / macro 0.5396，test IoU 0.6833 / macro 0.5249。test 更低是正常的（val 参与选模，test 没有），但**不能**把两个数混着报，也不能用 test 的 0.6833 去替换阶段 11 的 val 0.7775。

### 7.4 覆盖与产物

| 检查 | 结果 |
| --- | --- |
| 行数 / 缺失 / 重复 | **442 / 0 / 0**，顺序与清单完全一致 |
| 二值 PNG | 442 张，全部 **1070×1404**（高×宽），取值集合 **{0, 1}** |
| 产物 | val 139 文件 / 382 KB；test 449 文件 / 1.1 MB（含 442 PNG） |
| 浮点概率 | 全量**未保存**；只有 6 张 val 预览在内存中比对过 |

### 7.5 单图一致性（045，val 图，不是 test 成绩）

| 检查 | 结果 |
| --- | --- |
| 预处理（`prepare_image` vs `PCB1Dataset`） | `pixel_values` **位级相等**，几何字段逐个相等 |
| 同进程概率差 | **0.0** |
| predict 概率 vs 评价路径 | **0.0** |
| 二值掩码 | predict 127 757 px = val 127 757 px，**XOR 0** |

### 7.6 资源与耗时

| 项目 | val（132） | test（442） |
| --- | ---: | ---: |
| 墙钟 | 19.84 s | **60.89 s** |
| 其中 PNG 写盘 | 1.83 s | 6.13 s |
| 平均每张 | 0.150 s | 0.138 s |
| 推理峰值显存 | 未单独测量；不能沿用训练阶段的峰值 | 未单独测量 |

### 7.7 原训练 run 未被改动

评价前后各做一次全目录 `bytes/SHA-256` 清单：**21 个文件、0 改动、0 新增、0 删除**。评价产物全部写在新目录 `..._eval_001`。

### 7.8 验收清单

| 验收条件 | 结果 | 证据 |
| --- | --- | --- |
| 新进程从 run 重建，strict 0/0 | 通过 | `evaluation_meta.json.identity` |
| 使用 run 内 processor、非预训练目录 | 通过（`local_files_only`，四开关校验） | 同上 |
| 完整 132 张 val 逐项对齐 | 通过（29 项，0 差异） | `val/gate.json` |
| test 以 gate 为前置、身份匹配 | 通过（代码强制） | 同上 + `test/evaluation_meta.json` |
| 442 张全量、无缺失/重复 | 通过 | `test/metrics.json.coverage` |
| 原尺寸 0/1 PNG | 通过（1070×1404，{0,1}） | `stage12_evidence/consistency.json` |
| 逐图 JSON/CSV、macro、正常 FP、NA | 通过 | `test/per_image.{json,csv}` |
| 无 GT 单图预测 | 通过（127 757 px，与 val 一致） | `single_val_045_v2/` |
| 不据test选参，固定协议评价并保留必要技术复核 | 首次评价见eval001；验收复核另见联合验收报告 | 本文修订记录 |
| 原 run 只读 | 通过（21 文件未变） | 7.7 |

## 8. 遇到的问题、修复与未解决项

### 8.1 单图推理漏掉精度上下文（已修）

**现象**：predict 输出 127 755 px，val 同一图 127 757 px，XOR 4。

**定位**：`probability_map` 没有套 `evaluation_precision`，而本机 `cudnn.allow_tf32=True`，单图走 TF32、评价走 IEEE FP32，阈值边界上 4 个像素翻转。

**修复**：`probability_map(model, item, device, config)` 强制要求 config 并在内部开上下文；`predict.py` 传入。

**复验**：predict 127 757 px、概率差 **0.0**、XOR **0**（`consistency.json`）。没有放宽容差。

### 8.2 环境层面的两个坑（不是本阶段代码问题）

1. **scipy 从用户 site-packages 被解析**：直接调 `env\python.exe` 时，`transformers` 为了目标检测损失去 import `scipy`，命中 `C:\Users\...\AppData\Roaming\Python\312\site-packages` 里的旧 scipy，与 numpy 2.5.4 冲突（`cannot import name 'Inf'`）。走项目 runner（`conda run --prefix env`）时不启用 user site，问题消失。**结论：所有需要 transformers 的调用必须走项目 runner。**
2. **`Invoke-SegPython` 抛异常会吞掉 Python traceback**：排查时改用 `& $SegConda run ... > out 2> err` 直接取原始 stderr。

### 8.3 未解决/未验证

- **推理延迟未测**：只记录了评价墙钟；没有做"预热 20 次 + 计时 100 次"的稳定延迟基准，因此本阶段不报 GPU 前向延迟。阶段 13/后续如需，按固定 B1、CUDA 同步、中位/P95 另测。
- **图像级判定规则未定义**：正常图只报像素级误报（FP 像素、比例、有 FP 的图数），没有"图像判为异常"的阈值规则，因此没有检出率/误检率。
- **样本量限制**：40张异常test图，未计算macro置信区间或验证小样本稳定性；全局IoU面积加权。val的042/051/081与test的049/057/077均存在漏检，但不据此认定机制相同，也不调本次阈值。
- **单 seed、单数据集**：不宣称跨类别、跨产线或工业泛化。

## 9. 给用户的学习要点

1. **"同一个模型"要用身份证明，不能用目录名证明。** best SHA、结构 SHA、processor SHA、config SHA、清单 SHA 全部对上，才叫"同一个模型"；`strict=True` 只保证张量名对得上，不保证你加载的是那个 epoch。
2. **val 门禁的价值在于"先失败"。** 如果新进程复算和训练时不一致，最可能的三个原因是：eval 模式、归一化常数、还原顺序。这三件事都在门禁里逐字段暴露，比事后争论"分数怎么不一样"便宜得多。
3. **精度策略是一个会咬人的隐变量。** TF32 默认开启，训练评价显式关闭；单图推理忘关就差了 4 个像素。差异极小，但方向是"你以为在复现、其实换了数值口径"。
4. **关键是test不参与选择，而不是禁止技术复核。** 依据test挑阈值、轮次或后处理会引入选择偏差；同一冻结模型与协议的排错复核可以保留，但必须说明原因、保留全部结果，不以分数择优。
5. **面积加权会隐藏小目标失败。** test 全局 IoU 0.683 与 macro 0.525 差 0.16；049/057 两张完全没报，077 报了 351 px 全错。只报全局会把这类失败盖住。

自测问题：

1. 如果 val 门禁里只有 `global.foreground_iou` 不一致（差 1e-6），而 132 行逐图完全一致，最可能是什么原因？
   *线索*：§5.3 的字段分层、§7.2 的 29 项；要点是"全局由逐图累计而来，逐图一致而全局不一致通常意味着比较的基准记录来自不同 epoch 或不同阈值"。
2. 为什么 `probability_map` 要强制传 config，而不是给个默认 `None`？
   *线索*：§5.2；要点是默认值会让"忘记精度上下文"变成静默行为。
3. test 的 32/402 张正常图有误报，能不能说"假阳性率 8 %"？
   *线索*：§8.3；要点是不能——那 32 张是"像素级有任意 FP 的图数"，不是按图像判定的假阳性率。

## 10. 下一阶段与修订记录

阶段 13 直接消费本阶段的 `val/`、`test/`、`single_val_045_v2/` 与阶段 11 的 `epochs.jsonl`：

- 曲线只用训练日志，不重训；
- 展示图按本阶段 `display_rules.json` 与 `selection.json` 的规则和名单生成；若需要浮点概率，用**同一条** `inference` 通道对这些少量图重建，并核对二值结果与 `predictions/*.png` 完全一致；
- 失败分析只写"观察—证据—可能原因—下一项验证"，并提出切块/分辨率/采样/DINOv3 等候选对照，**不在本阶段实施**；
- test 已被查看，后续任何方法调整都必须披露这一事实。

| 日期 | 修改原因 | 改动与影响 | 修正后重新验证的证据 |
| --- | --- | --- | --- |
| 2026-11-01 | 初版 | 新增 `inference.py`/`evaluate.py`/`predict.py` 与 10 项单测；val 门禁 29/29 通过；test 442 张一次性评价（IoU 0.6833 / macro 0.5249 / 正常 FP 3 689 px）；修复单图 TF32 精度上下文后 XOR 0 | `reports/stage12-summary.json`、`reports/stage12_evidence/`、`reports/stage12-unit-tests.txt`（273 项）、`..._eval_001\{val,test}` |

### 10.1 2026-11-01独立验收修订

[阶段12—13联合验收](../../reports/2026-11-01-stage12-13-acceptance.md)已通过。原eval001保持只读，新增`pcb1_b0_v2_seed42_eval_acceptance_20261101`用于固定协议技术复核，没有重新训练或按分数择优。

- v2门禁74/74通过，绑定完整源码/入口脚本、processor、冻结权重/结构/配置、原图及GT提取清单SHA、运行环境与精度策略；test先核门禁再读test图像/GT字节。
- 修复v1缺失预览可跳过、processor被覆盖参数后检查、id2label比较遗漏、NaN/重复行/部分NA统计放行与输出目录保护不足；旧v1门禁拒绝使用。
- 实际hook观察132张val与442张test均为CUDA/B1/FP32/eval/inference_mode，grad/autocast/TF32关闭。每张预测PNG与首次评价相同；独立从实际PNG和原始GT重算混淆量逐图相等。
- 6张val预览概率差0.0/XOR0；无GT单图045预测127757像素，与完整val掩码XOR0、训练保存概率差0.0。
- 当前291项完整回归，11.593s，退出码0；21项阶段12定向测试覆盖实际无GT导出、原尺寸、门禁前不读test、数据篡改、缺预览、重复/NaN/NA和输出保护。详见`reports/stage12-13-unit-tests.txt`、`reports/stage12-fix-tests.txt`。
- 原训练21文件及eval001原件不变。首次计分元数据中的inference/predict SHA早于TF32修复，完整原计分源码文本未保存；提交7478742的修复后源码已另行归档，不冒称原计分版本。新复核源码逐文件SHA绑定，全部统计/预测复现一致。
- 当前CLI验证完整训练bundle与绝对路径：换机需另存整个21文件run和原数据/提取清单、按记录重建环境；不同根路径需要显式迁移协议，不能只复制best后照抄命令。

| 日期 | 修改原因 | 改动与影响 | 修正后重新验证的证据 |
| --- | --- | --- | --- |
| 2026-11-01 | 独立验收完善 | v2身份/原始字节门禁、严格processor与标签、缺失/NA/NaN/重复检查、原件与输出保护，纠正“一次评价”等同“禁止技术复核”的表述 | 联合验收、74门禁、291项回归、完整val/test原图PNG与GT检查、独立单图 |
