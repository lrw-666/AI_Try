# PCB1 SegFormer-B0 实验卡

- 日期 / run ID：2026-11-01 / `pcb1_b0_v2_seed42`
- 当前状态：正式训练（50 轮）与独立测试评价已完成，**已通过2026-11-01独立验收**；未做多 seed、未做跨数据集、未部署
- 任务：有监督背景/缺陷二值分割（PCB1，VisA）
- 数据版本与清单 SHA：train `f1522eb9b474d0ac…`、val `60a6c6cbf5ed7f5f…`、test `4bdc594148f208c7…`、`split_meta` `7c53a669aabebcbb…`（PCB1 v2 分组，seed 42）
- 划分：530 / 132 / 442；train 482 正常 + 48 异常，val 120 + 12，test 402 + 40
- 预训练：`nvidia/mit-b0`，revision `80983a413c30d36a39c20203974ae7807835e2b4`，权重 SHA `4af8348c2a802bf7…`（编码器 192/194 张量精确匹配）
- 模型与参数量：SegFormer-B0 + 二类解码头，3,714,658 参数（编码器 3,319,392 + 头 395,266），全量微调
- 输入 / 还原：512 letterbox；标签 padding=255 用于训练/验证预处理中的忽略区域，原尺寸评价 GT 仅 0/1。原图 1404×1070 尺度计分；`logits→512→裁 padding→原 H/W→softmax→阈值 0.5`
- 损失：`CE(ignore_index=255) + 0.5 × 正 GT 图逐图前景 Soft Dice`，eps 1e-6，Dice 在 FP32 上算
- 训练：物理 batch 2、累积 1、有效 batch 2；LR 6e-5（warmup 662 步线性 + cosine 到 1e-6）；50 epoch / 13 250 次计划更新；AMP FP16 + GradScaler；seed 42；`flip: random` p=0.5，每轮 `set_epoch`
- 环境：Python 3.12.16 / torch 2.10.0+cu126 / torchvision 0.25.0+cu126 / transformers 4.57.6 / CUDA 12.6 / RTX 4070 Laptop 8 188 MiB（驱动 617.42）
- 代码身份：git HEAD `18eadb8dc19a53fa6c98b6ec25558508838a60bb` + 未提交改动（14 个源文件 SHA 记录在 run 的 `code_identity_start.json`）；`source_and_config_unchanged = True`
- 实际完成轮数 / 最佳 epoch：50 / **39**，按 val 全局前景 IoU 选择（并列取较早轮次）
- val（132 张，原图 IEEE FP32，阈值 0.5）：IoU **0.7775**、Dice 0.8748、P 0.9398、R 0.8183；异常 macro IoU 0.5396；正常图 FP 1,495 px（11/120 张）
- test（442 张，同一冻结协议；原始评价与验收技术复核分别留档）：IoU **0.6833**、Dice 0.8119、P 0.9008、R 0.7390；异常 macro IoU 0.5249、macro Dice 0.6351；正常图 FP 3,689 px（32/402 张），比例 6.108e-06
- 训练耗时 / 显存：50 轮墙钟 2609.2 s（其中读图等待 898.0 s）、验证 533.4 s；峰值 allocated 806.3 / reserved 934.0 MiB
- 原始阶段 12 评价耗时：val 19.84 s、test 60.89 s（B1，test 含 442 张 0/1 PNG 写盘）；验收技术复核耗时见各评价目录 `evaluation_meta.json.timings`。**推理延迟未测**（没有做预热/多次计时的稳定基准）
- 失败案例：test 中 049、057 预测为空，077 的 351 px 全部为 FP；val 上 042（15 523 px）与 051 仍近乎全漏、081 仅 0.0105。详见 `failure-cases.csv` 与 `figures/anomaly-worst-cases.png`
- 局限：单模型、单 seed、单数据集；仅 40 张异常 test 图，未估计置信区间；全局按并集像素加权，macro 与全局差值不能单独证明面积效应；无图像级判定规则；`workers=0` 未做吞吐对照；恢复的跨进程数值限制保留（阶段 11）
- 独立预测命令：`python -m segformer_lab.predict --config configs/baseline.yaml --run <run> --image <img> --out <dir>`
- 复算 val 命令：`python -m segformer_lab.evaluate --config configs/baseline.yaml --run <run> --split val --out <dir>`
- 路径：run `E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42`；评价 `E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42_eval_acceptance_20261101`；本目录 `reports/pcb1_b0_v2_seed42`

验收记录：[阶段12—13联合验收](../2026-11-01-stage12-13-acceptance.md)。原始执行与验收技术复核分别保留，未按分数择优。
