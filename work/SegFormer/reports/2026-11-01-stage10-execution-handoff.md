# 阶段 10：完整训练集三轮试跑执行交接

日期：2026-11-01。本文为下一阶段实施指令，**没有启动 pilot 或正式训练**。执行前先读[第八、九阶段验收](./2026-11-01-stage08-09-acceptance.md)的最终结论。

## 1. 下一步要回答的问题

阶段 08 证明训练/保存通路，09 证明固定 8 张训练图可学习。阶段 10 改用完整 **530 张 train、132 张 val**，跑 **3 个完整 epoch**，核对增强、逐轮验证、学习率、保存与资源使用能否共同工作。它仍是短程排错，不是最终论文基线成绩。

本轮只执行阶段 10。三轮结束后返回验收，不自动启动 50 轮训练；不读取 test 图像或掩码，不进行测试推理、选参或报分。测试侧只允许检查清单完整性与哈希。

## 2. 可直接复制给实施 Agent 的指令

```text
请只实现并执行 SegFormer 主指南的阶段10：完整数据三轮pilot。
先读AI_Try/AGENTS.md、work/SegFormer/AGENTS.md、阶段索引、08/09最终验收及本文。
保留当前独立Conda环境、PCB1 v2分组清单、原始数据与固定MiT-B0快照；不安装到base，
不改全局配置、不覆盖已有run。所有阶段说明用中文MD，代码与小型证据放AI_Try。

先核实当前CLI：train.py目前专用于8图overfit，硬性要求8图/关闭翻转；不能简单把
pilot.yaml传给它，也不能为了pilot删掉08/09的边界。最小新增pilot入口或明确任务分支，
共享data/loss/metrics/evaluation/训练步和保存加载代码，避免两套损失或评价实现。

建立完整configs/pilot.yaml：train_ids_file=null，train全530、val全132；3个epoch，
512 letterbox、B2、累积1、workers0、seed42、AMP FP16，全网络微调。
训练flip=random、horizontal_flip_p=0.5，每轮开始显式dataset.set_epoch(epoch)；
val无增强，B1、FP32、eval.precision=ieee_fp32，原图GT与阈值0.5保持。
CE+0.5正GT逐图Dice/eps1e-6/ignore255不改，不加裁块、加权采样、后处理或DINO。

从原固定MiT-B0编码器和新二类头重新初始化，不能继承smoke/overfit权重或优化器。
复用baseline的AdamW分组、LR6e-5、weight_decay0.01、grad clip1.0；落实基线配置的
warmup_ratio=0.05、衰减及minLR1e-6，明确warmup/cosine的实际步数及off-by-one定义。
pilot按3轮预算建schedule，正式50轮会重新建schedule，不伪称两者学习率轨迹相同。
scheduler只在optimizer确实执行更新后step，AMP跳步不能推进scheduler/成功计数。

按epoch完整遍历，不用“补到795次成功更新”替代3轮数据覆盖；drop_last=false。
每轮265个B2 batch，三轮最多795次更新尝试；若有AMP跳步，成功更新可少于795，
日志须分别记录attempt/success/skip。每轮核实train ID覆盖530、图/标签同步与epoch变化。
采用已验收的有限跳步策略；非有限loss/参数、范数溢出或连续异常必须停下保留现场。

每轮在完整132张val上顺序评价，汇报全局前景IoU/Dice/P/R、异常图macro、正常误报
及NA原因；只按val全局前景IoU选best，不以8图overfit分数作达标线。
训练日志聚合CE按有效像素加权、Dice按正GT图数加权；不简单平均batch均值冒充全局loss。
仅保存固定6张val预览及必要失败图；不要把全部132张原尺寸概率/图像缓存到RAM。

保存last和按val选出的best：模型/结构/processor、optimizer/scaler/scheduler、epoch与
成功更新/尝试数、解析配置、数据/代码身份及必要RNG状态。用新进程严格加载已训练的
best及run内processor，对固定val样本复算并核对保存前后的原图结果。
若增加resume，只支持明确的完整epoch边界并实际验证；未实现完整恢复就如实说明，
不能仅保存了模型和optimizer便宣称已验证断点续训。

记录初始化/数据读取/训练step/完整验证/保存耗时，显存allocated/reserved与主机RAM；
逐步记录AMP scale、跳步和真实optimizer/scheduler计数。新run放资源目录，不覆盖旧run。
固化运行开始与结束的代码/配置/清单SHA、Git HEAD及未提交diff，避免运行时改源码。

交付docs/stages/10-pilot-design-and-implementation.md、独立配置、必要回归、三轮日志、
逐轮完整val指标、小型JSON证据、精选预览、checkpoint清单及新进程复验。
更新README和阶段索引。三轮结束或遇到无法解释的失败后停下，提交验收；不自行进入11。
```

## 3. 验收重点

| 重点 | 需要看到的真实证据 |
| --- | --- |
| 完整数据而非残留子集 | `train_ids_file=null`，train 530/val 132，各轮 ID 覆盖与清单 SHA；风险组不跨集 |
| 训练增强恢复 | `flip=random`，每轮 `set_epoch`，若干增强叠加图；val 不翻转 |
| 训练过程稳定 | 三轮真实 attempt/success/skip，非有限处理、optimizer 与 scheduler 成功步数一致，loss/LR 曲线 |
| 保存可复算 | best/last 完整来源、严格新进程加载，使用保存 processor，原图计分/预测对照通过 |
| 评价可信 | 全132张val；全局、macro及正常误报齐全；不加平滑项掩盖NA；不读test图像 |
| 资源与学习文档 | GPU/主存实测、耗时拆分、清晰中文实现说明；未完成项明确标注 |

不要求三轮达到某个固定 IoU。若短程验证效果低，先结合 loss、逐图预测与正常误报判断原因；不能因为 8 图拟合曾达到 0.946 就要求留出验证也达到该值。三轮效果低也不自动改损失、阈值或抽样协议。

大文件仍保存在 `E:\桌面\AI\segformer_resources\runs\pilot_...`。仓库只收配置、代码、文档、清单、小型日志/指标和少量精选图。运行入口须先通过 `--help` 核对，不把本文待实现的接口写成已执行结果。
