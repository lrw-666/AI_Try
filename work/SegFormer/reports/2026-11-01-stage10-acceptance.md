# 第十阶段验收与完善

日期：2026-11-01。范围是完整数据三轮 pilot 的实现、原始记录、保存状态及新进程复算；没有启动正式 50 轮训练或测试评价。

## 1. 阶段结论

**第十阶段通过验收，可以进入第 11 阶段。**三轮记录的数学、数据覆盖、完整保存状态和真实 CUDA 新进程复算均通过；该结论验证流程可靠性，不代表目前的小缺陷分割效果已满足论文要求。

| 项目 | 原始实测与复算 |
| --- | --- |
| 数据 | train 530（482 正常+48 异常），val 132（120 正常+12 异常），v2 清单未改 |
| 三轮覆盖 | 每轮 530 图各一次、265 个 B2 batch，drop_last=false；无遗漏或重复 |
| 更新与调度 | 795 尝试/795 成功/0 跳步；scheduler completed=795；warmup 40，cosine 755，末步 LR 1e-6 |
| 增强 | epoch 1/2/3，翻转 248/265/265，确定性计划及指纹一致；沿用已验收同步变换代码 |
| 验证全局 IoU | 0.010309→0.237012→0.608100 |
| 异常图宏平均 IoU | 0.003082→0.065552→0.220685 |
| 正常图误报 | 三轮均为 0 像素；最后仍有 7/12 异常图预测为空 |
| best | 按完整 val 全局前景 IoU 选 epoch 3；没有根据 test 选参数或权重 |

这些是短程 pilot 的留出验证读数，**不是正式基线或最终测试成绩**。最后一轮 TP/FP/FN/TN 为 230,093 / 37,131 / 111,156 / 197,922,580；全局 Dice 0.756296、precision 0.861049、recall 0.674267。

045/046/059 三张大缺陷合计占验证 GT 前景约 86%，支配全局 IoU；小缺陷效果仍不足。三轮不足以确定原因为欠训练、分辨率限制还是类别不平衡，也不能保证 50 轮必然改善。下一阶段先保持固定配方，不能因 pilot 读数直接修改损失或阈值。

## 2. 本轮完善

1. 原复验主要判定全局分数与预览概率，现补齐完整逐图/macro/正常图/NA 比较，防止全局相同而逐图结果不同仍被判通过。
2. 概率对照明确检查 ID 集合、shape、有限性和范围；每图误差须在原 `atol=1e-5` 内，且二值 XOR=0。不让 NaN 被聚合函数掩盖，也不以“小于容差”放过跨阈值的预测变化。
3. 补齐原始产物 bytes/SHA、数据与配置锚点、best/last/纯权重镜像的状态一致性，以及优化器、scaler、scheduler、epoch/step/LR 等保存状态检查。复验输出独立文件，不覆盖历史 run。
4. 起止身份比较覆盖数据清单、split_meta 与固定预览；发现变动应失败，不能仅写 false 后仍返回成功。当前三轮训练没有这类变动，不需要重训。
5. 修正 NA 解释：全局指标有定义不等于逐图没有 NA；正常 GT/预测均空时 IoU/Dice/precision/recall 是 null。异常图 precision 的 NA 数为 11/9/7。
6. 修正文档的产物数、计时因果、batch/LR 关系与未来验证样本量说明；保持 workers=0，不将可选吞吐优化变成正式训练的前置任务。

完整复验还定位了一个构造器副作用：本机 Transformers 的 `SegformerConfig.from_dict` 会给输入字典增加 `attn_implementation`。现使用深拷贝构造，避免验收器自行改动载入的元数据而误报配置不一致；新增回归已先复现失败再验证修复，不修改任何原始 checkpoint。

保存了准确的记录运行源码，以及三轮日志、逐图结果、原复验与身份记录，见[原始小型证据](./stage10_evidence/README.md)。新源码的 SHA 与旧训练源码不同属于验收修正，不能把两者混成同一版本。

## 3. 最终独立验证

| 验证 | 本轮实测 | 证据 |
| --- | --- | --- |
| 完整项目回归 | **231 项通过，10.516 s，退出码 0** | [测试日志](./stage10-unit-tests.txt)；原219项日志另存原件目录 |
| 新进程 CUDA 复验 | `ok=true`、`not_a_baseline=true`；13.7 s | [完整复验 JSON](./stage10-reload-check.json)、[日志](./stage10-reload-check.txt) |
| 产物与身份 | 30 个产物的 bytes/SHA、原 summary 的29项快照及起止身份通过；原run全部32文件未改 | JSON inventory/identity 及 [只读审计](./stage10-evidence-audit.json) |
| best / last 保存状态 | epoch=3、step=795；每份均有205个优化器参数状态，所有 step=795；scheduler=795；模型、scaler、RNG结构与原记录一致 | JSON 的 `checks.checkpoints`；这不是 resume 实验 |
| 完整原图分数 | 132 图的 per-image、macro、normal、NA、混淆量及 global 全部严格一致 | `checks.complete_val_metrics.fields_identical` 全部为 true |
| 六张固定预览概率 | 9,013,680 像素，最大概率差 **0.0**，XOR **0**，保持 `atol=1e-5` | `preview_probability_comparison` |
| 保存预处理 | 未覆盖 flags，直接加载保存 processor；实际传给完整val及预览Dataset | JSON 的 `processor` |

复验使用原隔离环境、FP32、B1、无增强和 IEEE 精度上下文，没有反向或优化更新。原始三轮代码与新复验代码分别保留哈希；当前代码身份及交付物摘要见 [stage10-acceptance.json](./stage10-acceptance.json)。

## 4. 图片和解释

已查看最终固定六张预览及早期漏检图；未见整体图像/掩码错位。

- [第三轮正常0011](./figures/stage10-epoch03-normal-0011.png)：GT 和预测都空，逐图 IoU 应为 NA。
- [第三轮异常045](./figures/stage10-epoch03-anomaly-045.png)：大区域已定位，边缘和引脚附近有多报，不能只看覆盖面积。
- [第三轮异常087](./figures/stage10-epoch03-anomaly-087.png)：原 GT 仍有细缺陷，预测为空，与最终漏检记录一致。
- [第一轮异常087](./figures/stage10-epoch01-anomaly-087.png)：保留早期对照；不冒充最终 epoch 的图。

## 5. 资源与限制

原三轮初始化及全部环节约 208.17 s；训练墙钟总 151.894 s、全量验证总 33.524 s、出图 1.643 s、保存 5.429 s。峰值 allocated/reserved 为 806.3/934 MiB，主机可用内存约 1.73 GiB。训练步中位约 105 ms 不含全部读图等待，不能用它直接估计总训练时间；多轮不意味着单步显存必然增长。

完整 epoch resume 尚未实现或实测；保存了 RNG/optimizer/scaler 不等于验证过恢复。当前 checkpoint 也没有完整记录 DataLoader generator 状态。下一阶段应先做独立短程恢复对照，再执行正式预算。workers>0 仍未验证，可继续采用 0。

原运行调用了已带来源校验的模型加载路径，但没有单独保存 `loaded.snapshot/source_sha/encoder_check` 的运行开始记录。当前来源文件及固定模型文件已核准，不能反向补写为当时曾独立冻结过的哈希；阶段 11 须补齐这类来源证据。增强方面核对了 epoch、覆盖及确定性翻转计划，实际像素同步沿用阶段 06 已验收的代码，并未逐张目击三轮所有增强图。

保持 test 隔离：仅核对其清单哈希，没有读取测试图像/掩码、推理或选参。验证集仍只有 12 张异常图；增加训练 epoch 只增加候选权重点，不增加验证样本量。

## 6. 学习材料与下一步

[第十阶段学习文档](../docs/stages/10-pilot-design-and-implementation.md)补充了共享 step、调度算术、加权损失、NA 和完整保存物核验的细节。

下一步仅按[阶段 11 执行交接](./2026-11-01-stage11-execution-handoff.md)实现正式单基线：seed 42、50 个完整 epoch、530 train/132 val，重新从固定预训练初始化，不接续 pilot。总预算 13,250 次 batch 尝试，按当前 Python round 规则 warmup 662；出现跳步不重复数据补成功数。

先补全正式入口和 epoch 边界恢复证据，再开始本阶段训练；完成后再次验收，不自动进入最终 test 评价。本轮尚未执行这些工作。
