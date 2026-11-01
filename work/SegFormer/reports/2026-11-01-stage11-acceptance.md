# 第十一阶段验收与完善

日期：2026-11-01。**正式50轮训练与保存模型通过验收，可进入阶段12。恢复的状态与数据顺序已验证，端到端续训数值对照未通过预设容差，不能称“恢复全面通过”。**本轮没有重跑正式训练，也没有读取test图像或进行测试评价。

## 1. 正式训练结果

| 项目 | 已独立核准的结果 |
| --- | --- |
| 数据与预算 | 530 train / 132 val，50完整epoch；每轮530唯一图，合计26500访次 |
| 更新 | 13250尝试 / 13243成功 / 7次AMP跳步；最长连续跳步1 |
| 调度 | 13250/662总预算与warmup；所有逐次LR、scheduler计数一致，跳步不推进 |
| 选模 | best为epoch39；按完整val全局前景IoU选出，不使用test |
| best验证 | IoU **0.777534**、Dice **0.874846**、precision **0.939779**、recall **0.818306** |
| 异常macro | IoU **0.539556**、Dice **0.620312** |
| 正常误报 | 1495像素，11/120张正常图出现任意FP，像素比约8.293e-6 |
| best状态 | 10330成功 / 10335尝试 / 5跳步；205个优化器参数step均10330 |
| last状态 | 13243成功 / 13250尝试 / 7跳步；205个优化器参数step均13243 |

以上是**验证集选模结果，不是最终测试成绩**。只有051预测全空，但042的TP=0、081只命中67/6373个GT像素，不能据“空预测只剩1张”判断小缺陷都已解决。验证集仅12张异常图，未计算置信区间或多seed稳定性。

冻结best SHA：`192c827bef00607f4258e5fadf0f1d97970ca5cc727a0add749be0fba5997485`。

## 2. 本轮独立验证

- **263项完整回归通过，11.006s，退出码0**，见[日志](./stage11-unit-tests.txt)。原242项日志另存原件目录。
- [独立审计](./stage11-evidence-audit.json)复算全部13250条更新和6600条逐图验证结果，覆盖、加权loss、LR、global/macro/normal/NA和选模均一致。更新日志按8位小数保存，加权重算差约1e-9，未冒称全精度逐值一致。
- [新进程CUDA复验](./stage11-reload-check.json)完成138次实际B1 IEEE FP32前向：完整132张val与6张固定预览；保存processor实际被使用，模型、输入、logits为FP32，eval/inference_mode、autocast/TF32关闭。
- best、last和镜像严格加载；完整状态、配置、optimizer分组及moment、scaler、scheduler、epoch/step/LR、RNG及generator结构核准。完整逐图/macro/global/normal/NA与保存值一致；预览概率最大差 **0.0**、XOR **0**。
- 原正式run **21个文件** 的大小/SHA前后未变；19项产物清单通过。权重、概率及批量图片没有移入Git。

## 3. 恢复代码修复与证据纠正

原代码存在明确风险：同run只比basename；resume事件行会破坏再次续训统计；保存processor未实际绑定训练Dataset；身份检查不完整；优化器缺slot可能被接受；恢复后manifest自引用过期。现已修复，并增加CPU故障回归。

当前恢复仅接收路径、配置、源码、数据/模型/processor身份一致，且日志与last一致的完整epoch边界。损坏或未提交尾日志会保留并拒绝，不承诺任意崩溃自动恢复。旧训练checkpoint因缺少新身份字段或源码版本不同，不能不经审核强行续训；原best仍可正常只读推理。

旧fixture报告不能按字面引用：其NPZ来自epoch1 best，而非epoch3 last；state_at_load读取的是已覆盖的epoch3检查点，且未完整比较所声称的RNG/generator。原报告、原脚本及A/B/D/E记录均保留在[原件目录](./stage11_evidence/README.md)，没有静默改写。

本轮另做36次小样本更新，不继承给正式模型：A连跑18次；B先6次、在run外冻结epoch1检查点并先复验，再续12次。新证据如下：

| 范围 | 结果 |
| --- | --- |
| 真正epoch1加载 | 模型、optimizer/groups、scaler、scheduler/LR、计数/best、Python/NumPy/Torch/CUDA RNG、实际loader generator精确一致；16项状态检查为true |
| 下一轮实际数据 | 真实解码6批，顺序与flip逐批一致 |
| epoch3 last结构 | 两侧计数/LR/scheduler、12条后续记录的顺序、flip和更新计数一致 |
| last数值对照 | 模型最大差约4.71834e-4，概率最大差 **1.5032291e-4**、XOR **33**，逐图分数不完全相同 |
| 原定数值门槛 | `probability_atol=1e-5` 且 `XOR=0`；**未通过，未放宽** |

[加载边界报告](./stage11-new-resume-boundary.json)、[真正last对照](./stage11-new-resume-continuation.json)、[纠正后的恢复汇总](./stage11-resume-fixture.json)明确区分结构通过与数值失败。两个独立前缀本身也有差异，本对照不足以将偏差唯一归于恢复边界、FP16或某个GPU kernel；严格确定性诊断仍未完成。

**因此只确认完整状态和数据顺序恢复，不提供逐位或1e-5内的续训轨迹保证。**正式50轮本身连续完成，没有用resume，这项限制不妨碍对既定best进行独立测试。

## 4. 其他完善与学习材料

新增完整只读验收器，旧验证入口委托到它；不再仅凭全局分数或误差比例判通过，必须核对每图/macro/NA、概率key/shape/有限性/XOR和完整保存状态。修复后的恢复使用实际文件SHA与保存processor，模型配置构造不改原字典。

[第十一阶段学习文档](../docs/stages/11-training-design-and-implementation.md)已重写：纠正warmup时点的错误因果、旧恢复结论和未经计算的“噪声”判断。正式曲线与精选图见 `reports/figures/stage11-*`；045的大区域较好，087仍有多报，0011为空预测，与逐图记录相符。

训练墙钟2609.179s、全量验证533.355s，保存8.053s；峰值allocated/reserved为806.3/934MiB。单步中位不是端到端耗时，不能把速度变化单独归于长跑或保存。workers>0未验证，可继续使用0；本轮未安装包、修改base或全局配置。

正式全量日志以无损gzip归档，解压内容逐字节匹配原件。原训练源码与当前修正源码分别保存SHA。机器摘要见 [stage11-acceptance.json](./stage11-acceptance.json)。

## 5. 下一步

按[阶段12执行交接](./2026-11-01-stage12-execution-handoff.md)，冻结best39、processor及协议，新进程先复算完整val，再对 **442张test（402正常+40异常）** 做独立评价及单图推理。不得依据test改阈值、epoch或后处理；必要技术复核可用同一冻结协议重跑并记录原因。

本轮未执行stage12。跨机器需另外复制已训练best和processor并核SHA，Git没有大权重，不能用重新下载的MiT-B0替代已训练模型。
