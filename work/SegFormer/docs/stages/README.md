# SegFormer-B0 PCB1 基线：阶段索引

本目录保存每个编号阶段的中文设计与实现文档（英文文件名、中文正文），按主指南步骤编号。
状态只写**实际完成情况**：未执行的阶段不预先标记为完成，受阻时写真实阻塞点。

- 主指南：`../../../../docs/research/2026-10-31-segformer-baseline-end-to-end-guide.md`
- 隔离规范：`../../../../docs/research/2026-10-31-segformer-environment-isolation-and-stage-docs.md`
- 文档模板：`../../../../docs/research/templates/segformer-stage-design-and-implementation-template.md`
- 项目根：`E:\桌面\AI\AI_Try\work\SegFormer`　资源根：`E:\桌面\AI\segformer_resources`

## 状态总览

| 阶段 | 名称 | 状态 | 文档 | 主要证据 |
| --- | --- | --- | --- | --- |
| 01 | 工作区准备与启动入口 | 已验证 | [01-workspace-design-and-implementation.md](./01-workspace-design-and-implementation.md) | `segformer_resources\audit\before-*`；三个脚本的实测输出 |
| 02 | 独立 Conda 环境与 GPU 验证 | 已验证 | [02-environment-design-and-implementation.md](./02-environment-design-and-implementation.md) | `reports\environment.json`；`audit\after-*`；`requirements-lock.txt`；`audit\negative-control-env-check.json` |
| 03 | 引入 PCB1、固定划分与标签规则 | 历史 v1 验收通过，当前划分由阶段 04 升为 v2 | [03-data-preparation-design-and-implementation.md](./03-data-preparation-design-and-implementation.md) | `manifests/archive/pcb1_v1`；第三阶段验收报告；原始数据未改 |
| 04 | 数据审计与可视化 | 已通过验收，v2 分组清单已重出并验证 | [04-data-audit-design-and-implementation.md](./04-data-audit-design-and-implementation.md) | 当前 `manifests`；`reports/data_audit`；第四阶段验收报告；62 项回归；资源目录 18 张五联图 |
| 05 | 预训练权重与模型搭建 | 已通过独立验收 | [05-model-design-and-implementation.md](./05-model-design-and-implementation.md) | 快照 `models\mit-b0\source.json`；`reports\model_download.json`、`model_loading.json`、`model_check.json`；[模型加载与核验报告](../../reports/2026-10-31-stage05-model-loading-and-check.md)；`audit\stage05-*.log` |
| 06 | Dataset、变换与配置 | 已通过独立验收 | [06-dataset-design-and-implementation.md](./06-dataset-design-and-implementation.md) | `reports\dataset_check.json`（39 项检查）；`segformer_resources\runs\dataset_check`（9 张五联图）；`audit\stage06-*.log`、`stage06-unit-tests.txt` |
| 07 | 损失与指标 | 已通过独立验收 | [07-loss-metrics-design-and-implementation.md](./07-loss-metrics-design-and-implementation.md) | [验收报告](../../reports/2026-11-01-stage07-acceptance.md)；`loss_metrics_check.json`（42 项）；`stage07-unit-tests.txt`（186 项）；`stage07-acceptance.json`；归档恢复图 |
| 08 | 20 步冒烟检查 | 已通过独立验收 | [08-smoke-design-and-implementation.md](./08-smoke-design-and-implementation.md) | `reports\stage08-gates.json`（5/5 技术门禁）；`runs\smoke_001\summary.json`、`updates.jsonl`、`reload_check.json`；`audit\stage08-*.log` |
| 09 | 8 张训练图过拟合检查 | 已通过独立验收 | [09-overfit-design-and-implementation.md](./09-overfit-design-and-implementation.md) | `runs\overfit_002\summary.json`、`evaluation_curve.json`、`eval\*`（仓库外）；`reports\stage09-summary.json`（仓库）；崩溃现场 `runs\overfit_001` 与 `runs\archive\overfit_001_attempt{1,2}_*`；回归 `runs\smoke_002_stage09_regression` |
| 10 | 完整训练集短程试跑 | 已通过独立验收 | [10-pilot-design-and-implementation.md](./10-pilot-design-and-implementation.md) | `runs\pilot_001\summary.json`、`epochs.jsonl`、`updates.jsonl`、`reload_check.json`（仓库外）；`reports\stage10-summary.json`、`reports\stage10-unit-tests.txt`、`reports\figures\stage10-*.png`（仓库） |
| 11 | 正式训练与可靠保存 | 正式训练及保存模型已验收；恢复数值复现未通过原容差 | [11-training-design-and-implementation.md](./11-training-design-and-implementation.md) | [验收](../../reports/2026-11-01-stage11-acceptance.md)；`reports/stage11-acceptance.json`、`stage11-reload-check.json`、`stage11-resume-fixture.json`、263项回归、`stage11_evidence`原件；权重留仓库外 |
| 12 | 独立推理与原图评价 | 未开始，已有执行交接 | — | [阶段12执行交接](../../reports/2026-11-01-stage12-execution-handoff.md) |
| 13 | 结果可视化、失败分析与实验卡 | 未开始 | — | — |

## 当前状态一句话

历史阶段09验收：步骤01—09通过，当时完整项目回归207项通过。第09阶段200/201/1跳步，固定训练子集全局IoU 0.008250→0.945710、异常macro 0.014414→0.760745、正常FP 873,511→85。该结果不代表泛化性能；当时完整val或test尚未评价。新进程保存物复验及当前代码smoke均通过，详见[联合验收](../../reports/2026-11-01-stage08-09-acceptance.md)。

**第10阶段已通过独立验收**：795/795更新、0跳步；最终231项回归通过，新进程完整132图的逐图/macro/global/NA一致，预览概率差0、XOR0。val全局IoU0.6081、异常macro0.2207，仍有7/12异常图预测空，不能当正式基线成绩。[验收入口](../../reports/2026-11-01-stage10-acceptance.md)。

**第11阶段正式训练与保存模型已验收，恢复数值复现有明确限制**：50轮，13250尝试 / 13243成功 / 7次AMP跳步；best39，val全局IoU0.7775、异常macro0.5396。当前263项回归，新进程完整132张val与6张预览通过；未读取test图像或执行阶段12。[验收记录](../../reports/2026-11-01-stage11-acceptance.md)。

## 阶段 11 关键结论

- 协议与覆盖：530 train / 132 val、B2、FP16AMP、workers0、随机水平翻转；每轮530唯一图、26500访次。预算13250、warmup662、cosine跨度12588，跳步不推进调度。
- 末次实际更新LR1.000045016763374e-6，更新后保存LR1.0000330735426696e-6。warmup在第3轮已达到base LR，未验证其与前3轮全背景的因果。
- best39：TP279246 / FP17894 / FN62003 / TN197941817；global IoU0.7775342969、Dice0.8748459012；异常macro IoU0.5395555278、Dice0.6203118681；正常1495FP像素、11/120张有FP。051预测空，042 TP0，081仅命中67/6373，不能只报global。
- 保存物：best/last/mirror、完整optimizer/scaler/scheduler/RNG状态核准；实际138次B1 CUDA IEEE FP32前向，完整逐图/macro/global/normal/NA相等，预览概率差0.0、XOR0。原run21文件大小/SHA不变。
- 代码修复：真实路径/身份与processor绑定、日志一致性与尾日志拒绝、完整优化器slot、重复resume事件和manifest自引用；新增完整只读复验器及真实边界/last检查器。当前263项回归，含21项正式训练、5项保存复验、6项恢复证据检查测试。
- 旧恢复证据范围已纠正，原报告/源码/记录未覆盖；新A/B共36次更新，真正epoch1载入16项状态及下一轮数据/翻转通过，但epoch3 last概率最大差1.5032291e-4、XOR33，**未通过1e-5/XOR0门槛**。前缀也有差异，严格确定性原因未定位，不提供数值续训保证。
- 正式50轮连续完成，没有用resume，可进入阶段12。旧checkpoint缺新恢复身份字段且源码不同，不能直接续训，保存best推理仍通过。workers>0未验证，单seed/12异常val不能支撑泛化结论。
- [学习文档](./11-training-design-and-implementation.md)、[机器摘要](../../reports/stage11-acceptance.json)、[独立复验](../../reports/stage11-reload-check.json)、[原件](../../reports/stage11_evidence/README.md)及精选图保存在Git；权重、概率、环境和数据留仓库外。

## 阶段 10 关键结论

- 入口：新增 `segformer_lab/pilot.py`（`--config/--out/--verify-run`），不修改阶段 09 的八图边界；`train_step` 与 `step_optimizer` 从 `training.py` 抽出共享，`train.py` 一行未改。
- 数据：`train_ids_file: null`，train 530（482N+48A）/ val 132（120N+12A）；每轮 265 个 B2 batch、末批 2 张，覆盖审计 530/530、重复 0、缺失 0；翻转 248/265/265 张，翻转指纹逐轮不同，`dataset.epoch` 1/2/3。
- 计数与调度：795 尝试 / 795 成功 / 0 跳步，`scheduler.completed` 795；warmup 40 步（0.05×795），LR 1.5e-6 → 第 40 步 6e-5 → 第 795 步 **1e-6**。调度只在 optimizer 真执行后前进，单测覆盖跳过分支。
- 损失（加权）：CE 0.118857→0.017636→0.012300；Dice 项 0.984199→0.962749→0.911114；总 0.610957→0.499010→0.467857。CE 是逐像素均值（前景仅 0.17 %），低 CE 不等于分割好。
- 验证（132 张、B1 IEEE FP32、原图、阈值 0.5）：IoU 0.0103→0.2370→**0.6081**；Dice 0.0204→0.3832→0.7563；P/R 第 3 轮 0.8610/0.6743；异常宏平均 IoU 0.0031→0.0656→0.2207，宏平均 Dice 0.0059→0.0985→0.2784；正常图 FP 三轮全 0；NA 为空。
- 逐图（第 3 轮）：12 张异常图中 **7 张预测前景为 0**（第 1、2 轮为 11、9 张）；045/046/059 三张占 val GT 前景 86 %，全局 IoU 由它们撑起，宏平均才 0.2207。"正常图 0 误报"是欠报，不是可靠。
- 保存与复验：best=epoch 3，`best.pt` SHA `d1698b28…`，strict 0/0，`model_state.pt` 与 best 逐张量相等；新进程全量 val 混淆量与全局指标逐位相同，6 张预览概率最大差 **0.0**、XOR **0 / 9 013 680 px**。
- 资源：初始化 3.593 s，脚本总计 **208.17 s**；三轮训练墙钟 61.06/45.20/45.63 s（其中读图等待 23.59/16.28/16.39 s），全量 val 14.30/9.51/9.71 s；单步中位 124.30/105.39/105.36 ms；峰值 allocated **806.3** / reserved **934.0 MiB**；主机 RAM 15.22 GiB、可用 1.73 GiB。
- 身份：git HEAD `0c5f0bd` + 工作区未提交改动（`training.py` `91c829d2…`、`pilot.py` `52e3aae7…`）；起末代码/配置/清单 SHA 与 `git diff` SHA 一致（`source_and_config_unchanged: true`）。
- **未实现 resume**（checkpoint 含 optimizer/scaler/scheduler/epoch/RNG，但没有在 epoch 边界实测恢复），`workers>0` 仍未验证；未读 test 图像、未做测试推理、未启动正式训练。

## 阶段 09 关键结论

- 八图4N4A，200成功/201尝试/1跳步；所有指标仅在这八张训练图计分。大缺陷占GT前景约96.43%，必须结合macro及逐图结果。
- 新进程208个模型状态张量全等且有限，205个优化器参数step均200；保存processor实际使用，B1 IEEE FP32原图计分与历史数值一致。旧split误标明确更正。
- 真实step/AMP边界、模式恢复、保存processor和局部图均已完善。第09原旧概率图缺失，概率对照为null；分数复算通过，不能混淆。
- 三次失败各进入第76次尝试，加记录运行201次，共429/500；不重跑200步。当前代码另建20步smoke，20/20/0，新进程概率差0、XOR0。
- [验收报告](../../reports/2026-11-01-stage08-09-acceptance.md)、[原始证据](../../reports/stage08_09_evidence/README.md)、[只读复验](../../reports/stage09-reload-check.json)和[机器摘要](../../reports/stage09-acceptance.json)均在仓库。

## 阶段 08 关键结论

- 固定子集：`manifests/overfit_train_ids.txt`（SHA `2ac81b81…`），4 正常 + 4 异常；异常前景 1 150 → 191 178 px，跨两个数量级。阶段 09 复用同一文件。
- 计数：成功更新 **20 / 尝试 20 / 跳步 0**；停止原因 `target reached`。
- 梯度：编码器 0.0931、解码头 0.0253；参数 `max|Δ|` 分别 6.014e-05 / 6.004e-05，符合一步 AdamW 幅度。
- 计时与显存：首步 3 423 ms，稳态中位 **58.5 ms**；峰值 allocated 806.5 / reserved **934.0 MiB**（20 步实测，不能当正式训练预算）。
- 评价：6 张固定 val，B1 FP32（TF32 关闭），IoU 0.0、误报 264 px；**只证通路，不是基线**。
- 保存重载：run 内结构 + 权重 `strict=True`，missing/unexpected 为 0；6 张图最大概率差 **0.0**，XOR **0**，混淆量与指标完全一致。
- 本轮已用真实CPU GradScaler回归覆盖跳步/范数边界，并使用保存processor重新复算旧smoke；原源码SHA不一致的历史局限保留，workers>0与完整resume仍未验证。

## 阶段 07 关键结论

- 组合损失：`L = 1.0×CE + 0.5×Dice`；CE 在**上采样到 512 的 logits** 上算，`ignore_index=255` 与当初画 padding 的分辨率一致。
- Dice 分子是逐像素交集 `2Σ(p·g)`；手算夹具（Σp=1.75, Σg=2, Σ(pg)=1.5）得 0.19999987 对 0.2，差来自 eps=1e-6 的 float32 舍入。
- 全正常 batch 的 Dice 为图相连的精确 0；Dice 单独反传得到零梯度，CE 仍监督正常图。CE 与 Dice 的数值计算均使用 FP32。
- 全忽略 batch 抛 `ValueError` 而不是返回 NaN；标签出现 0/1/255 之外的值同样拒绝。
- padding：同分辨率改 padding logits，loss 差 **0.0**；在 128 网格上改**完全落在 61 行 padding 内**的 0–9 行，loss 差也是 **0.0**。
- 指标：`int64` 累计 TP/FP/FN/TN 后只除一次；2×2 手算 IoU 0.5 / Dice 2/3 / P 1 / R 0.5 全部对上。
- 人工例子中「每图平均」0.505、「全局累计」2/101，不能混用；这不是一般大小关系。
- 特定合成恢复夹具：正确路径质心误差 **0.366 px**；漏裁 padding 为 **36.141 px**，不是通用偏移常数。
- 单像素夹具：512 采样丢掉的 1 个 GT 像素，用原掩码记 **FN=1 / IoU=0**，用放大回去的 512 标签记 **NA**。
- 真实 probe（6 张 val、随机解码头）：9,013,680 像素；相同 logits 改变累计分组严格一致。旧“12 px”统计错误已撤回；IEEE FP32 下 B2/B1 真 XOR 为 1，最大概率差 `1.4901161193847656e-7`，固定 B1 重复差为零。未定位具体 kernel。**这些是数值通路检查，不是基线性能。**
- `eval.precision: ieee_fp32` 和 `evaluation_precision` 上下文统一评价策略：关闭 TF32/autocast、退出恢复；训练 AMP 独立。`--skip-model` 仅夹具诊断，不会使全阶段通过。

## 本轮验收

[阶段01—02验收与完善记录](../../reports/2026-10-31-stage01-02-acceptance.md)：修复跨 PowerShell 的空变量删除、资源根传递与恢复问题，收紧环境检查器，补充完整可移植依赖。修正后真实 CUDA／随机 SegFormer 前后向通过，缓存越界 0。

[第三阶段验收与完善记录](../../reports/2026-10-31-stage03-acceptance.md)：独立复算划分，核对 1205 个文件与归档成员哈希，修复写入/协议保护后全套 23 项测试通过。

[第四阶段验收与完善记录](../../reports/2026-10-31-stage04-acceptance.md)：原 18 张预览逐看；新版 15 张字节未变、另 3 张重新查看；登记正常图 3 个疑点；加入逐原域追踪和分组校验；完整审计 13 项及项目回归 62 项通过。当前证据见 `reports/stage04-acceptance.json`。

[第五阶段模型加载与核验报告](../../reports/2026-10-31-stage05-model-loading-and-check.md)：固定 commit 下载、逐键核对编码器、键分类、有效区 CE、梯度与一步更新、batch 1/2 显存、快照未被覆盖；当前项目回归101项、加载条件5项、运行条件9项通过；已归档小型model_source.json，按固定commit+全文件SHA换机恢复，真实快照留仓库外。

[第六阶段验收](../../reports/2026-10-31-stage06-acceptance.md)：39项数据契约、134项回归通过，9张图逐看；修复skip-model假通过和source GT展示，完善子集/processor/seed门槛。完整身份与代码证据见reports/stage06-acceptance.json；workers>0仍未验证。

[第七阶段验收](../../reports/2026-11-01-stage07-acceptance.md)：42 项完整检查、186 项回归通过；明确 NA、原图 255、FP16 CE、真实 XOR、精度策略与前置身份保护，源码及小型证据保存在仓库。旧版和 TF32 对照分别归档，不覆盖失败事实。未训练，不报告模型质量。

## 下一步

只做[阶段12执行交接](../../reports/2026-11-01-stage12-execution-handoff.md)：冻结best39与保存结构/processor，新进程先对齐完整132图val，再评价442图test（402正常+40异常）。固定原图还原、0.5、B1 IEEE FP32；不能依据test选择权重/阈值/后处理。必要技术复核须同协议并保留原因和全部结果，不按成绩择优。

本轮未实施12或读取test图像。跨机器另行复制训练best及processor并核SHA。恢复数值限制与workers>0未验证如实带入后续；042/051/081的失败归因留阶段13。

## 维护约定

1. 同一阶段多次修正时，更新原文件并在第 10 节「修订记录」追加一行，注明哪些验证是修正后重跑的；不产生多份「最终版」。
2. 文档中的版本、路径、计数必须能在证据文件里找到，不允许出现「预计」「应该」当作结论。
3. 未运行的命令单独标注「待执行」，不与已执行结果混排。
