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
| 08 | 20 步冒烟检查 | 未开始 | — | — |
| 09 | 8 张训练图过拟合检查 | 未开始 | — | — |
| 10 | 完整训练集短程试跑 | 未开始 | — | — |
| 11 | 正式训练与可靠保存 | 未开始 | — | — |
| 12 | 独立推理与原图评价 | 未开始 | — | — |
| 13 | 结果可视化、失败分析与实验卡 | 未开始 | — | — |

## 当前状态一句话

步骤 01—07 已验收：PCB1 v2 分组清单、MiT-B0 快照、Dataset 契约未变；损失与原图指标经 **42 项完整检查、186 项项目回归**验证。原图恢复按「512→裁 padding→原尺寸」；评价 B1、显式 IEEE FP32；单像素漏检记 FN，正常空预测记 NA。**尚未进行本基线训练，没有训练后性能结果**。阶段 08、09 可同轮串行实施，但当前仍是未开始。

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

按[阶段 08—09 执行交接](../../reports/2026-11-01-stage08-09-execution-handoff.md)可同轮串行执行：08 完成 20 次成功更新、AMP、保存与新进程重载，内部门槛通过后，09 独立初始化并检查固定 8 张训练图的学习。两份配置、run、中文文档和证据独立，完成后合并交付验收；失败先停当前阶段，不自动进入 10/11。

## 维护约定

1. 同一阶段多次修正时，更新原文件并在第 10 节「修订记录」追加一行，注明哪些验证是修正后重跑的；不产生多份「最终版」。
2. 文档中的版本、路径、计数必须能在证据文件里找到，不允许出现「预计」「应该」当作结论。
3. 未运行的命令单独标注「待执行」，不与已执行结果混排。
