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
| 06 | Dataset、变换与配置 | 未开始 | — | — |
| 07 | 损失与指标 | 未开始 | — | — |
| 08 | 20 步冒烟检查 | 未开始 | — | — |
| 09 | 8 张训练图过拟合检查 | 未开始 | — | — |
| 10 | 完整训练集短程试跑 | 未开始 | — | — |
| 11 | 正式训练与可靠保存 | 未开始 | — | — |
| 12 | 独立推理与原图评价 | 未开始 | — | — |
| 13 | 结果可视化、失败分析与实验卡 | 未开始 | — | — |

## 当前状态一句话

PCB1 数据（v2 分组划分）与审计已通过验收；`nvidia/mit-b0` 已按 commit `80983a41…` 下载并逐键核验（编码器 192/192 张量数值完全一致），二类 SegFormer-B0 的人工张量前向/反向/一步更新与 batch 1·2 显存均已实测。**尚未实现 Dataset、未训练，无任何性能指标**。

## 阶段 05 关键结论

- 快照：`nvidia/mit-b0@80983a413c30d36a39c20203974ae7807835e2b4`，3 文件 14 450 344 B，权重 `pytorch_model.bin` 14 380 029 B（SHA-256 `4af8348c…27fc`），非 LFS 指针。
- 加载：总参数 3 714 658 全可训练；编码器 192/192 张量与检查点**数值完全相等**（max|diff| = 0.0）；检查点剩余的 2 个 `classifier.*` 是 ImageNet 分类头（257 000 参数）。
- 键分类：missing 16（全部 `decode_head.*`）、unexpected 2、mismatched 0 → 编码器无缺失、无形状冲突；`ignore_mismatched_sizes=True` 本次并未起作用。
- 前向：`[2,3,512,512]` → logits `[2,2,128,128]`（stride 4）→ 上采样 `[2,2,512,512]`；整幅 CE 与有效区 CE 差 0.0（≤1e-6；255 标签不直接参与归约）。
- 反向与更新：编码器 `patch_embeddings.0.proj.weight` 梯度范数 0.115848、解码头 `linear_c.0.proj.weight` 0.212860，均非零有限；一步 AdamW 后两者 `max|Δ|`=1.0e-4。
- 显存（先B1再B2、优化器创建前，FP32/AMP关闭）：504/1008 MiB reserved。接受batch2起步，阶段08再测真实训练。
- 快照完整性：来源与全部文件先校验哈希，再做加载；调试更新前后全部快照文件及source SHA不变；旧1058MiB数据仅作为历史。

## 本轮验收

[阶段01—02验收与完善记录](../../reports/2026-10-31-stage01-02-acceptance.md)：修复跨 PowerShell 的空变量删除、资源根传递与恢复问题，收紧环境检查器，补充完整可移植依赖。修正后真实 CUDA／随机 SegFormer 前后向通过，缓存越界 0。

[第三阶段验收与完善记录](../../reports/2026-10-31-stage03-acceptance.md)：独立复算划分，核对 1205 个文件与归档成员哈希，修复写入/协议保护后全套 23 项测试通过。

[第四阶段验收与完善记录](../../reports/2026-10-31-stage04-acceptance.md)：原 18 张预览逐看；新版 15 张字节未变、另 3 张重新查看；登记正常图 3 个疑点；加入逐原域追踪和分组校验；完整审计 13 项及项目回归 62 项通过。当前证据见 `reports/stage04-acceptance.json`。

[第五阶段模型加载与核验报告](../../reports/2026-10-31-stage05-model-loading-and-check.md)：固定 commit 下载、逐键核对编码器、键分类、有效区 CE、梯度与一步更新、batch 1/2 显存、快照未被覆盖；当前项目回归101项、加载条件5项、运行条件9项通过；已归档小型model_source.json，按固定commit+全文件SHA换机恢复，真实快照留仓库外。

## 下一步

步骤6：实现PCB1 Dataset与DataLoader，复用当前v2、同一geometry与一次processor归一化；检查同步翻转、稳定ID/几何、标签0/1/255和真实B2 eval/no_grad前向。完整指令见[第五阶段验收报告第7节](../../reports/2026-10-31-stage05-model-loading-and-check.md#7-下一步只执行阶段-06)。

## 维护约定

1. 同一阶段多次修正时，更新原文件并在第 10 节「修订记录」追加一行，注明哪些验证是修正后重跑的；不产生多份「最终版」。
2. 文档中的版本、路径、计数必须能在证据文件里找到，不允许出现「预计」「应该」当作结论。
3. 未运行的命令单独标注「待执行」，不与已执行结果混排。
