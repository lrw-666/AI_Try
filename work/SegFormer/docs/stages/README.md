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
| 07 | 损失与指标 | 未开始 | — | — |
| 08 | 20 步冒烟检查 | 未开始 | — | — |
| 09 | 8 张训练图过拟合检查 | 未开始 | — | — |
| 10 | 完整训练集短程试跑 | 未开始 | — | — |
| 11 | 正式训练与可靠保存 | 未开始 | — | — |
| 12 | 独立推理与原图评价 | 未开始 | — | — |
| 13 | 结果可视化、失败分析与实验卡 | 未开始 | — | — |

## 当前状态一句话

PCB1（v2 分组划分）审计通过，MiT-B0 快照按 commit `80983a41…` 固定并逐键核验；**Dataset/collate/DataLoader 已实现并通过 39 项数据契约检查**：B2 张量为 `float32[2,3,512,512]` 与 `int64[2,512,512]`（0/1/255、有效区 512×390、上下各 61 行 255），同步翻转与归一化均有逐值证据，真实 B2 eval 前向得 `[2,2,128,128]`。**尚未实现损失/指标/训练器，未训练，无性能指标**。

## 阶段 06 关键结论

- 数据集规模与身份：train 530 / val 132 / test 442；三份清单哈希与 `split_meta.json` 一致。
- 样本张量：`pixel_values float32 [2,3,512,512]`（CPU，min −1.9638 / max 2.6400），`labels int64 [2,512,512]`、唯一值 `{0,1,255}`。
- 几何：`1404×1070 → 512×390`，scale 0.364672，pad `L0 T61 R0 B61`，有效区 199 680 px；padding 共 124 928 个 255（2×122×512）。
- 标签语义：正常图有效区全 0、padding 全 255；异常087原ID0/3合并，512有效区前景313px非空。
- 同步翻转：`force` 路径下标签 `fliplr` 完全相等、图像 uint8 最大差 0；val/test均拒绝random与force。
- 归一化：与 `(x/255−mean)/std` 公式最大差 **0.0**；Dataset factory misses=1且共享同一实例（不含模型加载器），`do_resize=False`、`do_reduce_labels=False`。
- 原图 GT：collate 保留为 list（2 × `[1070,1404]` uint8，取值 `{0,1}`），不强行堆叠；train方向随增强，val/test为文件方向，B2 payload约3MB。
- 隔离：`__getitem__` 输出在 CPU，源码不含 CUDA/模型加载；真实 B2 eval+no_grad 前向得 `[2,2,128,128]` float32 有限，未创建优化器。

## 本轮验收

[阶段01—02验收与完善记录](../../reports/2026-10-31-stage01-02-acceptance.md)：修复跨 PowerShell 的空变量删除、资源根传递与恢复问题，收紧环境检查器，补充完整可移植依赖。修正后真实 CUDA／随机 SegFormer 前后向通过，缓存越界 0。

[第三阶段验收与完善记录](../../reports/2026-10-31-stage03-acceptance.md)：独立复算划分，核对 1205 个文件与归档成员哈希，修复写入/协议保护后全套 23 项测试通过。

[第四阶段验收与完善记录](../../reports/2026-10-31-stage04-acceptance.md)：原 18 张预览逐看；新版 15 张字节未变、另 3 张重新查看；登记正常图 3 个疑点；加入逐原域追踪和分组校验；完整审计 13 项及项目回归 62 项通过。当前证据见 `reports/stage04-acceptance.json`。

[第五阶段模型加载与核验报告](../../reports/2026-10-31-stage05-model-loading-and-check.md)：固定 commit 下载、逐键核对编码器、键分类、有效区 CE、梯度与一步更新、batch 1/2 显存、快照未被覆盖；当前项目回归101项、加载条件5项、运行条件9项通过；已归档小型model_source.json，按固定commit+全文件SHA换机恢复，真实快照留仓库外。

[第六阶段验收](../../reports/2026-10-31-stage06-acceptance.md)：39项数据契约、134项回归通过，9张图逐看；修复skip-model假通过和source GT展示，完善子集/processor/seed门槛。完整身份与代码证据见reports/stage06-acceptance.json；workers>0仍未验证。

## 下一步

阶段07：实现精确定义的CE+0.5前景Soft Dice、原图logits还原及int64累计指标，用手算数组/非对称几何夹具验证。完整指令见[第六阶段验收第7节](../../reports/2026-10-31-stage06-acceptance.md#7-下一步指令只执行阶段-07)。不启动训练。

## 维护约定

1. 同一阶段多次修正时，更新原文件并在第 10 节「修订记录」追加一行，注明哪些验证是修正后重跑的；不产生多份「最终版」。
2. 文档中的版本、路径、计数必须能在证据文件里找到，不允许出现「预计」「应该」当作结论。
3. 未运行的命令单独标注「待执行」，不与已执行结果混排。
