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
| 05 | 预训练权重与模型搭建 | 未开始 | — | — |
| 06 | Dataset、变换与配置 | 未开始 | — | — |
| 07 | 损失与指标 | 未开始 | — | — |
| 08 | 20 步冒烟检查 | 未开始 | — | — |
| 09 | 8 张训练图过拟合检查 | 未开始 | — | — |
| 10 | 完整训练集短程试跑 | 未开始 | — | — |
| 11 | 正式训练与可靠保存 | 未开始 | — | — |
| 12 | 独立推理与原图评价 | 未开始 | — | — |
| 13 | 结果可视化、失败分析与实验卡 | 未开始 | — | — |

## 当前状态一句话

PCB1 的完整解码、固定可视化和 v2 分组协议已通过验收；0042/0043 均在 train，官方 test 未变。512 下没有整张前景变空，但有 19 个微小原始连通域消失，已登记限制。**尚未下载预训练权重、未搭 Dataset、未训练，无性能指标**。

## 阶段 04 关键结论

- 图像 1104/1104 完整解码，全部 `1404×1070`、`RGB`；掩码 100/100 完整解码，全部单通道 `L`、尺寸一致。
- 掩码原始取值并集 `{0,1,2,3,4}`，共 9 种组合；最常见 `{0,2}`（46 张）。
- 异常前景占比中位数：train 0.188%、val 0.316%、test 0.194%（最大 12.73%）。
- 512 letterbox：`512×390`，上下各 61 px（23.83%）；前景比例保持率中位 99.94%，但 328 个原始 4 连通域中 19 个消失，涉及 12 张、总原面积 39 px。原图评价不能恢复训练时丢失的细节。
- 精确重复 0 组；0042/0043 按高可信同实物风险组完整重出 v2，官方 test 字节不变；当前跨集候选 0 对。旧版资料保留，不能混用。
- 固定预览：`runs\data_preview\{train,val}`，12 + 6 张五联图 + 2 张总览图；`manifests\val_preview_ids.txt` 固定了 3 正常 + 3 异常验证样例。

## 本轮验收

[阶段01—02验收与完善记录](../../reports/2026-10-31-stage01-02-acceptance.md)：修复跨 PowerShell 的空变量删除、资源根传递与恢复问题，收紧环境检查器，补充完整可移植依赖。修正后真实 CUDA／随机 SegFormer 前后向通过，缓存越界 0。

[第三阶段验收与完善记录](../../reports/2026-10-31-stage03-acceptance.md)：独立复算划分，核对 1205 个文件与归档成员哈希，修复写入/协议保护后全套 23 项测试通过。

[第四阶段验收与完善记录](../../reports/2026-10-31-stage04-acceptance.md)：原 18 张预览逐看；新版 15 张字节未变、另 3 张重新查看；登记正常图 3 个疑点；加入逐原域追踪和分组校验；完整审计 13 项及项目回归 62 项通过。当前证据见 `reports/stage04-acceptance.json`。

## 下一步

步骤 5：取得固定 revision 的 `nvidia/mit-b0`，搭建二类 SegFormer-B0，核验预训练 encoder 真加载、分类／解码头初始化和实际梯度、参数变化。人工张量调试不覆盖原始快照；不开始正式训练。可复制指令见[第四阶段验收报告第 7 节](../../reports/2026-10-31-stage04-acceptance.md#7-下一步指令只执行阶段-05)。

## 维护约定

1. 同一阶段多次修正时，更新原文件并在第 10 节「修订记录」追加一行，注明哪些验证是修正后重跑的；不产生多份「最终版」。
2. 文档中的版本、路径、计数必须能在证据文件里找到，不允许出现「预计」「应该」当作结论。
3. 未运行的命令单独标注「待执行」，不与已执行结果混排。
