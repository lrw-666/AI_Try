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
| 03 | 引入 PCB1、固定划分与标签规则 | 已通过本轮独立验收 | [03-data-preparation-design-and-implementation.md](./03-data-preparation-design-and-implementation.md) | `manifests\{train,val,test}.csv` 与 `split_meta.json`；`segformer_resources\data\visa\extraction_manifest.json`；`audit\stage03-*.log`、`audit\stage03-verification.txt`；本轮修复后的复核待完成 |
| 04 | 数据审计与可视化 | 未开始 | — | — |
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

PCB1 数据已导入并固定划分（train 530 / val 132 / test 442），三集合互斥、测试集与官方一致，1104 张图与 100 张掩码的文件头可解析、尺寸可读取，清单哈希已记录；**阶段 03 本轮验收与完善进行中，阶段 04 尚未开始，完整解码与像素级审计待做，未下载预训练权重，未训练**。

## 阶段 03 关键结论

- 官方 CSV 校验值一致（`c3331eed…2f09`），官方 pcb1 为 train 662（602 正常 / 60 异常）、test 442（402 / 40）。
- 派生清单 530 / 132 / 442；`train∩val`、`train∩test`、`val∩test` 均为 0；test 与官方 test 的行内容集合一致。
- 1104 张图的文件头均可解析、尺寸均为 `1404×1070`；100 张掩码文件头可解析、模式为单通道、尺寸与图像一致。少量掩码像素抽查得到 `{0,1}`、`{0,2}`（多类 ID，训练时按「非零→1」转换）；全量完整解码与取值统计留阶段 04。
- 重复运行：`files=0 / verified=1205`，三份 CSV 的 SHA-256 不变；原 TAR/ZIP 未被修改。

## 本轮验收

[阶段01—02验收与完善记录](../../reports/2026-10-31-stage01-02-acceptance.md)：修复跨PowerShell的空变量删除、资源根传递与恢复问题，收紧环境检查器，补充完整可移植依赖。修正后真实CUDA/随机SegFormer前后向通过，缓存越界0。

阶段 03 本轮验收正在修复提取临时文件、协议参数与清单发布保护，并复核原始数据和划分身份；最终验收结论待修复后的回归与真实数据检查完成。阶段 04 保持未开始。

[第三阶段验收与完善记录](../../reports/2026-10-31-stage03-acceptance.md)：独立复算划分，核对1205个文件与归档成员哈希，修复写入/协议保护后全套23项测试通过。阶段04仍未开始。

## 下一步

步骤 4：数据审计与可视化。读取三份清单，做图像解码审计（坏图、重复/近重复）、掩码取值分布统计、512 最近邻缩放后的前景保留率，并生成原图+掩码叠加图供人工检查；仍不训练、不下载权重。

## 维护约定

1. 同一阶段多次修正时，更新原文件并在第 10 节「修订记录」追加一行，注明哪些验证是修正后重跑的；不产生多份「最终版」。
2. 文档中的版本、路径、计数必须能在证据文件里找到，不允许出现「预计」「应该」当作结论。
3. 未运行的命令单独标注「待执行」，不与已执行结果混排。
