# 阶段 12：独立推理、原图还原与最终测试执行交接

日期：2026-11-01。本文只交接下一阶段，**本轮未实施阶段 12、未执行 test**。先以第 11 阶段最终验收记录为准；验收通过后再执行本文。只读检查已确认实际 `best.pt` SHA-256 与阶段 11 摘要一致。

后续授权更新：用户已要求原开发Agent连续完成阶段12—13，每阶段完成后分别提交并推送，最终独立验收待用户回来进行。现在执行时以[统一交接](./2026-11-01-stage12-13-continuous-execution-handoff.md)为总指令；它覆盖本文末尾“不自动commit/push或进入13”的旧边界，其余技术规格保持有效。此更新不表示已执行阶段12。

## 1. 已冻结的输入与边界

- 项目：`E:\桌面\AI\AI_Try\work\SegFormer`；资源：`E:\桌面\AI\segformer_resources`；沿用独立环境与项目 runner，不安装或升级包，不改 base、全局配置或持久环境变量。
- 训练 run：`runs\pcb1_b0_v2_seed42`，50 轮；仅按 **val 全局前景 IoU 最大、并列取较早轮次**选得 best epoch **39**、成功 step **10330**。本阶段不重新选择权重。
- best：`E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42\checkpoints\best.pt`。
- best SHA-256：`192c827bef00607f4258e5fadf0f1d97970ca5cc727a0add749be0fba5997485`（本轮通过 `Get-FileHash` 实核）。
- 使用 run 内 `model_config.json`、checkpoint 的 `model_config`/`model`、`processor/preprocessor_config.json` 和已保存协议；模型结构须相互核对。`model_state.pt` 若使用，须与 best 逐张量一致。
- 固定 PCB1 v2 分组、seed42、512 letterbox、背景0/缺陷1/padding ignore255、阈值 **0.5**、B1、**IEEE FP32**（关闭 TF32/autocast）、无增强/后处理/TTA。原损失定义保持不变，不训练、不 resume。
- 先完整 val **132=120正常+12异常**，通过一致性门禁后才读取 test 图像/GT；最终 test **442=402正常+40异常**，不得变成预览子集。
- 原索引中“test 只能评一次”应更正为：**协议冻结后允许必要技术复核或断点重跑**；必须同 checkpoint、同数据/代码/协议，记录原因、命令、失败和全部复核结果，不按 test 分数择优。

## 2. 可直接复制给实施 Agent 的指令

```text
请只完成 SegFormer 主指南阶段12：独立推理、原图还原与最终test评价。
先读根/项目AGENTS、主指南步骤12、环境隔离companion及阶段模板、阶段11最终验收、
docs/stages/11-training-design-and-implementation.md、阶段索引、阶段11执行交接、
configs/baseline.yaml、reports/stage11-summary.json及run内summary/resolved_config/身份清单。
先确认阶段11最终验收通过；若未通过，处理当前验收问题，不提前读取test图像/GT。
查实际代码再实施：当前不存在segformer_lab/evaluate.py、predict.py；evaluation.py只是
精度上下文，不能把它当独立评价CLI。新增专用evaluate/predict入口，共享现有
data/geometry/metrics.restore_defect_probability/evaluation数学实现，不另造一套还原与指标。
不调用train/pilot/train_full的训练或resume分支，不执行反向传播或optimizer.step；
只读验收器可构建优化器核对保存状态，不修改已有训练产物。

新进程从run内模型配置构建SegformerForSemanticSegmentation，再strict=True载入best模型状态；
missing/unexpected必须均为0。加载run内保存processor，local_files_only=True，使用其真实
归一化/缩放参数，关闭processor二次resize与reduce_labels。不用预训练目录重新搭建权重，
不从nvidia/mit-b0重新下载来替代已训练best。记录best、结构、processor的SHA与加载证据。
固定model.eval()+inference_mode、B1、IEEE FP32、workers0、flip=none、阈值>=0.5。
所有接口共用：低分辨率logits双线性至512(align_corners=False)→按几何记录裁padding
→双线性恢复原H/W→通道softmax取缺陷概率→阈值化。用原始GT非零合并的0/1标签计分，
正常图建立原尺寸全0 GT；不能把512 GT放大回去当原图GT，也不把原标注255误作padding。

先用完整132张val复算，不用val_preview_ids，不改阈值或放宽协议来凑一致。
对照训练run summary中epoch39的val（仓库stage11-summary仅是便携摘要）：逐image_id核对
132行计数/尺寸/GT与预测面积/TPFPFNTN/IoU Dice P R/NA；同时核对异常12图macro及其
有效分母、normal统计、global和所有NA原因。整数与NA结构要求相等，浮点使用预先声明
的容差并逐项报告差异。先核对阶段11最终验收实际提供的完整复验入口及--help，再复用它；
历史旧do_verify仅核confusion/global等且覆写原run报告，不能沿用旧exit0作为完整对照证据。
固定6张val预览还要与run内pre_save_probabilities.npz比较概率/二值XOR；报告实际差值，
概率atol=1e-5沿用旧复验，不能据test调整容差。同输入evaluate/predict也须概率一致、
原尺寸掩码一致；predict不读取GT或标签。所有新报告写新评价目录，保留原run只读。
val任何关键项不符先停在val定位，不进入test、不继续训练。门禁通过后冻结代码与身份清单。

完整test逐图处理并立即计数/写PNG/释放概率；默认不保存全test float概率图，
不能keep_probabilities=True累积全部442图，也不把全量logits/原图/GT常驻RAM。
共享流式逐图回调或等价接口完成写盘，累计int64 TP/FP/FN/TN后只除一次，防止溢出。
输出442行per_image.json和per_image.csv，保留image_id/normal或anomaly/原H,W/GT与预测面积/
TPFPFNTN/IoU Dice P R/NA原因/预测文件路径；清单顺序可追溯，漏图/重复为0。
metrics.json报全局前景IoU、Dice、P、R及混淆量，异常40图macro四指标及各自有效图数/NA。
normal单独报402图的FP像素总数/总像素比、每图预测前景ratio及median/P95/max、
有任意FP的图数及分母；这不是“图像级检出率”，本阶段没有定义图像级判定规则。
零分母写null/NA并记录原因，指标不加eps，不把正常空GT且空预测的IoU/Dice记为1，
macro只对有定义项计算并公开分母，不以正常样本冲淡异常macro。

输出原尺寸0/1二值PNG（本数据应为宽1404×高1070），明确0背景/1缺陷；如另存便于
显示的0/255版本，须独立目录并写明它只是显示约定，读取计分时非零转换为1。
单图predict只需--image，不要求mask/label；至少输出原尺寸二值PNG、热力图和叠加图，
可按需保存单图float32概率，概率热力图不可冒充真实浮点概率。复用同一预处理和还原函数。
展示规则在读test结果前写入元数据：异常IoU最低/recall最低、正常FP像素最多各取前k，
并列按image_id；异常全漏和正常有FP的计数及完整ID列表全保留。列出规则、k、选中ID，
允许补充固定样本，不能只挑漂亮图；所有442行原始统计必须保存，详细归因留阶段13。
042/051/081是val上已知漏检缺陷，记录为既有局限，不根据test再改阈值、轮次或后处理。
结论仅限单模型/单seed/PCB1该协议，不宣称跨类别、跨产线或工业泛化。

写evaluation_meta.json与artifact_manifest.json：命令/进程/解释器和实际包版本/设备/时间、
原run和best epoch/SHA、配置及resolved配置SHA、模型结构/processor各文件SHA、
源代码逐文件SHA和Git HEAD/未提交差异、官方CSV及train/val/test/split_meta SHA、
所用图像/GT原文件SHA（可复用已验证数据审计清单并校验）、阈值/还原/NA/展示规则、
输入输出相对路径/大小/SHA、val门禁与技术重跑原因。输入保存物只读，新输出目录拒绝覆盖。
不修改test annotations，不依test结果调参；必要技术复核保留各次结果，不能挑成绩最好的一次。
CPU/GPU性能比较非本阶段必做；若测延迟，固定B1、预热并CUDA同步，分别声明前向与
端到端范围及中位/P95，不把初始化/异步返回/读写时间混成稳定GPU延迟。
按模板交付中文docs/stages/12-evaluation-design-and-implementation.md，更新阶段索引和
项目README；Git只保留代码、配置、小报告、精选图，权重/全量预测等留资源根。未执行写未执行。
换机须单独复制已训练best、结构及processor并校验SHA；Git没有权重，不能靠重下mit-b0替代。
完成后返回实现文件、真实命令、val门禁、test全量结果、产物和局限；不自动commit/push或进入13。
```

## 3. 待实现的命令契约示例（本轮均未执行）

以下参数是给新增入口的实现约定，**当前不能直接运行**。先实现并核对 `--help`，再按实有参数执行；test 入口必须验证 `--val-gate` 与同一 best/协议/数据身份绑定且已通过。

```powershell
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
$SegTrainedRun = 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42'
$SegEvalRun = 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42_eval_001'
if (Test-Path -LiteralPath $SegEvalRun) { throw '评价目录已存在，另取新目录，不覆盖。' }
& '.\scripts\run_project.ps1' -PythonArgs @('-m','segformer_lab.evaluate','--help')
& '.\scripts\run_project.ps1' -PythonArgs @('-m','segformer_lab.predict','--help')
& '.\scripts\run_project.ps1' -PythonArgs @('-m','segformer_lab.evaluate','--config','configs/baseline.yaml','--run',$SegTrainedRun,'--split','val','--out',"$SegEvalRun\val")
if ($LASTEXITCODE -ne 0) { throw '独立val未通过，停止。' }
# 仅在val/gate.json明确通过、完整对照已核验后执行下一行；不能仅看进程exit0。
& '.\scripts\run_project.ps1' -PythonArgs @('-m','segformer_lab.evaluate','--config','configs/baseline.yaml','--run',$SegTrainedRun,'--split','test','--val-gate',"$SegEvalRun\val\gate.json",'--out',"$SegEvalRun\test")
if ($LASTEXITCODE -ne 0) { throw 'test技术失败，保留现场与原因，按冻结协议复核。' }
& '.\scripts\run_project.ps1' -PythonArgs @('-m','segformer_lab.predict','--config','configs/baseline.yaml','--run',$SegTrainedRun,'--image','E:\桌面\AI\segformer_resources\data\visa\pcb1\Data\Images\Anomaly\045.JPG','--out',"$SegEvalRun\single_val_045")
```

`045.JPG` 是固定 val 示例，单图输出不能称为独立 test 成绩。上述 `--run` 须强制从该目录加载已冻结 best/结构/processor；若另设 `--checkpoint`，也须校验相同 SHA。

## 4. 验收时对照

val 参考：TP **279246** / FP **17894** / FN **62003** / TN **197941817**；global IoU **0.7775342969**、Dice **0.8748459012**、P **0.9397792286**、R **0.8183056947**。正常120图FP **1495 px**、有FP **11图**。这些是已有 **val** 证据，不是 test 成绩；完整逐图/macro/NA以同一run的epoch39保存记录为准。

最终验收必须同时看到132图完整val对照通过、442图全量统计、40异常macro与402正常误报、原尺寸预测与单图一致性、输入输出身份清单和中文阶段文档；没有实际运行证据不得标为完成。
