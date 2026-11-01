# 阶段 11：正式单基线训练、监控与恢复执行交接

日期：2026-11-01。本文是待执行指令，**没有实现正式入口、验证续训或启动 50 轮训练**。执行以[阶段 10 最终验收](./2026-11-01-stage10-acceptance.md)结论为前提；先读该记录及主指南步骤 11。

## 1. 目标与边界

阶段 11 只完成固定 PCB1 v2 划分上的 **SegFormer-B0、seed 42、50 个完整 epoch 单基线**，每轮全量 val 选权重，交付可恢复 checkpoint 与中文学习文档。结束返回人工验收；不自动进入阶段 12 或测试评价，不自动提交/推送。

阶段 10 的三轮结果仅作流程证据：val 全局 IoU 约 0.6081，但异常图 macro IoU 约 0.2207，12 张异常图中 7 张预测前景为空。全局指标受大缺陷面积影响，不能把 0.608 当达标线，也不能保证 50 轮必然改善。阶段 11 不因这些读数改损失、阈值或采样协议；后续超参数变体须另建配置/run，与本次固定基线分开报告。

## 2. 可直接复制给实施 Agent 的指令

```text
请只实现并执行 SegFormer 主指南的阶段11：50轮正式单基线、逐轮验证与可靠恢复。
先读AI_Try/AGENTS.md、work/SegFormer/AGENTS.md、主指南步骤11、隔离规范、阶段索引、
阶段10最终验收/学习文档/执行交接、configs/baseline.yaml和pilot.yaml及实际代码。
只在阶段10验收通过后执行本阶段；已有用户逐阶段人工验收安排，不另加反复确认门禁。
保持现独立Conda环境与资源根；新进程用项目runner，保留base和全局配置，不覆盖旧run。

先解决入口边界：train.py仍是阶段09的8图overfit；pilot.py是阶段10入口、无resume，
目前也未把epochs严格限制为3，不能直接换成baseline.yaml就称为正式训练。
新增train_full.py，或实现有清晰守卫、独立stage/mode/schema的正式模式；共享已有
data/loss/metrics/evaluation/train_step/optimizer/scheduler，不复制另一套数学实现。
保留并补强阶段08/09/10的原边界，pilot仍限全量530/132、3轮原协议；回归确认兼容。

完善正式baseline配置并写出resolved_config：不得靠隐含defaults补关键协议。
显式写明seed42、train_ids_file=null、train530(482正常+48异常)、val132(120正常+12异常)、
50epoch、max_steps=null、B2、accumulation1、workers0、drop_last=false、shuffle=true，
CUDA FP16 AMP、全网络微调；训练flip=random/p=0.5，每轮set_epoch(epoch)。
precision=fp16_amp、lr_scheduler=warmup_cosine等入口字段须与实际行为一致并受校验。
workers0就是本轮可用设置；不强制先测workers2/4，不把性能优化作为正式基线前置任务。
保持原图分组清单与固定MiT-B0快照/revision，512 letterbox及padding/processor协议不改；
掩码>0合并为缺陷1、背景0、padding ignore255，图/标签同步翻转。不开裁块、加权采样、
复杂增强、后处理、TTA或DINO，不因pilot表现改变原始数据、标签、损失或指标协议。
CE+0.5正GT逐图前景Dice、eps1e-6、Dice FP32照旧；CE按有效像素、Dice按正GT图数
加权汇总，保留分子分母，不能以batch均值之均值代替正式epoch loss。

正式新run必须从固定原预训练编码器+新二类头重新初始化model/optimizer/scaler，
不能接pilot/smoke/overfit checkpoint或优化器；记录初始化与编码器逐键核对证据。
AdamW沿用原分组：LR6e-5、wd0.01、bias/归一化不衰减、grad clip1.0。
计划每轮ceil(530/2)=265个batch；50轮最多13250个batch尝试，正常完成恰为13250，
不因AMP跳步重复数据补成功数；计划更新预算50*265=13250，warmup_ratio=0.05。
沿用现WarmupCosineLR：max(1,int(round(13250*0.05)))=662；Python round的ties-to-even
使662.5取662，并非一律四舍五入向上。cosine跨度=12588；不可沿用pilot的795/40计划。
completed=k表示已完成的成功更新数；第k+1次更新前LR先apply并记录。
warmup时LR=6e-5*(k+1)/662，首步约9.06344e-8，第662次到6e-5；1e-6只为cosine
终点，不是warmup下界，不要擅自clip。后续progress=(k-662+1)/12588；无跳步时第
13250次更新用1e-6。scheduler仅在optimizer确实更新后推进，AMP跳步不推进/不补样本。
按50个完整epoch遍历，逐轮核实530张各一次；分别记录attempt/success/skip及AMP scale。
沿用已验收的有限跳步策略：连续3次AMP skip、非有限loss/参数或范数溢出应停下保留现场；
不要为凑13250次成功更新多遍历数据。若发生跳步，末LR可能未到floor，真实报告即可。

每轮完整132张val：B1、无增强、IEEE FP32、关闭TF32/autocast，按既有原图还原函数
恢复logits后softmax，固定阈值0.5；GT使用原尺寸二值掩码，ignore区域不参与训练指标。
只按val原图全局前景IoU选best，初始负无穷，并列取较早轮次；early_stopping=false。
同时报告全局IoU/Dice/P/R及TP/FP/FN/TN、异常macro IoU/Dice/P/R、132张逐图指标、
NA原因、正常图FP像素比与有任意FP的图数，不以全局IoU替代这些诊断。
逐轮记录12张异常图预测为空的数量与ID，关注pilot漏检的019/033/042/051/081/087/090；
这些val诊断不用于本轮换损失/阈值。只有12张异常val，结论须说明样本量与面积权重限制。
最终test始终隔离：test清单只允许哈希核对，不解码test图像/掩码、不推理、不选参报分。

每轮原子保存完整last，刷新best另存完整best，保存结构/processor与独立推理权重。
checkpoint含model、optimizer及分组、GradScaler、scheduler全部状态/总预算、完成epoch、
下一epoch、真实step/attempt/skip、best分数/epoch、当轮metrics、解析配置与数据/代码/
预训练身份；Python/NumPy/Torch/CUDA RNG和DataLoader generator.get_state()都必须保存。
现pilot只保存全局RNG，不能据此声称续训完整。恢复校验身份、设备状态与LR后再迭代loader，
恢复loader generator/set_epoch/模型train状态；避免模型构造等操作额外消耗已恢复的RNG。
普通resume只支持完整epoch边界，从last的下一epoch续到原50轮，保留13250/662计划，
不能按剩余epoch重建schedule。中断记录最后完整epoch；未完成epoch不冒充可恢复边界。

正式50轮前先实现并实测epoch边界恢复：独立resume_fixture与正式run完全分开，允许
小固定train子集+短预算。相同初始化/配置分别跑不中断对照和保存后新进程续训，至少跨
一个epoch边界；fixture配置/日志标记resume_fixture，不作基线结果、不继承给正式run。
比较加载瞬间model/optimizer/scaler/scheduler/RNG/generator、step/attempt/skip与当前LR，
并比较下一epoch的shuffle顺序、flip图案、若干训练更新以及固定val概率/指标。
状态、计数、LR和数据顺序必须正确；同环境可选deterministic模式检查exact resume。
数值比较的参数/概率/指标绝对相对容差须在对照前明确；若不能逐位一致，解释来源并验证，
如实标注恢复保证范围；不得把明显顺序错误或只加载权重写成resume已验证。
必要的短恢复校验和受影响回归通过后直接启动本阶段50轮，不反复小跑按val盲调配方，
也不因可选吞吐实验或未承诺的跨平台位级一致无限推迟正式训练。

资源输出用新pcb1_b0_v2_seed42正式run；fixture另建目录；普通新run拒绝覆盖，只有显式
resume可向匹配run续写，不能清空旧日志。先检查可用磁盘/内存与实际环境，记录真实版本。
验证流式逐图累加，不缓存全132张原尺寸概率；仅固定6张预览及少量必要失败图保留概率。
记录初始化/数据等待/训练step/全量val/出图/保存的耗时，GPU allocated/reserved峰值和
主机RAM，保存逐轮loss/LR/指标曲线；保持现稳定有限性检查，不为提速先删安全检查。
冻结起止代码/配置/清单SHA、预训练revision、Git HEAD和未提交diff；新增未跟踪源文件
也须哈希/归档。大权重/全量日志/批量图在资源目录；仓库仅代码配置、少量精选图和small JSON。

交付docs/stages/11-training-design-and-implementation.md中文文档，按模板讲设计/实际版本/
代码片段/准确命令/真实结果/恢复保证/局限/学习题；更新README和docs/stages/README.md。
保留正式逐轮及逐次更新日志、best/last清单与SHA、恢复对照证据、全量val逐图结果、
精选预览/曲线和reports/stage11-summary.json等小型证据。计划与已执行严格区分。
完成50轮或出现不能解释的故障后停止，报告实际完成epoch、best来源与下一步；不自动测试。
阶段12的新进程独立推理/最终test评价须之后单独获得阶段执行授权。
```

## 3. 命令接口与验收证据

以下是**待实现接口示例，尚未运行**；实施 Agent 应以实际 `--help` 核对，并把最终可复现命令写进阶段 11 文档。若选择显式 mode，替换成其真实接口，不能照搬主指南中过拟合入口的旧命令。

```powershell
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
$env:PSExecutionPolicyPreference = 'Bypass'
. '.\scripts\enter_project.ps1'
Invoke-SegPython -m segformer_lab.train_full --help
# 先完成独立resume_fixture恢复对照；通过后执行正式全新run：
Invoke-SegPython -m segformer_lab.train_full --config 'configs\baseline.yaml' --out 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42'
# 仅在正式run实际中断后，显式从其最后完整epoch恢复：
Invoke-SegPython -m segformer_lab.train_full --config 'configs\baseline.yaml' --resume 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42\checkpoints\last.pt' --out 'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42'
```

| 验收重点 | 必须看到的证据 |
| --- | --- |
| 正式身份与预算 | 独立新run、原预训练初始化、完整显式配置、530/132覆盖、50轮实际状态、13250/662计划与真实更新数 |
| 完整恢复 | fixture不中断/新进程续训对照；完整状态和loader RNG；下一epoch顺序/翻转/step/LR一致及数值容差结果 |
| 模型选择与诊断 | 每轮全132张val；best只按全局IoU；macro、逐图、正常FP、异常空预测及NA齐全 |
| 可追溯交付 | best/last及身份SHA、资源/耗时实测、流式val、精选图、小型JSON、中文阶段文档与两处索引更新 |

停下条件是数据/身份不一致、恢复关键状态或顺序错误、非有限异常/连续跳步、资源不足，或正式预算完成；保留失败现场与最后完整 epoch。验证分数低或未超过 pilot 不单独触发改配方。用户验收针对真实完成情况与协议遵守，不承诺效果改善。
