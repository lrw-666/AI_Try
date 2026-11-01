# 阶段 11：正式单基线、保存物验收与恢复边界

| 项目 | 当前记录 |
| --- | --- |
| 更新日期 | 2026-11-01 |
| 状态 | 正式训练与保存模型通过验收；恢复状态/数据顺序通过，续训数值复现未通过预设容差 |
| 正式 run | `E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42` |
| 协议 | PCB1 v2，530 train / 132 val，SegFormer-B0，seed42，50个完整epoch |
| 代码与数据身份 | 原训练源码在 `reports/stage11_evidence/recorded_source`；当前修复及全部SHA见 `reports/stage11-acceptance.json` |
| 范围 | 完成训练、按val选best、保存物复算和有界恢复对照；没有执行test推理 |

## 1. 本阶段回答什么

阶段10验证三轮完整数据流程，阶段11运行预先固定的50轮配方，产出可供独立推理的已训练best。训练和验证已完成；最终测试性能尚未产生。验证集用于选择epoch，本阶段数值不能称为test成绩。

原正式训练连续完成，没有使用resume。恢复代码和证据发现的问题不改变已完成的50轮权重，但必须修复未来恢复风险，并准确限定恢复保证。不能因为有checkpoint字段就把恢复写成“全通过”。

## 2. 技术栈与入口

现有专用环境：Python3.12.16、PyTorch2.10.0+cu126、transformers4.57.6、NumPy2.5.4，RTX4070 Laptop；本轮没有安装/升级包、改base或全局配置。

| 文件 | 职责 |
| --- | --- |
| `configs/baseline.yaml` | 正式50轮的完整显式配置，formal守卫 |
| `configs/resume_fixture.yaml` | 独立12图、3轮恢复排错配置，绝不继承给正式模型 |
| `manifests/resume_fixture_train_ids.txt` | 6正常+6异常，均为train成员 |
| `segformer_lab/train_full.py` | 正式/fixture控制流、完整epoch保存恢复、旧验证入口委托 |
| `segformer_lab/training.py` | 共享train_step、真实optimizer执行、AMP门槛、调度器 |
| `segformer_lab/full_check.py` | 只读核验正式best/last与完整132图；不训练、不评价test |
| `segformer_lab/resume_check.py` | 冻结epoch1状态/loader实测，以及真正epoch3 last模型对照 |
| `reports/stage11_evidence` | 原始日志无损压缩、小型状态记录和历史源码 |

## 3. 固定配方与调度算术

530 train=482正常+48异常，132 val=120正常+12异常；B2、累积1、workers0、drop_last=false，每轮265次batch尝试。全网络微调，训练随机水平翻转p=0.5，每轮set_epoch；val无增强、B1 IEEE FP32、原图GT、阈值0.5。

损失仍为CE+0.5正GT逐图前景Dice，eps1e-6、ignore255；两项数值计算FP32，正常图由CE监督。AdamW沿用bias/normalization不衰减分组、其余wd0.01、LR6e-5、clip1.0。没有增加裁块、采样权重、TTA或后处理。

```text
总尝试预算 = 50×265 = 13250
warmup = round(13250×0.05) = 662  # Python ties-to-even，不是663
cosine跨度 = 13250-662 = 12588
第1次成功更新LR = 6e-5/662 = 9.0634441e-8
第662次成功更新LR = 6e-5
无跳步时第13250次更新LR = 1e-6
```

1e-6是cosine终点，不是warmup下界。scheduler只按真实成功更新推进；本轮7次skip不补数据，所以成功数13243。最后一次实际更新使用LR约1.0000450e-6，更新后的下一步LR约1.0000331e-6；二者不是同一个时点。

第3轮已结束warmup并达到过6e-5，不能说“前三轮一直低于2.4e-5，第四轮才到base”。前三轮预测全空与学习率轨迹同时发生，但未做控制变量实验，不能把warmup认定为唯一原因。

## 4. 正式训练及验证结果

| 项目 | 已核准结果 |
| --- | --- |
| 完整epoch / train访次 | 50 / 26,500；每轮530个唯一ID，无重复或遗漏 |
| 尝试 / 成功 / 跳步 | 13,250 / 13,243 / 7；最长连续跳步1 |
| scheduler completed | 13,243，与成功数一致 |
| best | epoch39，按完整val全局前景IoU；不使用test |
| best的更新状态 | 10,330成功 / 10,335尝试 / 5跳步 |
| last的更新状态 | 13,243成功 / 13,250尝试 / 7跳步 |

best验证集结果：

| 指标 | 数值 |
| --- | ---: |
| 全局缺陷IoU / Dice | 0.777534 / 0.874846 |
| 全局precision / recall | 0.939779 / 0.818306 |
| TP / FP / FN / TN | 279246 / 17894 / 62003 / 197941817 |
| 异常图macro IoU / Dice | 0.539556 / 0.620312 |
| 正常图FP像素 | 1495 / 180273600，比例约8.293e-6 |
| 出现任意FP的正常图 | 11/120；这是像素误报诊断，不冒充另定义的图像级分类器 |

只有051预测全空，但这不代表只有一张漏检：042虽预测47像素却TP=0，IoU仍为0；081只命中67/6373个GT像素。大区域045/046较好，细缺陷仍是弱点。全局与macro回答不同问题，应并列报告。

50轮有更多候选epoch，但验证样本仍只有12张异常图。未计算置信区间或多seed稳定性，不能声称0.05波动都属于噪声。best/last差距小也不能证明两种选模策略等价；继续遵守预定best规则。

## 5. 保存物与独立推理验收

正式best SHA：`192c827bef00607f4258e5fadf0f1d97970ca5cc727a0add749be0fba5997485`。权重留在资源目录，不上传Git；跨机需另行复制已训练bundle并核哈希，重新下载MiT-B0不能替代训练后的best。

full_check先核原清单19项、配置/划分/快照身份和全部更新历史，再严格加载best、last、纯权重镜像；分别检查模型、优化器205个参数状态、各自step、scheduler、scaler、epoch/LR和RNG/loader结构。保存processor实际用于Dataset，不覆盖其设置。

真实新进程执行完整132图+固定6预览，共138次CUDA B1前向，FP32、eval/inference_mode、关闭autocast/TF32。所有逐图、macro、normal、global与NA字段一致；6预览概率最大差0.0、XOR0。原run21个文件的大小和SHA前后未变。

新完整验收器也拒绝NaN/Inf、概率shape/key差异、跨阈值XOR和状态/计数不一致。旧验证入口只gate全局分数，已委托到完整验收器；原报告保留为历史证据。

## 6. 恢复实现的修复

- 同run判定改为完整解析路径，不能用相同basename绕过；只从last完整epoch恢复。
- 日志过滤resume事件，避免第二次resume读取事件行后KeyError；未提交尾日志、部分JSON或与last不一致时保留现场并拒绝，不假称任意崩溃都能自动修复。
- 保存processor绑定实际train/val/preview Dataset，恢复不再回用外部snapshot的预处理。
- 身份核对覆盖实际快照文件、source、savedprocessor、三CSV、split_meta、train/eval/preview IDs、完整解析配置与代码版本；身份或源码改变时拒绝恢复。
- 模型配置用副本构造，避免Transformers原地加字段。optimizer所有slot、moments、step、分组和形状必须完整且有限；只保存部分状态不能静默继续。
- 结构重建完成后再恢复Python/NumPy/Torch/CUDA RNG及真实loader generator；保存模型镜像采用原子替换。
- 恢复后的清单排除自身，内嵌清单排除旧summary，避免自引用SHA和旧文件指纹过期。

当前修复代码会拒绝旧checkpoint缺失的新身份字段或旧源码SHA；这是保护，不应删除守卫强行续训。原正式run已完成，仍可只读加载best推理；没有把新的恢复代码伪装成原50轮实现。

## 7. 恢复证据：原报告不能按字面使用

旧A/B/D/E对照存在三个关键范围问题：NPZ来自epoch1 best而非epoch3 last；所谓加载状态读取了已覆盖的B epoch3 last；对Python/NumPy RNG和实际loader generator的核验不完整。A/B在边界之前的前缀也已不同。旧报告的硬件唯一成因解释和“全恢复通过”结论已撤回，原件与脚本保存在归档中。

本轮重做了有界、独立的12图对照，只有36次更新：A完整18次；B先6次后冻结run外epoch1-last.pt，独立复验，再续12次。没有重训正式50轮。

| 新对照 | 真实结果 |
| --- | --- |
| 冻结边界 | epoch1/next2、6次成功更新；完整checkpoint SHA已记录，继续训练不会覆盖它 |
| 加载瞬间 | 208个模型状态、optimizer/groups、scaler/scheduler/LR、step/best、全部四类RNG及真实loader generator均精确相等 |
| 下一轮loader | 真实解码6批，数据顺序与flip逐批一致 |
| 续训后的状态结构 | 两run均epoch3 last，计数/LR/scheduler及12条后续update的顺序/flip一致 |
| 最终模型最大差 | 约4.71834e-4；未声称逐位一致 |
| 真正last概率对照 | 最大差1.5032291e-4，XOR33，**超过预声明atol1e-5且XOR0的门槛** |

因此，本轮确认的是**完整状态加载与数据顺序恢复**，不是续训轨迹的数值复现。端到端数值对照未通过，不放宽容差、不把失败改名为通过。A/B的首轮日志也有差异，因此本对照不能单独定位边界或某个GPU kernel为偏差原因；严格确定性诊断尚未完成。

此限制不影响本次连续完成的正式模型用于独立测试。以后需要中断训练时，仅适用于日志与last一致的完整epoch边界；不能保证得到与不中断训练按位或在1e-5内相同的最终模型。

## 8. 本轮验证与复现

完整项目回归 **263项通过，11.006s**。原始正式数据审计见[独立报告](../../reports/stage11-evidence-audit.json)，当前保存物复验见[完整模型检查](../../reports/stage11-reload-check.json)，新边界与last对照见[恢复汇总](../../reports/stage11-resume-fixture.json)。

```powershell
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @(
  '-m','segformer_lab.train_full','--verify-run',
  'E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42',
  '--config','configs/baseline.yaml','--report','reports/stage11-reload-check.json',
  '--device','cuda'
)
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @('-m','unittest','discover','-v')
```

上面是只读保存物复核，不启动训练。大权重、完整概率与批量图片在资源目录，小型日志用无损gzip进入仓库，解压方法见[原件说明](../../reports/stage11_evidence/README.md)。本轮无包安装或全局环境改动。

## 9. 学习自测

1. 为什么best来自39轮，而最后一轮仍要保存last？两个文件各用于什么？
2. 为什么末尾“实际使用的LR”与“更新后已准备的LR”不同？跳步如何影响它们？
3. 全空预测数为1，为什么仍不能说只漏检1张？请对比042/051/081。
4. 比较best的NPZ，能否证明last续训后的概率一致？为什么必须保留不可变epoch1检查点？
5. exact state restore、exact data order、numeric continuation分别需要什么证据？

## 10. 下一步与修订

下一步按[阶段12执行交接](../../reports/2026-11-01-stage12-execution-handoff.md)：冻结best39和协议，新进程先复算val，再进行独立test442张评价与单图推理，不训练/调阈值/换best。可为排错和复现重复同一冻结协议，不能依据test择优改模型；“测试只能运行一次”不能禁止必要工程复核。

| 日期 | 修订 | 证据 |
| --- | --- | --- |
| 2026-11-01 初版 | 完成50轮与原恢复对照 | 原正式run和旧fixture原件 |
| 2026-11-01 独立验收 | 完整保存物复验、恢复安全修复；撤回旧恢复范围误述；新冻结边界和真正last对照，数值失败如实保留 | 263项回归、CUDA完整val复算、新恢复报告 |
