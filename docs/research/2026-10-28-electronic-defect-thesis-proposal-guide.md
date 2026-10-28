# 工业电子元器件视觉缺陷检测：开题路线建议与学习计划

更新日期：2026 年 10 月 28 日。开题答辩：2026 年 11 月 16—20 日；材料按 11 月 16 日提交、11 月 15 日前定稿安排。

本文结合项目文档、你在本次交流中的补充，以及论文原文、作者代码和官方资料编写。论文与代码链接已做资料核查；本轮没有安装训练环境、训练模型或验证论文成绩。下文的难度、学习顺序和工作量属于实施建议。

## 1. 先看结论：推荐怎样选

你的目标可以表述为：**在导师认可的电子工业视觉问题上，使用公开数据和成熟代码，用 Python/PyTorch 完成一项范围适中的硕士研究；在过程中掌握可迁移到视觉岗位的数据、训练、评测和推理能力，并深入理解一种现代视觉架构。**

结合“毕业优先、希望多学、创新要求适度、对未来就业有帮助”这几个条件，我的推荐顺序是：

| 优先级 | 研究路线 | 起步与主模型 | 采用条件 |
| --- | --- | --- | --- |
| **首选** | 有监督表面／外观缺陷分割 | **U-Net 基线 → SegFormer-B0** | 导师接受用已标注缺陷训练，并认可所选对象；VisA PCB 类可使用官方 high-shot 协议 |
| **备选** | 仅正常样本训练的异常检测与定位 | **PatchCore 基线 → Dinomaly**；EfficientAD 可作较轻的替代 | 导师强调缺陷稀缺、未知缺陷，或希望只用正常样本训练 |
| 条件选择 | 缺陷目标检测 | 成熟 YOLO 实现起步，之后按需了解 RT-DETR | 需要框和类别，且有合适框标注；使用 DeepPCB 时须接受“裸 PCB 线路缺陷”这一对象 |
| 拓展学习 | 扩散重建或缺陷合成 | DDPM 概念实验 → DiffusionAD 单类别实验 | 主线稳定后再学；有明确兴趣和实验依据时，也可以替代主线 |

**如果现在让我替你带一个方案见导师，我会优先带“U-Net → SegFormer 的有监督缺陷分割”，同时准备“PatchCore → Dinomaly 的正常样本异常检测”作为备选。**前者更直接地覆盖完整监督训练流程，也能自然学习 CNN、注意力和 Transformer；后者更贴近工业缺陷样本难收集的设定。

这与最初只考虑异常检测的建议有所调整：你已经明确，学习通用技术和未来求职也是重要目标，而 VisA 确实提供允许使用缺陷训练的公开协议。没有必要为了遵循 normal-only 这一研究设定，放弃已有的像素标注。

当前最需要导师确认的两个问题是：**对象能否包括 PCB 及板上元件；研究是否允许使用真实缺陷及其掩码训练。**这两个答案比先选哪个网络更影响最终方案。

## 2. 你的现状，以及本次如何使用项目资料

### 2.1 已确认的条件

- 你已完成此前整理的 5 篇综述阅读，对两个候选数据集已有初步了解。
- 导师建议的方向是“基于机器视觉的工业电子元器件表面瑕疵检测”；是否包括板级外观和装配异常，尚待讨论。
- 你是在职非全日制学生，没有小论文发表指标；学校目前没有向你明确必须有几个研究点、必须做系统等额外要求。
- 你希望以毕业和学习为核心，技术难度适中，代码尽量使用 Python；本建议不按你目前的工作内容或 C++ 经历安排。
- 你自述有 RTX 4060，必要时可租云 GPU；本轮没有确认已经跑通 PyTorch 训练。
- 历史计划中的学习预算约为每周 22 小时，并预留至少 20% 缓冲。这是排期参考，近期实际可用时间尚未重新确认。

算力补充：本次执行环境的只读查询返回 RTX 4070 Laptop GPU、8188 MiB，与口述型号不同；尚不确定是否对应你最终使用的训练机器。因此方案按“小显存单卡先起步”设计，正式训练前以目标机器查询结果为准。

### 2.2 已阅读综述如何转化为开题依据

项目中可对应到以下 5 篇综述。前四篇的本地译文题名、开头及结论已核对；第五篇本轮仅核对了 PDF 文件存在。你的阅读完成状态以本次陈述为准。

| 综述 | 开题中最有用的作用 |
| --- | --- |
| *AI-enabled defect detection in industrial products: A comprehensive survey, key insights and future research challenges* | 组织工业缺陷检测的整体任务与方法分类 |
| *A survey of deep learning for industrial visual anomaly detection* | 解释 normal-only 设定、记忆库、重建、蒸馏等技术路线 |
| *Automatic printed circuit board inspection: a comprehensible survey* | 区分裸 PCB 线路、板上元件与装配检查 |
| *Deep Learning Algorithms for Defect Detection on Electronic Assemblies: A Systematic Literature Review* | 连接电子组件场景、数据及监督方式 |
| 《基于计算机视觉的电子元器件表面缺陷检测综述》 | 支撑导师方向下的对象和表面瑕疵定义 |

**现在不需要重新系统阅读一轮综述。**下一步应是选定任务、看原图与标注、跑一个基线，再针对代码里不懂的模块查原始论文。

### 2.3 旧资料中需要更新的地方

- 2026 年 9 月的长期路线把部分研究和开题收敛安排到 2027—2028 年，不能继续作为本次 11 月开题的执行表。
- 旧学习计划中的 `Not started` 反映历史起点，不能覆盖你现在已完成的阅读进展，也不能据时间推定实验已经完成。
- MVTec 数据集卡仍写下载未完成；现在已有 `mvtec_anomaly_detection.tar.xz`，大小 **5,264,982,680 字节**，且归档开头可解析。未做全量解压、逐图解码和哈希校验，当前只能确认归档存在。
- VisA 归档大小 **1,929,840,640 字节**。既有数据集卡记录过图像／掩码计数与划分 CSV 路径核对；本轮未重做全部完整性检查。

## 3. 对原先三个选题条件的修正

| 原先想法 | 更适合开题的表述 |
| --- | --- |
| 能用公开电子元器件数据集训练 | 公开、可获取、许可清楚；**对象、缺陷、标注和训练协议与题目匹配**，而且训练、验证、测试划分可复现 |
| 主流成熟模型，有高质量检测论文和开源实现 | 优先选择经过同行评审、原理清楚、代码可读、预训练权重和数据处理流程可获得的方法；由原论文与维护中的框架交叉核对 |
| 将方法迁移到元器件数据集，并形成创新点 | 先复现与迁移，再针对具体困难做适度改进，例如细小缺陷、类别不平衡、边界定位或效率；用对照、消融和错误分析说明价值 |
| 达到硕士毕业要求即可 | 按导师和学校要求控制深度，形成完整、可解释、可复现的研究工作；**不以追榜或发表为前提，也不能仅凭跑通开源代码认定已满足学位要求** |

这里不需要提前证明一个从未出现过的新架构。对你的目标，更合适的工作形态是：**理解并复现成熟方法 → 找到一个具体困难 → 做一项有理由的适配或改进 → 用实验讲清效果与代价 → 做出可运行的 Python 演示。**

可以把“创新点”先写成“拟研究内容／预期改进点”。普通调参、使用预训练或迁移数据集，应如实称为实验与适配；在此基础上的针对性设计及验证，可与导师讨论如何组织成学位论文贡献，不必按顶会审稿标准提前卡死。

另一个术语区别：工业语境中的“缺陷检测”是宽泛说法，既可判断正常／异常，也可输出掩码；深度学习中的“目标检测”通常特指框和类别。**研究细裂纹、划痕和缺陷范围时，像素分割通常是自然的任务表达。**

## 4. 数据集推荐：先解决对象与训练协议

### 4.1 推荐总表

| 数据集 | 对象、标注和规模 | 最适合的用途 | 需要注意的边界 |
| --- | --- | --- | --- |
| **VisA：PCB1—4** | 含晶体管、电容、芯片等元件的板级图像；图像标签和异常掩码；四类合计 **4,416 张** | **有监督分割首选候选**，也支持 normal-only 对照 | 同时有划痕、熔损、缺失、错位等；不能全部称为单体元器件纯表面瑕疵 |
| **MVTec AD：transistor** | 晶体管；正常训练 **213**、正常测试 **60**、异常测试 **40**；像素掩码 | **最快的电子类异常检测基线** | 官方训练只有正常图；异常样本少，且不全是壳体表面缺陷；不能直接拿官方测试异常来训练后还沿用原协议 |
| **原版 Real-IAD：transistor1 等电子类** | 全库 30 类、5 视角、约 15 万张图；正常／异常及像素标注 | 单体电子零件方向的补充验证，优先只取一个电子类别、一个视角 | 下载需接受条件；按实物分组，避免同一件不同视角泄漏；不宜一开始做全库多视角 |
| **DeepPCB** | 裸板线路；1,500 对正常模板／待检图，6 类缺陷框标注 | 导师同意线路缺陷时，做监督框检测或模板比较 | 不是封装电子元器件表面，也不同于 VisA 板上元件场景 |
| **KolektorSDD2** | 工业表面缺陷；3,335 张，训练含 246 张异常、2,085 张正常 | 有监督分割的外部验证／训练练习 | 官方页未明确证明其对象就是电子元器件，不能作为“电子元器件”主数据的直接依据 |

来源：[VisA 官方说明](https://github.com/amazon-science/spot-diff)、[MVTec 官方论文](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf)、[Real-IAD 原版论文](https://arxiv.org/html/2403.12580v1)、[DeepPCB 作者仓库](https://github.com/tangsanli5201/DeepPCB)、[KolektorSDD2 官方页](https://vicos.si/resources/kolektorsdd2/)。

### 4.2 首选方案的数据：VisA high-shot

VisA 全库有 12 类、10,821 张图，包括 9,621 张正常和 1,200 张异常。官方同时提供 1-class、2-class high-shot 和 few-shot 划分。**有监督分割要用 high-shot 等允许异常训练的协议，不能沿用 1-class 的训练声明。**[官方仓库与划分入口](https://github.com/amazon-science/spot-diff)

本轮从项目中保存的官方代码 ZIP 重算 high-shot CSV，PCB 四类划分为：

| 类别 | 训练正常 | 训练异常 | 测试正常 | 测试异常 |
| --- | ---: | ---: | ---: | ---: |
| PCB1 | 602 | 60 | 402 | 40 |
| PCB2 | 601 | 60 | 400 | 40 |
| PCB3 | 604 | 60 | 402 | 40 |
| PCB4 | 603 | 60 | 402 | 40 |

建议先做 **PCB1 单类别二值分割**，然后再决定是否扩到 PCB2—4。每类只有 60 张训练异常，适合迁移学习、合理增强及小样本实验，不适合从零训练大模型。验证集从官方训练部分另留，保留官方测试部分；因此你实际用于优化参数的异常图会少于 60 张，要在清单中写明。

分割标签建议统一为 **0＝背景／正常、1＝缺陷**；原始缺陷类别 ID 大于 0 的位置合并为 1。正常图的掩码为全 0。需要做缺陷类型分类时，再核对细分类别和样本量。

官方数据为 **CC BY 4.0**，仓库代码许可另列；下载、准备脚本、划分 CSV 和指标工具可从[作者仓库](https://github.com/amazon-science/spot-diff)取得。这个仓库主要发布数据和工具，**不能把它当成已提供 SPD 全套预训练、分类和分割训练代码**。

### 4.3 导师坚持单体电子元器件时

建议把正常样本方案的数据组合改为：**MVTec `transistor` 起步，Real-IAD `transistor1` 补充**，然后根据原图确认壳体、引脚、表面和几何异常的范围。

Real-IAD 原版的官方单视角配置明确列有 `transistor1`、`regulator`、`audiojack`、`switch`、`terminalblock`、`usb` 等。优先只选一个类别与固定视角，下载后核对缺陷类型和掩码；这比一开始处理全库容易控制。[官方单视角配置](https://github.com/Tencent/AnomalyDetection_Real-IAD/blob/main/experiments/RealIAD-C1/config.yaml)

原版 Real-IAD 有仅正常训练的 UIAD 和训练中混入异常的 FUIAD 等设置，不能混用；五视角图像也不能当成互相独立的实物件。官方数据页需要登录并接受信息共享条件，声明研究用途与 **CC BY-NC-SA 4.0**。[官方数据入口](https://huggingface.co/datasets/Real-IAD/Real-IAD)

**如果严格要求“单体元器件＋真实缺陷监督训练＋只研究表面瑕疵”，当前两份主候选并未自动满足全部条件。**此时先核实 Real-IAD 的具体类别、可采用的监督划分，或请导师确认是否有合适数据，再承诺有监督题目。不要把测试图随意改成训练图来解决这个矛盾。

### 4.4 版本、许可和筛选提醒

- 本文的 MVTec AD 指原版二维数据集；原版 Real-IAD 指 CVPR 2024 数据，不混入 Real-IAD D³ 的三维／多模态设置。
- MVTec AD、Real-IAD、KolektorSDD2 均有非商业使用限制，学术演示应保留许可和引用；未来作品集展示也需遵循各自条款。[MVTec](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)、[Real-IAD](https://huggingface.co/datasets/Real-IAD/Real-IAD)、[KolektorSDD2](https://vicos.si/resources/kolektorsdd2/)
- 原版 MVTec、Real-IAD 中存在人工制作的缺陷，不能把公开基准称为完全自然产线缺陷分布。[MVTec 论文](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf)、[Real-IAD 论文](https://arxiv.org/html/2403.12580v1)
- 若为了贴合“表面瑕疵”筛掉缺失、错位等异常，要预先定义规则、公布保留样本和统计，并称为自定义子集。不能看结果后只保留容易的缺陷。

## 5. 首选路线：U-Net → SegFormer 的有监督缺陷分割

### 5.1 这条路线为什么适合你

你能在同一任务中完整接触：Python 图像处理、数据集与掩码、PyTorch 训练、CNN、迁移学习、Transformer、损失函数、指标与可视化。它不要求从零预训练视觉大模型，任务输出也直观：输入图像，输出缺陷概率图和区域。

SegFormer 使用分层 Transformer 编码器和多尺度特征，以及较简单的 MLP 分割头；适合在 U-Net 之后理解局部特征、全局关系和特征融合。它是成熟基准，不是 2026 年最新模型；选择它的理由是结构与实现适合学习。[SegFormer 原论文，NeurIPS 2021](https://arxiv.org/abs/2105.15203)、[维护中的 Transformers 实现](https://huggingface.co/docs/transformers/model_doc/segformer)

### 5.2 建议的实现顺序

1. **数据与最小训练闭环。**固定 PCB1 high-shot 的训练／验证／测试清单，检查图像、掩码、尺寸与类别，先用少量训练图确认模型能过拟合，排除标签和损失错误。
2. **U-Net 基线。**用预训练编码器建立可重复训练、验证、保存、加载和推理流程；观察细小缺陷及正常图误报。
3. **SegFormer-B0 对照。**复用相同数据清单、预处理和评测，先微调小模型；确认跑通后再考虑 B1。B0 的迁移效果与本机显存占用需要实测。
4. **只深入一个问题。**根据结果选择微小缺陷保真、类别不平衡或边界定位，做一项主要改进；保留原始模型和简单方案作对照。
5. **扩展验证与演示。**在第二个 PCB 类别或明确的外部数据上验证，并做 Python 批量推理和结果浏览。

模型实现可从 [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) 中的 U-Net 与 [Hugging Face SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) 起步。它们是框架实现，报告中应与论文作者代码区分；本轮未在你的环境运行。

### 5.3 可以形成论文内容的三个切入点

下面选一个作为主要研究内容即可，不要求三个都做。

| 切入点 | 具体问题与可做改进 | 最小验证 | 难度判断 |
| --- | --- | --- | --- |
| **细小缺陷与分辨率**，优先 | 整图缩小后划痕只剩少量像素；比较保留比例的输入、滑窗、重叠融合，必要时设计局部采样或尺度融合 | 与同等分辨率整图、普通滑窗比较；按缺陷面积分组，同时测耗时 | 中等，可逐步做 |
| **前景稀少与误报** | 正常像素多，模型容易全判背景或误报；比较采样、类别权重和 CE/BCE＋Dice 等策略 | 基础损失与改进损失对照，保留正常图，报告缺陷召回和误报 | 较易起步，分析要扎实 |
| **多尺度特征／边界恢复** | 小缺陷边界被平滑；研究浅层细节与深层特征如何融合，或一个轻量解码改进 | 原解码头、简单融合及所提改动的消融，报告小缺陷和边界结果 | 中等，适合学模型内部 |

建议先从“**细小缺陷分割与正常图误报的平衡**”观察问题，再决定改采样、损失还是解码头。研究内容可以是有证据的适配和改进，不必先给模块起新名字。

已有高分辨率切块、重叠融合和多尺度研究，因此“把图片切块”本身不能写成首次提出。可复现的相关工作是 [Divide and Conquer，CVPR Workshops 2024](https://arxiv.org/abs/2403.04932)。它主要研究异常检测切块集成，对监督分割的迁移仍需你自己实验。

### 5.4 Python 实现时最值得提前检查的细节

- **标签值。**采用两类分割时，标签应为 0／1；不要把数据中表示缺陷的 255 直接交给默认把 255 当 `ignore_index` 的损失。
- **背景保留。**SegFormer 的标签预处理使用 `do_reduce_labels=False`，避免将背景 0 错转为忽略类。
- **几何同步。**图像和掩码使用相同随机裁剪／翻转；掩码缩放用最近邻，图像使用适当图像插值。
- **输出尺寸。**检查 logits 的实际分辨率，再按明确规则恢复到标注尺寸；不能把低分辨率图与原图掩码直接比较。
- **正负样本。**训练和评测保留正常图；既看漏检，也看正常图上的误报。缺陷引导裁剪仅在训练集进行。

其中标签处理、损失输入与输出尺寸的实现依据见 [SegFormer 官方框架文档](https://huggingface.co/docs/transformers/model_doc/segformer)；其他项是针对本任务的数据处理建议。

## 6. 备选路线：PatchCore → Dinomaly 的异常检测与定位

### 6.1 什么时候改选它

导师更看重“缺陷样本稀少”“训练只用正常图”“对未预先标注的缺陷定位”时，优先选这条。它可以直接使用 MVTec 和 VisA 1-class 协议，也能自然接触预训练视觉 Transformer。

**PatchCore** 冻结预训练 CNN，提取正常局部特征，采样建记忆库，推理时用近邻距离产生异常分数。其训练阶段主要是建库，通常没有骨干反向传播；非常适合快速掌握异常热图和评价，但不能单靠它练完梯度训练。[CVPR 2022 原论文](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)、[作者代码](https://github.com/amazon-science/patchcore-inspection)

**Dinomaly** 使用冻结的预训练 DINOv2/ViT 编码器，训练瓶颈与 Transformer 特征重建网络，由特征差异定位异常。你能学习 token、注意力、冻结与微调、特征重建和损失；不用自己预训练 DINOv2。[CVPR 2025 原论文](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html)、[作者代码](https://github.com/guojiajeremy/Dinomaly)

### 6.2 控制难度的方法

- 先按类别分别训练，完成一个电子相关类别；统一多类别训练可以留到后期。
- PatchCore 先验证数据和指标；Dinomaly 先跑完整的前向、反向、保存和加载，再做正式实验。
- Dinomaly 作者默认 ViT-B/14、resize 448 后裁到 392、batch 16，实验使用 RTX 3090 24GB；瓶颈加 8 层解码器**不是一个很小的分类头**。[作者训练脚本](https://github.com/guojiajeremy/Dinomaly/blob/master/dinomaly_mvtec_sep.py)
- 可尝试 ViT-S/14、batch 1—2 和 224／280 输入进行学习性试跑；这是建议配置，没有本机验证，也不是原设置复现。后续恢复合理分辨率，必要时再用云 GPU。
- 作者类别分别训练脚本中的权重保存语句被注释，自己的工程要明确保存权重、配置和划分；不能只留控制台分数。[同一作者脚本](https://github.com/guojiajeremy/Dinomaly/blob/master/dinomaly_mvtec_sep.py)

研究内容可围绕特征层选择、较小模型、细节保真、多尺度重建或正常分数校准，选一项做深。更换骨干的实验本身也有学习价值；最终如何上升为论文改进，由实验和导师意见决定。

### 6.3 EfficientAD 的位置

如果更希望使用较小网络学习可训练方法，可将 Dinomaly 换成 EfficientAD。它有教师、学生和自编码器分支，以正常数据学习特征差异，同时关注推理效率。[WACV 2024 论文](https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf)

有维护中的 [Anomalib 实现](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/efficient_ad/README.md)，但不能称为论文作者官方代码。该实现还涉及教师权重、ImageNette 和特定归一化设置；不是只下载工业数据就完成全部准备。论文标题中的毫秒延迟不能当作你的电脑实测速度。

## 7. 其他技术怎样取舍

| 技术 | 对当前任务的价值 | 建议安排 |
| --- | --- | --- |
| **CNN、迁移学习、梯度训练** | 训练与调试基础，可迁移到大量视觉任务 | 用所选主线完成：监督路线通过 U-Net，异常路线可通过 Dinomaly；不要求两条都做 |
| **注意力、Transformer、ViT** | 理解现代视觉表征和特征交互 | 主线选 SegFormer 或 Dinomaly 之一深入 |
| **YOLO、RT-DETR** | 学习框检测、实例匹配和推理，岗位迁移面较广 | 如果最终标签是框，可升主线；若做分割，毕业前补一个短实践即可 |
| **蒸馏、轻量化** | 连接模型效果和计算成本 | 主线稳定后做一个小模型／大模型对照；按需学 EfficientAD |
| **自监督预训练／视觉基础模型** | 理解为什么预训练特征可迁移 | 学 DINOv2 使用与原理；不从零重训大规模预训练 |
| **扩散模型** | 学生成与重建，能用于缺陷重建或合成 | 作为拓展最合适；跑通后非常感兴趣，可与导师讨论替代主线 |
| SAM、视觉语言模型、多模态大模型 | 能拓展标注和视觉理解思路 | 当前任务没有迫切依赖，可在论文主线稳定后再选学 |

你提到的概念也需要分层：**注意力是机制，Transformer 是架构，ViT 是它在图像中的一种应用；扩散是另一类建模方式。**它们不是必须分别做四个项目的四项技术，选择一个视觉 Transformer 模型，就能在同一项目中学到前三者。

### 扩散建议：学得进去，但先给它较小的范围

推荐论文是 **DiffusionAD: Norm-Guided One-Step Denoising Diffusion for Anomaly Detection**，IEEE TPAMI 2025。它使用扩散重建与分割网络，并利用正常图像、合成异常和合成掩码训练；不是必须依赖 Stable Diffusion 的大模型方案。[出版记录](https://ieeexplore.ieee.org/abstract/document/11005495)、[论文全文](https://arxiv.org/html/2303.08730v4)、[作者代码](https://github.com/HuiZhang0812/DiffusionAD)

训练与推理均使用单步去噪重建，norm-guided 涉及两个噪声尺度；作者仍采用较长训练，不能把“单步”理解为训练成本低。作者实验包含 256×256 输入及额外纹理／前景准备。建议开题后先做一个类别的小实验：改变噪声尺度，观察正常细节保留、缺陷修复与定位之间的关系。训练时长与显存先测量，再决定是否租云。[论文实现细节](https://arxiv.org/html/2303.08730v4)、[作者实现配置](https://github.com/HuiZhang0812/DiffusionAD/blob/main/args/args1.json)

如果你做过小实验后发现更喜欢生成／重建，也愿意承担这部分成本，扩散可以成为主研选择；目前综合“难度适中＋完整 CV 能力”，我仍优先推荐监督分割。

## 8. 代表性论文与代码：按使用目的阅读

下面是资源清单，不是开题前必须逐篇精读的任务。**首选路线先读 U-Net、SegFormer、VisA 协议与一篇工业分割工作；备选路线先读 PatchCore、Dinomaly 和数据协议。**

| 用途 | 论文／发表 | 代码与阅读重点 |
| --- | --- | --- |
| 监督分割基础 | [U-Net: Convolutional Networks for Biomedical Image Segmentation，MICCAI 2015](https://arxiv.org/abs/1505.04597) | [原项目](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)、[现代 PyTorch 框架实现](https://github.com/qubvel-org/segmentation_models.pytorch)；学编码器、解码器、跳跃连接；原任务是医学图像，迁移工业数据需实验 |
| **首选主模型** | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers，NeurIPS 2021](https://arxiv.org/abs/2105.15203) | [作者代码](https://github.com/NVlabs/SegFormer)、[HF 实现](https://huggingface.co/docs/transformers/model_doc/segformer)；先理解分层特征与 MLP 头，使用维护中的 Python 实现更方便起步 |
| 工业监督分割依据 | [Segmentation-Based Deep-Learning Approach for Surface-Defect Detection，JIM，2019 在线发表](https://doi.org/10.1007/s10845-019-01476-x) | [作者代码](https://github.com/skokec/segdec-net-jim2019)；说明分割与缺陷判定的工业依据。原代码 Python 2.7／TensorFlow 1，适合读方法，当前不优先重建旧环境 |
| 工业监督分割补充 | [Mixed supervision for surface-defect detection: from weakly to fully supervised learning，Computers in Industry 2021](https://vicos.si/resources/mixed-segdec-net-comind2021/) | [作者 PyTorch 实现](https://github.com/vicoslab/mixed-segdec-net-comind2021)；学习标签使用、分割与图像判定；不要求把混合监督加入自己的论文 |
| 数据与任务设定 | [SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation，ECCV 2022](https://arxiv.org/abs/2207.14315) | [VisA 作者资源](https://github.com/amazon-science/spot-diff)；重点读数据、三种协议与指标，不把数据发布等同于全部训练代码发布 |
| **异常检测起点** | [Towards Total Recall in Industrial Anomaly Detection，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) | [PatchCore 作者代码](https://github.com/amazon-science/patchcore-inspection)；学局部特征、coreset、记忆库与近邻分数 |
| **ViT 异常检测候选** | [Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html) | [作者代码](https://github.com/guojiajeremy/Dinomaly)；学冻结 ViT、瓶颈、线性注意力与特征重建；先用类别分别训练设置 |
| 蒸馏与效率备选 | [EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies，WACV 2024](https://arxiv.org/abs/2303.14535) | [Anomalib 框架实现](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/efficient_ad/README.md)；理解特征蒸馏、局部／全局异常与成本 |
| 另一蒸馏参照 | [Anomaly Detection via Reverse Distillation From One-Class Embedding，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html) | [RD4AD 作者代码](https://github.com/hq-deng/RD4AD)；学教师编码器、瓶颈、学生解码器；按需看，不要求多跑一条线 |
| 高分辨率相关工作 | [Divide and Conquer: High-Resolution Industrial Anomaly Detection via Memory Efficient Tiled Ensemble，CVPR Workshops 2024](https://arxiv.org/abs/2403.04932) | [Anomalib 切块文档](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/data/input_tiling.html)；学分辨率、显存、位置与重叠融合的关系 |
| 框检测候选 | [DETRs Beat YOLOs on Real-time Object Detection，CVPR 2024](https://arxiv.org/abs/2304.08069) | [RT-DETR 作者代码](https://github.com/lyuwenyu/RT-DETR)；若选框检测，结合成熟小模型起步，按需学 Transformer 检测 |
| 扩散拓展 | [DiffusionAD，TPAMI 2025](https://ieeexplore.ieee.org/abstract/document/11005495) | [作者代码](https://github.com/HuiZhang0812/DiffusionAD)；学噪声、正常重建、合成异常与分割 |

基础原理按遇到的问题补读即可：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[ViT](https://arxiv.org/abs/2010.11929)、[DINOv2](https://arxiv.org/abs/2304.07193)及其[作者实现](https://github.com/facebookresearch/dinov2)、[DDPM](https://arxiv.org/abs/2006.11239)。不需要先学完所有论文才开始实验。

**关于“可复现”的准确理解：**上表区分了作者代码、框架实现和数据工具。公开代码不保证原生 Windows、当前 Python 版本或小显存卡直接跑通。应锁定成功运行的版本、权重、配置和数据清单；改变输入、训练预算或验证协议后，如实说明差异，不以必须追平论文数字为开题门槛。

## 9. 实验怎么做，才能同时支持论文与学习

### 9.1 两条路线分别使用自己的协议

| 项目 | 首选：监督分割 | 备选：normal-only 异常检测 |
| --- | --- | --- |
| 参数学习 | 正常与真实缺陷图＋掩码 | 正常图；按方法声明额外预训练／合成数据 |
| 验证 | 从官方训练部分留出，分层保留正常／异常 | 从正常训练部分留出正常校准集；若用真实异常挑模型，需明确改变了验证设定 |
| 模型选择 | 独立验证集，固定预算 | 固定训练步数或正常验证规则；不能用测试缺陷反复挑配置 |
| 最终测试 | 保留测试清单，冻结方案后评测 | 同样保留；区分开发诊断与未参与选择的最终测试 |
| 可否直接比较分数 | 与相同监督与划分的模型比较 | 与相同正常样本条件比较；不能把与监督模型的差异全归因于架构 |

原始图像先划分，再裁块和增强；多视角数据按实物件分组。验证阈值、后处理和训练轮数都应在最终测试之前确定。

框架默认设置需要看清：本次核查到 Anomalib 的 MVTecAD 数据模块默认 `val_split_mode=SAME_AS_TEST`。如果再据验证分数选模型、早停或校准阈值，测试信息就进入选择过程。自己实验时应明确配置独立划分，不直接照默认值汇报为独立验证。[官方源码](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py)

### 9.2 推荐报告的指标

| 要回答的问题 | 建议指标和记录 |
| --- | --- |
| 缺陷区域分得准不准 | **缺陷类 IoU、Dice**；写清逐图／整体汇总、空掩码处理；不只报被大面积背景主导的均值 |
| 正常件会不会被误判 | 在验证集确定阈值后的图像级误报率、误报图数／正常图数 |
| 缺陷件有没有漏掉 | 图像级召回、漏检数／异常图数；可按缺陷类型和大小分组 |
| 异常分数排序能力 | 异常检测路线报图像级 AUROC，像素级 AUROC／AP；按需要补 AUPRO，并注明积分范围与实现 |
| 小缺陷是否改进 | 预先固定面积分组，报告各组数量和结果；最终测试掩码只用于评估分组，不用于选模型 |
| 代价是否合适 | 单图端到端耗时、峰值显存／内存、模型或记忆库大小；明确输入、batch、硬件与预热 |

将像素概率图转成图像级判定时，先固定聚合方式，例如像素最大值或预设 top-k 均值。聚合规则、像素／图像阈值及最小连通区域面积均在训练／验证阶段确定，再用相同规则计算最终误报和漏检。

训练／测试异常图不多时，优先同时列出“错了几张”。例如 40 张异常中少错一张就是 2.5 个百分点，不能仅凭一次小幅提升判断稳定改进。正式论文的关键实验建议做 3 个随机种子；开题预实验先用固定 1 个种子即可，明确属于初步结果。

### 9.3 一张足够起步的对照表

首选路线采用下面的层次即可，不需要一次堆很多模型：

| 实验 | 模型／设置 | 要回答的问题 |
| --- | --- | --- |
| S0 | U-Net＋统一数据与基础损失 | 数据、训练和指标是否正确，CNN 基线有什么问题 |
| S1 | SegFormer-B0＋相同划分 | Transformer 路线在此数据上是否有价值，成本如何 |
| S2 | 选定主模型＋一项改进 | 针对问题的改变是否有效 |
| S3 | 去掉／替换改进中的关键设计 | 效果由哪部分带来，是否只是更多算力或更大分辨率 |
| S4 | 第二类别／外部数据 | 改进是否只适用于一个类别 |

U-Net 与 SegFormer 的差异同时包括预训练、规模、编码器和解码器等。不能仅凭 S0／S1 把全部差异归因于注意力；要研究注意力机制本身，需进一步控制变量。

备选路线把 S0／S1 换成 PatchCore／Dinomaly 即可，但保持相同 normal-only 设定。完成一个有解释的对照，比同时跑十种算法更适合当前阶段。

## 10. 如何把论文过程转化为以后能用的能力

这里按计算机视觉算法与视觉应用方向评估迁移价值，不依据你现在的工作内容。

企业官方岗位样本中，博世的视觉 AI 岗位同时要求检测／分割／异常检测、Python、模型训练与交付，并认可学术研究项目；另一 AOI 岗位强调 Python、深度学习框架、数据管理和从原数据到应用的过程，也列出 Transformer。这支持把完整项目能力作为重点，而不是只积累模型名。[博世视觉 AI 岗位](https://jobs.smartrecruiters.com/BoschGroup/744000146881229-ai-engineer-me)、[博世 AOI 岗位](https://jobs.smartrecruiters.com/BoschGroup/744000125446259-data-scientist-ps-rbcd)

上述只是岗位样本，不能代表所有招聘需求；论文选题与掌握某种架构都不直接保证录用。

| 能力 | 在论文项目里怎样练 | 最后应有的证据 |
| --- | --- | --- |
| Python、NumPy、OpenCV | 检查图像、掩码、增强、连通域和可视化 | 可复用的数据检查与处理脚本 |
| PyTorch 训练与调试 | Dataset/DataLoader、优化器、学习率、梯度、混合精度、断点恢复 | 能独立解释和修改的训练／验证代码 |
| CNN 与 Transformer | 从 U-Net 到 SegFormer，或从 CNN 特征到 DINOv2 | 张量形状、结构说明、至少一组受控实验 |
| 实验与数据能力 | 固定划分、跟踪配置、对比与消融、分析误报漏报 | 配置文件、结果表、失败案例、重跑说明 |
| 推理与模型交付 | Python 批量预测；按需做 ONNX 导出与一致性检查 | 命令行推理、演示界面、耗时与资源记录 |
| 技术表达 | 讲清任务、数据、模型、困难和取舍 | README、技术说明、短演示与答辩材料 |

推荐把最后的项目整理成：**一份 Python/PyTorch 仓库＋可重复实验配置＋一组可信对照＋可视化演示**。界面可用 Streamlit 或 Gradio 做小型浏览器演示；这是展示方式，不必另做复杂平台。

若毕业前还有余力，可补一次小型框检测实践和一次 ONNX 推理体验。扩散、VLM 等兴趣内容按需加入学习笔记，不必都变成论文研究点。

## 11. 从现在到开题：约 45 小时的可执行安排

按历史每周 22 小时和至少 20% 缓冲估算，11 月 15 日前安排约 40—50 小时主动工作较合理。下面的小时数是预算，GPU 自动运行另计，不代表已测工时。

| 时间 | 主动时间预算 | 重点 | 可见产出 |
| --- | ---: | --- | --- |
| **10/28—10/30** | 5 小时 | 抽看原图与掩码，核对数据归档；确定向导师讨论的两套对象和监督方式；同时起草报告目录 | 一页选题比较、样本拼图、数据／标签清单 |
| **10/31—11/3** | 10 小时 | 只跑一个基线：监督方案 U-Net／PCB1；异常方案 PatchCore／transistor | 能加载权重重跑的结果、热图／掩码、指标和耗时 |
| **11/4—11/8** | 12 小时 | 与导师收窄对象；尝试 SegFormer-B0 或 Dinomaly 的单类训练；整理实际失败问题 | 一张对照表、典型误报漏报、主研究问题和备选方案 |
| **11/9—11/11** | 6 小时 | 固定开题技术路线，做必要的小验证，完成报告初稿与研究计划 | 报告初稿、路线说明、预实验图表 |
| **11/12—11/13** | 4 小时 | 导师反馈、报告修改、PPT 主体 | 可审阅版本、答辩问答 |
| **11/14—11/15** | 8 小时 | 定稿、引用检查、演练、备份 | 提交材料、PPT、代码和结果备份 |

**开题前的最低目标：一个任务定义、一份数据与划分说明、一个真正跑通的基线、一组真实问题，以及后续实验计划。**能再跑通现代模型更好，但不要求开题时已完成全部论文实验或确定最终创新效果。

收缩规则：

- 一个环境问题累计约 2 小时仍无进展，记录错误，切换维护中的框架实现或较小配置；云 GPU 主要解决实测算力瓶颈。
- 到 11 月 3 日还没有训练闭环，就先完成一个基线，不增加模型数和数据集。
- 到 11 月 8 日应明确主任务；SegFormer、Dinomaly 二选一作为深入对象。
- 11 月 11 日以后以补证据和材料为主，不再引入扩散等新路线。
- 导师若改变对象范围，优先复用已经完成的数据、训练与评测流程，再调整数据集。

### 开题以后，按成果推进的顺序

1. 完成两个基线与可重复评测，读懂主要代码。
2. 用错误分析确定一项改进，完成对照与消融。
3. 扩展到第二类别或数据集，补充稳定性和成本分析。
4. 做 Python 推理演示，整理代码、图表和论文章节。
5. 有余力时做扩散或其他现代技术的小实验，再决定是否纳入论文。

## 12. 可以直接带去见导师的方案

### 方案 A：首选，监督分割

**暂定题目：**《基于深度学习的电子组件外观缺陷分割方法研究》。若导师同意突出 Transformer，可改为《基于多尺度视觉 Transformer 的电子组件外观缺陷分割方法研究》。

**对象与数据：**先使用 VisA PCB1，优先增加一个 PCB 类别验证，再按时间决定是否扩到四类；采用 high-shot 协议，允许使用正常及带缺陷掩码的图像训练。题目中的“电子组件／外观”用于容纳板级场景，需导师认可。

**研究内容：**建立 U-Net 和 SegFormer-B0 基线；围绕细小缺陷保真与误报问题，选择采样、损失或特征融合中的一项改进；通过消融、跨类别验证和效率评测分析效果，完成 Python 检测演示。

**学习收益：**完整监督训练、分割、迁移学习、CNN、Transformer、多尺度特征与实验工程。

### 方案 B：备用，正常样本异常检测

**暂定题目：**《基于预训练视觉特征的电子元器件外观异常检测与定位方法研究》。若最终严格限定表面缺陷，再在数据审核后调整措辞。

**对象与数据：**MVTec `transistor` 起步，Real-IAD `transistor1` 补充；若导师接受板级对象，可使用 VisA PCB 的 1-class 协议。

**研究内容：**建立 PatchCore 和 Dinomaly 基线；研究适合电子类细微缺陷的特征选择、重建或尺度处理；结合误报漏报、定位能力和资源消耗评价。若训练复杂度不合适，可用 EfficientAD 替代 Dinomaly。

**学习收益：**异常检测、预训练特征、ViT、注意力、重建／蒸馏与实验工程。

### 见导师时可这样说明

> 我已经完成前期综述阅读，想把题目收敛到公开数据可支撑、Python 可实现的视觉缺陷任务。优先考虑有监督分割，用 U-Net 建立基线，再用 SegFormer 学习和研究 Transformer；如果更希望突出缺陷稀缺，则采用 PatchCore 和 Dinomaly 的正常样本路线。开题前先拿出一个可运行基线和实际失败问题，之后做一项有针对性的改进、完整对照与演示，不以发表或追最高分为目标。需要先确定对象是否能包括 PCB／板上元件，以及是否允许缺陷样本参与训练。

开题报告可按“研究背景与对象 → 现状及方法分类 → 数据和任务定义 → 拟研究内容 → 技术路线 → 预实验与可行性 → 进度与预期成果”组织。硕士论文可在此基础上形成基础方法、改进方法、实验分析和系统实现等章节；章节数量与研究深度由导师确认。

## 附：本地资料入口与证据边界

- [当前研究方向](../../personal-planning/machine-vision-inspection-research-direction.md)
- [历史学习状态与时间预算](../../personal-planning/current-learning-status.md)
- [历史长期路线](../../personal-planning/machine-vision-inspection-roadmap-2026-2028.md)
- [MVTec AD 数据集卡](../../docs/dataset/MVTec%20AD/dataset-card.md)
- [VisA 数据集卡](../../docs/dataset/VisA/dataset-card.md)
- [五篇综述所在目录](../../docs/paper/260917/)

本文将“用户已确认事实”“本地归档／文档记录”“论文和官方代码事实”“实施建议”分别说明。尚待确定的是导师对象范围、监督方式、学校最终要求，以及实际训练兼容性和效果。下一步最有价值的证据是一个可以重跑的单类基线，而不是再增加一份模型名录。
