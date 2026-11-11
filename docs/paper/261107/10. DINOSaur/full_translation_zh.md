# 重新思考边缘端持续异常检测：在真实工业条件下开展基准评测

**原文：**[Rethinking Continual Anomaly Detection on the Edge: Benchmarking Under Realistic Industrial Conditions](<10-DINOSaur-Continual-Anomaly-Detection-2026.pdf>)。对应本目录20页PDF，arXiv:2605.24251v2，2026年6月29日。翻译日期：2026-11-11。

**作者：**Chad Weatherly、Sen Lin。**单位：**休斯敦大学。

**翻译说明：**按本地PDF逐段翻译摘要、正文第1—6节、参考文献[1]—[41]、附录A—E、图1—2图注和关键图内文字、表1—7全部数据、公式(1)—(13)及算法1—2。合并跨页续段，去除运行页眉页脚；图形不重绘，请结合原PDF查看。作者、模型和数据集等专名保留；文献题名中文化，同时保留原始书目信息。性能判断属于作者在该版本中的陈述。原文内部不一致处用译注说明，不静默改动数字或结论。

## 摘要

持续异常检测（CAD）旨在满足工业检测系统适应不断变化的生产条件的需求，但现有方法存在三个共同的关键缺口：评价不够真实、缺乏系统比较，以及没有考虑边缘部署约束。我们提出统一基准，将结构与逻辑异常上的离散任务评价、新的连续漂移协议、首次对所有已发表CAD方法的直接比较，以及边缘硬件上的计算效率剖析结合起来。结果表明，现有CAD方法并不能稳定优于采用简单经验回放的传统方法。受此启发，我们提出DINOSaur：一种无需训练的方法，将冻结的DINOv3骨干、按空间位置索引的核心集记忆，以及邻域限制的异常评分相结合。DINOSaur从结构设计上实现零遗忘，除几何漂移外，在每一种协议上都优于全部受评方法；在几何漂移下，所有方法都退化至随机水平。它在NVIDIA Jetson Orin Nano上实现低于100毫秒的推理，并能在30秒以内于设备端适应新任务。

**关键词：**持续学习；异常检测；工业检测；基准；边缘部署。

## 1 引言

工业4.0的发展提高了制造业的自动化程度，其中包括采用人工智能驱动的质量控制。在生产环境中，质量控制不佳可能造成极高成本；漏检缺陷会导致保修索赔、产品召回以及重大声誉损失[11]。异常检测（AD）使用主要由正常样本训练的模型识别缺陷产品，因此自然适合工业环境中的自动视觉检测[7]。

然而，现实制造过程从来不是真正静态的。环境条件波动、工具磨损退化、材料批次差异和生产参数随时间漂移，都会使正常产品外观的分布持续变化，对假定数据分布固定的AD系统构成根本挑战。正常分布变化后，已部署模型可能漏掉真实缺陷，也可能把正常变化标成异常，两者都有显著经济后果。持续学习（CL）提供一种框架，使模型在保留既有分布知识的同时适应新数据[32]。近期若干研究提出持续异常检测方法[19,23,31]，将AD架构与CL策略结合，在顺序任务的数据演化中保持性能。

尽管取得了这些进展，我们仍在现有CAD文献中发现三个关键缺口。

**（1）忽视计算可行性。**工业AD系统通常运行在计算资源有限的边缘设备上，为跟上生产吞吐量，往往需要每秒执行多次推理。近期CAD方法采用基于视觉语言模型（VLM）、扩散模型或大型视觉Transformer的大规模架构，却没有分析它们能否部署于受限环境[13,20,33,40]。近期工作虽然研究了边缘硬件上的高效异常检测[3,4]，但针对的是静态设置或单骨干回放设置。我们所关心的是专用CAD方法的可行性：它们为顺序学习与任务识别增加的机制，尚未接受边缘部署方面的检验。

**（2）评价不真实且范围狭窄。**现有CAD基准把不同产品类别作为不同任务，从而产生突然、离散的边界，例如从瓶子切换到电缆。现实工业漂移却是渐进的：同一种产品的外观会因过程变化而随时间改变。现有基准均未覆盖这种连续漂移。此外，评价仅限于表面缺陷等结构异常，忽视了缺件、排列错误等逻辑异常[6]；这是一个重要而未解决的挑战，尚无CAD方法接受过这方面的评价。

**（3）缺乏系统比较。**已经发表的三种CAD方法DNE[19]、IUF[31]和UCAD[23]，各自在不同实验条件下与不同基线独立比较，彼此从未得到全面评估。因此，我们无法理解它们各自的优势与不足，更难判断其在真实场景中的可行性。

为弥补上述缺口，本文作出以下贡献：

- 提出统一CAD评价框架，首次结合所有已发表CAD方法的直接比较、新的连续漂移协议、逻辑异常评价，以及边缘硬件上的计算效率剖析。
- 评价揭示，专用CAD方法并不能稳定优于采用简单经验回放的传统基线。受此启发，提出无需训练的DINOSaur：以一小部分计算成本优于全部受评方法，从而支持实际边缘部署。

代码与数据：<https://github.com/Continue-Edge-AI-Lab/Rethinking-Continual-AD>。

## 2 相关工作

**工业场景中的异常检测。**工业异常检测（IAD）大致分为两种范式。基于分布的方法[10,25,27]对正常样本特征的分布建模，通过近邻评分、多元高斯拟合或学习到的正常性得分，将分布外测试样本标为异常。基于重建的方法[35,36]使用正常图像训练编码器—解码器网络，以较高重建误差作为异常信号。统一或多类别AD[35]可同时处理多个类别，但假定一次性获得全部数据，不满足CAD的顺序数据约束。

**用于异常检测的大规模模型。**近期工作将大型基础模型用于AD，包括基于大语言模型的推理[13]、视觉语言提示[40]、上下文学习[41]和扩散模型[20,33]。虽然这些方法在基准上表现强劲，但需要数十亿参数和GPU级硬件，每幅图像的推理延迟以秒计，根本不适合要求每秒处理多幅图像的边缘部署。此外，大模型并不免疫于灾难性遗忘：将CLIP等视觉语言模型适配到顺序任务，会显著降低其零样本迁移能力[39]；多模态大语言模型微调后，同样无法保留预训练时的视觉理解能力[38]。计算不可行与易遗忘这双重挑战，进一步促使我们关注面向边缘CAD的轻量、免训练方法。

**持续学习。**CL研究从任务序列$\{D_1,D_2,\ldots,D_T\}$学习的问题，约束为：训练任务$t$时，只能访问数据集$D_t$，无法访问$D_1,\ldots,D_{t-1}$。每个任务对应一组独特数据$D_t=\{x_t,y_t\}$，$t\in[1,2,\ldots,T]$。在CL框架下，我们希望模型能不断学习新任务，并尽可能减少此前知识的灾难性遗忘[32]。

现有CL策略主要分为三类。基于正则化的方法在损失中加入惩罚项，限制权重更新以保留先前任务的重要知识。弹性权重巩固（EWC）[17]通过Fisher信息识别重要参数；突触智能（SI）[37]在线跟踪参数重要性；记忆感知突触（MAS）[1]根据输出敏感性计算重要性。基于架构的方法为不同任务分配专用参数或子网络，例如渐进神经网络[28]。基于记忆的方法存储先前任务信息，并利用它修改新任务学习，又可分为基于复习和基于梯度投影的方法。前者存储既有任务的原始样本或特征表示，在新任务训练时回放，例如经验回放[26]和暗经验回放[9]。后者[21,22,29]在记忆中保存旧任务的梯度信息，据此修改新任务的梯度方向。经验表明，基于记忆的方法在CL基准中始终取得最强性能[32]，因为它们通过保留对过去信息的访问，最直接地解决遗忘问题。

**持续异常检测。**迄今已有三种已发表方法专门研究真实工业环境中的CAD。DNE[19]保存来自视觉Transformer（ViT）骨干的逐任务嵌入统计量，仅在图像层面通过马氏距离评分。IUF[31]采用ViT编码器—解码器，配合正则化的潜空间分离和受约束的梯度更新。UCAD[23]结合对比学习、SAM[16]纹理分解和任务专用知识库。各方法的详细架构见附录A。

最近出现了若干CAD基准，但没有直接比较CAD方法。Bugarin等[8]在MVTec-AD上提出像素级CAD基准，将现成的AD方法（PatchCore、CFA、STFPM、EfficientAD、FastFlow等）包装为采用标准回放或记忆库策略的版本。Pezzè等[24]同样评测了回放条件下的现有AD方法，提出用超分辨率降低记忆占用的压缩回放策略，但没有新增检测架构。Continual-MEGA[18]提供大规模基准并评价零样本泛化；其受评方法包括CAD方法和基于CLIP的VLM，同时提出带混合专家适配器的CLIP基线，专门缓解大模型遗忘。这三者都依赖通用CL策略，主要是将回放用于常规AD骨干，没有围绕正常性的非平稳变化、无监督设置或边缘部署等CAD特有约束设计方法。

与我们的边缘关注最接近的是Barusco等：他们降低了边缘硬件上的静态AD成本[3]，并通过压缩回放研究边缘持续视觉异常检测[4]；但两项工作都把通用回放应用于单个静态骨干，没有比较专用CAD方法、连续漂移或逻辑异常。我们的基准还在以下方面不同：首次包含连续漂移协议，而不只考虑离散类别切换；首次借助MVTec-LOCO评价CAD的逻辑异常；首次在实际边缘硬件上剖析计算效率。

## 3 持续异常检测的统一基准

### 3.1 动机

该基准的核心动机，是当前CAD评价方式与工业AD系统实际运行方式脱节。在真实制造环境中，检测系统长期监测同一种产品。随着温度、湿度、照明等环境条件变化，工具退化，材料批次改变，或者生产过程漂移，产品外观逐步改变。这些变化使正常数据分布持续演进，而不是像现有IAD数据集模拟的那样发生突然的类别切换。例如，检测磁瓦表面缺陷的工厂会经历传感器逐渐退化、季节性照明变化和材料批次变化。这些都会改变“正常”的外观，但系统必须始终检出真实缺陷。这与检测系统突然从瓶子切换到电缆根本不同。

除分布漂移外，现有CAD评价的范围也狭窄。工业缺陷不限于表面的结构异常；缺件、排列错误等逻辑异常同样常见，需要根本不同的检测策略，但此前没有已发表CAD方法在这类异常上接受评价。

因此，本基准沿对应上述缺口的三个轴评价CAD方法：（1）评价的真实性与范围，覆盖标准离散任务协议、新的连续漂移协议，以及通过MVTec-LOCO开展的逻辑异常评价；（2）对所有现有CAD方法及传统AD基线进行系统比较；（3）剖析计算效率，以评估边缘部署可行性。

### 3.2 离散任务协议

离散任务协议采用AD与CAD的标准数据集[34]。

**MVTec-AD。**MVTec-AD[7]包含纹理和物体共15个类别，各类别作为独立任务顺序学习。训练只使用正常样本（无监督设置），反映制造业异常样本稀缺的现实。该协议作为与现有方法直接比较的基准。

**MVTec-LOCO。**MVTec-LOCO[6]包含5个类别，兼有结构异常和需要全局场景理解的逻辑异常。顺序协议与MVTec-AD相同，每个类别作为一个任务。

### 3.3 连续漂移协议

为模拟真实工业数据漂移，我们基于磁瓦缺陷（MTD）数据集[14]设计协议。MTD只有磁瓦这一产品类别，但包含气孔、裂纹、破损、磨损和不平整等多种缺陷，适合研究同一产品内部的分布偏移，而不是产品类别间的切换。

**通过渐进增强模拟漂移。**我们对MTD施加强度逐渐增加的图像增强，构建$T=10$个任务。设$A(x,\alpha)$为以强度$\alpha$作用于图像$x$的增强函数。对任务$t\in\{1,\ldots,T\}$，后续任务的强度更大：

$$\alpha_t>\alpha_{t-1},\quad\forall t>1.\tag{1}$$

各任务的数据为：

$$D_t=\{A(x,\alpha_t)\mid x\in D_{\mathrm{MTD}}\}.\tag{2}$$

其中$D_{\mathrm{MTD}}$是原始MTD数据集。随着$t$增加，图像失真更严重，模拟环境或过程的渐进漂移。

**增强类型。**我们运行三条独立实验轨道，分别模拟一种真实漂移：

- 颜色失真：逐渐改变色相、饱和度、亮度和对比度，模拟照明与材料批次变化。
- 高斯模糊：逐渐增大核尺寸和标准差，模拟传感器退化和焦点漂移。
- 几何失真：逐渐增强透视变换、旋转和扭曲，模拟方向变化和相机失准。

各类增强都对应有文献记录的工业变化来源[12]，因而提供可解释且有物理动机的漂移情景，而不是任意分布偏移。我们通过实验选择强度范围，确保磁瓦物体仍然可见。各类漂移10个强度等级的精确参数见附录C.3，示例见图1。

### 3.4 评价指标

图像级异常得分取图像块级得分的最大值。所有方法使用以下指标评价。

**图像级AUROC（ROC曲线下面积）。**衡量正常与异常图像之间、不依赖阈值的区分能力。报告最后一个任务训练完成后，对所有任务取平均的最终AUROC。

**图像级准确率。**依赖阈值，衡量正常与异常样本的分类性能。

**召回率（敏感度）。**专门衡量被正确识别的异常样本比例。在工业场景中，漏检（假阴性）的代价高于误报，因此尤为关键。准确率和召回率使用固定阈值，即训练得分的第97.5百分位；阈值选择细节见附录B。由于AUROC不依赖阈值，它仍是主要指标。

**像素级AUROC。**衡量缺陷区域定位，为图像级检测提供补充。在DINOSaur、PatchCore等基于分布的方法中，像素异常图直接由图像级预测使用的相同图像块得分得到，图像得分就是图像块得分的最大值。因此，我们预期二者紧密相关，并在第5.3节验证：对像素评分有意义的协议（排除变换填充占主导的几何漂移），二者的相关系数为$r=0.945$，并保持领先方法的排序，证实二者互为伴随结果。图像级AUROC仍是主要指标，因为它符合工业检测常见的通过/不通过二元决策；像素级AUROC则表明结论并非指标选择造成的假象。

**图1：MTD上各类漂移跨任务的渐进增强。**上：颜色失真（亮度、对比度、饱和度）。中：高斯模糊（核尺寸与$\sigma$逐渐增大）。下：几何失真（旋转、平移、缩放、剪切）。从左侧任务1的最小失真到右侧任务10的最大失真，强度递增。模糊和颜色增强保持缺陷掩码不变；几何增强对图像与缺陷掩码施加相同变换。关键图内标注：原始图像；任务1、3、5、7、10；漂移强度递增。展示的颜色幅度区间依次为$[0.00,0.05]$、$[0.10,0.15]$、$[0.20,0.25]$、$[0.30,0.35]$、$[0.45,0.50]$；模糊参数$(k,\sigma)$依次为$(1,0.5)$、$(5,1.5)$、$(9,2.5)$、$(13,3.5)$、$(19,5.0)$；几何示例的旋转/尺度依次为$(2^\circ,0.01)$、$(6^\circ,0.03)$、$(10^\circ,0.05)$、$(14^\circ,0.07)$、$(20^\circ,0.10)$。

**遗忘度量。**采用标准遗忘度量FM[32]量化灾难性遗忘。令$a_{t,j}$为模型训练任务$t$后在任务$j$上的性能。最终任务$T$训练后，任务$j$的遗忘为：

$$f_j=\max_{t\in\{1,\ldots,T-1\}}a_{t,j}-a_{T,j}.\tag{3}$$

平均遗忘度量为：

$$\mathrm{FM}=\frac{1}{T-1}\sum_{j=1}^{T-1}f_j.\tag{4}$$

FM为正表示性能退化，即灾难性遗忘；FM为零表示旧任务零遗忘。

**计算效率。**报告模型规模（总参数量）、推理延迟（毫秒/图像）和峰值内存占用。这些因素直接决定在NVIDIA Jetson、Raspberry Pi等硬件上的边缘部署可行性。

## 4 DINOSaur：DINO空间异常无监督识别

**图2：DINOSaur架构概览。**上：训练时，冻结的DINOv3 ViT-S/16提取CLS与图像块token，通过CLS原型构建及贪心核心集选择，存入任务专用记忆库。下：推理时，将测试图像CLS token与已存原型比较以路由任务，再通过邻域限制的近邻搜索，将图像块token与所选记忆库比较并评分。

图内标注：训练；输入图像$224\times224\times3$；DINOv3（冻结）；CLS token $z\in\mathbb R^e$；图像块token $p_i\in\mathbb R^d$；记忆构建；CLS原型$\mu^t\in\mathbb R^e$；贪心核心集（逐空间位置）；任务记忆库（任务专用存储）。推理部分：测试图像$224\times224\times3$；测试CLS token $z^{\mathrm{test}}\in\mathbb R^e$；测试图像块token $p_i^{\mathrm{test}}\in\mathbb R^d$；任务路由（最近CLS原型）；记忆$\mathcal M^{(t^*)}$（选定任务库）；邻域搜索（半径$r$、局部$(2r+1)^2$区域）；图像块得分（$\sqrt N\times\sqrt N$图）；异常得分。红色虚线框为异常评分，紫色虚线框为记忆构建。

### 4.1 动机

现有CAD方法的架构越来越复杂：UCAD引入SAM[16]，IUF训练完整ViT编码器—解码器，DNE需要任务专用分布估计。然而，正如第5节所示，这种复杂性并不总能带来更好性能。基于记忆的CL机制始终取得最强结果[32]；即使在边缘设备上，工业场景的长期存储也很少成为约束。因此，我们假设强大的冻结特征提取器配合非参数记忆库已经足够：它既能取得有竞争力的性能，又因为不更新模型权重而从设计上实现零遗忘。由此提出DINOSaur：结合冻结特征提取器、空间结构化记忆及基于邻域的异常评分的极简基线，专为边缘部署设计。概览见图2。

### 4.2 使用DINOv3提取特征

DINOSaur使用冻结DINOv3 ViT-S/16[30]作为特征骨干。它是DINOv3家族中最小的变体，仅有2100万参数，用于证明强大的CAD性能不需要大型骨干。更大的ViT-B/16、ViT-L/16可能改善检测质量，但推理延迟和内存占用也会增加；我们有意选择最小骨干，以建立体现边缘部署约束的下界。选择ViT版本还因为任务识别与路由需要CLS token。DINOv3采用学生—教师蒸馏和Gram锚定进行自监督训练，既产生全局CLS token，也产生空间区分性强的图像块特征，在生成具有层次概念理解的嵌入向量方面被认为达到当前先进水平。

对输入图像$x\in\mathbb R^{H\times W\times3}$，冻结编码器$\Phi$产生图像块级特征向量：

$$\{p_1,p_2,\ldots,p_N\}=\Phi(x),\qquad p_i\in\mathbb R^d.\tag{5}$$

其中$N=(H/s)\times(W/s)$是由图像块尺寸$s$决定的图像块数，$d$是特征维度。提取Transformer最后一层的特征，因为它们包含对图像块的描述性、层次性与语义理解。

冻结骨干消除了灾难性遗忘的主要来源——权重更新；DINOv3的自监督特征则无需任务专用适配，即可提供强语义表示。

### 4.3 按空间位置索引的记忆库

DINOSaur不把所有图像块特征汇总到单个扁平记忆库，而是维护一种按空间位置索引的记忆：每个图像块位置有独立核心集。给定当前任务的$D$幅训练图像，在$\sqrt N\times\sqrt N$图像块网格的每个位置$(i,j)$，收集训练集中该位置观测到的全部特征向量：

$$P_{i,j}=\{p^{(k)}_{i,j}\mid k=1,\ldots,D\},\qquad|P_{i,j}|=D.\tag{6}$$

随后在各位置独立执行贪心核心集选择[27]，迭代选择距当前核心集最远的点，在缩小记忆库的同时获得有代表性的位置专用特征集合。由此得到每位置大小为$M=\max(20,\lfloor D\rho\rfloor)$的核心集$C_{i,j}$，其中核心集比例$\rho\in(0,1]$为超参数。整个任务记忆库是空间结构化张量$\mathcal M\in\mathbb R^{\sqrt N\times\sqrt N\times M\times e}$，$e$表示嵌入维度。这一结构在记忆中保留图像空间布局，是下述邻域评分的关键。

### 4.4 邻域限制的异常评分

推理时，首先把测试图像路由至适当的任务专用记忆库（见第4.5节）。给定所选记忆库$\mathcal M$，每个测试图像块只与局部空间邻域内的记忆特征比较。对位置$(i,j)$、特征为$p^{\mathrm{test}}_{i,j}$的图像块，将半径$r$内的图像块核心集并集定义为邻域记忆：

$$\mathcal N^r_{i,j}=\bigcup_{\substack{i'\in[i-r,i+r]\\j'\in[j-r,j+r]}}C_{i',j'}.\tag{7}$$

图像块级异常得分为：

$$s_{i,j}=\min_{c\in\mathcal N^r_{i,j}}\|p^{\mathrm{test}}_{i,j}-c\|_2.\tag{8}$$

图像级得分取所有图像块位置的最大值：

$$S(x_{\mathrm{test}})=\max_{(i,j)}s_{i,j}.\tag{9}$$

邻域限制强制空间一致性：根据某位置“应该出现什么”来检测异常，并将每块的比较次数从$N M$降低为至多$(2r+1)^2M$。在$14\times14$网格、默认$r=3$下，比较量减少为约四分之一。

### 4.5 通过任务专用记忆实现持续学习

DINOSaur通过两个互补机制处理持续学习：任务专用记忆库，以及用于无监督任务识别的CLS token原型。

**1）任务专用记忆。**不把所有特征累积到单个不断增长的记忆库中，而是为每个任务$t$维护独立、按空间位置索引的记忆库$\mathcal M^{(t)}$。新任务到达时，其正常训练图像经过冻结编码器处理，构建并独立保存新的逐位置核心集，以优选嵌入向量减小内存占用。由于特征提取器$\Phi$从不更新，可处理任务数主要受内存和存储约束，而不受模型架构限制。

**2）CLS token原型。**DINOv3还为每幅图像产生全局CLS token $z\in\mathbb R^e$。训练任务$t$时，计算所有训练图像CLS token的均值$\mu_t$，作为紧凑任务原型。测试时，无监督任务路由选择最近原型：

$$t^*=\arg\min_t\|z_{\mathrm{test}}-\mu_t\|_2.\tag{10}$$

随后使用该任务记忆库$\mathcal M^{(t^*)}$评分。推理时不需要任务标签，对每个已存任务仅计算一次距离。

## 5 实验

### 5.1 实验设置

**受评方法。**我们评价以下方法：

- CAD方法：DNE[19]、IUF[31]和UCAD[23]，即三种已发表的持续异常检测方法。
- 带回放的传统AD基线：PatchCore[27]和EfficientAD[5]，加入经验回放[26]。回放预算与增强策略见附录B.2。
- DINOSaur（本文）：第4节提出的基线。

对所有方法，有官方实现则使用官方实现，否则忠实重现，并匹配所报告超参数。所有方法在每个任务上训练300轮。

> 译注：以上“所有方法训练300轮”沿原文保留；附录B及表6另明确规定，PatchCore与DINOSaur仅做一轮前向特征提取，不执行梯度训练。

**训练协议。**所有实验均为无监督设置，训练时只使用正常样本，以反映真实工业场景。模型按任务顺序训练，不能访问先前任务数据，即标准CL约束；例外是为PatchCore与EfficientAD提供较宽裕的回放缓冲区。所有协议的任务顺序见附录C。训练在单张NVIDIA RTX 4090上完成；边缘推理效率在NVIDIA Jetson Orin Nano和Raspberry Pi 5上剖析，见第5.4节。

### 5.2 主要结果

表1给出所有受评方法在MVTec-AD和MVTec-LOCO上的图像级AUROC、准确率、召回率与遗忘度量（FM）。主要发现如下：

- DINOSaur与带回放的PatchCore在两种协议中占优，专用CAD方法表现不足。
- DNE与IUF退化为预测“正常”（原文称MVTec-AD上召回率为零），揭示无监督设置中的根本架构局限。
- 专用CAD方法中，仅UCAD表现出有意义的学习，但在MVTec-AD的AUROC仍比PatchCore低超过12个百分点。
- DINOSaur与UCAD实现零遗忘；PatchCore的少量遗忘表明，仅回放就能缓解灾难性遗忘。

> 译注：原文第二点将DNE和IUF均称为零召回；表1中IUF召回率实际列为0.074。保留原文论述与表格值，二者不一致。

DINOSaur和PatchCore虽然同处第一梯队，但二者存在根本差异。PatchCore将所有图像块特征汇入单个扁平记忆库，采用全局近邻搜索评分，并依赖每个先前任务100个样本的经验回放缓解新任务到来时的遗忘。DINOSaur不需要回放，其空间索引记忆及邻域限制评分（第4.4节）施加了扁平记忆库不具备的空间一致性。这一架构差异直接体现于边缘性能：DINOSaur在Jetson上比PatchCore快23%（91.6对119毫秒，表4），同时在每种协议上取得更高AUROC。此外，DINOSaur可通过核心集比例$\rho$调节：对于较小或变化较少的任务，将$\rho$降至默认10%以下，可进一步减少存储与延迟，而AUROC损失较小（表5），为使用者提供明确的效率—精度调节手段。我们的PatchCore*基线与PatchcoreCL[2]在机制上等价：两者均不更新权重，持续保留特征访问，并采用按任务组织的检索。除了上述空间索引和邻域评分，DINOSaur还通过显式CLS原型任务路由与二者区分。

连续漂移结果（表2）在不同漂移类型上表现出不同模式。颜色和模糊漂移中，DINOSaur与PatchCore明显优于专用CAD方法，证实连续偏移下强特征比专门CL机制更重要。值得注意的是，连续漂移并不总比离散任务切换容易：DINOSaur和PatchCore在MTD颜色、模糊漂移上的AUROC与MVTec-AD相当或更低，而较弱的DNE、IUF不论协议如何都接近随机水平，表明失败更可能源于架构的根本局限，而不是特定协议造成的假象。

**表1：离散任务结果。**在全部任务上平均的图像级AUROC（↑）、准确率（↑）、召回率（↑）、遗忘度量（↓）。*表示加入经验回放。原表各列最优结果用粗体标识。

| 方法 | MVTec-AD AUROC | 准确率 | 召回率 | FM | MVTec-LOCO AUROC | 准确率 | 召回率 | FM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DNE[19] | 0.500 | 0.277 | 0.000 | 0.012 | 0.500 | 0.368 | 0.000 | -0.022 |
| IUF[31] | 0.508 | 0.300 | 0.074 | -0.066 | 0.572 | 0.436 | 0.234 | 0.002 |
| UCAD[23] | 0.798 | 0.635 | 0.552 | 0.000 | 0.629 | 0.537 | 0.389 | 0.000 |
| PatchCore*[27] | 0.925 | 0.808 | 0.804 | 0.023 | 0.743 | 0.655 | 0.521 | 0.038 |
| EfficientAD*[5] | 0.529 | 0.310 | 0.071 | 0.051 | 0.485 | 0.384 | 0.053 | 0.025 |
| DINOSaur（本文） | 0.968 | 0.829 | 0.991 | 0.000 | 0.768 | 0.709 | 0.845 | 0.000 |

**表2：连续漂移结果。**MTD颜色、几何与模糊漂移上的图像级AUROC（↑）、准确率（↑）、召回率（↑）、FM（↓），在全部任务上平均。*表示经验回放。为便于阅读，将原表的三组列拆成三张表，行列数值不变。

| 方法 | MTD-Color AUROC | 准确率 | 召回率 | FM |
| --- | ---: | ---: | ---: | ---: |
| DNE[19] | 0.520 | 0.529 | 0.308 | 0.021 |
| IUF[31] | 0.521 | 0.532 | 0.294 | -0.008 |
| UCAD[23] | 0.566 | 0.591 | 0.141 | 0.030 |
| PatchCore*[27] | 0.836 | 0.710 | 0.530 | -0.043 |
| EfficientAD*[5] | 0.565 | 0.573 | 0.080 | -0.007 |
| DINOSaur（本文） | 0.928 | 0.836 | 0.861 | 0.000 |

| 方法 | MTD-Geo AUROC | 准确率 | 召回率 | FM |
| --- | ---: | ---: | ---: | ---: |
| DNE[19] | 0.480 | 0.517 | 0.342 | 0.032 |
| IUF[31] | 0.525 | 0.588 | 0.048 | 0.002 |
| UCAD[23] | 0.480 | 0.548 | 0.107 | 0.007 |
| PatchCore*[27] | 0.509 | 0.564 | 0.042 | -0.008 |
| EfficientAD*[5] | 0.510 | 0.567 | 0.050 | -0.004 |
| DINOSaur（本文） | 0.519 | 0.567 | 0.050 | 0.013 |

| 方法 | MTD-Blur AUROC | 准确率 | 召回率 | FM |
| --- | ---: | ---: | ---: | ---: |
| DNE[19] | 0.555 | 0.573 | 0.062 | -0.003 |
| IUF[31] | 0.516 | 0.567 | 0.042 | 0.001 |
| UCAD[23] | 0.582 | 0.586 | 0.137 | 0.003 |
| PatchCore*[27] | 0.832 | 0.751 | 0.685 | -0.022 |
| EfficientAD*[5] | 0.566 | 0.564 | 0.051 | -0.010 |
| DINOSaur（本文） | 0.891 | 0.810 | 0.676 | -0.002 |

几何失真尤其困难，所有方法均退化至接近随机水平。这是因为几何变换根本改变图像块特征之间的空间关系，破坏基于分布的方法依赖的结构。相比之下，颜色和模糊失真只改变外观而保留空间结构，更适合基于图像块的检测。这一区别有实际意义：涉及机械形变或透视变化的工业情景，例如传送带失准、相机漂移，可能需要与光度变化情景根本不同的表示。DINOSaur在全部漂移中保持近零遗忘（FM≤0.013）；颜色漂移中，PatchCore甚至出现负FM（-0.043），表明存在后向迁移：继续用偏移数据训练，反而改善了较早任务的检测。

### 5.3 像素级评价

表3并列报告图像级与像素级AUROC。在像素评分有意义的协议上，即MVTec-AD、MVTec-LOCO、MTD颜色和MTD模糊，二者相关系数$r=0.945$，并保持领先方法的排序，说明像素级性能很大程度上伴随图像级检测性能。DINOSaur在这四种协议上均保持像素级领先。几何变换的仿射填充主导合并统计后的指标，因此省略几何像素AUROC。DNE没有空间图，像素项未定义。

**表3：图像级/像素级AUROC（↑）。**全部任务平均。图像级数值复录自表1、2。“—”表示DNE无空间图，或几何漂移的像素评分不可靠而省略。

| 方法 | AD图像 | AD像素 | LOCO图像 | LOCO像素 | Color图像 | Color像素 | Geo图像 | Geo像素 | Blur图像 | Blur像素 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| DNE[19] | 0.500 | — | 0.500 | — | 0.520 | — | 0.480 | — | 0.555 | — |
| IUF[31] | 0.508 | 0.525 | 0.572 | 0.606 | 0.521 | 0.557 | 0.525 | — | 0.516 | 0.585 |
| UCAD[23] | 0.798 | 0.822 | 0.629 | 0.651 | 0.566 | 0.564 | 0.480 | — | 0.582 | 0.537 |
| PatchCore*[27] | 0.925 | 0.945 | 0.743 | 0.742 | 0.836 | 0.770 | 0.509 | — | 0.832 | 0.730 |
| EfficientAD*[5] | 0.529 | 0.402 | 0.485 | 0.429 | 0.565 | 0.599 | 0.510 | — | 0.566 | 0.636 |
| DINOSaur（本文） | 0.968 | 0.947 | 0.768 | 0.810 | 0.928 | 0.875 | 0.519 | — | 0.891 | 0.857 |

### 5.4 计算效率

**表4：两种边缘平台上的计算效率。**延迟为30次单图运行的均值±标准差。存储是在设备上测得的每任务大小；“—”表示不按任务建立索引存储的权重更新方法。DINOSaur存储对应MVTec-AD。IUF超过Jetson内存（OOM）。FPS为每秒帧数，*表示回放。

| 方法 | 参数量（百万） | 每任务存储MB | Jetson延迟ms | Jetson FPS | Pi 5延迟ms | Pi 5 FPS |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| DNE[19] | 86.6 | 2.3 | 60.7±4.9 | 16.5 | 560.0±6.1 | 1.8 |
| IUF[31] | 631.1 | — | OOM | — | 3668.6±69.6 | 0.3 |
| UCAD[23] | 36.0 | 1.2 | 956.5±7.3 | 1.0 | 1058.0±15.4 | 0.9 |
| PatchCore*[27] | 9.3 | 5.0 | 119.0±2.4 | 8.4 | 1074.5±7.9 | 0.9 |
| EfficientAD*[5] | 8.1 | — | 34.7±0.1 | 28.8 | 1318.2±4.3 | 0.8 |
| DINOSaur（本文） | 21.6 | 7.1 | 91.6±9.6 | 10.9 | 630.6±10.2 | 1.6 |

表4比较所有方法在两种代表性边缘平台上的计算需求。工业检测通常要求在GPU内存有限的硬件上每秒处理多幅图像；无论检测指标多好，无法满足这些约束的方法都不是可行的CAD方案。

专用CAD方法中，IUF完全超出Jetson GPU内存，在Raspberry Pi上每幅图像需3.7秒。UCAD依赖SAM[16]，在两种设备上每图均近1秒。DNE总体最为高效：在Raspberry Pi最快（560毫秒、1.8 FPS），在Jetson也有竞争力（60.7毫秒）；但其AUROC接近随机，实际中速度优势并无意义。

在具有有效检测能力的方法中，DINOSaur在Jetson推理低于100毫秒（91.6毫秒，10.9 FPS），快于PatchCore的119毫秒。EfficientAD更快（34.7毫秒），但其持续学习检测性能崩溃。在仅使用CPU的Raspberry Pi 5上，DINOSaur为630.6毫秒，仅次于DNE。每任务存储（MVTec-AD为7.1MB）随$\rho$和训练集大小变化，详见附录B.2。

**设备端训练可行性。**对边缘部署，训练成本与推理同样重要。基于梯度的方法（DNE、IUF、UCAD、EfficientAD）每任务需在数百轮中执行前向和反向传播，内存需求远超推理。DINOSaur对每幅图像仅执行一次前向，再选择核心集，即可适应新任务，支持不依赖云连接的自主设备端适配。

### 5.5 分析与讨论

在四种有有效信号的协议上，没有专用CAD方法取得最高AUROC；DINOSaur均领先，带回放PatchCore其次。几何漂移下，所有方法均处于随机水平。这提出根本问题：以往CAD论文报告的性能收益，究竟来自真正的算法进步，还是实验协议的产物？我们认为无监督设置是关键因素。本文所有实验只用正常样本训练，显著难于此前DNE和IUF某些评价中采用、包含合成异常生成的有监督设置。这些方法接近随机的性能表明，其架构可能依赖训练中的异常示例来学习有意义的决策边界，而真实工业部署很少具备这一条件。移除这一支撑后，专用CL机制相对简单回放没有可测收益；对边缘受限部署，额外架构复杂性没有回报。

**任务顺序不变性。**由于骨干冻结，各任务获得独立记忆库，DINOSaur的最终性能完全不受任务顺序影响。这是其他受评方法无法提供的结构性保证，因为它们要么更新权重，要么跨任务回放。这消除了CL评价中的混杂因素，并简化部署。扩展分析见附录D。

**DINOSaur超参数敏感性。**DINOSaur有两个超参数：核心集比例$\rho$和邻域半径$r$。对$\rho\in\{1\%,2.5\%,5\%,10\%,20\%\}$、$r\in\{0,1,2,3,4\}$做网格搜索，表5报告全部五种协议的平均AUROC。性能随$\rho$单调提高，但收益递减（10%到20%提升1.5%）；$r$的影响较小但一致。选择$\rho=10\%$、$r=3$，平衡检测质量与边缘效率（Jetson推理低于100毫秒）。完整逐数据集结果见附录。

> 译注：原文称完整逐数据集超参数结果见附录，但本地20页PDF附录未列此分解表；不据此补造表格。表5具体数值与“单调提高”的概括亦有局部出入，均保留。

**表5：超参数敏感性。**每种核心集比例和邻域半径组合在五种协议上的平均图像级AUROC；默认组合加粗。

| 核心集比例ρ / 邻域半径r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 0.771 | 0.773 | 0.773 | 0.780 | 0.778 |
| 2.5% | 0.765 | 0.770 | 0.775 | 0.776 | 0.782 |
| 5% | 0.771 | 0.775 | 0.777 | 0.781 | 0.785 |
| 10% | 0.787 | 0.790 | 0.794 | **0.801** | 0.798 |
| 20% | 0.803 | 0.816 | 0.819 | 0.816 | 0.822 |

## 6 结论

我们提出统一CAD基准，针对文献中评价不真实、缺乏系统比较、忽视边缘约束三个缺口，表明现有CAD方法并不能稳定优于使用简单经验回放的传统AD基线。由此提出DINOSaur，将冻结DINOv3骨干、空间索引核心集记忆和任务识别/路由结合，在五种基准协议中的四种上实现先进性能并从结构上保证零遗忘（几何漂移下所有方法均退化至随机）。其免训练设计可在NVIDIA Jetson Orin Nano上于30秒内进行设备端适配，适用于自主工业部署。

**未来工作。**仍有三个开放方向。第一，几何漂移使全部方法退化至随机，解决这一问题可能需要超越现有冻结骨干的几何等变特征表示。第二，DINOSaur的邻域评分本质上是局部的；需要整体场景理解的逻辑异常，可能受益于在空间记忆外加入全局推理。最后，将基准扩展至没有显式任务边界的在线数据流，可使评价更接近真实工业部署。


## 参考文献

[1] **记忆感知突触：学习什么应该（不应该）遗忘。** 原始书目：Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., Tuytelaars, T.: Memory aware synapses: Learning what (not) to forget. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds.) Computer Vision – ECCV 2018, vol. 11207, pp. 144–161. Springer International Publishing.https://doi.org/10.1007/978-3-030-01219-9_9, https://link.springer.com/10.1007/978-3-030-01219-9_9, series Title: Lecture Notes in Computer Science

[2] **迈向医疗领域的持续视觉异常检测。** 原始书目：Barusco, M., Borsatti, F., Beda, N., Pezze, D.D., Susto, G.A.: Towards continual visual anomaly detection in the medical domain. In: Proceedings of the International Workshop on Personalized Incremental Learning in Medicine (PILM) (2025).https://doi.org/10.1145/3746259.3760434,https://arxiv.org/abs/2508.18013, arXiv:2508.18013

[3] **PaSTe：提高边缘端视觉异常检测的效率。** 原始书目：Barusco, M., Borsatti, F., Pezze, D.D., Paissan, F., Farella, E., Susto, G.A.: PaSTe: Improving the efficiency of visual anomaly detection at the edge. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE (2025),https://openaccess.thecvf.com/content/CVPR2025W/VAND/html/ Barusco_PaSTe_Improving_the_Efficiency_of_Visual_Anomaly_Detection_at_the_CVPRW_2025_paper.html

[4] **面向边缘视觉异常检测的内存高效持续学习。** 原始书目：Barusco, M., D’Antoni, L., Pezze, D.D., Borsatti, F., Susto, G.A.: Memory efficient continual learning for edgebased visual anomaly detection (2025),https://arxiv.org/abs/2503.02691, arXiv:2503.02691

[5] **EfficientAD：毫秒级延迟的精确视觉异常检测。** 原始书目：Batzner, K., Heckler, L., König, R.: EfficientAD: Accurate visual anomaly detection at millisecond-level latencies. pp. 128–138.https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_ Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html

[6] **超越凹痕与划痕：无监督异常检测与定位中的逻辑约束。** 原始书目：Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., Steger, C.: Beyond dents and scratches: Logical constraints in unsupervised anomaly detection and localization130(4), 947–969.https://doi.org/10.1007/s11263-02201578-9,https://doi.org/10.1007/s11263-022-01578-9

[7] **MVTec AD：面向无监督异常检测的全面真实世界数据集。** 原始书目：Bergmann, P., Fauser, M., Sattlegger, D., Steger, C.: MVTec AD — a comprehensive real-world dataset for unsupervised anomaly detection. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 9584–9592. IEEE.https://doi.org/10.1109/CVPR.2019.00982,https://ieeexplore.ieee.org/ document/8954181/

[8] **揭示不断变化世界中的异常：持续学习中的像素级异常检测基准。** 原始书目：Bugarin, N., Bugaric, J., Barusco, M., Pezze, D.D., Susto, G.A.: Unveiling the anomalies in an ever-changing world: A benchmark for pixel-level anomaly detection in continual learning. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). pp. 4065–4074. IEEE.https://doi.org/10. 1109/CVPRW63382.2024.00410,https://ieeexplore.ieee.org/document/10678644/

[9] **面向通用持续学习的暗经验：强大而简单的基线。** 原始书目：Buzzega, P., Boschini, M., Porrello, A., Abati, D., CALDERARA, S.: Dark experience for general continual learning: a strong, simple baseline. In: Advances in Neural Information Processing Systems. vol. 33, pp. 15920–15930. Curran Associates, Inc.,https://proceedings.neurips.cc/paper/2020/hash/ b704ea2c39778f07c617f6b7ce480e9e-Abstract.html

[10] **PaDiM：面向异常检测与定位的图像块分布建模框架。** 原始书目：Defard, T., Setkov, A., Loesch, A., Audigier, R.: PaDiM: A patch distribution modeling framework for anomaly detection and localization. In: Del Bimbo, A., Cucchiara, R., Sclaroff, S., Farinella, G.M., Mei, T., Bertini, M., Escalante, H.J., Vezzani, R. (eds.) Pattern Recognition. ICPR International Workshops and Challenges. pp. 475–489. Springer International Publishing.https://doi.org/10.1007/978-3-030-68799-1_35

[11] **自动检测系统将质量控制成本降低60%。** 原始书目：Dimitriou, E., Georgiou, A.: Automatic inspection systems cut quality control costs by 60%2(1), 23–33,https: //theeducationjournals.com/index.php/NJQIBE/article/view/147

[12] **面向异常检测的终身持续学习：新挑战、视角与见解。** 原始书目：Faber, K., Corizzo, R., Sniezynski, B., Japkowicz, N.: Lifelong continual learning for anomaly detection: New challenges, perspectives, and insights12, 41364–41380.https://doi.org/10.1109/ACCESS.2024.3377690, https://ieeexplore.ieee.org/document/10473036/?arnumber=10473036, conference Name: IEEE Access

[13] **AnomalyGPT：使用大型视觉语言模型检测工业异常。** 原始书目：Gu, Z., Zhu, B., Zhu, G., Chen, Y., Tang, M., Wang, J.: AnomalyGPT: Detecting industrial anomalies using large vision-language models38(3), 1932–1940.https://doi.org/10.1609/aaai.v38i3.27963,https://ojs. aaai.org/index.php/AAAI/article/view/27963

[14] **磁瓦表面缺陷的显著性。** 原始书目：Huang, Y., Qiu, C., Yuan, K.: Saliency of magnetic tile surface defects36(1), 85–96.https://doi.org/10.1007/ s00371-018-1588-5,https://doi.org/10.1007/s00371-018-1588-5

[15] **磁瓦表面缺陷显著性。** 原始书目：Huang, Y., Qiu, C., Yuan, K.: Surface defect saliency of magnetic tile. The Visual Computer36(1), 85–96 (2020). https://doi.org/10.1007/s00371-018-1588-5

[16] **分割一切。** 原始书目：Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollar, P., Girshick, R.: Segment anything. pp. 4015–4026.https://openaccess.thecvf.com/content/ ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html

[17] **克服神经网络中的灾难性遗忘。** 原始书目：Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., Hadsell, R.: Overcoming catastrophic forgetting in neural networks114(13), 3521–3526.https://doi.org/10.1073/pnas.1611835114, https://pnas.org/doi/full/10.1073/pnas.1611835114

[18] **Continual-MEGA：面向可泛化持续异常检测的大规模基准。** 原始书目：Lee, G., Oh, Y., Jang, G., Lee, S., Song, J., Cha, S., Yoo, Y.: Continual-MEGA: A large-scale benchmark for generalizable continual anomaly detection.https://doi.org/10.48550/arXiv.2506.00956,http://arxiv.org/ abs/2506.00956, version: 1

[19] **迈向工业异常检测中的持续适应。** 原始书目：Li, W., Zhan, J., Wang, J., Xia, B., Gao, B.B., Liu, J., Wang, C., Zheng, F.: Towards continual adaptation in industrial anomaly detection. In: Proceedings of the 30th ACM International Conference on Multimedia. pp. 2871–2880. MM ’22, Association for Computing Machinery.https://doi.org/10.1145/3503161.3548232, https://doi.org/10.1145/3503161.3548232

[20] **One-for-More：面向异常检测的持续扩散模型。** 原始书目：Li, X., Tan, X., Chen, Z., Zhang, Z., Zhang, R., Guo, R., Jiang, G., Chen, Y., Qu, Y., Ma, L., Xie, Y.: One-for-more: Continual diffusion model for anomaly detection. pp. 4766–4775.https://openaccess.thecvf. com/content/CVPR2025/html/Li_One-for-More_Continual_Diffusion_Model_for_Anomaly_Detection_CVPR_ 2025_paper.html

[21] **超越不遗忘：带后向知识迁移的持续学习。** 原始书目：Lin, S., Yang, L., Fan, D., Zhang, J.: Beyond not-forgetting: Continual learning with backward knowledge transfer35, 16165–16177,https://proceedings.neurips.cc/paper_files/paper/2022/hash/ 6728fcf94660c59c938319a6833a6073-Abstract-Conference.html

[22] **TRGP：用于持续学习的信赖域梯度投影。** 原始书目：Lin, S., Yang, L., Fan, D., Zhang, J.: TRGP: Trust region gradient projection for continual learning. In: International Conference on Learning Representations (ICLR).https://openreview.net/forum?id=iEvAf8i6JjO

[23] **利用对比学习提示实现无监督持续异常检测。** 原始书目：Liu, J., Wu, K., Nie, Q., Chen, Y., Gao, B.B., Liu, Y., Wang, J., Wang, C., Zheng, F.: Unsupervised continual anomaly detection with contrastively-learned prompt38(4), 3639–3647.https://doi.org/10.1609/aaai.v38i4. 28153,https://ojs.aaai.org/index.php/AAAI/article/view/28153

[24] **面向异常检测的持续学习方法。** 原始书目：Pezze, D.D., Anello, E., Masiero, C., Susto, G.A.: Continual learning approaches for anomaly detection16(4), 111.https://doi.org/10.1007/s12530-025-09732-7,https://doi.org/10.1007/s12530-025-09732-7

[25] **PANDA：为异常检测与分割适配预训练特征。** 原始书目：Reiss, T., Cohen, N., Bergman, L., Hoshen, Y.: PANDA: Adapting pretrained features for anomaly detection and segmentation. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2805– 2813. IEEE.https://doi.org/10.1109/CVPR46437.2021.00283,https://ieeexplore.ieee.org/document/ 9577693/

[26] **用于持续学习的经验回放。** 原始书目：Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., Wayne, G.: Experience replay for continual learning. In: Advances in Neural Information Processing Systems. vol. 32. Curran Associates, Inc.,https://papers.nips.cc/ paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html

[27] **迈向工业异常检测的完全召回。** 原始书目：Roth, K., Pemula, L., Zepeda, J., Scholkopf, B., Brox, T., Gehler, P.: Towards total recall in industrial anomaly detection. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 14298– 14308. IEEE.https://doi.org/10.1109/CVPR52688.2022.01392,https://ieeexplore.ieee.org/document/ 9879738/

[28] **渐进神经网络。** 原始书目：Rusu, A.A., Rabinowitz, N.C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., Pascanu, R., Hadsell, R.: Progressive neural networks.https://doi.org/10.48550/arXiv.1606.04671,http://arxiv.org/abs/1606. 04671

[29] **用于持续学习的梯度投影记忆。** 原始书目：Saha, G., Garg, I., Roy, K.: Gradient projection memory for continual learning. In: International Conference on Learning Representations (ICLR).https://openreview.net/forum?id=3AOj0RCNC2

[30] **DINOv3。** 原始书目：Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., Bojanowski, P.: DINOv3.https://doi.org/10.48550/arXiv.2508.10104,http://arxiv.org/abs/2508.10104

[31] **面向小缺陷检测的增量统一框架。** 原始书目：Tang, J., Lu, H., Xu, X., Wu, R., Hu, S., Zhang, T., Cheng, T.W., Ge, M., Chen, Y.C., Tsung, F.: An incremental unified framework for small defect inspection. In: Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler, T., Varol, G. (eds.) Computer Vision – ECCV 2024. pp. 307–324. Springer Nature Switzerland.https://doi.org/ 10.1007/978-3-031-72751-1_18

[32] **持续学习综合综述：理论、方法与应用。** 原始书目：Wang, L., Zhang, X., Su, H., Zhu, J.: A comprehensive survey of continual learning: Theory, method and application46(8), 5362–5383.https://doi.org/10.1109/TPAMI.2024.3367329,https://ieeexplore.ieee.org/ document/10444954

[33] **AnoDDPM：使用单纯形噪声的去噪扩散概率模型进行异常检测。** 原始书目：Wyatt, J., Leach, A., Schmon, S.M., Willcocks, C.G.: AnoDDPM: Anomaly detection with denoising diffusion probabilistic models using simplex noise. pp. 650–656.https://openaccess.thecvf.com/content/CVPR2022W/ NTIRE/html/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_ Simplex_CVPRW_2022_paper.html

[34] **IM-IAD：制造业工业图像异常检测基准。** 原始书目：Xie, G., Wang, J., Liu, J., Lyu, J., Liu, Y., Wang, C., Zheng, F., Jin, Y.: IM-IAD: Industrial image anomaly detection benchmark in manufacturing54(5), 2720–2733.https://doi.org/10.1109/TCYB.2024.3357213, https://ieeexplore.ieee.org/document/10443076/, conference Name: IEEE Transactions on Cybernetics

[35] **面向多类别异常检测的统一模型。** 原始书目：You, Z., Cui, L., Shen, Y., Yang, K., Lu, X., Zheng, Y., Le, X.: A unified model for multi-class anomaly detection35, 4571–4584,https://proceedings.neurips.cc/paper_files/paper/2022/hash/ 1d774c112926348c3e25ea47d87c835b-Abstract-Conference.html

[36] **DRÆM：用于表面异常检测的判别式训练重建嵌入。** 原始书目：Zavrtanik, V., Kristan, M., Skocaj, D.: DRÆM – a discriminatively trained reconstruction embedding for surface anomaly detection. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 8310–8319. IEEE.https://doi.org/10.1109/ICCV48922.2021.00822,https://ieeexplore.ieee.org/document/9710329/

[37] **通过突触智能实现持续学习。** 原始书目：Zenke, F., Poole, B., Ganguli, S.: Continual learning through synaptic intelligence. In: Proceedings of the 34th International Conference on Machine Learning. pp. 3987–3995. PMLR,https://proceedings.mlr.press/v70/ zenke17a.html

[38] **研究多模态大语言模型中的灾难性遗忘。** 原始书目：Zhai, Y., Tong, S., Li, X., Cai, M., Qu, Q., Lee, Y.J., Ma, Y.: Investigating the catastrophic forgetting in multimodal large language models. In: Conference on Parsimony and Learning, Proceedings of Machine Learning Research. vol. 234, pp. 202–227.https://proceedings.mlr.press/v234/zhai24a.html

[39] **防止视觉语言模型在持续学习中的零样本迁移退化。** 原始书目：Zheng, Z., Ma, M., Wang, K., Qin, Z., Yue, X., You, Y.: Preventing zero-shot transfer degradation in continual learning of vision-language models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Preventing_Zero-Shot_ Transfer_Degradation_in_Continual_Learning_of_Vision-Language_Models_ICCV_2023_paper.pdf

[40] **AnomalyCLIP：用于零样本异常检测的对象无关提示学习。** 原始书目：Zhou, Q., Pang, G., Tian, Y., He, S., Chen, J.: AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection.https://openreview.net/forum?id=buC4E91xZE

[41] **通过少样本示例提示的上下文残差学习迈向通用异常检测。** 原始书目：Zhu, J., Pang, G.: Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts. pp. 17826–17836.https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Toward_ Generalist_Anomaly_Detection_via_In-context_Residual_Learning_with_Few-shot_CVPR_2024_paper. html

## 附录

本附录提供正文引用的支持材料：扩展方法说明（A）、详细训练流程（B）、逐任务数据集规格（C）、扩展分析（D），以及DINOSaur完整算法伪代码（E）。

### A 扩展相关工作

本节扩展正文第2节的方法说明。大规模异常检测模型在正文中讨论（原文标作第2.1节）；这里提供三种专用持续异常检测方法的额外架构细节。

#### A.1 持续异常检测方法

**DNE（正常嵌入分布）[19]。**DNE提出首个CAD框架。训练各任务时，使用视觉Transformer ViT-B/16骨干提取特征嵌入，为各任务保存嵌入分布的均值向量、收缩协方差矩阵和样本数。推理时，根据已存逐任务统计量重构全局分布，通过马氏距离计算异常得分。DNE仅在图像级运行，不产生像素分割图。值得注意的是，特征嵌入没有被正则化为任务专用，因此随着任务数增加，保存的统计量可能无法准确刻画各任务独特的分布。

**IUF（增量统一框架）[31]。**IUF采用含ViT组件的编码器—解码器架构，嵌入维度64、4个注意力头、4层。潜空间通过V投影机制显式正则化，将各任务分布分离至不重叠区域。梯度更新通过权重正则项（$\beta=0.5$）约束，惩罚偏离前一任务参数的变化，以保护旧任务重要参数。训练目标结合重建损失、判别器损失和鼓励低秩潜表示的奇异值惩罚。

**UCAD（无监督持续异常检测）[23]。**UCAD结合对比学习和用于任务识别、适配的提示机制。利用分割一切模型SAM[16]分解图像纹理，再通过对比学习将任务专用特征分布保存于知识库用于推理。维护三种记忆组件：键记忆（最远点采样得到196个代表图像块）、知识记忆（贪心核心集选择得到196块），以及提示记忆（学习到的任务专用提示）。推理时使用键记忆识别任务，将测试图像路由到相应知识库。

### B 训练流程

所有实验共享训练框架。下面介绍公共配置及逐方法细节。

#### B.1 公共配置

各方法按固定顺序逐任务训练（各数据集顺序见附录C）。所有方法使用Adam，学习率$7.5\times10^{-4}$，权重衰减$10^{-4}$，训练批量12。输入缩放至$224\times224$像素，EfficientAD例外，使用$256\times256$；使用ImageNet统计量归一化：$\mu=[0.485,0.456,0.406]$，$\sigma=[0.229,0.224,0.225]$。全部训练在单张NVIDIA RTX 4090（24GB显存）上完成。

对支持经验回放的PatchCore和EfficientAD，每个新任务训练时回放来自旧任务的100个正常训练样本。回放样本不施加数据增强。DINOSaur不采用经验回放：冻结骨干与任务专用记忆库消除了回放旧数据的必要性。

#### B.2 各方法训练细节

表6汇总各方法关键训练参数。

**表6：所有方法的训练超参数与关键架构细节。**轮数表示每任务完整遍历训练集的次数。†仅需一轮前向特征提取，无梯度优化。*表示经验回放。

| 项目 | DNE | IUF | UCAD | PatchCore* | EfficientAD* | DINOSaur |
| --- | --- | --- | --- | --- | --- | --- |
| 骨干 | ViT-B/16 | 自定义ViT | 自定义ViT | ResNet-18 | PDN | DINOv3 ViT-S/16 |
| 骨干冻结 | 否 | 否 | 是 | 是 | 部分 | 是 |
| 嵌入维度 | 768 | 64 | 768 | 512 | 384 | 384 |
| 每任务轮数 | 300 | 300 | 300 | 1† | 300 | 1† |
| 损失 | 交叉熵 | 复合损失 | 对比损失 | — | 三部分MSE | — |
| 回放 | 否 | 否 | 否 | 是（100） | 是（100） | 否 |
| 可训练参数量（百万） | 85.98 | 631.07 | 仅提示 | 0.79 | 5.38 | 0.00 |

> 译注：表6将PatchCore可训练参数列为0.79百万，而下文明确称其无梯度更新；按原表保留，不擅自修正。

**DNE。**DNE采用线性分类头（768→2）和交叉熵，每任务微调ViT-B/16骨干300轮，总参数86.57百万。首任务后分类头冻结，后续任务只更新骨干。每任务训练后保存768维均值与$768\times768$收缩协方差矩阵，收缩系数$\alpha=0.5$，每任务约2.3MB。推理时根据全部任务统计量重建全局分布，以马氏距离评分。异常阈值为训练集异常得分第97.5百分位。

**IUF。**IUF训练编码器—解码器（631.07百万参数），使用复合损失：

$$L_{\mathrm{IUF}}=\lambda_1L_{\mathrm{recon}}+\lambda_2L_{\mathrm{disc}}+\lambda_3L_{\mathrm{svd}}.\tag{11}$$

其中$\lambda_1=1.0$、$\lambda_2=0.5$、$\lambda_3=8.0$。奇异值损失$L_{\mathrm{svd}}$惩罚超过截断值$t=1$的奇异值，鼓励低秩潜表示。每任务后保存模型权重；后续训练中，正则项（$\beta=0.5$）将参数拉向前任务权重。IUF仅在无监督设置下运行，实验未采用其有监督路径。异常阈值为第97.5百分位。

**UCAD。**UCAD使用冻结ViT骨干，以对比损失训练形状$(5,768)$的可学习提示参数：

$$L_{\mathrm{UCAD}}=\lambda_\alpha L_{\mathrm{neg}}-\lambda_\beta L_{\mathrm{pos}}.\tag{12}$$

其中$\lambda_\alpha=\lambda_\beta=1.0$，损失分量使用图像块特征之间的余弦相似度。每任务训练300轮后，更新键记忆（最远点采样196块）、知识记忆（贪心核心集196块）以及已学习提示。推理时，知识库中的5个近邻决定图像块异常得分，再上采样至原图分辨率。阈值设为第97.5百分位。

**PatchCore。**使用冻结ResNet-18骨干（9.33百万参数），每任务仅需一轮特征提取，无梯度更新。有意选择较小ResNet-18，以建立体现边缘部署条件的下界，与DINOSaur选择ViT-S/16一致。拼接中间层特征，经随机线性投影至512维，再通过自适应平均池化层（核3、步长1、填充1），产生每图$28\times28$空间特征图。记忆库保留累计特征的5%贪心核心集，至少500个样本。推理异常得分为距最近核心集条目的距离，并由该匹配的3个近邻重新加权。阈值为第97.5百分位。

**EfficientAD。**训练学生图像块描述网络（PDN）和自编码器，蒸馏预训练教师PDN知识。教师在ImageNet上采用独立Adam优化器（学习率$10^{-3}$），批量16，预训练60,000次迭代；使用ResNet34特征，并通过Welford运行统计量归一化，该统计量在10,000个样本上计算。学生与自编码器每任务联合训练300轮，采用三部分MSE损失：

$$L_{\mathrm{EAD}}=L_{\mathrm{ST}}+L_{\mathrm{AE}}+L_{\mathrm{STAE}}.\tag{13}$$

$L_{\mathrm{ST}}$使用困难样本挖掘（$p_{\mathrm{hard}}=0.999$），$L_{\mathrm{AE}}$将教师特征蒸馏给自编码器，$L_{\mathrm{STAE}}$对齐学生与自编码器输出。学生输出768维特征，分成两个384维部分；最终异常图等权（各0.5）结合学生—教师和学生—自编码器差异。学生/教师输入为$256\times256$，ResNet34特征提取器输入为$512\times512$。阈值为第97.5百分位。

**DINOSaur。**采用完全冻结的DINOv3 ViT-S/16骨干，21.6百万参数、可训练参数为0，每任务只需一轮特征提取。骨干在$14\times14$网格上输出196个384维图像块token，另输出384维CLS token。逐位置贪心核心集保留$\rho=10\%$图像块特征，每位置至少20个；保存CLS均值作为任务原型。推理时测试CLS与所有已存原型比较以路由任务，图像块在$r=3$的空间邻域（$7\times7$窗口）中评分。图像得分取块得分最大值，异常阈值为训练得分第97.5百分位。$\rho$与$r$通过$\{1\%,2.5\%,5\%,10\%,20\%\}\times\{0,1,2,3,4\}$网格搜索选择，详见正文；完整算法见附录E。

**回放策略细节。**对PatchCore和EfficientAD，每新任务训练时，从此前任务提供100个正常训练样本作为较宽裕的回放预算，使回放基线有充分机会保留旧知识。在MVTec-AD与MVTec-LOCO离散基准中，典型类别有200—400张训练图，100张占原训练集25%—50%。回放图像不增强。MTD连续漂移协议中，扩大增强参数范围，使其覆盖此前任务训练分布，为传统方法提供充分历史信息，从而隔离专用CL机制的作用。DINOSaur不回放：骨干冻结、记忆库逐任务独立，旧数据没有影响新任务记忆的机制，也不存在需要缓解的遗忘。

**DINOSaur逐任务存储。**存储取决于训练集大小$D$和核心集比例$\rho$。196个位置各保存$M=\max(20,\lfloor D\rho\rfloor)$个384维向量，另加384维CLS原型。使用32位浮点时，理论字节数为$196\times M\times384\times4+384\times4$，$M=20$时约5.8MB。全文报告设备实测存储。

当$\rho=10\%$时，多数MVTec-AD类别因$D<200$而受最小核心集20限制；较大类别（如hazelnut，$D=391$、$M=39$）与MTD（$D=443$、$M=44$）超过该下限。更高核心集比例或更大数据集会成比例增加存储。若每任务有数千张训练图，存储也相应增长，使用者应权衡核心集比例与目标边缘设备可用RAM。

**表7：ρ=10%时DINOSaur在各基准的任务存储。**D为每任务近似训练图像数。$\lfloor D\rho\rfloor<20$时使用每位置至少20个的核心集。存储是设备实测，包含序列化开销。

| 数据集 | 任务数 | 每任务D（约） | 每位置M | 总存储MB |
| --- | ---: | --- | --- | ---: |
| MVTec-AD | 15 | 60—391 | 20—39 | 106.8 |
| MVTec-LOCO | 5 | 357—572 | 35—57 | 35.6 |
| MTD（全部漂移类型） | 10 | 443 | 44 | 71.2 |

### C 数据集与任务规格

#### C.1 MVTec-AD

MVTec-AD[7]含15类工业检测对象与纹理。各类别独立为任务，按字母顺序训练：bottle（瓶子）、cable（电缆）、capsule（胶囊）、carpet（地毯）、grid（网格）、hazelnut（榛子）、leather（皮革）、metal_nut（金属螺母）、pill（药片）、screw（螺钉）、tile（瓷砖）、toothbrush（牙刷）、transistor（晶体管）、wood（木材）、zipper（拉链）。训练集只有无缺陷正常图像；测试集既有正常图，也有每类别多种缺陷。离散类别边界代表传统CAD评价范式。

#### C.2 MVTec-LOCO

MVTec-LOCO[6]在MVTec-AD基础上扩展出5类，专用于评价不能仅由局部特征比较检测的逻辑和结构异常。类别按字母序训练。除局部纹理异常外，该集还检验方法能否识别缺件、错位部件等需要全局推理的异常。

#### C.3 增强MTD

磁瓦缺陷MTD[15]提供单个产品类别、多种缺陷。为模拟真实连续漂移，对正常训练图像施加受控失真，设置10个递增强度，每级定义一个任务。任务按失真严重程度升序训练，模拟工业环境的渐进漂移。

**颜色漂移。**通过随机调整亮度、对比度和饱和度施加颜色失真。每个任务定义狭窄、不重叠的强度带，保证后续任务的失真逐渐增强且有界，不与此前任务重叠。10个区间为$[0.0,0.05]$、$[0.05,0.1]$、$[0.1,0.15]$、$[0.15,0.2]$、$[0.2,0.25]$、$[0.25,0.3]$、$[0.3,0.35]$、$[0.35,0.4]$、$[0.4,0.45]$、$[0.45,0.5]$。每幅训练图从当前带内均匀采样扰动幅度$v$，同时以乘法因子$(1\pm v)$作用于亮度、对比度和饱和度。这样，各任务位于失真空间不同区域，防止后续任务生成近似原图的图像。

**模糊漂移。**高斯模糊采用10组逐渐加重的核—标准差参数：$(1,0.5)$、$(3,1.0)$、$(5,1.5)$、$(7,2.0)$、$(9,2.5)$、$(11,3.0)$、$(13,3.5)$、$(15,4.0)$、$(17,4.5)$、$(19,5.0)$。首项为核尺寸，第二项为标准差。

**几何漂移。**同时结合旋转、平移、缩放和剪切，并逐渐增大幅度。10级名义参数元组（旋转度数、平移像素、尺度、剪切度数）为$(2,1,0.01,1)$、$(4,2,0.02,2)$、$(6,3,0.03,3)$、$(8,4,0.04,4)$、$(10,5,0.05,5)$、$(12,6,0.06,6)$、$(14,7,0.07,7)$、$(16,8,0.08,8)$、$(18,9,0.09,9)$、$(20,10,0.10,10)$。并不直接施加名义值，而从名义值下方的窄窗口采样实际参数：旋转$[\mathrm{nom}-2,\mathrm{nom}]$，各轴平移$[\mathrm{nom}-1,\mathrm{nom}]$，尺度$[\mathrm{nom}-0.01,\mathrm{nom}]$，剪切$[\mathrm{nom}-1,\mathrm{nom}]$。参数在任务初始化时采样一次，该任务全部图像保持相同参数，模拟稳定但发生偏移的工况，例如传送带逐渐失准。窄窗口设计保证各任务占据有界几何失真区域，不与此前任务重叠。

### D 扩展分析

#### D.1 详细遗忘分析

**MVTec-AD。**DINOSaur和UCAD均零遗忘（FM=0.000），原因根本不同：前者通过冻结骨干在设计上防止灾难性遗忘，后者通过任务专用机制隔离所学分布。IUF表现出较强后向迁移（-0.066），意味着继续训练实际上改善早期任务性能；但因AUROC接近随机，此现象意义有限。PatchCore较小的遗忘（0.023）说明，即使没有专门CL组件，经验回放也能有效缓解灾难性遗忘，尤其当基础方法较强时。

**MVTec-LOCO。**全部方法遗忘通常较低。DINOSaur与UCAD再次零遗忘；PatchCore（0.038）和EfficientAD（0.025）有少量退化。DNE负FM（-0.022）表示后向迁移，但接近随机的AUROC限制了实际意义。

**增强MTD。**所有方法在MTD上的遗忘度量都低于离散任务基准。颜色和模糊漂移中，PatchCore出现负FM（-0.043、-0.022），意味着继续在偏移数据上训练改善早期任务检测。DINOSaur各类漂移中近零遗忘（FM≤0.013），与其冻结骨干设计一致。

#### D.2 CL组件有帮助吗？扩展讨论

最引人注目的发现是，专门CL机制并不能稳定改善采用简单经验回放的传统AD方法。在MVTec-AD中，DNE（0.500）和IUF（0.508）接近随机；UCAD（0.798）有意义地学习，但仍远低于带回放PatchCore（0.925）和DINOSaur（0.968）。连续漂移也类似：DINOSaur和PatchCore主导颜色与模糊漂移；几何漂移下所有方法困难，IUF略领先（0.525，对比DINOSaur 0.519），但全部仍在随机水平，没有方法具有实质领先。四种有有效信号的协议中，专用CAD方法均未取得最高AUROC。

这提出根本问题：此前CAD论文报告的收益是真正算法进步，还是实验协议产物？以下因素可能造成差异：

- 既有研究可能受益于针对各自协议的大量超参数调优。
- 离散MVTec-AD类别协议可能不是评估CL能力的合适试验场，因为任务切换过于极端，即使一般的CL机制相对非CL基线也显得有效。
- 最关键的是，本文全部实验为只有正常训练样本的无监督设置，显著难于DNE、IUF此前某些评价采用的有监督设置。它们接近随机的性能表明，架构可能依赖异常训练示例学习有意义的决策边界，而真实工业部署很少拥有这一条件。
- 考虑边缘计算约束后，专门CL机制增加的复杂性没有明确回报。

#### D.3 对漂移类型的鲁棒性：扩展分析

比较两种评价范式可见，连续漂移并非总比离散任务切换容易。较强方法DINOSaur与PatchCore在MTD颜色、模糊漂移的AUROC与MVTec-AD相当或更低；几何漂移中所有方法接近随机。较弱CAD方法DNE、IUF不论协议都接近随机，说明失败并不依赖特定协议，而更可能是其架构在无监督设置中的根本局限。

几何失真最困难，可能因为它根本改变图像块特征间的空间关系。颜色和模糊只改变外观而保持空间结构，更适合基于分布的检测方法。实际意义是，涉及机械形变、透视变化的工业场景，例如传送带失准与相机漂移，可能需要与光度变化情景根本不同的表示。

### E DINOSaur算法

算法1、2给出完整训练与推理流程。符号沿用正文：$\Phi$为冻结DINOv3 ViT-S/16编码器，$p_{i,j}$为位置$(i,j)$的图像块特征，$z$为CLS token，$\mathcal M^{(t)}$为任务$t$的空间索引记忆库，$C_{i,j}$为该位置核心集，$\mu_t$为任务CLS原型，$\mathcal N^r_{i,j}$为半径$r$内的邻域记忆，$\rho$为核心集比例。

**算法1：DINOSaur训练（任务t）。**

输入：训练集$D_t=\{x_1,\ldots,x_{|D_t|}\}$、冻结编码器$\Phi$、核心集比例$\rho$。  
输出：记忆库$\mathcal M^{(t)}$、CLS原型$\mu_t$。

~~~text
/* 特征提取（原文注释：式4） */
1  对每幅训练图像 x_k ∈ D_t：
2      {p_1^(k), ..., p_N^(k)}, z^(k) ← Φ(x_k)
3  结束循环
/* 计算CLS原型 */
4  μ_t ← (1 / |D_t|) Σ_(k=1)^|D_t| z^(k)
/* 按空间位置索引的核心集选择（原文注释：式5） */
5  对 √N × √N 网格的每个空间位置 (i,j)：
6      P_(i,j) ← {p_(i,j)^(k) | k=1,...,|D_t|}
7      M ← max(20, floor(|D_t|·ρ))
8      C_(i,j) ← GreedyCoreset(P_(i,j), M)
9  结束循环
10 M^(t) ← {C_(i,j)}_(i,j)    // 保存空间索引记忆库
11 返回 M^(t), μ_t
~~~

**算法2：DINOSaur推理。**

输入：测试图像$x_{\mathrm{test}}$、冻结编码器$\Phi$、记忆库$\{\mathcal M^{(t)}\}_{t=1}^T$、原型$\{\mu_t\}_{t=1}^T$、邻域半径$r$。  
输出：图像级异常得分$S(x_{\mathrm{test}})$。

~~~text
/* 特征提取 */
1  {p_1^test,...,p_N^test}, z^test ← Φ(x_test)
/* 通过CLS原型进行任务路由（原文注释：式9） */
2  t* ← argmin_t ||z^test - μ_t||_2
/* 邻域限制异常评分（原文注释：式6—8） */
3  对 √N × √N 网格的每个空间位置 (i,j)：
4      N_(i,j)^r ← ⋃_(i'∈[i-r,i+r], j'∈[j-r,j+r]) C_(i',j')^(t*)
                            // 邻域记忆
5      s_(i,j) ← min_(c∈N_(i,j)^r) ||p_(i,j)^test - c||_2
                            // 图像块异常得分
6  结束循环
7  S(x_test) ← max_(i,j) s_(i,j)    // 图像级得分
8  返回 S(x_test)
~~~

> 译注：算法伪代码中的公式索引较正文少1：算法1注释“式4、5”实际对应正文式(5)、(6)，算法2“式9”实际对应式(10)，“式6—8”实际对应式(7)—(9)。以上保留原注释并说明对应关系。
