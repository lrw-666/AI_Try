# 面向复杂背景表面缺陷分割的变化感知孪生网络

**原文：**[Siamese network with change awareness for surface defect segmentation in complex backgrounds](<11-CADNet-Reference-Defect-Segmentation-2025.pdf>)。本译文对应本目录的 19 页出版社 PDF，*Scientific Reports* 15，11814（2025），DOI：10.1038/s41598-025-94733-4。翻译日期：2026-11-07；按本地 PDF 复核补全：2026-11-11。

**作者：**Biyuan Liu¹、Sijie Luo¹、Huiyao Zhan²、Yicheng Zhou³、Zhou Huang³˒*、Huaixin Chen¹˒*。

**单位：**¹电子科技大学资源与环境学院，中国成都 611731；²华南师范大学，中国汕尾 516600；³四川长虹电器股份有限公司，中国绵阳 621000。*通信邮箱：chowhuang23@gmail.com；huaixinchen@outlook.com。

**翻译说明：**覆盖本地 PDF 第 1—19 页全部正文、公式（1）—（16）、图 1—12 的图注及关键图内标注、表 1—7 全部行列数值、数据可用性、参考文献 [1]—[70] 和文末声明。原文没有另列补充附录；第 15 页是图 10 的整页展示。合并跨页、跨行续段，去除运行页眉页脚和孤立页码。原始图形不重绘，图中重要标注译出，并将带数值的关键图注信息列出，图形本身请结合上方 PDF 查看。作者、模型、数据集、期刊等专名保留；参考文献提供中文题名及完整原始书目。原文中公式索引、数据统计及实验差值的不一致以译注明确标出，不静默修正数值。论文的“最佳”“优于”等判断均保留其发表时的语境。

## 摘要

尽管深度视觉网络在区域级表面缺陷检测方面取得显著进展，但缺陷外观多样且可用数据有限，高质量逐像素缺陷检测仍具有挑战性。为解决过度依赖缺陷外观的问题并提高缺陷分割准确率，我们提出基于 Transformer 的变化感知孪生网络，将复杂背景下的缺陷分割表述为变化检测，以模拟人工检测过程。具体而言，我们引入新的多类别平衡对比损失来引导 Transformer 编码器，使其将各类缺陷编码为缺陷图像与无缺陷图像之间统一且类别无关的差异。这一差异通过距离图表示，再以跳跃连接方式送入变化感知解码器，辅助逐像素缺陷定位。此外，我们构建了一个合成数据集，在复杂且不连续的背景上下文中包含多类别液晶显示器（LCD）缺陷。在所提数据集及两个公开数据集上的评估表明，模型保持相对紧凑的规模，同时优于领先的语义分割方法。此外，在各种监督设置下，与半监督方法相比，我们的模型取得了新的最佳性能。代码和数据集见：<https://github.com/HATFormer/CADNet>。

**关键词：**表面缺陷分割；变化感知解码器；孪生网络；对比学习；基于 Transformer 的编码器。

## 引言

表面缺陷检测在制造业中至关重要，可避免潜在质量问题、经济损失甚至安全隐患。缺陷可能表现为污物、斑点、断裂等多种形式，常见于钢材 [1,2]、LED [3] 和磁瓦 [4] 等多类工业产品。与语义物体不同，表面缺陷通常没有规则形状、明确含义，或与背景连续的上下文，这使经验设计方法的应用更加困难 [5]。为了实现缺陷检测自动化，基于深度学习的方法已广泛用于多个层级的检测。**（1）图像级分类。**早期工作关注判断图像是否包含缺陷，而不给出具体逐像素位置 [6—8]。SegNet [1] 及其变体 [9,10] 将像素级标注作为辅助信息引入网络，但最终仍输出二分类结果。**（2）模糊级缺陷定位。**指在没有逐像素监督的情况下获得相对细粒度输出。例如，类别激活图 [11] 能可视化 CNN 作分类决策时关注的图像区域；它被用于仅有图像级监督的模糊 LED 缺陷定位 [3] 和工业异常定位 [12]。基于无缺陷样本建模的方法 [13—15] 在训练阶段建立无缺陷数据分布模型，随后评估异常与正常样本的分布偏离。基于重建的异常检测方法 [16—18] 则依据相似性度量重建正常数据实例，再通过异常数据与重建数据的像素级差异定位并识别异常 [19]。

虽然这些方法不需要大量训练数据，但缺少精细监督导致像素级预测不准确。**（3）细粒度分割。**它已越来越多地用于缺陷检测 [2,20—23]。追求零缺陷制造 [24] 与获得充分缺陷样本之间存在矛盾。为缓解像素标签标注不足，研究引入了视觉显著性 [25]、重复模式分析 [26] 和交互点击 [22] 等额外先验。此外，还采用伪标签 [5,27]、一致性正则化 [28] 等半监督技术，进一步增强方法。

梳理现有文献可以发现，大多数缺陷检测方法仍关注图像级分类 [6—8] 和模糊级定位 [3,11,12]，主要原因是实际生产环境中难以获取逐像素标签。部分研究尝试利用半监督技术构建细粒度分割方法 [5,22,26,27]。然而，对于 LCD 屏、PCB 和印刷材料等工业产品，复杂背景、不规则缺陷形状以及充分带标签样本的缺乏，仍使缺陷分割很困难。外观先验是指关于缺陷视觉特征的既有知识或假设 [30]，通常依靠从训练数据学到的特定模式、形状、纹理或其他视觉线索来定位和分类缺陷。然而，数据稀缺与缺陷外观多样之间存在固有矛盾，因此上述基于外观先验定位缺陷的方法并不可靠（见图 1（d）（e））。有限缺陷样本可能使实际数据分布的表示出现偏斜，导致这些基于外观的方法泛化表现下降 [5]。需要强调的是，在 PCB、LCD、印刷出版物等工业产品中，根据视觉特征定位缺陷十分困难。复杂且有时含混的背景图案会遮蔽缺陷，进一步增加检测复杂性（见图 1（a）（b）（c））。

我们将缺陷检测转为变化检测问题的动机，基于两个不言自明的事实：（1）获得无缺陷样本远比获得缺陷图像容易；（2）缺陷区域本质上对应无缺陷与缺陷样本之间的差异。即使对人类观察者，没有干净参考也难以识别缺陷区域。因此，我们提出基于数据模拟与变化特征建模的精确缺陷分割方法。该方法尤其适合背景图案相对稳定但复杂的表面缺陷，如 PCB、LCD 和印刷出版物。

现有研究仍存在若干空白。第一，缺少足够带标签样本来训练能够高精度分割缺陷的全监督模型。第二，大多数分割方法基于外观，因此有限标签下的高精度分割仍有挑战。第三，缺陷检测主要依靠样本比较这一基本概念，尚未得到充分重视。如前所述，需要设计一种利用易获取无缺陷样本、依据变化检测机制进行缺陷分割的方法。此外，标签效率与计算效率也是模型设计的重要考虑。解决这些问题，对满足实际生产环境需求和弥补当前研究空白至关重要。为此，我们提出在缺陷与无缺陷工业图像之间进行深特征差分的缺陷分割方法，特点是并行建模两类图像、对比特征学习及变化感知解码。具体而言，我们提出具有变化注意力机制的新型变化感知孪生网络，以解决逐像素缺陷检测。编码阶段，使用受多类别平衡对比损失（BCL）约束的 Transformer 孪生网络，提取无缺陷与缺陷样本间的差异特征；随后通过多阶段相减融合层级孪生特征对，并上采样到高分辨率。解码阶段，特征距离图通过跳跃连接进入解码器，充当变化区域注意力，辅助逐像素缺陷定位。与直接建模缺陷外观不同，我们将缺陷建模为无缺陷与缺陷图像间的差异。有趣的是，如图 1（f）（g）所示，这一结构在标签有限时也表现出较强泛化。这可归因于图 1（h）所示的深特征差异学习。作为比较，图 1（i）显示直接进行图像相减所得到的含噪结果。

**图 1：基于变化的方法与基于外观的方法在全监督、半监督和无监督设置下的缺陷分割样例。**（d）列结果来自 SegFormer [29]，（e）列来自 UAPS [5]。预测图中绿色表示漏检，红色表示误检；“Err”量化二者之和。我们的方法优于半监督方法，而且仅使用 10% 训练样本，就能得到与全监督模型有竞争力的结果。图内列标注：（a）无缺陷图像；（b）缺陷图像；（c）真实标注；（d）全监督（100%）；（e）半监督（10%）；（f）本文半监督（10%）；（g）本文无监督（0%）；（h）深特征差异；（i）图像相减。上方分组为“基于外观的方法”和“本文基于变化的方法”。（d）—（g）的 Err 值依次为：LCD 样例 1457、1881、407、2660；PCB 样例 878、2185、1082、1675。

此外，表面缺陷检测社区需要更有挑战性的数据集。小数据集占主导地位，妨碍全面评估现有模型。例如，常用 KolektorSSD [10]、DAGM2007 [9] 和 Severstal-Steel [10] 数据集的平均精度分别已达 100%、100% 和 98.7%。鉴于 LCD 快速成为主要显示技术，并广泛应用于计算机及手机，我们引入新的数据集，以促进 LCD 缺陷检测。贡献总结如下：

- 提出用于缺陷分割的变化感知孪生网络。建模机制依赖干净与缺陷图像间的变化特征，而非缺陷外观，从而支持合成数据监督和未见类别泛化。
- 为模拟背景稳定但复杂的表面缺陷，并进一步推动 LCD 缺陷检测领域，提出名为 SynLCD 的合成 LCD 缺陷数据集，将其作为与其他分割方法比较的基准。
- SynLCD、PKU-Market-PCB [31] 和 MvTec-AD [16] 上的实验表明，该网络优于当前主流的基于外观的分割方法。此外，与五种先进半监督分割方法比较，突出显示模型在不同监督程度下的优势。

## 相关工作

本节介绍不同检测粒度下的表面缺陷检测，以及变化检测方法。与本研究最相关的是基于重建和差分的异常检测方法。它们对重建图像与输入图像进行差分，识别一般表面缺陷的大致位置。相比之下，我们使用深特征变化检测，而非图像空间中的简单差分。我们关注缺陷细微且可能在重建时被掩盖的情形下的精确分割。这一聚焦对保持我们对核心问题的主要关注至关重要。

### 表面缺陷检测

**图像标签级检测。**Masci 等人 [6] 将 CNN 应用于钢材表面缺陷检测，突出 CNN 优于人工特征。Faghih-Roohi 等人 [7] 探索网络复杂度对缺陷检测性能的影响。Racki 等人 [8] 在分类任务外结合辅助分割标签，提出检测合成纹理异常的紧凑 CNN。SegNet [1] 将原本分开的分割与分类阶段合为端到端训练框架，改进这一方法。Božič 等人 [10] 探索从弱监督到全监督的不同程度监督对缺陷分类准确率的影响。此外，[32] 中关于特征提取与复杂数据分布处理的一般原则，可能为高效、低成本收集缺陷分类训练标签提供启发。[33] 强调组合多模型提升准确率的能力，包括预处理、利用多个特征描述子提取特征，以及使用不同分类器分类。尽管有这些进展，早期深度学习研究仍主要关注图像级缺陷检测，对逐像素缺陷定位关注有限。

**模糊及区域级检测。**受异常检测任务逐像素标注的限制，一些研究转向弱监督 [3,12] 和无监督学习 [19]。类别激活图 [11] 广泛用于仅凭图像级提示指示潜在异常区域 [3,12]。但这只是减轻标注劳动，没有解决数据稀缺这一根本问题。另一方面，丰富的无缺陷数据大大推动了无缺陷建模及重建方法。无缺陷建模着重建立正常样本的嵌入模型，通过衡量与潜在空间的偏离识别异常实例。借助图像块表征（如 PatchCore [13]、ReconPatch [14]）、感受野上采样 [34] 和基于归一化流模型的梯度反向传播 [15]，可模糊定位缺陷。重建模型通常被训练以重建无缺陷样本，识别其无法生成的异常实例。自编码器 [16] 和生成对抗网络（GAN）[17] 常用于重建过程。对输入与重建样本直接差分即可得到缺陷区域，如 EfficientAD [18] 的逐元素平方距离。然而，正常图像重建不准确引起假阳性检测，是普遍问题。总之，这些方法没有逐像素标注，因此哪些图像点属于异常仍不清楚，导致检测结果不清晰。

**逐像素检测。**近来，从语义分割模型拓展的像素级缺陷检测越来越受关注。He 等人 [21] 采用 FCN [35] 定位木材缺陷。Huang 和 Xiang [26] 稍作修改，将 DeepLab v3+ [36] 用于织物缺陷分割。Du 等人 [37] 将 U-Net [38] 扩展为双流结构，分割 X 射线图像缺陷。更近的研究使用注意力机制建模局部及全局上下文依赖。Dong 等人 [23] 用全局上下文注意力分割钢材表面缺陷。Yeung 等人 [39] 以边界感知模块细化 SegFormer [40]，实现基于 Transformer 的缺陷分割。[41] 将自注意力与双编码器—解码器结合，用于噪声背景下生物医学分割。[42] 强调多尺度特征学习对提高分割精度的重要性。缺陷分割增强了对缺陷样本的理解，但受精细标签成本限制。

因此，一些近期研究采用伪标签 [5,27] 和一致性正则化 [28] 等半监督技术。伪标签方法 [43,44] 通过预训练网络为未标注样本生成伪标签，利用额外训练信号潜在提升模型表现。然而，未标注样本中的预测噪声可能损害伪标签质量，限制其实用性。一致性正则化认为，未标注样本的预测在受控扰动下应保持一致，旨在最小化不同场景下预测差异。研究提出协同训练 [45]、平均教师 [46]、多头预测不确定性 [5] 等启发式方案。[47] 探索的主动学习技术也具有类似理念：在改善模型表现的同时最小化标注成本。表 6 比较有限标签下这些半监督方法与我们的变化建模架构。

### 变化检测

图像变化检测旨在识别不同时刻采集图像之间预先定义的差异 [48]。主要挑战在于区分语义变化与噪声变化，包括照明变化、饱和度变化及无关背景干扰 [49]。它广泛用于手写签名验证 [50]、街景 [51] 和遥感变化检测 [48]。ChangChip [52] 通过人工图像配准与比较识别 PCB 表面缺陷，但预处理时间长，图像相减的超参数也需要细调。Zagoruyko 等人 [53] 开创性地将 CNN 用于图像比较。Daudt 等人 [54] 进一步发展基于 FCN 的孪生架构，实现任意尺寸图像变化检测。若干研究 [51,55] 集中引入对比损失 [56]，其关键在于最小化未变化特征对距离、最大化变化特征对距离。然而，这些对比方法主要面向二值变化，对不同变化类别的关注不平衡，如图 7 所示。

在本研究情境下，最相关的是背景重建方法 [57,58]。这些创新工作从无标签数据重建无瑕疵图像，并对原图与重建图像进行差分映射，得到最终分割图。然而，重建图像质量及图像级差分成为其瓶颈。

> 译注：正文中的“ChangChip”按原文保留；参考文献 [52] 题名拼为“Changechip”。[16] 是 MVTec AD 数据集论文，原文在重建方法、自编码器等位置也引用了该编号，译文保留引文，不替换为其他文献。

## 方法

### 问题定义：外观建模与变化建模

LCD、PCB 和印刷产品（如书籍、图纸、商标）等工业材料，在无缺陷时具有相对一致的外观及表面图案。基于这一观察，我们简化表面缺陷图像 $x_{\mathrm{ng}}$ 的形成过程（ng 表示“not good”，即不良品）：将标准干净图像 $x_{\mathrm{ok}}$ 与 $x_{\mathrm{defect}}$ 按特定方式叠加，再进行全局非线性变换。可表述为：

$$x_{\mathrm{ng}}=\sigma(x_{\mathrm{ok}}\boxplus x_{\mathrm{defect}}), \tag{1}$$

其中 $\sigma$ 表示非线性全局变换（如材料批次差异、老化、照明和成像畸变），$\boxplus$ 表示特定叠加方式（如腐蚀、破损、混合和直接覆盖）。经典分割范式的模型 $f'$ 根据外观及上下文识别缺陷物体，依据式（1）的假设可写为：

$$\widehat x_{\mathrm{defect}}=f'(x_{\mathrm{ng}})=f'\!\left(\sigma(x_{\mathrm{ok}}\boxplus x_{\mathrm{defect}})\right). \tag{2}$$

这意味着模型 $f'$ 需要在非线性干扰 $\sigma$ 下，从复杂背景 $x_{\mathrm{ok}}$ 分离 $\widehat x_{\mathrm{defect}}$。然而，背景内容可能与缺陷十分相似，如图 5（g），使基于外观的区分不可靠。我们旨在将缺陷图像中的缺陷建模为它与无缺陷图像的差异：

$$\begin{aligned}\widehat x_{\mathrm{defect}}&=f(x_{\mathrm{ng}},\widehat x_{\mathrm{ok}}),\\&=\sigma(x_{\mathrm{ok}}\boxplus x_{\mathrm{defect}})\boxminus\widehat x_{\mathrm{ok}}.\end{aligned} \tag{3}$$

变化建模范式让模型学习深度相减函数 $\boxminus$，克服与缺陷外观相关的限制。易获取的无缺陷图像 $\widehat x_{\mathrm{ok}}$ 缓解了非线性变换 $\sigma$ 和复杂背景的干扰。

### 变化感知孪生网络

图 2 展示变化感知孪生网络流程。对比特征编码器提取缺陷与无缺陷样本的深特征差异。变化感知解码器融合编码器变化信息，辅助缺陷定位。特征距离图（DistMap）用于编码器与解码器之间的变化信息交互。具体而言，编码器包含一个高效 Transformer 主干，使用共享权重的四个 Transformer 块 [29,59]。随后通过多阶段相减融合层级特征，并在解码前上采样到高分辨率。解码阶段，DistMap 引入变化信息，以定位逐像素缺陷。整个网络由两个损失函数监督：交叉熵损失评估预测与相应真实标注的相似性，平衡对比损失区分缺陷与无缺陷区域特征。

**图 2：基于 Transformer 的变化感知缺陷检测网络 CADNet 的流程。**输入为待检图像（NG）和无缺陷参考图像（OK），用于深度变化建模。对比特征编码器由四阶段孪生 Transformer 构成，生成深特征距离图 DistMap；变化感知解码器利用 DistMap 促进准确缺陷定位。网络使用交叉熵及多类别平衡对比损失训练。注意，真实标注为多类别分割图。图内标注：Input patches＝输入图像块；Norm＝归一化；SR Attention＝序列缩减注意力；MLP＝多层感知机；Subtraction＝相减；UP2x＝上采样两倍；Change-information interaction＝变化信息交互；Change-aware Decoder＝变化感知解码器；Pixel-wise prediction＝逐像素预测；Balanced ContraLoss＝平衡对比损失；CrossEntropy Loss＝交叉熵损失；GT＝真实标注；“Supervision for contrastive feature／Final prediction”分别为对比特征／最终预测监督；四层特征尺度为 $1/4,1/8,1/16,1/32$，基本块重复两次。

### 对比特征编码器

我们设计高效 Transformer 编码器，以隐式特征比较度量学习对比特征，确保满足工业生产快速检测需求。由于处理配对输入带来双倍计算成本，为提高效率，我们借鉴序列缩减注意力 [29,60]，如图 3（a）。标准自注意力 [59] 的主要瓶颈是长序列输入下的二次复杂度：

$$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V, \tag{4}$$

其中矩阵 $Q,K,V$ 均为 $N\times C$，且 $d_k=N$。采用比率 $R$ 缩短序列 $K$：

$$\widehat K=\operatorname{reshape}\!\left(\frac NR,C\cdot R\right)(K), \tag{5}$$

$$K=\operatorname{linear}(C\cdot R,C)(\widehat K), \tag{6}$$

首先将 $K$ 重塑为 $\frac NR\times C\cdot R$，再用线性层处理长度为 $C\cdot R$ 的序列，产生 C 维序列。新 $K$ 因此为 $\frac NR\times C$，将自注意力复杂度由 $O(N^2)$ 降至 $O(N^2/R)$。每个序列缩减注意力（SRA）模块包含具有残差连接的序列缩减注意力单元及多层感知机（MLP）。每个 Transformer 阶段采用两个 SRA 模块，四阶段缩减率分别为 $\{8,4,2,1\}$。

> 译注：原文将注意力缩放维度写作 $d_k=N$，与通常以键向量维度作为缩放维度的记法不同，此处依本地 PDF 保留。式（5）（6）的 reshape／linear 记法亦按原文重写为 LaTeX，不额外更改算法。

层级 Transformer 块共享权重并行编码缺陷与无缺陷图像，因为图像对仅在很小的缺陷区域不同。金字塔特征记为 $\{f_m^n\mid m=0,1;\ n=0,1,2,3\}$，其中 $m$ 指两个孪生分支，$n$ 指四个特征层。位置 $(i,j)$ 的特征距离为：

$$\begin{aligned}\operatorname{DistMap}(i,j)&=\left\|f^{\mathrm{ng}}(i,j)-f^{\mathrm{ok}}(i,j)\right\|_2,\\f^{\mathrm{ng}}&=\operatorname{concat}(f_0^1,f_0^2,f_0^3,f_0^4),\\f^{\mathrm{ok}}&=\operatorname{concat}(f_1^1,f_1^2,f_1^3,f_1^4).\end{aligned} \tag{7}$$

其中 $f^{\mathrm{ng}}$ 与 $f^{\mathrm{ok}}$ 分别表示缺陷和无缺陷图像层级特征的拼接（concat）。层级特征调整到与 $F_0^1$ 或 $F_1^1$ 相同尺寸。采用对比损失（CL）作为约束：

$$\mathrm{CL}=\begin{cases}\operatorname{DistMap}(i,j)-\tau_{\mathrm{ok}},&y(i,j)=0,\\\max(0,\tau_{\mathrm{ng}}-\operatorname{DistMap}(i,j)),&y(i,j)=1,\end{cases} \tag{8}$$

其中 $y(i,j)$ 是真实标注，值为 0 或 1 分别表示该点未变化或变化。$\tau_{\mathrm{ok}}$ 和 $\tau_{\mathrm{ng}}$ 为非负阈值。$y(i,j)=0$（未变化点）时，期望特征距离减小到接近 0 的 $\tau_{\mathrm{ok}}$；$y(i,j)=1$（变化点）时，鼓励特征距离增大至 $\tau_{\mathrm{ng}}$。依据 [61]，我们将 $\tau_{\mathrm{ng}}$ 和 $\tau_{\mathrm{ok}}$ 设为 2.2 和 0.3。因为配对未变化点（无缺陷）并非完全相同，$\tau_{\mathrm{ok}}$ 取略大于 0 的正值；$\tau_{\mathrm{ng}}$ 取较大正值，鼓励不同或有缺陷点之间存在明显间隔。

> 译注：式（7）之前层级索引为 0—3，而公式中为 1—4；正文尺寸记号又使用大写 $F$，均沿用原文。式（8）第一分支没有 $\max$ 截断、平方或绝对值，不能擅自补入；当距离小于 $\tau_{\mathrm{ok}}$ 时，该表达式可为负，且与“减小到阈值”的文字描述并不完全一致。

原始对比损失面向二值变化检测。但当不止一种缺陷被建模为变化区域（即 $y\in1,2,\ldots,c$）时，各类样本量失衡会导致对比监督失衡。因此，我们提出用多类别平衡因子扩展它。根据某变化类别在总变化区域（即 $y(i,j)=1$）中的比例，平衡因子定义为：

$$B_p=\frac1{f_p}=\frac1{n_p}\sum_q^C n_q. \tag{9}$$

其中 $C$ 是总类别数，$f_q$ 是类别 $q$ 样本点占总变化样本点的比例，$n_q$、$n_p$ 分别为类别 $q$、$p$ 的点数。平衡对比损失（BCL）定义为：

$$\mathrm{BCL}=\begin{cases}\mathrm{CL},&y(i,j)=0,\\\displaystyle\sum_{c^l=0}^{C}B_l\cdot\mathrm{CL}(y(i,j)=c^l),&y(i,j)\in1,2,\ldots,C.\end{cases} \tag{10}$$

它更加侧重较少见的变化类别，使损失在不同变化类型间分配平衡。

> 译注：式（9）原文求和仅标 $q$ 与上限 $C$；式（10）原文使用 $c^l$、$B_l$，求和从 $c^l=0$ 到 $C$，同时将第二分支条件写为正类 1—C。这里忠实保留，不能据此自动认定它已清楚定义是否包含背景或零频类别如何处理。

**图 3：基本模块。**（a）序列缩减注意力利用空间缩减层，将自注意力模块复杂度从 $O(N^2)$ 降至 $O(N^2/R)$。（b）变化感知解码器基于三维（水平、垂直及深度）注意力模块，在全监督和半监督目标检测时，以不同方式利用携带变化信息的 DistMap。图内标注：Spatial reduction＝空间缩减；Reshape＝重塑；Linear＝线性；MatMul＝矩阵乘法；scale＝缩放；softmax＝softmax；Input／Output＝输入／输出；Spatial Att.／Channel Att.／Change Att.＝空间／通道／变化注意力；AvgPool＝平均池化；Conv+BN+ReLU＝卷积＋批归一化＋ReLU；FC＝全连接层；FST?＝是否全监督训练；Y／N＝是／否；Sigmoid＝sigmoid；Point-wise multiply／addition＝逐点乘法／加法；$H\times W\times C$ 为输入尺寸，水平、垂直及通道池化图尺寸分别标为 $H\times1\times1$、$1\times W\times1$、$1\times1\times C$。

### 变化感知解码器

注意力机制广泛用于上下文建模。但缺陷位置任意分布、与周围关联较弱，严重影响空间上下文。为此，提出名为变化感知解码器（CAD）的新变化注意力机制，引入变化信息辅助缺陷目标定位。具体而言，对比特征编码器获得的特征距离通过跳跃连接送入解码器，在全监督与半监督目标检测中扮演不同角色。CAD 结构如图 3（b）。

首先，将轻量 coordAttention [62] 扩展为三维注意力模块，在维持低参数成本的同时实现较高特征解码精度。受平衡对比损失约束，DistMap 在变化区域激活值高，在不变区域低。当前语义分割方法在带标签数据充分时已能有效检测缺陷。因此，将特征距离加到编码特征，以辅助定位缺陷：

$$\operatorname{Output}=\operatorname{ChangeAtt}(\operatorname{Input}\oplus\operatorname{DistMap}), \tag{11}$$

其中 $\oplus$ 表示逐点相加，ChangeAtt 为通道注意力（CA）、水平注意力（HA）与垂直注意力（VA）的组合：

$$\operatorname{ChangeAtt}(\cdot)=\operatorname{CA}(\cdot)\otimes\operatorname{HA}(\cdot)\otimes\operatorname{VA}(\cdot), \tag{12}$$

其中 $\otimes$ 表示逐点相乘。然而，标签有限时遇到外观未知的未见缺陷，依赖缺陷外观将不可靠。实际上，训练集上的缺陷模式过拟合可能导致测试集泛化更差。这时，变化信息成为缺陷定位的主要指示。因此，DistMap 作为空间线索，经归一化（Norm）后与编码特征相乘，以辅助这一过程：

$$\operatorname{ChangeAtt}(\cdot)=\operatorname{CA}(\cdot)\otimes\operatorname{Norm}(\operatorname{DistMap})\otimes(\cdot), \tag{13}$$

$$\operatorname{Output}=\operatorname{ChangeAtt}(\operatorname{Input}). \tag{14}$$

这里，乘法引入鲁棒先验，专门面向变化区域。DistMap 作为空间上下文先验，替代常规水平或垂直注意力，旨在引导模型识别变化区域中的潜在缺陷。特别地，图 7 表明 DistMap 对最终结果提供粗略表示，所谓缺陷区域与实际变化区域准确对齐。

### 损失函数

网络使用 BCL 和交叉熵损失训练。BCL 引导模型学习对比特征，如“对比特征编码器”小节所述。单点 $(i,j)$ 的交叉熵损失为：

$$\mathrm{CEL}=-\log\frac{e^{\widehat y(i,j,c^y)}}{\displaystyle\sum_{c^k=0}^{C-1}e^{\widehat y(i,j,c^k)}}, \tag{15}$$

其中 $c^y$ 为样本点真实类别，$C$ 为总类别数，$\widehat y(i,j,c^k)$ 表示类别 $c^k$ 的预测概率。训练总损失如下：

$$\mathrm{loss}=\lambda_1\mathrm{CEL}+\lambda_2\mathrm{BCL}. \tag{16}$$

实验中 $\lambda_1$ 和 $\lambda_2$ 均设为 1。

> 译注：原文将式（15）指数中的 $\widehat y$ 称为“预测概率”，但该写法通常将指数前数值解释为 logits；译文保留原式和原文字定义，不擅自将其改写成另一损失。

## 实验与结果

### 数据集

评估涉及三个数据集，包括具有复杂背景和微小纹理异常的自建合成 LCD 数据集与 PKU-Market-PCB [31]。此外，采用异常检测基准 MVTec-AD [16]，在全监督学习条件下验证方法的泛化能力。

**合成 LCD 缺陷数据集。**为验证模型在不同成像、生产条件和缺陷外观下的分割能力，我们构建了名为 SynLCD 的合成 LCD 缺陷数据集。在实际 LCD 检测中，专门设计显示图案以显露点、线、Mura 等不同缺陷 [63]。图案由纯色块、色图、文本块、灰度渐变和人脸构成。图 4 展示十种无缺陷显示图案；检测界面分为五个功能区，以全面评价显示性能：

- 右下角（人脸图像）：人类视觉系统对人脸特征高度敏感。该区域测试显示器准确呈现肤色、头发纹理、人脸轮廓等复杂生物细节的能力。色彩平衡、饱和度或纹理清晰度偏差可能意味着面板驱动问题或颜色校准缺陷。
- 左下角（灰度渐变）：采用由纯黑到白的渐变，评价亮度均匀性、对比度响应和背光一致性。缺陷屏幕可能出现条带（不自然的颜色阶跃）、亮度不均或背光泄漏（例如 Mura 缺陷），损害视觉平滑性。
- 右上角（文本图案）：用于评价高频细节呈现。低质量面板经常出现文字边缘锯齿、像素块缺失或背景噪声，这些问题与子像素排列异常、驱动电路不稳定或 TFT（薄膜晶体管）阵列缺陷有关。
- 人脸图像上方（RGB 色度图）：通过标准色轮或渐变测量色域覆盖范围和准确性。偏色、混叠、去饱和等异常提示彩色滤光片或信号处理算法存在问题。
- 左上角（纯色块）：包含红、绿、蓝、黑等基本颜色，用于测试颜色纯度和背光均匀性。局部亮度波动、色斑或暗点（例如线缺陷）等异常揭示制造瑕疵。

**图 4：无缺陷 LCD 图案。**在实际检测中，工业 LCD 显示图案由 RGB 色块、灰度渐变、色图、字符和人脸组成，用于显露点、线及 Mura 等不同类型缺陷 [63]。

SynLCD 包含位置与分布随机的三类缺陷样本：线缺陷、异常点（abpt）及混合缺陷，如图 5。部分缺陷与背景图案极为相似；线缺陷与背景对比度较低，且横贯整个屏幕。

**图 5：SynLCD 样本与数据集挑战。**（a）异常点缺陷样本；（b）线缺陷样本；（c）混合缺陷样本；（d）混合缺陷图像的二值标签；（e）RGB 偏差和不规则屏幕纹理；（f）非线性饱和度差异；（g）低对比度异常点与线缺陷。

**表 1：SynLCD 数据集统计详情。**

| 属性 | 类型 | 数值 | 说明 |
|---|---|---|---|
| 数量 | 背景图案 | 10 | 人脸、色图等变化 |
| 数量 | 缺陷类型 | 3 | 线、异常点、混合缺陷 |
| 数量 | 缺陷样本 | 10×300×3 | 每种类型、每种图案各 300 个样本 |
| 数量 | 无缺陷样本 | 10×900 | 亮度、对比度等变化 |
| 缺陷 | 形状 | 2 | 线和异常点 |
| 缺陷 | 颜色 | 5 | 黑、白、红、绿、蓝 |
| 缺陷 | 不透明度 | 10%—100% | 间隔 10% |
| 缺陷 | 宽度 | 3—33 像素 | 间隔 3%（原表如此） |
| 屏幕 | 亮度 | 偏置：1—6 | 间隔 1 |
| 屏幕 | 对比度 | alpha：0.5—1.5 | 间隔 0.1 |
| 屏幕 | ISO 噪声 | 10%—100% | 间隔 10% |
| 屏幕 | RGB 偏差 | 3—33 灰度级 | 间隔 3 |

表 1 给出统计详情。根据式（1）的假设，缺陷图像 $x_{ng}$ 由干净表面图像 $x_{ok}$ 经非线性整体表面变化 $\sigma$ 后叠加缺陷 $x_{defect}$ 形成。为生成线缺陷，首先将干净图像划分为 $K$ 个区域，然后在每个区域预绘颜色、透明度和宽度随机的线。这些线贯穿屏幕，模拟实际线缺陷。异常点倾向于出现在边缘、头发和文字等高频过渡区域。为生成异常点样本，我们把灰度阈值从 50 改变至 200，得到各阈值下的分割结果，并从中提取边缘点集合。随后用 K-means 对这些点随机聚类，为每个子类分配随机颜色、尺度和透明度。获得两类缺陷后，利用高斯模糊与泊松无缝融合 [64] 将其叠加到干净图像上；引入随机亮度、对比度、ISO 噪声和 RGB 色偏以增加多样性。为避免分类任务中的样本不平衡干扰，对图 4 每幅干净图像生成 300 个缺陷和无缺陷样本。总计有 4,200 个训练样本（七种标准背景图案，每种 600 个样本）和 1,800 个测试样本。

> 译注：表 1 的缺陷数量为 10×300×3、无缺陷数量为 10×900，合计 18,000；本段训练与测试数量合计为 6,000。原文两处统计不一致，均保留。

**PKU-Market-PCB。**该数据集 [31] 包含 1,386 幅图像及六种缺陷，用于验证模型在复杂背景和微小缺陷场景下的泛化。原始图像尺寸不一致。为简化训练，我们将原图缩放、裁剪为 1000×1000 子图，只保留包含缺陷的子图。最终 1,566 幅（70%）用于训练，676 幅（30%）用于测试。预处理数据随源码发布于 <https://github.com/HATFormer/CADNet>。

**MVTec-AD。**为进一步验证一般缺陷检测，我们采用广泛使用的异常检测基准 MVTec-AD [16] 比较。为更有效训练并精确分割缺陷，将原始数据重组为全监督训练数据。原始 5,354 幅图像及对应真值标注随机打乱，分为 3,747 幅（70%）训练图像及 1,607 幅（30%）测试图像。

### 实验设置与指标

**实现细节。**模型基于 MMSegmentation，在 RTX3090 GPU 上训练。输入缩放为 512×512；训练使用随机裁剪、翻转和颜色归一化等常见增强。全监督比较中，各数据集分别以 70% 和 30% 用于训练与测试。对比方法输入原数据集中的单幅样本；我们的方法同时输入该样本及对应干净图像。SynLCD 有图 4 的十幅固定干净图像；PKU-Market-PCB [31] 同样提供十幅干净图像；MVTec-AD [16] 每类选择一幅无缺陷图像，在训练与测试中共用。所有模型训练 30 轮，根据各模型显存占用，混合批量大小在 4—8 之间。

半监督比较中，带标签样本比例为 0%、5%、10%、15%（0% 实际是无监督，为简便纳入半监督比较）。为与使用无标签数据训练的 UAPS [5] 比较，遵循 [5] 的设置，加入 10% 无标签数据。

**指标。**采用语义分割指标评价逐像素预测，包括平均交并比（mIoU）、准确率（Acc）和 Fscore，亦见 [36,65]。TP、FP、FN 分别表示真阳性、假阳性和假阴性：

- 精确率 $P=TP/(TP+FP)$，召回率 $R=TP/(TP+FN)$；
- $Fscore=2PR/(P+R)$；
- $Acc=(TP+TN)/(TP+FN+FP+TN)$；
- $mIoU=\frac1C\sum_{i=0}^{C-1}\frac{TP_i}{TP_i+FP_i+FN_i}$。

通过参数量（Params）与十亿浮点运算次数（GFLOPs）衡量复杂度。所有表格中 ↑ 表示越高越好，↓ 表示越低越好。

**对比方法。**从两个方面评价：（1）全监督分割，在有充足像素标签时证明变化建模相较外观建模的优势，比较表 2 中六种语义分割方法。（2）半监督分割，评价标签不足时的鲁棒性，因为实际生产缺陷难以具备大量外观一致的样本，比较表 2 中五种先进半监督方法。

**表 2：用于比较的全监督与半监督分割方法概览。**

| 类型 | 方法 | 描述 |
|---|---|---|
| 全监督 | FCN [66] | 利用全卷积层对任意尺寸图像作稠密预测。 |
| 全监督 | PSPNet [67] | 利用金字塔池化聚合全局上下文，进行复杂场景解析。 |
| 全监督 | DeepLabV3+ [36] | 引入空洞空间卷积金字塔（ASPP），增强多尺度上下文。 |
| 全监督 | DANet [68] | 通过自注意力自适应整合空间和通道维度的语义依赖。 |
| 全监督 | OCRNet [69] | 引入对象上下文表示，利用像素与对象关系增强像素表示。 |
| 全监督 | SegFormer [40] | 将 Transformer 与轻量 MLP 解码器结合，简化语义分割框架。 |
| 半监督 | DCT [45] | 使用一个网络，确保同一样本不同视图的一致性。 |
| 半监督 | CPS [43] | 强制两个不同初始化的分割网络保持一致。 |
| 半监督 | UAMT [46] | 鼓励不同扰动下的预测一致性，估计不确定性以学习无标签数据。 |
| 半监督 | UCC [44] | 共享编码器、采用双解码器，通过数据增强约束解码器一致性。 |
| 半监督 | UAPS [5] | 单次前向传播中动态混合多头伪标签，进行不确定性正则化。 |

### 定量比较

#### 全监督分割

表 3 显示，我们的方法相较其他分割模型有显著改进。相较次优结果，四项指标（线 IoU、异常点 IoU、mIoU、mFscore）分别提高 12.65%、0.82%、8.17%、4.15%。表 4、5 中，模型在 PCB 和 MVTec-AD 的大多数指标上取得最佳结果。

在效率方面，模型参数量与 SegFormer 相当，两者计算开销均显著优于其他模型。相较 SegFormer，仅增加 1.84 GFLOPs，mIoU、mAcc、mFscore 分别提高 9.79%、6.42%、5.15%。这既体现适合资源受限工业设备的效率，也体现变化建模机制的优势。

> 译注：以上差值按原文保留；表 3 中本方法与 SegFormer 的直接差值为 mIoU 7.47、mAcc 5.37、mFscore 4.45 个百分点，与正文不一致。

**表 3：SynLCD 上与主流语义分割方法的比较。**原表用粗体、粗斜体、斜体标出各指标前三名。

| 方法 | 线 IoU↑ | 异常点 IoU↑ | mIoU↑ | mAcc↑ | mFscore↑ | 参数量（百万）↓ | GFLOPs↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCN [66] | 51.86 | 11.48 | 31.67 | 36.06 | 44.45 | 49.5 | 57.91 |
| PSPNet [67] | 79.00 | 52.54 | 65.77 | 71.56 | 78.58 | 12.76 | 54.27 |
| DeepLabV3+ [36] | 81.96 | 72.93 | 77.45 | 90.24 | 87.22 | 43.58 | 176.22 |
| DANet [68] | 79.92 | 57.04 | 68.48 | 76.27 | 80.74 | 49.82 | 199.05 |
| OCRNet [69] | 83.46 | 62.19 | 72.83 | 86.08 | 83.84 | 12.07 | 52.83 |
| SegFormer [40] | 82.99 | 69.62 | 76.31 | 83.68 | 86.39 | 3.72 | 6.37 |
| 本文 CADNet | 94.02 | 73.53 | 83.78 | 89.05 | 90.84 | 3.90 | 8.21 |

**表 4：PCB 数据集上的比较。**原表用三种字型标出各指标前三名。

| 方法 | 类别1 IoU↑ | 类别2 IoU↑ | 类别3 IoU↑ | 类别4 IoU↑ | 类别5 IoU↑ | 类别6 IoU↑ | mIoU↑ | mAcc↑ | mFscore↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FCN [66] | 50.13 | 69.19 | 68.65 | 45.45 | 50.35 | 36.36 | 53.35 | 60.80 | 68.79 |
| PSPNet [67] | 74.04 | 72.59 | 72.61 | 71.29 | 66.39 | 72.46 | 71.56 | 81.77 | 83.40 |
| DeepLabV3+ [36] | 75.39 | 73.56 | 74.22 | 73.57 | 69.94 | 76.47 | 73.85 | 82.10 | 84.94 |
| DANet [68] | 74.31 | 73.02 | 71.21 | 72.14 | 68.86 | 75.02 | 72.42 | 82.01 | 83.99 |
| OCRNet [69] | 76.08 | 73.00 | 73.78 | 75.98 | 71.13 | 78.13 | 74.68 | 83.45 | 85.48 |
| SegFormer [40] | 75.79 | 71.39 | 72.31 | 72.29 | 70.75 | 78.04 | 73.42 | 82.29 | 84.65 |
| 本文 CADNet | 77.21 | 73.98 | 75.08 | 79.95 | 76.47 | 82.44 | 77.52 | 85.87 | 87.31 |

**表 5：MVTec-AD 上的比较。**原表用三种字型标出前三名。该数据集有 15 类，此处报告其中六类。

| 方法 | 类别1 IoU↑ | 类别2 IoU↑ | 类别3 IoU↑ | 类别4 IoU↑ | 类别5 IoU↑ | 类别6 IoU↑ | mIoU↑ | mAcc↑ | mFscore↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FCN [66] | 76.10 | 60.14 | 35.93 | 69.73 | 13.51 | 79.65 | 58.14 | 64.84 | 70.00 |
| PSPNet [67] | 72.00 | 68.24 | 43.86 | 74.89 | 42.43 | 83.44 | 65.42 | 76.25 | 77.58 |
| DeepLabV3+ [36] | 76.65 | 63.48 | 41.18 | 72.31 | 34.93 | 81.12 | 63.77 | 77.59 | 76.19 |
| DANet [68] | 75.13 | 56.37 | 37.95 | 72.42 | 27.10 | 80.92 | 61.63 | 72.49 | 73.94 |
| OCRNet [69] | 70.89 | 65.18 | 45.67 | 65.47 | 35.41 | 81.51 | 59.89 | 68.98 | 72.31 |
| SegFormer [40] | 81.63 | 64.63 | 53.81 | 70.81 | 44.14 | 84.71 | 65.97 | 71.21 | 77.51 |
| 本文 CADNet | 82.60 | 74.16 | 61.19 | 73.06 | 52.69 | 86.41 | 71.35 | 80.85 | 82.24 |

#### 半监督分割

当有充足标签、缺陷外观定义明确时，SegFormer 等一般分割模型表现令人满意。但现实应用中缺陷样本有限、外观多样，基于外观建模不能保证稳健泛化。因此，我们深入探索标签有限、甚至完全无标签的缺陷分割。

LL、AA、LA、AL 实验的首字母表示在线（L）或异常点（A）上训练，第二个字母表示测试类型。图 6 显示，多数分割模型取得可接受的全监督结果，却因基于外观建模而无法检测未见缺陷（IoU、Acc、Fscore 等低于 0.5%）。变化感知模型在训练未见的缺陷外观上仍取得可观结果。LA（线缺陷训练、异常点测试）的准确率明显下降，可能是异常点尺寸更小、更难与背景区分。

**图 6：交叉测试性能比较。**推理样本不出现在训练阶段。LL、AA、LA、AL 首字母为训练类型，第二个字母为测试类型，L 表示线缺陷，A 表示异常点。图内 IoU、Acc、Fscore 分别为交并比、准确率、F 分数。

表 6 显示，模型在不同监督设置下优于五种先进半监督分割方法，表明其有潜力仅用少量真实生产样本灵活应用，显著降低数据采集与标注成本。

**表 6：SynLCD 上不同标签比例（0%—15%）下与先进半监督方法的比较。**所有模型先在异常点上预训练，再用线缺陷微调和测试；原表粗体为最优结果。

| 方法 | mIoU 0%↑ | mIoU 5%↑ | mIoU 10%↑ | mIoU 15%↑ | Fscore 0%↑ | Fscore 5%↑ | Fscore 10%↑ | Fscore 15%↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DCT [45] | 0.05 | 56.96 | 73.67 | 71.85 | 0.10 | 71.27 | 84.57 | 82.75 |
| UAMT [46] | 0.44 | 61.68 | 68.73 | 71.96 | 0.88 | 75.48 | 80.94 | 83.15 |
| CPS [43] | 1.09 | 65.07 | 65.63 | 76.02 | 2.15 | 78.29 | 78.70 | 85.68 |
| UCC [44] | 0.015 | 61.40 | 70.48 | 71.55 | 0.03 | 75.41 | 82.27 | 82.78 |
| UAPS [5] | 0.44 | 58.86 | 74.43 | 81.34 | 0.88 | 72.52 | 84.35 | 89.22 |
| 本文 CADNet | 46.89 | 82.93 | 84.52 | 84.71 | 63.84 | 90.87 | 91.64 | 91.72 |

### 消融研究

研究对比损失 CL、平衡对比损失 BCL、变化感知解码器 CAD 的影响。根据表 7 与图 7，可得如下结论。

- 以 CL 监督中间层，无额外计算开销即可显著改善大部分准确率指标。图 7 的无 CL 距离图与有 CL 距离图比较说明，对比约束能减少背景噪声，找到更有判别性的变化（缺陷）区域。无 CL 距离图中线比异常点更容易区分，说明对比约束不平衡。
- 图 7 中 CL 和 BCL 距离图说明，BCL 能有效增强异常点强度，在保持线 IoU 稳定时进一步提高异常点 IoU，因而提高整体 mIoU 和 mFscore。
- CAD 仅增加极少计算成本即可提升全部准确率指标。无 CAD 与有 CAD 的概率图比较表明，变化信息和空间上下文对于修复断线、减轻噪声检测具有重要作用。

**表 7：损失函数与解码器消融。**CEL 为交叉熵，CL 为对比损失，BCL 为平衡对比损失，CAD 为变化感知解码器。

| CEL | CL | BCL | CAD | 线 IoU↑ | 异常点 IoU↑ | mIoU↑ | mAcc↑ | mFscore↑ | 参数量↓ | GFLOPs↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ✓ | — | — | — | 84.21 | 73.00 | 78.61 | 85.09 | 87.91 | 3.72 | 8.16 |
| ✓ | ✓ | — | — | 89.40 | 70.17 | 79.78 | 85.22 | 88.43 | 3.72 | 8.16 |
| ✓ | — | ✓ | — | 89.56 | 72.96 | 81.26 | 87.32 | 89.43 | 3.72 | 8.16 |
| ✓ | — | ✓ | ✓ | 94.02 | 73.53 | 83.78 | 89.05 | 90.84 | 3.90 | 8.21 |

**图 7：消融可视化结果。**展示有无 CAD、CL、BCL 时的最终预测（pred）、输出前概率图（prob）与距离图 DistMap。图内 noCL／CL／BCL 表示无对比损失／对比损失／平衡对比损失；noCAM／CAM 为原图对有无注意力模块的命名。

### 定性结果

图 8 左侧两图的精确率—召回率曲线表明，变化感知网络在线和异常点检测上始终优于其他模型，尤其是在高召回率处。右侧两图的 Fscore—阈值曲线表明，模型在不同二值化阈值下始终具有较高 Fscore。此外，较大线缺陷相较异常点通常具有更高精确率和 Fscore。

**图 8：精确率—召回率（PR）和 Fscore—阈值（FT）曲线比较。**从左至右为线缺陷 PR、异常点 PR、线缺陷 FT、异常点 FT。Precision／Recall／Threshold 为精确率／召回率／阈值。

图 9、10 进一步展示 SynLCD 和 PCB 的可视化比较。为直观观察，全部多类别缺陷统一设为白色；绿色表示漏检，红色表示误检，黄色误差汇总漏检和误检。细线比粗线更容易漏检，因为特征提取下采样可能丢失信息。总体而言，其他模型效果弱于 CADNet，这与准确率指标一致；其在所有测试图像中存在大量漏检和误检。

**图 9：SynLCD 上的可视化比较。**白色为线与异常点缺陷，绿色为漏检，红色为误检；黄色误差（Err）汇总漏检与误检。图内 NG／GT／Err 为待检图像／真实标注／误差。

**图 10：PCB 上的可视化比较。**白色为缺陷，绿色为漏检，红色为误检；黄色误差（Err）汇总漏检与误检。

图 11 展示本方法与 SegFormer 在一般工业产品场景的结果。第 3、4 行的高层语义缺陷具有正常纹理，因此常规分割方法无法处理。图 12 展示本模型在全监督及无监督下的预测。有趣的是，虽然未见缺陷的 mIoU 下降，视觉影响并不明显；CADNet 的 mIoU 仍相当令人满意。以 SegFormer 在 COCO [70] 上为参照，其实时 B0 版本 mIoU 为 35.6%，非实时 B5 版本为 46.7%，解释了本方法为何能对未见对象得到可接受视觉结果。

**图 11：MVTec-AD 上的可视化比较。**结果体现加入对比约束后的优势。前两行及后两行为低对比度、复杂背景场景，第 3、4 行为高层语义异常。

**图 12：全监督及半监督训练设置下的模型预测可视化。**图内 LL、AA、LA、AL 的训练／测试含义同图 6；GT 为真实标注，Pred 为预测。

## 结论

计算机视觉进步推动工业缺陷检测显著发展，但细粒度分割仍面临两个主要障碍：（1）缺陷数据采集与标注有限，使多数数据驱动语义分割模型无效；（2）模型开发与应用阶段的外观不一致，成为许多外观建模算法的瓶颈，尤其是 LCD、PCB 和印刷产品等复杂背景缺陷。为此，我们基于“缺陷区域本质是缺陷图像与无缺陷图像的差异”这一假设，开发基于变化建模的框架，定位像素级多类别缺陷。

在自建 SynLCD 与两个公开数据集上进行了深入实验。模型在合理计算成本下优于六种领先分割模型，半监督分割也优于五种先进方法。尤其在其他模型结果崩溃时，CADNet 仍达到 46.89% mIoU 和 63.84% F1。消融验证了各组件有效性。

这一突破表明，变化建模机制比外观建模语义分割更有效。变化感知使模型在外观高度变化的现实场景中具有相当应用潜力。得益于高效变化建模架构，计算成本与标签需求均相对较低，使仅凭少量样本开发精简基础工业检测模型成为可能。

局限与后续研究如下。（1）模型对较大几何错位敏感；视角变化或传感器数据错位等非理想检测条件会影响实际性能，需要进一步研究异构数据有效对齐。（2）虽然已用一定数据增强，更先进方法可进一步提高鲁棒性，例如使用扩散模型合成扩充缺陷数据，以改善未见缺陷泛化。当前尚未探索，是重要后续方向。（3）虽然准确率较好、计算成本相对较低，在真实工业环境尤其大规模数据中仍可能有较高计算负担，未来可通过剪枝、硬件加速等优化效率。

## 数据可用性

本研究全部数据集见 <https://github.com/HATFormer/CADNet>。

收稿：2024 年 12 月 10 日；录用：2025 年 3 月 17 日。

## 参考文献

[1] Tabernik, D., Šela, S., Skvarč, J. & Skočaj, D. 《基于分割的表面缺陷检测深度学习方法》 Journal of Intelligent Manufacturing 31, 759–776 (2020).

[2] Huang, Z., Wu, J. & Xie, F. 《采用深度可分离 U 形网络的热轧钢带表面缺陷自动分割》 Materials Letters 301, 130271 (2021).

[3] Lin, H., Li, B., Wang, X., Shu, Y. & Niu, S. 《采用深度卷积神经网络的 LED 芯片自动缺陷检测》 Journal of Intelligent Manufacturing 30, 2525–2534 (2019).

[4] Liu, T. & Ye, W. 《磁瓦表面缺陷分类的半监督学习方法》 Machine Vision and Applications 33, 35 (2022).

[5] Sime, D. M., Wang, G., Zeng, Z. & Peng, B. 《面向半监督缺陷分割的不确定性感知动态混合伪标签》 Computers in Industry 152, 103995 (2023).

[6] Masci, J., Meier, U., Ciresan, D., Schmidhuber, J. & Fricout, G. 《采用最大池化卷积神经网络的钢材缺陷分类》 载于 The 2012 international joint conference on neural networks (IJCNN), 1–6 (IEEE, 2012).

[7] Faghih-Roohi, S., Hajizadeh, S., Núñez, A., Babuska, R. & De Schutter, B. 《用于铁轨表面缺陷检测的深度卷积神经网络》 载于 2016 International joint conference on neural networks (IJCNN), 2584–2589 (IEEE, 2016).

[8] Racki, D., Tomazevic, D. & Skocaj, D. 《用于纹理表面异常检测的紧凑卷积神经网络》 载于 2018 IEEE winter conference on applications of computer vision (WACV), 1331–1339 (IEEE, 2018).

[9] Božič, J., Tabernik, D. & Skočaj, D. 《缺陷检测两阶段神经网络的端到端训练》 载于 2020 25th International Conference on Pattern Recognition (ICPR), 5619–5626 (IEEE, 2021).

[10] Božič, J., Tabernik, D. & Skočaj, D. 《表面缺陷检测的混合监督：从弱监督到全监督学习》 Computers in Industry 129, 103459 (2021).

[11] Zhou, B., Khosla, A., Lapedriza, A., Oliva, A. & Torralba, A. 《学习用于判别定位的深层特征》 载于 Proceedings of the IEEE conference on computer vision and pattern recognition, 2921–2929 (2016).

[12] Lin, D. et al. 《采用三元组特征正则化、CAM 引导多路径解码 U-Net 的缺陷检测与分割》 Knowledge-Based Systems 228, 107272 (2021).

[13] Roth, K. et al. 《迈向工业异常检测的全面召回》 载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14318–14328 (2022).

[14] Hyun, J. et al. 《ReconPatch：用于工业异常检测的对比图块表示学习》 载于 Proceedings of the IEEE/ CVF Winter Conference on Applications of Computer Vision, 2052–2061 (2024).

[15] Rudolph, M., Wandt, B. & Rosenhahn, B. 《相同却不同：采用归一化流的半监督缺陷检测》 载于 Proceedings of the IEEE/CVF winter conference on applications of computer vision, 1907–1916 (2021).

[16] Bergmann, P., Fauser, M., Sattlegger, D. & Steger, C. 《MVTec AD：用于无监督异常检测的全面真实世界数据集》 载于 Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9592–9600 (2019).

[17] Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U. & Langs, G. 《利用生成对抗网络引导标志物发现的无监督异常检测》 载于 International conference on information processing in medical imaging, 146–157 (Springer, 2017).

[18] Batzner, K., Heckler, L. & König, R. 《EfficientAD：毫秒级延迟的准确视觉异常检测》 载于 Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 128–138 (2024).

[19] Ruff, L. et al. 《深层与浅层异常检测的统一综述》 Proceedings of the IEEE 109, 756–795 (2021).

[20] Gao, X., Jian, M., Hu, M., Tanniru, M. & Li, S. 《结合 FCN 与 Faster R-CNN 的盾构隧道快速多缺陷检测系统》 Advances in Structural Engineering 22, 2907–2921 (2019).

[21] He, T. et al. 《用于木材缺陷定位与识别的全卷积神经网络》 IEEE Access 7, 123453–123462 (2019).

[22] Du, W., Shen, H. & Fu, J. 《基于深度学习的 X 射线图像自动缺陷分割》 IEEE Transactions on Industrial Electronics 12912–12920 (2021).

[23] Dong, H. et al. 《PGA-Net：用于自动表面缺陷检测的金字塔特征融合与全局上下文注意力网络》 IEEE Transactions on Industrial Informatics 16, 7448–7458 (2019).

[24] Caiazzo, B. et al. 《迈向零缺陷制造范式：先进方法与开放挑战综述》 Computers in Industry 134, 103548 (2022).

[25] Luo, X. et al. 《MAMINet：用于表面缺陷检测的记忆注意多推理网络》 Computers in Industry 145, 103834 (2023).

[26] Huang, Y. & Xiang, Z. 《RPDNet：基于卷积神经网络与重复模式分析的自动织物缺陷检测》 Sensors 22, 6226 (2022).

[27] Xu, R., Hao, R. & Huang, B. 《采用自监督学习策略与分割网络的高效表面缺陷检测》 Advanced Engineering Informatics 52, 101566 (2022).

[28] Sime, D. M., Wang, G., Zeng, Z., Wang, W. & Peng, B. 《采用成对相似图一致性与集成交叉伪标签的半监督缺陷分割》 IEEE Transactions on Industrial Informatics (2022).

[29] Xie, E. et al. 《SegFormer：基于 Transformer 的简单高效语义分割设计》 载于 Advances in Neural Information Processing Systems 34, 12077–12090 (2021).

[30] Saad, A., Hamarneh, G. & Möller, T. 《利用形状与外观先验探索和可视化分割不确定性》 IEEE Transactions on Visualization and Computer Graphics 16, 1366–1375 (2010).

[31] Ding, R., Dai, L., Li, G. & Liu, H. 《TDD-Net：用于印刷电路板的微小缺陷检测网络》 CAAI Transactions on Intelligence Technology 4, 110–116 (2019).

[32] Ul Amin, S., Kim, B., Jung, Y., Seo, S. & Park, S. 《利用三维卷积和长短期记忆模块高效融合时空特征的视频异常检测》 Advanced Intelligent Systems 2300706 (2024).

[33] Amin, S. U., Taj, S., Hussain, A. & Seo, S. 《采用集成学习方法自动分析 COVID-19、结核病与肺炎胸部 X 射线图像》 Biomedical Signal Processing and Control 87, 105408 (2024).

[34] Deng, H. & Li, X. 《通过单类嵌入反向蒸馏进行异常检测》 载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9737–9746 (2022).

[35] Long, J., Shelhamer, E. & Darrell, T. 《用于语义分割的全卷积网络》 载于 Proceedings of the IEEE conference on computer vision and pattern recognition, 3431–3440 (2015).

[36] Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F. & Adam, H. 《采用空洞可分离卷积的语义图像分割编码器—解码器》 载于 Ferrari, V., Hebert, M., Sminchisescu, C. & Weiss, Y. (eds.) Computer Vision – ECCV 2018, 833–851 (Springer International Publishing, Cham, 2018).

[37] Du, W., Shen, H. & Fu, J. 《基于深度学习的 X 射线图像自动缺陷分割》 IEEE Transactions on Industrial Electronics 68, 12912–12920 (2020).

[38] Ronneberger, O., Fischer, P. & Brox, T. 《U-Net：用于生物医学图像分割的卷积网络》 载于 International Conference on Medical image computing and computer-assisted intervention, 234–241 (Springer, 2015).

[39] Yeung, C.-C. & Lam, K.-M. 《采用 Transformer 的缺陷语义分割注意边界感知融合》 IEEE Transactions on Instrumentation and Measurement (2023).

[40] Xie, E. et al. 《SegFormer：基于 Transformer 的简单高效语义分割设计》 Advances in Neural Information Processing Systems 34, 12077–12090 (2021).

[41] Agarwal, R. et al. 《用于生物医学 CT 图像分割的双编码器—解码器深层准循环自注意力》 IEEE Journal of Biomedical and Health Informatics (2024).

[42] Agarwal, R., Ghosal, P., Sadhu, A. K., Murmu, N. & Nandi, D. 《用于生物医学图像分割的多尺度双通道特征嵌入解码器》 Computer Methods and Programs in Biomedicine 257, 108464 (2024).

[43] Chen, X., Yuan, Y., Zeng, G. & Wang, J. 《采用交叉伪监督的半监督语义分割》 载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2613–2622 (2021).

[44] Fan, J., Gao, B., Jin, H. & Jiang, L. 《UCC：用于半监督语义分割的不确定性引导跨头协同训练》 载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9947–9956 (2022).

[45] Qiao, S., Shen, W., Zhang, Z., Wang, B. & Yuille, A. 《用于半监督图像识别的深度协同训练》 载于 Proceedings of the european conference on computer vision (eccv), 135–152 (2018).

[46] Yu, L., Wang, S., Li, X., Fu, C.-W. & Heng, P.-A. 《用于半监督三维左心房分割的不确定性感知自集成模型》 载于 Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part II 22, 605–613 (Springer, 2019).

[47] Amin, S. U., Hussain, A., Kim, B. & Seo, S. 《用于数据标注并改善分类模型整体性能的深度主动学习技术》 Expert Systems with Applications 228, 120391 (2023).

[48] Liu, B., Chen, H., Wang, Z., Xie, W. & Shuai, L. 《LSNet：用于遥感图像变化检测的极轻量孪生网络》 载于 IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium, 2358–2361 (IEEE, 2022).

[49] Liu, B., Chen, H., Li, K. & Yang, M. Y. 《采用多任务一致性约束、基于 Transformer 的多模态变化检测》 Information Fusion 108, 102358 (2024).

[50] Bromley, J., Guyon, I., LeCun, Y., Säckinger, E. & Shah, R. 《采用孪生时延神经网络的签名验证》 Advances in neural information processing systems 6 (1993).

[51] Guo, E. et al. 《学习度量变化：用于场景变化检测的全卷积孪生度量网络》 arXiv preprint arXiv:1810.09111 (2018).

[52] Fridman, Y., Rusanovsky, M. & Oren, G. 《ChangeChip：面向 PCB 缺陷检测的基于参考的无监督变化检测》 载于 2021 IEEE Physical Assurance and Inspection of Electronics (PAINE), 1–8 (IEEE, 2021).

[53] Zagoruyko, S. & Komodakis, N. 《学习通过卷积神经网络比较图像图块》 载于 Proceedings of the IEEE conference on computer vision and pattern recognition, 4353–4361 (2015).

[54] Caye Daudt, R., Le Saux, B. & Boulch, A. 《用于变化检测的全卷积孪生网络》 载于 2018 25th IEEE International Conference on Image Processing (ICIP), 4063–4067 (2018).

[55] Zhan, Y. et al. 《基于深度孪生卷积网络的光学航空图像变化检测》 IEEE Geoscience and Remote Sensing Letters 14, 1845–1849 (2017).

[56] Hadsell, R., Chopra, S. & LeCun, Y. 《通过学习不变映射进行降维》 载于 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06), vol. 2, 1735–1742 (IEEE, 2006).

[57] Sae-Ang, B.-I., Kumwilaisak, W. & Kaewtrakulpong, P. 《采用自编码器辅助模块的半监督缺陷分割学习》 Sensors 22, 2915 (2022).

[58] Lv, C., Shen, F., Zhang, Z., Xu, D. & He, Y. 《基于稳定背景重建的新型逐像素缺陷检测方法》 IEEE Transactions on Instrumentation and Measurement 70, 1–13 (2020).

[59] Vaswani, A. et al. 《注意力就是你所需要的全部》 Advances in neural information processing systems 30 (2017).

[60] Wang, W. et al. 《金字塔视觉 Transformer：无需卷积的通用稠密预测主干》 载于 Proceedings of the IEEE/CVF international conference on computer vision, 568–578 (2021).

[61] Chen, J. et al. 《DASNet：用于高分辨率卫星图像变化检测的双注意全卷积孪生网络》 IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 14, 1194–1206 (2020).

[62] Hou, Q., Zhou, D. & Feng, J. 《面向高效移动网络设计的坐标注意力》 载于 Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 13713–13722 (2021).

[63] Ming, W. et al. 《基于机器视觉的液晶显示器 Mura 缺陷检测综述》 Crystals 11, 1444 (2021).

[64] Pérez, P., Gangnet, M. & Blake, A. 《泊松图像编辑》 载于 ACM SIGGRAPH 2003 Papers, 313–318 (2003).

[65] Liu, B.-Y., Chen, H.-X., Huang, Z., Liu, X. & Yang, Y.-Z. 《ZoomInNet：采用跨尺度知识蒸馏的新型无人机图像小目标检测器》 Remote Sensing 13, 1198 (2021).

[66] Long, J., Shelhamer, E. & Darrell, T. 《用于语义分割的全卷积网络》 载于 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3431–3440 (2015).

[67] Zhao, H., Shi, J., Qi, X., Wang, X. & Jia, J. 《金字塔场景解析网络》 载于 Proceedings of the IEEE conference on computer vision and pattern recognition, 2881–2890 (2017).

[68] Fu, J. et al. 《用于场景分割的双注意力网络》 载于 Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 3146–3154 (2019).

[69] Yuan, Y., Chen, X. & Wang, J. 《用于语义分割的对象上下文表示》 载于 Vedaldi, A., Bischof, H., Brox, T. & Frahm, J.-M. (eds.) Computer Vision – ECCV 2020, 173–190 (Springer International Publishing, Cham, 2020).

[70] Lin, T.-Y. et al. 《Microsoft COCO：上下文中的常见物体》 载于 European conference on computer vision, 740–755 (Springer, 2014).

## 致谢

本论文获中国广东省“扬帆计划”重大项目（[2020] 05 号）支持。

## 作者贡献

B.L. 构思研究、制定方法并撰写初稿。S.L. 进行数据分析和可视化。H.Z. 开发和整理 SynLCD 合成 LCD 缺陷数据集并管理数据。Y.Z. 进行验证。Z.H. 对稿件作重要修订。H.C. 管理项目、获取经费、监督研究设计，并参与撰写和编辑。所有作者均审阅稿件。

## 声明

### 利益冲突

作者声明不存在利益冲突。

### 其他信息

通信及材料请求请联系 Z.H. 或 H.C.。重印与许可信息见 <https://www.nature.com/reprints>。

**出版者声明：**Springer Nature 对所发表地图及机构隶属信息中的管辖权主张保持中立。

**开放获取：**本文采用知识共享署名 4.0 国际许可，允许以任何媒介或格式使用、共享、改编、传播和复制，条件是适当注明原作者和来源、提供许可链接，并说明是否作了修改。除非材料署名另有说明，文中图像或其他第三方材料均包括在本文知识共享许可中。若材料不在该许可内，且拟定用途不为法律法规允许或超出允许范围，须直接向版权持有人取得许可。许可文本见 <https://creativecommons.org/licenses/by/4.0/>。

© 作者 2025。
