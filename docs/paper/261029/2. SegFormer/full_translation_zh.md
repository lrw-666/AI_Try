# SegFormer：基于 Transformer 的简单高效语义分割设计

**原文：**[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](<SegFormer- Simple and Efficient Design for Semantic Segmentation with Transformers.pdf>)。本译文对应本目录的 18 页 PDF，arXiv:2105.15203v3，2021 年 10 月 28 日。翻译日期：2026-10-30。

**作者：**Enze Xie¹、Wenhai Wang²、Zhiding Yu³、Anima Anandkumar³˒⁴、Jose M. Alvarez³、Ping Luo¹。

**单位：**¹香港大学；²南京大学；³NVIDIA；⁴加州理工学院。

**翻译说明：**按原文段落翻译正文、附录 A—D、图 1—7 的说明、表 1—7 及参考文献；合并排版造成的跨行、跨页续段，去除运行页眉、页脚和孤立页码。公式、模型名、数据集名和引文编号保留。原始图形不重绘，图注及关键图内文字提供中文，图形本身请结合上方 PDF 查看。原文个别公式或措辞存在不一致，明确以“译注”标记，不静默修改实验数字。原文关于“当前最佳”的判断均指论文发表时。

## 摘要

我们提出 SegFormer：一种简单、高效而功能强大的语义分割框架，将 Transformer 与轻量多层感知机（multilayer perceptron，MLP）解码器结合起来。SegFormer 有两个突出特点：第一，它包含一种新颖的层级式 Transformer 编码器，可以输出多尺度特征。该编码器不需要位置编码，从而避免了当测试分辨率与训练分辨率不同时，对位置编码进行插值所导致的性能下降。第二，SegFormer 不使用复杂的解码器。我们提出的 MLP 解码器聚合不同层的信息，将局部注意力和全局注意力结合起来，形成具有较强表达能力的表征。我们证明，这种简单而轻量的设计是利用 Transformer 实现高效分割的关键。通过扩展模型规模，我们得到从 SegFormer-B0 到 SegFormer-B5 的一系列模型，其性能和效率均显著优于先前方法。例如，SegFormer-B4 在 ADE20K 上仅使用 6400 万参数便达到 50.3% 的平均交并比（mean Intersection over Union，mIoU），模型规模约为此前最佳方法的五分之一，而 mIoU 高出 2.2 个百分点。我们最强的模型 SegFormer-B5 在 Cityscapes 验证集上达到 84.0% mIoU，并在 Cityscapes-C 上表现出优异的零样本鲁棒性。代码将发布于：<https://github.com/NVlabs/SegFormer>。

## 1 引言

语义分割是计算机视觉中的一项基础任务，能够支持许多下游应用。它与图像分类密切相关，只是其输出是逐像素的类别预测，而不是图像级预测。一项奠基性工作 [1] 指出并系统研究了这种关系，作者将全卷积网络（fully convolutional network，FCN）用于语义分割任务。此后，FCN 启发了大量后续研究，并成为密集预测中的主流设计选择。

由于分类与语义分割之间存在紧密联系，许多先进的语义分割框架都是 ImageNet 图像分类常用架构的变体。因此，主干架构设计一直是语义分割领域的活跃研究方向。从早期使用 VGG 的方法 [1, 2]，到近期采用更深、更强主干的方法 [3]，主干的演进确实显著推动了语义分割的性能边界。除主干架构外，另一类研究将语义分割表述为结构化预测问题，重点设计能够有效捕获上下文信息的模块和算子。其中一个代表是空洞卷积 [4, 5]：通过在卷积核中插入空隙来“膨胀”卷积核，从而扩大感受野。

**图 1：ADE20K 上的性能与模型效率。**所有结果均使用单模型、单尺度推理。SegFormer 达到新的最佳性能 51.0% mIoU，同时比先前方法更高效。横轴为参数量，单位为百万；纵轴为 ADE20K mIoU。图中比较点包括 SegFormer-B0—B5、Twins、SETR、Swin Transformer、DeepLabV3+/R101、HRNet-W48＋OCR、PVT、SemFPN 和 FCN-R50。图内数值表如下，参数量单位为百万，FLOPs 单位为十亿次运算，FPS 为每秒帧数：

| 方法 | mIoU | 参数量 | FLOPs | FPS |
| --- | ---: | ---: | ---: | ---: |
| SegFormer-B0 | 37.4 | 3.7M | 8.4G | 50.5 |
| FCN-R50 | 36.1 | 49.6M | 198.0G | 23.5 |
| SegFormer-B2 | 46.5 | 27.5M | 62.4G | 24.5 |
| DeepLabV3+/R101 | 44.1 | 62.7M | 255.1G | 14.1 |
| HRNet-W48＋OCR | 43.0 | 70.5M | 164.8G | 17.0 |
| SegFormer-B4 | 50.3 | 64.1M | 95.7G | 15.4 |
| SETR | 48.6 | 318.3M | 362.1G | 5.4 |

Transformer 在自然语言处理（natural language processing，NLP）领域取得巨大成功后，研究者近期对将其引入视觉任务产生了浓厚兴趣。Dosovitskiy 等人 [6] 提出了用于图像分类的视觉 Transformer（Vision Transformer，ViT）。遵循 NLP 中的 Transformer 设计，作者将图像划分为多个经过线性嵌入的图像块，再将其连同位置嵌入（positional embedding，PE）输入标准 Transformer，在 ImageNet 上取得了出色性能。在语义分割中，Zheng 等人 [7] 提出 SETR，证明了 Transformer 用于该任务的可行性。SETR 以 ViT 为主干，并采用若干 CNN 解码器提高特征分辨率。尽管性能较好，ViT 仍有一些局限：其一，输出的是单尺度低分辨率特征，而非多尺度特征；其二，在大图像上的计算成本很高。为克服这些局限，Wang 等人 [8] 提出了金字塔视觉 Transformer（PVT），通过金字塔结构将 ViT 自然扩展到密集预测任务。与对应的 ResNet 相比，PVT 在目标检测和语义分割中获得了明显提升。然而，包括 Swin Transformer [9]、Twins [10] 在内的这些新方法主要考虑 Transformer 编码器的设计，忽略了解码器对进一步提升性能的贡献。

本文提出 SegFormer，一种同时考虑效率、准确性和鲁棒性的先进 Transformer 语义分割框架。与先前方法不同，我们重新设计了编码器与解码器。主要创新包括：

- 一种无需位置编码的层级式 Transformer 编码器。
- 一种轻量的全 MLP 解码器，无需复杂、计算开销大的模块，即可获得较强的表征。
- 如图 1 所示，SegFormer 在三个公开语义分割数据集上，在效率、准确性和鲁棒性方面均达到了新的最佳水平。

首先，我们提出的编码器在对不同于训练分辨率的图像进行推理时，无需对位置编码进行插值。因此，编码器可以方便地适应任意测试分辨率，而不受这一插值问题的影响。此外，层级结构使编码器能够同时生成高分辨率的精细特征与低分辨率的粗粒度特征；相比之下，ViT 只能产生固定分辨率的单一低分辨率特征图。

其次，我们提出一种轻量 MLP 解码器。关键思路是利用 Transformer 所产生的特征：较低层的注意力往往具有局部性，而最高层的注意力具有很强的非局部性。通过聚合不同层的信息，MLP 解码器将局部注意力和全局注意力结合起来，从而以简单、直接的结构获得强大的表征能力。

我们在 ADE20K、Cityscapes 和 COCO-Stuff 三个公开数据集上，展示了 SegFormer 在模型规模、运行时间和准确性方面的优势。在 Cityscapes 上，轻量模型 SegFormer-B0 无需 TensorRT 等加速实现，即可以 48 FPS 达到 71.9% mIoU。与 ICNet [11] 相比，其延迟与性能分别获得了原文所述的 60% 和 4.2% 改善。最大的 SegFormer-B5 达到 84.0% mIoU，比 SETR [7] 高出 1.8 个百分点，同时快五倍。在 ADE20K 上，该模型达到新的最佳结果 51.8% mIoU，模型规模约为 SETR 的四分之一。此外，我们的方法对常见退化和扰动的鲁棒性显著优于已有方法，因此适合对安全性要求较高的应用。代码将公开发布。

> 译注：原文以“%”表述若干 mIoU 差值；本译文在差值明确时使用“个百分点”。上段关于 ICNet 延迟的“60%”沿用原文陈述；具体速度和准确率见表 2，不将其另行改算为新的实验结论。

## 2 相关工作

**语义分割。**语义分割可以看作图像分类从图像级向像素级的扩展。在深度学习时代 [12—16]，FCN [1] 是语义分割的基础性工作：它通过全卷积网络，以端到端方式完成像素到像素的分类。此后，研究者从多个方面改进 FCN，包括扩大感受野 [17—19, 5, 2, 4, 20]、完善上下文信息 [21—29]、引入边界信息 [30—37]、设计各种注意力模块 [38—46]，以及使用自动机器学习技术 [47—51]。这些方法通过引入许多经验性模块，显著提高了语义分割性能，但也使框架变得复杂、计算开销增大。近期方法证明了基于 Transformer 的架构对语义分割的有效性 [7, 46]，但这些方法的计算成本仍然较高。

**Transformer 主干。**ViT [6] 首次证明，纯 Transformer 可以在图像分类中达到先进性能。ViT 将每幅图像视为词元（token）序列，输入多个 Transformer 层完成分类。随后，DeiT [52] 进一步探索了 ViT 的数据高效训练策略与蒸馏方法。T2T ViT [53]、CPVT [54]、TNT [55]、CrossViT [56] 和 LocalViT [57] 等近期方法针对 ViT 进行了专门修改，进一步提升图像分类性能。

在分类之外，PVT [8] 首次将金字塔结构引入 Transformer，展示了纯 Transformer 主干在密集预测任务中相对 CNN 主干的潜力。此后，Swin [9]、CvT [58]、CoaT [59]、LeViT [60]、Twins [10] 等方法增强了特征的局部连续性，去除了固定大小的位置嵌入，从而提高 Transformer 在密集预测任务中的表现。

**面向特定任务的 Transformer。**DETR [52] 首次使用 Transformer 构建了无需非极大值抑制（non-maximum suppression，NMS）的端到端目标检测框架。其他工作还将 Transformer 用于跟踪 [61, 62]、超分辨率 [63]、重识别 [64]、图像着色 [65]、检索 [66] 和多模态学习 [67, 68]。在语义分割中，SETR [7] 使用 ViT [6] 主干提取特征，并取得了出色性能。然而，这些基于 Transformer 的方法效率较低，因此难以部署于实时应用。

> 译注：前文“DeiT [52]”的编号按原文保留，但参考文献 [52] 实际是 DETR 论文；本段 DETR 的编号则与参考文献相符。

## 3 方法

本节介绍 SegFormer：一种高效、鲁棒而强大的分割框架，不使用人工设计且计算成本高昂的模块。如图 2 所示，SegFormer 包含两个主要模块：第一，层级式 Transformer 编码器，用于产生高分辨率粗粒度特征与低分辨率精细特征；第二，轻量全 MLP 解码器，将这些多层级特征融合，生成最终的语义分割掩码。

给定尺寸为 $H\times W\times3$ 的图像，我们首先将其划分为 $4\times4$ 的图像块。与 ViT 使用的 $16\times16$ 图像块相比，更小的图像块有利于密集预测。接着，将图像块输入层级式 Transformer 编码器，得到原图分辨率 $\{1/4,1/8,1/16,1/32\}$ 的多层级特征，再将这些特征输入全 MLP 解码器，预测分辨率为 $\frac H4\times\frac W4\times N_{\mathrm{cls}}$ 的分割掩码，其中 $N_{\mathrm{cls}}$ 是类别数。下文将详细介绍编码器与解码器，并总结我们的方法与 SETR 的主要区别。

**图 2：SegFormer 框架。**框架包含两个主要模块：用于提取粗粒度与精细特征的层级式 Transformer 编码器，以及直接融合这些多层级特征、预测语义分割掩码的轻量全 MLP 解码器。“FFN”表示前馈网络。图内模块依次为：重叠图像块嵌入、Transformer 块 1—4、重叠图像块合并；每个块包含高效自注意力与 Mix-FFN。解码端对各尺度特征作 MLP 映射、上采样、拼接与 MLP 融合，最终输出 $H/4\times W/4\times N_{\mathrm{cls}}$ 的预测。“Encoder”“Decoder”“UpSample”分别为编码器、解码器、上采样。

### 3.1 层级式 Transformer 编码器

我们设计了一系列混合 Transformer 编码器（Mix Transformer，MiT），从 MiT-B0 到 MiT-B5。它们采用相同的总体架构，但规模不同。MiT-B0 是面向快速推理的轻量模型，而 MiT-B5 是用于取得最佳性能的最大模型。MiT 的设计部分受 ViT 启发，但针对语义分割进行了定制与优化。

**层级特征表示。**与只能生成单一分辨率特征图的 ViT 不同，本模块的目标是根据输入图像生成类似 CNN 的多层级特征。这些特征同时提供高分辨率粗粒度特征与低分辨率细粒度特征，通常可以提升语义分割性能。更具体地，给定分辨率为 $H\times W\times3$ 的输入图像，我们通过图像块合并，得到分辨率为 $\frac{H}{2^{i+1}}\times\frac{W}{2^{i+1}}\times C_i$ 的层级特征图 $F_i$，其中 $i\in\{1,2,3,4\}$，且 $C_{i+1}>C_i$。

> 译注：原文在此处及第 3 节开头写作“高分辨率粗粒度、低分辨率细粒度”，而引言写作“高分辨率精细、低分辨率粗粒度”。译文分别保留其原意；不要将这种不一致视为额外的结构定义。

**重叠图像块合并。**对于一个图像块，ViT 中的图像块合并过程将 $N\times N\times3$ 图像块映射为 $1\times1\times C$ 向量。这一过程可以自然扩展为将 $2\times2\times C_i$ 的特征块映射为 $1\times1\times C_{i+1}$ 向量，从而获得层级特征图。这样就能把 $F_1$ 的尺寸从 $\frac H4\times\frac W4\times C_1$ 缩小为 $F_2$ 的 $\frac H8\times\frac W8\times C_2$，并重复这一过程，得到其他层级的特征图。该过程最初用于组合互不重叠的图像块或特征块，因此不能保留图像块周围的局部连续性。我们改为使用重叠的图像块合并过程。为此定义 $K,S,P$：$K$ 是图像块大小，$S$ 是相邻图像块之间的步长，$P$ 是填充大小。实验采用 $K=7,S=4,P=3$ 以及 $K=3,S=2,P=1$，在进行重叠合并时产生与非重叠过程相同尺寸的特征。

**高效自注意力。**编码器的主要计算瓶颈是自注意力层。在原始多头自注意力中，$Q,K,V$ 的尺寸均为 $N\times C$，其中 $N=H\times W$ 为序列长度。自注意力计算为：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_{\mathrm{head}}}}\right)V.\tag{1}
$$

该过程的计算复杂度为 $O(N^2)$，对于高分辨率图像，成本过高。因此，我们使用文献 [8] 引入的序列缩减过程，通过缩减比 $R$ 减少序列长度：

$$
\begin{aligned}
\widehat K&=\operatorname{Reshape}\!\left(\frac NR,C\cdot R\right)(K),\\
K&=\operatorname{Linear}(C\cdot R,C)(\widehat K).
\end{aligned}\tag{2}
$$

其中，$K$ 是待缩减的序列，$\operatorname{Reshape}(N/R,C\cdot R)(K)$ 表示将其重排为 $\frac NR\times(C\cdot R)$ 的形状；$\operatorname{Linear}(C_{\mathrm{in}},C_{\mathrm{out}})(\cdot)$ 表示将输入的 $C_{\mathrm{in}}$ 维张量映射为 $C_{\mathrm{out}}$ 维输出的线性层。新的 $K$ 因而具有 $\frac NR\times C$ 的尺寸。自注意力的复杂度由 $O(N^2)$ 降至 $O(N^2/R)$。实验中，从阶段 1 到阶段 4，$R$ 分别设为 $[64,16,4,1]$。

**Mix-FFN。**ViT 使用位置编码（positional encoding，PE）引入位置信息，但其分辨率固定。因此，当测试分辨率与训练分辨率不同时，需要对位置编码进行插值，这往往导致准确率下降。为缓解这一问题，CPVT [54] 使用 $3\times3$ 卷积结合位置编码，实现数据驱动的位置编码。我们认为，语义分割实际上不一定需要显式位置编码。我们提出 Mix-FFN，利用零填充能够泄露位置信息的特性 [69]，直接在前馈网络（feed-forward network，FFN）中使用 $3\times3$ 卷积：

$$
x_{\mathrm{out}}=\operatorname{MLP}\!\left(\operatorname{GELU}\!\left(\operatorname{Conv}_{3\times3}(\operatorname{MLP}(x_{\mathrm{in}}))\right)\right)+x_{\mathrm{in}}.\tag{3}
$$

其中，$x_{\mathrm{in}}$ 是自注意力模块输出的特征。Mix-FFN 在每个 FFN 中组合 $3\times3$ 卷积与 MLP。实验将表明，一个 $3\times3$ 卷积就足以为 Transformer 提供位置信息。具体而言，我们使用逐通道卷积，以减少参数量并提高效率。

### 3.2 轻量全 MLP 解码器

SegFormer 使用仅由 MLP 层组成的轻量解码器，从而避免其他方法中常用的人工设计且计算成本较高的组件。如此简单的解码器之所以有效，关键在于我们的层级式 Transformer 编码器比传统 CNN 编码器具有更大的有效感受野（effective receptive field，ERF）。

全 MLP 解码器包含四个主要步骤。首先，MiT 编码器的多层级特征 $F_i$ 通过 MLP 层统一通道维度。其次，将各特征上采样到原图分辨率的四分之一，并进行拼接。然后，使用一个 MLP 层融合拼接后的特征。最后，通过另一个 MLP 层，根据融合特征预测分辨率为 $\frac H4\times\frac W4\times N_{\mathrm{cls}}$ 的分割掩码 $M$，其中 $N_{\mathrm{cls}}$ 是类别数。因此，解码器可表示为：

$$
\begin{aligned}
\widehat F_i&=\operatorname{Linear}(C_i,C)(F_i),&&\forall i,\\
\widehat F_i&=\operatorname{Upsample}\!\left(\frac W4\times\frac W4\right)(\widehat F_i),&&\forall i,\\
F&=\operatorname{Linear}(4C,C)(\operatorname{Concat}(\widehat F_i)),&&\forall i,\\
M&=\operatorname{Linear}(C,N_{\mathrm{cls}})(F).
\end{aligned}\tag{4}
$$

其中，$M$ 表示预测掩码，$\operatorname{Linear}(C_{\mathrm{in}},C_{\mathrm{out}})$ 的输入与输出向量维度分别为 $C_{\mathrm{in}}$ 和 $C_{\mathrm{out}}$。

> 译注：式 (4) 的上采样尺寸在原 PDF 中排为 $W/4\times W/4$，此处原样保留；相邻正文及模型结构给出的目标尺寸是 $H/4\times W/4$。复现时应依据实际图像高、宽处理。

**有效感受野分析。**在语义分割中，维持足够大的感受野、纳入上下文信息一直是核心问题 [5, 19, 20]。我们使用有效感受野 [70] 这一工具，可视化并解释 MLP 解码器为何在 Transformer 上如此有效。图 3 展示了 DeepLabv3+ 与 SegFormer 四个编码阶段及解码头的有效感受野，可以得到以下观察：

- 即使在最深的阶段 4，DeepLabv3+ 的有效感受野仍然相对较小。
- SegFormer 的编码器在较低阶段自然地产生类似卷积的局部注意力，而在阶段 4 能产生高度非局部的注意力，有效捕获上下文。
- 如图 3 放大的区域所示，MLP 头的有效感受野（蓝框）与阶段 4（红框）不同：除非局部注意力外，它还具有明显更强的局部注意力。

CNN 的有限感受野使其需要借助 ASPP [18] 等上下文模块扩大感受野，但这不可避免地增加了模型负担。我们的解码器利用 Transformer 的非局部注意力，以简单结构获得更大的感受野。然而，同样的解码器在 CNN 主干上效果不佳，因为整个模型的感受野仍受阶段 4 的有限感受野约束；表 1d 将验证这一点。

更重要的是，我们的解码器实质上利用了 Transformer 所诱导的特征，使高度局部与非局部注意力可以同时出现。通过统一这两种信息，MLP 解码器只增加少量参数，就能得到互补且强大的表征。这也是推动我们进行这一设计的另一个关键原因。仅使用阶段 4 的非局部注意力并不足以取得良好结果，表 1d 将对此加以验证。

**图 3：Cityscapes 上的有效感受野。**结果为 100 幅图像的平均值。上行为 DeepLabv3+，下行为 SegFormer；展示两种架构四个阶段及解码头的有效感受野，建议放大查看。图内“Stage-1”—“Stage-4”为阶段 1—4，“Head”为解码头。

### 3.3 与 SETR 的关系

与 SETR [7] 相比，SegFormer 采用了多项更高效、更强大的设计：

- 我们只使用 ImageNet-1K 进行预训练，而 SETR 中的 ViT 使用更大的 ImageNet-22K 预训练。
- SegFormer 编码器使用层级架构，规模小于 ViT，能够同时捕获高分辨率粗粒度特征和低分辨率精细特征；SETR 的 ViT 编码器只能生成单一低分辨率特征图。
- 我们去除了编码器中的位置嵌入，而 SETR 使用固定形状的位置嵌入，当推理与训练分辨率不同时会降低准确率。
- 我们的 MLP 解码器比 SETR 的解码器更紧凑，计算需求更低，额外计算开销可以忽略。SETR 则需要包含多个 $3\times3$ 卷积的重型解码器。

## 4 实验

### 4.1 实验设置

**数据集。**我们使用 Cityscapes [71]、ADE20K [72] 和 COCO-Stuff [73] 三个公开数据集。ADE20K 是场景解析数据集，包含 20,210 幅图像，覆盖 150 个细粒度语义概念。Cityscapes 是面向驾驶场景语义分割的数据集，包含 5,000 幅精细标注的高分辨率图像，具有 19 个类别。COCO-Stuff 覆盖 172 个标签，包含约 164,000 幅图像，其中训练集 118,000 幅、验证集 5,000 幅、test-dev 集 20,000 幅、test-challenge 集 20,000 幅。

**实现细节。**我们采用 mmsegmentation 代码库¹，在配备 8 块 Tesla V100 的服务器上训练。编码器在 ImageNet-1K 上预训练，解码器随机初始化。训练时使用 0.5—2.0 倍随机缩放、随机水平翻转和随机裁剪进行数据增强；ADE20K、Cityscapes、COCO-Stuff 的裁剪大小分别为 $512\times512$、$1024\times1024$、$512\times512$。对于最大的 B5 模型，遵循文献 [9]，在 ADE20K 上使用 $640\times640$ 裁剪。所有模型均使用 AdamW 优化器，在 ADE20K 和 Cityscapes 上训练 160,000 次迭代，在 COCO-Stuff 上训练 80,000 次迭代；消融实验例外，仅训练 40,000 次迭代。ADE20K 与 COCO-Stuff 的批量大小为 16，Cityscapes 为 8。初始学习率为 0.00006，默认使用指数为 1.0 的多项式学习率调度。为保持简单，我们不采用在线困难样本挖掘（OHEM）、辅助损失、类别平衡损失等常见技巧。评估 ADE20K 与 COCO-Stuff 时，将图像短边缩放到训练裁剪尺寸，并保持长宽比；Cityscapes 使用 $1024\times1024$ 窗口进行滑窗测试。语义分割性能以 mIoU 衡量。

¹ mmsegmentation：<https://github.com/open-mmlab/mmsegmentation>。

### 4.2 消融研究

**模型规模的影响。**首先分析增大编码器规模对性能和模型效率的影响。图 1 展示 ADE20K 上性能与效率如何随编码器规模变化，表 1a 汇总三个数据集的结果。首先值得注意的是解码器相对于编码器的规模：轻量模型的解码器仅有 40 万参数；MiT-B5 模型的解码器仅占总参数量的约 4%。总体而言，增大编码器规模会在所有数据集上带来稳定的性能提升。轻量模型 SegFormer-B0 紧凑、高效，同时保持具有竞争力的性能，说明其适合实时应用。另一方面，最大的 SegFormer-B5 在三个数据集上均取得最佳结果，展示了我们 Transformer 编码器的潜力。

**MLP 解码器通道维度 $C$ 的影响。**接着分析第 3.2 节中解码器通道维度 $C$ 的影响。表 1b 展示准确率、浮点运算量和参数量随 $C$ 的变化。$C=256$ 时，性能与计算成本已具有较强竞争力。增大 $C$ 能提高性能，但会增大模型并降低效率。有趣的是，通道维度超过 768 后，性能趋于饱和。因此，实时模型 SegFormer-B0、B1 采用 $C=256$，其余模型采用 $C=768$。

**表 1：与模型规模、编码器和解码器设计有关的消融实验。**下列表格中，参数量单位为百万，浮点运算量单位为十亿次；向下箭头表示越低越好，向上箭头表示越高越好。SS/MS 分别表示单尺度/多尺度测试。

**表 1a：三个数据集上的准确率、参数量与计算量随模型规模的变化。**

| 编码器 | 编码器参数 | 解码器参数 | ADE20K 运算量↓ | ADE20K mIoU（SS/MS）↑ | Cityscapes 运算量↓ | Cityscapes mIoU（SS/MS）↑ | COCO-Stuff 运算量↓ | COCO-Stuff mIoU（SS）↑ |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| MiT-B0 | 3.4 | 0.4 | 8.4 | 37.4 / 38.0 | 125.5 | 76.2 / 78.1 | 8.4 | 35.6 |
| MiT-B1 | 13.1 | 0.6 | 15.9 | 42.2 / 43.1 | 243.7 | 78.5 / 80.0 | 15.9 | 40.2 |
| MiT-B2 | 24.2 | 3.3 | 62.4 | 46.5 / 47.5 | 717.1 | 81.0 / 82.2 | 62.4 | 44.6 |
| MiT-B3 | 44.0 | 3.3 | 79.0 | 49.4 / 50.0 | 962.9 | 81.7 / 83.3 | 79.0 | 45.5 |
| MiT-B4 | 60.8 | 3.3 | 95.7 | 50.3 / 51.1 | 1240.6 | 82.3 / 83.9 | 95.7 | 46.5 |
| MiT-B5 | 81.4 | 3.3 | 183.3 | 51.0 / 51.8 | 1460.4 | 82.4 / 84.0 | 111.6 | 46.7 |

**表 1b：ADE20K 上，准确率随解码器 MLP 维度 $C$ 的变化。**

| $C$ | 运算量↓ | 参数量↓ | mIoU↑ |
| ---: | ---: | ---: | ---: |
| 256 | 25.7 | 24.7 | 44.9 |
| 512 | 39.8 | 25.8 | 45.0 |
| 768 | 62.4 | 27.5 | 45.4 |
| 1024 | 93.6 | 29.6 | 45.2 |
| 2048 | 304.4 | 43.4 | 45.6 |

**表 1c：Cityscapes 不同测试分辨率下，Mix-FFN 与位置编码（PE）的比较。**

| 推理分辨率 | 编码方式 | mIoU↑ |
| --- | --- | ---: |
| 768×768 | PE | 77.3 |
| 1024×2048 | PE | 74.0 |
| 768×768 | Mix-FFN | 80.5 |
| 1024×2048 | Mix-FFN | 79.8 |

**表 1d：CNN 和 Transformer 编码器搭配 MLP 解码器时，在 ADE20K 上的准确率。**S4 表示仅使用阶段 4 特征，S1—4 表示使用阶段 1—4 特征。

| 编码器 | 运算量↓ | 参数量↓ | mIoU↑ |
| --- | ---: | ---: | ---: |
| ResNet50（S1—4） | 69.2 | 29.0 | 34.7 |
| ResNet101（S1—4） | 88.7 | 47.9 | 38.7 |
| ResNeXt101（S1—4） | 127.5 | 86.8 | 39.8 |
| MiT-B2（S4） | 22.3 | 24.7 | 43.1 |
| MiT-B2（S1—4） | 62.4 | 27.7 | 45.4 |
| MiT-B3（S1—4） | 79.0 | 47.3 | 48.6 |

**Mix-FFN 与位置编码的比较。**本实验分析将 Transformer 编码器中的位置编码替换为 Mix-FFN 的效果。分别训练使用位置编码和 Mix-FFN 的编码器，再在 Cityscapes 上用两种分辨率推理：$768\times768$ 的滑窗测试，以及 $1024\times2048$ 的整图测试。

表 1c 显示，在给定分辨率下，使用 Mix-FFN 明显优于位置编码。此外，该方法对测试分辨率的变化不那么敏感：使用位置编码时，不同测试分辨率之间的准确率下降为 3.3 个百分点；采用 Mix-FFN 后，降幅仅为 0.7 个百分点。这说明 Mix-FFN 能生成更准确、更鲁棒的编码器。

> 译注：源文将上述 3.3% 的下降描述为使用较低分辨率时发生，但表 1c 中较低分辨率对应的数值反而较高。这里按表格数值将其表述为两种分辨率间的差值，不改变结果。

**有效感受野评估。**第 3.2 节指出，相比其他 CNN 模型，Transformer 具有更大的有效感受野，MLP 解码器因此受益。为量化这一点，我们将 MLP 解码器与 ResNet、ResNeXt 等 CNN 编码器结合。如表 1d 所示，搭配 CNN 编码器的准确率明显低于搭配所提出 Transformer 编码器的准确率。直观而言，CNN 的感受野较小，因此 MLP 解码器不足以完成全局推理；而 Transformer 编码器搭配 MLP 解码器则取得最佳效果。此外，对于 Transformer 编码器，需要结合低层局部特征与高层非局部特征，不能只使用高层特征。

### 4.3 与先进方法的比较

下面在 ADE20K [72]、Cityscapes [71] 和 COCO-Stuff [73] 上，与已有方法进行比较。

**ADE20K 与 Cityscapes。**表 2 汇总这两个数据集上的参数量、浮点运算量、延迟和准确率。表格上半部分为实时方法，包括已有先进方法及使用轻量 MiT-B0 编码器的 SegFormer；下半部分更关注性能，报告使用更强编码器的相关方法及我们的结果。

**表 2：ADE20K 和 Cityscapes 上与先进方法的比较。**SegFormer 在参数量、运算量、速度和准确率方面有明显优势。对于 SegFormer-B0，将图像短边缩放为 $\{1024,768,640,512\}$，以获得不同的速度—准确率权衡。参数量单位为百万、FLOPs 单位为十亿次；“—”表示原文未报告。为便于阅读，原表拆分如下。

| 类别 | 方法 | 编码器 | 参数量↓ | ADE20K FLOPs↓ | ADE20K FPS↑ | ADE20K mIoU↑ | Cityscapes FLOPs↓ | Cityscapes FPS↑ | Cityscapes mIoU↑ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 实时 | FCN [1] | MobileNetV2 | 9.8 | 39.6 | 64.4 | 19.7 | 317.1 | 14.2 | 61.5 |
| 实时 | ICNet [11] | — | — | — | — | — | — | 30.3 | 67.7 |
| 实时 | PSPNet [17] | MobileNetV2 | 13.7 | 52.9 | 57.7 | 29.6 | 423.4 | 11.2 | 70.2 |
| 实时 | DeepLabV3+ [20] | MobileNetV2 | 15.4 | 69.4 | 43.1 | 34.0 | 555.4 | 8.4 | 75.2 |
| 实时 | SegFormer（本文，短边 1024） | MiT-B0 | 3.8 | 8.4 | 50.5 | 37.4 | 125.5 | 15.2 | 76.2 |
| 实时 | SegFormer（本文，短边 768） | MiT-B0 | 3.8 | — | — | — | 51.7 | 26.3 | 75.3 |
| 实时 | SegFormer（本文，短边 640） | MiT-B0 | 3.8 | — | — | — | 31.5 | 37.1 | 73.7 |
| 实时 | SegFormer（本文，短边 512） | MiT-B0 | 3.8 | — | — | — | 17.7 | 47.6 | 71.9 |
| 非实时 | FCN [1] | ResNet-101 | 68.6 | 275.7 | 14.8 | 41.4 | 2203.3 | 1.2 | 76.6 |
| 非实时 | EncNet [24] | ResNet-101 | 55.1 | 218.8 | 14.9 | 44.7 | 1748.0 | 1.3 | 76.9 |
| 非实时 | PSPNet [17] | ResNet-101 | 68.1 | 256.4 | 15.3 | 44.4 | 2048.9 | 1.2 | 78.5 |
| 非实时 | CCNet [41] | ResNet-101 | 68.9 | 278.4 | 14.1 | 45.2 | 2224.8 | 1.0 | 80.2 |
| 非实时 | DeepLabV3+ [20] | ResNet-101 | 62.7 | 255.1 | 14.1 | 44.1 | 2032.3 | 1.2 | 80.9 |
| 非实时 | OCRNet [23] | HRNet-W48 | 70.5 | 164.8 | 17.0 | 45.6 | 1296.8 | 4.2 | 81.1 |
| 非实时 | GSCNN [35] | WideResNet38 | — | — | — | — | — | — | 80.8 |
| 非实时 | Axial-DeepLab [74] | AxialResNet-XL | — | — | — | — | 2446.8 | — | 81.1 |
| 非实时 | Dynamic Routing [75] | Dynamic-L33-PSP | — | — | — | — | 270.0 | — | 80.7 |
| 非实时 | Auto-Deeplab [50] | NAS-F48-ASPP | — | — | — | 44.0 | 695.0 | — | 80.3 |
| 非实时 | SETR [7] | ViT-Large | 318.3 | — | 5.4 | 50.2 | — | 0.5 | 82.2 |
| 非实时 | SegFormer（本文） | MiT-B4 | 64.1 | 95.7 | 15.4 | 51.1 | 1240.6 | 3.0 | 83.8 |
| 非实时 | SegFormer（本文） | MiT-B5 | 84.7 | 183.3 | 9.8 | 51.8 | 1447.6 | 2.5 | 84.0 |

在 ADE20K 上，SegFormer-B0 仅用 380 万参数和 8.4G FLOPs 达到 37.4% mIoU，在参数量、运算量和延迟方面均优于其他实时方法。例如，相比使用 MobileNetV2 的 DeepLabV3+，SegFormer-B0 快 7.4 FPS，mIoU 高 3.4 个百分点。SegFormer-B5 则超过包括此前最佳 SETR 在内的所有方法，达到新的最佳结果 51.8% mIoU，比 SETR 高 1.6 个百分点，同时效率明显更高。

表 2 也表明，这些优势在 Cityscapes 上同样成立。图像短边为 1024 时，SegFormer-B0 达到 15.2 FPS 和 76.2% mIoU，相比 DeepLabV3+ 提高 1.3 个百分点，速度约为其两倍。短边为 512 时，SegFormer-B0 达到 47.6 FPS 和 71.9% mIoU，比 ICNet 快 17.3 FPS，mIoU 高 4.2 个百分点。SegFormer-B5 达到最佳的 84.0% mIoU，比所有已有方法至少高 1.8 个百分点，同时比 SETR [7] 快五倍，模型约为其四分之一。

在 Cityscapes 测试集上，我们遵循常用设置 [20]，将验证图像合并至训练集，分别报告仅用 ImageNet-1K 预训练，以及额外使用 Mapillary Vistas [76] 的结果。如表 3 所示，仅使用 Cityscapes 精细标注数据和 ImageNet-1K 预训练，本方法即达到 82.2% mIoU，超过包括 SETR 在内的所有方法；后者还使用了 ImageNet-22K 预训练和额外的 Cityscapes 粗标注数据。使用 Mapillary 预训练后，本方法达到新的最佳结果 83.1% mIoU。图 4 展示了 Cityscapes 上的定性结果：SegFormer 比 SETR 提供更精细的细节，比 DeepLabV3+ 提供更平滑的预测。

**表 3：Cityscapes 测试集上与先进方法的比较。**IM-1K、IM-22K、Coarse、MV 分别表示 ImageNet-1K、ImageNet-22K、Cityscapes 粗标注集和 Mapillary Vistas。在额外数据相同或更少的情况下，SegFormer 优于对照方法。

| 方法 | 编码器 | 额外数据 | mIoU |
| --- | --- | --- | ---: |
| PSPNet [17] | ResNet-101 | IM-1K | 78.4 |
| PSANet [43] | ResNet-101 | IM-1K | 80.1 |
| CCNet [41] | ResNet-101 | IM-1K | 81.9 |
| OCNet [21] | ResNet-101 | IM-1K | 80.1 |
| Axial-DeepLab [74] | AxiaiResNet-XL | IM-1K | 79.9 |
| SETR [7] | ViT | IM-22K | 81.0 |
| SETR [7] | ViT | IM-22K、Coarse | 81.6 |
| SegFormer | MiT-B5 | IM-1K | 82.2 |
| SegFormer | MiT-B5 | IM-1K、MV | 83.1 |

**图 4：Cityscapes 上的定性结果。**与 SETR 相比，SegFormer 在物体边界附近预测出明显更精细的掩码；与 DeepLabV3+ 相比，SegFormer 减少了红色标示的长距离错误。建议在屏幕上查看。

**COCO-Stuff。**最后，我们在完整 COCO-Stuff 数据集上评估 SegFormer。由于已有方法没有报告该完整数据集的结果，我们复现了最具代表性的 DeepLabV3+、OCRNet 和 SETR。本数据集上的 FLOPs 与 ADE20K 上报告的相同。如表 4 所示，SegFormer-B5 仅用 8470 万参数即达到 46.7% mIoU，比 SETR 高 0.9 个百分点，模型规模约为其四分之一。总体而言，这些结果证明，SegFormer 在语义分割的准确率、计算成本和模型规模方面具有优势。

**表 4：完整 COCO-Stuff 数据集上的结果。**数据集包含 COCO 2017 的全部约 164,000 幅图像，覆盖 172 类。参数量单位为百万。

| 方法 | 编码器 | 参数量 | mIoU |
| --- | --- | ---: | ---: |
| DeepLabV3+ [20] | ResNet50 | 43.7 | 38.4 |
| OCRNet [23] | HRNet-W48 | 70.5 | 42.3 |
| SETR [7] | ViT | 305.7 | 45.8 |
| SegFormer | MiT-B5 | 84.7 | 46.7 |

### 4.4 对自然退化的鲁棒性

模型鲁棒性对自动驾驶等安全关键任务非常重要 [77]。本实验评估 SegFormer 对常见退化和扰动的鲁棒性。遵循文献 [77]，我们构建 Cityscapes-C：在 Cityscapes 验证集上施加由算法生成的 16 类退化，分属噪声、模糊、天气和数字变换四类。我们将本方法与文献 [77] 报告的 DeepLabV3+ 变体及其他方法比较，结果见表 5。

本方法显著优于先前方法，对高斯噪声的相对提升最高达到 588%，对雪天天气的相对提升最高达到 295%。结果表明 SegFormer 具有较强鲁棒性，我们期望它能使重视鲁棒性的安全关键应用受益。

**表 5：Cityscapes-C 上的主要结果。**DLv3+、MBv2、R、X 分别表示 DeepLabv3+、MobileNetv2、ResNet、Xception。对照方法的 mIoU 来自文献 [77]。原宽表按退化类别拆分，所有数值保留。

| 方法 | 无退化 | 运动模糊 | 散焦模糊 | 玻璃模糊 | 高斯模糊 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLv3+（MBv2） | 72.0 | 53.5 | 49.0 | 45.3 | 49.1 |
| DLv3+（R50） | 76.6 | 58.5 | 56.6 | 47.2 | 57.7 |
| DLv3+（R101） | 77.1 | 59.1 | 56.3 | 47.7 | 57.3 |
| DLv3+（X41） | 77.8 | 61.6 | 54.9 | 51.0 | 54.7 |
| DLv3+（X65） | 78.4 | 63.9 | 59.1 | 52.8 | 59.2 |
| DLv3+（X71） | 78.6 | 64.1 | 60.9 | 52.0 | 60.4 |
| ICNet | 65.9 | 45.8 | 44.6 | 47.4 | 44.7 |
| FCN8s | 66.7 | 42.7 | 31.1 | 37.0 | 34.1 |
| DilatedNet | 68.6 | 44.4 | 36.3 | 32.5 | 38.4 |
| ResNet-38 | 77.5 | 54.6 | 45.1 | 43.3 | 47.2 |
| PSPNet | 78.8 | 59.8 | 53.2 | 44.4 | 53.9 |
| GSCNN | 80.9 | 58.9 | 58.4 | 41.9 | 60.1 |
| SegFormer-B5 | 82.4 | 69.1 | 68.6 | 64.1 | 69.8 |

| 方法 | 高斯噪声 | 脉冲噪声 | 散粒噪声 | 斑点噪声 |
| --- | ---: | ---: | ---: | ---: |
| DLv3+（MBv2） | 6.4 | 7.0 | 6.6 | 16.6 |
| DLv3+（R50） | 6.5 | 7.2 | 10.0 | 31.1 |
| DLv3+（R101） | 13.2 | 13.9 | 16.3 | 36.9 |
| DLv3+（X41） | 17.0 | 17.3 | 21.6 | 43.7 |
| DLv3+（X65） | 15.0 | 10.6 | 19.8 | 42.4 |
| DLv3+（X71） | 14.9 | 10.8 | 19.4 | 41.2 |
| ICNet | 8.4 | 8.4 | 10.6 | 27.9 |
| FCN8s | 6.7 | 5.7 | 7.8 | 24.9 |
| DilatedNet | 15.6 | 14.0 | 18.4 | 32.7 |
| ResNet-38 | 13.7 | 16.0 | 18.2 | 38.3 |
| PSPNet | 11.0 | 15.4 | 15.4 | 34.2 |
| GSCNN | 5.5 | 2.6 | 6.8 | 24.7 |
| SegFormer-B5 | 57.8 | 63.4 | 52.3 | 72.8 |

| 方法 | 亮度 | 对比度 | 饱和度 | JPEG 压缩 |
| --- | ---: | ---: | ---: | ---: |
| DLv3+（MBv2） | 51.7 | 46.7 | 32.4 | 27.2 |
| DLv3+（R50） | 58.2 | 54.7 | 41.3 | 27.4 |
| DLv3+（R101） | 59.2 | 54.5 | 41.5 | 37.4 |
| DLv3+（X41） | 63.6 | 56.9 | 51.7 | 38.5 |
| DLv3+（X65） | 65.9 | 59.1 | 46.1 | 31.4 |
| DLv3+（X71） | 68.0 | 58.7 | 47.1 | 40.2 |
| ICNet | 41.0 | 33.1 | 27.5 | 34.0 |
| FCN8s | 53.3 | 39.0 | 36.0 | 21.2 |
| DilatedNet | 52.7 | 32.6 | 38.1 | 29.1 |
| ResNet-38 | 60.0 | 50.6 | 46.9 | 14.7 |
| PSPNet | 60.4 | 51.8 | 30.6 | 21.4 |
| GSCNN | 75.9 | 61.9 | 70.7 | 12.0 |
| SegFormer-B5 | 81.0 | 77.7 | 80.1 | 58.8 |

| 方法 | 雪 | 飞溅 | 雾 | 霜 |
| --- | ---: | ---: | ---: | ---: |
| DLv3+（MBv2） | 13.7 | 38.9 | 47.4 | 17.3 |
| DLv3+（R50） | 12.0 | 42.0 | 55.9 | 22.8 |
| DLv3+（R101） | 11.9 | 47.8 | 55.1 | 22.7 |
| DLv3+（X41） | 18.2 | 46.6 | 57.6 | 20.6 |
| DLv3+（X65） | 19.3 | 50.7 | 63.6 | 23.8 |
| DLv3+（X71） | 18.8 | 50.4 | 64.1 | 20.2 |
| ICNet | 6.3 | 30.5 | 27.3 | 11.0 |
| FCN8s | 11.3 | 31.6 | 37.6 | 19.7 |
| DilatedNet | 12.5 | 32.3 | 34.7 | 19.2 |
| ResNet-38 | 13.5 | 45.9 | 52.9 | 22.2 |
| PSPNet | 8.4 | 42.7 | 34.4 | 16.2 |
| GSCNN | 12.4 | 47.3 | 67.9 | 32.6 |
| SegFormer-B5 | 40.7 | 68.4 | 78.5 | 49.9 |

## 5 结论

本文提出 SegFormer：一种简单、简洁而强大的语义分割方法，包含无需位置编码的层级式 Transformer 编码器，以及轻量全 MLP 解码器。它避免了先前方法常见的复杂设计，同时实现高效率和高性能。SegFormer 不仅在常用数据集上取得新的最佳结果，也展示了较强的零样本鲁棒性。我们希望它能成为语义分割的可靠基线，推动进一步研究。其局限之一是：虽然最小的 370 万参数模型已小于已知的 CNN 模型，但它能否在仅有 100k 内存的边缘设备芯片上良好运行，仍不清楚。我们将其留作未来工作。

## 致谢

感谢 Ding Liang、Zhe Chen 和 Yaojun Liu 提供富有洞见的讨论；没有这些讨论，本文将难以完成。

## 附录 A MiT 系列的详细信息

本节列出混合 Transformer（MiT）编码器的重要超参数。通过改变这些参数，可以方便地将编码器从 B0 扩展到 B5。

MiT 的超参数总结如下：

- $K_i$：阶段 $i$ 中重叠图像块嵌入的图像块大小。
- $S_i$：阶段 $i$ 中重叠图像块嵌入的步长。
- $P_i$：阶段 $i$ 中重叠图像块嵌入的填充大小。
- $C_i$：阶段 $i$ 输出的通道数。
- $L_i$：阶段 $i$ 的编码器层数。
- $R_i$：阶段 $i$ 中高效自注意力的缩减比。
- $N_i$：阶段 $i$ 中高效自注意力的注意力头数。
- $E_i$：阶段 $i$ 中前馈层 [78] 的扩展比。

表 6 展示 MiT 系列的详细信息。为便于讨论，我们将编码器命名为 B0 至 B5：B0 是面向实时应用的最小模型，B5 是面向高性能的最大模型。

**表 6：MiT 系列的详细设置。**设计遵循 ResNet [12] 的原则：第一，随网络加深，空间分辨率降低、通道维度增加；第二，将大部分计算分配给阶段 3。原表按阶段展开如下，所有取值按 PDF 保留。

| 阶段 | 输出空间尺寸 | 重叠图像块嵌入的共有参数 | 模型 | $C_i$ | $R_i$ | $N_i$ | $E_i$ | $L_i$ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | $H/4\times W/4$ | $K_1=7,S_1=4,P_1=3$ | B0 | 32 | 8 | 1 | 8 | 2 |
| 1 | 同上 | 同上 | B1 | 64 | 8 | 1 | 8 | 2 |
| 1 | 同上 | 同上 | B2 | 64 | 8 | 1 | 8 | 3 |
| 1 | 同上 | 同上 | B3 | 64 | 8 | 1 | 8 | 3 |
| 1 | 同上 | 同上 | B4 | 64 | 8 | 1 | 8 | 3 |
| 1 | 同上 | 同上 | B5 | 64 | 8 | 1 | 4 | 3 |
| 2 | $H/8\times W/8$ | $K_2=3,S_2=2,P_2=1$ | B0 | 64 | 4 | 2 | 8 | 2 |
| 2 | 同上 | 同上 | B1 | 128 | 4 | 2 | 8 | 2 |
| 2 | 同上 | 同上 | B2 | 128 | 4 | 2 | 8 | 3 |
| 2 | 同上 | 同上 | B3 | 128 | 4 | 2 | 8 | 3 |
| 2 | 同上 | 同上 | B4 | 128 | 4 | 2 | 8 | 8 |
| 2 | 同上 | 同上 | B5 | 128 | 4 | 2 | 4 | 6 |
| 3 | $H/16\times W/16$ | $K_3=3,S_3=2,P_3=1$ | B0 | 160 | 2 | 5 | 4 | 2 |
| 3 | 同上 | 同上 | B1 | 320 | 2 | 5 | 4 | 2 |
| 3 | 同上 | 同上 | B2 | 320 | 2 | 5 | 4 | 6 |
| 3 | 同上 | 同上 | B3 | 320 | 2 | 5 | 4 | 18 |
| 3 | 同上 | 同上 | B4 | 320 | 2 | 5 | 4 | 27 |
| 3 | 同上 | 同上 | B5 | 320 | 2 | 5 | 4 | 40 |
| 4 | $H/32\times W/32$ | $K_4=3,S_4=2,P_4=1$ | B0 | 256 | 1 | 8 | 4 | 2 |
| 4 | 同上 | 同上 | B1 | 512 | 1 | 8 | 4 | 2 |
| 4 | 同上 | 同上 | B2 | 512 | 1 | 8 | 4 | 3 |
| 4 | 同上 | 同上 | B3 | 512 | 1 | 8 | 4 | 3 |
| 4 | 同上 | 同上 | B4 | 512 | 1 | 8 | 4 | 3 |
| 4 | 同上 | 同上 | B5 | 512 | 1 | 8 | 4 | 3 |

> 译注：正文的序列长度缩减比使用 $[64,16,4,1]$，表 6 的 $R_i$ 列为 $[8,4,2,1]$；译文保留原文两处数值及符号，不将二者混抄。

**表 7：混合 Transformer 编码器。**原表的 Top 1 为分类 Top-1 准确率，参数量单位为百万。

| 方法 | GFLOPs | 参数量（M） | Top-1 |
| --- | ---: | ---: | ---: |
| MiT-B0 | 0.6 | 3.7 | 70.5 |
| MiT-B1 | 2.1 | 14.0 | 78.7 |
| MiT-B2 | 4.0 | 25.4 | 81.6 |
| MiT-B3 | 6.9 | 45.2 | 83.1 |
| MiT-B4 | 10.1 | 62.6 | 83.6 |
| MiT-B5 | 11.8 | 82.0 | 83.8 |

## 附录 B 更多掩码预测定性结果

图 5 给出 Cityscapes、ADE20K 和 COCO-Stuff 上更多定性结果，并与 SETR、DeepLabV3+ 比较。

相比 SETR，SegFormer 在物体边界附近的掩码细节明显更精细，因为我们的 Transformer 编码器能提取比 SETR 分辨率高得多的特征，保留更多精细纹理信息。相比 DeepLabV3+，SegFormer 的长距离错误更少，这得益于 Transformer 编码器比卷积网络更大的有效感受野。

**图 5：Cityscapes、ADE20K 和 COCO-Stuff 上的定性结果。**第一行为 Cityscapes，第二行为 ADE20K，第三行为 COCO-Stuff。各列为 SegFormer、SETR、DeepLabV3+，建议放大查看。

## 附录 C 更多有效感受野可视化

图 6 选取了一些有代表性的图像，展示 DeepLabV3+ 与 SegFormer 的有效感受野。除感受野更大外，SegFormer 的有效感受野对图像上下文更加敏感。可以看到，SegFormer 的有效感受野学习到了道路、汽车和建筑的模式，而 DeepLabV3+ 的有效感受野呈现相对固定的模式。这些结果也表明，Transformer 编码器比卷积网络具有更强的特征提取能力。

**图 6：Cityscapes 上的有效感受野。**图中展示两种架构四个阶段及解码头的有效感受野。“Stage-1”—“Stage-4”和“Head”分别表示阶段 1—4 和解码头；各组对照模型为 DeepLabV3+ 与 SegFormer。

## 附录 D Cityscapes-C 上 DeepLabV3+ 与 SegFormer 的更多比较

本节详细展示 SegFormer 与 DeepLabV3+ 的零样本鲁棒性比较。遵循文献 [77]，对于四种噪声分别测试三个严重程度，对于其余十二种退化和扰动分别测试五个严重程度。

如图 7 所示，随着严重程度增加，DeepLabV3+ 的性能明显下降；SegFormer 的性能则相对稳定。此外，在所有退化/扰动类型和全部严重程度下，SegFormer 均明显优于 DeepLabV3+，表现出优异的零样本鲁棒性。

**图 7：Cityscapes-C 上 SegFormer 与 DeepLabV3+ 的零样本鲁棒性比较。**蓝线表示 SegFormer，橙线表示 DeepLabV3+；横轴为退化严重程度，纵轴为 mIoU。遵循文献 [77]，噪声测试三个严重程度，其余退化测试五个严重程度。十六个子图依次为：高斯噪声、散粒噪声、脉冲噪声、斑点噪声；运动模糊、散焦模糊、玻璃模糊、高斯模糊；雪、飞溅、雾、霜；亮度、对比度、饱和度、JPEG 压缩。原图中的“Impluse”“Forst”为源文拼写，中文按其含义翻译。

## 参考文献

以下保留原文作者顺序、年份、文献类型与编号，题名译为中文；作者姓名和会议/期刊缩写保留，便于检索。“原文引用页”是 PDF 中条目末尾的回指页码，不是文章出版页码。

[1] Jonathan Long、Evan Shelhamer、Trevor Darrell。《用于语义分割的全卷积网络》。CVPR，2015。原文引用页：1、2、7。

[2] Liang-Chieh Chen、George Papandreou、Iasonas Kokkinos、Kevin Murphy、Alan L. Yuille。《结合深度卷积网络与全连接条件随机场的语义图像分割》。ICLR，2015。原文引用页：1、2。

[3] Hang Zhang、Chongruo Wu、Zhongyue Zhang、Yi Zhu、Haibin Lin、Zhi Zhang、Yue Sun、Tong He、Jonas Mueller、R. Manmatha 等。《ResNeSt：分裂注意力网络》。arXiv，2020。原文引用页：1。

[4] Liang-Chieh Chen、George Papandreou、Iasonas Kokkinos、Kevin Murphy、Alan L. Yuille。《DeepLab：结合深度卷积网络、空洞卷积与全连接条件随机场的语义图像分割》。TPAMI，2017。原文引用页：2。

[5] Fisher Yu、Vladlen Koltun。《通过空洞卷积进行多尺度上下文聚合》。ICLR，2016。原文引用页：2、5。

[6] Alexey Dosovitskiy、Lucas Beyer、Alexander Kolesnikov、Dirk Weissenborn、Xiaohua Zhai、Thomas Unterthiner、Mostafa Dehghani、Matthias Minderer、Georg Heigold、Sylvain Gelly 等。《一幅图像相当于 16×16 个词：用于大规模图像识别的 Transformer》。arXiv，2020。原文引用页：2、3。

[7] Sixiao Zheng、Jiachen Lu、Hengshuang Zhao、Xiatian Zhu、Zekun Luo、Yabiao Wang、Yanwei Fu、Jianfeng Feng、Tao Xiang、Philip H. S. Torr 等。《借助 Transformer，从序列到序列的视角重新思考语义分割》。CVPR，2021。原文引用页：2、3、5、7、8。

[8] Wenhai Wang、Enze Xie、Xiang Li、Deng-Ping Fan、Kaitao Song、Ding Liang、Tong Lu、Ping Luo、Ling Shao。《金字塔视觉 Transformer：无需卷积的通用密集预测主干》。arXiv，2021。原文引用页：2、3、4。

[9] Ze Liu、Yutong Lin、Yue Cao、Han Hu、Yixuan Wei、Zheng Zhang、Stephen Lin、Baining Guo。《Swin Transformer：采用移位窗口的层级式视觉 Transformer》。arXiv，2021。原文引用页：2、3、6。

[10] Xiangxiang Chu、Zhi Tian、Yuqing Wang、Bo Zhang、Haibing Ren、Xiaolin Wei、Huaxia Xia、Chunhua Shen。《Twins：重新审视视觉 Transformer 中的空间注意力设计》。arXiv，2021。原文引用页：2、3。

[11] Hengshuang Zhao、Xiaojuan Qi、Xiaoyong Shen、Jianping Shi、Jiaya Jia。《ICNet：面向高分辨率图像的实时语义分割》。ECCV，2018。原文引用页：2、7。

[12] Kaiming He、Xiangyu Zhang、Shaoqing Ren、Jian Sun。《用于图像识别的深度残差学习》。CVPR，2016。原文引用页：2、11。

[13] Alex Krizhevsky、Ilya Sutskever、Geoffrey E. Hinton。《使用深度卷积神经网络进行 ImageNet 分类》。NeurIPS，2012。

[14] Karen Simonyan、Andrew Zisserman。《用于大规模图像识别的极深卷积网络》。arXiv，2014。

[15] Xiang Li、Wenhai Wang、Xiaolin Hu、Jian Yang。《选择性卷积核网络》。CVPR，2019。

[16] Wenhai Wang、Xiang Li、Tong Lu、Jian Yang。《混合连接网络》。IJCAI，2018。原文引用页：2。

[17] Hengshuang Zhao、Jianping Shi、Xiaojuan Qi、Xiaogang Wang、Jiaya Jia。《金字塔场景解析网络》。CVPR，2017。原文引用页：2、7、8。

[18] Maoke Yang、Kun Yu、Chi Zhang、Zhiwei Li、Kuiyuan Yang。《用于街景语义分割的 DenseASPP》。CVPR，2018。原文引用页：5。

[19] Chao Peng、Xiangyu Zhang、Gang Yu、Guiming Luo、Jian Sun。《大卷积核至关重要：通过全局卷积网络改进语义分割》。CVPR，2017。原文引用页：2、5。

[20] Liang-Chieh Chen、Yukun Zhu、George Papandreou、Florian Schroff、Hartwig Adam。《采用空洞可分离卷积的语义图像分割编码器—解码器》。ECCV，2018。原文引用页：2、5、7、8。

[21] Yuhui Yuan、Jingdong Wang。《OCNet：用于场景解析的对象上下文网络》。arXiv，2018。原文引用页：2、8。

[22] Changqian Yu、Jingbo Wang、Changxin Gao、Gang Yu、Chunhua Shen、Nong Sang。《用于场景分割的上下文先验》。CVPR，2020。

[23] Yuhui Yuan、Xilin Chen、Jingdong Wang。《用于语义分割的对象上下文表征》。arXiv，2019。原文引用页：7、8。

[24] Hang Zhang、Kristin Dana、Jianping Shi、Zhongyue Zhang、Xiaogang Wang、Ambrish Tyagi、Amit Agrawal。《用于语义分割的上下文编码》。CVPR，2018。原文引用页：7。

[25] Yizhou Zhou、Xiaoyan Sun、Zheng-Jun Zha、Wenjun Zeng。《上下文增强的语义分割》。CVPR，2019。

[26] Guosheng Lin、Anton Milan、Chunhua Shen、Ian Reid。《RefineNet：用于高分辨率语义分割的多路径细化网络》。CVPR，2017。

[27] Rudra P. K. Poudel、Ujwal Bonde、Stephan Liwicki、Christopher Zach。《ContextNet：探索实时语义分割中的上下文与细节》。arXiv，2018。

[28] Tianyi Wu、Sheng Tang、Rui Zhang、Yongdong Zhang。《CGNet：一种轻量的上下文引导语义分割网络》。arXiv，2018。

[29] Junjun He、Zhongying Deng、Lei Zhou、Yali Wang、Yu Qiao。《用于语义分割的自适应金字塔上下文网络》。CVPR，2019。原文引用页：3。

[30] Henghui Ding、Xudong Jiang、Ai Qun Liu、Nadia Magnenat Thalmann、Gang Wang。《用于场景分割的边界感知特征传播》。ICCV，2019。原文引用页：3。

[31] Gedas Bertasius、Jianbo Shi、Lorenzo Torresani。《采用边界神经场的语义分割》。CVPR，2016。

[32] Xiangtai Li、Xia Li、Li Zhang、Guangliang Cheng、Jianping Shi、Zhouchen Lin、Shaohua Tan、Yunhai Tong。《通过解耦主体与边缘监督改进语义分割》。arXiv，2020。

[33] Yuhui Yuan、Jingyi Xie、Xilin Chen、Jingdong Wang。《SegFix：与模型无关的分割边界细化》。ECCV，2020。

[34] Mingmin Zhen、Jinglu Wang、Lei Zhou、Shiwei Li、Tianwei Shen、Jiaxiang Shang、Tian Fang、Long Quan。《利用迭代金字塔上下文联合进行语义分割与边界检测》。CVPR，2020。

[35] Towaki Takikawa、David Acuna、Varun Jampani、Sanja Fidler。《Gated-SCNN：用于语义分割的门控形状卷积神经网络》。ICCV，2019。原文引用页：7。

[36] Changqian Yu、Jingbo Wang、Chao Peng、Changxin Gao、Gang Yu、Nong Sang。《学习用于语义分割的判别性特征网络》。CVPR，2018。

[37] Liang-Chieh Chen、Jonathan T. Barron、George Papandreou、Kevin Murphy、Alan L. Yuille。《利用 CNN 的任务专用边缘检测与判别式训练的域变换进行语义图像分割》。CVPR，2016。原文引用页：3。

[38] Jun Fu、Jing Liu、Haijie Tian、Yong Li、Yongjun Bao、Zhiwei Fang、Hanqing Lu。《用于场景分割的双注意力网络》。CVPR，2019。原文引用页：3。

[39] Xiaolong Wang、Ross Girshick、Abhinav Gupta、Kaiming He。《非局部神经网络》。CVPR，2018。

[40] Zilong Zhong、Zhong Qiu Lin、Rene Bidart、Xiaodan Hu、Ibrahim Ben Daya、Zhifeng Li、Wei-Shi Zheng、Jonathan Li、Alexander Wong。《用于语义分割的压缩与注意力网络》。CVPR，2020。

[41] Zilong Huang、Xinggang Wang、Lichao Huang、Chang Huang、Yunchao Wei、Wenyu Liu。《CCNet：用于语义分割的纵横交叉注意力》。ICCV，2019。原文引用页：7、8。

[42] Hanchao Li、Pengfei Xiong、Jie An、Lingxue Wang。《用于语义分割的金字塔注意力网络》。arXiv，2018。

[43] Hengshuang Zhao、Yi Zhang、Shu Liu、Jianping Shi、Chen Change Loy、Dahua Lin、Jiaya Jia。《PSANet：用于场景解析的逐点空间注意力网络》。ECCV，2018。原文引用页：8。

[44] Xia Li、Zhisheng Zhong、Jianlong Wu、Yibo Yang、Zhouchen Lin、Hong Liu。《用于语义分割的期望最大化注意力网络》。ICCV，2019。

[45] Yue Cao、Jiarui Xu、Stephen Lin、Fangyun Wei、Han Hu。《GCNet：非局部网络与压缩—激励网络的结合及拓展》。ICCVW，2019。

[46] Enze Xie、Wenjia Wang、Wenhai Wang、Peize Sun、Hang Xu、Ding Liang、Ping Luo。《使用 Transformer 分割自然场景中的透明物体》。IJCAI，2021。原文引用页：3。

[47] Albert Shaw、Daniel Hunter、Forrest Landola、Sammy Sidhu。《SqueezeNAS：用于更快语义分割的快速神经架构搜索》。ICCVW，2019。原文引用页：3。

[48] Wuyang Chen、Xinyu Gong、Xianming Liu、Qian Zhang、Yuan Li、Zhangyang Wang。《FasterSeg：搜索更快的实时语义分割模型》。arXiv，2019。

[49] Yanwei Li、Lin Song、Yukang Chen、Zeming Li、Xiangyu Zhang、Xingang Wang、Jian Sun。《学习用于语义分割的动态路由》。CVPR，2020。

[50] Chenxi Liu、Liang-Chieh Chen、Florian Schroff、Hartwig Adam、Wei Hua、Alan L. Yuille、Li Fei-Fei。《Auto-DeepLab：用于语义图像分割的层级神经架构搜索》。CPVR，2019。原文引用页：7。〔译注：CPVR 为原文拼写。〕

[51] Vladimir Nekrasov、Hao Chen、Chunhua Shen、Ian Reid。《通过辅助单元快速搜索紧凑的语义分割神经架构》。CVPR，2019。原文引用页：3。

[52] Nicolas Carion、Francisco Massa、Gabriel Synnaeve、Nicolas Usunier、Alexander Kirillov、Sergey Zagoruyko。《使用 Transformer 的端到端目标检测》。ECCV，2020。原文引用页：3。

[53] Li Yuan、Yunpeng Chen、Tao Wang、Weihao Yu、Yujun Shi、Zihang Jiang、Francis E. H. Tay、Jiashi Feng、Shuicheng Yan。《Tokens-to-Token ViT：在 ImageNet 上从头训练视觉 Transformer》。arXiv，2021。原文引用页：3。

[54] Xiangxiang Chu、Zhi Tian、Bo Zhang、Xinlong Wang、Xiaolin Wei、Huaxia Xia、Chunhua Shen。《用于视觉 Transformer 的条件位置编码》。arXiv，2021。原文引用页：3、4。

[55] Kai Han、An Xiao、Enhua Wu、Jianyuan Guo、Chunjing Xu、Yunhe Wang。《Transformer 中的 Transformer》。arXiv，2021。原文引用页：3。

[56] Chun-Fu Chen、Quanfu Fan、Rameswar Panda。《CrossViT：用于图像分类的交叉注意力多尺度视觉 Transformer》。arXiv，2021。原文引用页：3。

[57] Yawei Li、Kai Zhang、Jiezhang Cao、Radu Timofte、Luc Van Gool。《LocalViT：为视觉 Transformer 引入局部性》。arXiv，2021。原文引用页：3。

[58] Haiping Wu、Bin Xiao、Noel Codella、Mengchen Liu、Xiyang Dai、Lu Yuan、Lei Zhang。《CvT：为视觉 Transformer 引入卷积》。arXiv，2021。原文引用页：3。

[59] Weijian Xu、Yifan Xu、Tyler Chang、Zhuowen Tu。《协同尺度卷积注意力图像 Transformer》。arXiv，2021。原文引用页：3。

[60] Ben Graham、Alaaeldin El-Nouby、Hugo Touvron、Pierre Stock、Armand Joulin、Hervé Jégou、Matthijs Douze。《LeViT：披着卷积网络外衣、实现更快推理的视觉 Transformer》。arXiv，2021。原文引用页：3。

[61] Peize Sun、Yi Jiang、Rufeng Zhang、Enze Xie、Jinkun Cao、Xinting Hu、Tao Kong、Zehuan Yuan、Changhu Wang、Ping Luo。《TransTrack：使用 Transformer 的多目标跟踪》。arXiv，2020。原文引用页：3。

[62] Tim Meinhardt、Alexander Kirillov、Laura Leal-Taixe、Christoph Feichtenhofer。《TrackFormer：使用 Transformer 的多目标跟踪》。arXiv，2021。原文引用页：3。

[63] Hanting Chen、Yunhe Wang、Tianyu Guo、Chang Xu、Yiping Deng、Zhenhua Liu、Siwei Ma、Chunjing Xu、Chao Xu、Wen Gao。《预训练图像处理 Transformer》。arXiv，2020。原文引用页：3。

[64] Shuting He、Hao Luo、Pichao Wang、Fan Wang、Hao Li、Wei Jiang。《TransReID：基于 Transformer 的目标重识别》。arXiv，2021。原文引用页：3。

[65] Manoj Kumar、Dirk Weissenborn、Nal Kalchbrenner。《图像着色 Transformer》。arXiv，2021。原文引用页：3。

[66] Alaaeldin El-Nouby、Natalia Neverova、Ivan Laptev、Hervé Jégou。《训练用于图像检索的视觉 Transformer》。arXiv，2021。原文引用页：3。

[67] Alec Radford、Jong Wook Kim、Chris Hallacy、Aditya Ramesh、Gabriel Goh、Sandhini Agarwal、Girish Sastry、Amanda Askell、Pamela Mishkin、Jack Clark 等。《通过自然语言监督学习可迁移的视觉模型》。arXiv，2021。原文引用页：3。

[68] Ronghang Hu、Amanpreet Singh。《只需 Transformer：使用统一 Transformer 进行多模态多任务学习》。arXiv，2021。原文引用页：3。

[69] Md Amirul Islam、Sen Jia、Neil D. B. Bruce。《卷积神经网络编码了多少位置信息？》。arXiv，2020。原文引用页：4。

[70] Wenjie Luo、Yujia Li、Raquel Urtasun、Richard Zemel。《理解深度卷积神经网络中的有效感受野》。arXiv，2017。原文引用页：5。

[71] Marius Cordts、Mohamed Omran、Sebastian Ramos、Timo Rehfeld、Markus Enzweiler、Rodrigo Benenson、Uwe Franke、Stefan Roth、Bernt Schiele。《用于语义城市场景理解的 Cityscapes 数据集》。CVPR，2016。原文引用页：6、8。

[72] Bolei Zhou、Hang Zhao、Xavier Puig、Sanja Fidler、Adela Barriuso、Antonio Torralba。《通过 ADE20K 数据集进行场景解析》。CVPR，2017。原文引用页：6、8。

[73] Holger Caesar、Jasper Uijlings、Vittorio Ferrari。《COCO-Stuff：上下文中的物体类与背景类》。CVPR，2018。原文引用页：6、8。

[74] Huiyu Wang、Yukun Zhu、Bradley Green、Hartwig Adam、Alan L. Yuille、Liang-Chieh Chen。《Axial-DeepLab：用于全景分割的独立轴向注意力》。ECCV，2020。原文引用页：7、8。

[75] Yanwei Li、Lin Song、Yukang Chen、Zeming Li、Xiangyu Zhang、Xingang Wang、Jian Sun。《学习用于语义分割的动态路由》。CVPR，2020。原文引用页：7。

[76] Gerhard Neuhold、Tobias Ollmann、Samuel Rota Bulò、Peter Kontschieder。《用于街景语义理解的 Mapillary Vistas 数据集》。ICCV，2017。原文引用页：8。

[77] Christoph Kamann、Carsten Rother。《语义分割模型鲁棒性的基准评估》。CVPR，2020。原文引用页：8、9、10、14。

[78] Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Łukasz Kaiser、Illia Polosukhin。《只需注意力机制》。NeurIPS，2017。原文引用页：10。

## 术语对照

| 原文术语 | 本文译法 |
| --- | --- |
| semantic segmentation | 语义分割 |
| dense prediction | 密集预测 |
| backbone | 主干 |
| hierarchical feature representation | 层级特征表示 |
| overlapping patch embedding / merging | 重叠图像块嵌入 / 合并 |
| positional encoding / embedding | 位置编码 / 位置嵌入 |
| efficient self-attention | 高效自注意力 |
| feed-forward network | 前馈网络 |
| depth-wise convolution | 逐通道卷积 |
| effective receptive field | 有效感受野 |
| mean Intersection over Union | 平均交并比 |
| corruption / perturbation | 退化 / 扰动 |
| ablation study | 消融研究 |
| zero-shot robustness | 零样本鲁棒性 |
