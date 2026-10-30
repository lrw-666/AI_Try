# 一幅图像相当于16×16个词：用于大规模图像识别的Transformer

原题：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

原文：[本地PDF](<./An Image is Worth 16x16 Words- Transformers for Image Recognition at Scale.pdf>)。版本：arXiv:2010.11929v2 [cs.CV]，2021年6月3日；发表于ICLR 2021。翻译范围：原PDF全部22页，包括摘要、正文、致谢、参考文献、附录A–D、公式(1)–(8)、图1–14、表1–9及脚注1–3。正文跨页续段已合并，原页眉、页码和排版断词已移除。

标识约定：作者姓名、模型和数据集名称、机构与会议缩写、引用的作者—年份标识及URL保留，以便与原文检索对照。首次出现的主要术语附英文。本文件以文字译文呈现；原始图形请对照上述PDF，图注与关键图内英文标签在对应位置译出；表格和公式在本文件中重排。图表按相应正文位置安放，不改变编号或数据。下文中单独标出的“译注”仅说明原文排印问题，不属于作者论述。

作者：Alexey Dosovitskiy\*†、Lucas Beyer\*、Alexander Kolesnikov\*、Dirk Weissenborn\*、Xiaohua Zhai\*、Thomas Unterthiner、Mostafa Dehghani、Matthias Minderer、Georg Heigold、Sylvain Gelly、Jakob Uszkoreit、Neil Houlsby\*†。

\* 技术贡献相同；† 共同指导。

Google Research，Brain Team。电子邮件：`{adosovitskiy, neilhoulsby}@google.com`。

## 摘要

尽管Transformer架构已经成为自然语言处理任务事实上的标准，但它在计算机视觉中的应用仍然有限。在视觉领域，注意力要么与卷积网络结合使用，要么在保持卷积网络整体结构不变的情况下替换其中某些组件。我们表明，没有必要如此依赖CNN：将纯Transformer直接应用于图像块（image patch）序列，就能在图像分类任务上取得很好的表现。在大量数据上进行预训练（pre-training），再迁移到多个中小规模图像识别基准（ImageNet、CIFAR-100、VTAB等）时，视觉Transformer（Vision Transformer，ViT）相较于最先进的卷积网络取得了优异结果，而且训练所需的计算资源大幅减少。[^1]

## 1 引言

基于自注意力的架构，尤其是Transformer（Vaswani et al., 2017），已经成为自然语言处理（NLP）的首选模型。主流方法是在大型文本语料库上预训练，再在较小的任务专用数据集上微调（fine-tuning）（Devlin et al., 2019）。得益于Transformer的计算效率与可扩展性，人们已能训练规模空前、参数超过1000亿的模型（Brown et al., 2020; Lepikhin et al., 2020）。随着模型和数据集不断增大，性能仍未显现出饱和迹象。

但在计算机视觉中，卷积架构仍然占据主导地位（LeCun et al., 1989; Krizhevsky et al., 2012; He et al., 2016）。受到NLP成功的启发，多项工作尝试将类似CNN的架构与自注意力结合（Wang et al., 2018; Carion et al., 2020），其中一些甚至完全替换了卷积（Ramachandran et al., 2019; Wang et al., 2020a）。后一类模型虽然理论上高效，但由于使用专门设计的注意力模式，尚未能在现代硬件加速器上有效扩大规模。因此，在大规模图像识别中，经典的类ResNet架构仍然是最先进的方法（Mahajan et al., 2018; Xie et al., 2020; Kolesnikov et al., 2020）。

受Transformer在NLP中成功扩展规模的启发，我们尝试以尽可能少的修改，将标准Transformer直接应用于图像。为此，我们把图像切分为图像块，将这些块的线性嵌入（embedding）组成序列，作为Transformer的输入。图像块的处理方式与NLP应用中的词元（token，即词）相同。我们在图像分类任务上以监督方式训练该模型。

在ImageNet这样的中等规模数据集上训练且不使用强正则化时，这些模型的准确率较为一般，比大小相当的ResNet低几个百分点。这种表面上令人失望的结果或许是可以预料的：Transformer缺少CNN固有的一些归纳偏置（inductive bias），例如平移等变性和局部性，所以在数据量不足时不能很好地泛化。

但如果在更大的数据集上训练模型（1400万至3亿幅图像），情况就会改变。我们发现，大规模训练胜过归纳偏置。我们的视觉Transformer（ViT）在足够大的规模上预训练后，迁移到数据点较少的任务时能够取得优异结果。在公开的ImageNet-21k数据集或内部JFT-300M数据集上预训练后，ViT在多个图像识别基准上接近或超过最先进水平。具体而言，最佳模型在ImageNet上达到88.55%的准确率，在ImageNet-ReaL上达到90.72%，在CIFAR-100上达到94.55%，在包含19项任务的VTAB套件上达到77.63%。

[^1]: 微调代码和预训练模型可在[项目代码库](https://github.com/google-research/vision_transformer)获取。

## 2 相关工作

Vaswani et al.（2017）为机器翻译提出了Transformer，此后它已成为许多NLP任务中最先进的方法。大型Transformer模型通常先在大规模语料库上预训练，再针对当前任务微调：BERT（Devlin et al., 2019）使用去噪自监督预训练任务，而GPT系列工作使用语言建模作为预训练任务（Radford et al., 2018; 2019; Brown et al., 2020）。

将自注意力直接应用于图像，会要求每个像素关注其他所有像素。其代价随像素数量呈二次增长，因而无法扩展到实际使用的输入尺寸。因此，以往为将Transformer用于图像处理，已经尝试过若干近似方法。Parmar et al.（2018）仅在各查询像素的局部邻域内应用自注意力，而不是应用于全局。这类局部多头点积自注意力模块可以完全替代卷积（Hu et al., 2019; Ramachandran et al., 2019; Zhao et al., 2020）。另一条研究路线中，稀疏Transformer（Child et al., 2019）使用可扩展的全局自注意力近似，使其可以应用于图像。扩展注意力的另一种方式是在大小不同的块中应用它（Weissenborn et al., 2019），极端情况下只沿单个坐标轴应用（Ho et al., 2019; Wang et al., 2020a）。许多专门设计的注意力架构在计算机视觉任务上展示了令人期待的结果，但要在硬件加速器上高效实现它们，需要复杂的工程工作。

与我们工作最相关的是Cordonnier et al.（2020）的模型，它从输入图像中提取2×2大小的图像块，然后在其上应用完整的自注意力。该模型与ViT非常相似，但我们的工作进一步证明，大规模预训练能使原生Transformer与最先进的CNN相媲美，甚至表现更好。此外，Cordonnier et al.（2020）使用2×2像素的小图像块，使模型只能应用于低分辨率图像，而我们也能处理中等分辨率图像。

将卷积神经网络（CNN）与各种形式的自注意力结合也引起了广泛兴趣，例如对特征图进行增强以用于图像分类（Bello et al., 2019），或利用自注意力进一步处理CNN的输出，用于目标检测（Hu et al., 2018; Carion et al., 2020）、视频处理（Wang et al., 2018; Sun et al., 2019）、图像分类（Wu et al., 2020）、无监督目标发现（Locatello et al., 2020），或者统一的文本—视觉任务（Chen et al., 2020c; Lu et al., 2019; Li et al., 2019）。

最近另一个相关模型是图像GPT（iGPT）（Chen et al., 2020a），它在降低图像分辨率并缩小颜色空间之后，将Transformer应用于图像像素。该模型以无监督方式作为生成模型训练，得到的表示随后可以微调，或通过线性探测（linear probing）评估分类性能，在ImageNet上取得的最高准确率为72%。

越来越多的论文探索在超过标准ImageNet数据集的规模上开展图像识别，我们的工作也是其中之一。使用额外的数据来源，可以在标准基准上取得最先进的结果（Mahajan et al., 2018; Touvron et al., 2019; Xie et al., 2020）。此外，Sun et al.（2017）研究了CNN性能如何随数据集大小扩展，Kolesnikov et al.（2020）和Djolonga et al.（2020）则对CNN从ImageNet-21k、JFT-300M等大规模数据集迁移学习进行了实证探索。我们同样关注后两个数据集，但训练的是Transformer，而不是以往工作中基于ResNet的模型。

## 3 方法

在模型设计上，我们尽可能遵循原始Transformer（Vaswani et al., 2017）。这种刻意保持简单的设置有一个优点：可扩展的NLP Transformer架构及其高效实现几乎可以直接使用。

### 3.1 视觉Transformer（ViT）

**图1：模型概览。** 我们把图像切分为固定大小的图像块，分别进行线性嵌入，加上位置嵌入（position embedding），再将得到的向量序列送入标准Transformer编码器。为了执行分类，我们使用标准做法：在序列中加入一个额外的、可学习的“分类词元”。Transformer编码器示意图受Vaswani et al.（2017）启发。原始图形见PDF第3页。

图中标签：Vision Transformer（ViT）=视觉Transformer；Class=类别（Bird=鸟，Ball=球，Car=汽车）；MLP Head=MLP分类头；Transformer Encoder=Transformer编码器；Patch + Position Embedding=图像块嵌入加位置嵌入；Extra learnable [class] embedding=额外的可学习`[class]`嵌入；Linear Projection of Flattened Patches=展平图像块的线性投影；Multi-Head Attention=多头注意力；Norm=归一化；Embedded Patches=嵌入后的图像块；L×=重复L次。

图1给出了模型概览。标准Transformer接收一维词元嵌入序列作为输入。为处理二维图像，我们把图像$\mathbf{x}\in\mathbb{R}^{H\times W\times C}$重排为展平的二维图像块序列$\mathbf{x}_p\in\mathbb{R}^{N\times(P^2\cdot C)}$，其中$(H,W)$是原始图像的分辨率，$C$是通道数，$(P,P)$是每个图像块的分辨率，$N=HW/P^2$是得到的图像块数量，同时也是Transformer的有效输入序列长度。Transformer在所有层中使用固定的潜在向量维度$D$，因此我们先展平图像块，再通过可训练的线性投影将其映射到$D$维（公式1）。我们将该投影的输出称为图像块嵌入。

与BERT的`[class]`词元类似，我们在图像块嵌入序列的开头添加一个可学习的嵌入（$\mathbf{z}_0^0=\mathbf{x}_{\mathrm{class}}$），它在Transformer编码器输出端的状态（$\mathbf{z}_L^0$）作为图像表示$\mathbf{y}$（公式4）。在预训练和微调期间，都在$\mathbf{z}_L^0$之后连接一个分类头。预训练时，分类头由一个具有单隐藏层的MLP实现；微调时，则由单个线性层实现。

我们将位置嵌入加到图像块嵌入上，以保留位置信息。我们使用标准的可学习一维位置嵌入，因为没有观察到更复杂、具备二维感知能力的位置嵌入带来显著性能提升（附录D.4）。得到的嵌入向量序列作为编码器的输入。

Transformer编码器（Vaswani et al., 2017）由多头自注意力（multiheaded self-attention，MSA；见附录A）层与MLP模块交替构成（公式2、3）。每个模块前应用层归一化（Layernorm，LN），每个模块后使用残差连接（Wang et al., 2019; Baevski & Auli, 2019）。MLP包含两层，并使用GELU非线性激活。

$$
\mathbf{z}_0=[\mathbf{x}_{\mathrm{class}};\mathbf{x}_p^1\mathbf{E};\mathbf{x}_p^2\mathbf{E};\cdots;\mathbf{x}_p^N\mathbf{E}]+\mathbf{E}_{\mathrm{pos}},\qquad
\mathbf{E}\in\mathbb{R}^{(P^2\cdot C)\times D},\quad
\mathbf{E}_{\mathrm{pos}}\in\mathbb{R}^{(N+1)\times D}.
\tag{1}
$$

$$
\mathbf{z}'_\ell=\operatorname{MSA}(\operatorname{LN}(\mathbf{z}_{\ell-1}))+\mathbf{z}_{\ell-1},\qquad\ell=1\ldots L.
\tag{2}
$$

$$
\mathbf{z}_\ell=\operatorname{MLP}(\operatorname{LN}(\mathbf{z}'_\ell))+\mathbf{z}'_\ell,\qquad\ell=1\ldots L.
\tag{3}
$$

$$
\mathbf{y}=\operatorname{LN}(\mathbf{z}_L^0).
\tag{4}
$$

**归纳偏置。** 我们注意到，视觉Transformer所具有的图像专用归纳偏置比CNN少得多。在CNN中，局部性、二维邻域结构和平移等变性被内置于整个模型的每一层。在ViT中，只有MLP层具有局部性和平移等变性，自注意力层则是全局的。模型极少使用二维邻域结构：一处是在模型开始时把图像切成图像块，另一处是在微调时针对不同分辨率的图像调整位置嵌入（如下文所述）。除此之外，位置嵌入在初始化时不携带任何关于图像块二维位置的信息，图像块之间的所有空间关系都必须从头学习。

**混合架构。** 除原始图像块之外，输入序列也可以由CNN的特征图构成（LeCun et al., 1989）。在这种混合模型中，将图像块嵌入投影$\mathbf{E}$（公式1）应用于从CNN特征图中提取的块。一个特例是图像块的空间大小为1×1，这意味着只需展平特征图的空间维度，再投影到Transformer的维度，就可得到输入序列。分类输入嵌入和位置嵌入按照前述方法添加。

### 3.2 微调与更高分辨率

我们通常先在大型数据集上预训练ViT，再在规模较小的下游任务上微调。为此，我们移除预训练的预测头，接上一个零初始化的$D\times K$前馈层，其中$K$是下游类别数。使用比预训练时更高的分辨率进行微调通常有益（Touvron et al., 2019; Kolesnikov et al., 2020）。输入更高分辨率的图像时，我们保持图像块大小不变，从而增大有效序列长度。视觉Transformer可以处理任意序列长度（受内存限制），但预训练的位置嵌入可能不再有意义。因此，我们根据预训练位置嵌入在原始图像中的位置，对它们进行二维插值。注意，这种分辨率调整和图像块提取，是将有关图像二维结构的归纳偏置人工注入视觉Transformer的仅有两处环节。

## 4 实验

我们评估了ResNet、视觉Transformer（ViT）以及混合模型的表示学习能力。为了解每种模型的数据需求，我们在不同大小的数据集上预训练，并在多项基准任务上评估。若考虑模型预训练的计算成本，ViT表现非常有利，以更低的预训练成本，在大多数识别基准上达到最先进水平。最后，我们还开展了一项小规模自监督实验，表明自监督ViT未来值得期待。

### 4.1 实验设置

**数据集。** 为探索模型的可扩展性，我们使用了具有1000个类别、130万幅图像的ILSVRC-2012 ImageNet数据集（下文简称ImageNet），它的超集ImageNet-21k（具有21000个类别、1400万幅图像）（Deng et al., 2009），以及JFT（Sun et al., 2017）（具有18000个类别、3.03亿幅高分辨率图像）。按照Kolesnikov et al.（2020）的方法，我们针对下游任务的测试集对预训练数据集进行去重。我们将这些数据集上训练的模型迁移到多项基准任务：分别使用原始验证标签和清理后的ReaL标签（Beyer et al., 2020）的ImageNet、CIFAR-10/100（Krizhevsky, 2009）、Oxford-IIIT Pets（Parkhi et al., 2012）以及Oxford Flowers-102（Nilsback & Zisserman, 2008）。这些数据集的预处理遵循Kolesnikov et al.（2020）。

我们还在包含19项任务的VTAB分类套件上评估（Zhai et al., 2019b）。VTAB对多样化任务的小样本迁移进行评估，每项任务使用1000个训练样本。任务分为三组：自然类（Natural），即类似上述Pets、CIFAR等任务；专业类（Specialized），即医学和卫星图像；结构类（Structured），即定位等需要几何理解的任务。

**模型变体。** 我们基于BERT（Devlin et al., 2019）采用的配置设置ViT，汇总见表1。“Base”和“Large”模型直接沿用BERT，我们另外加入了更大的“Huge”模型。下文用简写表示模型大小和输入图像块大小，例如ViT-L/16表示输入图像块为16×16的“Large”变体。注意，Transformer序列长度与图像块大小的平方成反比，因此图像块越小，模型计算成本越高。

**表1：视觉Transformer各模型变体的详细配置。**

| 模型 | 层数 | 隐藏维度$D$ | MLP维度 | 注意力头数 | 参数量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

对于基线CNN，我们使用ResNet（He et al., 2016），但将批归一化层（Ioffe & Szegedy, 2015）替换为组归一化（Wu & He, 2018），并使用标准化卷积（Qiao et al., 2019）。这些修改有助于迁移（Kolesnikov et al., 2020），我们将修改后的模型称为“ResNet（BiT）”。对于混合模型，我们将中间特征图输入ViT，图像块大小为一个“像素”。为试验不同序列长度，我们采取以下两种方式之一：（i）使用常规ResNet50第4阶段的输出；（ii）移除第4阶段，把相同数量的层放入第3阶段（保持总层数不变），再使用扩展后第3阶段的输出。方式（ii）的序列长度为原来的4倍，ViT模型的计算成本也更高。

**训练与微调。** 所有模型（包括ResNet）均采用Adam（Kingma & Ba, 2015）训练，其中$\beta_1=0.9$、$\beta_2=0.999$，批大小4096，并使用较高的权重衰减0.1；我们发现，这有利于所有模型的迁移（附录D.1表明，与常见做法不同，在我们的设置下，Adam训练ResNet略优于SGD）。我们使用线性学习率预热和衰减，细节见附录B.1。微调时，所有模型均使用带动量的SGD，批大小为512，见附录B.1.1。对于表2中的ImageNet结果，我们以更高分辨率微调：ViT-L/16使用512，ViT-H/14使用518；同时还采用系数为0.9999的Polyak & Juditsky（1992）平均（Ramachandran et al., 2019; Wang et al., 2020b）。

**指标。** 我们以少样本准确率或微调准确率报告下游数据集上的结果。微调准确率反映模型在对应数据集微调后的性能。少样本准确率通过求解正则化最小二乘回归得到：把训练图像子集的冻结表示映射到$\{-1,1\}^K$目标向量。这种形式使我们能够以闭式形式求得精确解。虽然我们主要关注微调性能，但在微调成本过高时，有时也使用线性少样本准确率进行快速的训练中即时评估。
### 4.2 与最先进方法比较

我们首先将最大的两个模型ViT-H/14和ViT-L/16与文献中最先进的CNN比较。第一个比较对象是Big Transfer（BiT）（Kolesnikov et al., 2020），它使用大型ResNet进行监督迁移学习。第二个是Noisy Student（Xie et al., 2020），即大型EfficientNet，采用半监督学习，在ImageNet以及去除标签的JFT-300M上训练。目前，Noisy Student在ImageNet上处于最先进水平，BiT-L则在这里报告的其他数据集上处于最先进水平。所有模型均在TPUv3硬件上训练，我们报告各模型预训练所需的TPUv3核心天数，也就是用于训练的TPUv3核心数（每芯片2核）乘以以天为单位的训练时间。

表2给出了结果。在JFT-300M上预训练的较小模型ViT-L/16，在所有任务上均超过了在同一数据集预训练的BiT-L，而且训练所需计算资源显著减少。更大的ViT-H/14进一步提升了性能，尤其是在ImageNet、CIFAR-100和VTAB套件等更具挑战性的数据集上。有趣的是，该模型的预训练计算量仍然远低于以往最先进的方法。但我们也注意到，预训练效率不仅受架构选择影响，也可能受到训练日程、优化器、权重衰减等其他参数影响。第4.4节给出了不同架构性能与计算量关系的受控研究。最后，在公开ImageNet-21k上预训练的ViT-L/16，在大多数数据集上也表现良好，而且预训练所需资源更少：使用一个标准的8核云TPUv3，大约30天就能完成训练。

**表2：在常用图像分类基准上与最先进方法比较。** 报告的准确率为三次微调运行的均值与标准差。在JFT-300M上预训练的视觉Transformer，在所有数据集上都优于基于ResNet的基线，而且预训练计算资源大幅减少。在更小的公开ImageNet-21k数据集上预训练的ViT也有良好表现。\* Touvron et al.（2020）报告了略有提高的88.5%结果。

| 数据集／成本 | 我们的方法-JFT（ViT-H/14） | 我们的方法-JFT（ViT-L/16） | 我们的方法-I21k（ViT-L/16） | BiT-L（ResNet152x4） | Noisy Student（EfficientNet-L2） |
| --- | ---: | ---: | ---: | ---: | ---: |
| ImageNet | 88.55 ± 0.04 | 87.76 ± 0.03 | 85.30 ± 0.02 | 87.54 ± 0.02 | 88.4/88.5\* |
| ImageNet ReaL | 90.72 ± 0.05 | 90.54 ± 0.03 | 88.62 ± 0.05 | 90.54 | 90.55 |
| CIFAR-10 | 99.50 ± 0.06 | 99.42 ± 0.03 | 99.15 ± 0.03 | 99.37 ± 0.06 | − |
| CIFAR-100 | 94.55 ± 0.04 | 93.90 ± 0.05 | 93.25 ± 0.05 | 93.51 ± 0.08 | − |
| Oxford-IIIT Pets | 97.56 ± 0.03 | 97.32 ± 0.11 | 94.67 ± 0.15 | 96.62 ± 0.23 | − |
| Oxford Flowers-102 | 99.68 ± 0.02 | 99.74 ± 0.00 | 99.61 ± 0.02 | 99.63 ± 0.03 | − |
| VTAB（19项任务） | 77.63 ± 0.23 | 76.28 ± 0.46 | 72.72 ± 0.21 | 76.29 ± 1.70 | − |
| TPUv3核心天数 | 2.5k | 0.68k | 0.23k | 9.9k | 12.3k |

图2将VTAB任务拆分为各个任务组，并与该基准以前的最先进方法比较：BiT；VIVI，即在ImageNet和Youtube上共同训练的ResNet（Tschannen et al., 2020）；以及S4L，即ImageNet上的监督学习加半监督学习（Zhai et al., 2019a）。在自然类和结构类任务上，ViT-H/14超过了BiT-R152x4和其他方法。在专业类任务上，前两个模型表现相近。

**图2：按自然类、专业类和结构类任务组细分的VTAB性能。** 原始图形见PDF第6页。纵轴Accuracy [%]=准确率[%]；四个分图依次是VTAB（19项任务）、自然类（7项任务）、专业类（4项任务）、结构类（8项任务）。图例保留为ViT-H/14、BiT-L（R152x4）、VIVI-Ex-100%（R50x3）、S4L（R50x1）。

### 4.3 预训练数据需求

视觉Transformer在大型JFT-300M数据集预训练时表现良好。它的视觉归纳偏置比ResNet少，那么数据集大小究竟有多关键？我们进行了两组实验。

首先，我们在规模递增的数据集上预训练ViT模型：ImageNet、ImageNet-21k和JFT-300M。为提高较小数据集上的性能，我们优化了三个基本正则化参数：权重衰减、丢弃法（dropout）和标签平滑。图3显示了在ImageNet上微调后的结果（其他数据集上的结果见表5）。[^2] 在最小的ImageNet数据集上预训练时，尽管使用了适度正则化，ViT-Large模型的表现仍不如ViT-Base模型。在ImageNet-21k上预训练时，两者表现相近。只有在JFT-300M上，才充分体现出更大模型的优势。图3还显示了不同大小的BiT模型所覆盖的性能区间。BiT CNN在ImageNet上优于ViT，但在更大的数据集上，ViT反超。

**图3：迁移到ImageNet。** 在小数据集上预训练时，大型ViT模型表现比BiT ResNet差（阴影区域）；在更大数据集上预训练时，它们表现出色。同样，随着数据集增大，较大的ViT变体会超过较小的变体。原始图形见PDF第7页。纵轴ImageNet Top1 Accuracy [%]=ImageNet Top-1准确率[%]；横轴Pre-training dataset=预训练数据集（ImageNet、ImageNet-21k、JFT-300M）。图例中的模型标识BiT、ViT-B/32、ViT-B/16、ViT-L/32、ViT-L/16、ViT-H/14保留。

其次，我们在JFT-300M的随机子集（900万、3000万和9000万幅图像）以及完整数据集上训练模型。我们不针对较小子集增加正则化，所有设置均使用相同超参数。这样，我们评估的是模型的内在属性，而不是正则化的效果。不过，我们确实采用早停，并报告训练过程中达到的最佳验证准确率。为节省计算量，我们报告少样本线性准确率，而不是完整微调准确率。图4给出了结果。在较小数据集上，视觉Transformer比计算成本相当的ResNet更容易过拟合。例如，ViT-B/32略快于ResNet50；在900万子集上，它的表现差得多，但在9000万及以上的子集上表现更好。ResNet152x2与ViT-L/16之间也是如此。这一结果强化了这样的直觉：卷积归纳偏置对较小数据集有用，但对于更大的数据集，直接从数据学习相关模式已经足够，甚至可能更加有利。

**图4：ImageNet线性少样本评估与预训练规模的关系。** 使用较小预训练数据集时，ResNet表现更好，但它比ViT更早达到平台；在更大规模预训练时，ViT表现更好。ViT-b指所有隐藏维度均减半的ViT-B。原始图形见PDF第7页。纵轴Linear 5-shot ImageNet Top1 [%]=ImageNet线性5样本Top-1准确率[%]；横轴Number of JFT pre-training samples=JFT预训练样本数（10M、30M、100M、300M为图示刻度）。图例为ViT-B/32、ViT-b/32、ResNet50x1（BiT）、ResNet152x2（BiT）。

总体来看，ImageNet上的少样本结果（图4）以及VTAB上的小数据结果（表2），都显示了极低数据量迁移的潜力。进一步分析ViT的少样本特性，是令人期待的未来研究方向。

[^2]: 注意，在ImageNet上预训练的模型也会微调，微调仍在ImageNet上进行。这是因为微调时提高分辨率可以改善性能。

### 4.4 规模扩展研究

我们通过评估从JFT-300M迁移后的性能，对不同模型进行了受控的规模扩展研究。在该设置中，数据规模不会成为模型性能的瓶颈，我们考察每个模型的性能与预训练成本的关系。模型集合包括：7个ResNet，即预训练7个轮次的R50x1、R50x2、R101x1、R152x1、R152x2，以及预训练14个轮次的R152x2和R200x3；6个视觉Transformer，即预训练7个轮次的ViT-B/32、B/16、L/32、L/16，以及预训练14个轮次的L/16和H/14；5个混合模型，即预训练7个轮次的R50+ViT-B/32、B/16、L/32、L/16，以及预训练14个轮次的R50+ViT-L/16（对于混合模型，模型名末尾的数字表示ResNet骨干中的总下采样倍率，而不是图像块大小）。

图5给出了迁移性能与总预训练计算量的关系（计算成本的详细说明见附录D.5）。附录表6列出了各模型的详细结果。可以观察到一些规律。第一，视觉Transformer在性能与计算量的权衡上优于ResNet。达到相同性能时（在5个数据集上取平均），ViT所需计算量约为其1/2至1/4。第二，在计算预算较小时，混合模型略优于ViT，但这种差异随模型增大而消失。这个结果有些出人意料，因为原本可能预期卷积的局部特征处理在任何模型大小下都会帮助ViT。第三，在所尝试的范围内，视觉Transformer似乎没有达到饱和，这为未来进一步扩展规模提供了动力。

**图5：不同架构的性能与预训练计算量关系，包括视觉Transformer、ResNet和混合模型。** 相同计算预算下，视觉Transformer通常优于ResNet。模型较小时，混合模型比纯Transformer有所改善，但这种差距在更大模型上消失。原始图形见PDF第7页。两个分图为ImageNet、Average-5（5个数据集的平均值）；纵轴Transfer accuracy [%]=迁移准确率[%]；横轴Total pre-training compute [exaFLOPs]=总预训练计算量[exaFLOPs]；图例Transformer（ViT）=Transformer（ViT）、ResNet（BiT）、Hybrid=混合模型，标注的ViT-L/16与ViT-L/32为模型名。

### 4.5 审视视觉Transformer

为了初步了解视觉Transformer如何处理图像数据，我们分析了它的内部表示。视觉Transformer的第一层将展平的图像块线性投影到较低维空间（公式1）。图7左侧显示了学习得到的嵌入滤波器的前几个主成分。这些成分类似一些合理的基函数，可用于低维表示每个图像块内部的精细结构。

投影之后，将可学习的位置嵌入加到图像块表示上。图7中间显示，模型学会了通过位置嵌入之间的相似度编码图像中的距离，也就是说，更近的图像块往往具有更相似的位置嵌入。此外，行列结构也显现出来：同一行或同一列中的图像块具有相似的嵌入。最后，在更大网格上，有时可以看到正弦结构（附录D）。位置嵌入能学会表示二维图像拓扑，这解释了为什么人工设计的二维感知嵌入变体没有带来改进（附录D.4）。

自注意力使ViT即便在最底层也能整合整个图像的信息。我们研究了网络在多大程度上利用这种能力。具体而言，我们根据注意力权重计算信息在图像空间中被整合时所跨越的平均距离（图7右侧）。这种“注意力距离”类似于CNN中的感受野大小。我们发现，有些注意力头在最底层就已经关注图像的大部分区域，说明模型确实利用了全局整合信息的能力。其他注意力头在低层则始终保持较小的注意力距离。在Transformer前应用ResNet的混合模型中，这种高度局部化的注意力不那么明显（图7右侧），表明它可能起到与CNN早期卷积层类似的作用。此外，注意力距离随网络深度增大而增加。从全局看，我们发现模型会关注与分类语义相关的图像区域（图6）。

**图6：输出词元对输入空间的注意力的代表性示例。** 细节见附录D.7。原始图形见PDF第8页；Input=输入，Attention=注意力。

**图7：** 左：ViT-L/32对RGB值进行初始线性嵌入所用的滤波器。中：ViT-L/32位置嵌入的相似度。每个小图显示指定行列图像块的位置嵌入与所有其他图像块位置嵌入之间的余弦相似度。右：随注意力头和网络深度变化的关注区域大小。每个点表示某层16个注意力头之一在图像间平均的注意力距离。细节见附录D.7。原始图形见PDF第9页。

图中标签：RGB embedding filters (first 28 principal components)=RGB嵌入滤波器（前28个主成分）；Position embedding similarity=位置嵌入相似度；Input patch row/column=输入图像块行／列；Cosine similarity=余弦相似度（色标−1至1）；Mean attention distance (pixels)=平均注意力距离（像素）；Network depth (layer)=网络深度（层）；Head 1/2/3/...=注意力头1/2/3/……；ViT-L/16为右侧模型名称。

### 4.6 自监督

Transformer在NLP任务上表现出色，但它的成功很大程度上不仅源于出色的可扩展性，也源于大规模自监督预训练（Devlin et al., 2019; Radford et al., 2018）。我们还对用于自监督的掩码图像块预测进行了初步探索，模仿BERT使用的掩码语言建模任务。经过自监督预训练，较小的ViT-B/16模型在ImageNet上达到79.9%的准确率，比从头训练显著提高2个百分点，但仍比监督预训练低4个百分点。更多细节见附录B.1.2。对比式预训练（Chen et al., 2020b; He et al., 2020; Bachman et al., 2019; Hénaff et al., 2020）的探索留待未来工作。

## 5 结论

我们探索了将Transformer直接应用于图像识别。与以往在计算机视觉中使用自注意力的工作不同，除最初的图像块提取步骤之外，我们没有在架构中引入图像专用归纳偏置。我们将图像解释为图像块序列，并使用NLP中的标准Transformer编码器处理它。这种简单而可扩展的策略，与大型数据集预训练结合时，效果出人意料地好。因此，视觉Transformer在许多图像分类数据集上达到或超过最先进水平，同时预训练成本相对低廉。

这些初步结果虽然令人鼓舞，但仍有许多挑战。其一是将ViT应用于检测和分割等其他计算机视觉任务。我们的结果结合Carion et al.（2020）的结果，说明这种方法具有潜力。另一项挑战是继续探索自监督预训练方法。初步实验表明自监督预训练带来了改进，但它与大规模监督预训练之间仍有很大差距。最后，进一步扩大ViT规模很可能会提升性能。

## 致谢

这项工作在Berlin、Zürich和Amsterdam开展。我们感谢Google的许多同事提供帮助，尤其感谢Andreas Steiner在基础设施和代码开源发布方面提供关键帮助；感谢Joan Puigcerver和Maxim Neumann在大规模训练基础设施方面提供帮助；感谢Dmitry Lepikhin、Aravindh Mahendran、Daniel Keysers、Mario Lučić、Noam Shazeer、Ashish Vaswani和Colin Raffel提供有益讨论。
## 参考文献

Samira Abnar and Willem Zuidema. 量化Transformer中的注意力流。ACL，2020。

Philip Bachman, R Devon Hjelm, and William Buchwalter. 通过最大化跨视图互信息学习表示。NeurIPS，2019。

Alexei Baevski and Michael Auli. 用于神经语言建模的自适应输入表示。ICLR，2019。

I. Bello, B. Zoph, Q. Le, A. Vaswani, and J. Shlens. 注意力增强卷积网络。ICCV，2019。

Lucas Beyer, Olivier J. Hénaff, Alexander Kolesnikov, Xiaohua Zhai, and Aäron van den Oord. 我们已经完成ImageNet的研究了吗？arXiv，2020。

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 语言模型是少样本学习器。arXiv，2020。

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. 使用Transformer进行端到端目标检测。ECCV，2020。

Mark Chen, Alec Radford, Rewon Child, Jeff Wu, and Heewoo Jun. 从像素进行生成式预训练。ICML，2020a。

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. 视觉表示对比学习的简单框架。ICML，2020b。

Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. UNITER：通用图像—文本表示学习。ECCV，2020c。

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. 使用稀疏Transformer生成长序列。arXiv，2019。

Jean-Baptiste Cordonnier, Andreas Loukas, and Martin Jaggi. 论自注意力与卷积层之间的关系。ICLR，2020。

J. Deng, W. Dong, R. Socher, L. Li, Kai Li, and Li Fei-Fei. ImageNet：大型分层图像数据库。CVPR，2009。

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT：用于语言理解的深度双向Transformer预训练。NAACL，2019。

Josip Djolonga, Jessica Yung, Michael Tschannen, Rob Romijnders, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Matthias Minderer, Alexander D’Amour, Dan Moldovan, Sylvan Gelly, Neil Houlsby, Xiaohua Zhai, and Mario Lucic. 论卷积神经网络的鲁棒性与可迁移性。arXiv，2020。

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 用于图像识别的深度残差学习。CVPR，2016。

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. 用于无监督视觉表示学习的动量对比。CVPR，2020。

Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, and Tim Salimans. 多维Transformer中的轴向注意力。arXiv，2019。

Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. 用于目标检测的关系网络。CVPR，2018。

Han Hu, Zheng Zhang, Zhenda Xie, and Stephen Lin. 用于图像识别的局部关系网络。ICCV，2019。

Zilong Huang, Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, and Thomas S. Huang. CCNet：用于语义分割的十字交叉注意力。ICCV，2020。

Olivier J. Hénaff, Aravind Srinivas, Jeffrey De Fauw, Ali Razavi, Carl Doersch, S. M. Ali Eslami, and Aaron van den Oord. 利用对比预测编码实现数据高效的图像识别。ICML，2020。

Sergey Ioffe and Christian Szegedy. 批归一化：通过减少内部协变量偏移加速深度网络训练。2015。

Diederik P. Kingma and Jimmy Ba. Adam：一种随机优化方法。ICLR，2015。

Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big Transfer（BiT）：通用视觉表示学习。ECCV，2020。

Alex Krizhevsky. 从微小图像中学习多层特征。技术报告，2009。

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. 使用深度卷积神经网络进行ImageNet分类。NIPS，2012。

Y. LeCun, B. Boser, J. Denker, D. Henderson, R. Howard, W. Hubbard, and L. Jackel. 将反向传播应用于手写邮政编码识别。Neural Computation，1:541–551，1989。

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard：通过条件计算与自动分片扩展巨型模型。arXiv，2020。

Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. VisualBERT：面向视觉与语言的简单高性能基线。Arxiv，2019。

Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. 使用槽注意力进行以物体为中心的学习。arXiv，2020。

Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT：为视觉与语言任务预训练任务无关的视觉语言表示。NeurIPS，2019。

Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. 探索弱监督预训练的极限。ECCV，2018。

M. Nilsback and A. Zisserman. 面向大量类别的自动花卉分类。ICVGIP，2008。

Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. 猫和狗。CVPR，2012。

Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. 图像Transformer。ICML，2018。

B. T. Polyak and A. B. Juditsky. 通过平均加速随机逼近。SIAM Journal on Control and Optimization，30(4):838–855，1992。doi: 10.1137/0330046。URL：[https://doi.org/10.1137/0330046](https://doi.org/10.1137/0330046)。

Siyuan Qiao, Huiyu Wang, Chenxi Liu, Wei Shen, and Alan Yuille. 权重标准化。arXiv预印本arXiv:1903.10520，2019。

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 通过无监督学习改善语言理解。技术报告，2018。

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 语言模型是无监督多任务学习器。技术报告，2019。

Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jon Shlens. 视觉模型中的独立自注意力。NeurIPS，2019。

Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. 重新审视深度学习时代数据不可思议的有效性。ICCV，2017。

Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. VideoBERT：视频与语言表示学习的联合模型。ICCV，2019。

Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herve Jegou. 修正训练与测试分辨率的差异。NeurIPS，2019。

Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herve Jegou. 修正训练与测试分辨率的差异：FixEfficientNet。arXiv预印本arXiv:2003.08237，2020。

Michael Tschannen, Josip Djolonga, Marvin Ritter, Aravindh Mahendran, Neil Houlsby, Sylvain Gelly, and Mario Lucic. 视频诱导视觉不变性的自监督学习。IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR）论文集，2020年6月。

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 你所需要的就是注意力。NIPS，2017。

Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-DeepLab：用于全景分割的独立轴向注意力。ECCV，2020a。

Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-DeepLab：用于全景分割的独立轴向注意力。arXiv预印本arXiv:2003.07853，2020b。

Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao. 学习用于机器翻译的深层Transformer模型。ACL，2019。

Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. 非局部神经网络。CVPR，2018。

Dirk Weissenborn, Oscar Täckström, and Jakob Uszkoreit. 扩展自回归视频模型。ICLR，2019。

Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Masayoshi Tomizuka, Kurt Keutzer, and Peter Vajda. 视觉Transformer：用于计算机视觉的基于词元的图像表示与处理。arxiv，2020。

Yuxin Wu and Kaiming He. 组归一化。ECCV，2018。

Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. 使用Noisy Student进行自训练以改善ImageNet分类。CVPR，2020。

Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. S4L：自监督半监督学习。ICCV，2019a。

Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. 使用视觉任务适配基准开展表示学习的大规模研究。arXiv预印本arXiv:1910.04867，2019b。

Hengshuang Zhao, Jiaya Jia, and Vladlen Koltun. 探索用于图像识别的自注意力。CVPR，2020。

## 附录

### A 多头自注意力

标准$qkv$自注意力（self-attention，SA；Vaswani et al., 2017）是神经架构中常用的组成模块。对于输入序列$\mathbf{z}\in\mathbb{R}^{N\times D}$中的每个元素，我们对序列中所有值$\mathbf{v}$进行加权求和。注意力权重$A_{ij}$依据序列中两个元素各自的查询表示$\mathbf{q}^i$与键表示$\mathbf{k}^j$之间的两两相似度计算。

$$
[\mathbf{q},\mathbf{k},\mathbf{v}]=\mathbf{z}\mathbf{U}_{qkv},\qquad
\mathbf{U}_{qkv}\in\mathbb{R}^{D\times3D_h}.
\tag{5}
$$

$$
A=\operatorname{softmax}\left(\mathbf{q}\mathbf{k}^{\top}/\sqrt{D_h}\right),\qquad
A\in\mathbb{R}^{N\times N}.
\tag{6}
$$

$$
\operatorname{SA}(\mathbf{z})=A\mathbf{v}.
\tag{7}
$$

多头自注意力（MSA）是SA的扩展：并行执行$k$个称为“头”的自注意力运算，然后对拼接后的输出进行投影。为在改变$k$时保持计算量和参数量不变，$D_h$（公式5）通常设为$D/k$。

$$
\operatorname{MSA}(\mathbf{z})=[\operatorname{SA}_1(\mathbf{z});\operatorname{SA}_2(\mathbf{z});\cdots;\operatorname{SA}_k(\mathbf{z})]\mathbf{U}_{msa},\qquad
\mathbf{U}_{msa}\in\mathbb{R}^{k\cdot D_h\times D}.
\tag{8}
$$

### B 实验细节

#### B.1 训练

表3汇总了不同模型的训练设置。我们发现，在ImageNet上从头训练模型时，强正则化至关重要。使用dropout时，除$qkv$投影之外的每个全连接层后面都会应用dropout，并且在位置嵌入与图像块嵌入相加后立即应用。混合模型的训练设置与对应ViT完全相同。最后，所有训练都以224分辨率进行。

**表3：训练超参数。** 所有模型训练时批大小为4096，学习率预热10000步。对于ImageNet，我们发现额外使用全局范数为1的梯度裁剪是有益的。训练分辨率为224。

| 模型 | 数据集 | 轮次 | 基础学习率 | 学习率衰减 | 权重衰减 | Dropout |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| ViT-B/{16,32} | JFT-300M | 7 | $8\cdot10^{-4}$ | 线性 | 0.1 | 0.0 |
| ViT-L/32 | JFT-300M | 7 | $6\cdot10^{-4}$ | 线性 | 0.1 | 0.0 |
| ViT-L/16 | JFT-300M | 7/14 | $4\cdot10^{-4}$ | 线性 | 0.1 | 0.0 |
| ViT-H/14 | JFT-300M | 14 | $3\cdot10^{-4}$ | 线性 | 0.1 | 0.0 |
| R50x{1,2} | JFT-300M | 7 | $10^{-3}$ | 线性 | 0.1 | 0.0 |
| R101x1 | JFT-300M | 7 | $8\cdot10^{-4}$ | 线性 | 0.1 | 0.0 |
| R152x{1,2} | JFT-300M | 7 | $6\cdot10^{-4}$ | 线性 | 0.1 | 0.0 |
| R50+ViT-B/{16,32} | JFT-300M | 7 | $8\cdot10^{-4}$ | 线性 | 0.1 | 0.0 |
| R50+ViT-L/32 | JFT-300M | 7 | $2\cdot10^{-4}$ | 线性 | 0.1 | 0.0 |
| R50+ViT-L/16 | JFT-300M | 7/14 | $4\cdot10^{-4}$ | 线性 | 0.1 | 0.0 |
| ViT-B/{16,32} | ImageNet-21k | 90 | $10^{-3}$ | 线性 | 0.03 | 0.1 |
| ViT-L/{16,32} | ImageNet-21k | 30/90 | $10^{-3}$ | 线性 | 0.03 | 0.1 |
| ViT-* | ImageNet | 300 | $3\cdot10^{-3}$ | 余弦 | 0.3 | 0.1 |

##### B.1.1 微调

所有ViT模型均使用动量为0.9的SGD微调。我们对学习率进行小规模网格搜索，范围见表4。为此，我们从训练集中划出小子集作为开发集（Pets和Flowers各10%，CIFAR为2%，ImageNet为1%），在剩余数据上训练。对于最终结果，我们在整个训练集上训练，并在对应测试数据上评估。微调ResNet和混合模型时使用完全相同的设置，唯一例外是在ImageNet上额外将0.06加入学习率搜索。此外，对于ResNet，我们还运行Kolesnikov et al.（2020）的设置，并在这次运行与我们搜索的结果中选择最佳值。最后，除非另有说明，所有微调实验都以384分辨率运行（使用与训练不同的分辨率进行微调是常见做法（Kolesnikov et al., 2020））。

**表4：微调超参数。** 所有模型均使用余弦学习率衰减，批大小为512，不使用权重衰减，梯度裁剪的全局范数为1。除非另有说明，微调分辨率为384。

| 数据集 | 步数 | 基础学习率 |
| --- | ---: | --- |
| ImageNet | 20 000 | {0.003, 0.01, 0.03, 0.06} |
| CIFAR100 | 10 000 | {0.001, 0.003, 0.01, 0.03} |
| CIFAR10 | 10 000 | {0.001, 0.003, 0.01, 0.03} |
| Oxford-IIIT Pets | 500 | {0.001, 0.003, 0.01, 0.03} |
| Oxford Flowers-102 | 500 | {0.001, 0.003, 0.01, 0.03} |
| VTAB（19项任务） | 2 500 | 0.01 |

将ViT模型迁移到另一个数据集时，我们移除整个分类头（两个线性层），替换为单个零初始化线性层，其输出维度为目标数据集所需的类别数。我们发现，这比仅重新初始化最后一层稍微更稳健。

对于VTAB，我们遵循Kolesnikov et al.（2020）的协议，并对所有任务使用相同超参数设置。学习率为0.01，训练2500步（表4）。我们在两个学习率和两种训练日程上进行小规模搜索，选择在每项含200个样本的验证集上VTAB得分最高的设置。我们遵循Kolesnikov et al.（2020）的预处理方式，唯一例外是不使用任务专用输入分辨率。相反，我们发现视觉Transformer在所有任务上都最受益于高分辨率（384×384）。

##### B.1.2 自监督

我们在初步自监督实验中采用掩码图像块预测目标。具体而言，我们对50%的图像块嵌入进行扰动：将其替换为可学习的`[mask]`嵌入（80%）、另一个随机图像块嵌入（10%），或保持原样（10%）。这种设置与Devlin et al.（2019）用于语言的设置非常相似。最后，使用各扰动图像块的对应表示，预测其3比特平均颜色（即总计512种颜色）。

我们在JFT上以4096的批大小，将自监督模型训练100万步（约14个轮次）。使用Adam，基础学习率为$2\cdot10^{-4}$，预热10000步，并采用余弦学习率衰减。对于预训练预测目标，我们尝试了以下设置：（1）只预测平均3比特颜色（即从512种颜色中做1次预测）；（2）并行预测16×16图像块下采样后的4×4版本，颜色为3比特（即从512种颜色中做16次预测）；（3）使用L2对完整图像块回归（即对3个RGB通道进行256次回归）。令人惊讶的是，我们发现这些方式都相当有效，不过L2稍差。我们只报告方案（1）的最终结果，因为它的少样本性能最好。我们也试验了Devlin et al.（2019）使用的15%扰动率，但按我们的少样本指标评估，结果同样稍差。

最后，我们想指出，我们采用的掩码图像块预测方法，并不需要如此大量的预训练，也不需要JFT这样的大型数据集，就能在ImageNet分类上带来相似的性能收益。具体来说，我们观察到，预训练100000步之后，下游性能提升开始出现边际收益递减；在ImageNet上预训练时，也观察到了相似收益。
### C 补充结果

我们报告了与正文各图对应的详细结果。表5对应正文图3，显示不同ViT模型在规模递增的数据集ImageNet、ImageNet-21k和JFT-300M上预训练后的迁移性能。表6对应正文图5，显示不同大小的ViT、ResNet和混合模型的迁移性能，以及估算的预训练计算成本。

**表5：在ImageNet、ImageNet-21k或JFT300M上预训练的视觉Transformer，在不同数据集上的Top-1准确率（%）。** 这些数值对应正文图3。模型在384分辨率下微调。注意，这里的ImageNet结果未使用表2结果所采用的额外技术（Polyak平均和512分辨率图像）。

| 预训练数据集 | 评估数据集 | ViT-B/16 | ViT-B/32 | ViT-L/16 | ViT-L/32 | ViT-H/14 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ImageNet | CIFAR-10 | 98.13 | 97.77 | 97.86 | 97.94 | − |
| ImageNet | CIFAR-100 | 87.13 | 86.31 | 86.35 | 87.07 | − |
| ImageNet | ImageNet | 77.91 | 73.38 | 76.53 | 71.16 | − |
| ImageNet | ImageNet ReaL | 83.57 | 79.56 | 82.19 | 77.83 | − |
| ImageNet | Oxford Flowers-102 | 89.49 | 85.43 | 89.66 | 86.36 | − |
| ImageNet | Oxford-IIIT-Pets | 93.81 | 92.04 | 93.64 | 91.35 | − |
| ImageNet-21k | CIFAR-10 | 98.95 | 98.79 | 99.16 | 99.13 | 99.27 |
| ImageNet-21k | CIFAR-100 | 91.67 | 91.97 | 93.44 | 93.04 | 93.82 |
| ImageNet-21k | ImageNet | 83.97 | 81.28 | 85.15 | 80.99 | 85.13 |
| ImageNet-21k | ImageNet ReaL | 88.35 | 86.63 | 88.40 | 85.65 | 88.70 |
| ImageNet-21k | Oxford Flowers-102 | 99.38 | 99.11 | 99.61 | 99.19 | 99.51 |
| ImageNet-21k | Oxford-IIIT-Pets | 94.43 | 93.02 | 94.73 | 93.09 | 94.82 |
| JFT-300M | CIFAR-10 | 99.00 | 98.61 | 99.38 | 99.19 | 99.50 |
| JFT-300M | CIFAR-100 | 91.87 | 90.49 | 94.04 | 92.52 | 94.55 |
| JFT-300M | ImageNet | 84.15 | 80.73 | 87.12 | 84.37 | 88.04 |
| JFT-300M | ImageNet ReaL | 88.85 | 86.27 | 89.99 | 88.28 | 90.33 |
| JFT-300M | Oxford Flowers-102 | 99.56 | 99.27 | 99.56 | 99.45 | 99.68 |
| JFT-300M | Oxford-IIIT-Pets | 95.80 | 93.40 | 97.11 | 95.83 | 97.56 |

**表6：模型规模扩展实验的详细结果。** 这些结果对应正文图5。表中给出了多个数据集上的迁移准确率，以及预训练计算量（exaFLOPs）。

| 模型 | 轮次 | ImageNet | ImageNet ReaL | CIFAR-10 | CIFAR-100 | Pets | Flowers | exaFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-B/32 | 7 | 80.73 | 86.27 | 98.61 | 90.49 | 93.40 | 99.27 | 55 |
| ViT-B/16 | 7 | 84.15 | 88.85 | 99.00 | 91.87 | 95.80 | 99.56 | 224 |
| ViT-L/32 | 7 | 84.37 | 88.28 | 99.19 | 92.52 | 95.83 | 99.45 | 196 |
| ViT-L/16 | 7 | 86.30 | 89.43 | 99.38 | 93.46 | 96.81 | 99.66 | 783 |
| ViT-L/16 | 14 | 87.12 | 89.99 | 99.38 | 94.04 | 97.11 | 99.56 | 1567 |
| ViT-H/14 | 14 | 88.08 | 90.36 | 99.50 | 94.71 | 97.11 | 99.71 | 4262 |
| ResNet50x1 | 7 | 77.54 | 84.56 | 97.67 | 86.07 | 91.11 | 94.26 | 50 |
| ResNet50x2 | 7 | 82.12 | 87.94 | 98.29 | 89.20 | 93.43 | 97.02 | 199 |
| ResNet101x1 | 7 | 80.67 | 87.07 | 98.48 | 89.17 | 94.08 | 95.95 | 96 |
| ResNet152x1 | 7 | 81.88 | 87.96 | 98.82 | 90.22 | 94.17 | 96.94 | 141 |
| ResNet152x2 | 7 | 84.97 | 89.69 | 99.06 | 92.05 | 95.37 | 98.62 | 563 |
| ResNet152x2 | 14 | 85.56 | 89.89 | 99.24 | 91.92 | 95.75 | 98.75 | 1126 |
| ResNet200x3 | 14 | 87.22 | 90.15 | 99.34 | 93.53 | 96.32 | 99.04 | 3306 |
| R50x1+ViT-B/32 | 7 | 84.90 | 89.15 | 99.01 | 92.24 | 95.75 | 99.46 | 106 |
| R50x1+ViT-B/16 | 7 | 85.58 | 89.65 | 99.14 | 92.63 | 96.65 | 99.40 | 274 |
| R50x1+ViT-L/32 | 7 | 85.68 | 89.04 | 99.24 | 92.93 | 96.97 | 99.43 | 246 |
| R50x1+ViT-L/16 | 7 | 86.60 | 89.72 | 99.18 | 93.64 | 97.03 | 99.40 | 859 |
| R50x1+ViT-L/16 | 14 | 87.12 | 89.76 | 99.31 | 93.89 | 97.36 | 99.11 | 1668 |

### D 补充分析

#### D.1 ResNet中的SGD与Adam

ResNet通常使用SGD训练，我们采用Adam作为优化器相当不常见。这里给出促使我们作出这一选择的实验。具体来说，我们比较了使用SGD和Adam在JFT上预训练的两个ResNet（50x1和152x2）的微调性能。对于SGD，我们使用Kolesnikov et al.（2020）推荐的超参数。结果见表7。在大多数数据集上以及总体平均值上，Adam预训练均优于SGD预训练。这为在JFT上预训练ResNet时选择Adam优化器提供了依据。注意，这里的绝对数值低于Kolesnikov et al.（2020）报告的数值，因为我们只预训练7个轮次，而不是30个轮次。

**表7：使用Adam和SGD预训练的ResNet模型的微调结果。**

| 数据集 | ResNet50：Adam | ResNet50：SGD | ResNet152x2：Adam | ResNet152x2：SGD |
| --- | ---: | ---: | ---: | ---: |
| ImageNet | 77.54 | 78.24 | 84.97 | 84.37 |
| CIFAR10 | 97.67 | 97.46 | 99.06 | 99.07 |
| CIFAR100 | 86.07 | 85.17 | 92.05 | 91.06 |
| Oxford-IIIT Pets | 91.11 | 91.00 | 95.37 | 94.79 |
| Oxford Flowers-102 | 94.26 | 92.06 | 98.62 | 99.32 |
| 平均值 | 89.33 | 88.79 | 94.01 | 93.72 |

#### D.2 Transformer的形状

我们对Transformer架构不同维度的扩展进行了消融实验，以了解哪些维度最适合扩展到超大模型。图8显示不同配置在ImageNet上的5样本性能。所有配置均基于一个8层ViT模型，其中$D=1024$、$D_{\mathrm{MLP}}=2048$、图像块大小为32；这也是所有曲线的交点。可以看到，扩展深度带来的改进最大，直到64层仍清晰可见。但在16层之后，就已经可以看到边际收益递减。有趣的是，扩展网络宽度似乎带来的变化最小。减小图像块大小，从而增加有效序列长度，在不引入参数的情况下，带来了出人意料地稳健的改进。这些发现表明，相比参数量，计算量可能是更好的性能预测指标；如果一定要有所侧重，扩展规模应当更侧重深度，而非宽度。总体而言，我们发现按比例扩展所有维度能带来稳健改进。

**图8：扩展视觉Transformer的不同模型维度。** 原始图形见PDF第16页。左纵轴ImageNet 5shot=ImageNet 5样本性能；右纵轴Average 5shot=平均5样本性能；横轴Relative Compute=相对计算量；图例Models=模型，All=全部维度，Depth=深度，Patch size=图像块大小，Width MLP=MLP宽度，Width=宽度。

#### D.3 分类头类型与class词元

为尽量接近原始Transformer模型，我们使用了一个额外的`[class]`词元，将其作为图像表示。随后，利用一个小型多层感知机（MLP），将该词元的输出变换为类别预测；该MLP仅有一个隐藏层，其非线性激活为tanh。

这一设计继承自用于文本的Transformer模型，我们在正文所有实验中均采用该设计。最初我们曾尝试仅使用图像块嵌入，对其进行全局平均池化（GAP），再接一个线性分类器，就像ResNet的最终特征图一样，但表现很差。然而，我们发现，这既不是额外词元导致的，也不是GAP运算导致的。实际上，性能差异完全可以由它们需要不同学习率来解释，见图9。

**图9：class词元分类器与全局平均池化分类器的比较。** 二者效果相近，但需要不同的学习率。原始图形见PDF第17页。纵轴ImageNet linear 5-shot accuracy [%]=ImageNet线性5样本准确率[%]；横轴Epochs of training=训练轮次；图例CLS-Token, lr=8e-4=CLS词元，学习率$8\times10^{-4}$；GAP, lr=8e-4=全局平均池化，学习率$8\times10^{-4}$；GAP, lr=3e-4=全局平均池化，学习率$3\times10^{-4}$。

#### D.4 位置嵌入

我们对利用位置嵌入编码空间信息的不同方式进行了消融实验，尝试了以下情况：

- 不提供位置信息：将输入视为一个图像块袋。
- 一维位置嵌入：将输入视为按光栅顺序排列的图像块序列（本文其他所有实验的默认设置）。
- 二维位置嵌入：将输入视为二维图像块网格。在这种情况下，学习两组嵌入，各对应一个坐标轴，即X嵌入与Y嵌入，每组维度为$D/2$。然后，根据输入图像块的坐标，拼接X和Y嵌入，得到该图像块最终的位置嵌入。
- 相对位置嵌入：使用图像块间的相对距离而不是绝对位置编码空间信息。为此，我们使用一维相对注意力，定义所有可能图像块对之间的相对距离。因此，对任意给定的一对图像块（一个在注意力机制中作为查询，另一个作为键／值），都有一个偏移量$p_q-p_k$，每个偏移量对应一个嵌入。然后，我们执行一次额外的注意力运算：仍使用原始查询（查询的内容），但将相对位置嵌入作为键。再将相对注意力的logit作为偏置项，在应用softmax之前，加到主注意力（基于内容的注意力）的logit上。

除了编码空间信息的不同方式，我们还尝试了将这些信息融入模型的不同方式。对于一维和二维位置嵌入，我们尝试了三种情况：（1）在模型的stem之后、将输入送入Transformer编码器之前，立即将位置嵌入加到输入上（本文其他所有实验的默认设置）；（2）在每层开头学习位置嵌入并加到输入上；（3）在每层开头将同一组可学习位置嵌入加到输入上（层间共享）。

**表8：使用ViT-B/16模型进行位置嵌入消融研究的结果，在ImageNet线性5样本设置下评估。**

| 位置嵌入 | 默认／stem处 | 每一层 | 每一层，层间共享 |
| --- | ---: | ---: | ---: |
| 无位置嵌入 | 0.61382 | N/A | N/A |
| 一维位置嵌入 | 0.64206 | 0.63964 | 0.64292 |
| 二维位置嵌入 | 0.64001 | 0.64046 | 0.64022 |
| 相对位置嵌入 | 0.64032 | N/A | N/A |

表8汇总了ViT-B/16上的这项消融研究结果。可以看到，不使用位置嵌入的模型与使用位置嵌入的模型之间性能差距很大，但不同位置信息编码方式之间几乎没有差别。我们推测，因为Transformer编码器处理的是图像块级输入，而不是像素级输入，所以空间信息编码方式的差异不那么重要。更具体地说，图像块级输入的空间维度远小于原始像素级输入，例如14×14，而非224×224；在这种分辨率下，对不同位置编码策略来说，学习空间关系的表示都同样容易。即便如此，网络学得的位置嵌入相似度的具体模式，仍取决于训练超参数（图10）。

**图10：采用不同超参数训练的模型的位置嵌入。** 原始图形见PDF第18页。图中Input patch row/column=输入图像块行／列；Cosine similarity=余弦相似度（色标−1至1）。三个模型均为ViT-L16，从左至右设置为：7个轮次、学习率0.0002、权重衰减0.01；7个轮次、学习率0.0004、权重衰减0.1；14个轮次、学习率0.0004、权重衰减0.1。

#### D.5 实测计算成本

我们还关注这些架构在所用硬件上的实际速度；由于硬件并行通道宽度、缓存大小等细节，理论FLOPs并不总能准确预测实际速度。为此，我们在TPUv3加速器上测量了主要研究模型的推理速度；推理速度与反向传播速度相差一个与模型无关的常数因子。

图12左侧显示了不同输入大小下，一个核心每秒可处理的图像数量。每个点都表示在很广的批大小范围内测得的峰值性能。可以看到，ViT随图像尺寸在理论上的四次方增长，直到最大模型与最高分辨率的组合下才刚刚开始显现。

另一个值得关注的量是每个模型在一个核心上可容纳的最大批大小，这个值越大，越有利于扩展到大型数据集。图12右侧显示相同模型集合的这个量。结果表明，与ResNet相比，大型ViT模型在内存效率上有明显优势。

**图12：** 左：不同架构在不同输入大小下的实际耗时测量。ViT模型的速度与类似ResNet相当。右：不同架构在不同输入大小下，设备上每核能够容纳的最大批大小。ViT模型显然更节省内存。原始图形见PDF第19页。左纵轴Peak inference speed [img/sec/core]=峰值推理速度[图像／秒／核]；右纵轴Largest per-core batch-size=每核最大批大小；横轴Input size [px]=输入尺寸[像素]（64、128、224、384、512）；图例模型名R50x1、R50x2、ViT-B/32、ViT-L/32、ViT-B/16、ViT-L/16、ViT-H/14、R152x4保留。

#### D.6 轴向注意力

轴向注意力（Axial Attention）（Huang et al., 2020; Ho et al., 2019）是一种简单但有效的技术，用于在组织为多维张量的大输入上执行自注意力。其基本思想是执行多个注意力运算，每个运算仅沿输入张量的单个轴进行，而不是把输入展平后应用一维注意力。在轴向注意力中，每次注意力沿特定轴混合信息，同时保持其他轴上的信息相互独立。沿着这一思路，Wang et al.（2020b）提出AxialResNet模型：将ResNet50中所有卷积核大小为3×3的卷积替换为轴向自注意力，即添加相对位置编码的行注意力和列注意力。我们实现了AxialResNet作为基线模型。[^3]

此外，我们修改了ViT，使其以二维形状处理输入，而不是以一维图像块序列处理；并引入轴向Transformer模块。在这种模块中，原本的一次自注意力加一个MLP，被替换为行自注意力加一个MLP，然后再接列自注意力加一个MLP。

图13展示了Axial ResNet、Axial-ViT-B/32和Axial-ViT-B/16在JFT预训练后，于ImageNet线性5样本设置下的性能与预训练计算量的关系，计算量分别用FLOPs和推理时间（每秒样本数）衡量。可以看到，Axial-ViT-B/32和Axial-ViT-B/16的性能都优于对应的ViT-B模型，但代价是更多计算。这是因为在Axial-ViT中，每个全局自注意力Transformer模块都被替换为两个轴向Transformer模块，一个做行自注意力，另一个做列自注意力。虽然轴向情况下自注意力作用的序列更短，但每个Axial-ViT模块都多了一个MLP。对于AxialResNet，尽管从准确率与计算量的权衡来看似乎合理（图13左），但朴素实现在TPU上极慢（图13右）。

**图13：基于轴向注意力的模型，其ImageNet线性5样本Top-1准确率与速度的关系，速度分别用FLOPs数量（左）和推理时间（原文此处也写作“左”）衡量。** 原始图形见PDF第20页。译注：根据图内横轴，推理速度实际位于右图。纵轴ImageNet 5-shot linear top-1 accuracy=ImageNet线性5样本Top-1准确率；左横轴Total compute [exaFLOPs]=总计算量[exaFLOPs]；右横轴Peak inference speed [img/sec/core]=峰值推理速度[图像／秒／核]；图中AxialViT-B/16、ViT-B/16、AxialViT-B/32、ViT-B/32、AxialResNet50、ResNet50为模型名。

[^3]: 我们的实现基于开源PyTorch实现[axial-deeplab](https://github.com/csrhddlam/axial-deeplab)。实验中，我们复现了Wang et al.（2020b）报告的准确率；但与该开源实现一样，我们的实现在TPU上速度非常慢。因此，我们无法用它开展广泛的大规模实验。经过精心优化的实现，或许能使这些实验成为可能。

#### D.7 注意力距离

为了解ViT如何使用自注意力整合整幅图像的信息，我们分析了不同层中注意力权重所跨越的平均距离（图11）。这种“注意力距离”类似于CNN的感受野大小。在较低层中，不同注意力头的平均注意力距离差异很大：有些头关注图像的大部分区域，另一些则关注查询位置或其附近的小区域。随着深度增大，所有注意力头的注意力距离都增加。在网络的后半部分，大多数头会广泛关注各词元。

**图11：随注意力头和网络深度变化的关注区域大小。** 在128幅示例图像上，对查询像素与所有其他像素之间的距离按注意力权重进行加权平均，从而计算注意力距离。每个点显示某层16个头之一在图像间平均的注意力距离。图像宽度为224像素。原始图形见PDF第18页。左模型ViT-L/16，右模型R50x1 + ViT-L/16；纵轴Mean attention distance (pixels)=平均注意力距离（像素）；横轴Network depth (layer)=网络深度（层）；Head 1/2/3/...=注意力头1/2/3/……。

#### D.8 注意力图

为计算输出词元到输入空间的注意力图（图6和图14），我们使用了注意力展开（Attention Rollout）（Abnar & Zuidema, 2020）。简要来说，我们先对ViT-L/16所有注意力头的权重取平均，再递归相乘所有层的权重矩阵。这样就考虑了注意力经过所有层时在词元之间发生的混合。

**图14：更多与图6相同类型的注意力图示例（随机选取）。** 原始图形见PDF第21页；图中编号1–128用于标识128个示例。

#### D.9 ObjectNet结果

我们还按Kolesnikov et al.（2020）的评估设置，在ObjectNet基准上评估了旗舰ViT-H/14模型，得到82.1%的Top-5准确率和61.7%的Top-1准确率。

#### D.10 VTAB细分结果

表9显示了各项VTAB-1k任务上取得的得分。

**表9：按任务细分的VTAB-1k性能。** 为便于阅读，原表的任务列在这里转置为行，所有得分与原表一一对应。模型名中JFT与I21k分别保留原文标识。

| 任务 | ViT-H/14（JFT） | ViT-L/16（JFT） | ViT-L/16（I21k） |
| --- | ---: | ---: | ---: |
| Caltech101 | 95.3 | 95.4 | 90.8 |
| CIFAR-100 | 85.5 | 81.9 | 84.1 |
| DTD | 75.2 | 74.3 | 74.1 |
| Flowers102 | 99.7 | 99.7 | 99.3 |
| Pets | 97.2 | 96.7 | 92.7 |
| Sun397 | 65.0 | 63.5 | 61.0 |
| SVHN | 88.9 | 87.4 | 80.9 |
| Camelyon | 83.3 | 83.6 | 82.5 |
| EuroSAT | 96.7 | 96.5 | 95.6 |
| Resisc45 | 91.4 | 89.7 | 85.2 |
| Retinopathy（视网膜病变） | 76.6 | 77.1 | 75.3 |
| Clevr-Count（计数） | 91.7 | 86.4 | 70.3 |
| Clevr-Dist（距离） | 63.8 | 63.1 | 56.1 |
| DMLab | 53.1 | 49.7 | 41.9 |
| dSpr-Loc（位置） | 79.4 | 74.5 | 74.7 |
| dSpr-Ori（朝向） | 63.3 | 60.5 | 64.9 |
| KITTI-Dist（距离） | 84.5 | 82.2 | 79.9 |
| sNORB-Azim（方位角） | 33.2 | 36.2 | 30.5 |
| sNORB-Elev（仰角） | 51.2 | 51.1 | 41.7 |
| 平均值（Mean） | 77.6 | 76.3 | 72.7 |
