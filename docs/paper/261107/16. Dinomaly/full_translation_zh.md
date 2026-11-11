# Dinomaly：多类别无监督异常检测中的“少即是多”哲学

**原文：**[Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection](<16-Dinomaly-2025.pdf>)。本译文对应本目录 11 页 CVPR 2025 会议论文 PDF，论文集页码 20405—20415。翻译日期：2026-11-07；按本地 PDF 复核补全：2026-11-11。

**作者：**Jia Guo¹、Shuai Lu²、Weihang Zhang²†、Fang Chen³、Huiqi Li²†、Hongen Liao¹˒³✉。

**单位：**¹清华大学生物医学工程学院，中国北京；²北京理工大学，中国北京；³上海交通大学生物医学工程学院，中国上海。邮箱：guojia.jeremy@gmail.com；lushuaie@163.com；zhangweihang@bit.edu.cn；chen-fang@sjtu.edu.cn；huiqili@bit.edu.cn；liao@tsinghua.edu.cn。✉通讯作者；†项目启动时的原导师。

**翻译说明：**逐段覆盖本地 PDF 全部 11 页，包括摘要、正文第 1—5 节、图 1—5 图注与关键标注、表 1—5 全部数据、公式 (1)—(6)、脚注、致谢和参考文献 [1]—[70]。原文提及的附录 A—F 及 A 系列表格不在该本地 PDF 中，译文保留这些引用，不臆造未收录内容。合并跨行跨页续段，去除运行页眉页脚与开放获取水印。图形不重绘，请结合 PDF 查看；编号、数字、模型和数据集名保留，原文不一致处以译注说明。“最佳”等判断均属于论文发表时作者的陈述。

## 摘要

近期研究强调一种实用的无监督异常检测（UAD）设定：为多类别图像建立统一模型。虽然已有多项进展处理这一挑战性任务，多类别设定下的检测性能仍远低于先进的逐类别独立模型。我们旨在弥合这一显著差距。本文提出 Dinomaly，一种极简的重建式异常检测框架，采用纯 Transformer 架构，不依赖复杂设计、额外模块或专门技巧。在仅包含注意力和多层感知机（MLP）的强大框架中，我们发现四个对多类别异常检测至关重要的简单组件：(1) 可扩展的基础 Transformer，提取通用且有区分力的特征；(2) 含噪瓶颈，由现成 Dropout 完成全部噪声注入；(3) 天然无法聚焦的线性注意力；(4) 不强制逐层、逐点重建的宽松重建。我们在 MVTec-AD、VisA、Real-IAD 等流行异常检测基准上开展大量实验。Dinomaly 在三个数据集上分别取得 99.6%、98.7%、89.3% 的图像级 AUROC，不仅优于先进多类别 UAD 方法，还达到最先进的逐类别 UAD 纪录。代码：<https://github.com/guojiajeremy/Dinomaly>。

## 1 引言

无监督异常检测（UAD）旨在从正常图像中检测异常模式，并进一步定位异常区域。由于潜在异常多样且稀少，该任务采用无监督范式，对仅包含正常样本的可用训练集建模。UAD 广泛应用于工业缺陷检测 [3]、医学疾病筛查 [13]、视频监控 [37] 等，解决这些场景中难以收集和标注所有可能异常的问题。

传统 UAD 为每个物体类别建立独立模型，如图 1(a)。但这种一类一模型设定需要大量模型存储开销 [60]，尤其当应用需要许多类别时。对 UAD 来说，正常模式的紧致边界对区分异常至关重要。当类别多样导致正常类内部模式过于复杂，其分布就难以度量，从而损害检测性能。近期，UniAD [60] 及后续研究提出训练统一模型进行多类别异常检测（MUAD），如图 1(b)。在该设定下，无论正常或异常都直接将输入复制为输出的“恒等映射”，会损害传统方法性能 [60]。这一现象源于多类别正常模式的多样性，促使网络泛化到未见模式。

两年来，研究者提出邻域掩码注意力 [60]、合成异常 [68]、向量量化 [36]、扩散模型 [16, 59]、状态空间模型 Mamba [17] 等方法处理 MUAD。然而，先进 MUAD 与逐类别 UAD 仍有不可忽略的性能差距，限制了统一模型的实际应用，如图 1(c)。另外，既有方法采用精心设计、可能不直观的模块与架构，通用性和易用性有限 [18, 36]。

**图 1：Dinomaly 的设定、基准比较与扩展。**(a) 逐类别独立 UAD 任务设定；(b) 模型统一的多类别 UAD；(c) 在 MVTec-AD [3]、VisA [70]、Real-IAD [54] 上与此前先进方法比较；(d) Dinomaly 的扩展性质。三个基准分别包含 15、12、30 类，纵轴为图像级 AUROC（%）。图中 Dinomaly 为 99.6、98.7、89.3；MUAD 对比项 ViTAD/MambaAD 在 MVTec-AD 上为 98.3/98.6，ReContrast/MambaAD 在 VisA 上为 95.5/94.3，在 Real-IAD 上为 86.4/86.3；逐类别对照在三个子图分别为 99.6/99.6、98.1/97.8、89.4/88.5。下排分别展示模型规模扩展（ViT-Small 37.4M、ViT-Base 148.0M、ViT-Large 275.3M 参数；横轴 GMACs）、输入尺寸扩展（$280^2,336^2,392^2,448^2$；纵轴像素 AP）与预训练扩展（横轴 ImageNet 线性探测准确率；纵轴图像 AUROC），†表示默认设定。

本文希望以多类别统一模型追平逐类别异常检测性能。我们提出 Dinomaly，一种完全由纯 Transformer 块 [51]（具体为自注意力和 MLP）构建的极简重建式 UAD 框架。首先，实证研究自监督预训练视觉 Transformer（ViT）[12] 用作提取重建目标的特征编码器时的扩展规律。随后，引入三个直接却关键的要素，在不增加复杂性或计算负担的前提下解决 MUAD 中的恒等映射。第一，激活 MLP 内置 Dropout，替代精心设计的伪异常与特征噪声，防止网络同时恢复正常和异常模式。第二，利用线性注意力的“副作用”：作为 Softmax 注意力的高效对应物，它难以聚焦局部区域，因此可以阻止相同信息直接传递。第三，既有方法逐层、逐区域重建，蒸馏出的解码器即使面对异常区域，也能很好模仿编码器。为此，我们将多个层视为整体分组，优化时丢弃重建良好区域，放宽重建约束。

为验证 MUAD 下的有效性，我们在 MVTec AD [3]（15 类）、VisA [70]（12 类）、Real-IAD（30 类）等广泛使用的基准上开展大量实验。图 1 表明，基础规模 Dinomaly 在三者上取得此前未有的 99.6%、98.7%、89.3% 图像级 AUROC，显著超过此前先进方法。此外，可扩展性是 Dinomaly 的关键特点。继续扩大模型，可将性能进一步提高至 99.8%、98.9%、90.1%；缩小参数和输入尺寸，则可为计算受限场景提供高效方案。

## 2 相关工作

**多类别 UAD。**UniAD [60] 首次引入多类别异常检测，希望以统一模型检测不同类别的异常。在这一设定下，传统 UAD 常遇到“恒等捷径”：正常与异常样本在推理时都能有效恢复 [60]。一般认为，这来自多类别正常模式的多样性，驱动网络泛化到未见模式，违背基于认知能力的方法的基本假设。许多近期研究致力于解决此问题 [14, 31, 36, 59, 60]。

UniAD [60] 用邻域掩码注意力和特征抖动缓解捷径。HVQ-Trans [36] 用向量量化（VQ）Transformer 使异常产生较大特征差异。LafitE [59] 使用潜空间扩散模型，并引入特征编辑缓解该问题。DiAD [16] 也用扩散处理多类别 UAD。OmniAL [68] 聚焦统一设定下的异常定位，通过合成伪异常防止恒等重建。ReContrast [14] 尝试通过两个编码器间交叉重建缓解恒等映射。ViTAD [5] 抽象出统一的特征重建 UAD 框架，采用 Transformer 构件。MambaAD [17] 探索近期状态空间模型 Mamba 在多类别 UAD 中的应用。更多相关工作见附录 A。

> 译注：此处 ViTAD 的引文 [5] 按原文保留，但参考文献 [5] 为 SimSiam 论文；参考文献中 ViTAD 对应 [65]。第 4.4 节另一次引用 ViTAD 为 [64]，同样保留原编号。

## 3 方法

### 3.1 Dinomaly 框架

“我无法创造的，我就不能理解。”——Richer Feynman

> 译注：人名拼写“Richer Feynman”依原文照录。

从已知事物中识别异常是人类先天能力，也是探索世界的重要途径。类似地，我们构建依赖人工神经网络认知特性的重建框架。Dinomaly 包含编码器、瓶颈和重建解码器，如图 2。不失一般性，使用具有 12 个 Transformer 层的预训练 ViT [12] 作编码器，提取不同语义尺度、信息丰富的特征图。瓶颈是简单 MLP（亦称前馈网络 FFN），收集编码器 8 个中间层的特征表征。解码器类似编码器，包含 8 个 Transformer 层。训练时，解码器通过最大化特征图余弦相似性，学习重建编码器中间特征；推理时，期望其重建正常区域，而由于从未见过异常，在异常区域失败。

**图 2：Dinomaly 框架，由简单纯粹的 Transformer 构件构成。**图中依次为输入图像、冻结的自监督预训练 Transformer、层分组求和、含噪 MLP 瓶颈、Transformer 解码器、宽松重建，以及由测试时重建误差产生的异常图。省略较低层并丢弃较深层。标准注意力块含 LayerNorm、多头注意力、LayerNorm、MLP 与残差相加；不聚焦注意力块将多头注意力换成多头线性注意力。MLP 由线性层、GELU、线性层及含噪时启用的 Dropout 组成，⊕为相加。

**基础 Transformer。**基础模型，尤其在大规模数据上预训练的 ViT [12, 33]，可作为特定视觉任务的基础和起点。它们采用对比学习（MoCov3 [6]、DINO [4]）、掩码图像建模（MAE [19]、SimMIM [57]、BEiT [40]）及二者结合（iBOT [69]、DINOv2 [39]）等自监督机制，产生适用于图像级和像素级任务的通用特征。

由于 UAD 缺乏监督，先进方法大多使用预训练网络提取有区分力的特征。近期工作 [28, 43, 65] 初步发现，自监督模型鲁棒通用的特征优于领域特定的 ImageNet 特征。本文通过系统分析基础 ViT，率先研究 UAD 模型扩展行为，概览见图 1(d)。全面评估包括预训练策略（图 5）、模型规模（表 4）、输入分辨率（表 5），详见第 4.4 节。权衡检测性能和效率，默认采用 DINOv2-Register [7] 预训练的 ViT-Base/14 作为编码器。

### 3.2 含噪瓶颈

“你所需要的只是 Dropout。”

既有研究 [14, 60, 68] 将多类别多样样本训练导致的 UAD 性能下降归因于“恒等映射”；本文将其重新表述为“过度泛化”。泛化能力是神经网络优点，使其在未见测试集上也能良好工作。但对利用神经网络认知特性的无监督异常检测而言，泛化并不总是所需。多类别 UAD 增加图像及模式多样性后，解码器会将重建能力泛化到未见异常样本，导致基于重建误差的检测失败。

解决恒等映射的直接方法，是将“重建”转为“恢复”。以往工作不直接从正常输入重建正常图像或特征，而是在输入图像 [62, 67] 或编码器特征 [59, 60] 上添加扰动作为伪异常，同时仍让解码器恢复无异常图像或特征，形成类似去噪的框架。但此类方法采用启发式、人工设计的异常生成策略，不一定跨领域、数据集和方法通用。本文采用简单优雅的 Dropout。自 Hinton 等 [21] 于 2014 年将其作为过拟合补救措施引入以来，Dropout 已成为包括 Transformer 在内神经架构的基础组件。Dinomaly 在 MLP 瓶颈中用 Dropout 随机丢弃神经激活。其作用并非缓解过拟合，而可解释为扰乱正常表征的伪特征异常，类似去噪自编码器 [52, 53]。无需任何专门模块，这一简单组件就能内在地迫使解码器无论测试图像是否异常都恢复正常特征，从而缓解恒等映射。

> 译注：上段年份 2014 按正文保留；其引用 [21] 在参考文献中的年份为 2012。

### 3.3 不聚焦的线性注意力

“彼之砒霜，吾之蜜糖。”

Softmax 注意力是 Transformer 的关键机制，使模型能够关注输入词元序列的不同部分。形式上，给定长度 $N$ 的输入序列 $X\in\mathbb R^{N\times d}$，注意力首先将其变为查询矩阵 $Q$、键矩阵 $K$ 和值矩阵 $V$，三者均属于 $\mathbb R^{N\times d}$：

$$Q=XW^Q,\qquad K=XW^K,\qquad V=XW^V,\tag{1}$$

其中 $W^Q,W^K,W^V\in\mathbb R^{d\times d}$ 为可学习参数。以查询键相似性计算注意力图，Softmax 注意力输出为¹

$$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}(QK^{\mathsf T})V.\tag{2}$$

回到 MUAD，先前方法 [36, 60] 建议用注意力代替卷积，因为卷积很容易学习恒等映射。但两种算子都可能因过度关注对应输入位置来产生输出而形成恒等映射：

$$\text{卷积核}=\begin{bmatrix}0&0&0\\0&1&0\\0&0&0\end{bmatrix},\qquad\text{注意力图}=\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}.$$

是否有简单方法阻止注意力关注相同信息？Dinomaly 利用一类不含 Softmax 的注意力——线性注意力——的“不聚焦能力”。线性注意力被提出，用于降低标准 Softmax 注意力相对于词元数的计算复杂度 [26]。以简单激活函数 $\phi(\cdot)$（通常 $\phi(x)=\operatorname{elu}(x)+1$）代替 Softmax，可将计算顺序从 $(QK^{\mathsf T})V$ 改为 $Q(K^{\mathsf T}V)$。形式上：

$$\operatorname{LA}(Q,K,V)=(\phi(Q)\phi(K^{\mathsf T}))V=\phi(Q)(\phi(K^{\mathsf T})V),\tag{3}$$

计算复杂度从 $O(N^2d)$ 降到 $O(Nd^2)$。复杂度与表达能力之间的权衡是两难问题。既有研究 [15, 48] 将线性注意力在有监督任务中的性能下降归因于无法聚焦：缺少 Softmax 非线性重加权后，它无法集中关注与查询相关的前景、邻域等重要区域。但在本文场景中，这正是重建解码器偏好的性质。

为研究注意力如何传播信息，我们训练两个 Dinomaly 变体，分别在解码器中以标准 Softmax 或线性注意力作空间混合器，并可视化注意力图。如图 3，Softmax 往往聚焦查询的确切区域，而线性注意力将注意力扩散到整幅图像。这意味着，线性注意力受无法聚焦所迫，利用更多长程信息恢复各位置特征，降低重建时把未见模式的相同信息传至下一层的机会。当然，它还减少计算，且不损失性能。

¹注意力完整形式为 $\operatorname{Softmax}(QK^{\mathsf T}/\sqrt d)V$。为叙述简单，省略常数分母；也省略拼接多个注意力的多头机制。

**图 3：Softmax 注意力与线性注意力。**(a) 注意力图可视化；(b) 注意力分布。(a) 展示解码器第 $l=1,\ldots,8$ 层注意力图，取 12 个头平均；输入示出图像与查询位置。(b) 横轴为键到查询的距离，纵轴为注意力权重；Softmax 在近查询位置有尖峰，线性注意力分布更分散。

### 3.4 宽松重建

“攥得越紧，拥有的越少。”

**宽松约束。**特征重建/蒸馏 UAD 的先驱 [10, 46] 受到知识蒸馏 [20] 启发。多数方法通过对应解码器层 [10, 46, 65]（图 4(a)）或最后一层 [58, 60]（图 4(b)）蒸馏特定编码器层，例如 ResNet 三个阶段各自最后一层。直觉上，更多编码器解码器特征对（图 4(c)）能利用不同层更多信息区分异常。然而，知识蒸馏直觉也意味着，逐层监督越多，学生（解码器）就越能模仿教师（编码器）[30]，这反而损害依赖编码器解码器差异检测异常的 UAD。这也体现为恒等映射。由于柱状 Transformer 层自上而下保持一致，我们把目标层特征图全部相加为一个整体组，放宽逐层约束，如图 4(d)。这可看作放松层间一一对应，为解码器提供更多自由度，使它在未见输入模式上可以与编码器表现明显不同。浅层特征含有利于精确定位的低层视觉特性，因此还可分成低语义层组与高语义层组，如图 4(e)。

**图 4：重建约束方案。**(a) 逐层（稀疏）；(b) 单层对拼接层；(c) 逐层（密集）；(d) 宽松组对组、单组（本文）；(e) 宽松组对组、双组（本文）。图例：编码器特征、解码器特征、重建箭头、拼接、相加；灰色提示为省略较低层和丢弃较深层。

**宽松损失。**根据上述分析，我们也通过丢弃特征图中的部分点，放松逐点重建损失。直接采用困难挖掘全局余弦损失 [14]，在训练中分离余弦距离较低、恢复良好的特征点梯度。令 $f_E,f_D$ 为编码器和解码器的分组特征图：

$$\mathcal L_{\mathrm{global-hm}}=D_{\cos}(\mathcal F(f_E),\mathcal F(\hat f_D)),\tag{4}$$

$$\hat f_D(h,w)=\begin{cases}\operatorname{sg}(f_D(h,w))_{0.1},&\text{若 }D_{\cos}(f_D,f_E)<k\%_{\mathrm{batch}},\\f_D(h,w),&\text{否则},\end{cases}\tag{5}$$

$$D_{\cos}(a,b)=1-\frac{a^{\mathsf T}\cdot b}{\|a\|\,\|b\|}.\tag{6}$$

其中 $D_{\cos}$ 为余弦距离，$\mathcal F(\cdot)$ 为展平操作，$f_D(h,w)$ 为位置 $(h,w)$ 的特征点，$\operatorname{sg}(\cdot)_{0.1}$ 表示将梯度缩为原来的十分之一。²条件 $D_{\cos}(f_D(h,w),f_E(h,w))<k\%_{\mathrm{batch}}$ 选择批次内余弦距离较小的 $k\%$ 特征点进行梯度缩小。总损失是所有编码器解码器特征对的 $\mathcal L_{\mathrm{global-hm}}$ 平均值。

²完全停止梯度偶尔会导致优化不稳定。

## 4 实验

**表 1：多类别 UAD 设定下的性能（%）。**†表示专为 MUAD 设计的方法。Dinomaly↑ 表示三个数据集训练迭代数分别扩大至 20,000、20,000、100,000（原为 10,000/10,000/50,000）。I 表示图像级，P 表示像素级。

| 数据集 | 方法 | I-AUROC | I-AP | I-F1-max | P-AUROC | P-AP | P-F1-max | P-AUPRO |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MVTec-AD [3] | RD4AD [10] | 94.6 | 96.5 | 95.2 | 96.1 | 48.6 | 53.8 | 91.1 |
| MVTec-AD | SimpleNet [34] | 95.3 | 98.4 | 95.8 | 96.9 | 45.9 | 49.7 | 86.5 |
| MVTec-AD | DeSTSeg [67] | 89.2 | 95.5 | 91.6 | 93.1 | 54.3 | 50.9 | 64.8 |
| MVTec-AD | UniAD [60]† | 96.5 | 98.8 | 96.2 | 96.8 | 43.4 | 49.5 | 90.7 |
| MVTec-AD | ReContrast [14]† | 98.3 | 99.4 | 97.6 | 97.1 | 60.2 | 61.5 | 93.2 |
| MVTec-AD | DiAD [18]† | 97.2 | 99.0 | 96.5 | 96.8 | 52.6 | 55.5 | 90.7 |
| MVTec-AD | ViTAD [65]† | 98.3 | 99.4 | 97.3 | 97.7 | 55.3 | 58.7 | 91.4 |
| MVTec-AD | MambaAD [17]† | 98.6 | 99.6 | 97.8 | 97.7 | 56.3 | 59.2 | 93.1 |
| MVTec-AD | Dinomaly（本文） | 99.6 | 99.8 | 99.0 | 98.4 | 69.3 | 69.2 | 94.8 |
| MVTec-AD | Dinomaly↑ | 99.7 | 99.8 | 99.2 | 98.4 | 69.3 | 69.2 | 94.7 |
| VisA [70] | RD4AD [10] | 92.4 | 92.4 | 89.6 | 98.1 | 38.0 | 42.6 | 91.8 |
| VisA | SimpleNet [34] | 87.2 | 87.0 | 81.8 | 96.8 | 34.7 | 37.8 | 81.4 |
| VisA | DeSTSeg [67] | 88.9 | 89.0 | 85.2 | 96.1 | 39.6 | 43.4 | 67.4 |
| VisA | UniAD [60]† | 88.8 | 90.8 | 85.8 | 98.3 | 33.7 | 39.0 | 85.5 |
| VisA | ReContrast [14]† | 95.5 | 96.4 | 92.0 | 98.5 | 47.9 | 50.6 | 91.9 |
| VisA | DiAD [18]† | 86.8 | 88.3 | 85.1 | 96.0 | 26.1 | 33.0 | 75.2 |
| VisA | ViTAD [65]† | 90.5 | 91.7 | 86.3 | 98.2 | 36.6 | 41.1 | 85.1 |
| VisA | MambaAD [17]† | 94.3 | 94.5 | 89.4 | 98.5 | 39.4 | 44.0 | 91.0 |
| VisA | Dinomaly（本文） | 98.7 | 98.9 | 96.2 | 98.7 | 53.2 | 55.7 | 94.5 |
| VisA | Dinomaly↑ | 98.9 | 99.0 | 96.4 | 98.8 | 53.8 | 55.8 | 94.5 |
| Real-IAD [54] | RD4AD [10] | 82.4 | 79.0 | 73.9 | 97.3 | 25.0 | 32.7 | 89.6 |
| Real-IAD | SimpleNet [34] | 57.2 | 53.4 | 61.5 | 75.7 | 2.8 | 6.5 | 39.0 |
| Real-IAD | DeSTSeg [67] | 82.3 | 79.2 | 73.2 | 94.6 | 37.9 | 41.7 | 40.6 |
| Real-IAD | UniAD [60]† | 83.0 | 80.9 | 74.3 | 97.3 | 21.1 | 29.2 | 86.7 |
| Real-IAD | ReContrast [14]† | 86.4 | 84.2 | 77.4 | 97.8 | 31.6 | 38.2 | 91.8 |
| Real-IAD | DiAD [18]† | 75.6 | 66.4 | 69.9 | 88.0 | 2.9 | 7.1 | 58.1 |
| Real-IAD | ViTAD [65]† | 82.7 | 80.2 | 73.7 | 97.2 | 24.3 | 32.3 | 84.8 |
| Real-IAD | MambaAD [17]† | 86.3 | 84.6 | 77.0 | 98.5 | 33.0 | 38.7 | 90.5 |
| Real-IAD | Dinomaly（本文） | 89.3 | 86.8 | 80.2 | 98.8 | 42.8 | 47.1 | 93.9 |
| Real-IAD | Dinomaly↑ | 89.5 | 86.9 | 80.4 | 98.9 | 43.3 | 47.4 | 94.2 |

### 4.1 实验设定

**数据集。**MVTec-AD [3] 包含 15 类（5 类纹理、10 类物体），训练集共 3,629 幅正常图像；测试集 1,725 幅，其中正常 467、异常 1,258。VisA [70] 含 12 类，按官方划分得到训练集 8,659 幅正常图像，测试集 2,162 幅，其中正常 962、异常 1,200。Real-IAD [54] 是近期发布的大规模 UAD 数据集，含 30 类不同物体。遵循包括全部视角的官方划分，训练集正常图像 36,465 幅，测试集 114,585 幅，其中正常 63,256、异常 51,329。

**指标。**遵循既有工作 [17, 65]，采用 7 项指标。图像级异常检测以接收者操作特征曲线下面积（AUROC）、平均精确率（AP）、最优阈值下 F1（F1-max）评估；像素级异常定位以 AUROC、AP、F1-max 和逐区域重叠曲线下面积（AUPRO）评估。数据集结果是全部类别的平均。

**实现细节。**默认使用 DINOv2-R [7] 预训练的 ViT-Base/14（图像块边长 14）作为编码器。含噪瓶颈丢弃率默认为 0.2，在多样的 Real-IAD 上增至 0.4。采用双组宽松约束，异常图为两组逐点余弦距离的平均。输入先缩放至 $448^2$，再中心裁剪至 $392^2$，因此特征图 $28^2$ 足够大，适合定位。使用带 AMSGrad [41] 的 StableAdamW [56]，训练比 AdamW [35] 更稳定；学习率 $2\times10^{-3}$，$\beta=(0.9,0.999)$，权重衰减 $10^{-4}$。MVTec-AD 和 VisA 训练 10,000 次迭代，Real-IAD 训练 50,000 次。详细设定见附录 B。

### 4.2 与先进多类别 UAD 方法比较

我们比较 Dinomaly 与最先进的 UAD、MUAD 方法 [10, 14, 17, 18, 34, 65, 67]。表 1 显示，Dinomaly 在所有数据集和所有指标上显著超过对比方法。在最常用的 MVTec-AD 上，图像级性能为 99.6/99.8/99.0（%），像素级为 98.4/69.3/69.2/94.8，比此前最佳分别提高 1.0/0.2/1.2 和 0.7/9.1/7.7/1.6。这表明 MUAD 下 MVTec-AD 图像级性能已近饱和。在 VisA 上，图像级为 98.7/98.9/96.2，像素级为 98.7/53.2/55.7/94.5，分别提高 3.2/2.5/4.2 和 0.2/5.3/5.1/2.6。在含 30 类、每类 5 个相机视角的 Real-IAD 上，图像级为 89.3/86.8/80.2，像素级为 98.8/42.8/47.1/93.9，分别提高 3.0/2.2/3.2 和 0.3/4.9/5.4/3.4，表明能够扩展至极复杂场景。逐类别性能和定性可视化见附录 E、F。其他流行基准 MPDD [24]、BTAD [38]、UniMedical [66] 也取得优良结果，I-AUROC 分别为 97.2、95.4、84.9，见附录表 A13。

### 4.3 与先进逐类别 UAD 比较

表 2 还比较了 Dinomaly 与先进逐类别方法。MUAD 下 Dinomaly 的表现可与为每类别建立独立模型的传统方法 [10, 34, 45] 相当。MVTec-AD、VisA 上，多类别 Dinomaly（第一行）相较逐类别版本（第二行）几乎没有性能下降。Real-IAD 类别和视角更多、更复杂，多类别版本有中等程度下降，但仍可与先进逐类别方法相当。

**表 2：传统逐类别 UAD 设定性能（%）。**n/a 为不可用；第一行专列 MUAD 结果作对照。

| 方法 | MVTec I-AUROC | MVTec P-AUROC | MVTec P-AUPRO | VisA I-AUROC | VisA P-AUROC | VisA P-AUPRO | Real-IAD I-AUROC | Real-IAD P-AUROC | Real-IAD P-AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dinomaly（MUAD） | 99.6 | 98.4 | 94.8 | 98.7 | 98.7 | 94.5 | 89.3 | 98.8 | 93.9 |
| Dinomaly | 99.7 | 99.9 | 95.0 | 98.9 | 98.9 | 95.1 | 92.0 | 99.1 | 95.1 |
| RD4AD [10] | 98.5 | 97.8 | 93.9 | 96.0 | 90.1 | 70.9 | 87.1 | n/a | 93.8 |
| PatchCore [45] | 99.1 | 98.1 | 93.5 | 94.7 | 98.5 | 91.8 | 89.4 | n/a | 91.5 |
| SimpleNet [34] | 99.6 | 98.1 | 90.0 | 97.1 | 98.2 | 90.7 | 88.5 | n/a | 84.6 |

### 4.4 消融研究

**总体消融。**我们验证含噪瓶颈（NB）、线性注意力（LA）、宽松约束（LC）和宽松损失（LL）的有效性。第一行已很强的基线为不含噪声的 MLP 瓶颈、Softmax 注意力、密集逐层监督及全局余弦损失的 Dinomaly，非常类似 ViTAD [64] 和 RD4AD [10] 的 ViT 版本。MVTec-AD 与 VisA 结果分别见表 3、A4。NB 和 LL 可以直接提升性能；存在 NB 时，LA 和 LC 进一步提升。LC 单独使用没有益处，因为没有注入噪声时，它会使重建过于容易。

**表 3：MVTec-AD 上 Dinomaly 各要素消融（%）。**NB 为含噪瓶颈；LA 为线性注意力；LC 为双组宽松约束；LL 为宽松损失。MVTec-AD 已近饱和，VisA 结果另见表 A4。—表示未采用。

| NB | LA | LC | LL | I-AUROC | I-AP | I-F1-max | P-AUROC | P-AP | P-F1-max | P-AUPRO |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| — | — | — | — | 98.41 | 99.09 | 97.41 | 97.18 | 62.96 | 63.82 | 92.95 |
| ✓ | — | — | — | 99.06 | 99.54 | 98.31 | 97.62 | 66.22 | 66.70 | 93.71 |
| — | ✓ | — | — | 98.54 | 99.21 | 97.62 | 97.20 | 62.94 | 63.73 | 93.09 |
| — | — | ✓ | — | 98.35 | 99.04 | 97.43 | 97.10 | 61.05 | 62.73 | 92.60 |
| — | — | — | ✓ | 99.03 | 99.45 | 98.19 | 97.62 | 64.10 | 64.96 | 93.34 |
| ✓ | ✓ | — | — | 99.27 | 99.62 | 98.63 | 97.85 | 67.36 | 67.33 | 94.16 |
| ✓ | — | ✓ | — | 99.50 | 99.72 | 98.87 | 98.14 | 68.16 | 68.24 | 94.23 |
| ✓ | — | ✓ | ✓ | 99.52 | 99.73 | 98.92 | 98.20 | 68.25 | 68.34 | 94.17 |
| ✓ | ✓ | ✓ | — | 99.57 | 99.78 | 99.00 | 98.20 | 67.93 | 68.21 | 94.50 |
| ✓ | ✓ | ✓ | ✓ | 99.60 | 99.78 | 99.04 | 98.35 | 69.29 | 69.17 | 94.79 |

**模型可扩展性。**既有工作 [10, 60, 65] 报告异常检测网络不遵循“扩展规律”。例如 RD4AD [10] 发现 WideResNet50 编码器优于 WideResNet101；ViTAD [65] 发现 ViT-Small 优于 ViT-Base。相反，表 4 表明 Dinomaly 受益于扩展。配备 ViT-Small 已达到先进水平，ViT-Large 则创造更高纪录。这使用户可按场景计算资源选择适当模型规模。与其他方法计算成本比较见表 A11。训练计划也可扩展，在不增加推理成本的情况下提高性能，如图 1 的 Dinomaly↑。

**表 4：MVTec-AD 上 ViT 模型规模扩展（%）。**Im/s 为每秒图像吞吐量，在 NVIDIA RTX3090、批量 16 下测量；VisA、Real-IAD 见表 A3。†默认。

| 架构 | 参数量 | MACs | Im/s | I-AUROC | I-AP | I-F1-max | P-AUROC | P-AP | P-F1-max | P-AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Small | 37.4M | 26.3G | 153.6 | 99.26 | 99.67 | 98.72 | 98.07 | 68.29 | 67.78 | 94.36 |
| ViT-Base† | 148.0M | 104.7G | 58.1 | 99.60 | 99.78 | 99.04 | 98.35 | 69.29 | 69.17 | 94.79 |
| ViT-Large | 275.3M | 413.5G | 24.2 | 99.77 | 99.92 | 99.45 | 98.54 | 70.53 | 70.04 | 95.09 |

**输入可扩展性。**Dinomaly 与输入图像较小的既有方法比较，可能看似不公平，但表 5 表明，提高它们的输入尺寸不仅无益，反而损害性能，尤其图像级检测。因此遵循常见“各取最优”的比较策略。相反，Dinomaly 的异常定位受益于输入尺寸扩大，小输入时仍保持先进性能。细节见附录表 A2。

**表 5：MVTec-AD 上输入尺寸扩展（%）。**†默认。对比方法增大输入后性能退化。每行图像级依次为 AUROC/AP/F1-max，像素级依次为 AUROC/AP/F1-max/AUPRO。

| 方法 | 输入尺寸 | 图像级 | 像素级 |
| --- | ---: | --- | --- |
| RD4AD | $256^2$† | 94.6/96.5/96.1 | 96.1/48.6/53.8/91.1 |
| RD4AD | $320^2$ | 93.2/96.9/95.6 | 95.7/55.1/57.5/91.1 |
| RD4AD | $384^2$ | 91.9/96.2/95.0 | 94.9/52.1/55.3/90.8 |
| ReContrast | $256^2$† | 98.3/99.4/97.6 | 97.1/60.2/61.5/93.2 |
| ReContrast | $320^2$ | 98.2/99.2/97.5 | 96.8/61.8/62.6/93.3 |
| ReContrast | $384^2$ | 95.2/98.0/96.4 | 96.5/57.7/59.5/92.6 |
| Dinomaly | $280^2$ | 99.6/99.8/99.3 | 98.2/65.2/66.3/93.6 |
| Dinomaly | $336^2$ | 99.6/99.8/99.2 | 98.3/67.2/67.8/94.2 |
| Dinomaly | $392^2$† | 99.6/99.8/99.0 | 98.4/69.3/69.2/94.8 |

> 译注：表 5 中 RD4AD 默认行图像级第三项为 96.1，而表 1 同方法同设定的 I-F1-max 为 95.2。此处保留原表数值，未修订。

**ViT 基础模型。**我们广泛研究 DeiT [50]、MAE [19]、D-iGPT [44]、MOCOv3 [6]、DINO [4]、iBot [69]、DINOv2 [39]、DINOv2-R [7] 等预训练基础模型的影响。图 5 表明，Dinomaly 对主干选择鲁棒，几乎全部基础模型都能达到先进水平，图像 AUROC 超过 98%。唯一明显例外为 MAE，先前也报告其不经微调时，在 kNN、线性探测等无监督任务上较弱 [39]。不同主干预训练分辨率不同，因此最优输入尺寸也不同。有趣的是，异常检测性能与基础模型的 ImageNet 线性探测准确率（冻结主干，仅调整线性分类器）强相关，提示未来只需采用更强基础模型即可继续提升。详细结果与分析见附录及表 A1。

**图 5：采用不同 ViT 基础模型的 Dinomaly 图像级 AUROC，以及这些基础模型在 ImageNet 上的线性探测准确率。**MIM 为掩码图像建模，CL 为对比学习。左纵轴为 MVTec-AD I-AUROC，右纵轴为 ImageNet 线性探测准确率；横轴模型和输入分别为 DeiT/$448^2$、MAE/$224^2$、D-iGPT/$224^2$、MOCOv3/$224^2$、DINO/$224^2$、iBOT/$224^2$、DINOv2/$392^2$、DINOv2-R/$392^2$。下方预训练类型分别标注监督、MIM、CL、CL＋MIM；虚线参照为 UniAD 和此前最佳水平。

更多实验与结果详见附录 C，涵盖不同预训练基础模型、各组件消融、超参数优化及其他深入分析。

## 5 结论

本文提出极简 UAD 框架 Dinomaly，解决 MUAD 模型性能不足的问题。四个关键要素——基础 Transformer、含噪 MLP 瓶颈、线性注意力和宽松重建——无需花哨模块或技巧，即可提升挑战性 MUAD 下的性能。MVTec AD、VisA、Real-IAD 上的大量实验展示其优于以往统一多类别模型，甚至优于近期逐类别模型，表明在复杂场景部署统一模型而不发生严重性能退化是可行的。

## 致谢

作者感谢中国国家自然科学基金（U22A2051、82027807）、国家重点研发计划（2022YFC2405200）、清华—佛山创新专项基金（2021THFS0104），以及清华大学智能医疗研究院（2022ZLB001）的支持。


## 参考文献

[1] Samet Akcay, Amir Atapour-Abarghouei, and Toby P Breckon. 《GANomaly：通过对抗训练进行半监督异常检测》 载于 Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6, 2018, Revised Selected Papers, Part III 14 , 页 622–637. Springer, 2019. 1

[2] Paul Bergmann, Sindy Löwe, Michael Fauser, David Sattlegger, and Carsten Steger. 《通过将结构相似性用于自编码器改善无监督缺陷分割》arXiv 预印本 arXiv:1807.02011, 2018. 1

[3] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. 《MVTec AD：用于无监督异常检测的全面真实世界数据集》 载于 Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 页 9592–9600, 2019. 1, 2, 6, 7

[4] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. 《自监督视觉 Transformer 中涌现的性质》 载于 Proceedings of the IEEE/CVF international conference on computer vision, 页 9650–9660, 2021. 3, 8, 2

[5] Xinlei Chen and Kaiming He. 《探索简单的孪生表示学习》 载于 Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 页 15750–15758, 2021. 3

[6] Xinlei Chen, Saining Xie, and Kaiming He. 《训练自监督视觉 Transformer 的实证研究》 InProceedings of the IEEE/CVF international conference on computer vision, 页 9640–9649, 2021. 3, 8, 2

[7] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. 《视觉 Transformer 需要寄存器》 arXiv 预印本 arXiv:2309.16588, 2023. 3, 6, 8, 1, 2

[8] Thomas Defard, Aleksandr Setkov, Angelique Loesch, and Romaric Audigier. 《PaDiM：用于异常检测与定位的图块分布建模框架》 载于 International Conference on Pattern Recognition, 页 475–489. Springer, 2021. 1

[9] David Dehaene and Pierre Eline. 《通过建模感知特征进行异常定位》 arXiv 预印本 arXiv:2008.05369, 2020. 1

[10] Hanqiu Deng and Xingyu Li. 《通过单类嵌入反向蒸馏进行异常检测》 载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 页 9737–9746, 2022. 5, 6, 7, 1, 8, 9, 10

[11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 《ImageNet：大规模分层图像数据库》 载于 IEEE Conference on Computer Vision and Pattern Recognition, 页 248–255, 2009. 2

[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 《一幅图像相当于 16×16 个词：大规模图像识别的 Transformer》 arXiv 预印本 arXiv:2010.11929, 2020. 2, 3

[13] Jia Guo, Shuai Lu, Lize Jia, Weihang Zhang, and Huiqi Li. 《用于医学图像无监督异常检测的编码器—解码器对比》IEEE Transactions on Medical Imaging, 2023. 1

[14] Jia Guo, Shuai Lu, Lize Jia, Weihang Zhang, and Huiqi Li. 《ReContrast：通过对比重建进行领域特定异常检测》 载于 Advances in Neural Information Processing Systems (NeurIPS) , 页 10721–10740, 2023. 2, 3, 5, 6

[15] Dongchen Han, Xuran Pan, Yizeng Han, Shiji Song, and Gao Huang. 《FLatten Transformer：采用聚焦线性注意力的视觉 Transformer》 载于 Proceedings of the IEEE/CVF International Conference on Computer Vision, 页 5961– 5971, 2023. 5

[16] Haoyang He, Jiangning Zhang, Hongxu Chen, Xuhai Chen, Zhishan Li, Xu Chen, Yabiao Wang, Chengjie Wang, and Lei Xie. 《DiAD：基于扩散的多类别异常检测框架》 arXiv 预印本 arXiv:2312.06607, 2023. 1, 3

[17] Haoyang He, Yuhu Bai, Jiangning Zhang, Qingdong He, Hongxu Chen, Zhenye Gan, Chengjie Wang, Xiangtai Li, Guanzhong Tian, and Lei Xie. 《MambaAD：探索用于多类别无监督异常检测的状态空间模型》 arXiv 预印本 arXiv:2404.06564, 2024. 1, 3, 6, 7, 8, 10

[18] Haoyang He, Jiangning Zhang, Hongxu Chen, Xuhai Chen, Zhishan Li, Xu Chen, Yabiao Wang, Chengjie Wang, and Lei Xie. 《基于扩散的多类别异常检测框架》 载于 Proceedings of the AAAI Conference on Artificial Intelligence, 页 8472–8480, 2024. 2, 6, 7, 8, 9, 10

[19] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. 《掩码自编码器是可扩展的视觉学习者》 载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 页 16000– 16009, 2022. 3, 8, 2, 5

[20] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 《蒸馏神经网络中的知识》 arXiv 预印本 arXiv:1503.02531, 2015. 5

[21] Geoffrey E Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R Salakhutdinov. 《通过防止特征检测器协同适应改善神经网络》arXiv 预印本 arXiv:1207.0580, 2012. 4

[22] Chaoqin Huang, Haoyan Guan, Aofan Jiang, Ya Zhang, Michael Spratling, and Yan-Feng Wang. 《基于配准的少样本异常检测》 载于 European Conference on Computer Vision, 页 303–319. Springer, 2022. 7

[23] Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, and Onkar Dabeer. 《WinCLIP：零／少样本异常分类与分割》 arXiv 预印本 arXiv:2303.14814, 2023. 7

[24] Stepan Jezek, Martin Jonak, Radim Burget, Pavel Dvorak, and Milos Skotak. 《基于深度学习的金属零件缺陷检测：复杂条件下现有方法的评价》 载于 2021 13th International Congress on Ultra Modern Telecommunications and Control Systems and Workshops (ICUMT), 页 66–71. IEEE, 2021. 7, 6

[25] Xi Jiang, Jianlin Liu, Jinbao Wang, Qiang Nie, Kai Wu, Yong Liu, Chengjie Wang, and Feng Zheng. 《SoftPatch：含噪数据的无监督异常检测》 Advances in Neural Information Processing Systems , 35:15433–15445, 2022. 6, 7

[26] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc ¸ois Fleuret. 《Transformer 是循环神经网络：采用线性注意力的快速自回归 Transformer》 载于 International conference on machine learning, 页 5156–5165. PMLR, 2020. 5

[27] Sungwook Lee, Seunghyun Lee, and Byung Cheol Song. 《CFA：面向目标异常定位的耦合超球面特征适配》 IEEE Access , 10: 78446–78454, 2022. 1

[28] Yujin Lee, Harin Lim, and Hyunsoo Yoon. 《Selformaly：迈向任务无关的统一异常检测》 arXiv 预印本 arXiv:2307.12540, 2023. 3

[29] Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, and Tomas Pfister. 《CutPaste：用于异常检测与定位的自监督学习》 载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 页 9664–9674, 2021. 1

[30] Chen Liang, Jiahui Yu, Ming-Hsuan Yang, Matthew Brown, Yin Cui, Tuo Zhao, Boqing Gong, and Tianyi Zhou. 《多模态基础模型的逐模块自适应蒸馏》 Advances in Neural Information Processing Systems, 36, 2024. 5

[31] Jiangqi Liu and Feng Wang. 《用于多类别工业异常检测的混合注意力自编码器》 arXiv 预印本 arXiv:2309.12700, 2023. 2

[32] Wenqian Liu, Runze Li, Meng Zheng, Srikrishna Karanam, Ziyan Wu, Bir Bhanu, Richard J Radke, and Octavia Camps. 《迈向变分自编码器的视觉解释》 InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 页 8642–8651, 2020. 1

[33] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. 《Swin Transformer：采用移位窗口的分层视觉 Transformer》 InIEEE/CVF International Conference on Computer Vision , 页 10012–10022, 2021. 3

[34] Zhikang Liu, Yiming Zhou, Yuansheng Xu, and Zilei Wang. 《SimpleNet：用于图像异常检测与定位的简单网络》 arXiv 预印本 arXiv:2303.15140, 2023. 6, 7, 1, 8, 9, 10

[35] Ilya Loshchilov and Frank Hutter. 《解耦权重衰减正则化》 arXiv 预印本 arXiv:1711.05101, 2017. 6

[36] Ruiying Lu, YuJie Wu, Long Tian, Dongsheng Wang, Bo Chen, Xiyang Liu, and Ruimin Hu. 《用于多类别无监督异常检测的分层向量量化 Transformer》 arXiv 预印本 arXiv:2310.14228, 2023. 1, 2, 4

[37] Amira Ben Mabrouk and Ezzeddine Zagrouba. 《智能视频监控系统中的异常行为识别综述》 Expert Systems with Applications, 91:480– 491, 2018. 1

[38] Pankaj Mishra, Riccardo Verk, Daniele Fornasier, Claudio Piciarelli, and Gian Luca Foresti. 《VT-ADL：用于图像异常检测与定位的视觉 Transformer 网络》 载于 2021 IEEE 30th International Symposium on Industrial Electronics (ISIE), 页 01–06. IEEE, 2021. 7, 6

[39] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. 《DINOv2：在无监督下学习鲁棒视觉特征》arXiv 预印本 arXiv:2304.07193, 2023. 3, 8, 2

[40] Z Peng, L Dong, H Bao, Q Ye, and F Wei. 《BEiT v2：采用向量量化视觉标记器的掩码图像建模》 arXiv 预印本 arXiv:2208.06366, 2022. 3, 2

[41] Sashank J Reddi, Satyen Kale, and Sanjiv Kumar. 《Adam 的收敛性及其他问题》 arXiv 预印本 arXiv:1904.09237, 2019. 6, 1

[42] Tal Reiss, Niv Cohen, Liron Bergman, and Yedid Hoshen. 《PANDA：适配预训练特征以进行异常检测与分割》 载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 页 2806–2814, 2021. 7

[43] Tal Reiss, Niv Cohen, Eliahu Horwitz, Ron Abutbul, and Yedid Hoshen. 《异常检测需要更好的表示》 载于 European Conference on Computer Vision, 页 56–68. Springer, 2022. 3, 7

[44] Sucheng Ren, Zeyu Wang, Hongru Zhu, Junfei Xiao, Alan Yuille, and Cihang Xie. 《使 Image-GPT 重新成为强大的视觉表示学习器》 arXiv 预印本 arXiv:2312.02147, 2023. 8, 2

[45] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Sch¨olkopf, Thomas Brox, and Peter Gehler. 《迈向工业异常检测的全面召回》 载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 页 14318–14328, 2022. 7, 1, 3

[46] Mohammadreza Salehi, Niousha Sadjadi, Soroosh Baselizadeh, Mohammad H Rohban, and Hamid R Rabiee. 《用于异常检测的多分辨率知识蒸馏》 载于 Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 页 14902–14912, 2021. 5, 1

[47] Thomas Schlegl, Philipp Seeb ¨ock, Sebastian M Waldstein, Georg Langs, and Ursula Schmidt-Erfurth. 《f-AnoGAN：采用生成对抗网络的快速无监督异常检测》 Medical image analysis, 54:30–44, 2019. 1

[48] Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, and Hongsheng Li. 《高效注意力：具有线性复杂度的注意力》 载于 Proceedings of the IEEE/CVF winter conference on applications of computer vision, 页 3531– 3539, 2021. 5

[49] Shelly Sheynin, Sagie Benaim, and Lior Wolf. 《用于少样本异常检测的分层变换判别生成模型》 载于 Proceedings of the IEEE/CVF International Conference on Computer Vision, 页 8495–8504, 2021. 1

[50] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv’e J’egou. 《通过注意力训练数据高效的图像 Transformer 并进行蒸馏》 arXiv 预印本 arXiv:2012.12877, 2021. 8, 2

[51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 《注意力就是你所需要的全部》 载于 Advances in Neural Information Processing Systems, 页 5998–6008, 2017. 2

[52] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 《利用去噪自编码器提取与组合鲁棒特征》 载于 Proceedings of the 25th international conference on Machine learning , 页 1096–1103, 2008. 4

[53] Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. 《堆叠去噪自编码器：采用局部去噪准则在深层网络中学习有用表示》 载于 Journal of Machine Learning Research, 页 3371–3408, 2010. 4

[54] Chengjie Wang, Wenbing Zhu, Bin-Bin Gao, Zhenye Gan, Jianning Zhang, Zhihao Gu, Shuguang Qian, Mingang Chen, and Lizhuang Ma. 《Real-IAD：用于通用工业异常检测基准评测的真实世界多视角数据集》arXiv 预印本 arXiv:2403.12580, 2024. 2, 6, 7, 3

[55] Guodong Wang, Shumin Han, Errui Ding, and Di Huang. 《用于异常检测的学生—教师特征金字塔匹配》 载于 The British Machine Vision Conference (BMVC), 2021. 1

[56] Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer, Ari Morcos, Ali Farhadi, and Ludwig Schmidt. 《大规模视觉语言模型的稳定低精度训练》 Advances in Neural Information Processing Systems, 36: 10271–10298, 2023. 6, 1

[57] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. 《SimMIM：掩码图像建模的简单框架》 载于 Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 页 9653–9663, 2022. 3

[58] Jie Yang, Yong Shi, and Zhiquan Qi. 《DFR：用于无监督异常分割的深度特征重建》 arXiv 预印本 arXiv:2012.07122, 2020. 5, 1

[59] Haonan Yin, Guanlong Jiao, Qianhui Wu, Borje F Karlsson, Biqing Huang, and Chin Yew Lin. 《LAFITE：采用特征编辑进行无监督多类别异常检测的潜在扩散模型》 arXiv 预印本 arXiv:2307.08059, 2023. 1, 2, 3, 4

[60] Zhiyuan You, Lei Cui, Yujun Shen, Kai Yang, Xin Lu, Yu Zheng, and Xinyi Le. 《多类别异常检测的统一模型》 arXiv 预印本 arXiv:2206.03687, 2022. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

[61] Weihao Yu, Chenyang Si, Pan Zhou, Mi Luo, Yichen Zhou, Jiashi Feng, Shuicheng Yan, and Xinchao Wang. 《面向视觉的 MetaFormer 基线》 IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 4

[62] Vitjan Zavrtanik, Matej Kristan, and Danijel Skočaj. 《DRAEM：用于表面异常检测的判别式训练重建嵌入》 载于 Proceedings of the IEEE/CVF International Conference on Computer Vision, 页 8330– 8339, 2021. 4, 1

[63] Vitjan Zavrtanik, Matej Kristan, and Danijel Skocaj. 《通过修补重建进行视觉异常检测》 Pattern Recognition, 112:107706, 2021. 1

[64] Dingwen Zhang, Guohai Huang, Qiang Zhang, Jungong Han, Junwei Han, Yizhou Wang, and Yizhou Yu. 《探索多模态磁共振图像脑肿瘤分割的任务结构》 IEEE Transactions on Image Processing, 29:9032–9043, 2020. 7

[65] Jiangning Zhang, Xuhai Chen, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, Ming-Hsuan Yang, and Dacheng Tao. 《探索普通 ViT 重建以进行多类别无监督异常检测》 arXiv 预印本 arXiv:2312.07495, 2023. 3, 5, 6, 7

[66] Jiangning Zhang, Haoyang He, Zhenye Gan, Qingdong He, Yuxuan Cai, Zhucun Xue, Yabiao Wang, Chengjie Wang, Lei Xie, and Yong Liu. 《ADER：多类别视觉异常检测的综合基准》 arXiv 预印本 arXiv:2406.03262, 2024. 7, 2, 6

[67] Xuan Zhang, Shiyu Li, Xi Li, Ping Huang, Jiulong Shan, and Ting Chen. 《DeSTSeg：用于异常检测的分割引导去噪学生—教师模型》 载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 页 3914–3923, 2023. 4, 6, 1, 7, 8, 9, 10

[68] Ying Zhao. 《OmniAL：用于无监督异常定位的统一卷积神经网络框架》 载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 页 3924–3933, 2023. 1, 3

[69] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. 《iBOT：采用在线标记器的图像 BERT 预训练》 arXiv 预印本 arXiv:2111.07832 , 2021. 3, 8, 2

[70] Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. 《用于异常检测与分割的找不同自监督预训练》 载于 European Conference on Computer Vision , 页 392–408. Springer, 2022. 2, 6, 7, 3
