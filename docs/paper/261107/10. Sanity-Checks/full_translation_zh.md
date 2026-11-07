# 显著图的合理性检验

**原文：**[Sanity Checks for Saliency Maps](<Sanity Checks for Saliency Maps.pdf>)。本译文对应本目录 11 页 NeurIPS 2018 会议论文 PDF。翻译日期：2026-11-07。

**作者：**Julius Adebayo∗、Justin Gilmer♯、Michael Muelly♯、Ian Goodfellow♯、Moritz Hardt♯†、Been Kim♯。

**单位与联系：**♯Google Brain；†University of California Berkeley；juliusad@mit.edu；{gilmer,muelly,goodfellow,mrtz,beenkim}@google.com。∗本研究在 Google AI Residency Program 期间完成。

**翻译说明：**覆盖本地 PDF 全部 11 页，包括摘要、正文第 1—6 节、图 1—6 图注和关键标注、脚注、致谢与参考文献 [1]—[36]。合并跨行、跨页续段，去除运行页眉页脚。图形不重绘，请结合原 PDF 阅读。原 PDF 多处指向未随此 PDF 收录的附录，并保留未解析图号“??”；译文照录并明确标注。这里的完整覆盖限于本地 11 页 PDF，不包含未提供的外部补充材料。公式在原文中均未编号，不另造编号。

## 摘要

显著性方法已经成为一种流行工具，用于突出输入中被认为与学习模型预测相关的特征。研究者提出了多种显著性方法，其设计往往以图像数据上的视觉吸引力为指导。本文提出一种可操作的方法论，用于评估某种方法能够和不能够提供哪些解释。我们发现，仅依赖视觉评估可能产生误导。通过大量实验，我们证明现有的一些显著性方法既独立于模型，也独立于数据生成过程。因此，未能通过所提检验的方法，不适合用于对数据或模型敏感的任务，例如发现数据离群点、解释模型所学到的输入输出关系，以及调试模型。我们借助图像边缘检测来解释这些发现，因为边缘检测既不需要训练数据，也不需要模型。针对线性模型和单层卷积神经网络的理论分析支持了我们的实验发现。²

²复现这些发现的全部代码将发布于 <https://goo.gl/hBmhDt>。

## 1 引言

随着机器学习的复杂度和影响力不断增长，人们寄希望于解释方法，希望借此阐明学习模型的重要方面 [1, 2]。解释有可能帮助满足监管要求 [3]、帮助实践者调试模型 [4, 5]，还可能揭示模型所学到的偏见或其他非预期影响 [6, 7]。显著性方法³是一类越来越流行的工具，旨在突出输入（通常是一幅图像）中的相关特征。尽管该领域令人振奋，近期也涌现了大量贡献 [8—21]，解释机器学习模型的这项有价值工作仍面临方法论挑战：很难评估模型解释的适用范围和质量。缺乏有原则的指导，使实践者在众多竞争方法之间难以抉择。

我们提出一种基于随机化检验、可实际操作的方法论，用于评价解释方法是否充分。我们以神经网络图像分类中的多种显著性方法实例化这一分析；然而，这套方法论原则上适用于任何解释方法。关键在于，所提随机化检验易于实现，有助于评估某种解释方法是否适合当前任务。

我们在广泛实验中，将该方法论应用于许多已有显著性方法、模型架构和数据集。令人意外的是，一些得到广泛应用的显著性方法既独立于模型所用的训练数据，也独立于模型参数。因此，这些方法无法帮助完成依赖模型的任务，例如模型调试，也无法帮助完成依赖数据中输入输出关系的任务。

³这里指旨在解释已训练模型的广义可视化与归因方法。这些方法常用于解释深度神经网络，尤其是图像数据上的网络。

**图 1：若干常见方法的显著图与边缘检测器的比较。**图中给出了在 ImageNet 上训练的 Inception v3 模型对 3 个输入的显著性掩码。边缘检测器的输出与某些显著性方法的输出极其相似。事实上，边缘检测器也会生成掩码，突出那些看似与模型类别预测相关的特征。我们发现，与边缘检测器最相似的方法（SSIM 指标见附录），即 Guided Backprop 及其变体，对我们的随机化检验只有极弱的敏感性。图中各列依次为原始图像、梯度、SmoothGrad、Guided BackProp、Guided GradCAM、积分梯度、积分梯度 SmoothGrad、梯度乘输入、边缘检测器；各行为 Junco 鸟、玉米、浅麦色㹴犬。

图 1 将标准显著性方法的输出与边缘检测器的输出进行比较，以说明这一点。边缘检测器不依赖模型或训练数据，却能产生与显著图在视觉上相似的结果。这表明，判断解释是否对底层模型和数据敏感时，视觉检查并不是可靠指南。

我们的方法论源于统计随机化检验：比较自然实验与人为随机化实验。我们重点研究一般框架的两种实例：模型参数随机化检验和数据随机化检验。

**模型参数随机化检验**比较显著性方法在已训练模型上的输出，以及同一方法在架构相同但随机初始化、未经训练网络上的输出。如果显著性方法依赖于模型学到的参数，那么我们应当期望两者输出显著不同。相反，如果输出相似，就可以推断显著图对模型的某些性质不敏感，此处即对模型参数不敏感。特别是，这样的显著图输出无法帮助完成模型调试之类必然依赖模型参数的任务。

**数据随机化检验**比较某种显著性方法应用于带标签数据集训练的模型时的结果，以及应用于相同架构、但在所有标签经过随机置换的数据副本上训练的模型时的结果。如果显著性方法依赖数据标签，我们同样应当期望两种情况的输出显著不同。反之，对置换标签不敏感，就表明该方法并不依赖原始数据中实例（如图像）与标签之间的关系。

更一般地说，任何解释方法都具有一组不变性，即某些数据或模型变换不会改变该方法的输出。如果发现某个不变性与当前任务要求不相容，我们就可以有把握地排除这种方法。因此，可将这些检验看作将方法用于实践之前应进行的合理性检验。

**我们的贡献**

1. 提出两种具体且易于实施的检验，评估解释方法的适用范围和质量：模型参数随机化检验与数据随机化检验。这些检验广泛适用于解释方法。
2. 在不同数据集和模型架构上，对多种解释方法开展大量实验，一致发现部分被测方法既独立于模型参数，也独立于模型训练数据的标签。
3. 在被测方法中，Gradients 和 GradCAM 通过合理性检验，而 Guided BackProp 和 Guided GradCAM 未通过。其他情形则出现视觉感知与排序相分离的现象，我们在结果中予以说明。
4. 因此，我们的发现意味着，未通过这些检验的显著性方法无法支持那些要求解释忠实于模型或数据生成过程的任务。
5. 通过对线性模型、简单单层卷积求和池化架构的系列分析，以及与边缘检测器的比较，解释上述发现。

## 2 方法与相关工作

在形式化设定中，输入是向量 $x\in\mathbb R^d$。模型描述函数 $S:\mathbb R^d\to\mathbb R^C$，其中 $C$ 是分类问题的类别数。解释方法给出映射 $E:\mathbb R^d\to\mathbb R^d$，将输入映射为与输入形状相同的对象。

下面简要介绍我们考察的部分解释方法。补充材料中提供了更深入的概述。我们的目标不是穷尽评估所有既有解释方法，而是说明检验方法如何应用于若干值得关注的情况。

输入 $x$ 的**梯度解释**为 $E_{\mathrm{grad}}(x)=\partial S/\partial x$ [8, 22, 23]。梯度量化了在输入附近的小邻域中，每一输入维度的变化会使预测 $S(x)$ 改变多少。

**梯度⊙输入。**另一种解释是输入与梯度的逐元素乘积，记作 $x\odot\frac{\partial S}{\partial x}$。它可以应对“梯度饱和”，并减少视觉弥散 [13]。

**积分梯度（IG）**也通过累加输入的不同缩放版本来处理梯度饱和 [14]。输入 $x$ 的 IG 定义为

$$E_{\mathrm{IG}}(x)=(x-\bar x)\times\int_0^1\frac{\partial S(\bar x+\alpha(x-\bar x))}{\partial x}\,d\alpha,$$

其中 $\bar x$ 是“基线输入”，表示原始输入 $x$ 中某一特征的缺失。

**引导反向传播（GBP）**[9] 建立在“DeConvNet”解释方法 [10] 基础上，对应于这样一种梯度解释：反向传播通过 ReLU 单元时，将负梯度项置零。

**Guided GradCAM。**Selvaraju 等 [19] 提出的 GradCAM 解释对应于类别得分（logit）相对于 DNN 最后一个卷积单元特征图的梯度。为获得像素级粒度，可通过逐元素乘积将 GradCAM 与 Guided Backpropagation 结合。

**SmoothGrad（SG）**[16] 通过对输入含噪副本的解释取平均，减轻显著图的噪声和视觉弥散 [14, 13]。给定解释映射 $E$，其定义为

$$E_{\mathrm{sg}}(x)=\frac1N\sum_{i=1}^N E(x+g_i),\qquad g_i\sim\mathcal N(0,\sigma^2),$$

其中噪声向量 $g_i$ 从正态分布独立同分布采样。

### 2.1 相关工作

**其他方法与相似性。**除梯度方法外，还有方法针对模型的每个样本“学习”解释 [20, 17, 12, 15, 11, 21]。近期，M. Ancona [24] 表明，对采用零基线且无偏置的 ReLU 网络，$\epsilon$-LRP 与 DeepLift（Rescale）解释等价于输入⊙梯度。类似地，Lundberg 和 Lee [18] 提出 SHAP 解释，近似 Shapley 值，并统一若干已有方法。

**脆弱性。**Ghorbani 等 [25] 和 Kindermans 等 [26] 都提出针对显著性方法的攻击，表明可以用非预期方式操纵所得解释。Nie 等 [27] 对基于反向传播的方法作了理论评估，发现 Guided BackProp 和 DeconvNet 在特定条件下对网络重新参数化不变，尤其包括随机高斯初始化。他们特别指出，这两种方法似乎都在执行部分输入恢复。我们对 Guided BackProp 及其变体也得到类似发现。此外，我们工作的不同之处在于提出了可操作的合理性检验，用于评估解释方法。类似地，Mahendran 和 Vedaldi [28] 也表明，一些基于反向传播的显著性方法缺乏神经元区分能力。

**现有评估方法。**Samek 等 [29] 和 Montavon 等 [30] 都提出用输入扰动程序评估显著性方法质量。Dabkowski 和 Gal [17] 提出基于熵的指标，量化解释掩码捕获的相关信息量。显著图在目标定位任务上的表现也被用作评估标准。Montavon 等 [30] 讨论将解释的连续性与选择性用作评估指标。

**随机化。**我们的标签随机化检验受 Zhang 等 [31] 启发，但用途完全不同。

### 2.2 可视化与相似性指标

我们介绍可视化方式，并概述评估两个解释之间相似性所用的指标。

**可视化。**显著图采用两种可视化方式。第一种为绝对值（ABS）：对归一化⁴后的映射取绝对值。第二种为发散式可视化：保持映射原样，以不同颜色表示正、负重要性。

**相似性指标。**定量比较采用以下指标：取绝对值后的 Spearman 秩相关、不取绝对值的 Spearman 秩相关（发散式）、结构相似性指数（SSIM），以及两个映射的梯度直方图（HOG）之间的 Pearson 相关。在 ImageNet 样本上，我们不取绝对值计算 SSIM 和 HOG 相似性。⁵这些指标涵盖广义相似性；然而，量化人类视觉感知仍是活跃研究领域。

⁴我们把映射归一化至 $[-1.0,1.0]$。这种归一化可能忽略某些显著性方法的特性。例如，积分梯度的归因值之和等于输出值。这一性质通常无法可视化。我们认为，这类性质不会改变人们感知输出可视化的方式。

⁵这些指标的校准讨论见附录。

## 3 模型参数随机化检验

模型参数编码模型在训练过程中从数据中学到的内容，并决定测试集性能。因此，显著性方法若要用于模型调试，就应当对模型参数敏感。

举例来说，考虑线性函数 $f(x)=w_1x_1+w_2x_2$，输入 $x\in\mathbb R^2$。模型对输入 $x$ 的行为的梯度解释由参数值 $(w_1,w_2)$ 给出，它们对应函数对各坐标的敏感性。因此，改变模型参数会改变解释。

所提模型参数随机化检验评估解释方法对模型参数的敏感性。我们进行两类随机化。第一类将模型所有权重随机重新初始化，既包含整体重新初始化，也包含级联方式。第二类每次独立随机化一层，其他层保持不变。两种情况下，都比较随机权重网络的解释与原始权重模型的解释。

### 3.1 级联随机化

**概述。**级联随机化从顶层开始，逐步向下直到最底层随机化模型权重。该过程从顶层到底层破坏学到的权重。图 2 可视化了多种显著性方法的级联随机化；图 3、4 展示 Spearman 指标、SSIM 和 HOG 相似性。

**梯度表现出敏感性，而 Guided BackProp 具有不变性。**梯度图对模型参数敏感，GradCAM 掩码也表现出敏感性。另一方面，在所有架构和数据集上，无论模型如何退化，Guided BackProp 和 Guided GradCAM 都没有变化。

**图 2：Inception v3（ImageNet）上的级联随机化。**第一列展示 Junco 鸟的原始解释。从左至右，各列表示直到对应“块”（包括该块）的网络权重及其他可训练变量已全部随机化。图中展示 17 个随机化块。坐标（Gradient, mixed_7b）表示从 Logits 顶层直到 mixed_7b 都已重新初始化的网络的梯度解释。最后一列对应权重全部重新初始化的网络。图内行标：梯度、梯度-SG、梯度⊙输入、引导反向传播、GradCAM、积分梯度、积分梯度-SG、Guided GradCAM；列标依次为原始解释、logits、mixed_7c、mixed_7b、mixed_7a、mixed_6e、mixed_6d、mixed_6c、mixed_6b、mixed_6a、mixed_5d、mixed_5c、mixed_5b、conv2d_4a_3x3、conv2d_3b_1x1、conv2d_2b_3x3、conv2d_2a_3x3、conv2d_1a_3x3。箭头表示从顶层到底层的级联随机化；另示原始图像。

**视觉评估的危险。**从视觉上看，积分梯度和梯度⊙输入与原始掩码极其相似。事实上，图 2 中即使多个块已经随机化，仍可辨认鸟的结构。这种视觉相似性体现在取绝对值的秩相关（图 3 上）、SSIM 和 HOG 指标（图 4）中。然而，重新初始化会扰乱映射符号，因而顶层一经随机化，不取绝对值的 Spearman 秩相关几乎立即降至零（图 3 下）。这种视觉感知与数值排序之间的分离表明，简单观察掩码不能区分结构相似而参数迥异的网络。我们在讨论部分解释其原因。

### 3.2 独立随机化

**概述。**作为另一种模型参数随机化检验，我们独立逐层随机化，目的是分离解释对各层的依赖。因此，可以评估显著性掩码对低层与高层权重的依赖。

**结果。**级联随机化与独立逐层随机化的结果相互对应（见附录图 ??、??、?? 和 ??）。与前述观察相同，在所有网络中，无论独立随机化哪一层，Guided Backprop 和 Guided GradCAM 掩码都几乎不变。同样，对于梯度⊙输入和积分梯度方法，输入结构在视觉上得以保留。

**图 3：级联随机化的相似性指标。**展示 Inception v3/ImageNet、CNN/Fashion MNIST 和 MLP/MNIST 的结果；MLP/Fashion MNIST 和 CNN/MNIST 见附录。所有图中，纵轴是原始解释与随机化到该层/块时所得解释的秩相关，横轴是从输出层开始的 DNN 层/块。黑色竖虚线标出连续随机化从顶层开始的位置。上：取绝对值的 Spearman 秩相关；下：不取绝对值的 Spearman 秩相关。图注说明：Inception v3/ImageNet 不取 ABS 时，IG、梯度乘输入和梯度重合；MLP/MNIST 中 IG 与梯度乘输入重合。图内 original 为原始模型，output 为输出层，hidden 为隐藏层，Conv2d/Mixed 为对应层块名称。

**图 4：级联随机化的相似性指标。**展示 Inception v3 级联随机化时，原始输入掩码与新生成掩码之间的 HOG 相似性和 SSIM。图注说明：SSIM/Inception v3/ImageNet 中，IG 与梯度⊙输入重合，而 GradCAM、Guided GradCAM 和 Guided BackProp 聚集在顶部。横轴为与图 2 对应的随机化层/块。

## 4 数据随机化检验

能否实现准确预测，取决于数据所编码的实例（如图像）与标签之间的关系。如果通过随机化标签人为打破这种关系，那么任何预测模型都无法优于随机猜测。数据随机化检验评估解释方法对实例标签关系的敏感性。对随机化标签不敏感的解释方法，不可能解释依赖数据生成过程中实例标签关系的机制。例如，将诊断随机分配给 CT 扫描后，如果解释没有变化，那么显然它原本就没有解释 CT 扫描与正确诊断之间的关系（文献 [32] 将 Guided BackProp 用作二维超声阴影检测流程的一部分）。

数据随机化检验中，我们置换训练标签，并在随机化训练数据上训练模型。模型若在随机化训练数据上获得较高训练准确率，就必须记忆随机标签，无法利用原始数据结构。Zhang 等 [31] 已表明，先进深度神经网络很容易拟合随机标签。

实验中，对每一模型数据集组合置换训练标签，并将模型训练至超过 95% 的训练集准确率。注意，测试准确率始终不优于随机猜测标签（考虑采样误差）。对每个所得模型，分别在相同测试输入集合上，计算真实标签训练模型与相应随机置换标签训练模型的解释。

**图 5：真实模型与随机标签训练模型的解释。**左上：CNN 对 MNIST 测试集数字 0 的掩码绝对值可视化。右上：同一 CNN 和数字 0 的发散色彩显著性掩码。左下：显著性方法的 Spearman 秩相关柱状图（取绝对值），比较随机标签训练模型与真实标签训练模型的解释相似性。右下：MLP 的显著性方法 Spearman 秩相关柱状图（不取绝对值）。CNN、MLP 在 Fashion MNIST 上的对应图见附录。关键标注：真实标签、随机标签、绝对值可视化、发散式可视化、取绝对值/不取绝对值的秩相关；方法包括 Gradient、Gradient-SG、Guided BackProp、GradCAM、Guided GradCAM、Integrated Gradients、Integrated Gradients-SG、Gradient⊙Input。

**梯度敏感。**我们再次发现梯度及其 SmoothGrad 变体显著改变。此外，GradCAM 掩码也发生变化，变得更不连贯。

**仅依赖视觉检查可能产生误导。**对于 Guided BackProp，我们观察到视觉变化；然而，若只简单观察图像，掩码仍会突出输入中一些看似合理的部分。例如，发散掩码（图 5 右）显示，对随机标签训练网络，Guided BackProp 掩码仍在数字的大部分区域赋予正相关性。

对于梯度⊙输入和积分梯度，我们也观察到所得掩码的视觉变化，尤其是归因符号变化。尽管如此，输入结构在掩码中仍然十分明显。这一效应对 MNIST 这种背景为零的稀疏输入尤其突出，不过在 Fashion MNIST 上也观察到类似效应（见附录）。仅凭视觉检查，分析者完全可能把随机标签训练网络产生的积分梯度和梯度⊙输入掩码误认为有效解释。

## 5 讨论

下面退一步解释这些发现。首先，讨论模型架构对神经网络解释的影响。其次，考察近似输入与梯度逐元素乘积的方法，因为若干局部解释具有这种形式 [33, 18]。我们通过实验表明，输入“结构”会主导梯度，尤其对稀疏输入。第三，借助线性模型解释梯度方法的行为。随后考察单层卷积加求和池化架构，证明该模型的显著性解释主要捕获边缘。最后回到边缘检测器，比较未通过合理性检验的方法与边缘检测器。

### 5.1 模型架构作为先验的作用

深度神经网络架构对网络产生的表征有重要影响。多项结果说明，随机初始化模型可以成为强有力的分类先验 [34, 35]。此外，在单一输入上训练随机初始化网络，无需其他训练数据，就可执行去噪、超分辨率和图像修补 [36]。这些既有工作说明，随机初始化网络也对应非平凡的表征。不依赖模型参数或训练数据的解释，仍可能依赖模型架构，因此可以提供关于架构内置先验的有用信息。但在这种情况下，解释方法只应被用于那些我们认为仅凭架构知识就足以提供有用解释的任务。

### 5.2 逐元素输入梯度乘积

一些方法，例如 $\epsilon$-LRP、DeepLift 和积分梯度，在 ReLU 这样的分段线性函数上近似输入与梯度的逐元素乘积。为进一步理解发现，我们考察输入固定而梯度随机化时，输入梯度乘积 $E(x)=x\odot\frac{\partial S}{\partial x}$ 会如何变化。为此进行如下实验：对输入 $x$ 采样两个随机向量 $u,v$（同时考虑截断正态和均匀分布），分别计算 $x\odot u$ 与 $x\odot v$。随后，随着噪声增大，使用全部考虑的指标比较两者相似性。在 ImageNet 样本上，我们发现输入确实主导乘积（见附录图 ??）。即使带噪梯度向量剧烈变化，这种主导性仍然存在。该实验表明，当梯度像通常情况那样在视觉上呈现噪声时，近似“输入乘梯度”的方法可能主要返回输入本身。

### 5.3 简单模型分析

为更好理解发现，我们分析被测显著性方法在两个简单模型上的输出：线性模型与单层求和池化卷积网络。在线性模型上，显著性方法返回的系数直观衡量模型对相应变量的敏感性；然而，应用于随机卷积时，这些方法似乎会产生类似边缘检测器的视觉伪象。

**图 6：单层求和池化卷积架构产生的解释。**展示梯度、SmoothGrad、积分梯度和 Guided BackProp 解释（相似性指标见附录）。图内 RGB 为彩色图，Gray Scale 为灰度图；Gradient、Gradient SmoothGrad、IG、GBP 分别表示上述四种方法。

**线性模型。**考虑线性模型 $f:\mathbb R^d\to\mathbb R$，定义为 $f(x)=w\cdot x$，其中 $w\in\mathbb R^d$ 为权重。对梯度有 $E_{\mathrm{grad}}(x)=\frac{\partial(w\cdot x)}{\partial x}=w$。类似地，对 SmoothGrad 有 $E_{\mathrm{sg}}(x)=w$，因为梯度独立于输入，对含噪输入的梯度求平均仍得到相同权重。这种情况下，积分梯度化为“梯度⊙输入”：

$$E_{\mathrm{IG}}(x)=(x-\bar x)\odot\int_0^1\frac{\partial f(\bar x+\alpha(x-\bar x))}{\partial x}\,d\alpha=(x-\bar x)\odot\int_0^1w\alpha\,d\alpha=(x-\bar x)\odot w/2.$$

> 译注：上式严格照录 PDF 第 8 页，包括积分内 $w\alpha$ 与末项 $w/2$。按通常积分梯度对路径点求梯度的定义，线性模型的梯度为 $w$，不会产生这一 $1/2$；原文此处与其前述定义存在不一致，未静默更正。

因此，基本梯度方法应用于线性模型时会通过合理性检验。随机模型的梯度会返回白噪声图像，而积分梯度会返回输入图像的含噪版本。这里没有考虑 Guided Backprop 和 GradCAM，因为二者都未针对上述线性模型定义。

**单层求和池化卷积模型。**下面说明，对单层卷积网络应用相同方法，可能产生具有误导性的视觉伪象，除非作进一步分析。考虑作用于灰度图像 $x\in\mathbb R^{n\times n}$ 的单层卷积网络。令 $w\in\mathbb R^{3\times3}$ 为 $3\times3$ 卷积滤波器，其元素为 $w_{ij}$，$i,j\in\{-1,0,1\}$。记 $w*x\in\mathbb R^{n\times n}$ 为卷积输出。网络输出可写为

$$l(x)=\sum_{i=1}^n\sum_{j=1}^n\sigma(w*x)_{ij},$$

其中 $\sigma$ 是逐点 ReLU 非线性。具体而言，网络先对输入使用单个 $3\times3$ 卷积滤波器，再施加 ReLU，最后对整个卷积层求和池化得到输出。其架构类似于 [34]。如图 6，不同显著性方法确实像边缘检测器一样工作。这意味着，网络的卷积结构导致了其中某些方法的边缘检测行为。

为理解这些方法应用于简单架构后为何在视觉上像边缘检测器，考察梯度 $\frac{\partial}{\partial x_{ij}}l(x)$ 的闭式形式。令 $a_{ij}=\mathbf1\{(w*x)_{ij}\ge0\}$ 表示卷积层 ReLU 单元的激活模式。对 $i,j\in[2,n-1]$，有

$$\frac{\partial}{\partial x_{ij}}l(x)=\sum_{k=-1}^1\sum_{l=-1}^1\sigma'((w*x)_{i+k,j+l})w_{kl}=\sum_{k=-1}^1\sum_{l=-1}^1a_{i+k,j+l}w_{kl}.$$

回顾：$x<0$ 时 $\sigma'(x)=0$，否则为 1。因此，像素 $x_{ij}$ 附近的 $3\times3$ 激活模式唯一决定 $\partial/\partial x_{ij}$。现在就能理解，为何显著性掩码中会出现边缘：图像中对应“边缘”的区域，与周围像素具有不同激活模式。相比之下，更均匀区域内的像素具有相同激活模式，因此 $\frac{\partial}{\partial x_{ij}}l(x)$ 也相同。类似原理可能也适用于堆叠卷积层。

### 5.4 边缘检测器的情况

粗略说来，边缘检测器是一种突出图像中剧烈过渡的经典工具。值得注意的是，它通常无需训练，也不依赖任何预测模型，只是给定输入图像的函数。与前述一些显著性方法一样，边缘检测对模型和数据变换保持不变。

图 1 中，边缘检测器生成的图像与某些显著性方法的输出极为相似。事实上，它也可以突出那些看似与模型类别预测相关的特征。但此时，人类观察者若把突出边缘解释成类别预测的解释，就有确认偏误的风险。附录图 ?? 定性比较了输入图像的显著图，以及同一输入图像与边缘检测器输出的逐元素乘积。结果确实极其相似，说明显著性方法主要利用图像边缘。

虽然边缘检测是基础且有用的图像处理技术，但通常不被认为是解释方法，因为它不涉及模型或训练数据。根据我们的发现，可以合理地将一些显著性方法理解为隐式执行类似边缘检测、分割或去噪的无监督图像处理技术。要将这些方法与对模型敏感的解释区分开，仅靠视觉检查是不够的。

## 6 结论与未来工作

我们实验方法的目标，是指导研究者评估模型解释方法的适用范围。我们希望这些方法能够成为设计新解释方法时的合理性检验。结果表明，只观察解释的视觉效果，可能会偏好那些图像令人信服，却对模型和数据生成过程缺乏敏感性的方法。

解释方法的不变性提供了一条具体途径，可以排除方法对某些任务的适用性。我们主要关注模型随机化与标签随机化下的不变性。还有许多其他变换值得研究，可以帮助理解本文评估过和未评估的方法。沿着这一方向，我们希望本文成为更严格评估新解释方法的踏脚石，而非对已有方法作最终裁决。

## 致谢

感谢 Google PAIR 团队为本文使用的方法提供开源实现。感谢 Martin Wattenberg 和 Google Brain 团队其他成员的批判性反馈，使工作得到改进。最后，感谢匿名审稿人帮助完善论文的反馈。

## 参考文献

[1] Alfredo Vellido, José David Martín-Guerrero, Paulo JG Lisboa. 使机器学习模型可解释。ESANN, 12:163–172. Citeseer, 2012。

[2] Finale Doshi-Velez, Mason Kortz, Ryan Budish, Chris Bavitz, Sam Gershman, David O’Brien, Stuart Schieber, James Waldo, David Weinberger, Alexandra Wood. 法律之下的 AI 问责：解释的作用。arXiv:1711.01134, 2017。

[3] Bryce Goodman, Seth Flaxman. 欧盟关于算法决策与“解释权”的规定。arXiv:1606.08813, 2016。

[4] Gabriel Cadamuro, Ran Gilad-Bachrach, Xiaojin Zhu. 调试机器学习模型。ICML Workshop on Reliable Machine Learning in the Wild, 2016。

[5] Jorge Casillas, Oscar Cordón, Francisco Herrera Triguero, Luis Magdalena. 模糊建模中的可解释性问题。卷 128. Springer, 2013。

[6] Himabindu Lakkaraju, Ece Kamar, Rich Caruana, Jure Leskovec. 黑盒模型的可解释且可探索近似。arXiv:1707.01154, 2017。

[7] Fulton Wang, Cynthia Rudin. 因果递降规则列表。arXiv:1510.05189, 2015。

[8] Karen Simonyan, Andrea Vedaldi, Andrew Zisserman. 深入卷积网络：可视化图像分类模型与显著图。arXiv:1312.6034, 2013。

[9] Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller. 追求简单：全卷积网络。arXiv:1412.6806, 2014。

[10] Matthew D Zeiler, Rob Fergus. 可视化并理解卷积网络。European Conference on Computer Vision, 818–833. Springer, 2014。

[11] Maximilian Alber Klaus-Robert Müller Dumitru Erhan Been Kim Sven Dähne Pieter-Jan Kindermans, Kristof T. Schütt. 学习如何解释神经网络：PatternNet 与 PatternAttribution。International Conference on Learning Representations, 2018。<https://openreview.net/forum?id=Hkn7CBaTW>。

> 译注：文献 [11] 与 [24] 的作者排列按本地 PDF 的异常排版照录；不据外部版本重新排列。

[12] Luisa M Zintgraf, Taco S Cohen, Tameem Adel, Max Welling. 可视化深度神经网络决策：预测差异分析。arXiv:1702.04595, 2017。

[13] Avanti Shrikumar, Peyton Greenside, Anna Shcherbina, Anshul Kundaje. 不只是黑盒：通过传播激活差异学习重要特征。arXiv:1605.01713, 2016。

[14] Mukund Sundararajan, Ankur Taly, Qiqi Yan. 深度网络的公理化归因。Doina Precup, Yee Whye Teh 编，Proceedings of the 34th International Conference on Machine Learning, PMLR 70:3319–3328. International Convention Centre, Sydney, Australia, 2017 年 8 月 6—11 日。<http://proceedings.mlr.press/v70/sundararajan17a.html>。

[15] Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. 为什么我应该信任你？解释任意分类器的预测。Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135–1144. ACM, 2016。

[16] Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg. SmoothGrad：通过添加噪声去除噪声。arXiv:1706.03825, 2017。

[17] Piotr Dabkowski, Yarin Gal. 面向黑盒分类器的实时图像显著性。Advances in Neural Information Processing Systems, 6970–6979, 2017。

[18] Scott M Lundberg, Su-In Lee. 解释模型预测的统一方法。Advances in Neural Information Processing Systems, 4768–4777, 2017。

[19] Ramprasaath R Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, Dhruv Batra. Grad-CAM：你为什么这么说？arXiv:1611.07450, 2016。

[20] Ruth C Fong, Andrea Vedaldi. 通过有意义扰动对黑盒作可解释说明。arXiv:1704.03296, 2017。

[21] Jianbo Chen, Le Song, Martin Wainwright, Michael Jordan. 学习解释：模型解释的信息论视角。Jennifer Dy, Andreas Krause 编，Proceedings of the 35th International Conference on Machine Learning, PMLR 80:883–892. Stockholmsmässan, Stockholm, Sweden, 2018 年 7 月 10—15 日。<http://proceedings.mlr.press/v80/chen18j.html>。

[22] Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pascal Vincent. 可视化深度网络的高层特征。University of Montreal, 1341(3):1, 2009。

[23] David Baehrens, Timon Schroeter, Stefan Harmeling, Motoaki Kawanabe, Katja Hansen, Klaus-Robert Müller. 如何解释单个分类决策。Journal of Machine Learning Research, 11(Jun):1803–1831, 2010。

[24] A. C. Öztireli M. Gross M. Ancona, E. Ceolini. 迈向更好地理解深度神经网络中基于梯度的归因方法。International Conference on Learning Representations (ICLR 2018), 2018。

[25] Amirata Ghorbani, Abubakar Abid, James Zou. 神经网络的解释是脆弱的。arXiv:1710.10547, 2017。

[26] Pieter-Jan Kindermans, Sara Hooker, Julius Adebayo, Maximilian Alber, Kristof T Schütt, Sven Dähne, Dumitru Erhan, Been Kim. 显著性方法的（不）可靠性。arXiv:1711.00867, 2017。

[27] Weili Nie, Yang Zhang, Ankit Patel. 对基于反向传播可视化中令人困惑行为的理论解释。ICML, 2018。

[28] Aravindh Mahendran, Andrea Vedaldi. 显著反卷积网络。European Conference on Computer Vision, 120–135. Springer, 2016。

[29] Wojciech Samek, Alexander Binder, Grégoire Montavon, Sebastian Lapuschkin, Klaus-Robert Müller. 评估深度神经网络所学内容的可视化。IEEE Transactions on Neural Networks and Learning Systems, 28(11):2660–2673, 2017。

[30] Grégoire Montavon, Wojciech Samek, Klaus-Robert Müller. 解释和理解深度神经网络的方法。Digital Signal Processing, 2017。

[31] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. 理解深度学习需要重新思考泛化。In Proc. 5th ICLR, 2017。

[32] Qingjie Meng, Christian Baumgartner, Matthew Sinclair, James Housden, Martin Rajchl, Alberto Gomez, Benjamin Hou, Nicolas Toussaint, Jeremy Tan, Jacqueline Matthew 等。二维超声中的自动阴影检测。2018。

[33] Marco Ancona, Enea Ceolini, Cengiz Öztireli, Markus Gross. 迈向更好地理解深度神经网络中基于梯度的归因方法。In Proc. 6th ICLR, 2018。

[34] Andrew M Saxe, Pang Wei Koh, Zhenghao Chen, Maneesh Bhand, Bipin Suresh, Andrew Y Ng. 关于随机权重和无监督特征学习。ICML, 1089–1096, 2011。

[35] Guillaume Alain, Yoshua Bengio. 使用线性分类器探针理解中间层。arXiv:1610.01644, 2016。

[36] Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky. 深度图像先验。arXiv:1711.10925, 2017。
