# 迈向工业异常检测的完全召回

**原文：**[Towards Total Recall in Industrial Anomaly Detection](<Towards Total Recall in Industrial Anomaly Detection.pdf>)。本译文对应本目录 11 页 CVPR 2022 正式论文集 PDF，原刊页码 14318—14328。翻译日期：2026-11-07。

**作者：**Karsten Roth¹˒*、Latha Pemula²、Joaquin Zepeda²、Bernhard Schölkopf²、Thomas Brox²、Peter Gehler²。

**单位：**¹图宾根大学；²Amazon AWS。* 本工作在 Amazon AWS 研究实习期间完成。

**翻译说明：**覆盖本地 PDF 的摘要、正文 1—5 节、影响与局限、致谢、图 1—6、表 1—6、算法 1、公式 (1)—(8) 和全部参考文献 [1]—[60]。正文提及的独立补充材料不包含在这份 11 页 PDF 中，因此不另行补入。合并跨页段落并去除运行页码；图形不重绘，图注与关键图内文字中文化。所有数字和公式按原文保留，疑点以译注标出。

## 摘要

发现缺陷部件的能力是大规模工业制造的关键组成部分。本文解决的一项特殊挑战是冷启动问题：仅用正常（无缺陷）示例图像拟合模型。虽然可以针对每类设计人工方案，我们的目标是构建能自动在多种不同任务上同时良好工作的系统。表现最好的方法将 ImageNet 模型嵌入与离群点检测模型结合。本文沿此方向提出 PatchCore，使用尽可能具有代表性的正常图像块特征记忆库。PatchCore 在检测与定位方面达到先进水平，同时保持有竞争力的推理时间。在具有挑战性、广泛使用的 MVTec AD 基准上，图像级异常检测 AUROC 最高为 99.6%，与第二好的竞争方法相比，误差降低一半以上。我们还在另外两个数据集及少样本场景中取得有竞争力的结果。代码：<https://github.com/amazon-research/patchcore-inspection>。

## 1 引言

检测图像中不寻常模式的能力深植于人类认知。人只需见到少量正常实例，就能区分数据中的预期变化与离群点。本文研究该问题的计算版本：用于工业图像数据视觉检查的冷启动¹异常检测。许多工业场景都存在这一问题：正常示例图像容易获取，但完整描述预期缺陷变化既昂贵又复杂。此任务自然可视为分布外检测，模型需区分来自训练数据分布的样本及其支撑集之外的样本。工业视觉缺陷分类尤其困难，缺陷既可能是细小划痕等微妙变化，也可能是部件缺失等较大结构缺陷 [5]。图 1 展示 MVTec AD 基准示例及本文方法结果。

¹ 通常也称单类分类（one-class classification，OCC）。

**图 1：MVTec 基准数据集示例。**图像上叠加 PatchCore 的分割结果。橙色边界表示真实分割图中异常的轮廓；玻璃破损、划痕、烧伤或结构变化等异常以蓝—橙渐变色表示。

现有冷启动工业视觉异常检测通过自编码方法 [12,36,44]、GAN [2,39,43] 或其他无监督适配 [42,56] 学习正常分布。近期 [4,10] 提议使用 ImageNet 分类的通用深层表示，不对目标分布作适配。尽管没有适配，这些模型的异常检测仍很强，甚至有良好的缺陷空间定位。其关键原理是测试样本与正常样本的特征匹配，同时利用深层表示的多尺度特性。高分辨率特征负责细微、细粒度缺陷分割，而更高抽象层级特征负责结构偏差及完整图像级异常检测。这种非适配方法的内在缺点是高抽象层级上的匹配置信度有限：ImageNet 训练得到的高层抽象特征与工业环境需要的抽象特征重叠很少。此外，可提取高层表示数量较少，有效限制了测试时能利用的正常上下文。

本文提出 PatchCore，通过 (1) 最大化测试时可用正常信息，(2) 降低对 ImageNet 类别的偏置，(3) 保持高速推理，有效缓解这一问题。只要有一个图像块异常，整幅图像即可判为异常 [14,56]；基于这一事实，PatchCore 使用局部聚合的中层图像块特征。中层网络图像块特征使其以高分辨率运行，同时最小化 ImageNet 类别偏置；局部邻域特征聚合又保证保留充分空间上下文，由此产生广泛的记忆库，使测试时能充分利用正常上下文。为实现实用性，PatchCore 还将贪心核心集子采样 [1] 引入正常特征库，既减少图像块记忆库冗余，又显著降低存储需求与推理时间，使其适用于真实工业场景。

在多样的 MVTec AD [5] 及专门的 Magnetic Tile Defects（MTD）[26] 工业异常检测基准上的深入实验证明其能力。它在两个数据集上取得先进的图像级检测分数，在 MVTec AD 上几乎完美（最高 99.6% AUROC），将此前方法检测误差减半以上，并取得先进工业异常定位性能。它无需在当前数据集上训练即可保持快速推理，因而对工业异常检测实用应用很有吸引力。进一步实验还表明，它只用一小部分正常训练数据就能达到现有方法性能，样本效率很高。

## 2 相关工作

多数异常检测模型依赖学习正常数据内在表示的能力，例如通过自编码模型 [44]。为更好估计正常特征分布，已有扩展采用高斯混合模型 [60]、生成对抗训练目标 [2,39,43]、对预定物理增强的不变性 [25]、隐藏特征对重新输入重建图像的鲁棒性 [29]、原型记忆库 [21]、注意力引导 [52]、结构目标 [7,59] 或受约束表示空间 [38]。也可以使用其他无监督表示学习，如 GAN [13]、预测预定几何变换 [20] 或归一化流 [42]。有了正常及新测试表示，异常检测就可转化为重建误差 [44]、到 k 个近邻的距离 [18]，或在特征上微调 OC-SVM [46]、SVDD [50,56] 等单类分类模型。多数方法自然通过逐像素重建误差完成定位，也能使用 GradCAM [47]、XRAI [28] 等显著性方法作异常分割 [42,45,52]。

**工业异常检测。**通过学习正常表示实现一般异常检测的文献很多，但工业图像有其自身挑战 [5]。从 [4] 开始，近期工作表明，无需适配当前数据，使用 ImageNet [16] 等大型外部自然图像数据集预训练模型就能达到先进检测性能。这催生了进一步复用预训练特征的方法，如 SPADE [10]：用包含多级特征的记忆库进行细粒度 kNN [18] 异常分割及图像级异常检测。类似地，PaDiM [14] 使用局部约束的特征袋方式 [8]，估计图像块级特征分布矩（均值和协方差），用于图像块级马氏距离 [33]。它与在完整图像上研究的 [40] 相似。为更好考虑自然预训练数据与工业图像的分布偏移，可以进一步适配，例如 [6,45] 的师生知识蒸馏 [24]，或在预训练网络特征上训练归一化流 [17,30,42]。

PatchCore 的具体组件与 SPADE、PaDiM 最相关。SPADE 使用预训练主干提取的正常特征记忆库，并分别处理图像级和像素级异常检测。PatchCore 同样使用记忆库，但采用具有邻域感知能力的图像块特征，这是更高性能的关键：它保留更多正常上下文，纳入更合适的归纳偏置。此外，记忆库经过核心集子采样，以更高性能实现低推理成本。核心集长期用于基础 kNN、kMeans [22] 和混合模型 [19]：找到能最好近似可用集合结构的子集，以明显更低成本得到近似解 [1,9]。近期核心集也进入深度学习，如网络剪枝 [34]、主动学习 [48]、通过增加小批次的有效数据覆盖改善 GAN 训练 [49] 或表示学习 [41]。后三者成功使用贪心核心集选择。我们同样采用贪心机制，近似记忆库的特征空间覆盖。

最后，本文图像级异常检测和异常分割均基于图像块，与 PaDiM 相关，其目标是提高异常检测敏感性。我们使用高效图像块特征记忆库，测试时所有图像块均可访问；PaDiM 则将检测限定为每个位置特定的马氏距离。这样 PatchCore 对图像对齐依赖更小，并用更大的正常上下文估计异常。此外，训练、测试图像无需像 PaDiM 那样形状相同。PatchCore 还用具有局部感知能力的图像块分数考虑局部空间变化，降低 ImageNet 类别偏置。

## 3 方法

PatchCore 包含以下部分，依次介绍：聚合到记忆库的局部图像块特征（第 3.1 节）、提升效率的核心集缩减（第 3.2 节）、最终完成检测和定位判决的完整算法（第 3.3 节）。

### 3.1 局部感知图像块特征

以 $\mathcal X_N$ 表示训练时所有正常图像，$\forall x\in\mathcal X_N:y_x=0$，其中 $y_x\in\{0,1\}$ 表示图像正常（0）或异常（1）。测试样本集合记为 $\mathcal X_T$，$\forall x\in\mathcal X_T:y_x\in\{0,1\}$。遵循 [4,10,14]，PatchCore 使用 ImageNet 预训练网络 $\phi$。由于特定网络层级特征很重要，以 $\phi_{i,j}=\phi_j(x_i)$ 表示数据集 $\mathcal X$ 中图像 $x_i$ 在网络第 $j$ 层级的特征。除非另有说明，按照文献惯例，$j$ 索引 ResNet 类 [23] 架构（如 ResNet-50、WideResNet-50 [57]）的特征图，$j\in\{1,2,3,4\}$ 表示相应空间分辨率块的最终输出。

一种表示选择是网络特征层级的最后一级，[4,10] 采用这一做法，但有两个问题。首先，会丢失更多局部正常信息 [14]；测试异常类型无法预知，因此有损下游性能。其次，ImageNet 预训练网络很深、抽象程度很高的特征偏向自然图像分类任务，与冷启动工业异常检测及当前数据重叠很小。

因此我们提议以中间层或中层图像块表示构建记忆库 $\mathcal M$，利用已有训练上下文，避免特征过于通用或过度偏向 ImageNet 分类。对 ResNet 类架构，例如使用 $j\in[2,3]$。为形式化图像块表示，扩展此前记号：特征图 $\phi_{i,j}\in\mathbb R^{c^*\times h^*\times w^*}$ 是深度 $c^*$、高度 $h^*$、宽度 $w^*$ 的三维张量；$\phi_{i,j}(h,w)=\phi_j(x_i,h,w)\in\mathbb R^{c^*}$ 表示位置 $h\in\{1,\ldots,h^*\}$、$w\in\{1,\ldots,w^*\}$ 的 $c^*$ 维特征切片。若每个 $\phi_{i,j}$ 感受野大于 1，它实际对应图像块表示。理想情况下，每个图像块表示的感受野应足够大，以捕获有意义、对局部空间变化鲁棒的异常上下文。虽然可以通过带步长池化及深入网络实现，但所得特征会更偏向 ImageNet、更不适合当前异常检测，同时训练成本增加，有效特征图分辨率下降。

因此，在构建每个图像块表示时作局部邻域聚合，在不损失空间分辨率或特征图可用性的情况下，扩大感受野并增强对微小空间偏差的鲁棒性。令图像块大小 $p$ 为奇数（对应所考虑的邻域大小），引入邻域：

$$
\mathcal N_p^{(h,w)}=\{(a,b)\mid a\in[h-\lfloor p/2\rfloor,\ldots,h+\lfloor p/2\rfloor],\ b\in[w-\lfloor p/2\rfloor,\ldots,w+\lfloor p/2\rfloor]\}.\tag{1}
$$

位置 $(h,w)$ 的局部感知特征为：

$$
\phi_{i,j}(\mathcal N_p^{(h,w)})=f_{agg}(\{\phi_{i,j}(a,b)\mid(a,b)\in\mathcal N_p^{(h,w)}\}).\tag{2}
$$

$f_{agg}$ 为邻域特征向量的聚合函数。PatchCore 使用自适应平均池化，类似对各特征图局部平滑，在 $(h,w)$ 产生预定维数 $d$ 的单一表示。对所有 $h\in\{1,\ldots,h^*\}$、$w\in\{1,\ldots,w^*\}$ 均执行，因而保留特征图分辨率。特征图张量 $\phi_{i,j}$ 的局部感知图像块特征集合为：

$$
\mathcal P_{s,p}(\phi_{i,j})=\{\phi_{i,j}(\mathcal N_p^{(h,w)})\mid h,w\bmod s=0,\ h<h^*,\ w<w^*,\ h,w\in\mathbb N\}.\tag{3}
$$

可选步长参数 $s$ 除第 4.4.2 节消融外设为 1。经验上，与 [10,14] 类似，聚合多个特征层级有一定好处。但为保留特征通用性和空间分辨率，PatchCore 只使用中间两个层级 $j$、$j+1$。计算 $\mathcal P_{s,p}(\phi_{i,j+1})$，并将每个元素与所用最低层级（最高分辨率）的对应图像块特征聚合。具体对前者作双线性缩放，使两个集合大小相同。

对所有正常训练样本 $x_i\in\mathcal X_N$，记忆库定义为：

$$\mathcal M=\bigcup_{x_i\in\mathcal X_N}\mathcal P_{s,p}(\phi_j(x_i)).\tag{4}$$

**图 2：PatchCore 概览。**正常样本分解为具有邻域感知能力的图像块特征记忆库，再通过贪心核心集子采样降低冗余和推理时间。测试时，只要至少一个图像块异常，便判图像异常；对各图像块特征评分，得到像素级异常分割。图内模块为：训练、正常样本、预训练编码器、局部感知图像块特征、核心集子采样、记忆库、最近邻搜索、测试、测试样本、异常分数、异常分割。

### 3.2 核心集缩减的图像块特征记忆库

随 $\mathcal X_N$ 增大，$\mathcal M$ 会非常大，测试推理时间与存储需求也增大。SPADE [10] 同时使用低层和高层特征图，已注意到其异常分割存在此问题。受计算限制，SPADE 需要用较弱的图像级异常检测预选特征图：该机制依赖整图深层表示，即最后特征图全局平均。这会产生低分辨率、偏向 ImageNet 的整图表示，可能损害检测与定位。

让 $\mathcal M$ 在图像尺寸与数量较大时仍可有效搜索，就能用有利于检测和分割的图像块比较解决这些问题，但必须保留记忆库编码的正常特征覆盖。遗憾的是，随机子采样，特别是缩减几个数量级时，会丢失覆盖中编码的大量信息（另见第 4.4.2 节）。本文使用核心集子采样缩减 $\mathcal M$，可降低推理时间并保留性能。

概念上，核心集选择寻找 $S\subset A$，使在 $S$ 上计算的解能尽量接近、特别是更快地近似 $A$ 上问题的解 [1]。不同问题的核心集不同。由于 PatchCore 使用近邻计算，采用极小极大设施选址核心集选择，见 [48,49]，确保子集 $\mathcal M_C$ 与原记忆库在图像块特征空间覆盖近似：

$$
\mathcal M_C^*=\arg\min_{\mathcal M_C\subset\mathcal M}\max_{m\in\mathcal M}\min_{n\in\mathcal M_C}\|m-n\|_2.\tag{5}
$$

精确计算为 NP-hard [54]，因此采用 [48] 的迭代贪心近似。为进一步降低选择时间，遵循 [49]，利用 Johnson–Lindenstrauss 定理 [11]，通过随机线性投影 $\psi:\mathbb R^d\to\mathbb R^{d^*}$（$d^*<d$）降低元素维数。算法 1 总结此过程。用 PatchCore−$n$% 表示原记忆库子采样至 $n$%，如 PatchCore−1% 为缩小 100 倍。图 3 直观比较贪心核心集与随机选择的空间覆盖。

**图 3：核心集与随机子采样比较。**二维蓝色数据分别采样自 (a) 多模态分布、(b) 均匀分布，红色为所选样本。上行为核心集，下行为随机。核心集更好近似空间支撑集；随机方法在多模态时漏掉簇，在 (b) 中也较不均匀。图内“Coverage”为覆盖比例；(a) 三列分别为 5.0%、10.0%、20.0%，(b) 两列分别为 2.0%、10.0%。

**算法 1：PatchCore 记忆库。**输入：预训练 $\phi$、层级 $j$、正常数据 $\mathcal X_N$、步长 $s$、图像块大小 $p$、核心集目标大小 $l$、随机线性投影 $\psi$。输出：图像块级记忆库 $\mathcal M$。

$$
\begin{aligned}
&\mathcal M\leftarrow\varnothing;\\
&\text{对每个 }x_i\in\mathcal X_N:\quad\mathcal M\leftarrow\mathcal M\cup\mathcal P_{s,p}(\phi_j(x_i));\\
&\text{应用贪心核心集选择：}\quad\mathcal M_C\leftarrow\varnothing;\\
&\text{对 }i\in[0,\ldots,l-1]:\\
&\qquad m_i\leftarrow\arg\max_{m\in\mathcal M-\mathcal M_C}\min_{n\in\mathcal M_C}\|\psi(m)-\psi(n)\|_2;\\
&\qquad\mathcal M_C\leftarrow\mathcal M_C\cup\{m_i\};\\
&\mathcal M\leftarrow\mathcal M_C.
\end{aligned}
$$

> 译注：原算法从空核心集开始，未单独说明首个点的初始化方式；此处保留原算法，不增补实现约定。

### 3.3 用 PatchCore 检测异常

利用正常图像块特征记忆库 $\mathcal M$，对测试图像 $x^{test}$，定义 $\mathcal P(x^{test})=\mathcal P_{s,p}(\phi_j(x^{test}))$。取测试图像块特征到各自在记忆库中最近邻 $m^*$ 的最大距离 $s^*$，估计图像级异常分数 $s\in\mathbb R$：

$$
\begin{aligned}
m^{test,*},m^*&=\arg\max_{m^{test}\in\mathcal P(x^{test})}\arg\min_{m\in\mathcal M}\|m^{test}-m\|_2,\\
s^*&=\|m^{test,*}-m^*\|_2.
\end{aligned}\tag{6}
$$

为得到 $s$，用缩放 $w$ 调整 $s^*$，考虑相邻图像块行为：若最接近异常候选 $m^{test,*}$ 的记忆库特征 $m^*$ 本身远离邻近样本，意味着它已是少见正常情况，我们增加异常分数：

$$
s=\left(1-\frac{\exp\|m^{test,*}-m^*\|_2}{\sum_{m\in\mathcal N_b(m^*)}\exp\|m^{test,*}-m\|_2}\right)s^*.\tag{7}
$$

$\mathcal N_b(m^*)$ 表示 $\mathcal M$ 中距离特征 $m^*$ 最近的 $b$ 个图像块特征。重加权比仅取最大图像块距离更鲁棒。有了 $s$，分割可直接得到。式 (7)（原文称“第一行”）的图像级分数通过 $\arg\max$ 要计算每个图像块分数，因而可类似 [14]，按空间位置重新对齐分数，同步计算分割图。为匹配原始输入分辨率，使用双线性插值上采样；另用核宽度 $\sigma=4$ 的高斯函数平滑，没有优化该参数。

> 译注：原文把 $m^*$ 称为“测试图像块特征”，但式 (6) 定义其为记忆库最近邻；译文按公式身份表述。上段“增加异常分数”沿用原文措辞，式 (7) 的乘子本身为相对最大距离的重加权，不改写公式。

## 4 实验

### 4.1 实验细节

**数据集。**为研究工业异常检测，大部分实验在 MVTec AD [5] 上进行。它含 15 个子数据集，共 5354 幅图像，其中测试集 1725 幅。每个子集针对一种产品，训练集只有正常数据；测试集同时有正常和不同缺陷类型的异常样本及真实异常掩码。按 [10,14,56]，图像先缩放至 $256\times256$，再中心裁剪至 $224\times224$。不使用数据增强，因为这要求预先知道哪些增强保持类别。

还用 [42] 所用的 MTD [26] 研究更专门的工业异常任务。它有 925 幅无缺陷、392 幅异常磁瓦图像，照明和图像大小各异。同 [42]，20% 无缺陷图像在测试时评估，其余用于冷启动训练。

最后，按 [52,14]，在 Mini Shanghai Tech Campus（mSTC）上评估冷启动异常定位，展示非工业适用潜力。它是原 STC [32] 的子采样版，训练、测试视频均每五帧取一帧，包含 12 个场景的行人视频。训练视频是正常行人行为，测试视频可能包含打斗或骑车等行为。为比较冷启动实验，遵循既有协议 [14,52]，不用任何异常监督，图像缩放为 $256\times256$。

**评估指标。**图像级异常检测使用生成分数的 AUROC。按既有工作，在 MVTec 上计算类别平均 AUROC [2,10,14]。分割采用像素级 AUROC 和 PRO，均遵循 [6]。PRO 考虑异常连通分量的重叠和恢复，更好处理 MVTec AD 不同异常尺寸；详见 [6]。

### 4.2 MVTec AD 异常检测

表 1 为图像级结果，报告记忆库保留 25%、10%、1% 三种设置。所有设置下，PatchCore 平均检测显著更高，且各子数据集均稳定较高（详细比较见独立补充 B）。从 PaDiM 的 2.1% 误差降到 PatchCore−25% 的 0.9%，即误差减少 57%，对工业检查有现实意义。在最优 F1 阈值下，1725 幅图像仅 42 幅错误，三分之一类别完全解决。独立补充 B 还显示，在 F1 最优和完全召回工作点，相比 SPADE、PaDiM，分类错误也更低。PatchCore 错分图像少于 50 幅。此外，其像素 AUROC（表 2，98.1 对 PaDiM 97.5）及 PRO（表 3，93.5 对 92.1）均为先进分割结果。图 1 定性体现准确定位。

**表 1：MVTec AD [5] 异常检测（AUROC）。**PaDiM* 为 [14] 使用针对问题选择主干的结果。错分数为 F1 最优阈值下假阳性与假阴性之和；其他方法无单独分数，因而只对 PatchCore 计算。

| 指标 | SPADE [10] | PatchSVDD [56] | DifferNet [42] | PaDiM [14] | Mah.AD [40] | PaDiM* [14] | PatchCore−25% | PatchCore−10% | PatchCore−1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AUROC ↑ | 85.5 | 92.1 | 94.9 | 95.3 | 95.8 | 97.9 | 99.1 | 99.0 | 99.0 |
| 误差 ↓ | 14.5 | 7.9 | 5.1 | 4.7 | 4.2 | 2.1 | 0.9 | 1.0 | 1.0 |
| 错分数 ↓ | − | − | − | − | − | − | 42 | 47 | 49 |

**表 2：MVTec AD [5] 异常分割（像素级 AUROC）。**

| 指标 | AE SSIM [5] | γ-VAE + grad. [15] | CAVGA-Rw [52] | PatchSVDD [56] | SPADE [10] | PaDiM [14] | PatchCore−25% | PatchCore−10% | PatchCore−1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AUROC ↑ | 87 | 88.8 | 89 | 95.7 | 96.0 | 97.5 | 98.1 | 98.1 | 98.0 |
| 误差 ↓ | 13 | 11.2 | 11 | 4.3 | 4.0 | 2.5 | 1.9 | 1.9 | 2.0 |

**表 3：以 PRO [%] [5,10] 衡量的 MVTec AD [5] 异常检测性能。**〔译注：原表标题写“检测”，此指标实际对应文中讨论的分割。〕

| 指标 | AE SSIM [5] | Student [6] | SPADE [10] | PaDiM [14] | PatchCore−25% | PatchCore−10% | PatchCore−1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRO ↑ | 69.4 | 85.7 | 91.7 | 92.1 | 93.4 | 93.5 | 93.1 |
| 误差 ↓ | 30.6 | 14.3 | 8.3 | 7.9 | 6.6 | 6.5 | 6.9 |

核心集记忆库子采样还使 PatchCore−1% 可用于更高分辨率（如 280/320，而非 224）及集成系统，推理时间仍低于默认分辨率的 PatchCore−10%。表 4 展示进一步提高的图像与像素检测结果，详见独立补充；部分误差再次降低一半以上，例如图像 AUROC 误差从 1% 到 0.4%。

**表 4：更高分辨率/更大主干/集成的 PatchCore−1%。**核心集允许使用计算较昂贵的设置，同时保持快速推理。

| 设置 | 指标 | AUROC | 像素 AUROC | PRO |
| --- | --- | ---: | ---: | ---: |
| DenseN-201 & RNext-101 & WRN-101 (2+3)，尺寸 320 | 分数 ↑ | 99.6 | 98.2 | 94.9 |
| 同上 | 误差 ↓ | 0.4 | 1.8 | 5.6 |
| WRN-101 (2+3)，尺寸 280 | 分数 ↑ | 99.4 | 98.2 | 94.4 |
| 同上 | 误差 ↓ | 0.6 | 1.8 | 5.6 |
| WRN-101 (1+2+3)，尺寸 280 | 分数 ↑ | 99.2 | 98.4 | 95.0 |
| 同上 | 误差 ↓ | 0.8 | 1.6 | 5.0 |

> 译注：表 4 第一组 PRO=94.9 而误差=5.6，两者与 100 不相加一致；均忠实保留原表，不自行修正。

### 4.3 推理时间

我们关注的另一维度是推理时间。表 5 与 SPADE [10]、PaDiM [14] 的重实现比较，使用 WideResNet50，尽可能在 GPU 操作，实现细节见独立补充 A。时间包含主干前向。无子采样的 PatchCore−100% 联合图像/像素检测时间低于 SPADE，性能更高。核心集能进一步加速，甚至低于 PaDiM，同时保持先进图像检测与分割。最后，研究用近似最近邻搜索 IVFPQ [27] 加速 PatchCore−100%；这与核心集正交，也能用于 SPADE，但后者已明显弱于 PatchCore−1%。近似搜索使性能下降，图像级检测尤甚，而时间仍高于 PatchCore−1%。即使下降后，其性能仍高于其他方法。结合核心集与近似近邻可进一步降低时间，扩展到更大数据集。

**表 5：MVTec AD 每幅图像平均推理时间。**分数依次为图像 AUROC、像素 AUROC、PRO。

| 方法 | 分数 | 时间 [s] |
| --- | --- | ---: |
| PatchCore−100% | (99.1, 98.0, 93.3) | 0.6 |
| PatchCore−10% | (99.0, 98.1, 93.5) | 0.22 |
| PatchCore−1% | (99.0, 98.0, 93.1) | 0.17 |
| PatchCore−100% + IVFPQ | (98.0, 97.9, 93.0) | 0.2 |
| SPADE | (85.3, 96.6, 91.5) | 0.66 |
| PaDiM | (95.4, 97.3, 91.8) | 0.19 |

### 4.4 消融研究

研究局部感知图像块特征及核心集缩减的消融。独立补充还展示不同主干一致性（C.2）、图像分辨率增大时的可扩展性（C.3）、剩余错误定性分析（C.4）。

#### 4.4.1 局部感知图像块特征及层级

通过改变式 (1) 的邻域大小、评估异常检测变化，研究局部感知特征的重要性（原文此处指第 3.3 节）。图 4 上半部分显示局部性与全局上下文有明确最佳平衡，因此选择 $p=3$。深入网络也可获得更全局上下文（如 [10,14]），但代价是分辨率下降、ImageNet 类别偏置增强（第 3.1 节）。将 WideResNet50 前三个块编号 1—3，图 4 下半部分再次显示高度局部预测、更多全局上下文及 ImageNet 偏置之间的最优点。层级 2 单独已能达到先进性能，加入后续层级特征（2+3，默认设置）仍有益。

**图 4：局部感知与网络特征深度对检测性能的影响。**PRO 结果在独立补充中。图内纵轴为检测 AUROC、分割 AUROC；上排横轴为邻域大小，下排为所选网络特征层级。上排图例“Without Coreset”“With Coreset”分别为无核心集、使用核心集。

#### 4.4.2 核心集子采样的重要性

图 5 比较三种记忆库子采样：贪心核心集、随机、以及学习一组与目标保留比例 $p_{target}$ 对应的基代理。第三种方法采样代理 $p_i\in\mathcal P\subset\mathbb R^d$，$|\mathcal P|=p_{target}|\mathcal M|$，最小化基重建目标，寻找最能描述记忆库 $\mathcal M$ 的 $N$ 个代理：

$$
\mathcal L_{rec}(m_i)=\left\|m_i-\sum_{p_k\in\mathcal P}\frac{e^{\|m_i-p_k\|_2}}{\sum_{p_j\in\mathcal P}e^{\|m_i-p_j\|_2}}p_k\right\|_2^2.\tag{8}
$$

图 5 中核心集优于其他选择。未经子采样的性能与小两个数量级的核心集记忆库相当。子采样库冗余也显著更少。我们记录测试时使用的库样本比例：未子采样时不足 30%，核心集缩减至 1% 后接近 95%。在某些保留比例区间（约 50%—10%），检测与定位联合性能甚至部分提高。最后，通过增加式 (3) 步长缩减库会损失分辨率上下文，性能更差：$s=2$ 图像检测 AUROC 为 97.6%，$s=3$ 为 96.8%。

**图 5：不同子采样方法的性能保留。**PRO 在独立补充中。横轴为子采样保留比例，左右纵轴分别为检测 AUROC、分割 AUROC；图例 Coreset、Random、Learned 分别为核心集、随机、学习代理。

### 4.5 少样本异常检测

真实检查常只有有限正常数据。除完整 MVTec AD 外，还研究更少训练样本：从 1 个（正常训练数据总量的 0.4%）到 50 个（21%），与使用同主干 WideResNet50 的 SPADE [10]、PaDiM [14] 重实现比较。图 6 汇总结果，详见独立补充 C.1。只使用五分之一正常训练数据，PatchCore 仍能匹配此前先进性能。与 [42] 的 16 样本实验比较，PatchCore 超过其在预训练特征上适配归一化流的方法。与 [10] 图像级记忆方法相比，只需 5/1 个正常样本就能分别匹配定位/检测性能。

**图 6：PatchCore 的样本效率明显高于竞争方法，仅用部分正常数据便匹配此前最佳。**为可比性，PaDiM 和 SPADE 均以 WideResNet50 重实现。三图分别为检测 AUROC、分割 AUROC、分割 PRO；下横轴为所用总比例，上横轴为每个数据集样本数（2、5、10、16、20、50）；虚线为使用 100% 数据的此前最佳性能；图例为 PaDiM、PatchCore−10、SPADE，并标出 DifferNet。

### 4.6 其他基准评估

另外在 ShanghaiTech Campus（STC）[32] 及 MTD [26] 上评估。第 4.1 节所述 STC 评估遵循 [52,14,10]。在 STC 视频子采样版 mSTC 上报告无监督定位，图像缩放为 $256\times256$ [14]。其检测上下文更接近 ImageNet 自然图像，因此使用更深的第 3、4 层级特征，除此之外不调 PatchCore 超参数。表 6 上部展示先进定位，说明可良好迁移到此领域。最后，MTD 图像大小各异，PaDiM 这类空间刚性方法无法直接应用；其正常数据已有与异常样本类似的高度变化 [42]。按 [42] 协议测量图像检测，性能匹配甚至略高于 [42]（表 6 下）。

**表 6：mSTC [32,52] 异常分割及 MTD [26] 异常检测，与 [42] 报告结果比较。**

| mSTC 方法 | CAVGA-Ru [52] | SPADE [10] | PaDiM [14] | PatchCore−10 |
| --- | ---: | ---: | ---: | ---: |
| 像素级 AUROC [%] | 85 | 89.9 | 91.2 | 91.8 |

| MTD 方法 | GANomaly [2] | 1-NN [35] | DifferNet [42] | PatchCore−10 |
| --- | ---: | ---: | ---: | ---: |
| AUROC [%] | 76.6 | 80.0 | 97.7 | 97.9 |

## 5 结论

本文提出冷启动异常检测算法 PatchCore，只利用正常示例知识，在测试时检测并分割异常。通过 ImageNet 预训练网络提取局部感知正常图像块表示并构建记忆库，保留测试时最大量正常上下文；同时以核心集子采样最小化运行时间，在两者之间取得平衡。所得系统在工业基准上以低计算成本达到先进冷启动图像检测及定位性能。MVTec 图像检测 AUROC 超过 99%，在相关小训练集场景具有最高样本效率。

**更广泛影响。**自动工业异常检测是计算机视觉最成功的应用之一，PatchCore 的改进对该领域实践者有价值。本工作专注工业异常检测，负面社会影响有限。虽然基本方法可能用于更有争议领域的检测系统，但我们认为本文改进尚不足以改变这类系统的社会应用。

**局限。**PatchCore 无需专门适配当前问题领域即可有效进行工业异常检测，但适用性总体受预训练特征可迁移性限制。可将其有效性与所用特征适配结合来解决；这一有趣扩展留待未来。

**致谢。**感谢 Yasser Jadidi、Alex Smola 支持计算基础设施设置。K.R. 感谢 International Max Planck Research School for Intelligent Systems（IMPRS-IS）及 European Laboratory for Learning and Intelligent Systems（ELLIS）博士项目支持。

## 参考文献

[1] Pankaj Agarwal, Sariel Har, Peled Kasturi, and R Varadarajan. 《通过核心集进行几何近似》。 Combinatorial and Computational Geometry, 52, 11 2004. 2, 4

[2] Samet Akcay, Amir Atapour-Abarghouei, and Toby P Breckon. 《GANomaly：通过对抗训练进行半监督异常检测》。 In Asian Conference on Computer Vision, pages 622–637. Springer, 2018. 1, 2, 5, 8, 4

[3] Jerone Andrews, Thomas Tanay, Edward Morton, and Lewis Griffin. 《用于异常检测的迁移表示学习》。 07 2016. 4

[4] Liron Bergman, Niv Cohen, and Yedid Hoshen. 《深度最近邻异常检测》。 CoRR, abs/2002.10445, 2020. 1, 2, 3

[5] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. 《MVTec AD：用于无监督异常检测的全面真实世界数据集》。 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. 1, 2, 5, 6, 7, 4

[6] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. 《不知情的学生：使用判别潜嵌入的师生异常检测》。 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , June 2020. 2, 5, 7, 4

[7] Paul Bergmann, Sindy Löwe, Michael Fauser, David Sattlegger, and Carsten Steger. 《将结构相似性应用于自编码器以改善无监督缺陷分割》。 Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, 2019. 2

[8] Wieland Brendel and Matthias Bethge. 《使用局部特征袋模型近似 CNN 在 ImageNet 上表现出乎意料地好》。 In International Conference on Learning Representations, 2019. 2

[9] Kenneth L. Clarkson. 《核心集、稀疏贪心近似与 Frank–Wolfe 算法》。ACM Trans. Algorithms, 6(4), Sept. 2010. 2

[10] Niv Cohen and Yedid Hoshen. 《使用深度金字塔对应进行子图像异常检测》。 CoRR, abs/2005.02357, 2020. 1, 2, 3, 4, 5, 6, 7, 8

[11] Sanjoy Dasgupta and Anupam Gupta. 《Johnson 和 Lindenstrauss 定理的一个初等证明》。 Random Structures & Algorithms, 22(1):60–65, 2003. 4

[12] Diana Davletshina, Valentyn Melnychuk, Viet Tran, Hitansh Singla, Max Berrendorf, Evgeniy Faerman, Michael Fromm, and Matthias Schubert. 《X 射线图像的无监督异常检测》。 2020. 1

[13] Lucas Deecke, Robert Vandermeulen, Lukas Ruff, Stephan Mandt, and Marius Kloft. 《使用生成对抗网络进行图像异常检测》。 In Michele Berlingerio, Francesco Bonchi, Thomas Gärtner, Neil Hurley, and Georgiana Ifrim, editors, Machine Learning and Knowledge Discovery in Databases, pages 3–17, Cham, 2019. Springer International Publishing. 2

[14] Thomas Defard, Aleksandr Setkov, Angelique Loesch, and Romaric Audigier. 《PaDiM：用于异常检测与定位的图像块分布建模框架》。 In Alberto Del Bimbo, Rita Cucchiara, Stan Sclaroff, Giovanni Maria Farinella, Tao Mei, Marco Bertini, Hugo Jair Escalante, and Roberto Vezzani, editors, Pattern Recognition. ICPR International Workshops and Challenges , pages 475–489, Cham, 2021. Springer International Publishing. 2, 3, 5, 6, 7, 8, 1, 4

[15] David Dehaene, Oriel Frigo, Sébastien Combrexelle, and Pierre Eline. 《通过在正常数据流形上进行迭代能量投影实现异常定位》。 In International Conference on Learning Representations, 2020. 6, 4

[16] J. Deng, W. Dong, R. Socher, L. Li, Kai Li, and Li Fei-Fei. 《ImageNet：大规模层级图像数据库》。 In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009. 2

[17] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. 《使用 Real NVP 进行密度估计》。 In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. 2

[18] Eleazar Eskin, Andrew Arnold, Michael Prerau, Leonid Portnoy, and Sal Stolfo. 《无监督异常检测的几何框架》。 pages 77–101. Springer US, Boston, MA, 2002. 2

[19] Dan Feldman, Matthew Faulkner, and Andreas Krause. 《通过核心集进行混合模型的可扩展训练》。 In J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 24, pages 2142–2150. Curran Associates, Inc., 2011. 2

[20] Izhak Golan and Ran El-Yaniv. 《使用几何变换进行深度异常检测》。 In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31, pages 9758–9769. Curran Associates, Inc., 2018. 2, 4

[21] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. 《记忆正常性以检测异常：用于无监督异常检测的记忆增强深度自编码器》。 In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) , October 2019. 2

[22] Sariel Har-Peled and Akash Kushal. 《用于 k-median 和 k-means 聚类的更小核心集》。 Discrete and Computational Geometry, 37:3–19, 12 2007. 2

[23] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 《用于图像识别的深度残差学习》。 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016. 3, 6

[24] Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. 《蒸馏神经网络中的知识》。 In NIPS Deep Learning and Representation Learning Workshop, 2015. 2

[25] Chaoqing Huang, Jinkun Cao, Fei Ye, Maosen Li, Ya Zhang, and Cewu Lu. 《用于异常检测的逆变换自编码器》。 CoRR, abs/1911.10676, 2019. 2, 4

[26] Yibin Huang, C. Qiu, and K. Yuan. 《磁瓦表面缺陷显著性》。 The Visual Computer, 36:85–96, 2018. 2, 5, 8

[27] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 《使用 GPU 进行十亿级相似性搜索》。 IEEE Transactions on Big Data, pages 1–1, 2019. 6, 1

[28] Andrei Kapishnikov, Tolga Bolukbasi, Fernanda Viegas, and Michael Terry. 《XRAI：通过区域实现更好的归因》。 In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019. 2

[29] Ki Hyun Kim, Sangwoo Shim, Yongsub Lim, Jongseob Jeon, Jeongwoo Choi, Byungchan Kim, and Andre S. Yoon. 《RAPP：通过沿投影路径重建进行新颖性检测》。 In International Conference on Learning Representations, 2020. 2

[30] Durk P Kingma and Prafulla Dhariwal. 《Glow：使用可逆 1×1 卷积的生成流》。 In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. 2

[31] Wenqian Liu, Runze Li, Meng Zheng, Srikrishna Karanam, Ziyan Wu, Bir Bhanu, Richard J. Radke, and Octavia Camps. 《迈向变分自编码器的视觉解释》。 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020. 4

[32] W. Liu, D. Lian W. Luo, and S. Gao. 《用于异常检测的未来帧预测：一个新基线》。 In 2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 5, 8

[33] Prasanta Chandra Mahalanobis. 《论统计学中的广义距离》。 Proceedings of the National Institute of Sciences (Calcutta), 2:49–55, 1936. 2

[34] Ben Mussay, Margarita Osadchy, Vladimir Braverman, Samson Zhou, and Dan Feldman. 《通过核心集进行与数据无关的神经网络剪枝》。 In International Conference on Learning Representations, 2020. 2

[35] Tiago S. Nazaré, Rodrigo Fernandes de Mello, and Moacir A. Ponti. 《预训练 CNN 是监控视频异常检测的良好特征提取器吗？》。 CoRR, abs/1811.08495, 2018. 8

[36] Duc Tam Nguyen, Zhongyu Lou, Michael Klar, and Thomas Brox. 《使用多假设预测进行异常检测》。 In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research , pages 4800–4809. PMLR, 09–15 Jun 2019. 1

[37] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 《PyTorch：命令式高性能深度学习库》。 2019. 1

[38] Pramuditha Perera, Ramesh Nallapati, and Bing Xiang. 《OCGAN：使用受约束潜表示 GAN 的单类新颖性检测》。 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. 2

[39] Stanislav Pidhorskyi, Ranya Almohsen, Donald A. Adjeroh, and Gianfranco Doretto. 《使用对抗自编码器的生成式概率新颖性检测》。 In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, page 6823–6834, Red Hook, NY , USA, 2018. Curran Associates Inc. 1, 2

[40] Oliver Rippel, Patrick Mertens, and Dorit Merhof. 《在预训练深度特征中建模正常数据分布以检测异常》。 In 2020 25th International Conference on Pattern Recognition (ICPR) , pages 6726–6733, 2021. 2, 6, 4

[41] Karsten Roth, Timo Milbich, Samarth Sinha, Prateek Gupta, Bjorn Ommer, and Joseph Paul Cohen. 《重新审视深度度量学习的训练策略与泛化性能》。 In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 8242–8252. PMLR, 13–18 Jul 2020. 3

[42] Marco Rudolph, Bastian Wandt, and Bodo Rosenhahn. 《大同小异的 DifferNet：使用归一化流进行半监督缺陷检测》。 In Winter Conference on Applications of Computer Vision (WACV), Jan. 2021. 1, 2, 5, 6, 7, 8, 4

[43] Mohammad Sabokrou, Mohammad Khalooei, Mahmood Fathy, and Ehsan Adeli. 《用于新颖性检测的对抗学习单类分类器》。 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018. 1, 2

[44] Mayu Sakurada and Takehisa Yairi. 《使用具有非线性降维能力的自编码器进行异常检测》。 In Proceedings of the MLSDA 2014 2nd Workshop on Machine Learning for Sensory Data Analysis, MLSDA’14, page 4–11, New York, NY , USA, 2014. Association for Computing Machinery. 1, 2

[45] Mohammadreza Salehi, Niousha Sadjadi, Soroosh Baselizadeh, Mohammad Hossein Rohban, and Hamid R. Rabiee. 《用于异常检测的多分辨率知识蒸馏》。 2020. 2

[46] Bernhard Schölkopf, Robert C. Williamson, Alex J. Smola, John Shawe-Taylor, and John C. Platt. 《用于新颖性检测的支持向量方法》。 In Advances in Neural Information Processing Systems 12, pages 582–588, Cambridge, MA, USA, June 2000. Max-Planck-Gesellschaft, MIT Press. 2

[47] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra. 《Grad-CAM：通过基于梯度的定位提供深度网络的视觉解释》。 In 2017 IEEE International Conference on Computer Vision (ICCV), pages 618–626, 2017. 2

[48] Ozan Sener and Silvio Savarese. 《卷积神经网络的主动学习：核心集方法》。 In International Conference on Learning Representations, 2018. 2, 4

[49] Samarth Sinha, Han Zhang, Anirudh Goyal, Yoshua Bengio, Hugo Larochelle, and Augustus Odena. 《Small-GAN：使用核心集加速 GAN 训练》。 In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 9005–9015. PMLR, 13–18 Jul 2020. 3, 4

[50] David M. J. Tax and Robert P. W. Duin. 《支持向量数据描述》。 Machine Learning, 54:45–66, 2004. 2

[51] Guido Van Rossum and Fred L. Drake. 《Python 3 参考手册》。 CreateSpace, Scotts Valley, CA, 2009. 1

[52] Shashanka Venkataramanan, Kuan-Chuan Peng, Rajat Vikram Singh, and Abhijit Mahalanobis. 《图像中注意力引导的异常定位》。 In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision – ECCV 2020 , pages 485–503, Cham, 2020. Springer International Publishing. 2, 5, 6, 8, 4

[53] Ross Wightman. 《PyTorch 图像模型》。 <https://github.com/rwightman/pytorch-image-models>, 2019. 1

[54] Laurence A. Wolsey and George L. Nemhauser. 《整数与组合优化》。 Wiley Series in Discrete Mathematics and Optimization. Wiley, 2014. 4

[55] Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, and Kaiming He. 《用于深度神经网络的聚合残差变换》。 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , July 2017. 6

[56] Jihun Yi and Sungroh Yoon. 《Patch SVDD：用于异常检测与分割的图像块级 SVDD》。 In Proceedings of the Asian Conference on Computer Vision (ACCV) , November 2020. 1, 2, 5, 6, 4

[57] Sergey Zagoruyko and Nikos Komodakis. 《宽残差网络》。 In Edwin R. Hancock Richard C. Wilson and William A. P. Smith, editors, Proceedings of the British Machine Vision Conference (BMVC) , pages 87.1–87.12. BMVA Press, September 2016. 3, 1, 6

[58] Shuangfei Zhai, Yu Cheng, Weining Lu, and Zhongfei Zhang. 《用于异常检测的深度结构化能量模型》。 In Maria Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1100–1109, New York, New York, USA, 20–22 Jun 2016. PMLR. 4

[59] Zhou Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. 《图像质量评价：从误差可见性到结构相似性》。 IEEE Transactions on Image Processing , 13(4):600–612, 2004. 2

[60] Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Daeki Cho, and Haifeng Chen. 《用于无监督异常检测的深度自编码高斯混合模型》。 In International Conference on Learning Representations, 2018. 2
