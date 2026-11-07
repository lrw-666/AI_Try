# 通过概念瓶颈模型实现可解释视觉异常检测

**原文：**[Explainable Visual Anomaly Detection via Concept Bottleneck Models](<Explainable Visual Anomaly Detection via Concept Bottleneck Models.pdf>)。本译文对应本目录 27 页 PDF，arXiv:2511.20088v3，修订日期 2026 年 10 月 12 日；按预印本引用。翻译日期：2026-11-07。

**作者：**Arianna Stropeni、Valentina Zaccaria、Francesco Borsatti、Davide Dalle Pezze、Manuel Barusco、Gian Antonio Susto。

**单位：**University of Padova 信息工程系，意大利 Padova。邮箱依次为 arianna.stropeni@studenti.unipd.it；valentina.zaccaria@unipd.it；francesco.borsatti.1@phd.unipd.it；davide.dallepezze@unipd.it；manuel.barusco@phd.unipd.it；gianantonio.susto@unipd.it。

**翻译说明：**逐段覆盖本地 PDF 全部 27 页：摘要、正文第 1—7 节、参考文献 [1]—[48]、补充材料第 8—17 节、图 1—12 的图注和关键图内标注、表 1—6 的全部数据、公式 (1)—(14)。合并跨行跨页续段，去除运行页眉页脚；不重绘图形，请结合 PDF 查看。模型、数据集和引文编号保留。正文与补充材料关于融合、概念分支分类器等描述的差异按原文保留，并以译注提示。文中“最佳”等判断均属作者在该版本中的陈述。

**版权说明：**©2026。本文版权归作者所有，可将未经修改的文档以印刷或电子形式自由分发。

## 摘要

近年来，视觉异常检测（Visual Anomaly Detection，VAD）因较高的可解释性而受到广泛关注：它通过突出图像中的异常区域提供视觉解释。然而，这类视觉解释虽然有帮助，却缺乏直接且语义明确、对用户有意义的解释。为解决这一局限，我们提出将概念瓶颈模型（Concept Bottleneck Models，CBM）扩展到 VAD 场景。通过学习有意义的概念，网络能够给出人类可理解的异常描述，以新颖且更有洞察力的方式解释异常。

主要贡献包括：(i) 首次将 CBM 扩展到 VAD，提出基于概念的异常解释框架；(ii) 评估从有监督到无监督异常场景的多种监督机制；(iii) 提出双分支架构，将提供概念级解释的 CBM 概念分支与提供像素级异常定位的视觉分支结合，连接语义和空间可解释性。在三个成熟 VAD 基准上的评估表明，概念感知视觉异常检测（Concept-Aware Visual Anomaly Detection，CONVAD）的性能可与经典 VAD 方法相当（MVTec AD 上 I-AUROC 最高 0.97），同时提供更丰富的概念驱动解释。包含 582 份回答的用户研究显示，CONVAD 的解释在 73.5% 的情况下被评为有效（先进 VLM 解释方法 Anomaly-OV 为 60.7%），而只有 3.1% 被评为误导（对方为 11.7%）。除解释质量外，CBM 还支持人在回路中的纠正：在具有挑战性的无监督场景中，概念级干预进一步提升检测能力，部分缩小与完全监督性能之间的差距。

## 1 引言

视觉异常检测（VAD）旨在检测图像异常，同时突出导致异常的像素。这些能力在制造、医学和监控等领域尤其重要 [2, 6, 8]。

**图 1：（左）CONVAD 架构；（右）CONVAD 输出示例：**输入图像、异常热力图、最活跃和不活跃概念列表，以及预测。图内概念分支包括特征提取器、概念嵌入、概念异常得分；示例概念为均匀表面 −0.43、卵形形状 0.98、不规则外壳 1.69。视觉分支包括学生网络、特征图、像素级异常分割和视觉异常得分，两分支进入得分融合。右侧活跃概念为可见裂纹、玻璃碎片、锐利边缘、反射高光；不活跃概念为一致颜色、异物颗粒、琥珀色、圆柱形、光滑表面；最终预测为异常。

虽然 VAD 模型能够生成异常热力图，但这些视觉定位本身是否足以实现可解释性仍值得质疑。实际上，确定异常发生在何处，有时不足以理解究竟什么是异常，因为这依赖上下文以及潜在的正常性定义。本文采用基于概念瓶颈模型 [17] 的不同 VAD 范式，以称为“概念”的语义解释补充空间异常热力图。与标准 VAD 不同，CBM 依赖有监督概念标注，显式学习驱动最终预测的、可由人理解的中间概念。将 CBM 纳入 VAD 后，系统不仅提供图像级和像素级标签，还能提供每个预测背后的一组可解释概念，帮助最终用户理解模型输出并改善决策。此外，用户可以干预中间概念激活，帮助模型正确预测，实现标准 VAD 模型无法提供的无缝人机协作。

CBM 的潜在局限是需要有监督训练。这与传统 VAD 假定只有正常样本可用的设定不同，因为收集和标注异常样本成本很高，尤其在工业场景中。尽管如此，越来越多近期研究表明，将有限标签信息引入 VAD 可以显著提高异常检测性能 [25, 34]。这种监督可以是弱标签，也可以是少量真实异常样本。总体而言，这些发现表明，在标注投入与模型能力之间进行平衡既切实可行，也有益处。沿此方向，我们采用有限监督下的有监督设定，考虑只有极少真实异常或使用合成异常的场景。极少的标注投入换来了可解释性提升和更有效的人机协作。

贡献概括如下：

(i) **将 CBM 适配到 VAD。**据我们所知，这是首次成功将 CBM 适配到 VAD。该适配并非平凡，需要解决多项挑战。CBM 的训练需要“概念数据集”，其中每个样本均以预定义词表中的概念标注。VAD 尚无此类数据集，因此，为广泛评估方法，我们采用基于 VLM 的自动流程提取有意义的概念并标注工业图像。这是其他领域 CBM 文献中已存在的方法 [21, 33, 43]。我们公开发布三个主要工业 VAD 基准（MVTec AD、VisA、Real-IAD）所对应的概念数据集，以促进后续研究，同时公开实验所用的全部代码（见补充材料）。

(ii) **评估监督机制与合成异常生成。**我们比较异常数据可用程度不同的多种训练设定：使用全部真实异常的完全监督、仅使用少量真实样本的半监督，以及完全依赖专门流程生成合成缺陷的设定。这使我们能够深入分析性能与标注投入之间的权衡，逐步减少对真实异常数据的依赖，接近标准 VAD 范式。每种机制都与先进和标准基线比较。

(iii) **同时提供概念级与像素级解释的双分支架构。**VAD 方法具有像素级异常定位能力，标准 CBM 只能进行样本级预测，因而不具备这一能力。我们提出新的双分支架构：有监督的 CBM 概念分支提供概念级可解释性，视觉分支提供像素级异常定位。二者把概念解释丰富的语义与异常热力图精确的空间定位结合起来。

(iv) **跨数据集鲁棒性评估。**在 MVTec AD、VisA、Real-IAD 三个成熟工业 VAD 基准上验证方法，展示其在不同工业领域和缺陷类型间的泛化性与鲁棒性。

本文其余结构如下：第 2 节回顾相关 VAD 文献；第 3 节介绍 CBM 架构和训练方法背景；第 4 节提出 CONVAD，详述如何将 CBM 适配到 VAD；第 5 节介绍实验设定、实现细节和监督场景；第 6 节讨论结果；第 7 节提出未来研究方向。

## 2 相关工作

视觉异常检测的大量研究产生了众多方法，大致可分为四组：

**1. 基于特征的方法。**这类方法依赖预训练神经网络提取的特征，利用其中丰富的语义和结构信息，将偏离正常模式的情况检测为异常。例如 PatchCore [27]、STFPM [37]、FastFlow [45] 和 PaDiM [11]。

**2. 基于重建的方法。**假定仅在正常数据上训练的模型无法准确重建异常区域，因此重建误差可以指示异常。例如 AnoGAN [29]、f-AnoGAN [30] 和 UniAD [44]。与特征方法相比，它们需要训练生成模型，计算成本较高。

**3. 合成异常方法。**这类方法用合成异常扩充数据集，学习显式异常模式。CutPaste [18]、DRAEM [46]、SuperSimpleNet [24] 等早期方法使用基础或不真实的增强；近期方法则利用先进生成模型和视觉语言模型（VLM），生成保真度更高的异常，提升鲁棒性和性能 [15, 34]。值得注意的是，CONVAD 所用生成模型已被验证能够生成真实感工业异常 [15]，支持其适用于本任务。

较少被探索的另一种设定是半监督：DevNet [22]、DRA [13] 等方法将少量真实异常与正常数据结合，不依赖合成生成。

这些方法虽可提供异常图形式的视觉解释，却无法产生人类可理解的语义解释。

**4. 基于 VLM 的异常检测。**近期 VLM 方法将异常检测与异常的自然语言描述结合 [14, 16, 40, 42, 48]。虽然零样本性能较强（本文不以此为目标），它们却需要测试时运行 VLM，带来显著计算开销，可能限制工业部署。此外，零样本方法依赖先验语义知识，而非数据集统计，因此可能将语义上不常见、但在数据集中普遍存在的模式判为异常。例如，贯穿整个数据集的变色仍可能被 VLM 方法标为异常，而 CONVAD 和传统方法会将其视为正常。另外，其文本解释往往属于事后解释，不一定忠实反映驱动预测的因素，会产生幻觉或不一致解释 [36]。相比之下，CONVAD 在视觉特征和预测之间强制设置语义瓶颈，将决策建立在显式可解释概念之上，并支持干预。

提高可解释性的另一方向是概念解释。一些工作研究了计算机视觉中检测与解释分布外（OOD）样本的概念方法 [10, 19, 31]。但它们通常在类别层面定义异常，例如以狗训练模型后将猫图像视为异常。这不同于现实 VAD：异常通常是原本正常物体内的细微缺陷。据我们所知，以往尚无工作研究针对 VAD 的概念方法。我们通过探索 CBM 在 VAD 中的应用并评估效果，填补这一空白。

## 3 预备知识：概念瓶颈模型

CBM [17] 是一种通过中间的人类可解释概念集合进行预测的可解释神经架构。其关键优势在于，与大型 VLM 不同，CBM 需要的内存极少，推理快速，同时仍能提供人类可理解的解释，因此适合资源受限或实时场景。形式上，令 $x\in\mathcal X$ 为输入，$c\in\mathcal C$ 为包含 $k$ 个概念的向量，$y\in\mathcal Y$ 为目标标签。CBM 包含两个函数：概念提取器 $g:\mathcal X\to\mathcal C$ 将输入 $x$ 映射为预测概念向量 $\hat c=g(x)$；标签预测器 $f:\mathcal C\to\mathcal Y$ 将 $\hat c$ 映射为最终输出 $\hat y=f(\hat c)$。因此，CBM 为复合函数 $\hat y=f(g(x))$。

概念层充当瓶颈，迫使预测 $\hat y$ 完全通过概念预测依赖于 $x$。这提高下游输出可解释性，并支持下文讨论的测试时干预。

标准 CBM 需要有监督学习，即训练时可访问真实三元组 $\{x^i,c^i,y^i\}_{i=1}^m$。在本文设定中，$x$ 是原始图像，$y\in\{0,1\}$ 表示是否存在异常，真实概念为二值，即 $\mathcal C=\{0,1\}^k$。

**图 2：通过 VLM 概念标注创建概念数据集的流程。**步骤为：1. 上下文感知列表创建；2. 分组与过滤；3. 数据集自动标注。原始概念示例：孔洞、纹理不规则、锐利边界、卵形形状、线状断裂；分组示例：自然外观/有机外观，以及不均匀纹理/纹理不规则/纹理变化。最终词表示例为紧实外观、不规则纹理，每幅图像对应 1/0 二值标注。

### 3.1 CBM 训练

CBM 可采用不同策略训练。本文采用联合端到端训练，它持续优于其他范式（独立训练和顺序训练）。其他范式细节与消融见补充材料。

**联合训练**同时优化概念提取器 $g$ 与标签预测器 $f$，最小化

$$\hat f,\hat g=\arg\min_{f,g}\sum_i\mathcal L_Y(f(g(x^i)),y^i)+\lambda\sum_{i,j}\mathcal L_{C_j}(g_j(x^i),c_j^i).\tag{1}$$

其中 $\mathcal L_Y$ 为标签预测损失，$\mathcal L_{C_j}$ 为第 $j$ 个概念损失，$\lambda$ 平衡两个目标。

注意，预测概念 $g_j(x^i)$ 是连续 logit，而非真实概念向量中的二值变量。分类器 $f$ 直接处理这些 logit。

### 3.2 CBM 测试时干预

CBM 的关键性质是支持测试时干预。用户或领域专家发现概念预测错误时，可以手动令 $\hat c_j=c_j$ 进行纠正。纠正后概念向量 $\tilde c$ 输入标签预测器 $f$，得到更新后、理想情况下更准确的输出 $\tilde y=f(\tilde c)$。在 $f$ 处理概念 logit 的联合训练模型中，为避免极端值，干预将相应 logit 替换为训练 logit 分布的第 5 或第 95 百分位数（详见补充材料）。该机制允许专家无需重新训练就能改进预测，这一能力来自 CBM 瓶颈结构。

## 4 方法

### 4.1 概念数据集流程

CBM 的关键挑战是缺少概念标注数据集，包括 VAD 在内的许多领域都不存在这类数据。既有工作采用基于 VLM 的自动流程解决，例如 Label-free CBMs [21]、LaBo [43] 和 VLG-CBMs [33]。但此前研究聚焦自然图像多分类，与本文设定不同。VAD 概念具有领域专属性且粒度精细。此外，工业异常高度依赖数据集和类别，在 VLM 训练数据中代表不足，可能导致泛化空洞或幻觉概念。因此，我们设计专门面向异常的概念提取流程和提示词（详见补充材料），并用人工标注真值验证。该方法受其他领域现有概念提取方法启发，仅作适应 VAD 所需的少量修改。

如图 2，流程分为三步：

**1. 上下文感知列表创建。**不同于脱离领域上下文生成概念的既有方法，我们引入针对每个物体类别定制的上下文感知范式。采样代表性子集 $X=\{x_i\in D\mid i=1,\ldots,N\}$，占数据集 5%，确保覆盖全部缺陷类型。令 $M$ 为 VLM，$p_t$ 为注入上下文（物体类别、缺陷类型）且采用思维链策略 [39] 的文本提示（设计详见补充材料）。模型先生成自然语言图像描述，再从中提取最多五个语义概念。将提取约束于视觉描述，可避免模型关注非视觉属性并限制幻觉。所得概念集合为

$$V=\{v_{ij}\mid i=1,\ldots,N;\ j=1,\ldots,|v_i|\},\qquad v_i=M(x_i,p_t).\tag{2}$$

**2. 分组与过滤。**原始概念经两个子步骤细化：(i) **概念分组：**将完整列表和少样本示例 [7] 反馈给 VLM，合并形态和语义相关属性，将数百个概念缩减至不到五十个。(ii) **概念过滤：**遵循 [21]，用 CLIP ViT-B/32 文本编码器 $E_T$ [23] 编码概念；对余弦相似度超过 0.9 的每对概念，丢弃其中一个以去除近重复项：

$$\cos(E_T(v_i),E_T(v_j))=\frac{E_T(v_i)\cdot E_T(v_j)}{\|E_T(v_i)\|\,\|E_T(v_j)\|}.\tag{3}$$

**3. 数据集自动标注。**为每幅图像标注二值概念标签。使用带类别与缺陷信息的提示查询 VLM，要求对照最终概念列表检查每幅图像，为每个概念返回布尔赋值（见补充材料）。在覆盖全部类别的 400 幅人工标注图像子集上检验自动标注质量，达到 91.3% 准确率（详情见补充材料）。

图 3 展示 MVTec 榛子类别所得词表及按缺陷类型划分的分布。正常类与异常类的概念存在有意义的差异，证实它们对下游任务具有信息价值。异常类概念既包括一般性指示（色调不均、表面不连续），也包括精细描述（可见墨迹、内部较暗、孔洞）；后者在不同缺陷类型上具有不同分布，因此能对特定缺陷进行更精确、更有区分力的描述。值得注意的是，概念并非正常或异常类别所独占，许多概念同时出现于两类，只是频率不同。因此，单个概念与最终标签之间并非一一对应，分类依赖它们的共同作用。

**图 3：MVTec hazelnut 类别的概念词表及不同缺陷类型分布示例。**各行：正常、孔洞、印记、切口、裂纹。横轴概念依次为：紧实外观、完整外壳、均匀外观、自然外观、颜色偏差、内部较暗、可见缝隙、可见墨迹、孔洞、裂纹、劈裂、断裂、线状中断、可见印记、不规则外壳、结构损伤、纹理异常、表面瑕疵、表面不连续、破损外壳、色调不均。

### 4.2 CONVAD 架构

现代 VAD 方法提供像素级异常定位，标准 CBM 却不能。我们为 CBM 增加空间定位分支，弥合这一差距，使其共同生成概念解释和细粒度异常热力图。图 1 所示架构含两分支：(i) **概念分支** $\phi_{\mathrm{cbm}}$，输出概念预测和异常得分；(ii) **视觉分支** $\phi_{\mathrm{vis}}$，作为学生在正常数据上匹配概念分支特征，产生空间异常定位及额外异常得分。两个异常得分按式 (9) 合并。分支间共享特征提取器，鼓励语义解释与空间解释保持一致。

#### 4.2.1 训练

训练依次分为两个阶段。首先端到端训练 $\phi_{\mathrm{cbm}}$，优化概念预测与异常检测目标。然后冻结其权重，遵循学生教师 VAD 范式 [3, 12, 37, 47]，仅在正常样本上训练学生 $\phi_{\mathrm{vis}}$，匹配 $\phi_{\mathrm{cbm}}$ 的中间特征。

**概念分支。**定义每个概念的损失与异常检测损失：

$$\mathcal L_{C_j}=\mathcal L_{\mathrm{BCE}}(c_j,\hat c_j,\alpha),\qquad\mathcal L_Y=\mathcal L_{\mathrm{BCE}}(y,\hat y,\alpha).\tag{4}$$

其中

$$\mathcal L_{\mathrm{BCE}}(z,\hat z;\alpha)=-\frac1n\sum_{i=1}^n\left(\alpha z^i\log\hat z^i+(1-z^i)\log(1-\hat z^i)\right).$$

权重因子 $\alpha\in\mathbb R$ 补偿概念和异常预测中的类别不平衡，取不平衡比 $\alpha=(n-\sum_{i=1}^n z^i)/(\sum_{i=1}^n z^i)$。两个损失组合为

$$\mathcal L_{\mathrm{cbm}}=\frac1{1+\lambda k}\left(\mathcal L_Y+\lambda\sum_{j=1}^k\mathcal L_{C_j}\right),\tag{5}$$

其中 $k$ 为概念数量，$\lambda\in\mathbb R$ 控制异常检测与概念预测目标之间的权衡。

**视觉分支。**冻结 $\phi_{\mathrm{cbm}}$ 后，仅在正常样本上优化 $\phi_{\mathrm{vis}}$，匹配其中间特征表征：

$$\mathcal L_{\mathrm{vis}}=\frac1{|X_n|}\sum_{x\in X_n}\frac1F\sum_{f=1}^F\left\|\phi^{(f)}_{\mathrm{cbm}}(x)-\phi^{(f)}_{\mathrm{vis}}(x)\right\|_2^2,\tag{6}$$

其中 $X_n$ 为正常训练样本集合，$F$ 为固定的中间特征层数量 [3, 37]，$\phi^{(f)}_{\mathrm{cbm}}$、$\phi^{(f)}_{\mathrm{vis}}$ 分别表示对应网络的第 $f$ 层。

#### 4.2.2 推理

给定测试图像 $x$，模型产生三个输出：概念解释、空间热力图和异常得分。

**概念解释。**给定预测概念激活 $\hat c=(\hat c_1,\ldots,\hat c_k)$ 与最终线性层学到的权重 $w=(w_1,\ldots,w_k)$，概念 $c_j$ 对异常得分的贡献正比于

$$\delta_j(x)=w_j\cdot\hat c_j(x).\tag{7}$$

随后按 $|\delta_j(x)|$ 排序列出概念，突出对最终预测影响最大的概念。

与经典 VAD 相比，CONVAD 有两个主要优势。第一，模型本身可解释，排序后的概念贡献为异常预测提供直接、人类可读的理由。第二，可将最相关概念激活 $\hat c$ 展示给专家验证，让人通过纠正错误概念参与干预，无需重新训练即可直接影响最终决策 [9]。

**异常热力图。**推理时，两网络都处理输入 $x$，由于异常区域处于学生训练分布之外，利用特征图差异计算异常热力图：

$$H_{\mathrm{vis}}(x)=\frac1F\sum_{f=1}^F U\left(D^{(f)}(x)\right),\qquad D^{(f)}_{h,w}(x)=\left\|\phi^{(f)}_{\mathrm{cbm},h,w}(x)-\phi^{(f)}_{\mathrm{vis},h,w}(x)\right\|_2^2,\tag{8}$$

其中 $U(\cdot)$ 表示双线性上采样至原始分辨率 $H\times W$，每层的平方差在特征通道间取平均。这将 CBM 分支产生的文本解释与异常区域的空间可视化结合。示意见图 1。

**异常得分。**为使两分支处于共同尺度，使用小型留出集 $X_{\mathrm{cal}}$ 校准视觉分支。对热力图空间维度最大池化，得到逐图像视觉得分 $s_{\mathrm{vis}}(x)=\max_{h,w}H^{(h,w)}_{\mathrm{vis}}(x)$；选取使 $X_{\mathrm{cal}}$ 上 F1 最大的阈值 $\tau^*$，获得二值预测 $\hat y_{\mathrm{vis}}(x)=\mathbf1[s_{\mathrm{vis}}(x)\ge\tau^*]\in\{0,1\}$。概念分支以 $\hat c(x)$ 为条件，输出 sigmoid 异常概率 $s_{\mathrm{cbm}}(x)=\sigma(\phi_{\mathrm{cbm}}(x))\in[0,1]$。最终得分结合两个信号：

$$s(x)=\max\left(\hat y_{\mathrm{vis}}(x),s_{\mathrm{cbm}}(x)\right).\tag{9}$$

任一分支检测到异常就足以标记该图像，使检测更能抵御单分支失效。值得注意的是，如表 2 右侧所示，不同监督水平下，约 88%—91% 的最终预测由概念分支驱动，意味着视觉分支主要为概念层未捕获的分布外样本提供后备。因此，绝大多数检测由概念级解释驱动。其他融合策略的消融见补充材料：它们虽取得更高 AUROC，却持续降低与概念分支的一致性。

### 4.3 合成异常生成

遵循既有工作 [15, 34]，我们采用生成方法生成真实感合成异常。纳入这些样本后，CONVAD 可以在所有监督机制下工作，包括没有真实异常图像的半监督和无监督设定。

具体而言，使用文生图与图像编辑系统修改正常图像，得到合成异常。令 $G_{\mathrm{edit}}(x,p)$ 为编辑函数，输入图像 $x\in\mathbb R^{H\times W\times3}$ 和文本提示 $p$。从正常图像 $x_n$ 出发，生成程序产生 $x_a=G_{\mathrm{edit}}(x_n,p_a)$，其中 $p_a$ 包含所需缺陷类型，并保证与原图一致。该操作旨在引入异常，同时保持图像其余部分不变。

提示经过精心设计，以控制异常生成（细节见补充材料）。每条提示指定：(i) 缺陷类型，例如划痕、裂纹、凹痕；(ii) 物体类型，即被修改物体或表面，例如金属板、塑料零件；(iii) 姿态与视角细节，确保编辑图像保留原始相机角度、方向和光照。

## 5 实验设定

### 5.1 场景与基线

本文在训练监督类型和数量不同的多个场景中评估 CONVAD，采用与 [41] 相同的定义。

**有监督：**使用真实数据集中的正常和异常图像训练。由于没有在相同数据集和相同条件下训练的专门全监督 VAD 方法，我们与基于 UNet [26] 的有监督异常检测（AD）比较。

**半监督：**使用全部可用正常图像、合成异常以及少量真实异常训练。具体而言，每种缺陷采用单样本设定，每类别平均约 5 幅真实异常图像。作为比较，在相同设定下评估半监督 DevNet [22] 和 DRA [13]，每类别使用 5 幅异常图像。

**无监督：**训练仅使用真实数据集中的正常图像和合成样本。遵循既有工作 [18, 20, 41]，由于训练不使用真实异常样本，这仍视为无监督设定。在该场景中，只从合成异常图像提取概念，因此概念提取期间模型从未接触真实缺陷样本；此外，得分融合所需最大化 F1 的阈值只使用合成异常和正常图像校准，推理则使用真实异常，校准与测试数据完全互斥。比较方法为 PatchCore [27]、STFPM [37]、Tiny-Dinomaly [5]、AnomalyGPT [14] 和零样本 Anomaly-OV [42]。注意，Anomaly-OV 完全零样本，不需要任何训练图像；AnomalyGPT 虽基于 VLM，却需要在推理时提供正常参考图像，因此属于无监督场景。

### 5.2 实现细节

**模型。**概念数据集标注流程主要使用 Gemma3 [35] 作为 VLM，仅概念过滤阶段使用 CLIP ViT-B/32 文本编码器 [23]。合成异常生成流程的文生图与编辑系统采用 Gemini 2.5 Flash Image（Nano Banana）。CBM 使用 MobileNet-v2 特征提取器 [28]。

PatchCore 和 STFPM 采用与我们相同的 MobileNet 主干；UNet 也采用 MobileNet-v2 编码器，以确保可比性。Tiny-Dinomaly 使用轻量 DeiT 主干；AnomalyGPT 使用 Vicuna-7B 和 ImageBind-Huge 图像编码器；Anomaly-OV 使用 LLaVA-OneVision 0.5B，均与各自原论文一致。

**数据集。**主要实验结果基于 MVTec AD，VisA 和 Real-IAD 等工业基准上的进一步结果证实了跨数据集鲁棒性。

**超参数优化。**使用 Optuna 库 [1] 实现贝叶斯优化，通过树结构 Parzen 估计器（TPE）算法 [38] 选择最佳超参数配置（详见补充材料）。

**训练细节。**CONVAD 结果取三个不同随机种子运行的平均。推理使用占真实数据集 20% 的留出集。训练和数据增强的其他细节见补充材料。完整实现在开源库 MoViAD [4] 中提供。

### 5.3 指标

VAD 算法可以按对应不同分析层次的多个标准评估：

**图像级：**把每幅图像视为不同样本，用接收者操作特征曲线下面积 AUROC 评价检测，记作 I-AUC。

**像素级：**把每幅图像中的每个像素视为独立样本，计算常用 AUROC，记作 P-AUC。

**概念级：**对 CONVAD，用平均 AUROC 评估概念预测准确性，记作 C-AUC。概念为二值，因此分别计算各概念的 AUROC，再取平均。每项由 $w_j$ 加权，即第 $j$ 个概念 $c_j$ 的真值为真的实例数，$k$ 为概念总数：

$$\mathrm{C\text{-}AUC}:=\frac{\sum_{j=1}^k w_j\cdot\mathrm{AUROC}(c_j)}{\sum_{j=1}^k w_j}.\tag{10}$$

## 6 结果

第 6.1 节在各监督机制下比较 CONVAD 与先进 VAD 方法；第 6.2 节报告感知解释质量的用户研究；第 6.3 节评估作为人机协作机制的概念级干预；最后，第 6.4 节消融概念和视觉分支贡献，并分析像素级性能。

**表 1：MVTec AD 上，CONVAD 与其他 VAD 方法在无监督、半监督和有监督场景中的比较。**✗表示该方法不能生成相应输出（像素定位或概念级预测）。内存单位 MB，↓表示越低越好。为便于阅读，原横向表转置为逐方法行，所有数据保留。

| 场景 | 方法 | I-AUC | P-AUC | C-AUC | 内存 [MB]↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| 零样本 | Anomaly-OV | 0.94 | ✗ | ✗ | 1,768 |
| 无监督（零异常样本） | CONVAD | 0.81 | 0.91 | 0.73 | 17.6 |
| 无监督（零异常样本） | PatchCore | 0.86 | 0.96 | ✗ | 38.4 |
| 无监督（零异常样本） | STFPM | 0.88 | 0.88 | ✗ | 11.1 |
| 无监督（零异常样本） | Tiny-Dinomaly | 0.91 | 0.96 | ✗ | 36.5 |
| 无监督（零异常样本） | AnomalyGPT | 0.97 | 0.93 | ✗ | 30,520 |
| 半监督 | CONVAD | 0.87 | 0.92 | 0.72 | 17.6 |
| 半监督 | DevNet | 0.80 | 0.92 | ✗ | 44.7 |
| 半监督 | DRA | 0.76 | ✗ | ✗ | 44.7 |
| 有监督 | CONVAD | 0.96 | 0.95 | 0.86 | 17.6 |
| 有监督 | UNet | 0.65 | 0.77 | ✗ | 10.3 |

### 6.1 CBM 场景

表 1 比较 CONVAD 与先进 VAD 在 MVTec AD 上的有监督、半监督及无监督结果；VisA 和 Real-IAD 的其他结果见表 2 左侧。

**有监督。**相较传统 UNet 基线，CONVAD 性能更高，同时提供概念解释，使领域专家能够理解预测。虽然有监督设定可访问所有可用异常，它们仍然稀少，限制了 UNet 性能，也突出了我们方法的数据效率。三个基准（MVTec AD、VisA、Real-IAD）的进一步有监督结果证实 CONVAD 的有效性。

CONVAD 与 UNet 内存占用相近（17.6 MB 与 10.3 MB）；但 UNet 即使需要完全监督，检测性能仍显著更低（I-AUC：0.65 与 0.96；P-AUC：0.77 与 0.95），这说明其有限泛化能力使其难以成为实际工业检测的可行选择。

**半监督。**CONVAD 优于两种成熟半监督 VAD 基线 DevNet 和 DRA。这说明其内置的概念归纳偏置可以成为有效正则化，提高对异常训练数据不足的抵抗力，并改善有限监督下的泛化。关键因素是合成异常生成流程，它产生与真实异常域紧密对齐的样本（图 5），即使真实异常有限，也能提供更有代表性的概念监督。细节见第 6.5 节。

**无监督。**CONVAD 的 I-AUC 为 0.81，低于规模相近的 STFPM（0.88）和 Tiny-Dinomaly（0.91），后者内存需求约为两倍（36.5 MB 对 17.6 MB）。我们认为这属于可接受代价，因为这些基线都不提供概念解释，且大部分差距可以通过干预弥补（第 6.3 节）。更高检测得分的方法有明显代价：AnomalyGPT 的 I-AUC 最佳（0.97），但需要超过 30 GB 内存，接近 CONVAD 的 1,750 倍，且在解释质量用户研究中排名最差（第 6.2 节），44.2% 的解释被评为无帮助。类似地，Anomaly-OV 检测较强（I-AUC：0.94），却不生成像素级热力图，而这在工业 VAD 中是基础输出。相比之下，CONVAD 在紧凑内存占用下同时提供概念解释、像素热力图和异常得分，并可利用人在回路中的干预提高推理时检测性能（第 6.3 节）。

**表 2（左）：MVTec AD、VisA 和 Real-IAD 上的平均结果。**比较有监督概念分支、PatchCore（WideResNet-50；虽然无监督，仍是强基线）与 AnomalyGPT（4 个正常样本）。

| 数据集 | CONVAD C-AUC | CONVAD I-AUC | PatchCore I-AUC | AnomalyGPT I-AUC |
| --- | ---: | ---: | ---: | ---: |
| MVTec AD | 0.86 | 0.97 | 0.99 | 0.96 |
| VisA | 0.76 | 0.94 | 0.93 | 0.91 |
| Real-IAD | 0.88 | 0.94 | 0.90 | — |

**表 2（右）：MVTec AD 上不同监督水平的 CONVAD 消融。**报告概念分支、视觉分支及融合的 I-AUC；概念比例为由概念分支驱动最终预测的样本比例。

| 监督水平 | 概念 I-AUC | 视觉 I-AUC | 融合 I-AUC | 概念比例（%） |
| --- | ---: | ---: | ---: | ---: |
| 无监督 | 0.77 | 0.90 | 0.81 | 88.3 |
| 半监督 | 0.85 | 0.91 | 0.87 | 90.1 |
| 有监督 | 0.97 | 0.96 | 0.96 | 91.3 |

**表 3：用户研究结果（582 份回答，32 名参与者）。**左：解释的感知帮助程度。右：解释质量信号；有效表示解释帮助用户验证预测是否正确，误导表示说服用户相信错误预测（按式 (11) 计数）。

| 方法 | 有帮助 | 部分有帮助 | 无帮助↓ |
| --- | ---: | ---: | ---: |
| AnomalyGPT [14] | 33.2% | 22.6% | 44.2% |
| Anomaly-OV [42] | 59.2% | 22.5% | 18.3% |
| CONVAD | 55.6% | 28.6% | 15.8% |

| 信号 | AGPT | A-OV | CONVAD |
| --- | ---: | ---: | ---: |
| 有效 | 33.2% | 60.7% | 73.5% |
| 误导↓ | 18.4% | 11.7% | 3.1% |
| 无用↓ | 48.4% | 27.6% | 23.5% |

### 6.2 用户研究

我们对 32 名参与者开展在线研究，评估 MVTec 和 VisA 图像上的感知解释质量（详见补充材料）。向每名参与者展示产品图像、模型预测、解释，以及可用时的异常热力图，然后询问：(Q1) 预测看起来是否正确？(Q2) 解释是否有帮助？收集 CONVAD、AnomalyGPT [14] 与 Anomaly-OV [42] 的回答，在方法和类别之间平衡，共 582 份。

将两项回答组合成一个质量信号。对某份回答，令 $q_1\in\{\text{是},\text{否},\text{不确定}\}$ 表示用户对预测的判断（Q1），$q_2\in\{\text{是},\text{部分},\text{否}\}$ 表示感知帮助程度（Q2），$m\in\{0,1\}$ 表示预测是否真正符合真值（服务器记录，对用户隐藏）。解释有帮助且用户判断符合真相时为**有效**；有帮助却引导用户违背真相时为**误导**。对 $N$ 份回答，

$$\begin{aligned}
\mathrm{Effective}&=\frac1N\left|\{q_1\ne\text{不确定}\land q_2\ne\text{否}\land(q_1=\text{是}\Leftrightarrow m=1)\}\right|,\\
\mathrm{Misled}&=\frac1N\left|\{q_1\ne\text{不确定}\land q_2\ne\text{否}\land(q_1=\text{是}\Leftrightarrow m=0)\}\right|.
\end{aligned}\tag{11}$$

其余为**无用**（$q_2=\text{否}$ 或 $q_1=\text{不确定}$），故 $\mathrm{Useless}=1-\mathrm{Effective}-\mathrm{Misled}$。在安全关键的工业检测中，有用解释必须使操作人员能够独立验证预测正确性，在误报和漏检向下游传播前同时捕获二者。

表 3 显示，CONVAD 在 84.2% 情况下被评为有帮助或部分有帮助，与 Anomaly-OV 的 81.7% 相当，尽管完全“有帮助”的比例略低（55.6% 对 59.2%）。更重要的是，CONVAD 无帮助比例最低（15.8%，另两者为 18.3% 和 44.2%），误导解释也最少（3.1%，另两者为 11.7% 和 18.4%），表明 Anomaly-OV 在帮助程度上的边际优势伴随着显著更高的误导操作人员风险。与 VLM 方法的事后语言表述相比，概念得分与异常预测的内在耦合减少了解释信号噪声，避免看似合理的文本掩盖错误。

### 6.3 干预

我们在无监督设定下评估概念级干预，因为此时模型对异常域先验知识最少，最可能受益于人类反馈。如图 4，干预概念数量增加时，异常检测性能持续提高；纠正大部分概念后，多数 MVTec 类别接近完美 AUC。这证实人在回路中的改进对实际及安全关键应用有价值，实践者可以在标注投入和检测准确度间权衡。

transistor 和 toothbrush 两类发生退化：二者概念数量少，干预强加的硬标签过度约束预测。尽管如此，总体趋势为正；相较缺乏概念瓶颈、无法暴露结构化人工纠正机制的标准 VAD，这一能力是 CONVAD 的独特优势。

### 6.4 消融研究

表 2 右侧按监督水平报告概念分支、视觉分支与融合的贡献。视觉分支在像素级优于 STFPM：表 1 中 P-AUC 从 STFPM/MobileNet-v2 的 0.88 提升至 0.91，这得益于在概念预测任务上微调教师网络，注入领域知识并改善特征提取 [3]。与 PatchCore 相比，视觉分支以较低 P-AUC（0.91 对 0.96）换取较高 I-AUC（0.90 对 0.86），内存不到其一半，同时加入概念解释。值得注意的是，视觉分支不仅向最终用户提供直观视觉解释，还能在概念分支失效时增强模型鲁棒性。此外，CONVAD 视觉分支基于已证实可有效检测新颖数据的学生教师范式，进一步提高对概念分支可能遗漏的未见异常的鲁棒性。CONVAD 并不绑定特定视觉分支架构，任何基于共享特征提取器的方法都可充当视觉分支，因此方法具有广泛适用性。这里选择 STFPM，是因其高效且具代表性。

**图 4：无监督设定下，随干预概念数量增加，各 MVTec 类别的性能增益。**纵轴为相对于基线的 AUC 增量（百分点）；图例为干预 20%、50%、80% 概念。横轴为 bottle、cable、capsule、carpet、grid、hazelnut、leather、metal_nut、pill、screw、tile、toothbrush、transistor、wood、zipper。

### 6.5 合成到真实的对齐

生成异常是否有用，取决于其概念 logit 表征是否与真实缺陷对齐。我们对视觉复杂的 metal_nut 类别，用二维 t-SNE 嵌入检验这一点（图 5）。无监督时，合成异常形成与正常样本分离的紧密簇，但真实测试异常大多落在正常区域内：生成缺陷与真实缺陷失配，概念迁移较差（C-AUC 0.59、I-AUC 0.66）。这属于困难类别的现象：hazelnut 等简单类别中，合成和真实异常重叠且迁移良好（见补充材料）。每种缺陷只加入一个真实异常（半监督），就会将真实异常拉向合成簇，使 C-AUC 提高至 0.70、I-AUC 至 0.84：极少监督也能有效弥合合成与真实之间的差距。

**图 5：metal_nut 概念 logit 嵌入的 t-SNE，可视化（左）无监督和（右）半监督设定。**颜色编码划分/标签：训练正常、训练异常（生成）、测试正常、测试异常（真实）、训练异常（真实）；形状编码缺陷类型：●正常、■弯曲、◀颜色、＋翻面、★划痕。左侧生成异常与真实测试异常分别聚类；右侧每种缺陷加入一个真实异常后，两者趋于对齐。

## 7 结论

本文提出 CONVAD，将 CBM 适配到 VAD，连接高性能检测与人类可理解的推理。它把提供概念级解释的有监督 CBM 概念分支与像素定位视觉分支相结合，提供现有 VAD 方法所没有的双重可解释性。

在 MVTec AD、VisA 和 Real-IAD 上，CONVAD 在提供透明、以概念为依据的预测时，检测性能可与先进方法相当。概念级干预显著提升性能，证实 CBM 形式支持的人机协作价值。不同监督机制的比较表明，完全使用合成异常的无监督设定也可行，而少量真实异常结合合成缺陷，是稳健且成本有效的策略。

主要局限是依赖异常训练样本：合成生成降低了这种依赖，但还不能与真实异常匹敌。未来将提高合成质量，并利用视觉分支开展新颖性检测，捕获概念分支独自无法发现的未见缺陷，进一步接近完全无监督、可解释且高效的 VAD。我们还计划设计有效统一概念与视觉分支的策略，使其处理相同信号，同时不损失性能。

## 参考文献

[1] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, Masanori Koyama. Optuna：下一代超参数优化框架。Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2623–2631, 2019。

[2] Jinan Bao, Hanshi Sun, Hanqiu Deng, Yinsheng He, Zhaoxiang Zhang, Xingyu Li. BMAD：医学异常检测基准。2024。<https://arxiv.org/abs/2306.11876>。

[3] Manuel Barusco, Francesco Borsatti, Davide Dalle Pezze, Francesco Paissan, Elisabetta Farella, Gian Antonio Susto. PASTE：提高边缘端视觉异常检测效率。Proceedings of the Computer Vision and Pattern Recognition Conference, 4026–4035, 2025。

[4] Manuel Barusco, Francesco Borsatti, Arianna Stropeni, Davide Dalle Pezze, Gian Antonio Susto. MoViAD：视觉异常检测模块化库。arXiv:2507.12049, 2025。

[5] Manuel Barusco, Francesco Borsatti, David Petrovic, Davide Dalle Pezze, Gian Antonio Susto. 边缘端持续视觉异常检测：基准与高效解决方案。arXiv:2604.06435, 2026。

[6] Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger. MVTec AD：面向无监督异常检测的综合真实世界数据集。2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584–9592, 2019。doi:10.1109/CVPR.2019.00982。

[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell 等。语言模型是少样本学习器。Advances in Neural Information Processing Systems, 33:1877–1901, 2020。

[8] Robin Chan, Krzysztof Lis, Svenja Uhlemeyer, Hermann Blum, Sina Honari, Roland Siegwart, Pascal Fua, Mathieu Salzmann, Matthias Rottmann. SegmentMeIfYouCan：异常分割基准。2021。<https://arxiv.org/abs/2104.14812>。

[9] Kushal Chauhan, Rishabh Tiwari, Jan Freyberg, Pradeep Shenoy, Krishnamurthy Dvijotham. 交互式概念瓶颈模型。Proceedings of the AAAI Conference on Artificial Intelligence, 37:5948–5955, 2023。

[10] Jihye Choi, Jayaram Raghuram, Ryan Feng, Jiefeng Chen, Somesh Jha, Atul Prakash. 分布外检测器的概念解释。Proceedings of the 40th International Conference on Machine Learning, 5817–5837. PMLR, 2023 年 7 月。ISSN:2640-3498。<https://proceedings.mlr.press/v202/choi23e.html>。

[11] Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier. PaDiM：用于异常检测与定位的图像块分布建模框架。2020。<https://arxiv.org/abs/2011.08785>。

[12] Hanqiu Deng, Xingyu Li. 通过单类嵌入的反向蒸馏实现异常检测。2022。<https://arxiv.org/abs/2201.10703>。

[13] Choubo Ding, Guansong Pang, Chunhua Shen. 同时捕捉灰天鹅与黑天鹅：开放集有监督异常检测。Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7388–7398, 2022。

[14] Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Ming Tang, Jinqiao Wang. AnomalyGPT：使用大型视觉语言模型检测工业异常。Proceedings of the AAAI Conference on Artificial Intelligence, 38:1932–1940, 2024。

[15] Jinwei Hu, Francesco Borsatti, Arianna Stropeni, Davide Dalle Pezze, Manuel Barusco, Gian Antonio Susto. MIRAGE：面向视觉异常检测、与模型无关的真实感工业异常生成与评估。arXiv:2603.13507, 2026。

[16] Chaoqin Huang, Aofan Jiang, Jinghao Feng, Ya Zhang, Xinchao Wang, Yanfeng Wang. 适配视觉语言模型以实现可泛化的医学图像异常检测。Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11375–11385, 2024 年 6 月。

[17] Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, Percy Liang. 概念瓶颈模型。International Conference on Machine Learning, 5338–5348. PMLR, 2020。

[18] Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, Tomas Pfister. CutPaste：用于异常检测和定位的自监督学习。2021。<https://arxiv.org/abs/2104.04015>。

[19] Tongkun Liu, Bing Li, Xiao Du, Bingke Jiang, Xiao Jin, Liuyi Jin, Zhuo Zhao. 面向可调及逻辑工业视觉检测的部件感知异常检测框架。Advanced Engineering Informatics, 58:102161, 2023 年 10 月。ISSN:1474-0346。doi:10.1016/j.aei.2023.102161。<https://www.sciencedirect.com/science/article/pii/S1474034623002896>。

[20] Zhikang Liu, Yiming Zhou, Yuansheng Xu, Zilei Wang. SimpleNet：用于图像异常检测与定位的简单网络。Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20402–20411, 2023。

[21] Tuomas Oikarinen, Subhro Das, Lam M Nguyen, Tsui-Wei Weng. 无标签概念瓶颈模型。arXiv:2304.06129, 2023。

[22] Guansong Pang, Choubo Ding, Chunhua Shen, Anton van den Hengel. 基于偏差网络的可解释深度少样本异常检测。arXiv:2108.00462, 2021。

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark 等。从自然语言监督中学习可迁移视觉模型。International Conference on Machine Learning, 8748–8763. PMLR, 2021。

[24] Blaž Rolih, Matic Fučka, Danijel Skočaj. SuperSimpleNet：统一无监督与有监督学习，实现快速可靠的表面缺陷检测。International Conference on Pattern Recognition, 47–65. Springer, 2024。

[25] Blaž Rolih, Matic Fučka, Danijel Skočaj. 不遗漏任何标签：适用于所有监督机制的统一表面缺陷检测模型。Journal of Intelligent Manufacturing, 1–21, 2025。

[26] Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net：生物医学图像分割卷积网络。International Conference on Medical Image Computing and Computer-Assisted Intervention, 234–241. Springer, 2015。

[27] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler. 迈向工业异常检测的全面召回。2022。<https://arxiv.org/abs/2106.08265>。

[28] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. MobileNetV2：倒残差与线性瓶颈。Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 4510–4520, 2018。

[29] Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, Georg Langs. 使用生成对抗网络进行无监督异常检测，以指导标志物发现。2017。<https://arxiv.org/abs/1703.05921>。

[30] Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Georg Langs, Ursula Schmidt-Erfurth. f-AnoGAN：使用生成对抗网络进行快速无监督异常检测。Medical Image Analysis, 54:30–44, 2019。ISSN:1361-8415。doi:10.1016/j.media.2019.01.010。<https://www.sciencedirect.com/science/article/pii/S1361841518302640>。

[31] Laya Rafiee Sevyeri, Ivaxi Sheth, Farhood Farahnak, Samira Ebrahimi Kahou, Shirin Abbasinejad Enger. 通过概念解释实现透明异常检测。2023 年 11 月。arXiv:2310.10702 [cs]。<http://arxiv.org/abs/2310.10702>。

[32] Sungbin Shin, Yohan Jo, Sungsoo Ahn, Namhoon Lee. 进一步审视概念瓶颈模型的干预程序。International Conference on Machine Learning. PMLR, 2023。

[33] Divyansh Srivastava, Ge Yan, Lily Weng. VLG-CBM：以视觉语言指导训练概念瓶颈模型。Advances in Neural Information Processing Systems, 37:79057–79094, 2024。

[34] Han Sun, Yunkang Cao, Hao Dong, Olga Fink. 未见视觉异常生成。Proceedings of the Computer Vision and Pattern Recognition Conference, 25508–25517, 2025。

[35] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière 等。Gemma 3 技术报告。arXiv:2503.19786, 2025。

[36] Miles Turpin, Julian Michael, Ethan Perez, Samuel Bowman. 语言模型并不总是说出它们所想的：思维链提示中的不忠实解释。Advances in Neural Information Processing Systems, 36:74952–74965, 2023。

[37] Guodong Wang, Shumin Han, Errui Ding, Di Huang. 用于异常检测的学生教师特征金字塔匹配。2021。<https://arxiv.org/abs/2103.04257>。

[38] Shuhei Watanabe. 树结构 Parzen 估计器：理解算法组件及其在改善经验性能中的作用。arXiv:2304.11127, 2023。

[39] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, F. Xia, Quoc Le, Denny Zhou. 思维链提示激发大型语言模型的推理能力。ArXiv, abs/2201.11903, 2022。<https://api.semanticscholar.org/CorpusID:246411621>。

[40] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, Yanning Zhang. VadCLIP：适配视觉语言模型进行弱监督视频异常检测。Proceedings of the AAAI Conference on Artificial Intelligence, 38:6074–6082, 2024。

[41] Guoyang Xie, Jinbao Wang, Jiaqi Liu, Jiayi Lyu, Yong Liu, Chengjie Wang, Feng Zheng, Yaochu Jin. IM-IAD：制造业工业图像异常检测基准。IEEE Transactions on Cybernetics, 54(5):2720–2733, 2024。

[42] Jiacong Xu, Shao-Yuan Lo, Bardia Safaei, Vishal M Patel, Isht Dwivedi. 迈向使用多模态大型语言模型的零样本异常检测与推理。Proceedings of the Computer Vision and Pattern Recognition Conference, 20370–20382, 2025。

[43] Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, Mark Yatskar. 瓶中语言：语言模型指导的概念瓶颈，用于可解释图像分类。Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19187–19197, 2023。

[44] Zhiyuan You, Lei Cui, Yujun Shen, Kai Yang, Xin Lu, Yu Zheng, Xinyi Le. 多类别异常检测统一模型。Advances in Neural Information Processing Systems, 35:4571–4584, 2022。

[45] Jiawei Yu, Ye Zheng, Xiang Wang, Wei Li, Yushuang Wu, Rui Zhao, Liwei Wu. FastFlow：通过二维归一化流实现无监督异常检测与定位。arXiv:2111.07677, 2021。

[46] Vitjan Zavrtanik, Matej Kristan, Danijel Skočaj. DRAEM：用于表面异常检测的判别式训练重建嵌入。Proceedings of the IEEE/CVF International Conference on Computer Vision, 8330–8339, 2021。

[47] Xuan Zhang, Shiyu Li, Xi Li, Ping Huang, Jiulong Shan, Ting Chen. DeSTSeg：用于异常检测、分割引导的去噪学生教师模型。Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3914–3923, 2023。

[48] Jiaqi Zhu, Shaofeng Cai, Fang Deng, Beng Chin Ooi, Junran Wu. LLM 理解视觉异常吗？揭示 LLM 的零样本异常检测能力。Proceedings of the 32nd ACM International Conference on Multimedia, 48–57, 2024。

## 8 补充材料

> 译注：原 PDF 补充材料从独立页码 1 重新编号，但章节编号继续为 8—17；本译文保留章节、图表与公式的原始编号。

## 9 提示词细节

### 9.1 概念提取提示词

我们在提示中加入上下文线索，指导模型提取有意义的概念：指定图中物体、是否异常，以及存在异常时的具体缺陷。随后采用思维链提示策略，构造两条连续提示：

1. “描述图像，包含所有可见相关特征的信息。只关注能够看到的内容，避免推测或假定。”
2. “根据以下描述，提取五个最有意义的概念。概念的定义应使人观察图像时，能够明确用‘是’或‘否’回答其是否存在。”

需要说明，这一步仅应用于占数据集 5% 的小子集，因此只需少量带标签异常图像。如果没有带标签异常样本，也可对合成图像使用相同流程，但须注意提取概念的质量受限于生成质量。

### 9.2 数据集标注提示词

标注提示与概念提取提示非常相似：每幅图像都包含物体信息，以及异常存在时的异常类型。

“我提供了一幅 {category} 的图像。该图像被分类为 {label}，[这意味着它存在可见缺陷或异常，具体为 {defect type}。] 在已知上述信息的情况下，从以下概念列表中选择图中可以清楚看到的概念。以如下形式的 JSON 对象输出结果：{概念 1: True, 概念 2: False, …}。”

上述提示确保只关注二值概念。

### 9.3 异常生成提示词

利用 Google Nano Banana 的能力生成异常图像。假定已有一组正常图像，以及领域专家提供、希望加入合成图像的缺陷列表。随后查询大型语言模型（本文为 GPT-5），要求为待加入的缺陷提供十个同义表达，以丰富变化，再按以下结构构造提示：

“修改这幅图像，在 {name of the object} 上加入 {type of defect}，保持相同角度、视图和姿态。”

部分对 VLM 尤为困难的类别与缺陷类型，需要人工调整提示，并丢弃部分生成图像。所得合成图像示例见图 6。

**图 6：生成质量较好的合成异常图像示例。**图内示例分别是 leather/color（皮革/颜色）、transistor/bent_lead（晶体管/引脚弯曲）、pill/combined（药片/组合缺陷）、wood/scratch（木材/划痕）、screw/manipulated_front（螺钉/前端变形）、carpet/hole（地毯/孔洞）。

## 10 概念标注流程评估

除正文报告的指标外（自动标注流程在 400 幅人工标注图像子集上准确率 91.3%），我们还将 hazelnut 类别的完整自动标注数据集与其真值版本比较，并计算精确率和召回率以作更详细分析。表 4 概括结果。总体上，预测概念质量良好，尤其是更常见于正常图像的概念；异常概念的精确率和召回率变化很大，主要涉及描述图像弥散特征的概念，例如表面不连续、色调不均等。这些特征更难被 VLM 捕获，领域专家的判断也可能有主观性。

**表 4：hazelnut 类别预测概念的准确率、精确率与召回率均值和标准差。**

| 指标 | 全部 | 异常概念 | 正常概念 |
| --- | ---: | ---: | ---: |
| 准确率 Acc | 0.93 ± 0.08 | 0.92 ± 0.08 | 0.98 ± 0.01 |
| 精确率 Prec | 0.76 ± 0.24 | 0.71 ± 0.23 | 0.99 ± 0.00 |
| 召回率 Rec | 0.77 ± 0.28 | 0.72 ± 0.28 | 0.99 ± 0.01 |

## 11 其他训练细节

每个模型最多训练 100 个 epoch，采用十个 epoch 后触发的早停机制。性能在五个 epoch 后进入平台期时，允许将学习率降低为原来的十分之一。

**基于变换的数据增强。**施加变换以提高泛化，并谨慎选择能保留各概念是否存在的变换，确保属性标签完整。具体包括：以 0.5 概率水平和垂直翻转、随机旋转 25 度、幅度 0.2 的亮度抖动和幅度 0.2 的对比度抖动。

**CBM 训练。**在 CNN 特征提取器后接 $k$ 个并行线性层作为概念预测器；用含八个神经元的简单前馈神经网络检测异常。特征提取器先直接在目标数据集上预训练，以多类分类目标学习图中物体类别；概念预测训练期间微调其最后几层。

> 译注：正文第 4.2.2 节用“最终线性层”权重定义式 (7) 的概念贡献；此处又将异常预测器描述为含八个神经元的前馈网络。原文没有在这两处明确解释它们的对应关系，译文分别保留。

## 12 CBM 训练模式

除正文介绍的联合范式外，CBM 还可采用另外两种模式训练。

**独立训练：**先训练概念提取器，从输入预测概念：

$$\hat g=\arg\min_g\sum_{i,j}\mathcal L_{C_j}(g_j(x^i),c_j^i),\tag{12}$$

其中 $\mathcal L_{C_j}$ 为第 $j$ 个概念预测损失。另行使用真实概念向量 $c$ 作为输入，训练标签预测器：

$$\hat f=\arg\min_f\sum_i\mathcal L_Y(f(c^i),y^i),\tag{13}$$

其中 $\mathcal L_Y$ 为下游标签预测损失。注意部署期间，$\hat f$ 的输入是预测概念 $\hat c=\hat g(x)$，而非真实概念。这可能造成训练与推理输入分布偏移，从而降低性能。

**顺序训练：**先按独立设定训练 $g$；然后直接以预测概念 $\hat c=\hat g(x)$ 而非真实概念训练 $f$。

表 5 比较三种 CBM 范式在 MVTec AD 上的结果。鉴于这些发现，我们采用联合范式训练，因为其与独立范式相近，但 I-AUC 略优。

**表 5：MVTec AD 上三种 CBM 学习范式的概念预测与异常检测性能。**

| 模型类型 | C-AUC | I-AUC |
| --- | ---: | ---: |
| 联合 Joint | 0.86 | 0.97 |
| 顺序 Sequential | 0.85 | 0.95 |
| 独立 Independent | 0.87 | 0.96 |

## 13 融合与校准策略消融

我们在 MVTec AD 无监督设定下分析：(i) 视觉和概念分支得分校准；(ii) 得分融合。表 6 报告结果，包括与概念分支的一致率（Agree），即组合得分的二值预测与概念分支相同的样本百分比。

(i) **校准：**将视觉和概念异常得分归一化为各自经验测试分布的百分位秩，无需监督即可映射到共同尺度。与在小型留出集上训练的有监督保序回归相比，基于秩的归一化持续表现更好。

(ii) **融合：**比较所提最大值算子、算术平均、对两个校准得分作逻辑回归，以及额外使用概念概率的逻辑回归。虽然学习式融合取得更高 AUROC，但持续降低与概念分支的一致性。由于解释仅忠实于概念分支，我们保留最大值算子，将改善融合策略留待未来研究。

**表 6：MVTec AD 无监督设定下，校准策略（列）和融合策略（行）的消融。**

| 融合 | 秩校准 AUROC | 秩校准 Agree | 保序回归 AUROC | 保序回归 Agree |
| --- | ---: | ---: | ---: | ---: |
| Max 最大值 | 0.81 | 87.31% | 0.76 | 84.62% |
| Mean 平均值 | 0.88 | 82.95% | 0.87 | 73.29% |
| Learned 学习式 | 0.90 | 79.49% | 0.87 | 73.28% |
| Learned+C 学习式加概念 | 0.91 | 76.52% | 0.88 | 78.50% |

> 译注：正文第 4.2.2 节描述先以留出集最大化 F1 阈值二值化视觉得分，再按式 (9) 与概念概率取最大；本节则讨论对经验测试分布进行百分位秩校准。这些表述并不完全相同；此外表 6 的 87.31% 与表 2 的 88.3% 也不同。译文分别保留，未将其统一为同一实验定义或数值。

## 14 用户研究其他细节

参与者为 32 名研究生/研究人员，无需领域专长。通过每名参与者的确定性种子和逆频率采样分配样本，在方法、数据集、类别和标签之间平衡；不同方法的刺激材料大多不重叠（图像复用 7.7%），各方法保持自身自然正确/错误比例，未强制配比。CONVAD 相对于 AnomalyGPT 的帮助程度优势具有显著性（$\chi^2$，$p<0.001$；以参与者为簇的 bootstrap 95% 置信区间不含零），与 Anomaly-OV 则相当（$p=0.36$）。我们承认两项局限：只有 CONVAD/AnomalyGPT 提供热力图，Anomaly-OV 不生成；各方法刺激材料未配对匹配。

## 15 干预程序

第 6.3 节展示推理时手动将预测概念改为真值对性能的影响。但该程序需要专家监督，可能成本较高，因此已有若干策略通过优先关注更有可能提升性能的重要概念来降低成本。我们遵循 [32] 的 UCP（不确定概念预测）启发式，计算每个概念预测基于熵的不确定性：

$$H(c_i)=-\left(p_i\cdot\log(p_i)+(1-p_i)\cdot\log(1-p_i)\right).\tag{14}$$

随后按熵降序排列，基于如下想法：预测越不确定的概念，越可能混淆模型并导致错误预测。如第 3 节所述，联合模型用预测概念 logit 执行主任务，因此将其替换为训练分布第 5 或第 95 百分位数。

## 16 概念 logit 分析

在完全使用合成异常训练、使用真实异常评估的无监督设定中，部分类型获得较高概念级 AUC（C-AUC）和图像级 AUC（I-AUC），其他类型却无法有效迁移到真实数据。两个极端例子是迁移良好的 hazelnut 和迁移不佳的 metal nut。我们研究训练和测试样本的概念 logit 嵌入，以分析差异。测试集仅含真实正常与真实异常；训练集在 SAG 场景中可能包含合成图像。同类别内比较的所有模型使用相同随机种子、训练数据和评估集，且评估集只包含未被任何模型用于弱监督的真实异常。

良好的瓶颈应确保：(i) 若概念集足够有预测力，正常与异常样本应分离；(ii) 相同或相近缺陷类型（例如榛子的“孔洞”或“裂纹”）应紧密聚类。此外，如果合成异常与真实异常域充分对齐，它们的概念 logit 表征应与真实异常一致，表明迁移有效。为获得定性认识，我们绘制二维 t-SNE 概念 logit 可视化。

hazelnut 的 SAG 场景中，平均 C-AUC 为 0.89、I-AUC 为 0.99。图 7 显示训练正常与训练异常（合成）明确分离。测试正常与训练正常位于相同区域，这是预期的，因为二者都是实际图像、域一致。另外，测试异常（真实）被映射到合成训练异常所在的大致区域，这是理想情况。在异常区域内，“print”形成独立子簇，而“crack”“cut”“hole”的嵌入重叠更多，反映其视觉相似性。这些模式与高 AUC 一致，说明为榛子生成的合成异常有效捕获真实异常的相关特征。

但 metal nut 的 SAG 场景中，平均 C-AUC 降至 0.59、I-AUC 至 0.66。如图 8(a)，多数真实测试异常位于训练正常和测试正常区域，而非靠近合成训练异常。仅“flip”因与其他图像视觉差异较大而形成略微分离的簇，其他缺陷散落于正常样本之间。合成与真实异常嵌入的这种定性失配解释了 AUC 下降，也提示异常生成质量不佳。

**图 7：SAG 场景中 hazelnut 的 t-SNE 嵌入。**颜色：训练正常、训练异常、测试正常、测试异常。形状表示缺陷类型：正常、裂纹、切口、孔洞、印记。

**图 8：metal_nut 的概念 logit 嵌入 t-SNE 表征。**黑色标记突出有监督加 SAG 场景中使用的少量真实样本。(a) SAG；(b) 有监督（每种缺陷 1 个异常样本）＋SAG；(c) 有监督（每种缺陷 3 个异常样本）＋SAG。颜色图例为训练正常、训练异常、测试正常、测试异常、训练异常（真实数据）；形状为正常、弯曲、颜色、翻面、划痕。

分析有监督＋SAG 的嵌入，可更深入理解为何少量真实异常与合成异常结合就能缩小性能差距。对于 metal nut，在有监督（每种缺陷 1 个异常样本）＋SAG 场景中，平均 C-AUC 提升至 0.70、I-AUC 至 0.84，图 8(b) 显示合成与真实异常更好对齐。采用有监督（每种缺陷 3 个异常样本）＋SAG，相较标准半监督每缺陷一个样本，C-AUC 增至 0.76、I-AUC 至 0.89。同时，真实异常表征更接近合成异常，不同缺陷簇分离更明确（图 8(c)）。这些结果说明，使用合成异常训练时，极少真实数据监督也能缓解域偏移，促使概念空间编码更一致的缺陷特定结构，无需收集大量稀有且难以获取的异常。

## 17 其他概念词表分析

我们报告三个基准中代表性物体类别的概念分布（图 9—12），说明提取概念如何捕获有意义的数据特征。这些分布突出正常与异常样本之间的明确差异，表明针对特定缺陷类型激活的概念，可从可解释性角度帮助识别故障性质。

**图 9：capsule 类别（MVTec AD）的概念分布。**行标：正常、戳孔、印刷不良、挤压、划痕、裂纹。横轴概念：均匀纹理、光滑表面、穿刺痕迹、可见裂隙、印字模糊、表面磨损、外观不对称、压痕、表面凹陷、双色外观、字符变形、边缘变形、局部凹痕、形状变形、线状缺陷、轮廓扭曲。

**图 10：metal_nut 类别（MVTec AD）的概念分布。**行标：正常、弯曲、划痕、颜色、翻面。横轴概念：规则图案、内螺纹、圆形轮廓、圆润边缘、银色、六边形形状、暗色、面错位、轮廓弯曲、螺纹变形、线状痕迹、轮廓扭曲、划伤区域、颜色不均、几何不规则、表面磨损、金属表面损伤。

**图 11：pcb2 类别（VisA）的概念分布。**行标：正常、熔化、缺失、划痕、弯曲＋熔化、熔化＋划痕、弯曲、划痕＋缺失。横轴概念：蚀刻残留、元件焊接、裸铜、表面残留、未装件焊盘、空洞、边界不规则、线状痕迹、弯曲元件、变形表面、倾斜表面、表面磨损、划伤区域。

**图 12：rolled_strip_base 类别（Real-IAD）的概念分布。**行标：正常、凹坑、污染、缺件。横轴概念：均匀颜色、笔直边缘、卷边、平底、反射高光、暗斑、表面沉积物、边缘不完整、圆形凹陷、下陷区域、凹陷性、变色区域、表面凹痕。
