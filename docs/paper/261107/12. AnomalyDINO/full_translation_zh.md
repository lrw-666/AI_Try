# AnomalyDINO：利用 DINOv2 增强基于图像块的少样本异常检测

**原文：**[AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2](<AnomalyDINO- Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2.pdf>)。本译文对应本目录 11 页 WACV 2025 正式论文集开放获取 PDF，原刊页码 1319—1329；除开放获取水印外，与接收版本相同，最终论文集版本见 IEEE Xplore。翻译日期：2026-11-07。

**作者：**Simon Damm¹、Mike Laszkiewicz¹、Johannes Lederer²、Asja Fischer¹。

**单位：**¹德国波鸿鲁尔大学计算机科学系；²德国汉堡大学数学、计算机科学与自然科学学院。邮箱：{simon.damm, mike.laszkiewicz, asja.fischer}@rub.de；johannes.lederer@uni-hamburg.de。

**翻译说明：**覆盖本地 PDF 摘要、正文第 1—5 节、致谢、图 1—3、表 1—4 全部行列、公式 (1)—(4)、脚注 1—5 和参考文献 [1]—[51]。文中引用的附录 A—D 并未附在这份 11 页 PDF 中，译文保留相关交叉引用，不将其视为本地已提供材料。图形不重绘，图注与关键标注翻译；合并断行续段、去除运行页码。所有数值保留，“最佳”均指论文发表时。

## 摘要

多模态基础模型的近期进展为少样本异常检测设立了新标准。本文探索仅靠高质量视觉特征是否足以媲美现有最先进视觉—语言模型。我们通过将 DINOv2 适配用于单样本和少样本异常检测，并聚焦工业应用，肯定了这一点。这种方法不但可与现有技术竞争，在许多设置下甚至能够超过它们。纯视觉方法 AnomalyDINO 遵循成熟的图像块级深度最近邻范式，可实现图像级异常预测及像素级异常分割。其方法简单、无需训练，因而不需要额外数据作微调或元学习。尽管简单，它在单样本、少样本异常检测中达到先进结果，例如将 MVTec-AD 单样本 AUROC 从 93.1% 提高到 96.6%。较低额外开销结合优秀少样本性能，使其成为工业等环境快速部署的有力候选。

## 1 引言

机器学习中的异常检测（AD）尝试识别明显偏离正常数据分布 $p_{norm}(x)$ 的实例。异常因而使人怀疑其“由不同机制生成”[16]，通常指示关键、罕见或未预见事件。可靠区分异常与正常样本的能力，在安全 [40]、医疗 [14,41]、工业检查等领域很有价值。本文聚焦工业：全自动系统需要检测缺陷或缺失部件，防止下游产品失灵、对潜在危险报警，或分析问题以优化生产线。图 1 右侧给出此环境中的异常样本。

过去几年，工业图像异常检测受到极大关注。基准上接近最优的结果似乎表明该问题已基本解决。例如 Mousakhan 等 [28] 在常见 MVTec-AD [2]、VisA [51] 上分别报告 99.8%、98.9% AUROC。最流行技术用训练数据训练异常分类器 [38]，或训练生成模型，结合基于重建 [26,28,42] 或似然 [9,35] 的异常评分。但这些方法属于全样本设置，依赖足量训练数据。考虑数据集获取困难、快速易部署方法的吸引力，以及快速适应正常分布协变量偏移的要求 [22]，少样本和零样本异常检测受到越来越多关注。

然而，少样本方法高度依赖有意义特征，或如 [33] 所述，“异常检测需要更好的表示”。随着基础模型——以无监督/自监督方式在海量数据上训练的大规模模型 [5,30,31]——的可用性与能力提升，这类表示已可获得。基础模型已经提高了少样本检测性能，主要是结合语言与视觉的多模态方法 [4,19,21,50]。

本文转而聚焦纯视觉方法。动机是，人类标注者仅凭视觉特征便能完成少样本异常检测，不需要对象或预期异常类型的额外文字描述，而异常类型通常事先未知。

**图 1：AnomalyDINO 基于单个完好参考样本进行异常检测。**示例为 MVTec-AD 的 Screw。将可能增强后的参考样本正常图像块表示收集到记忆库 $\mathcal M$。测试时，如适用则通过掩码选择相关图像块。它们与库中正常表示的距离形成异常图，并通过统计量 $q$ 聚合成异常分数 $s(x_{test})$。掩码与特征提取均使用 DINOv2（$f$）。右侧及附录 A 图 4—7 展示其他类别。图内文字依次为参考样本、可选增强、收集正常图像块表示、推理、测试样本、可选掩码、最近邻匹配、异常图与分数。

AnomalyDINO 遵循成熟的图像块级深度最近邻 AD 框架 [34,46]，以 DINOv2 [30] 为主干。我们为单样本场景仔细设计预处理，使用 DINOv2 的零样本分割能力，从而避免另一个分割模型的额外开销。测试时，根据图像块表示到正常记忆库最近对应表示的较大距离检测异常。

由于简单，与 [7,21] 等复杂方法相比，它极易部署于工业。尽管如此，它在 MVTec-AD [2] 少样本异常检测中达到新的先进性能，在 VisA [51] 超过除一种以外的全部竞争方法。

本文结构如下：第 2 节回顾相关研究，区分零样本、少样本及批量零样本技术的设置；第 3 节介绍 AnomalyDINO；附录 D 详述其批量零样本扩展；第 4 节给出实验。附录 A、C 分别提供更多结果与消融，附录 B 讨论失败案例。复现实验代码：<https://github.com/dammsi/AnomalyDINO>。

**贡献。**

- 提出简单、免训练而高效的图像块视觉异常检测技术 AnomalyDINO，建立于 DINOv2 的高质量特征表示之上。
- 广泛分析证明效率和有效性，在性能及推理速度上超过其他多模态少样本技术。具体地，在 MVTec-AD 达到先进少样本结果，如单样本 AUROC 从 93.1% 提高到 96.6%，使少样本与全样本差距减半。VisA 结果不仅可与其他少样本方法竞争，还为全部免训练少样本检测器建立新的先进水平，并在所有方法中取得最佳定位。

## 2 相关工作

**视觉基础模型。**多模态基础模型已成为众多任务的强大工具，见 [3,6,17,23,25,29,31]。与视觉 AD 最相关的有 CLIP [31] 或近期 LLM [29] 多模态方法，也有 DINO [5,30] 等纯视觉方法。CLIP [31] 在带文本标注的图像对数据集上训练，从自然语言描述学习视觉概念。其对比目标对齐图像、文本编码器的嵌入，优化对应图文对相似性。视觉—语言公共空间可用于多个下游任务，例如通过计算与一组类别提示的相似性，实现零样本图像分类。

DINO [5,30] 采用基于视觉 Transformer [12] 的自监督师生框架，以多视图策略训练 ViT [11] 预测软化的教师输出，学习鲁棒、高质量下游特征。DINOv2 [30] 结合 DINO 与图像块级重建技术 [49]，扩展至更大架构与数据集。DINO 特征同时包含局部、全局信息，对多视图与裁剪鲁棒，并受益于大规模预训练，因此适合异常检测。GroundingDINO [25] 建立于 DINO 框架，重点改进文本与视觉对齐，增强详细对象定位及多模态理解。

> 译注：上一句关于 GroundingDINO 与 DINO 框架的关系为原文表述。本文不据此额外推导其与 DINOv2 自监督方法的架构或训练关系。

**异常检测。**给定预定正常性概念，AD 检测偏离该概念的测试样本 [36,45]。本文聚焦工业图像的低层感知异常，不针对语义异常，而针对工业产品图像的划痕等低层特征（如图 1）。已有工作训练异常分类器 [38]，或训练支持重建/似然 AD 的生成模型 [9,26,28,42,48]。

成熟的深度最近邻 AD 按测试实例到正常实例特征记忆库 $\mathcal M$ 中最近邻的距离评分。至少可追溯至 2002 年 [13]，后来先在图像级 [1] 普及，继而出现开创性图像块方法：Patch SVDD [46] 用 DeepSVDD [37] 与自监督损失混合训练特征提取器；PatchCore [34] 使用 ImageNet 预训练分类模型提取图像块特征。更多方法建立在这一简单而有效的 AD 方式之上 [8,24,43]。除 kNN，也使用马氏距离等“经典”方法 [10]。显然，性能高度依赖特征提取器 $f$。传统选择为 ResNet [18]、Wide ResNet [47]、ViT [11]，多以监督任务预训练。

另一类工作基于预训练视觉—语言模型的零样本分类成功。先定义描述正常和异常的提示集，再将文本嵌入与图像嵌入比较 [7,19,21,50]。视觉嵌入接近异常提示文本嵌入的图像被判异常。但这些方法要么需要大量提示工程（如 [7] 描述正常样本就用 $35\times7$ 个提示），要么需要微调提示或其嵌入。另一类少样本 AD 基于多模态聊天机器人，需要更复杂提示及文本输出解释技术 [44]。这类方法不需要记忆库，因此能作零样本 AD。

**少样本/零样本检测器分类。**既有工作采用不同 AD 设置，使评估比较复杂。我们按样本数设置、训练要求、基础模型模态给出分类。样本数设置有零样本、少样本、批量零样本。前两者由预测测试样本前可处理的正常训练样本数量区分；批量零样本不逐例推理，而基于整个测试批次，通常是完整测试集。例如 [24] 利用大多数像素正常的事实，在一批图像之间匹配图像块。[22] 则基于批归一化作用，部署无参数异常检测器。

训练要求分为免训练、微调、元训练。免训练不需要训练；微调使用少数可访问样本修改模型；元训练是在与测试数据相关的数据集上训练。例如 [22] 在 MVTec-AD 除被测类别外的所有类别训练；[50,7] 测 MVTec-AD 时在 VisA 训练，反之亦然。最后区分视觉模型（如预训练 ViT）与视觉—语言模型（如 CLIP）。详见表 1。

**表 1：近期少样本、零样本 AD 方法分类。**† 表示最初提出为全样本检测器、后被当作少样本检测器评估的方法，如 [34]。表内 GNN 的训练类型原样保留。

| 方法 | 设置 | 训练类型 | 模态 |
| --- | --- | --- | --- |
| DN2 [1] | 少样本 | 免训练 | 视觉 |
| SPADE [8] | 少样本† | 免训练 | 视觉 |
| PaDiM [10] | 少样本† | 微调 | 视觉 |
| PatchCore [34] | 少样本 | 免训练 | 视觉 |
| PatchCore–opt [39] | 少样本 | 免训练 | 视觉 |
| MuSc [24] | 批量零样本 | 免训练 | 视觉 |
| GraphCore [43] | 少样本 | 训练（GNN） | 视觉 |
| WinCLIP [19] | 零/少样本 | 免训练 | 视觉＋语言 |
| APRIL-GAN [7] | 零/少样本 | 元训练 | 视觉＋语言 |
| ADP [21] | 少样本 | 微调 | 视觉＋语言 |
| AnomalyCLIP [50] | 零样本 | 元训练 | 视觉＋语言 |
| GPT4-V [44] | 零样本 | 免训练 | 视觉＋语言 |
| ACR [22] | 批量零样本 | 元训练 | / |
| AnomalyDINO（本文） | 少样本 | 免训练 | 视觉 |

## 3 匹配图像块表示以实现视觉异常检测

本节介绍利用 DINOv2 提取有意义图像块特征的 AnomalyDINO。基于成熟深度最近邻方法 [1,8,13,24,34,43,46]，先将相关正常图像块表示收集到记忆库 $\mathcal M$，再计算各测试图像块到库中最近正常图像块的距离。适当聚合这些距离，得到图像级异常分数。

与先前方法 [34,46] 的区别是，将记忆库概念专门适配到少样本，利用 DINOv2 强特征，设计结合零样本掩码与增强的流程，并提出更鲁棒聚合统计量。DINOv2 具有强图像块特征与掩码能力，是此场景理想主干。它还降低特征工程复杂性，无需考虑选哪些表示/层及如何聚合，从而简化框架；同时增强输入分辨率灵活性，可选任意 14 的倍数。以下详述方法。

### 3.1 通过图像块（不）相似性检测异常

简要回顾图像块深度最近邻 AD。假定有合适提取器 $f:\mathcal X\to\mathcal F^n$，将图像 $x\in\mathcal X$ 映射为图像块特征元组 $f(x)=(\mathbf p_1,\ldots,\mathbf p_n)$，$\mathcal X$ 为图像空间，$\mathcal F$ 为特征空间。$n$ 取决于分辨率与图像块大小（DINOv2 为 $14\times14$ 像素）。给定 $k\ge1$ 个正常参考样本 $X_{ref}:=\{x^{(i)}\mid i\in[k]\}$，其中 $[k]:=\{1,\ldots,k\}$，收集正常图像块并存入：

$$
\mathcal M:=\bigcup_{x^{(i)}\in X_{ref}}\{\mathbf p_j^{(i)}\mid f(x^{(i)})=(\mathbf p_1^{(i)},\ldots,\mathbf p_n^{(i)}),\ j\in[n]\}.\tag{1}
$$

对测试样本 $x_{test}$，提取 $f(x_{test})=(\mathbf p_1,\ldots,\mathbf p_n)$，检查其与库的一致程度。对测试图像块 $\mathbf p\in\mathcal F$，用最近邻得到与参考图像块的最小距离：

$$d_{NN}(\mathbf p;\mathcal M):=\min_{\mathbf p_{ref}\in\mathcal M}d(\mathbf p,\mathbf p_{ref}).\tag{2}$$

实验采用余弦距离：

$$d(\mathbf x,\mathbf y):=1-\frac{\langle\mathbf x,\mathbf y\rangle}{\|\mathbf x\|\|\mathbf y\|}.\tag{3}$$

以合适统计量 $q$ 聚合，得到图像级分数：

$$s(x_{test}):=q(\{d_{NN}(\mathbf p_1;\mathcal M),\ldots,d_{NN}(\mathbf p_n;\mathcal M)\}).\tag{4}$$

全文将 $q$ 定义为最异常 1% 图像块的平均距离，即 $q(\mathcal D):=\operatorname{mean}(H_{0.01}(\mathcal D))$，其中 $H_{0.01}$ 含集合最高 1% 值。这可理解为 99% 分位数尾部风险值的经验估计 [27]，适合多种设置，因为兼顾两个要求：希望分数依赖最大图像块距离，它们可能提供最强异常信号；也希望对单个极大距离有一定鲁棒性，少样本稀疏记忆库尤其如此。但可以换统计量以满足特殊情况。例如，预期异常覆盖图像大部分时，高百分位数可能合适；若异常很局部、只影响少数甚至一个图像块，$q$ 应对最高距离敏感。通常采用最大像素异常分数；像素级分数一般由上采样至整图分辨率并平滑获得。

遵循 [34]，用双线性上采样及高斯平滑（$\sigma=4.0$）将图像块距离转为像素分数，定位潜在缺陷。图 1 和附录 A 图 4—7 展示异常图。另将 AnomalyDINO 扩展至批量零样本，见附录 D 图 18。

### 3.2 丰富记忆库与过滤相关图像块

少样本 AD 的主要挑战是从有限正常样本有效捕获正常性概念。一种有效策略是使用旋转等简单增强（受 [43] 启发），增加正常图像块变化。测试变化很可能显著大于参考数据 $X_{ref}$，因此此点关键。全样本方法则相反，试图缩小库以降低推理时间，如 [34]。

为避免无关测试区域导致虚高分数，我们建议为目标物体设置掩码。这能降低误报，尤其少样本参考无法充分捕获背景自然变化时，见附录 C 图 10。是否掩码、增强，应取决于物体具体特点。设计有效预处理的挑战与考虑详见附录 C.1。

**掩码。**从背景中划分主要物体，可减少误报并提高低数据条件的鲁棒性。为最小化开销，也用 DINOv2 掩码：对图像块特征第一主成分作阈值化 [30]。经验上有时掩码错误，常见于物体占图像块约 50% 以上的近景。为此，检查 PCA 掩码是否正确捕获第一个参考样本的物体，再据此决定是否应用，形成图 2“掩码测试”。每个物体只测一次，因为输出很一致。另外用膨胀和形态学闭运算去除小孔洞、间隙。附录 C 图 14、15 是示例，表 8 为逐物体测试结果，图 10 展示背景噪声下的益处。一般不对纹理（如 MVTec-AD 的 Wood、Tile）掩码。

**图 2：MVTec-AD 掩码测试。**上行 Capsule、Hazelnut 掩码成功；下行 Cable、Transistor 某些本应属于物体的区域被误判背景，以红色突出。逐物体结果见附录 C.1。

**旋转。**旋转参考样本能更好模拟库内正常性变化，可能改善检测。例如图 1 的 Screw 需要旋转不变特征；但有时旋转有害，因为目标整体或局部旋转本身可能就是异常，见图 13。

考虑两种设置。正文聚焦“不知情”（agnostic）设置，始终对参考样本作旋转增强。也考虑知道潜在旋转的“知情”（informed）设置。工业/医疗通常可控制采集过程或对齐测试图，因此知情设置合理，也可能降低推理时间。掩码、旋转效果及两设置比较见附录 C.1。

## 4 实验

**AnomalyDINO 默认设置。**DINOv2¹ 主干可选不同蒸馏尺寸，从小型 ViT-S（$21\times10^6$ 参数）至巨型 ViT-G（$1.1\times10^9$ 参数）。为低延迟，默认用最小模型，记为 AnomalyDINO-S，评估输入短边 448、672 像素两种分辨率。默认用上述“不知情”预处理，见附录 C。

¹ DINOv2 [30] 代码/权重采用 Apache 2.0 许可。

**数据集。**实验采用工业高分辨率图像数据集 MVTec-AD² 与 VisA³。MVTec-AD 有十五类，表现单物体或纹理，每类最多八种异常。VisA 有十二类，将多样异常统一为“bad”类别。其若干类别存在多个物体和复杂结构，可视为更具挑战性。

² MVTec-AD [2] 采用 CC BY-NC-SA 4.0 许可。

³ VisA [51] 采用 CC BY 4.0 许可。

**评估指标。**按 [7,19,24]，图像检测与像素分割各用三个指标。检测用图像异常分数计算 AUROC、最优阈值 F1（F1-max）、平均精度 AP；分割用像素分数计算 AUROC、F1-max、逐区域重叠 PRO [2]。正常、异常像素高度不平衡，MVTec-AD 正常像素 97.26%，VisA 更达 99.45% [24]，因此不建议只用分割 AUROC [32]。每实验重复三次，报告均值与标准差⁴。

⁴ 随机性来自参考样本 $X_{ref}$。为便于复现，三次运行分别取训练集第一组 $k$、第二组 $k$、第三组 $k$ 个样本。

**基线。**比较 SPADE [8]、PaDiM [10]、PatchCore [34]、WinCLIP+ [19]、APRIL-GAN [7] 等现代零/少样本模型。ACR [22]、MuSc [24] 研究批量零样本（表 1），与本文不同，附录 D 将本文方法适配至该设置。APRIL-GAN、AnomalyCLIP 需要相关数据集训练，与本文免训练不同。表 2、3 中 † 为 [19] 结果，* 为 WinCLIP 重实现 [21]（J. Jeong 亦为共同作者）；其余取原始论文。ADP [21] 报告通常稍好的 $ADP_\ell$（可访问类别标签）。GraphCore [43] 未报告标准差。

### 4.1 少样本异常检测与异常分割

**结果。**表 2、3 分别汇总 MVTec-AD、VisA。MVTec-AD 上，在所有 $k\in\{1,2,4,8,16\}$ 和所报告检测指标中均达到先进性能，超过需要额外训练集的 ADP、APRIL-GAN。定位也优秀：448 与 672 分辨率的检测相近，但更高分辨率提升定位。样本数增加时也明显改善。

**表 2：MVTec-AD 异常检测。**图像级分类与像素级分割的定量结果，单位均为 %。原表以粗体表示各样本数最佳、下划线表示第二好方法、灰框表示最佳免训练结果（见表 1）；此处保留全部数值及标记，改为统一 Markdown 表格。GraphCore 的“±/”表示未报告标准差，“/”为原表缺项。

| 样本数 | 方法 | 分类 AUROC | 分类 F1-max | 分类 AP | 分割 AUROC | 分割 F1-max | 分割 PRO |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | SPADE† | 82.9±2.6 | 91.1±1.0 | 91.7±1.2 | 92.0±0.3 | 44.5±1.0 | 85.7±0.7 |
| 1 | PatchCore† | 83.4±3.0 | 90.5±1.5 | 92.2±1.5 | 92.0±1.0 | 50.4±2.1 | 79.7±2.0 |
| 1 | GraphCore | 89.9±/ | / | / | 95.6±/ | / | / |
| 1 | WinCLIP+ | 93.1±2.0 | 93.7±1.1 | 96.5±0.9 | 95.2±0.5 | 55.9±2.7 | 87.1±1.2 |
| 1 | APRIL-GAN | 92.0±0.3 | 92.4±0.2 | 95.8±0.2 | 95.1±0.1 | 54.2±0.0 | 90.6±0.2 |
| 1 | AnomalyDINO-S (448) | 96.5±0.4 | 96.0±0.2 | 98.1±0.3 | 96.3±0.1 | 57.9±0.8 | 91.7±0.1 |
| 1 | AnomalyDINO-S (672) | 96.6±0.4 | 95.8±0.5 | 98.2±0.2 | 96.8±0.1 | 60.2±1.1 | 92.7±0.1 |
| 2 | SPADE† | 81.0±2.0 | 90.3±0.8 | 90.6±0.8 | 91.2±0.4 | 42.4±1.0 | 83.9±0.7 |
| 2 | PatchCore† | 86.3±3.3 | 92.0±1.5 | 93.8±1.7 | 93.3±0.6 | 53.0±1.7 | 82.3±1.3 |
| 2 | GraphCore | 91.9±/ | / | / | 96.9±/ | / | / |
| 2 | WinCLIP+ | 94.4±1.3 | 94.4±0.8 | 97.0±0.7 | 96.0±0.3 | 58.4±1.7 | 88.4±0.9 |
| 2 | ADPℓ | 95.4±0.9 | / | / | / | / | / |
| 2 | APRIL-GAN | 92.4±0.3 | 92.6±0.1 | 96.0±0.2 | 95.5±0.0 | 55.9±0.5 | 91.3±0.1 |
| 2 | AnomalyDINO-S (448) | 96.7±0.8 | 96.5±0.4 | 98.1±0.7 | 96.5±0.2 | 58.5±0.5 | 92.0±0.2 |
| 2 | AnomalyDINO-S (672) | 96.9±0.7 | 96.1±0.3 | 98.2±0.5 | 97.0±0.2 | 61.0±0.5 | 93.1±0.2 |
| 4 | SPADE† | 84.8±2.5 | 91.5±0.9 | 92.5±1.2 | 92.7±0.3 | 46.2±1.3 | 87.0±0.5 |
| 4 | PatchCore† | 88.8±2.6 | 92.6±1.6 | 94.5±1.5 | 94.3±0.5 | 55.0±1.9 | 84.3±1.6 |
| 4 | GraphCore | 92.9±/ | / | / | 97.4±/ | / | / |
| 4 | WinCLIP+ | 95.2±1.3 | 94.7±0.8 | 97.3±0.6 | 96.2±0.3 | 59.5±1.8 | 89.0±0.8 |
| 4 | ADPℓ | 96.2±0.8 | / | / | / | / | / |
| 4 | APRIL-GAN | 92.8±0.2 | 92.8±0.1 | 96.3±0.1 | 95.9±0.0 | 56.9±0.1 | 91.8±0.1 |
| 4 | AnomalyDINO-S (448) | 97.6±0.1 | 97.0±0.3 | 98.4±0.3 | 96.7±0.1 | 59.2±0.4 | 92.4±0.1 |
| 4 | AnomalyDINO-S (672) | 97.7±0.2 | 96.6±0.0 | 98.7±0.1 | 97.2±0.1 | 61.8±0.1 | 93.4±0.1 |
| 8 | GraphCore | 95.9±/ | / | / | 97.8±/ | / | / |
| 8 | WinCLIP+ | 94.6±0.1∗ | / | / | / | / | / |
| 8 | ADPℓ | 97.0±0.2 | / | / | / | / | / |
| 8 | APRIL-GAN | 93.1±0.2 | 93.1±0.2 | 96.4±0.2 | 96.2±0.1 | 57.7±0.2 | 92.4±0.2 |
| 8 | AnomalyDINO-S (448) | 98.0±0.1 | 97.4±0.1 | 99.0±0.2 | 97.0±0.1 | 59.6±0.2 | 92.7±0.0 |
| 8 | AnomalyDINO-S (672) | 98.2±0.2 | 97.4±0.2 | 99.1±0.1 | 97.4±0.1 | 62.3±0.1 | 93.8±0.1 |
| 16 | WinCLIP+ | 94.8±0.1∗ | / | / | / | / | / |
| 16 | ADPℓ | 97.0±0.3 | / | / | / | / | / |
| 16 | APRIL-GAN | 93.2±0.1 | 93.0±0.1 | 96.5±0.1 | 96.4±0.0 | 58.5±0.1 | 92.6±0.1 |
| 16 | AnomalyDINO-S (448) | 98.3±0.1 | 97.7±0.2 | 99.3±0.0 | 97.1±0.1 | 60.0±0.1 | 92.9±0.1 |
| 16 | AnomalyDINO-S (672) | 98.4±0.1 | 97.6±0.1 | 99.3±0.0 | 97.5±0.0 | 62.7±0.1 | 94.0±0.1 |

VisA（表 3）中，$k\in\{1,2\}$ 时 APRIL-GAN 最强；AnomalyDINO 稳定第二，4 样本结果相当，$k\in\{8,16\}$ 时创造新先进水平。这是因为它比 APRIL-GAN 更受益于丰富记忆库。我们推测，APRIL-GAN 更依赖元学习（测 VisA 时在 MVTec-AD 训练，反之亦然），而不是学习当前参考样本的正常特征。AnomalyDINO 也超过所有其他免训练方法。

VisA 分割（表 3）表明，AnomalyDINO 在全部考虑指标中稳定最强。448 分辨率已强，672 的优势在 VisA 更明显，归因于异常区域更小，较小有效图像块有益，且场景比 MVTec-AD 更复杂。

**表 3：VisA 异常检测。**图像级分类与像素级分割的定量结果，单位均为 %。原表以粗体表示各样本数最佳、下划线表示第二好方法、灰框表示最佳免训练结果（见表 1）；此处保留全部数值及标记，改为统一 Markdown 表格。GraphCore 的“±/”表示未报告标准差，“/”为原表缺项。

| 样本数 | 方法 | 分类 AUROC | 分类 F1-max | 分类 AP | 分割 AUROC | 分割 F1-max | 分割 PRO |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | SPADE† | 79.5±4.0 | 80.7±1.9 | 82.0±3.3 | 95.6±0.4 | 35.5±2.2 | 84.1±1.6 |
| 1 | PaDiM† | 62.8±5.4 | 75.3±1.2 | 68.3±4.0 | 89.9±0.8 | 17.4±1.7 | 64.3±2.4 |
| 1 | PatchCore† | 79.9±2.9 | 81.7±1.6 | 82.8±2.3 | 95.4±0.6 | 38.0±1.9 | 80.5±2.5 |
| 1 | WinCLIP+ | 83.8±4.0 | 83.1±1.7 | 85.1±4.0 | 96.4±0.4 | 41.3±2.3 | 85.1±2.1 |
| 1 | APRIL-GAN | 91.2±0.8 | 86.9±0.6 | 93.3±0.8 | 96.0±0.0 | 38.5±0.3 | 90.0±0.1 |
| 1 | AnomalyDINO-S (448) | 85.6±1.5 | 83.1±1.1 | 86.6±1.3 | 97.5±0.1 | 41.9±0.5 | 90.7±0.5 |
| 1 | AnomalyDINO-S (672) | 87.4±1.2 | 84.3±0.5 | 89.0±1.0 | 97.8±0.1 | 45.1±0.9 | 92.5±0.5 |
| 2 | SPADE† | 80.7±5.0 | 81.7±2.5 | 82.3±4.3 | 96.2±0.4 | 40.5±3.7 | 85.7±1.1 |
| 2 | PaDiM† | 67.4±5.1 | 75.7±1.8 | 71.6±3.8 | 92.0±0.7 | 21.1±2.4 | 70.1±2.6 |
| 2 | PatchCore† | 81.6±4.0 | 82.5±1.8 | 84.8±3.2 | 96.1±0.5 | 41.0±3.9 | 82.6±2.3 |
| 2 | WinCLIP+ | 84.6±2.4 | 83.0±1.4 | 85.8±2.7 | 96.8±0.3 | 43.5±3.3 | 86.2±1.4 |
| 2 | ADPℓ | 86.9±0.9 | / | / | / | / | / |
| 2 | APRIL-GAN | 92.2±0.3 | 87.7±0.3 | 94.2±0.3 | 96.2±0.0 | 39.3±0.2 | 90.1±0.1 |
| 2 | AnomalyDINO-S (448) | 88.3±1.8 | 84.8±1.2 | 89.2±1.3 | 97.8±0.1 | 44.2±0.3 | 91.7±0.5 |
| 2 | AnomalyDINO-S (672) | 89.7±1.3 | 86.3±1.2 | 90.7±0.8 | 98.0±0.1 | 47.6±0.5 | 93.4±0.6 |
| 4 | SPADE† | 81.7±3.4 | 82.1±2.1 | 83.4±2.7 | 96.6±0.3 | 43.6±3.6 | 87.3±0.8 |
| 4 | PaDiM† | 72.8±2.9 | 78.0±1.2 | 75.6±2.2 | 93.2±0.5 | 24.6±1.8 | 72.6±1.9 |
| 4 | PatchCore† | 85.3±2.1 | 84.3±1.3 | 87.5±2.1 | 96.8±0.3 | 43.9±3.1 | 84.9±1.4 |
| 4 | WinCLIP+ | 87.3±1.8 | 84.2±1.6 | 88.8±1.8 | 97.2±0.2 | 47.0±3.0 | 87.6±0.9 |
| 4 | ADPℓ | 88.4±0.4 | / | / | / | / | / |
| 4 | APRIL-GAN | 92.6±0.4 | 88.4±0.5 | 94.5±0.3 | 96.2±0.0 | 40.0±0.1 | 90.2±0.1 |
| 4 | AnomalyDINO-S (448) | 91.3±0.8 | 87.5±1.0 | 91.8±0.7 | 98.0±0.0 | 46.1±0.3 | 92.5±0.2 |
| 4 | AnomalyDINO-S (672) | 92.6±0.9 | 88.8±0.9 | 92.9±0.7 | 98.2±0.0 | 49.4±0.3 | 94.1±0.1 |
| 8 | WinCLIP+ | 85.0±0.0∗ | / | / | / | / | / |
| 8 | ADPℓ | 89.2±0.1 | / | / | / | / | / |
| 8 | APRIL-GAN | 93.0±0.2 | 88.8±0.2 | 94.9±0.3 | 96.3±0.0 | 40.2±0.1 | 90.2±0.0 |
| 8 | AnomalyDINO-S (448) | 92.6±0.1 | 88.6±0.2 | 92.9±0.2 | 98.2±0.0 | 47.6±0.5 | 93.3±0.2 |
| 8 | AnomalyDINO-S (672) | 93.8±0.3 | 90.0±0.1 | 94.3±0.4 | 98.4±0.0 | 51.1±0.4 | 94.8±0.2 |
| 16 | WinCLIP+ | 85.0±0.1∗ | / | / | / | / | / |
| 16 | ADPℓ | 90.1±0.5 | / | / | / | / | / |
| 16 | APRIL-GAN | 93.2±0.2 | 89.0±0.1 | 95.2±0.1 | 96.3±0.0 | 40.6±0.1 | 90.2±0.1 |
| 16 | AnomalyDINO-S (448) | 93.8±0.1 | 89.9±0.3 | 94.2±0.3 | 98.3±0.0 | 48.6±0.3 | 93.8±0.2 |
| 16 | AnomalyDINO-S (672) | 94.8±0.2 | 90.9±0.2 | 95.3±0.3 | 98.5±0.0 | 52.5±0.5 | 95.3±0.2 |

我们还适配至批量零样本（附录 D），改动直接且相对简单，尤其与 MuSc 比。没有使用 [24] 的“受约束图像级邻域重评分”或“多阶局部邻域聚合”等额外技术。因此表 4 的结果颇为令人满意。

尽管简单，AnomalyDINO 在所有设置均能与其他方法竞争，多数甚至领先，例如大幅缩小 MVTec-AD 少样本与全样本差距。免训练模型中，它基本全面胜出。这不仅说明方法本身的优点，也更一般地凸显强视觉特征相对于高度工程化架构的价值。更多定性结果见附录 A，语义异常检测等局限及失败案例见附录 B。

**表 4：其他设置的检测结果，全部为 AUROC（%）。**

| 设置 | 方法 | MVTec-AD | VisA |
| --- | --- | ---: | ---: |
| 零样本 | WinCLIP | 91.8 | 78.1 |
| 零样本 | AnomalyCLIP | 91.5 | 82.1 |
| 零样本 | APRIL-GAN | 86.1 | 78.0 |
| 批量零样本 | ACR | 85.8 | / |
| 批量零样本 | MuSc | 97.8 | 92.8 |
| 批量零样本 | AnomalyDINO-S (448) | 93.0 | 89.7 |
| 批量零样本 | AnomalyDINO-S (672) | 94.2 | 90.7 |

### 4.2 消融研究

额外实验评估具体设计选择，完整消融见附录 C，此处简述主要发现。图 3 从性能与运行时间两个关键方面支持消融。

**图 3：MVTec-AD 单样本检测 AUROC 与每样本推理时间。**方法名括号为输入分辨率。除注明外均在单块 NVIDIA A40 测量。ADP、WinCLIP+ 没有官方代码。横轴平均推理时间（ms，对数刻度），纵轴检测 AUROC（%）。图例为有预处理的 AnomalyDINO、无预处理的 AnomalyDINO、ImageNet 预训练其他主干、其他方法。AnomalyDINO-S 分辨率为 224、448、672、896；其他点为 ViT-B-16 (224)、ViT-L-16 (224)、PatchCore (224)、WinCLIP+ (240，非官方重实现或 Amazon EC2 G4dn 报告)、APRIL-GAN (518)、ADP (512，GeForce RTX 3090、2-shot 报告)。

**推理时间。**比较单样本时间（图 3，注意对数轴），AnomalyDINO 显著快于先进少样本竞争方法。例如 AnomalyDINO-S 在 448 分辨率处理一幅约 60 ms。只有 PatchCore 及使用 ImageNet 训练 ViT 主干的 AnomalyDINO 更快，但它们牺牲性能。增强增加记忆库，理论上增加近邻搜索时间，但实践影响可忽略⁵。更详细分析见附录 C 表 9。

⁵ 实现使用 GPU 加速近邻搜索 [20]，因此时间仅略增。库进一步增大时，建议使用核心集子采样 [34] 等缩减技术。

**预处理。**图 3 比较有无预处理的检测 AUROC。四种 AnomalyDINO 配置加入本文预处理后，AUROC 约提高 2%，而推理时间无显著变化。高分辨率时合适预处理影响似乎更大。另比较默认不知情与知情设置，前者稍好，见图 11、12。适当预处理对某些类别带来大幅改善。原则上旋转可能损害检测，但只在极少示例中观察到。完整讨论见附录 C.1。

**聚合统计量。**附录 C.2 将默认经验尾部风险值与其他合适聚合比较。测试图像块到库最高 1% 距离的均值，优于标准选择——上采样并平滑后图像块距离的最大值。

**架构尺寸/选择。**也对 ImageNet 预训练 ViT 评估该框架，见图 3、附录 C.4。这些主干略高于 PatchCore，但特征弱于 DINOv2，且不兼容本文掩码流程，说明 DINOv2 确实适合视觉 AD。DINOv2 等 ViT 架构易处理不同分辨率，448 在性能与时间间平衡最好，更高分辨率还可取得更高检测 AUROC。不同蒸馏尺寸 ViT-S、ViT-B、ViT-L 的完整结果见附录 C.3 图 16。差别不大，较大主干也在 MVTec-AD（所有 $k$）、VisA（$k\ge8$）取得先进结果。有趣的是，更大架构未必更强。最小模型在主要为单物体、较简单场景的 MVTec-AD 最好；更大架构在常有多个复杂物体的 VisA 更好。这表明，为具体任务取得“恰到好处”的抽象程度很重要，少样本尤其如此。

## 5 结论

本文提出纯视觉单/少样本异常检测，以 DINOv2 [30] 图像块表示之间相似性为基础。精心设计利用零样本分割、简单增强，以多样且相关的特征填充正常记忆库，同时最小化误报。工业要求快速部署、易调试纠错，以及迅速适应正常数据分布的协变量偏移。本流程的简单性与计算效率满足这些要求。AnomalyDINO 在 MVTec-AD 达到先进结果，在 VisA 与所有竞争方法抗衡并超过所有其他免训练方法。因此它兼具更好性能与更低推理时间，尤其相对复杂视觉—语言方法。简单与强性能使其成为工业实践的优秀候选，也为持续研究中的少样本乃至全样本异常检测提供有效基线。

**后续研究方向。**本文具体流程重视简单和高吞吐，但各部分易替换为更复杂方案。例如简单掩码可换成更专门、自适应的技术，对其他 DINOv2 方法也可能有用；简单上采样、平滑可换成 [15] 等更精细方法。值得研究能否进一步提高检测、定位，缩小少样本与全样本差距。我们也计划改善批量零样本性能。

**致谢。**作者感谢德国研究基金会（DFG）TRR 391“面向能源与交通转型的时空统计”资助。

## 参考文献

[1] Liron Bergman, Niv Cohen, and Yedid Hoshen. 《深度最近邻异常检测》。CoRR, abs/2002.10445, 2020. 3, 4

[2] Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger. 《MVTec 异常检测数据集：用于无监督异常检测的全面真实世界数据集》。International Journal of Computer Vision, 129(4):1038–1059, 2021. 1, 2, 5, 6

[3] Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon, Niladri Chatterji, Annie Chen, Kathleen Creel, Jared Quincy Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, Stefano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor Gale, Lauren Gillespie, Karan Goel, Noah Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte Khani, Omar Khattab, Pang Wei Koh, Mark Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, Jure Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir Mirchandani, Eric Mitchell, Zanele Munyikwa, Suraj Nair, Avanika Narayan, Deepak Narayanan, Ben Newman, Allen Nie, Juan Carlos Niebles, Hamed Nilforoshan, Julian Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, Joon Sung Park, Chris Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Rob Reich, Hongyu Ren, Frieda Rong, Yusuf Roohani, Camilo Ruiz, Jack Ryan, Christopher Ré, Dorsa Sadigh, Shiori Sagawa, Keshav Santhanam, Andy Shih, Krishnan Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Tramèr, Rose E. Wang, William Wang, Bohan Wu, Jiajun Wu, Yuhuai Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You, Matei Zaharia, Michael Zhang, Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, and Percy Liang. 《论基础模型的机遇与风险》。arXiv preprint arXiv:2108.07258, 2022. 2

[4] Yunkang Cao, Xiaohao Xu, Chen Sun, Yuqi Cheng, Zongwei Du, Liang Gao, and Weiming Shen. 《通过混合提示正则化无需训练地分割任意异常》。arXiv preprint arXiv:2305.10724, 2023. 1

[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. 《自监督视觉 Transformer 中涌现的性质》。In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 1, 2, 3

[6] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 《视觉表示对比学习的简单框架》。In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 1597–1607. PMLR, 13–18 Jul 2020. 2

[7] Xuhai Chen, Yue Han, and Jiangning Zhang. 《面向 CVPR 2023 VAND 研讨会挑战赛赛道 1 和 2 的零/少样本异常分类与分割方法：零样本 AD 第一名、少样本 AD 第四名》。arXiv preprint arXiv:2305.17382, 2023. 2, 3, 4, 5, 6

[8] Niv Cohen and Yedid Hoshen. 《使用深度金字塔对应进行子图像异常检测》。CoRR, abs/2005.02357, 2020. 3, 4, 6

[9] Tae Hyun Kim Daehyun Kim, Sungyong Baik. 《SanFlow：用于异常检测与定位的语义感知归一化流》。In In Advances in Neural Information Processing Systems (NeurIPS), 2023. 1, 3

[10] Thomas Defard, Aleksandr Setkov, Angelique Loesch, and Romaric Audigier. 《PaDiM：用于异常检测与定位的图像块分布建模框架》。In Alberto Del Bimbo, Rita Cucchiara, Stan Sclaroff, Giovanni Maria Farinella, Tao Mei, Marco Bertini, Hugo Jair Escalante, and Roberto Vezzani, editors, Pattern Recognition. ICPR International Workshops and Challenges - Virtual Event, January 10-15, 2021, Proceedings, Part IV , volume 12664 of Lecture Notes in Computer Science , pages 475–489. Springer, 2020. 3, 6

[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 《一幅图像值 16×16 个词：大规模图像识别的 Transformer》。CoRR, abs/2010.11929, 2020. 3

[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 《一幅图像值 16×16 个词：大规模图像识别的 Transformer》。In International Conference on Learning Representations, 2021. 3

[13] Eleazar Eskin, Andrew Arnold, Michael Prerau, Leonid Portnoy, and Sal Stolfo. 《无监督异常检测的几何框架：检测无标签数据中的入侵》。Applications of data mining in computer security, pages 77–101, 2002. 3, 4

[14] Tharindu Fernando, Harshala Gammulle, Simon Denman, Sridha Sridharan, and Clinton Fookes. 《医疗异常检测的深度学习综述》。ACM Comput. Surv., 54(7), jul 2021. 1

[15] Stephanie Fu, Mark Hamilton, Laura E. Brandt, Axel Feldmann, Zhoutong Zhang, and William T. Freeman. 《FeatUp：适用于任意分辨率特征的模型无关框架》。In The Twelfth International Conference on Learning Representations, 2024. 8

[16] Douglas M Hawkins. 《离群点识别》。volume 11. Springer, 1980. 1

[17] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. 《掩码自编码器是可扩展的视觉学习器》。Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000–16009, 2022. 2

[18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 《用于图像识别的深度残差学习》。CoRR, abs/1512.03385, 2015. 3

[19] Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, and Onkar Dabeer. 《WinCLIP：零/少样本异常分类与分割》。In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19606–19616, 2023. 1, 3, 5, 6

[20] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 《使用 GPU 进行十亿级相似性搜索》。IEEE Transactions on Big Data, 7(3):535–547, 2019. 8

[21] Sangkyung Kwak, Jongheon Jeong, Hankook Lee, Woohyuck Kim, Dongho Seo, Woojin Yun, Wonjin Lee, and Jinwoo Shin. 《通过个性化进行少样本异常检测》。IEEE Access, 2024. 1, 2, 3, 6, 20

[22] Aodong Li, Chen Qiu, Marius Kloft, Padhraic Smyth, Maja Rudolph, and Stephan Mandt. 《通过批归一化进行零样本异常检测》。Advances in Neural Information Processing Systems, 36, 2024. 1, 3, 4, 6, 21

[23] Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, and Jianfeng Gao. 《多模态基础模型：从专门模型到通用助手》。arXiv preprint arXiv:2309.10020, 2023. 2

[24] Xurui Li, Ziming Huang, Feng Xue, and Yu Zhou. 《MuSc：通过无标签图像互相评分进行零样本工业异常分类与分割》。In International Conference on Learning Representations, 2024. 3, 4, 5, 6, 7, 21

[25] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. 《Grounding DINO：将 DINO 与定位预训练结合用于开放集目标检测》。arXiv preprint arXiv:2303.05499, 2023. 2, 3

[26] Victor Livernoche, Vineet Jain, Yashar Hezaveh, and Siamak Ravanbakhsh. 《论用于异常检测的扩散建模》。CoRR, abs/2305.18593, 2023. 1, 3

[27] Alexander J McNeil. 《面向风险管理者的极值理论》。Departement Mathematik ETH Zentrum , 12(5):217–37, 1999. 4

[28] Arian Mousakhan, Thomas Brox, and Jawad Tayyub. 《使用条件去噪扩散模型进行异常检测》。CoRR, abs/2305.15956, 2023. 1, 3

[29] OpenAI. 《GPT-4 技术报告》。arXiv preprint arXiv:2303.08774, 2024. 2

[30] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 《DINOv2：无需监督地学习鲁棒视觉特征》。Transactions on Machine Learning Research, 2024. 1, 2, 3, 5, 8, 18

[31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 《通过自然语言监督学习可迁移视觉模型》。CoRR, abs/2103.00020, 2021. 1, 2

[32] Mehdi Rafiei, Toby P Breckon, and Alexandros Iosifidis. 《论异常检测的像素级性能评估》。arXiv preprint arXiv:2310.16435, 2023. 6

[33] Tal Reiss, Niv Cohen, Eliahu Horwitz, Ron Abutbul, and Yedid Hoshen. 《异常检测需要更好的表示》。In European Conference on Computer Vision, pages 56–68. Springer, 2022. 1

[34] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler. 《迈向工业异常检测的完全召回》。In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14318–14328, 2022. 2, 3, 4, 5, 6, 8, 19

[35] Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn, and Bastian Wandt. 《用于图像缺陷检测的全卷积跨尺度流》。In IEEE/CVF Winter Conference on Applications of Computer Vision, WACV 2022, Waikoloa, HI, USA, January 3-8, 2022, pages 1829–1838. IEEE, 2022. 1

[36] Lukas Ruff, Jacob R Kauffmann, Robert A Vandermeulen, Grégoire Montavon, Wojciech Samek, Marius Kloft, Thomas G Dietterich, and Klaus-Robert Müller. 《深度与浅层异常检测的统一综述》。Proceedings of the IEEE, 109(5):756–795, 2021. 3

[37] Lukas Ruff, Robert Vandermeulen, Nico Goernitz, Lucas Deecke, Shoaib Ahmed Siddiqui, Alexander Binder, Emmanuel Müller, and Marius Kloft. 《深度单类分类》。In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 4393–4402. PMLR, 10–15 Jul 2018. 3

[38] Lukas Ruff, Robert A. Vandermeulen, Nico Görnitz, Alexander Binder, Emmanuel Müller, Klaus-Robert Müller, and Marius Kloft. 《深度半监督异常检测》。In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. 1, 3

[39] João Santos, Triet Tran, and Oliver Rippel. 《优化 PatchCore 以进行少样本/多样本异常检测》。arXiv preprint arXiv:2307.10792, 2023. 3

[40] Md Amran Siddiqui, Jack W Stokes, Christian Seifert, Evan Argyle, Robert McCann, Joshua Neil, and Justin Carroll. 《结合解释和专家反馈使用异常检测来发现网络攻击》。In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2872–2876. IEEE, 2019. 1

[41] Yu Tian, Fengbei Liu, Guansong Pang, Yuanhong Chen, Yuyuan Liu, Johan W. Verjans, Rajvinder Singh, and Gustavo Carneiro. 《用于医学图像无监督异常检测与分割的自监督伪多类预训练》。Medical Image Anal. , 90:102930, 2023. 1

[42] Julia Wolleb, Florentin Bieder, Robin Sandkühler, and Philippe C Cattin. 《用于医学异常检测的扩散模型》。In International Conference on Medical image computing and computer-assisted intervention, pages 35–45. Springer, 2022. 1, 3

[43] Guoyang Xie, Jinbao Wang, Jiaqi Liu, Feng Zheng, and Yaochu Jin. 《突破工业视觉少样本异常检测的极限：GraphCore》。arXiv preprint arXiv:2301.12082, 2023. 3, 4, 5, 6

[44] Xiaohao Xu, Yunkang Cao, Yongqi Chen, Weiming Shen, and Xiaonan Huang. 《定制视觉—语言基础模型以进行多模态异常检测与推理》。arXiv preprint arXiv:2403.11083, 2024. 3

[45] Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu. 《广义分布外检测综述》。arXiv preprint arXiv:2110.11334, 2021. 3

[46] Jihun Yi and Sungroh Yoon. 《Patch SVDD：用于异常检测与分割的图像块级 SVDD》。In Proceedings of the Asian conference on computer vision, 2020. 2, 3, 4

[47] Sergey Zagoruyko and Nikos Komodakis. 《宽残差网络》。In Richard C. Wilson, Edwin R. Hancock, and William A. P. Smith, editors, Proceedings of the British Machine Vision Conference 2016, BMVC 2016, York, UK, September 19-22, 2016. BMVA Press, 2016. 3

[48] Hui Zhang, Zheng Wang, Zuxuan Wu, and Yu-Gang Jiang. 《DiffusionAD：用于异常检测的去噪扩散》。arXiv preprint arXiv:2303.08730, 2023. 3

[49] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. 《使用在线分词器的图像 BERT 预训练》。In International Conference on Learning Representations, 2022. 3

[50] Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, and Jiming Chen. 《AnomalyCLIP：用于零样本异常检测的对象无关提示学习》。In The Twelfth International Conference on Learning Representations, 2024. 1, 3, 4

[51] Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. 《用于异常检测与分割的找不同自监督预训练》。In European Conference on Computer Vision , pages 392–408. Springer, 2022. 1, 2, 5
