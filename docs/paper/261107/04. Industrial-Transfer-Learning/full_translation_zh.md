# 重新思考工业检测中的迁移学习：跨RGB与X射线任务比较DINOv3和ImageNet预训练

**原文：**[Rethinking Transfer Learning for Industrial Inspection: DINOv3 vs. ImageNet Pretraining Across RGB and X-ray Tasks](<Rethinking Transfer Learning for Industrial Inspection- DINOv3 vs. ImageNet Pretraining Across RGB and X-ray Tasks.pdf>)。对应本目录9页PDF，arXiv:2605.23472v1，2026年5月22日。翻译日期：2026-11-07。

**作者：**Mehdi Gharbage¹˒²、Céline Teulière²、Pierre Bouges¹、Thierry Chateau²。

**单位：**¹米其林轮胎制造商；²法国克莱蒙费朗，克莱蒙奥弗涅大学、法国国家科学研究中心（CNRS）、帕斯卡研究所。

**翻译说明：**逐段翻译本地PDF的摘要、正文第1—6节、致谢、图1—4说明和关键图内文字、表1—2及全部38条参考文献。本版本未附独立附录。合并排版造成的断行和跨页续段，去除运行页眉、页脚和孤立页码。保留模型、数据集、引文编号和实验数字；参考文献作者、出处等专名保留，题名提供中文。原图不重绘，图形请结合PDF查看。正文与表格不一致的地方保留原文并明确作译注，不将原文数值改写为本项目的实验结论。

## 摘要

在网络规模数据上预训练的视觉基础模型，近期在许多下游任务中展示了较强的迁移能力，但它们对工业视觉检测的有效性仍不明确。工业数据与网络数据存在显著差异，而且往往要求细粒度的密集预测，这引出了一个问题：现代自监督预训练能否优于以有监督ImageNet初始化为基础的传统迁移学习范式？本文比较使用ImageNet有监督分类预训练或DINOv3蒸馏预训练的ConvNeXt主干，并将它们与传统ResNet-50基线联系起来考察。我们在四个下游数据集上评价语义分割、实例分割和目标检测，涵盖RGB表面缺陷检测与X射线缺陷检测。我们还研究主干冻结和完全微调两种适配方式。结果表明，DINOv3在冻结迁移中没有明显优势，但在RGB任务上完全微调后能够提供更强的初始化，使收敛更快、最终性能更好。然而，在X射线模态偏移下，无论冻结还是微调，ImageNet有监督预训练仍更有效。总体而言，我们的发现表明，现代视觉基础模型有望用于有监督RGB工业检测，但其可迁移性很大程度上取决于下游适配和目标模态。

## 1 引言

视觉检测是工业质量控制的重要组成部分，其目标通常是在具有挑战性的成像条件下检测、定位和分割微小缺陷。与自然图像基准[30]不同，工业检测数据通常具有较强的领域特征，包括标注数据有限、前景与背景严重不平衡、局部模式细粒度，以及不同产品和采集条件之间较大的外观变化。当从传统RGB图像转向X射线等模态时，这些困难会更加突出，因为这些模态的视觉统计特性与网络规模自然图像显著不同。因此，为工业检测设计可迁移的视觉表征仍然是一项核心挑战。

**图1：冻结主干下不同预训练方式的比较。**比较在卷积主干ConvNeXt上使用DINOv3蒸馏预训练或ImageNet有监督预训练，考察冻结主干的线性探测设置下，下游图像分类、语义分割和目标检测的表现。柱形表示两种预训练方法的相对性能差距：正值表示DINOv3占优，负值表示ImageNet有监督预训练占优。网络规模基准中的图像分类和ADE20K语义分割结果来自既有工作[32]，Pascal VOC、Severstal、RarePlanes和GDXray的结果来自本文探测设置。图例“自然图像”“工业图像”分别对应原图的两种颜色。纵轴原式为$100\log(P_{\mathrm{DINOv3预训练}}/P_{\mathrm{ImageNet预训练}})$，坐标另标有“×10”；图中柱上标注值如下，按原图保留，不将其改算成普通百分比差值：

| 任务 | 数据集 | 图中标注值 |
| --- | --- | ---: |
| 分类 | IN-ReAL | +2.4 |
| 语义分割 | ADE20K | +23.6 |
| 语义分割 | Severstal | +0.3 |
| 语义分割 | Rubber | -0.6 |
| 目标检测 | VOC | -4.6 |
| 目标检测 | RarePlanes | -1.7 |
| 目标检测 | GDXray | -44.4 |

多年来，计算机视觉中占主导地位的迁移学习范式一直依赖有监督ImageNet预训练[21]，后者已成为下游识别和定位任务的标准初始化策略。在工业检测中，这种方法仍被广泛使用，因为在任务专用标注稀缺时，它提供了一种实用方案[2,3,16]。

然而，视觉基础模型近期取得的进展[22,27—29,32]，使研究重点转向在网络数据上进行大规模自监督预训练，期望学到更加通用、可复用的视觉表征。特别是DINOv3[32]等现代密集特征导向模型表明，大容量架构通过规模化训练，可能不仅能为全局识别提供更强特征，也能为分割和检测等结构化预测任务提供更强特征。

这种潜力能否延伸到工业检测，仍不清楚。一方面，工业任务要求空间上精确的表征，而且往往涉及与自然图像显著不同的视觉模式。另一方面，如果基础模型特征在这种分布偏移下仍能很好地迁移，它们就可能重新定义工业视觉系统的标准初始化策略。对于有监督工业场景，这个问题尤为重要，因为目标并非仅仅相对于正常样本检测异常，而是为语义分割、实例分割和目标检测学习兼顾类别与定位的预测器。

本文通过对传统和现代迁移学习范式进行受控比较来研究这一问题。我们将ImageNet有监督预训练作为传统基线，并在现代卷积主干ConvNeXt[25]上，将其与DINOv3蒸馏进行比较。具体而言，我们评价分别使用ImageNet有监督分类或DINOv3蒸馏预训练的ConvNeXt-T主干，并将这些结果与较强的ResNet-50基线联系起来分析。研究涵盖四个下游数据集上的三类有监督工业视觉任务，即语义分割、实例分割和目标检测，同时包括RGB表面检测和X射线缺陷检测。我们进一步分析从冻结主干到完全微调等不同适配方式下的迁移表现，以确定DINOv3表征能否直接使用，还是其优势主要通过下游适配显现。

结果表明，答案需要作细致区分。在RGB工业数据集上，冻结主干时DINOv3没有明显优势，但完全微调后会成为最强的初始化，使语义分割和RGB实例级定位收敛更快、最终性能更好。相比之下，这种优势没有延伸到X射线图像：无论冻结还是完全微调，ImageNet有监督预训练都更优。这些发现表明，现代视觉基础模型对工业检测的价值，与其说在于直接而普适的迁移，不如说在于当目标数据与自然图像统计特征足够接近时，能够为微调提供较强的先验。

本文的贡献如下：

- 对有监督工业检测中的迁移学习开展受控研究，覆盖语义分割、实例分割和目标检测。
- 在卷积主干上，于冻结和微调两种适配方式下，比较传统ImageNet有监督预训练与DINOv3蒸馏。
- 表明大规模自监督预训练的收益高度依赖下游任务与目标模态：在RGB任务上有明显收益，在X射线模态偏移下的迁移则有限。

## 2 相关工作

### 2.1 面向密集预测的视觉基础模型

过去几年，视觉基础模型已成为计算机视觉的重要研究方向。其基本前提是：大容量架构在网络规模图像集合上训练后，可以学习通用视觉表征，并有效迁移到多种下游任务。这一趋势自然延伸了长期存在的迁移学习范式，在该范式中，ImageNet有监督预训练[30]已成为许多视觉模型的标准初始化策略。尤其在目标标注有限时，ImageNet预训练被广泛用于下游识别和定位任务，通常能够改善优化和数据效率[21]，因为它使模型获得边缘、纹理和物体部件等通用的低层及中层视觉线索。

然而，有监督分类数据的规模与多样性仍然受到内在限制，这推动了自监督学习的发展，以便利用更大规模的无标签图像集合。早期自监督方法，包括对比方法[7,9]和联合嵌入方法[4,8,10,18]，引入了无需人工监督即可学习表征的前置任务目标。但这些目标主要针对全局图像级表征优化，而非密集视觉特征。较新的研究[15,33]表明，全局前置任务与密集预测并不总是一致，较强的全局性能也不一定转化为较强的密集表征[32]。这一认识促成了更明确地促进密集视觉学习的训练策略[22,27,32,33,38]。其中，掩码图像建模方法[22,38]将密集表征学习作为核心设计目标，而对比方法和联合嵌入方法[27,32,33]也逐步得到适配，以更好地支持密集视觉特征。

在这些模型中，DINOv3[32]与本文尤其相关，因为它明确以规模化保留和改善密集表征为设计目标。DINOv3将DINO框架适配到密集预测，并发布从大规模教师模型蒸馏得到的视觉Transformer和ConvNeXt主干，因此特别适合用于研究强领域偏移下向有监督工业分割与检测的迁移。

### 2.2 面向工业检测的有监督迁移学习

有监督迁移学习长期以来一直是工业检测中实用且广泛采用的范式，因为获取大规模缺陷标注通常成本高昂，而且依赖具体应用。在视觉基础模型兴起之前，标准做法是通过ImageNet有监督预训练[30]初始化卷积主干，再针对缺陷分类、目标检测和分割等下游任务进行微调。这种策略在数据较少的工业场景中尤其有吸引力，因为相比从头训练，预训练模型可以改善优化并减少过拟合[21]。

VISION等基准建设工作通过提供具有像素级和实例级标注的检测与分割工业数据集[5]，反映了这种有监督设置持续的重要性；近期综述也明确区分了此类闭集监督形式与开放集及异常检测范式[12]。相同的迁移学习方案也被用于X射线图像：尽管存在显著模态差距，在自然图像上预训练的卷积网络仍被适配到物体识别和检测[3,19]。更近的研究进一步表明，低数据量条件下的有监督工业分割仍然具有挑战性，因而需要比传统ImageNet初始化CNN更强的可迁移表征[24]。

### 2.3 将基础模型适配到工业检测

近期工作开始探索如何将基础模型适配到工业视觉检测，但其中相当多的文献关注异常检测，而非有监督缺陷识别与定位。尤其是，一些方法以通用视觉或视觉语言基础模型为基础，采用提示工程、提示调优或参数高效微调等任务专用适配策略。代表性例子包括：基于CLIP的异常检测方法，为工业零样本迁移学习静态和动态提示[6]；以及基于SAM的方法，引入专用调优模块和适配器，以改善工业领域偏移下的异常分割[36]。更广泛地看，近期工业基础模型综述将提示学习、微调和领域适应视为将通用基础模型带入制造应用的核心机制[37]。

这些方法表明，基础模型经过处理后能够在工业场景中发挥作用，但通常依赖额外的适配方法、辅助调优或领域专用改进，而非直接评价预训练视觉表征本身是否已经足以支持迁移。

一个值得注意的例外是AnomalyDINO[13]。它研究能否不经过额外微调或元学习，直接利用冻结的DINOv2视觉特征完成工业异常检测。其较强结果表明，基础模型特征已经可以为图像块级异常评分提供有力基础。然而，这一研究路线仍然属于异常检测设置，其目标是识别相对于正常状态的偏离，而非利用语义标签完成有监督缺陷检测或分割。因此，这些方法不能完全满足有监督工业检测的要求。在后者中，模型必须针对语义分割、实例分割和目标检测等任务产生具有类别感知能力、空间上精确的密集预测。

相关少样本缺陷分割研究进一步表明，视觉基础模型在工业场景中具有显著潜力，但也强调了超越常规异常检测基准，认真进行任务适配和评价的必要性[24]。这些观察促使我们关注一个互补问题：在强领域偏移和模态偏移下，近期面向密集预测的视觉基础模型能否有效迁移到有监督工业检测任务，以及实际上需要多大程度的适配。

## 3 方法

### 3.1 研究目标

本文旨在评价：相较于实践中常用的传统ImageNet有监督预训练，在网络数据上进行大规模自监督预训练，能否为有监督工业视觉检测提供更强的迁移学习范式。具体而言，我们研究经DINOv3蒸馏的卷积网络主干，能否改善向要求空间精确预测的工业任务的迁移，即语义分割、实例分割和目标检测。

除了总体性能，本文还考察：从大规模自然图像数据中学习的表征，在工业图像所具有的强分布偏移下是否仍然有效，包括X射线等更具挑战性的图像模态。通过这一评价，我们希望确定DINOv3预训练能否成为有监督工业缺陷检测中新的实用迁移学习标准。

### 3.2 比较研究

为回答这一问题，我们采用受控比较研究，而不是提出新的任务专用方法。传统基线使用通过ImageNet有监督预训练初始化的ResNet-50主干，这仍然是工业视觉的标准迁移学习方案。随后，在两种不同预训练设置下评价更强的ConvNeXt卷积主干[25]：ImageNet有监督分类预训练和DINOv3蒸馏。这样的设计使我们能够减少架构差异带来的偏差，更直接地关注预训练范式本身的影响。

评价针对三种有监督工业检测任务进行：语义分割、实例分割和目标检测。对于每种设置，我们研究冻结主干和完全适配主干两种方式。在不同预训练主干上应用相同的适配协议，可以与传统ResNet基线进行公平比较，并评价各类表征迁移到工业任务的容易程度。

### 3.3 卷积网络主干

我们主要评价卷积网络，原因有二。首先，卷积网络在工业检测中仍然是较强且广泛采用的基线，而该领域的密集预测任务通常要求精确定位微小缺陷、重复纹理和精细结构模式。其局部性、平移等变性等归纳特性[25,34]很适合此类场景，其特征层级也容易与标准分割和检测解码器集成。

其次，将主要比较限制在卷积主干内，能够更好地将预训练的影响与架构的影响分离。尤其是，比较采用ImageNet有监督目标预训练或采用DINOv3蒸馏的ConvNeXt主干，为评价大规模自监督预训练能否给工业检测带来稳定收益提供了更受控的设置。虽然视觉Transformer[14]，特别是在扩大规模后，也可能具有较强的迁移能力，但我们将其作用留待未来分析，以便当前研究集中于预训练的影响。

## 4 实验设置

### 4.1 架构细节

为评价预训练卷积主干在不同有监督工业视觉任务间的可迁移性，我们将每种主干放入标准密集预测框架中实例化。语义分割采用Mask2Former[11]解码器架构，实例分割采用Mask R-CNN[20]，目标检测采用Faster R-CNN[17]。

本文关注预训练的影响，而非重新设计架构。因此，除传统ResNet-50基线外，各实验中的下游架构保持不变。遵循文献[21]，我们将ResNet-50中的批归一化替换为组归一化，以确保在检测和分割设置中稳定优化。除此之外，不对主干或解码器架构作额外修改。

> 译注：此处Faster R-CNN后面的引文编号[17]按原文保留；本PDF参考文献[17]的题名实际为《Fast R-CNN》。

### 4.2 数据集

我们在四个数据集上开展研究，它们覆盖互补的工业检测设置和任务形式。通过这些数据集，可以评价语义分割、实例分割和目标检测间的迁移，同时考察不同外观统计特性和图像模态。表1总结了数据集的类别数、划分和预处理细节。

**表1：本文使用的工业数据集概况。**†对于GDXray，按序列划分训练集与验证集：验证集包含序列3、32、45、54、61、87；除序列55外，其余序列用于训练。

| 项目 | Severstal | Rubber Rings | GDXray | RarePlanes |
| --- | ---: | ---: | ---: | ---: |
| 任务 | 语义分割 | 语义分割 | 目标检测 | 实例分割 |
| 模态 | RGB | RGB | X射线 | RGB |
| 类别数 | 4 | 9 | 1 | 3 |
| 训练样本数 | 5332 | 158 | 706 | 4474 |
| 验证样本数 | 1334 | 39 | 356 | 1113 |
| 划分方式 | 随机 | 分层随机 | 自定义† | 官方 |
| 测试时图像短边 | 384 | 800 | 800 | 800 |

#### 4.2.1 语义分割数据集

**Severstal。**使用Severstal数据集[1]评价钢材表面缺陷的语义分割。该数据集具有工业表面检测的代表性：缺陷表现为局部纹理和结构变化，背景则大体均匀。

**Rubber Rings。**进一步在真实橡胶圈缺陷数据集[24]上评价语义分割。相比Severstal，该数据集更侧重工业质量控制，其目标是识别制造部件上的缺陷。此外，该数据集的训练样本更少，因而更具挑战性，也更能代表真实工业场景。

#### 4.2.2 检测与定位数据集

**RarePlanes。**对于实例分割和定位，我们使用RarePlanes[31]中的真实航拍图像。尽管该数据集严格来说不是工业检测基准，但它与具有挑战性的检测场景具有若干共同特征，包括小目标、多尺度结构以及大范围背景上的显著外观变化。因此，我们将其作为代理场景，用于评价预训练主干如何迁移到不同于自然图像数据集的细粒度实例级定位任务。

**GDXray Castings。**为研究更强模态偏移下的迁移，我们使用GDXray[26]的铸件子集，其中包含带缺陷标注的X射线图像。该数据集对本文尤为重要，因为它能够用来评价：大规模自然图像预训练中学到的表征，迁移到工业X射线检测时是否仍然有效。因此，它是评价基础模型特征超越传统RGB图像之后的鲁棒性的重要基准。

### 4.3 训练协议

我们的目标不是通过广泛的超参数搜索分别优化每个数据集，而是在受控且一致的协议下评价不同预训练策略的可迁移性。因此，训练过程尽可能接近所用下游架构的默认训练方案，只在必要时作如下所述的修改。这样的设计使性能差异主要能够归因于预训练表征，而非针对特定基准的优化。

**训练方案。**在语义分割任务中，Severstal与Rubber Rings都存在较强的前景/背景不平衡，缺陷区域只占图像面积的一小部分。为了减少只含背景的裁剪块数量，训练时采用缺陷感知裁剪策略：以$p=0.7$的概率，约束裁剪块至少包含某个已标注缺陷区域的一部分。对于GDXray Castings目标检测，按照文献[16]调整锚框配置，使其更匹配数据集中的小缺陷实例。除这些任务专用调整外，所有其他设置均遵循Detectron2[35]的默认配置。

**调度。**同一任务类别内部保持相同的学习率调度策略。对于Mask2Former语义分割，所有实验均采用其原始实现的默认调度。对于目标检测和实例分割，采用文献[21]中的策略：先进行线性预热，再进行多阶段学习率衰减。RarePlanes使用$3\times$调度，GDXray Castings使用更长的$6\times$调度。

**超参数。**为保持公平比较，我们不针对各个数据集单独调节超参数，而是在各架构类别内部，使实验超参数保持不变。语义分割保留Mask2Former的默认配置。对于检测和实例分割，ResNet-50主干采用文献[21]的训练方案，ConvNeXt主干采用标准ConvNeXt超参数设置[25]。这种受控协议旨在确保观察到的差异主要反映预训练特征的质量。

所有实验均使用Detectron2框架[35]实现，该框架为语义分割、实例分割和目标检测提供统一的训练与评价环境。

**图2：Severstal验证集上的mIoU学习曲线。**使用Mask2Former，主干为通过ImageNet有监督分类或DINOv3蒸馏预训练的ConvNeXt-T。横轴为迭代次数，单位为$10^3$；纵轴为mIoU。图中“DINOv3”“ImageNet有监督”表示两种初始化；“冻结”“微调”表示主干的两种适配方式。

## 5 结果与分析

表2给出了所有比较主干与预训练策略的最终验证性能，图2—4展示了冻结和完全微调设置下相应的学习曲线。可以得到两项主要观察。首先，预训练的影响高度依赖目标领域：在RGB数据集上，DINOv3蒸馏在冻结设置中没有明显优势，只有完全微调后才稳定带来收益；在X射线图像上，无论哪种设置，它都不及ImageNet有监督初始化。其次，主干类别本身也很重要：原文指出，不论采用何种预训练策略，完全微调后ConvNeXt都一致优于传统ResNet-50基线，说明更强的卷积架构本身已经能为工业视觉检测提供扎实的迁移收益。

更具体地说，完全适配之后，经DINOv3蒸馏的ConvNeXt在所有RGB任务上取得最佳结果；在X射线模态偏移下，ImageNet预训练的ConvNeXt仍然是最强选择。这些结果表明，网络数据上的大规模自监督预训练可以为RGB工业检测提供更强的初始化，但当目标模态与自然图像统计特性相差过远时，其可迁移性会显著下降。

**表2：四个下游评价数据集上，主干预训练策略与微调方式的比较。**Severstal和Rubber Rings语义分割报告mIoU；RarePlanes报告掩码mAP@[0.5:0.95]；GDXray报告边界框mAP@50。所有结果均来自各数据集使用的验证划分，并在训练最后一次迭代时报告。无论哪种设置，下游任务头都在目标数据集上优化；主干则根据适配设置冻结或完全微调。所有实验固定随机种子为42。前两个数据集的解码器为Mask2Former，RarePlanes为Mask R-CNN，GDXray为Faster R-CNN。

| 主干 | 预训练方法 | 预训练数据 | 微调方式 | Severstal | Rubber Rings | RarePlanes | GDXray |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| ResNet-50 | 有监督分类 | ImageNet-1k | 完全微调 | 63.28 | 73.87 | 78.39 | 24.42 |
| ConvNeXt-T | 有监督分类 | ImageNet-1k | 冻结主干 | 62.04 | 73.25 | 72.89 | 21.32 |
| ConvNeXt-T | 有监督分类 | ImageNet-1k | 完全微调 | 62.97 | 73.26 | 82.88 | **29.74** |
| ConvNeXt-T | DINOv3蒸馏 | LVD-1689M | 冻结主干 | 62.40 | 72.32 | 70.36 | 7.88 |
| ConvNeXt-T | DINOv3蒸馏 | LVD-1689M | 完全微调 | **64.01** | **75.60** | **84.50** | 27.84 |

> 译注：本段“完全微调后ConvNeXt一致优于ResNet-50”的原文概括，并不覆盖表2中的所有单项比较，例如ImageNet预训练ConvNeXt-T的Severstal与Rubber Rings结果均低于对应ResNet-50。此处保留作者原来的概括，具体数值以表2为准。

**图3：RarePlanes验证集上的边界框mAP学习曲线。**使用Mask R-CNN，主干为通过ImageNet有监督分类或DINOv3蒸馏预训练的ConvNeXt-T。横轴为迭代次数，单位为$10^3$；纵轴原文标为“边界框AP@IoU[0.50:0.95]”。图例区分DINOv3、ImageNet有监督，以及冻结、微调。

### 5.1 语义分割

Severstal与Rubber Rings的语义分割结果，清楚地区分了探测与完全微调两种设置。主干冻结时，两种预训练策略的差距有限。在Severstal上，ImageNet有监督分类预训练和DINOv3蒸馏预训练的ConvNeXt-T表现非常接近，mIoU分别为62.04与62.40，说明在该设置中，将主干仅用作固定特征提取器时，DINOv3并未明显优于传统预训练。Rubber Rings呈现类似趋势，在冻结设置下，ImageNet预训练甚至略强于DINOv3，mIoU分别为73.25与72.32。总体而言，这些结果表明，对于工业RGB语义分割，DINOv3预训练的优势不会立即体现在线性探测式协议下；这不同于自然图像基准上经常报告的大幅提升。

主干完全微调后则表现不同。在这种设置中，经DINOv3蒸馏的ConvNeXt-T在两个数据集上都取得最佳性能。原文称，其同时优于传统ResNet-50基线（+10.59 mIoU）和ImageNet预训练的ConvNeXt-T（+10.29 mIoU）。这一结果尤其值得注意，因为ImageNet预训练的ConvNeXt-T在两个数据集上都未能比冻结版本改善，而DINOv3预训练在适配后产生了明确增益。这表明，尽管DINOv3特征作为工业表面缺陷的冻结描述子时没有较强优势，但它为下游微调提供了更有利的初始化。

> 译注：上段的“+10.59”“+10.29”，以及“ImageNet预训练的ConvNeXt-T在两个数据集上均未优于冻结版本”，都按原文保留，但与表2的相应数值不一致。表2中ImageNet完全微调相对冻结版本为62.97对62.04、73.26对73.25；不能将该段未经核对的增益用于本课题的涨分预期。

学习曲线进一步支持上述解释。如图2所示，冻结主干时，两种预训练策略的优化行为相近，与最终接近的mIoU值一致。相比之下，完全微调时DINOv3初始化收敛更快、最终性能更高，表明预训练期间学到的密集视觉特征促进了下游适配。

这些结果表明，对于工业RGB语义分割，DINOv3主要是更优的微调初始化，而不是更强的冻结特征提取器。

### 5.2 RGB实例分割与定位

RarePlanes的结果与语义分割表现出类似行为。主干冻结时，ImageNet有监督预训练优于DINOv3蒸馏；但完全微调后，这一趋势反转，经DINOv3蒸馏的ConvNeXt-T取得最佳结果，即84.50掩码mAP，高于ImageNet预训练的ConvNeXt-T（82.88）和ResNet-50基线（78.39）。学习曲线证实了这一行为：冻结设置下ImageNet预训练仍略强；主干完全适配后，DINOv3的收敛更好，最终性能也更高。

一种可能的解释是，DINO自蒸馏主要学习全局语义不变性，而Mask R-CNN更加依赖实例级定位和边界框回归。已有工作表明，当预训练与多目标数据及回归感知目标相一致时，检测会获得更大收益[23]；DINO式联合嵌入方法在非目标中心数据集上，也不如掩码图像建模方法鲁棒[15]。这或许能解释，为何DINOv3在冻结检测迁移中的增益有限，却仍能为微调提供更强初始化。

不过，我们应记住，学生主干仍然是卷积架构，其迁移行为仍受到局部性、权重共享、下采样和多通道等CNN特有归纳偏置的影响[34]。这意味着，DINOv3在冻结迁移中增益有限，也可能部分反映了从视觉Transformer蒸馏的表征，与ConvNeXt学生所具有、对定位导向迁移仍很重要的卷积友好先验之间的不匹配；即使相同初始化在完全微调后会非常有效。

> 译注：本节和表2表注使用“掩码mAP”，而图3标题及纵轴使用“边界框mAP/AP”。这是本地PDF的口径差异，译文保留，不把两种指标自行视为相同。

### 5.3 X射线缺陷检测

GDXray Castings的结果表现出与RGB数据集显著不同的行为，突出了模态偏移下迁移网络预训练表征的困难。主干冻结时，性能差距已经很大：ImageNet预训练的ConvNeXt-T达到21.32边界框mAP@50，而经DINOv3蒸馏的对应模型降至7.88。完全微调后，这种差距仍然存在，ImageNet预训练取得最佳结果29.74，DINOv3蒸馏为27.84。

这些结果表明，在ConvNeXt上通过DINOv3蒸馏学到的表征，即使经过下游适配，也不能有效迁移到X射线图像。与RGB表面检测不同，X射线图像在外观统计、纹理结构和物体成像方面都与自然网络图像存在较大差异，这可能削弱自监督预训练所编码语义先验的实用性。相比之下，ImageNet有监督初始化似乎为此检测任务提供了更稳定的起点。总体而言，GDXray结果表明，DINOv3的收益没有延伸到强模态偏移，传统有监督预训练在此设置下仍更可靠。

**图4：GDXray验证集上的边界框AP学习曲线。**使用Faster R-CNN，主干为通过ImageNet有监督分类或DINOv3蒸馏预训练的ConvNeXt-T。横轴为迭代次数，单位为$10^2$；纵轴为“边界框AP@IoU=0.50”。图例区分DINOv3、ImageNet有监督，以及冻结、微调。

## 6 讨论

本文表明，DINOv3预训练卷积主干在工业视觉检测任务中的表现，高度依赖下游任务与目标模态。对于RGB数据集，与传统ImageNet有监督预训练相比，DINOv3预训练特征没有明显优势，但主干完全微调后提供了最强初始化，使语义分割和RGB实例级定位收敛更快、最终性能更好。相反，在X射线缺陷检测中，这种优势消失，无论冻结还是完全微调，ImageNet有监督预训练都更优。总体来看，这些观察表明，现代视觉基础模型在工业检测中的收益，与其说来自直接而通用的迁移，不如说来自当目标数据与自然图像统计特性足够接近时，能够提供较强的微调先验。

这些观察有助于澄清将现代视觉基础模型用于工业迁移学习时的三个实际问题。

**DINOv3表征的直接可迁移性。**仅在有限程度上成立。在本文实验中，主干冻结时，DINOv3并未稳定优于传统ImageNet预训练，在X射线模态偏移下还明显失效。这说明，网络规模预训练特征与工业图像并非普遍一致。然而，RGB任务完全微调后的增益表明，在允许充分适配时，这些表征仍能提供有价值的初始化。

**与传统ImageNet迁移的区别。**结果表明，两种策略的迁移方式不同。ImageNet预训练仍然是鲁棒基线，尤其是在冻结设置和强模态偏移下。相比之下，DINOv3作为固定特征提取器的收益有限，但为RGB任务的下游适配提供了更强的起点。从这个意义上说，DINOv3作为通用冻结描述子的效果较弱，作为微调初始化的效果则更强。

**迈向工业专用自监督预训练。**自监督学习为具有广泛可迁移表征的视觉基础模型开辟了道路，DINOv3等面向密集特征的方法可能成为新的迁移学习标准。但通用视觉表征的概念仍受预训练数据分布约束，而该分布目前仍以自然网络图像为主。对于工业检测，视觉模式往往受到特定材料、采集设置和非标准模态塑造，因而一个重要的后续方向是工业专用自监督预训练。这样的方向或许能促成基础模型的发展，使其不仅能够迁移，而且在内在特性上就更符合真实工业环境。

## 致谢

本工作在FACTOLAB（UCA、CNRS与米其林的联合实验室）以及I-SITE CAP 20-25框架下的“创新交通与生产系统”国际研究中心开展，得到米其林轮胎制造商支持。

## 参考文献

[1] 《Severstal：钢材缺陷检测》. https://www.kaggle.com/c/severstal-steel-defect-detection. 原文引用页：4。

[2] Samet Akcay and Toby Breckon. 《迈向自动威胁检测：X射线安检成像中深度学习进展综述》.Pattern Recognition, 122:108245, 2022. 原文引用页：1。

[3] Samet Akc ¸ay, Mikolaj E. Kundegorski, Michael Devereux, and Toby P. Breckon. 《利用卷积神经网络迁移学习进行X射线行李安检图像中的物体分类》. InICIP, pages 1057–1061. IEEE, 2016. 原文引用页：1, 3。

[4] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. 《通过联合嵌入预测架构从图像中进行自监督学习》. InCVPR, pages 15619–15629, 2023. 原文引用页：2。

[5] Haoping Bai, Shancong Mou, Tatiana Likhomanenko, Ramazan Gokberk Cinbis, Oncel Tuzel, Ping Huang, Jiulong Shan, Jianjun Shi, and Meng Cao. 《VISION数据集：基于视觉的工业检测基准》, 2023. 原文引用页：3。

[6] Yunkang Cao, Jiangning Zhang, Luca Frittoli, Yuqi Cheng, Weiming Shen, and Giacomo Boracchi. 《AdaCLIP：使用混合可学习提示适配CLIP以进行零样本异常检测》. InECCV, pages 55–72, Cham, 2025. Springer Nature Switzerland. 原文引用页：3。

[7] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. 《通过对比聚类分配进行视觉特征的无监督学习》. In NeurIPS, pages 9912–9924. Curran Associates, Inc., 2020. 原文引用页：2。

[8] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv ´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. 《自监督视觉Transformer中的涌现特性》. In ICCV, pages 9650–9660, 2021. 原文引用页：2。

[9] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 《视觉表征对比学习的简单框架》. InProceedings of the 37th International Conference on Machine Learning, pages 1597– 1607. PMLR, 2020. 原文引用页：2。

[10] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. 《使用动量对比学习改进基线》, 2020. 原文引用页：2。

[11] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. 《用于通用图像分割的掩码注意力掩码Transformer》. InCVPR, pages 1290–1299, 2022. 原文引用页：4。

[12] Yuqi Cheng, Yunkang Cao, Haiming Yao, Wei Luo, Cheng Jiang, Hui Zhang, and Weiming Shen. 《真实工业表面缺陷检测综合综述：挑战、方法与前景》.Journal of Manufacturing Systems, 84:152–172, 2026. 原文引用页：3。

[13] Simon Damm, Mike Laszkiewicz, Johannes Lederer, and Asja Fischer. 《AnomalyDINO：利用DINOv2增强基于图像块的少样本异常检测》. In2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1319–1329, 2025. 原文引用页：3。

[14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 《一幅图像相当于16×16个词：用于大规模图像识别的Transformer》. InICLR, 2020. 原文引用页：4。

[15] Alaaeldin El-Nouby, Gautier Izacard, Hugo Touvron, Ivan Laptev, Herv´e Jegou, and Edouard Grave. 《自监督预训练是否必须使用大规模数据集？》?, 2021. 原文引用页：2, 7。

[16] Max Ferguson, Ronay Ak, Yung-Tsun Tina Lee, and Kincho H. Law. 《利用卷积神经网络自动定位铸件缺陷》. In2017 IEEE International Conference on Big Data (Big Data), pages 1726–1735, 2017. 原文引用页：1, 5。

[17] Ross Girshick. 《Fast R-CNN》. InICCV, pages 1440–1448, 2015. 原文引用页：4。

[18] Jean-Bastien Grill, Florian Strub, Florent Altch ´e, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, koray kavukcuoglu, Remi Munos, and Michal Valko. 《自举自身潜在表征：一种新的自监督学习方法》. InNeurIPS, pages 21271–21284. Curran Associates, Inc., 2020. 原文引用页：2。

[19] Taimur Hassan, Meriem Bettayeb, Samet Akc ¸ay, Salman Khan, Mohammed Bennamoun, and Naoufel Werghi. 《检测X射线图像中的违禁物品：一种轮廓候选学习方法》. InICIP, pages 2016–2020. IEEE, 2020. 原文引用页：3。

[20] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Girshick. 《Mask R-CNN》. InICCV, pages 2961–2969, 2017. 原文引用页：4。

[21] Kaiming He, Ross Girshick, and Piotr Dollar. 《重新思考ImageNet预训练》. InICCV, pages 4918–4927, 2019. 原文引用页：1, 2, 3, 4, 5。

[22] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. 《掩码自编码器是可扩展的视觉学习器》. InCVPR, pages 16000–16009, 2022. 原文引用页：1, 2。

[23] Ming Li, Jie Wu, Xionghui Wang, Chen Chen, Jie Qin, Xuefeng Xiao, Rui Wang, Min Zheng, and Xin Pan. 《AlignDet：对齐目标检测中的预训练与微调》. InICCV, pages 6866–6876, 2023. 原文引用页：7。

[24] Tongkun Liu, Bing Li, Xiao Jin, Yupeng Shi, Qiuying Li, and Xiang Wei. 《结合度量学习与视觉基础模型探索通用工业场景中的少样本缺陷分割》.Optics & Laser Technology, 192: 114078, 2025. 原文引用页：3, 4。

[25] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. 《面向2020年代的卷积网络》. InCVPR, pages 11976–11986, 2022. 原文引用页：2, 4, 5。

[26] Domingo Mery, Vladimir Riffo, Uwe Zscherpel, German Mondrag´on, Iv ´an Lillo, Irene Zuccar, Hans Lobel, and Miguel Carrasco. 《GDXray：用于无损检测的X射线图像数据库》.J Nondestruct Eval, 34(4):42, 2015. 原文引用页：5。

[27] Maxime Oquab, Timoth ´ee Darcet, Th´eo Moutakanni, Huy V . V o, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 《DINOv2：无需监督学习鲁棒视觉特征》.Transactions on Machine Learning Research, 2023. 原文引用页：1, 2。

[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 《从自然语言监督中学习可迁移视觉模型》. InProceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.

[29] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer. 《SAM 2：分割图像和视频中的任意对象》. InICLR, 2024. 原文引用页：1。

[30] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. 《ImageNet大规模视觉识别挑战》.Int J Comput Vis, 115(3):211–252, 2015. 原文引用页：1, 2, 3。

[31] Jacob Shermeyer, Thomas Hossler, Adam Van Etten, Daniel Hogan, Ryan Lewis, and Daeil Kim. 《RarePlanes：合成数据起飞》. InProceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 207– 217, 2021. 原文引用页：4。

[32] Oriane Sim ´eoni, Huy V . V o, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha ¨el Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timoth´ee Darcet, Th´eo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herv ´e J ´egou, Patrick Labatut, and Piotr Bojanowski. 《DINOv3》, 2025. 原文引用页：1, 2。

[33] Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, and Lei Li. 《用于自监督视觉预训练的密集对比学习》. InCVPR, pages 3024–3033, 2021. 原文引用页：2。

[34] Zihao Wang and Lei Wu. 《深度卷积网络归纳偏置的理论分析》.NeurIPS, 36: 74289–74338, 2023. 原文引用页：4, 7。

[35] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. 《Detectron2》.https://github.com/facebookresearch/detectron2, 2019. 原文引用页：5。

[36] Hui-Yue Yang, Hui Chen, Ao Wang, Kai Chen, Zijia Lin, Yongliang Tang, Pengcheng Gao, Yuming Quan, Jungong Han, and Guiguang Ding. 《通过自感知调优，利用SAM进行可提示异常分割》.AAAI, 39 (12):13017–13025, 2025. 原文引用页：3。

[37] Shuxuan Zhao, Sichao Liu, Yishuo Jiang, Bo Zhao, Youlong Lv, Jie Zhang, Lihui Wang, and Ray Y . Zhong. 《面向智能制造的工业基础模型：系统综述》.Journal of Manufacturing Systems, 82: 420–448, 2025. 原文引用页：3。

[38] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. 《使用在线词元化器的图像BERT预训练》. InICLR, 2021. 原文引用页：2。

## 术语对照

| 原文术语 | 本文译法 |
| --- | --- |
| supervised pretraining | 有监督预训练 |
| distillation | 蒸馏 |
| frozen backbone | 冻结主干 |
| full finetuning | 完全微调 |
| dense prediction | 密集预测 |
| linear probing | 线性探测 |
| modality shift | 模态偏移 |
| inductive bias | 归纳偏置 |
| defect-aware cropping | 缺陷感知裁剪 |
| bounding box / mask mAP | 边界框／掩码平均精度均值 |
