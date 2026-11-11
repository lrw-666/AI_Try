# MVTec AD 2 数据集：面向无监督异常检测的进阶场景

**原文：**[The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection](<17-MVTec-AD2-Advanced-Anomaly-Detection-2025.pdf>)。本译文对应本目录 21 页 PDF，arXiv:2503.21622v1，2025 年 3 月 27 日。翻译日期：2026-11-11。[arXiv 原文](https://arxiv.org/abs/2503.21622)。

**作者：**Lars Heckler-Kram¹˒²、Jan-Hendrik Neudeck¹、Ulla Scheler¹、Rebecca König¹、Carsten Steger¹。

**单位：**¹ 德国 MVTec Software GmbH；² 德国慕尼黑工业大学。

**翻译说明：**按本地 PDF 翻译摘要、正文第 1—6 节、局限与未来工作、参考文献 [1]—[37]、作者简介及补充材料 A—D；保留公式 (1)、图 1—22 图注和表 1—12 全部数据。跨页续段合并，运行页眉页脚和页码省略；图像不重绘，图内类别和说明以文字翻译。数值、模型名、设备型号和引用编号保留；“当前最佳”等均指原文时点。参考文献保留原始书目信息并附中文题名。

## 摘要

近年来，在 MVTec AD、VisA 等现有异常检测基准上，分割 AU-PRO 性能开始饱和，先进模型的竞争差距往往不到一个百分点。这种区分能力不足使模型无法得到有意义的比较，从而阻碍领域进展，尤其考虑到机器学习结果本身具有随机性。我们提出 MVTec AD 2：由八种异常检测场景组成、包含超过 8,000 幅高分辨率图像的数据集。它涵盖以往数据集没有考虑的、具有挑战性且与工业高度相关的检测用例，包括透明和重叠物体、暗场和背光照明、正常数据变化很大的物体，以及极小缺陷。我们全面评估了先进方法，表明其平均 AU-PRO 仍低于 60%。此外，数据集提供照明条件变化的测试场景，以评估方法面对真实分布偏移的鲁棒性。我们提供公开可访问的评估服务器，保存测试集逐像素精确真值：[评估服务器](https://benchmark.mvtec.com/)。所有图像数据可从 [数据集页面](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) 获取。

**关键词：**异常检测；数据集；基准；评估指标；深度学习。

## 1 引言

过去几年，计算机视觉中的无监督异常检测与定位受到广泛关注。训练时只能访问无异常数据，而测试时要识别异常数据，这是自动驾驶 [1,2]、医疗 [3,4]、视频监控 [5,6] 和视觉检测 [7,8] 等领域共同面临的挑战。在工业环境中，视觉异常检测尤其是质量保证的重要工具。此处不仅要求系统将制成品图像判为不合格或合格，还要提供与异常区域对应的逐像素精确异常图。可靠的缺陷定位保证系统的可解释性与可信度，有助于及早发现生产线上的系统性故障，并使后续处理能对缺陷严重程度作出有依据的判断。

工业异常检测领域的进展主要由合适数据集的可用性推动。MVTec AD [7,9]、VisA [8] 等早期数据集着力于提供广泛的工业异常检测场景、定义任务，并建立评估指标和基准协议标准。此后又引入逻辑约束等新异常类型 [10]，以及三维信息等额外模态 [11,12]。尽管现有数据集尚未覆盖工业视觉检测的全部设置，例如不同照明类型，但从实际角度看，研究界的共同努力已经促成大量可部署到真实生产系统的方法。同时，MVTec AD、VisA 等成熟基准开始饱和。先进模型的性能差距往往不到一个百分点；深度学习模型的随机性以及对适当超参数的依赖，使真正创新与增量改进难以区分。

此外，不一致的评估设置妨碍有意义的模型比较。分割 AU-PRO [9] 等指标虽然定义明确，但有时使用的是稍加修改或近似的版本。更严重的是，依据测试数据调节超参数这一常见做法不恰当地使用了测试集，因为这违反了无监督异常检测的基本定义：不知道将出现哪些缺陷。

为推动异常检测研究，我们提出含八种进阶异常检测场景的 MVTec AD 2 数据集与基准（图 1）。相较现有数据集，它给当前先进方法带来显著挑战（表 1）：在 MVTec AD 等成熟数据集上 AU-PRO₀.₃₀ 超过 90% 的方法，在本数据集上最高只有 58.7%。新数据集聚焦于工业检测高度相关、却被现有数据集忽视的设置（表 2），包括大图中的小缺陷、图像边缘的缺陷、正常状态变化很大的物体类别、透明和反光物体，以及实例可能重叠且位置和数量随机变化的散装物品。

**图 1：MVTec AD 2 的物体。**每种物体均显示一幅无缺陷图像和一幅异常图像，异常以红色轮廓标出。异常区域特写显示逐像素精确真值标签。类别为膨胀管（Wall Plugs）、核桃（Walnuts）、织物（Fabric）、大米（Rice）、小瓶（Vial）、水果果冻（Fruit Jelly）、金属薄板（Sheet Metal）、罐（Can）；Anomaly-Free/Anomalous 分别为无异常/有异常。

**表 1：先进方法在成熟数据集和 MVTec AD 2 上的异常分割 AU-PRO₀.₃₀（%）及平均性能。成熟数据集已饱和，新数据集带来挑战和真正方法改进的机会。**

| 方法 | MVTec AD [7] | VisA [8] | MVTec AD 2（本文） |
| --- | --- | --- | --- |
| PatchCore | 92.7 | 79.7 | 53.8 |
| RD | 93.9 | 89.0 | 53.0 |
| RD++ | 95.0 | 89.3 | 54.3 |
| EfficientAD | 93.5 | 94.0 | 58.7 |
| MSFlow | 91.2 | 88.4 | 52.7 |
| SimpleNet | 89.6 | 68.9 | 46.4 |
| DSR | 90.8 | 68.1 | 49.0 |
| 平均 | 92.4 | 82.4 | 52.6 |

MVTec AD 2 还从两个重要方面扩展现有基准。首先，每个场景都在至少四种照明条件下拍摄。因此，其设计为评估方法对照明变化所致真实分布偏移的鲁棒性提供了新可能。这种变化自然发生于同一模型部署到不同机器，以及照明设备老化的工业应用中。

其次，真值评估只能在我们的公开基准服务器上进行，从而防止针对测试集优化。这样，MVTec AD 2 为不同方法更标准化、因而更公平的比较铺平道路。本文有三项贡献：

- 提出高度挑战性的新数据集 MVTec AD 2，当前最佳模型仅达到 58.7% AU-PRO₀.₃₀，为视觉异常检测的显著改进提供机会。
- 首次能够评估针对各物体场景设计的照明条件变化所致真实分布偏移下的异常检测鲁棒性。
- 提供公共评估服务器，使研究界更可靠地衡量领域进展；同时评估七种先进方法并讨论实际适用性。

## 2 相关工作

**表 2：按不同判据定性比较数据集。**本文数据集提供正常状态变化大的真实图像及不同照明，用于研究分布偏移；另一项主要贡献是高难度（表 1），此表略去。不关注逻辑异常或三维数据。

| 判据 | MVTec AD | MVTec LOCO AD | VisA | MVTec 3D AD | Eye-candies | Real-IAD | MVTec AD 2（本文） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 真实图像 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| 正常数据变化大* | ✗ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| 图像边缘缺陷 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 每图物体数量变化* | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 结构缺陷 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 逻辑缺陷 | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 二维数据 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 三维数据 | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| 不同照明条件 | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |
| 透明物体 | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |

*指训练与验证正常数据允许的变化。

工业视觉质量控制依赖异常检测系统可靠地定位，即分割异常区域。定量评估此类方法需要具有逐像素精确分割真值的数据集。本节简要回顾最常用的具有逐像素异常掩码标签的数据集，并概述现有异常检测方法。

### 2.1 异常检测数据集

MVTec AD [7] 包含覆盖 15 种物体类别的 5,354 幅高分辨率彩色图像。它推动了工业异常检测研究，但如今性能几乎饱和，最佳模型平均 AU-PRO 达 97.8% [13]。类似地，VisA [8] 包含 12 类物体的 10,821 幅图像，其中四种是每幅图像实例数量固定的多物体设置。它扩展了科研可用的标准异常检测场景，但也与 MVTec AD 一样接近被解决 [14,15]。

MVTec LOCO AD [10] 聚焦逻辑异常：缺陷图像中的元素保持相同总体外观，却违反逻辑约束，例如盒内元素数量不正确。目前最佳模型在该数据集上的定位性能为 79.8% [14]。较新的 Real-IAD [16] 聚焦多视角以及种类更多、以单个实例拍摄的非形变物体，也包括 MVTec AD、VisA 中出现的物体类型。它以庞大规模著称，含 30 种物体的超过 150,000 幅高分辨率图像，但物体正常状态变化较小。此外，即使仅采用原图 1/20 大小的分辨率，现有模型在无监督设置下已达到 94.4% 像素 AU-PRO₀.₃₀ [16,17]。这说明缺陷与正常数据分布容易区分。其他数据集开拓了三维 [11]、多姿态 [18] 等异常检测领域。Eyecandies [12] 提供十类糖果的合成数据，包括透明与非对称物体。合成渲染除能提供完美真值，还能模拟不同照明条件。

我们认为当前数据集不利于新研究主要有两个原因。第一，现有数据集过于饱和，模型间难以形成明显区分。近期数据集虽然覆盖新问题设置，但在所选物体上实质重复了已有难度水平，因此发表时就已达到超过 90% 的解决程度。第二，尽管上述数据集覆盖了广泛应用，仍缺少重要现实用例。例如，现代设备的生产速度可能要求对散装样品进行视觉检测，其位置、数量变化很大。此外，实际系统可能需要未经重新训练就部署到环境条件（如照明）略有差异的多台设备。MVTec AD 2 提供足够困难且多样的八种二维异常检测场景，使研究者基于真实数据开发解决这些挑战的方法。

### 2.2 无监督异常检测方法

得益于研究者过去几年的严谨工作，已有多种方法。重建方法 [19–21] 恢复图像的无异常状态，并计算输入与重建结果的差异。其他方法将正常数据分布显式存入记忆库 [22]，或通过保存无异常数据嵌入的统计属性 [23,24]、用归一化流映射至预定义分布 [25–28] 等方式隐式存储。学生—教师方法 [29,30,14,17,31] 在一定程度上结合了两种原则：学生学习模仿预训练教师在无缺陷训练数据上的输出，测试时异常会造成两个模型输出偏离。利用大型语言模型，还可比较正常状态的文本嵌入与测试图像嵌入以检测异常 [32,33]。此外，一些框架施加合成异常来区分正常与异常数据分布 [34–36]。

MVTec AD [7]、VisA [8] 等当前基准接近饱和，妨碍新方法开发。当先进模型之间的性能差落在随机波动范围内时，也难以区分真正进步与增量改进。因此需要新挑战，尤其是标准二维视觉检测设置。MVTec AD 2 提供进阶场景，以促进新方法设计，并支持更公平、更有意义的比较。

## 3 MVTec AD 2 数据集

### 3.1 进阶异常检测场景

MVTec AD 2 由工业应用中八种具有挑战性的无监督异常检测场景组成，共含 8,004 幅高分辨率图像。图 1 展示正常和异常实例。各场景突出特定难点：散装物品（膨胀管、核桃、大米）包含以无控制空间模式摆放、相互重叠与遮挡的物体；纹理物体（织物、金属薄板）正常变化较大；反光金属物体（金属薄板、罐）和透明物体（小瓶、水果果冻）分别带来反射与畸变挑战。缺陷包括划痕、孔洞等表面瑕疵、异物污染以及部件缺失。表 3 总结源于多种真实应用的场景难点。

此外，我们设计异常覆盖图像整个高度与宽度，并直接出现在边缘（图 2）。这给使用中心裁剪训练的模型带来新挑战，并可评估常见填充伪影。

**图 2：MVTec AD、VisA 和 MVTec AD 2 的缺陷分布。**图中显示各数据集归一化的真值标签分布。前两者的缺陷标签主要集中于图像中心，而 MVTec AD 2 在边缘具有大量缺陷，可测试方法对边界伪影的鲁棒性。

为覆盖工业视觉检测的多样性，除 RGB 图像外，数据集还包含单通道灰度图像（金属薄板、小瓶、膨胀管）。它也使用工业实践常见而现有数据集忽视的照明，如背光（小瓶、水果果冻）和暗场（金属薄板）[37，第 2.1.5 章]。图像分辨率为 2.6—5 百万像素，宽高比不同，详见表 4。

### 3.2 遵循无监督设置的数据集设计

数据集包含训练、验证和测试集。遵循无监督设置，训练与验证集只含正常数据。训练数据仅在常规照明下采集，因为实际照明变化主要发生于测试阶段。测试数据分为公开示例测试集 TEST_pub 和两个私有测试集 TEST_priv、TEST_priv,mix。TEST_pub 包含所有照明条件下的少量正常与异常图像及对应分割真值，便于本地测试和初步性能估计。规模大得多的私有测试集只公开图像，其真值私有，只能通过基准服务器评估（第 5 节）。TEST_priv 与训练集照明相同；TEST_priv,mix 显示与 TEST_priv 相同的场景，但每幅测试图像随机选择已见或未见照明，正常与异常数据均如此。由此可直接比较真实分布偏移下的模型行为。数据从 [数据集页面](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) 下载，以 CC BY-NC-SA 4.0 许可发布。

### 3.3 未见照明引起的分布偏移

每种物体都通过调整曝光时间，提供比训练图像更亮或更暗的图像。其他测试照明偏移根据各场景物体设计。例如，暗场照明采用两个不同照射角度；反光表面增加 LED 点光源产生反射；散装物品增加点光源，使场景光照分布不均。所有类别至少有四种照明条件，图 3 给出示例。变化照明时，物体位置等其余参数和相机参数（曝光时间除外）保持不变。

### 3.4 高质量逐像素精确真值

为保证高质量分割真值并覆盖全部异常，采集图像时先现场制作粗标注，再由人工标注员遵循物体专用准则精修。精修标签经过人工与自动检查，以减少不完整或虚假区域。我们核验不同场景间真值的全局一致性，开发软件识别可能误标的微小区域或孔洞。若某缺陷在不同照明下外观明显变化，也相应调整标注。最后人工筛选正常数据，确保没有缺陷，从而严格遵守无监督训练与验证要求。

**表 3：各物体的采集设置、缺陷和主要挑战。**

| 物体 | 采集设置 | 出现的缺陷 | 主要挑战 |
| --- | --- | --- | --- |
| 罐 | 汽水罐的漫射明场正面照明 | 印刷缺陷；划痕 | 纹理表面的微小缺陷；反光；罐旋转带来较大正常变化；大宽高比 |
| 织物 | 织物检测的漫射明场正面照明 | 切口和孔洞；颜色不一致；松线和多余布片 | 微小低对比缺陷；印花变化带来较大正常变化 |
| 水果果冻 | 含水果块果冻的漫射明场背光和正面照明 | 不同纹理与尺寸的异物污染；表面划痕 | 成分外观与数量变化造成正常变化大；半透明；低对比缺陷 |
| 大米 | 散装米粒的漫射明场正面照明 | （半）透明塑料污染；异物污染 | 极低对比缺陷；透明污染物 |
| 金属薄板 | 金属片的定向暗场正面照明 | 划痕、切口、孔洞；异物 | 随机表面反射；大宽高比；每图多个不同大小缺陷 |
| 小瓶 | 含透明起泡液体单瓶的漫射明场背光 | 异物污染；二维码损坏或缺失；液位偏差；瓶盖打开或缺失 | 小瓶和缺陷透明；小而低对比缺陷；旋转、气泡与二维码造成正常变化大 |
| 膨胀管 | 多个膨胀管的漫射明场正面照明 | 划痕、切口、缺失部件；异物或碎片污染；尺寸错误 | 散装实例接触、重叠、被边框截断；数量和位置变化 |
| 核桃 | 多个核桃的漫射明场正面照明 | 裂纹和孔洞；异物或不完整核桃污染 | 接触、重叠、被边框截断；单体大小形状结构变化大；数量与位置变化 |

### 3.5 图像采集设置

下面详细介绍表 3 的各物体采集设置。为在 TEST_pub、TEST_priv,mix 测试数据中引入照明变化（第 3.2 节），我们改变曝光时间，并安装额外光源；除非另有说明，额外光源与常规照明组合使用（第 3.3 节）。表 5 列出硬件，表 6 列出各条件的光源。

**罐。**大型条形灯从上方照亮场景，使整个视野照明均匀。罐前方背景是白纸，有助于产生漫射均匀照明。测试采集时，将两个额外点光源对准罐，在金属表面及标签上产生镜面高光。

**织物。**面光源从上方照亮布片。测试采集安装两个额外点光源：第一个以相对织物表面的低角度放置，形成跨图像照明梯度；第二个制造额外杂散光。

**表 4：数据集统计。**各场景训练、验证、测试图像数及尺寸。测试括号内为无异常/有异常数量。

| 物体 | 训练 | 验证 | TEST_pub | TEST_priv | TEST_priv,mix | 测试总计 | 尺寸 W×H |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 罐 | 412 | 46 | 162 (72/90) | 321 (145/176) | 321 (145/176) | 804 (362/442) | 2232×1024 |
| 织物 | 387 | 43 | 156 (66/90) | 314 (133/181) | 314 (133/181) | 784 (332/452) | 2448×2048 |
| 水果果冻 | 263 | 37 | 80 (20/60) | 255 (71/184) | 255 (71/184) | 590 (162/428) | 2100×1520 |
| 大米 | 313 | 35 | 132 (42/90) | 277 (96/181) | 277 (96/181) | 686 (234/452) | 2448×2048 |
| 金属薄板 | 137 | 19 | 114 (24/90) | 142 (36/106) | 142 (36/106) | 398 (96/302) | 4224×1056 |
| 小瓶 | 291 | 41 | 140 (35/105) | 276 (78/198) | 276 (78/198) | 692 (191/501) | 1400×1900 |
| 膨胀管 | 293 | 33 | 150 (60/90) | 232 (96/136) | 232 (96/136) | 614 (252/362) | 2448×2048 |
| 核桃 | 432 | 48 | 150 (60/90) | 228 (93/135) | 228 (93/135) | 606 (246/360) | 2448×2048 |

TEST_pub：所有照明的图像和真值均公开。TEST_priv：图像公开，真值仅在服务器，照明与训练相同。TEST_priv,mix：图像公开，真值仅在服务器，包含已见与未见照明。

**水果果冻。**面光源提供背光，点光源提供正面照明。光线透过半不透明果冻，令内部缺陷可见，同时突出前侧表面缺陷。塑料杯置于聚甲基丙烯酸甲酯板上，杯底在板上形成反射；反射中可见的缺陷同样标注。测试时在上方安装额外 LED 点光源，检验方法对杂散照明的鲁棒性。

**大米。**大型面光源照明，避免干扰阴影。测试数据由两个额外点光源产生照明梯度。

**金属薄板。**采用环形灯以低入射角形成暗场照明，使表面不规则处向相机反射光线，有效识别金属表面微小划痕与孔洞。与其他类别一样，测试采集改变常规照明曝光时间；还采用入射角更陡的第二个环形灯并改变曝光时间。两个环形灯单独使用，形成不同照明模式，尤其影响薄板轮廓。

**小瓶。**面光源背光有助于检测完全或部分透明污染物，例如微小塑料薄膜碎片。测试采集另安装三个 LED 点光源：第一个位于瓶上方，产生不均匀的自上而下照明；另两个产生瓶表面的额外镜面高光。

**膨胀管。**大型面光源从上方照亮散装膨胀管。测试时两个 LED 点光源以低入射角照射物体，形成强亮度梯度。

**核桃。**与膨胀管设置类似，面光源配合两个额外 LED 点光源，在测试数据中形成不规则照明。

**表 5：图像采集相机与镜头。**

| 物体 | 相机型号 | 镜头型号（焦距） |
| --- | --- | --- |
| 罐 | IDS GV-5280CP-C-HQ | Tamron MA111F16VIR (16mm) |
| 织物 | IDS GV-5280CP-C-HQ | Tamron MA111F16VIR (16mm) |
| 水果果冻 | IDS GV-5280CP-C-HQ | Tamron MA111F16VIR (16mm) |
| 大米 | Daheng MER2-503-36U3M | Moritex ML-U1614MP9 (16mm) |
| 金属薄板 | SVS-Vistek exo834MTLGEC | Tamron MA111F16VIR (16mm) |
| 小瓶 | Daheng MER2-503-36U3M | Moritex ML-U1614MP9 (16mm) |
| 膨胀管 | Daheng MER2-503-36U3M | Moritex ML-U1614MP9 (16mm) |
| 核桃 | IDS GV-5280CP-C-HQ | Computar M1214-MP2 (12mm) |

**表 6：测试集不同照明使用的光源。金属薄板的两环形灯各使用三种不同曝光时间。**

| 物体 | 常规、欠曝、过曝 | 额外光源偏移 1 | 偏移 2 | 偏移 3 |
| --- | --- | --- | --- | --- |
| 罐 | B | B, S1 | B, S2 | B, S1, S2 |
| 织物 | F | F, S1 | F, S2 | F, S1, S2 |
| 水果果冻 | F, S1 | F, S1, S2 | - | - |
| 大米 | F | F, S1 | F, S2 | F, S1, S2 |
| 金属薄板 | R1 | R2 | R2 | R2 |
| 小瓶 | F | F, S1 | F, S2 | F, S3 |
| 膨胀管 | F | F, S1 | F, S2 | F, S1, S2 |
| 核桃 | F | F, S1 | F, S2 | F, S1, S2 |

B：条形灯；F：面光源；R：环形灯；S：点光源。

**图 3：若干物体的照明条件变化。**除常规曝光外，每个场景也在轻度欠曝和过曝下拍摄。额外光源造成物体特定外观变化，如小瓶反射、膨胀管照明不均，以及大米和核桃轻微色温变化。图内 regular、underexposed、overexposed、additional light sources 分别为常规、欠曝、过曝和额外光源。

## 4 基准

### 4.1 评估方法

作为基线，我们评估学生—教师方法 EfficientAD [14]、Reverse Distillation [17]、Reverse Distillation Revisited [31]，记忆库方法 PatchCore [22]，归一化流方法 MSFlow [28]，以及模仿异常数据的方法 SimpleNet [36]、Dual Subspace Re-Projection [35]。均采用官方实现及原论文对 MVTec AD 报告的默认设置，不按场景调节训练时长等超参数。除非明确说明，图像缩放为 256×256。下面介绍实现和参数细节。

**逆向蒸馏（RD）。**使用 [17] 官方实现。其输入尺寸必须是 32 的倍数；不满足时，高和宽分别采用最接近且更小的可行尺寸。表 1 的 MVTec AD 结果来自 [17]，VisA 使用相同参数获得。

**重新审视逆向蒸馏（RD++）。**使用 [31] 官方实现。实现基于 RD，尺寸约束类似，采用相同缩放过程。表 1 的 MVTec AD 结果来自 [31]，VisA 使用相同参数。

**EfficientAD。**遵循 [14] 实现细节。MVTec AD 2 采用 EfficientAD-S，因为其延迟低于稍大的 EfficientAD-M 而性能接近。不过表 1 使用 EfficientAD-M 展示该方法在 MVTec AD、VisA 上的总体最佳表现。

**PatchCore。**使用 [22] 官方实现及默认集成设置：WideResNet-101、ResNeXT-101、DenseNet-201 骨干，近似贪心核心集子采样，采样比例 0.01%，最终聚合嵌入维度 384。关闭中心裁剪，以便检测边缘缺陷。

**MSFlow。**使用 [28] 官方实现，但为与其他方法可比，表 1 训练和推理采用 256×256 而非 512×512。VisA 使用与 MVTec AD 相同参数。

**SimpleNet。**使用 [36] 官方实现和 WideResNet50 特征提取器。遵循 [14]，不设置场景专用训练时长，统一训练 160 轮。

**双子空间重投影（DSR）。**使用 [35] 官方实现。因训练时间长，最大图像尺寸只训练大米类别。

### 4.2 评估指标

#### 4.2.1 与阈值无关的指标

定位质量采用像素级逐区域重叠（PRO）曲线下面积 [9]（AU-PRO），即在不同阈值下对误报率 FPR 积分。PRO 定义为：

$$\mathrm{PRO}=\frac1k\sum_{i=1}^{|D_{\mathrm{test}}|}\sum_{j=1}^{k_i}\frac{|P_{\mathrm{ano}}\cap C_{i,j}|}{|C_{i,j}|}.\tag{1}$$

其中 Cᵢⱼ 为真值图像 i 中连通分量 j 的异常像素集合，P_ano 为给定阈值 t 下预测异常像素的集合。与同样可评价像素定位、以真阳性率 TPR 对误报率构成的接收者操作特征曲线下面积 AU-ROC 不同，AU-PRO 平等考虑每个异常区域。存在多个区域时，TPR 被大异常主导，可能多数缺陷完全未检出却高估定位质量；PRO 不受异常尺寸差异污染（图 4）。工业应用要求不论大小检测每个缺陷，故选择 AU-PRO。

**图 4：PRO 与 TPR 比较。**给定异常真值和不同阈值化预测，PRO 平等考虑每个区域，TPR 被大缺陷主导而高估质量。四种情况：(i) PRO=(4/4+2/2+20/20)/3=1，TPR=26/26=1；(ii) PRO=(2/4+1/2+10/20)/3=0.5，TPR=13/26=0.5；(iii) PRO=(4/4+2/2+0/20)/3=0.67，TPR=6/26=0.23；(iv) PRO=(0/4+0/2+20/20)/3=0.33，TPR=(0+0+20)/26=0.77。图例为真值、阈值化预测、异常像素、预测异常像素、正常像素。

通常 PRO 曲线积分至 FPR=0.3（AU-PRO₀.₃₀）。然而，现代工业生产的高质量标准高度依赖准确分割，因此我们降低误报像素容忍度，将 MVTec AD 2 积分上限设为 0.05（AU-PRO₀.₀₅）。图 5 展示影响。FPR=30% 的异常图完全不符合人们对有意义异常图的期望。对本数据集大图中的小缺陷，收紧上限尤其重要：5 百万像素图像中一个 10 像素缺陷，即使 FPR 仅为 5%，误分区域也超过缺陷本身的 25,000 倍。不过表 1 仍用 AU-PRO₀.₃₀，以说明即便宽松评估下数据集仍很难；详值见补充材料。

**图 5：降低 AU-PRO 积分上限以促成更有意义的异常图。**展示 MSFlow 在 256×256 大米图像上的示例异常图，并阈值化至指定 FPR。常见上限 FPR=0.3 允许分割缺陷区域过大。图内含图像与真值叠加、预测、FPR=0.3/0.05 的阈值化预测；色标为 0.0、0.2、0.4、0.6、0.8、1.0、1.2、1.4、1.6。

#### 4.2.2 依赖阈值的指标

AU-PRO 等指标对总体异常图质量尤其有用。但工业实际的首要目标是有依据、可解释地判断受检零件是否有缺陷，因此必须在像素和图像层面施加阈值以分开两类。我们将异常图阈值化为二值分割图，用像素精确率与召回率的调和平均 F₁ 与真值比较。由于没有缺陷图像可用，基线分割阈值设为无缺陷验证图像逐像素异常值的均值加三倍标准差。从实践角度，整图合格或不合格应由是否检测到异常区域得出。因此同一分割阈值同时用作分类阈值，最大像素值作为图像异常分数；阈值化后同样用 F₁ 衡量图像分类。

#### 4.2.3 运行时间与内存消耗

实际部署必须仔细考虑可用硬件，因为算力和内存有限，运行时间与占用很快成为关键约束。我们在 NVIDIA RTX 2080Ti 上测量所有方法单图推理时间和峰值内存。

总体遵循 [14] 流程，并提供测量代码。由于着重展示较大图像的影响，没有为推理效率优化架构。具体为以 batch size=1、float32 先预热 1,000 次，再测量 1,000 次前向传播的平均时间。使用 Python time，从样本传至 GPU 前开始，到异常图在 CPU 可用时结束。内存用官方 PyTorch 功能 `torch.cuda.memory_stats()['reserved_bytes.all.peak']` 测量。PatchCore 推理时间取决于由训练图像数决定的记忆库大小，因此合并 TEST_pub 与 TEST_priv，在预热后约 200 个样本上平均。

### 4.3 结果

#### 4.3.1 与阈值无关的指标

表 7 给出 AU-PRO₀.₀₅ 定位结果。照明不变的常规设置下，EfficientAD 在 TEST_priv 平均为 30.8%，总体最佳，其后依次是 PatchCore、RD++、RD、MSFlow、SimpleNet、DSR。总体定位质量仍有很大改进空间，罐类别尤其困难。表 7 也表明 RD 等方法更耐受照明偏移：RD 在 TEST_priv,mix 仅下降 1.4 个百分点，而 MSFlow 从 24.3% 降到 11.9%，下降 12.4 个百分点。以往无法充分研究的此类鲁棒性差异既说明数据集贡献，也凸显未来方法开发必须考虑环境变化。

**表 7：TEST_priv / TEST_priv,mix 的分割 AU-PRO₀.₀₅（%）。所有方法在 TEST_priv 上均低于 31%；两测试集差距体现照明变化鲁棒性。**

| 物体 | PatchCore | RD | RD++ | EfficientAD | MSFlow | SimpleNet | DSR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 罐 | 4.7 / 4.6 | 7.0 / 7.5 | 7.7 / 7.0 | 9.6 / 1.3 | 6.7 / 0.8 | 8.4 / 1.9 | 13.9 / 3.5 |
| 织物 | 11.0 / 12.0 | 3.5 / 3.6 | 4.8 / 5.3 | 22.2 / 13.0 | 14.1 / 14.3 | 6.6 / 7.3 | 6.8 / 5.3 |
| 水果果冻 | 46.7 / 46.7 | 48.2 / 48.2 | 54.4 / 54.4 | 50.5 / 47.6 | 49.4 / 38.3 | 39.8 / 38.5 | 36.0 / 34.2 |
| 大米 | 25.6 / 18.5 | 11.2 / 11.4 | 12.2 / 10.2 | 27.6 / 4.3 | 21.5 / 12.2 | 8.7 / 4.2 | 7.8 / 8.3 |
| 金属薄板 | 15.2 / 13.0 | 9.5 / 8.8 | 9.3 / 8.9 | 11.8 / 5.3 | 11.6 / 7.7 | 12.0 / 9.1 | 18.0 / 16.1 |
| 小瓶 | 62.2 / 59.1 | 62.1 / 60.3 | 63.0 / 57.0 | 55.6 / 47.7 | 38.0 / 4.0 | 47.8 / 23.3 | 50.0 / 48.1 |
| 膨胀管 | 12.8 / 9.9 | 19.7 / 12.2 | 15.4 / 10.1 | 20.3 / 1.2 | 12.6 / 0.2 | 5.7 / 1.9 | 3.9 / 6.5 |
| 核桃 | 51.8 / 44.5 | 50.2 / 48.3 | 49.7 / 48.9 | 48.8 / 33.0 | 40.4 / 18.1 | 40.0 / 23.4 | 25.7 / 17.1 |
| 平均 | 28.8 / 26.0 | 26.4 / 25.0 | 27.1 / 25.2 | 30.8 / 19.2 | 24.3 / 11.9 | 21.1 / 13.7 | 20.3 / 17.4 |

#### 4.3.2 依赖阈值的指标

表 8 报告阈值化异常图的像素 F₁。常规设置下 MSFlow 总体最佳，在 TEST_priv 达 21.8%。除照明变化的挑战外，阈值化后的性能不一定与无阈值指标相关。例如 PatchCore 的 F₁ 明显差于所有其他方法，却取得类似 AU-PRO。因此仅用无阈值指标不足以评估实际适用性，新方法开发与合适阈值的确定应当同时处理。

**表 8：两个私有测试集二值图像的分割 F₁（%）。像素正常/异常阈值仅由无缺陷验证图像确定。**

| 物体 | PatchCore | RD | RD++ | EfficientAD | MSFlow | SimpleNet | DSR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 罐 | 0.3 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 | 0.8 / 0.1 | 5.0 / 0.1 | 0.6 / 0.1 | 0.4 / 0.1 |
| 织物 | 11.5 / 9.8 | 2.6 / 2.2 | 2.9 / 2.3 | 7.6 / 1.0 | 22.0 / 4.1 | 21.6 / 10.2 | 7.9 / 5.0 |
| 水果果冻 | 8.7 / 8.2 | 22.5 / 22.7 | 26.9 / 26.7 | 20.8 / 18.2 | 47.6 / 38.1 | 25.1 / 23.0 | 17.9 / 17.2 |
| 大米 | 3.8 / 4.2 | 7.0 / 3.9 | 9.5 / 2.9 | 15.0 / 0.5 | 19.1 / 1.8 | 11.6 / 1.0 | 1.5 / 1.4 |
| 金属薄板 | 1.8 / 1.1 | 41.3 / 39.2 | 40.9 / 37.7 | 9.3 / 3.8 | 13.0 / 7.6 | 14.6 / 2.8 | 13.9 / 14.4 |
| 小瓶 | 2.3 / 2.2 | 28.0 / 28.3 | 28.2 / 22.8 | 30.5 / 26.5 | 23.3 / 6.2 | 31.9 / 17.5 | 28.2 / 27.9 |
| 膨胀管 | 0.0 / 0.0 | 1.9 / 0.8 | 1.3 / 0.9 | 4.4 / 0.3 | 0.1 / 0.2 | 1.0 / 0.3 | 0.4 / 0.4 |
| 核桃 | 1.2 / 1.3 | 41.2 / 36.7 | 44.1 / 40.5 | 34.6 / 13.3 | 44.5 / 14.3 | 35.2 / 14.3 | 17.0 / 9.6 |
| 平均 | 3.7 / 3.4 | 18.1 / 16.7 | 19.2 / 16.7 | 15.4 / 8.0 | 21.8 / 9.0 | 17.7 / 8.7 | 10.9 / 9.5 |

#### 4.3.3 以更高代价达到更高性能

结果表明当前先进方法受到本数据集挑战。然而，数据集设计关注小而低对比缺陷等因素，因此我们假设部分异常在缩放至标准 256×256 时消失。于是将输入尺寸增大至原始宽高各一半开展实验。

图 6 的大米示例表明，提高分辨率可大幅提升性能。例如 PatchCore、MSFlow 在最大尺寸的 AU-PRO₀.₀₅ 约翻倍，但时间和内存同时增长超过一个数量级。PatchCore 定位最好，却每图需要 2 秒；SimpleNet 最大尺寸则超出可用内存。工业应用几乎总有算力、内存或极短运行时间约束，因此可能根本不能部署这类模型。MSFlow 在相似定位水平下扩展效率明显更好。为进一步把时间与内存确立为重要指标，我们提供更标准的记录代码，并要求排行榜参赛者披露图像尺寸，作为这些特性的代理。

**图 6：放大输入。**推理时间和内存迅速超过许多工业生产周期的实时要求，横轴为对数。所有大米类别测量均在 NVIDIA RTX 2080 Ti 上完成。

## 5 用于标准化、公平性能评估的服务器

如第 3 节所述，服务器保存 TEST_priv、TEST_priv,mix 的逐像素真值，主要目标是保证公平、标准的比较。两个测试集只公开图像，强调直到部署时才知道缺陷性质的无监督设置。私有真值防止过拟合与过度调参，保证评估完整性。实际中准确定位对信任系统合格/缺陷决策至关重要，因此着重分割性能。除 AU-PRO 外，我们鼓励研究稳健阈值获取方法，因为异常图最终决定物体分类。因此可将分割阈值随连续异常图一起提交。

评估 TEST_priv 或 TEST_priv,mix 并进入排行榜的步骤如下：

1. 用无异常训练数据训练模型。
2. 用无缺陷验证图像确定训练时长、二值分割阈值等超参数；可选用 TEST_pub 初步估计性能。
3. 在私有测试图像推理，按要求目录将异常图存为 TIFF：`/mvtec_ad_2/{object_name}/{private,private_mixed}/anomaly_images/test/{good,bad}/{image_name}.tiff`。
4. 执行本地结构检查；成功后自动压缩为 zip。所需代码可从 [数据集网站](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) 下载。
5. 上传压缩数据。
6. 接收性能结果。

服务器位于 [MVTec Benchmark](https://benchmark.mvtec.com/)，更多说明见补充材料或数据集网站。

## 6 结论

我们提出新无监督异常检测数据集 MVTec AD 2，以以往未覆盖的进阶场景扩展现有数据集，包括透明、重叠物体，暗场与背光，正常变化大，相对图像尺寸极小的缺陷，以及遍布整幅图像的异常。对七种先进方法的定位评估显示巨大改进空间：AU-PRO₀.₃₀ 低于 60%；收紧误报容忍度采用 AU-PRO₀.₀₅ 后，甚至低于约 30%。增大输入通常有帮助，但必须提升现有模型在时间和内存方面的扩展效率。首次提供含照明变化的非合成测试数据，支持研究真实偏移鲁棒性。评估服务器同时考虑无阈值与有阈值指标，有助于更公平、更标准的研究评价。

### 局限与未来工作

尽管已评估七种先进模型，我们计划在服务器报告更多方法。比本文实验更复杂的阈值估计或能改善有阈值与无阈值指标相关性；完全免除阈值选择的新方法也有很大创新空间。对于 TEST_priv 与 TEST_priv,mix 差异较大的方法，值得研究标准亮度变化等充分增强能否弥合差距，或必须用先进增强；同时应研究增强是否损害常规测试性能。大输入需要进一步研究切块、多尺度等高效策略。最后，建议建立公开基准服务器作为统一硬件环境衡量时间与内存，并可纳入仅 4 GB 内存的 [NVIDIA Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano) 或 int8 量化运行的 [Hailo-8 AI Accelerator](https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator)，以恰当反映常见工业约束。

## 版权声明

© 2025 IEEE。允许个人使用本材料。其他所有用途，无论现有或未来媒介，包括为广告或宣传重印/再版、创建新的合集、转售、向服务器或邮件列表再分发，或在其他作品中复用本文任何受版权保护的部分，均须获得 IEEE 许可。

## 参考文献

[1] H. Blum, P .-E. Sarlin, J. Nieto, R. Siegwart, and C. Cadena, “Fishyscapes: A benchmark for safe semantic segmentation in autonomous driving,” in IEEE International Conference on Computer Vision Workshops (ICCVW), 2019, pp. 2403–2412. 1

中文题名：Fishyscapes：面向自动驾驶安全语义分割的基准。

[2] D. Hendrycks, S. Basart, M. Mazeika, M. Mostajabi, J. Steinhardt, and D. Song, “A benchmark for anomaly segmentation,” arXiv preprint arXiv:1911.11132v1, 2019. 1

中文题名：异常分割基准。

[3] P . Seeb ¨ock, J. I. Orlando, T. Schlegl, S. M. Waldstein, H. Bogunovi´c, S. Klimscha, G. Langs, and U. Schmidt-Erfurth, “Exploiting epistemic uncertainty of anatomy segmentation for anomaly detection in retinal OCT,” IEEE Transactions on Medical Imaging, vol. 39, no. 1, pp. 87–98, 2020. 1

中文题名：利用解剖结构分割的认知不确定性检测视网膜 OCT 异常。

[4] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby et al. , “The multimodal brain tumor image segmentation benchmark (BRATS),” IEEE Transactions on Medical Imaging , vol. 34, no. 10, pp. 1993–2024, 2015. 1

中文题名：多模态脑肿瘤图像分割基准（BRATS）。

[5] T. S. Nazare, R. F. de Mello, and M. A. Ponti, “Are pre-trained CNNs good feature extractors for anomaly detection in surveillance videos?” arXiv preprint arXiv:1811.08495, 2018. 1

中文题名：预训练 CNN 是监控视频异常检测的良好特征提取器吗？。

[6] W.-X. Li, V . Mahadevan, and N. Vasconcelos, “Anomaly detection and localization in crowded scenes,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 1, pp. 18–32, 2013. 1

中文题名：拥挤场景中的异常检测与定位。

[7] P . Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “MVTec AD — A comprehensive real-world dataset for unsupervised anomaly detection,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9584–9592. 1, 2, 3, 4, 14

中文题名：MVTec AD：用于无监督异常检测的综合真实世界数据集。

[8] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer, “Spot-thedifference self-supervised pre-training for anomaly detection and segmentation,” in Computer Vision – ECCV 2022. Cham: Springer Nature Switzerland, 2022, pp. 392–408. 1, 2, 3, 4

中文题名：用于异常检测与分割的找不同自监督预训练。

[9] P . Bergmann, K. Batzner, M. Fauser, D. Sattlegger, and C. Steger, “The MVTec Anomaly Detection Dataset: A Comprehensive RealWorld Dataset for Unsupervised Anomaly Detection,” International Journal of Computer Vision , vol. 129, no. 4, pp. 1038–1059, 2021. 1, 8

中文题名：MVTec 异常检测数据集：用于无监督异常检测的综合真实世界数据集。

[10] ——, “Beyond dents and scratches: Logical constraints in unsupervised anomaly detection and localization,” International Journal of Computer Vision, vol. 130, no. 4, p. 947–969, 2022. 1, 3

中文题名：超越凹痕和划痕：无监督异常检测与定位中的逻辑约束。

[11] P . Bergmann, X. Jin, D. Sattlegger, and C. Steger, “The MVTec 3D-AD dataset for unsupervised 3D anomaly detection and localization,” in Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications — Volume 5: VISAPP . SciTePress, 2022, pp. 202–213. 1, 3

中文题名：用于无监督三维异常检测与定位的 MVTec 3D-AD 数据集。

[12] L. Bonfiglioli, M. Toschi, D. Silvestri, N. Fioraio, and D. De Gregorio, “The Eyecandies dataset for unsupervised multimodal anomaly detection and localization,” in Proceedings of the 16th Asian Conference on Computer Vision (ACCV 2022), 2022. 1, 3

中文题名：用于无监督多模态异常检测与定位的 Eyecandies 数据集。

[13] H. Li, J. Hu, B. Li, H. Chen, Y. Zheng, and C. Shen, “Target before shooting: Accurate anomaly detection and localization under one millisecond via cascade patch retrieval,”IEEE Transactions on Image Processing, vol. 33, pp. 5606–5621, 2024. 3

中文题名：射击前先瞄准：通过级联图块检索在一毫秒内准确检测与定位异常。

[14] K. Batzner, L. Heckler, and R. K ¨onig, “EfficientAD: Accurate visual anomaly detection at millisecond-level latencies,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024, pp. 128–138. 3, 7, 8, 9

中文题名：EfficientAD：毫秒级延迟下的准确视觉异常检测。

[15] H. Zhang, Z. Wang, Z. Wu, and Y.-G. Jiang, “DiffusionAD: Normguided one-step denoising diffusion for anomaly detection,” arXiv preprint arXiv:2303.08730v3, 2023. 3

中文题名：DiffusionAD：用于异常检测的正常性引导单步去噪扩散。

[16] C. Wang, W. Zhu, B.-B. Gao, Z. Gan, J. Zhang, Z. Gu, S. Qian, M. Chen, and L. Ma, “Real-IAD: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection,” arXiv preprint arXiv:2403.12580, 2024. 3

中文题名：Real-IAD：用于多功能工业异常检测基准的真实世界多视角数据集。

[17] H. Deng and X. Li, “Anomaly detection via reverse distillation from one-class embedding,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 9737–9746. 3, 7

中文题名：通过单类嵌入的逆向蒸馏检测异常。

[18] Q. Zhou, W. Li, L. Jiang, G. Wang, G. Zhou, S. Zhang, and H. Zhao, “PAD: A dataset and benchmark for pose-agnostic anomaly detection,” in Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. 3

中文题名：PAD：姿态无关异常检测的数据集与基准。

[19] P . Bergmann, S. L¨owe, M. Fauser, D. Sattlegger, and C. Steger, “Improving unsupervised defect segmentation by applying structural similarity to autoencoders,” in Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications — Volume 5: VISAPP . SciTePress, 2019, pp. 372–380. 3

中文题名：通过对自编码器应用结构相似性改善无监督缺陷分割。

[20] V . Zavrtanik, M. Kristan, and D. Sko ˇcaj, “DRAEM — A discriminatively trained reconstruction embedding for surface anomaly detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, pp. 8330–8339. 3

中文题名：DRAEM：用于表面异常检测的判别式训练重建嵌入。

[21] X. Zhang, N. Li, J. Li, T. Dai, Y. Jiang, and S.-T. Xia, “Unsupervised surface anomaly detection with diffusion probabilistic model,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 6782–6791. 3

中文题名：基于扩散概率模型的无监督表面异常检测。

[22] K. Roth, L. Pemula, J. Zepeda, B. Sch ¨olkopf, T. Brox, and P . V . Gehler, “Towards total recall in industrial anomaly detection,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 14 298–14 308. 3, 7

中文题名：迈向工业异常检测的完全召回。

[23] N. Cohen and Y. Hoshen, “Sub-image anomaly detection with deep pyramid correspondences,” arXiv preprint arXiv:2005.02357v1, 2020. 3

中文题名：采用深度金字塔对应的子图像异常检测。

[24] T. Defard, A. Setkov, A. Loesch, and R. Audigier, “PaDiM: A patch distribution modeling framework for anomaly detection and localization,” in Pattern Recognition. ICPR International Workshops and Challenges 2021, Proceedings, Part IV , ser. Lecture Notes in Computer Science, vol. 12664. Springer, 2020, pp. 475–489. 3

中文题名：PaDiM：用于异常检测与定位的图块分布建模框架。

[25] D. Gudovskiy, S. Ishizaka, and K. Kozuka, “CFLOW-AD: Realtime unsupervised anomaly detection with localization via conditional normalizing flows,” in 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 1819–1828. 3

中文题名：CFLOW-AD：通过条件归一化流实时无监督异常检测与定位。

[26] M. Rudolph, T. Wehrbein, B. Rosenhahn, and B. Wandt, “Fully convolutional cross-scale-flows for image-based defect detection,” in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2022, pp. 1829–1838. 3

中文题名：用于图像缺陷检测的全卷积跨尺度流。

[27] J. Yu, Y. Zheng, X. Wang, W. Li, Y. Wu, R. Zhao, and L. Wu, “FastFlow: Unsupervised anomaly detection and localization via 2D normalizing flows,” arXiv preprint arXiv:2111.07677v2, 2021. 3

中文题名：FastFlow：通过二维归一化流实现无监督异常检测与定位。

[28] Y. Zhou, X. Xu, J. Song, F. Shen, and H. T. Shen, “MSFlow: Multiscale flow-based framework for unsupervised anomaly detection,” IEEE Transactions on Neural Networks and Learning Systems , pp. 1– 14, 2024. 3, 7

中文题名：MSFlow：用于无监督异常检测的多尺度流框架。

[29] P . Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “Uninformed students: Student–teacher anomaly detection with discriminative latent embeddings,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 4182–4191. 3

中文题名：无知的学生：采用判别潜在嵌入的学生—教师异常检测。

[30] M. Rudolph, T. Wehrbein, B. Rosenhahn, and B. Wandt, “Asymmetric student-teacher networks for industrial anomaly detection,” in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2023, pp. 2591–2601. 3

中文题名：用于工业异常检测的非对称学生—教师网络。

[31] T. D. Tien, A. T. Nguyen, N. H. Tran, T. D. Huy, S. T. Duong, C. D. T. Nguyen, and S. Q. H. Truong, “Revisiting reverse distillation for anomaly detection,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 24 511–24 520. 3, 7

中文题名：重新审视用于异常检测的逆向蒸馏。

[32] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer, “Winclip: Zero-/few-shot anomaly classification and segmentation,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 19 606–19 616. 3

中文题名：WinCLIP：零/少样本异常分类与分割。

[33] Z. Gu, B. Zhu, G. Zhu, Y. Chen, M. Tang, and J. Wang, “AnomalyGPT: Detecting industrial anomalies using large vision-language models,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 3, pp. 1932–1940, 2024. 3

中文题名：AnomalyGPT：利用大型视觉语言模型检测工业异常。

[34] H. M. Schl ¨uter, J. Tan, B. Hou, and B. Kainz, “Natural synthetic anomalies for self-supervised anomaly detection and localization,” in Computer Vision – ECCV 2022 , S. Avidan, G. Brostow, M. Ciss ´e, G. M. Farinella, and T. Hassner, Eds. Cham: Springer Nature Switzerland, 2022, pp. 474–489. 4

中文题名：用于自监督异常检测与定位的自然合成异常。

[35] V . Zavrtanik, M. Kristan, and D. Sko ˇcaj, “DSR — A dual subspace re-projection network for surface anomaly detection,” in Computer Vision – ECCV 2022 , S. Avidan, G. Brostow, M. Ciss ´e, G. M. Farinella, and T. Hassner, Eds. Cham: Springer Nature Switzerland, 2022, pp. 539–554. 4, 7, 8

中文题名：DSR：用于表面异常检测的双子空间重投影网络。

[36] Z. Liu, Y. Zhou, Y. Xu, and Z. Wang, “Simplenet: A simple network for image anomaly detection and localization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 20 402–20 411. 4, 7, 8

中文题名：SimpleNet：用于图像异常检测与定位的简单网络。

[37] C. Steger, M. Ulrich, and C. Wiedemann, Machine Vision Algorithms and Applications, 2nd ed. Weinheim: Wiley-VCH, 2018. 4

中文题名：机器视觉算法与应用（第 2 版）。

## 作者简介

**Lars Heckler-Kram** 在慕尼黑工业大学（TUM）学习机电一体化与机器人，2022 年获硕士学位。目前在 MVTec Software GmbH 和 TUM 计算、信息与技术学院攻读博士，研究兴趣聚焦工业视觉异常检测。

**Jan-Hendrik Neudeck** 于 2020 年在 TUM 获计算机科学硕士，专攻计算机视觉与深度学习，此后任 MVTec 研究工程师。

**Ulla Scheler** 在卡尔斯鲁厄理工学院和达姆施塔特工业大学学习计算机科学，获计算机科学硕士后于 2023 年加入 MVTec 任研究工程师。

**Rebecca König** 在 TUM 学习数学，2017 年获科学与工程数学硕士，此后任 MVTec 研究工程师，撰写或合作发表多篇计算机视觉深度学习论文。

**Carsten Steger** 在 TUM 学习计算机科学，1998 年获博士。1996 年共同创立 MVTec 并领导研究部门，在计算机视觉与机器视觉领域撰写或合作发表超过 100 篇科学出版物，包括多部机器视觉教材。2011 年获任 TUM 计算机视觉荣誉教授，2013—2021 年任德国模式识别协会（DAGM）技术委员会成员，2018—2021 年担任发言人。

## 补充材料 A：数据集与评估服务器访问

[数据集页面](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) 提供所有图像和 TEST_pub 真值，可进行小规模本地测试。也可下载推理时间与内存测量片段、通用评估代码及 PyTorch 数据加载器，便于集成。服务器位于 [benchmark.mvtec.com](https://benchmark.mvtec.com/)。

数据以 CC BY-NC-SA 4.0 发布，托管于 [MVTec 数据集网站](https://www.mvtec.com/company/research/datasets)。该服务器已托管早期 MVTec 异常检测数据集，至今仍完整可用，证明是下载可用性和长期保存的稳定方案。结构遵循 MVTec AD [7] 等已有约定与格式。因为只有图像，读取无需特殊措施。

## 补充材料 B：额外定量结果

### B.1 与阈值无关的结果

如正文所示，过高积分上限的 AU-PRO 对小缺陷数据集意义有限。但为完整报告，仍提供当前视觉异常检测标准指标 AU-PRO₀.₃₀。表 9 给出 256×256 输入的两个私有测试集结果。

正文图 6 展示大米类别增大输入时性能与计算成本的权衡。表 10、11 给出所有物体较大尺寸下 AU-PRO₀.₀₅。例如 PatchCore 从 256×256 的 TEST_priv 28.8%（正文表 7），提高到原始宽高各一半的 62.3%。但直接增大输入很快超过系统内存与时间约束；即使最大测试尺寸，仍有显著提升空间。

**表 9：256×256 输入下两个私有测试集的异常分割 AU-PRO₀.₃₀（%）。**

| 物体 | PatchCore | RD | RD++ | EfficientAD | MSFlow | SimpleNet | DSR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 罐 | 21.6 / 18.1 | 42.9 / 30.1 | 44.7 / 29.4 | 38.1 / 24.4 | 33.4 / 15.8 | 36.3 / 20.7 | 50.0 / 26.3 |
| 织物 | 34.6 / 35.3 | 22.3 / 25.6 | 26.2 / 29.5 | 46.8 / 41.3 | 38.2 / 38.4 | 25.7 / 26.0 | 23.7 / 25.6 |
| 水果果冻 | 74.0 / 74.0 | 78.9 / 79.0 | 80.4 / 80.3 | 79.5 / 78.0 | 78.8 / 73.5 | 71.8 / 70.3 | 70.1 / 69.8 |
| 大米 | 50.9 / 43.1 | 30.2 / 31.3 | 34.2 / 32.9 | 52.1 / 19.1 | 48.5 / 38.8 | 25.5 / 24.3 | 28.4 / 28.0 |
| 金属薄板 | 39.8 / 34.2 | 26.6 / 25.1 | 27.6 / 25.1 | 36.1 / 24.5 | 35.8 / 29.0 | 32.3 / 22.9 | 46.9 / 45.2 |
| 小瓶 | 90.5 / 89.2 | 91.3 / 90.0 | 91.5 / 87.3 | 88.7 / 85.2 | 79.1 / 39.7 | 82.5 / 67.5 | 88.1 / 85.2 |
| 膨胀管 | 37.4 / 34.8 | 51.0 / 41.9 | 51.0 / 40.5 | 51.5 / 17.9 | 35.5 / 9.7 | 27.4 / 12.9 | 23.6 / 26.8 |
| 核桃 | 81.7 / 77.4 | 81.2 / 79.6 | 78.9 / 78.0 | 76.5 / 62.3 | 72.1 / 51.8 | 69.4 / 56.1 | 60.8 / 54.5 |
| 平均 | 53.8 / 50.8 | 53.0 / 50.3 | 54.3 / 50.4 | 58.7 / 44.1 | 52.7 / 37.1 | 46.4 / 37.6 | 49.0 / 45.2 |

**表 10：512×512 输入下两个私有测试集的异常分割 AU-PRO₀.₀₅（%）。**

| 物体 | PatchCore | RD | RD++ | EfficientAD | MSFlow | SimpleNet | DSR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 罐 | 8.4 / 7.0 | 10.5 / 10.7 | 13.1 / 12.6 | 14.2 / 2.4 | 24.2 / 1.3 | 19.0 / 2.3 | 19.8 / 3.3 |
| 织物 | 23.7 / 23.5 | 36.8 / 39.6 | 41.2 / 43.0 | 44.5 / 28.3 | 31.7 / 28.2 | 22.5 / 23.6 | 30.3 / 28.7 |
| 水果果冻 | 64.9 / 64.8 | 50.6 / 50.4 | 58.4 / 58.4 | 55.4 / 53.5 | 63.7 / 52.3 | 54.9 / 51.7 | 45.8 / 42.9 |
| 大米 | 33.6 / 15.5 | 13.1 / 12.0 | 16.0 / 14.4 | 29.9 / 3.1 | 27.2 / 19.0 | 10.1 / 6.1 | 20.5 / 15.9 |
| 金属薄板 | 27.3 / 24.1 | 20.0 / 20.1 | 24.3 / 22.0 | 21.7 / 12.4 | 23.4 / 9.5 | 19.0 / 14.1 | 10.1 / 8.4 |
| 小瓶 | 72.6 / 68.4 | 67.6 / 64.7 | 71.9 / 69.1 | 60.4 / 55.3 | 53.2 / 4.6 | 56.7 / 32.5 | 23.2 / 14.1 |
| 膨胀管 | 34.0 / 21.3 | 43.9 / 21.3 | 35.5 / 19.2 | 33.2 / 2.3 | 31.4 / 0.3 | 24.8 / 6.9 | 6.3 / 7.3 |
| 核桃 | 70.5 / 61.4 | 55.8 / 53.5 | 55.5 / 52.4 | 60.8 / 45.4 | 59.0 / 35.7 | 49.3 / 32.6 | 44.1 / 30.2 |
| 平均 | 41.9 / 35.8 | 37.3 / 34.0 | 39.5 / 36.4 | 40.0 / 25.3 | 39.2 / 18.9 | 32.0 / 21.2 | 25.0 / 18.8 |

**表 11：原图宽高各一半输入下两个私有测试集的 AU-PRO₀.₀₅（%）。因训练时长，DSR 仅评估大米。**

| 物体 | PatchCore | RD | RD++ | EfficientAD | MSFlow | SimpleNet | DSR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 罐 | 12.8 / 10.2 | 15.0 / 13.7 | 18.2 / 15.8 | 20.8 / 3.3 | 26.6 / 2.5 | 21.9 / 3.2 | - |
| 织物 | 69.0 / 59.1 | 81.1 / 77.9 | 82.1 / 77.8 | 72.5 / 45.4 | 82.9 / 60.6 | 66.0 / 55.6 | - |
| 水果果冻 | 71.5 / 70.8 | 54.9 / 54.7 | 63.3 / 62.8 | 54.5 / 52.9 | 72.8 / 67.6 | 64.9 / 63.0 | - |
| 大米 | 47.8 / 28.9 | 27.4 / 26.2 | 31.2 / 28.5 | 37.7 / 5.4 | 47.1 / 32.7 | 21.9 / 12.3 | 16.0 / 13.5 |
| 金属薄板 | 72.4 / 54.0 | 54.2 / 51.4 | 57.9 / 46.8 | 52.0 / 37.2 | 42.2 / 11.9 | 46.8 / 34.1 | - |
| 小瓶 | 75.8 / 72.2 | 69.9 / 67.7 | 73.2 / 58.8 | 61.4 / 56.4 | 56.1 / 1.7 | 56.1 / 38.8 | - |
| 膨胀管 | 68.4 / 53.7 | 55.1 / 39.5 | 52.8 / 24.7 | 42.2 / 12.4 | 65.1 / 1.1 | 25.6 / 10.5 | - |
| 核桃 | 80.4 / 71.6 | 71.7 / 64.6 | 64.1 / 59.1 | 65.0 / 54.9 | 72.6 / 53.8 | 62.7 / 55.7 | - |
| 平均 | 62.3 / 52.6 | 53.7 / 49.5 | 55.3 / 46.8 | 50.8 / 33.5 | 58.2 / 29.0 | 45.7 / 34.1 | - |

### B.2 依赖阈值的结果

图像分类直接基于方法输出异常图。阈值仍为正常验证图像均值加三倍标准差；任一像素超阈值，整图即判为不合格。表 12 给出 256×256 输入的分类 F₁。有趣的是，PatchCore、MSFlow 的无阈值指标虽强，分类却落后。PatchCore 的差表现可归因于归一化：默认在整个输入数据集上将异常图归一化至 [0,1]，以组合集成模型输出。正常与异常图混合推理时有效；但估计阈值仅输入无缺陷数据，造成基线阈值过高，F₁ 显著低于其他五种最佳方法。

**表 12：256×256 输入下两个私有测试集的分类 F₁（%）。阈值只由无缺陷验证图像确定；任一像素异常即整图异常。**

| 物体 | PatchCore | RD | RD++ | EfficientAD | MSFlow | SimpleNet | DSR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 罐 | 41.4 / 57.8 | 50.9 / 66.5 | 51.1 / 65.9 | 57.5 / 67.1 | 15.0 / 64.4 | 48.2 / 67.0 | 70.8 / 70.8 |
| 织物 | 1.1 / 1.1 | 74.0 / 74.4 | 78.0 / 76.9 | 73.4 / 73.3 | 55.8 / 68.9 | 65.4 / 72.9 | 73.1 / 73.1 |
| 水果果冻 | 9.3 / 8.3 | 84.1 / 84.1 | 85.8 / 85.2 | 84.2 / 84.2 | 64.1 / 78.6 | 84.6 / 84.3 | 83.8 / 83.8 |
| 大米 | 2.2 / 2.2 | 78.8 / 78.8 | 78.0 / 78.5 | 79.2 / 79.2 | 53.5 / 77.5 | 69.8 / 76.3 | 79.0 / 79.0 |
| 金属薄板 | 1.9 / 1.9 | 43.4 / 44.4 | 35.8 / 43.2 | 86.4 / 85.1 | 65.1 / 74.9 | 82.0 / 83.4 | 85.5 / 85.5 |
| 小瓶 | 4.0 / 4.0 | 83.3 / 83.5 | 83.7 / 83.5 | 83.5 / 83.5 | 75.6 / 81.8 | 84.8 / 83.7 | 83.5 / 83.5 |
| 膨胀管 | 1.4 / 0.0 | 73.9 / 73.9 | 72.2 / 73.4 | 74.0 / 73.6 | 4.2 / 53.4 | 69.1 / 73.4 | 74.1 / 73.9 |
| 核桃 | 2.9 / 4.3 | 78.2 / 77.3 | 79.5 / 77.2 | 76.1 / 76.1 | 75.6 / 76.6 | 82.6 / 76.3 | 74.4 / 74.4 |
| 平均 | 8.0 / 9.9 | 70.8 / 72.9 | 70.5 / 73.0 | 76.8 / 77.8 | 51.1 / 72.0 | 73.3 / 77.1 | 78.0 / 78.0 |

## 补充材料 C：缺陷示例

图 7—14 给出每种物体缺陷示例。

**图 7：罐的 TEST_pub 示例。**罐随机旋转，并用闪光薄膜覆盖条码，增加正常允许变化。所示缺陷为印刷错误、颜色变化与划痕；其他缺陷包括罐标签缺失。

**图 8：膨胀管的 TEST_pub 示例。**所示缺陷为切口、划痕、污染和破损实例；其他缺陷包括弯曲、部件缺失、尺寸错误。数量与位置变化，重叠造成遮挡。缺陷遍布图像，包括边缘附近。

**图 9：核桃的 TEST_pub 示例。**缺陷为裂纹、孔洞及异物污染，此处为榛子。与膨胀管类似，位置与数量随机，核桃颜色、形状、大小还有很大自然变化。缺陷遍布图像及边缘附近。

**图 10：织物的 TEST_pub 示例。**图案及方向可变。所示缺陷为切口、孔洞、多余布片；其他包括变色与多余纱线。缺陷遍布图像及边缘附近。

**图 11：水果果冻的 TEST_pub 示例。**所示缺陷为金属薄片、蔬菜碎片、毛发等异物；其他包括更多污染类型或玻璃划痕。为完整起见，也标注装置底部的缺陷反射。不同颜色水果、果冻折射特性及成分随机组合使正常数据非常多样。

**图 12：大米的 TEST_pub 示例。**所示缺陷为塑料薄膜和在形状、尺寸、颜色上刻意选择与米相似的异物；其他包括毛发。缺陷遍布图像及边缘附近。

**图 13：金属薄板的 TEST_pub 示例。**所示缺陷为划痕、切口、孔洞；其他包括覆盖材料的异物。正常数据变化大，镜面高光与阴影进一步增强变化。缺陷遍布图像及边缘附近。

**图 14：小瓶的 TEST_pub 示例。**所示缺陷为异物和塑料薄膜；其他包括二维码多余或缺失、液体过多或过少、毛发、瓶盖打开或缺失。二维码、气泡、瓶旋转和液体畸变造成正常数据变化大。

## 补充材料 D：定性结果

图 15—22 给出正常与异常示例及 256×256 输入下五种最佳方法的异常输出图。为方便可视化，色图上下限固定为各方法在 TEST_pub 所有输出中的最小与最大像素异常分数。

**图 15：**罐的无异常、有异常示例及五种最佳评估方法的输出图。异常图像真值宜放大查看。

**图 16：**织物的无异常、有异常示例及五种最佳评估方法的输出图。异常图像真值宜放大查看。

**图 17：**水果果冻的无异常、有异常示例及五种最佳评估方法的输出图。

**图 18：**大米的无异常、有异常示例及五种最佳评估方法的输出图。

**图 19：**金属薄板的无异常、有异常示例及五种最佳评估方法的输出图。

**图 20：**小瓶的无异常、有异常示例及五种最佳评估方法的输出图。

**图 21：**膨胀管的无异常、有异常示例及五种最佳评估方法的输出图。

**图 22：**核桃的无异常、有异常示例及五种最佳评估方法的输出图。

图 15—22 的列从左至右均为：输入、PatchCore、RD、RD++、EfficientAD、MSFlow、真值；行分别为无异常和有异常。原文图 17 中“五种最佳”重复一次，本译文消除该排版重复。
