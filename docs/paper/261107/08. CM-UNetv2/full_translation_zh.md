# CM-UNetv2：面向精确PCB缺陷检测与边界恢复的增强语义分割模型

**原文：**[CM-UNetv2: An Enhanced Semantic Segmentation Model for Precise PCB Defect Detection and Boundary Restoration](<08-CM-UNetv2-PCB-Defect-Segmentation-2025.pdf>)。Sensors，2025，25，4919。[DOI原文链接](https://doi.org/10.3390/s25164919)。翻译日期：2026-11-11。

**作者：**Qiyang Guo、Yajun Chen（通信作者）、Yirui Zhu、Dongle Chen。

**单位：**中国西安理工大学信息科学系，西安710048。邮箱：2230821075@stu.xaut.edu.cn（Q.G.）；2240820017@stu.xaut.edu.cn（Y.Z.）；2240821096@stu.xaut.edu.cn（D.C.）。通信邮箱：chenyj@xaut.edu.cn；电话：+86-29-6112-3868。

**出版信息：**收稿2025年7月9日；修订2025年8月6日；接受2025年8月8日；发表2025年8月9日。版权©2025作者；被许可方MDPI，瑞士巴塞尔。本文按[知识共享署名CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)开放获取。

**翻译说明：**对应本目录25页PDF，翻译摘要、正文第1—6节、公式(1)—(11)、图1—11图注、表1—10全部行列数值、作者贡献及声明、参考文献[1]—[48]题名和出版者声明。跨页段落合并，删除运行页眉页码，图形不重绘。保留原文内部不一致之处并提供译注，不据常识替换原文；性能结论均指原文实验范围与发表时的比较。PDF内无附录。

## 摘要

PCB在电子制造中发挥关键作用，准确检测缺陷对于确保产品质量和可靠性至关重要。然而，PCB缺陷通常很小、形状不规则，并嵌入复杂纹理，传统方法难以检测。本文提出语义分割网络CM-UNetv2，通过包含四项关键创新的三个架构模块解决上述问题。首先，在编码器中加入并行图块感知注意力（PPA）模块，通过结合局部、全局和串行卷积的多分支注意力增强多尺度特征表示。其次，提出双流跳跃引导（DSSG）模块，以两条独立跳跃路径解耦语义细化与空间信息保留，实现更精细的细节保存。第三，设计频域引导上下文Mamba（FGCMamba）解码器，集成两种新机制：空间引导交叉注意力（SGCA）增强空间和语义特征对齐；频域自注意力求解器（FSAS）在频域高效计算全局注意力，改善边界恢复并降低计算开销。在MeiweiPCB和KWSD2数据集上的实验表明，CM-UNetv2在小目标检测、边界精度和整体分割鲁棒性方面取得先进性能。

**关键词：**PCB；语义分割；多尺度特征提取；神经网络。

## 1 引言

随着电子产品日益普及，印制电路板（PCB）制造已成为现代工业生产的基础环节[1]。作为电子设备核心部件，PCB必须具有良好的稳定性和抗干扰能力，同时支持高集成度、紧凑设计和高速数据传输。然而，即使微小缺陷也可能造成严重后果：产品质量下降，返工或报废成本增加，上市延迟，以及短路、元件失效、过热等潜在安全风险。因此，PCB缺陷检测已得到广泛研究，研究者提出多种算法提高准确性和效率。

质量控制是制造过程的重要方面。为满足不断增长的市场需求，必须严格管理质量并提高生产效率。因此，表面缺陷检测对提升PCB制造质量至关重要[2]。传统方法[3,4]主要依靠人工目视，效率低、漏检率高。计算机视觉表面缺陷检测[5,6]则因其理论意义和实际价值而备受关注。在PCB表面缺陷自动检测中，准确分割缺陷区域尤其重要，因为其生成的二值图便于计算机系统识别缺陷。

消费电子普及显著增加了高质量PCB批量生产需求，但大规模生产中的质量保证面临重大挑战。现有PCB图像检测存在边缘分割不准、大尺度目标分割区域缺失等问题。缺陷可能导致电路失效，因此面向缺陷定位和分类的检测非常重要。开发高效检测方法既可提高制造质量和效率，也推动现代工业生产。

为此，我们提出CM-UNetv2，增强细粒度结构提取及复杂场景鲁棒性。架构包含用于多尺度提取的PPA编码器、处理两条并行特征流的DSSG（一路经MSAA细化，另一路直接送入解码器保留原空间结构），以及结合FSAS和SGCA、改善边界和纹理重建的FGCMamba解码器。主要贡献如下：

1. 在编码阶段引入PPA，通过多分支注意力捕获局部和全局上下文，产生用于渐进编码、语义增强和空间跳跃连接的三路输出。
2. 提出DSSG以平衡跳跃连接特征：一路经MSAA处理，另一路将原始编码器特征直接传给解码器。
3. 引入FGCMamba块，将DSSG与解码器结合，处理两条路径，并利用SGCA结合细化语义和空间细节，实现精确解码融合。
4. 在FGCMamba解码器中加入FSAS，以统一注意力替代传统空间和通道注意力，在降低复杂度的同时增强细粒度表示，平衡建模效率与准确性。

## 2 相关工作

### 2.1 PCB缺陷检测

传统计算机视觉在表面检测中取得显著成功[1,7,8]。例如，[9]提出半导体晶圆自动缺陷簇识别系统；[10]开发两阶段混合一对多支持向量机（SVM），用于滚动轴承自动故障诊断；[2]将新型多重分形特征与支持向量数据描述结合，实现织物自动缺陷检测。

随着深度学习广泛用于视觉任务，它也越来越多用于表面检测。[11]提出CNN自动缺陷检测策略，提高工件表面异常识别精度；[12]使用深度CNN识别木材特征并自动分类激光扫描图像中的缺陷；[13]结合CNN与注意力，精确分割复杂背景煤炭图像的异物；[6]将Faster R-CNN[9]与稠密尺度不变特征结合检测铁路扣件缺陷；[14]进一步验证深度学习在工业视觉检测中的有效性。

深度学习在PCB表面检测中也得到广泛应用。Zhang等[15]提出基于深度特征学习的裸板检测，用预训练CNN提取图像特征，再以滑窗分类，显著改善划痕、断路等缺陷检测。Liu等[16]开发轻量YOLO-pdd，将YOLOv5骨干与多尺度模块结合，实现序列图像高效检测。

Li等[17]提出基于FPN的改进PCB检测模型，显著提升多尺度识别。[18]使用去噪卷积自编码器重建缺陷图像特征并插值恢复，最终利用差分图定位缺陷，整体准确率97.5%。Gustilo等[19]设计适合移动设备的轻量CNN识别网络，用迁移学习补偿训练数据不足，检测率达85%。

尽管有效，现有PCB方法多侧重分类或粗定位，缺少缺陷形状、边界细节，限制了精细分析和准确修复中的用途。语义分割则提供像素定位，可精确提取轮廓、测量面积，更适于高精度工业应用。

### 2.2 基于深度学习的PCB分割方法

表面缺陷分割在目标检测识别中非常重要。传统方法通常结合人工特征与分类器[20,21]，而端到端深度学习近期显著推动语义分割。广泛使用的全卷积网络FCN[22]以卷积替换传统CNN全连接层，接受任意尺寸输入。

带跳跃连接的经典编码器—解码器U-Net[23]在医学分割中成功，并推广至工业。DeepLabV3+系列[24]用空洞卷积在保持分辨率时扩大感受野，改善上下文信息。PSANet[25]通过金字塔池化聚合多尺度信息。注意力也用于增强表示，例如[26]设计通道和空间注意力增强重要区域响应，[27]以情境注意力建模多尺度依赖。虽然公开基准表现强，但固定感受野限制全局建模。

近年来PCB语义分割进展明显。区别于分类或目标检测，语义分割提供像素级定位，适合必须准确判定缺陷形状、边界和位置的工业制造。

不少研究提出定制U-Net。例如Li和Liu提出VGG16编码器改进U-Net，加入动态混合注意力DHAM及轻量特征融合RGSM，仅2200万参数，mIoU81.74%、mPA87.33%，推理30 FPS，适合实时应用[18]。另一研究用去噪自编码器重建输入，并以差分图定位，实现有效无监督分割[28]。

为增强全局感知，研究者引入Transformer。Liu等结合Swin Transformer骨干和Cascade Mask R-CNN解码器，利用Stable Diffusion生成缺陷变体，显著改善小目标检测分割[29]。PCB-DDTR（细节与依赖Transformer）通过局部细节和全局上下文双分支提高复杂缺陷分割[30]。CA-TransSeg引入变化感知孪生Transformer，融合全局对比与变化图，检测微小表面变化，适合检测前后比较[31]。轻量化设计也成为研究重点。SEPDNet（简单有效的PCB缺陷网络）使用更少参数取得与更深模型相当的性能，在高分辨率图像上保持良好的速度与精度平衡[32]。

然而，微小缺陷密集分布、纹理模糊及不规则故障模式，尤其叠加丝印痕迹或氧化等工业噪声时，对现有分割模型构成重大挑战。这类场景需要更好地保存空间细节，并提供更自适应的语义推理。因此，有必要回顾语义分割新进展，寻找更适用方法。

### 2.3 语义分割方法

2010年代后期，CNN成为语义分割主流[33]。Ronneberger等2015年提出医学U-Net，编码器通过最大池化下采样，随后在解码器与语义特征结合，准确重建边界[23]。2018年Zhou等提出DLinkNet，以空洞卷积提取遥感道路并改善空间连通[34]。同年Chen等提出DeepLabv3+，通过多空洞率的ASPP模块提取不同尺度特征[24]。总体上，传统CNN主要依靠人工设计卷积核整合高低层特征[35]。

为减少对密集像素标注的依赖，出现多种掩码建模策略。掩码监督学习[36]随机遮蔽训练输入图块，鼓励模型从不完整线索推断语义，提高泛化。掩码生成蒸馏MGD[37]采用师生框架，学生从部分遮蔽输入重建教师完整特征，改善稠密预测特征质量。掩码协同对比MCC[38]结合掩码建模与对比学习，在遮蔽局部特征和全局语义预测间构造正样本对，改善弱监督语义一致性。这些努力减少标注成本、提高标签效率。但多数方法仍敏感于掩码模式，往往假定可见区域有较强空间相关；遮蔽区域语义复杂或面积大时可能退化。

Transformer因全局关系捕捉能力而流行。Chen等2021年提出TransUNet，引入Vision Transformer，以自注意力学习特征内部相关并给像素分配自适应权重[39]。SegFormer[40]和SETR进一步简化结构并保留层级注意力建模。Swin-Unet把Swin块引入U-Net，以窗口自注意力保留层级空间表示[41]。但Transformer尤其在高分辨率下计算和显存成本很高；缺乏局部归纳偏置也限制小型、不规则物体的细结构建模。

最初Mamba通过选择性序列扫描处理一维序列，尤其是NLP[42]。Xu等全面综述视觉Mamba，指出其在空间和多尺度结构建模中的潜力与局限[43]。为适应二维图像，Liu等2024年提出二维选择扫描2D-SSM[44]：展平特征、沿方向扫描以捕获长依赖，再恢复二维结构。同年Zhao等提出遥感分割及变化检测RS-Mamba，利用全局建模提升效率和质量[45]；Liu等提出结合ResNet编码器和CSMamba的CM-UNet，增强全局局部融合[46]。这些方法虽扩展感受野，但对不规则、多尺度对象仍有困难，因此应考虑CNN与Mamba混合方案。

## 3 方法

图1的CM-UNetv2含三个部分：PPA编码器、DSSG和FGCMamba解码器。编码器通过并行局部及全局分支扩展通道，结合注意力细化多尺度特征。每个分辨率上PPA生成X和X_skip。在DSSG中，保留原始结构的X_skip直接送解码器，X则经MSAA多尺度融合和自适应细化增强语义。两流均输入FGCMamba，经SGCA和FSAS结合，提高分割精度与细节重建。此外，编码器和解码器之间设置轻量FGCMamba，仅保留频域自注意力以简化机制、提高效率。

**图1：**所提出的CM-UNetv2网络结构。

### 3.1 并行图块感知注意力模块

ResNet等传统CNN反复下采样扩大感受野，但易丢失细结构。因此在编码器引入PPA，通过并行多分支和集成注意力增强多尺度表示。

核心是多分支提取。图2给定输入$F\in\mathbb{R}^{H'\times W'\times C'}$，逐点卷积产生调整特征$F'\in\mathbb{R}^{H'\times W'\times C'}$。局部、全局、串行卷积三分支分别提取同尺寸的$F_{local},F_{global},F_{conv}$，逐元素求和得到融合表示$\widetilde F$。

**图2：**模块由多分支融合与顺序注意力细化两部分组成。融合阶段结合图块大小p=2、p=4的局部及全局图块感知分支，以及串行卷积分支，增强多尺度表示；聚合特征再经通道和空间注意力细化。输出分为渐进编码、语义细化和空间跳跃连接三路。

局部和全局分支通过调整p并逐块处理F'增强感受野多样性。具体将F'划为不重叠p×p图块，形成形状$(p\times p,H'/p,W'/p,C)$张量。通道平均池化将其降为$(p\times p,H'/p,W'/p)$，再经前馈网络FFN[47]及激活计算空间注意力权重。

为加强特征选择，采用基于余弦相似度的token重加权。令$d=H'W'/(p\times p)$，token记为$(t_i)_{i=1}^{C'}$，$t_i\in\mathbb{R}^{d}$。按其与任务嵌入$\xi\in\mathbb{R}^{C'}$的相似度重加权：

$$\widehat t_i=P\cdot\operatorname{sim}(t_i,\xi)\cdot t_i.\tag{1}$$

其中$P\in\mathbb{R}^{C'\times C'}$为可学习投影，sim为归一化余弦相似度。

重加权后对每个token做通道选择，重塑并插值回原空间维度，得到局部和全局特征，形状均为$\mathbb{R}^{H'\times W'\times C'}$。卷积分支串联三个3×3卷积，将输出$F_{conv1},F_{conv2},F_{conv3}$相加形成F_conv。

融合特征$\widetilde F$上采样为$\mathbb{R}^{H\times W\times C'}$，再施加顺序注意力。通道图$M_c\in\mathbb{R}^{1\times1\times C'}$和空间图$M_s\in\mathbb{R}^{H'\times W'\times1}$作用为：

$$F_c=M_c(\widetilde F)\otimes\widetilde F,\tag{2}$$
$$F_s=M_s(F_c)\otimes F_c,\tag{3}$$
$$F''=\delta\bigl(\beta(\operatorname{dropout}(F_s))\bigr).\tag{4}$$

⊗表示逐元素乘法，δ为ReLU，β为批归一化，$F''\in\mathbb{R}^{H\times W\times C'}$为PPA最终输出。

实现中输出支持三路：一路传下一PPA维持渐进编码；一路进入MSAA作语义细化；一路作为保留空间细节的X_skip供解码器整合。这样既保持编码层连续，又支持DSSG双流引导。

### 3.2 双流跳跃引导

CM-UNet使用多尺度注意力聚合MSAA，在每个解码阶段动态融合编码特征，为跳跃连接注入丰富全局语义，增强复杂场景解释。它聚合多尺度特征，细化空间与语义表示，兼顾细节和广泛上下文，对准确分割十分重要。但传输前融合往往削弱空间细节，虽增强语义，却损害空间精度，对小目标等尤其关键。

因此提出双路径跳跃架构DSSG，利用PPA双输出明确解耦语义增强与空间保留。PPA产生语义丰富流X及高分辨率空间流X_skip：X经MSAA多尺度细化，X_skip绕过语义变换直接传解码器，保留细结构。

从建模看，DSSG将表示空间显式分叉，鼓励语义抽象和空间保留分至正交路径。形式为：

$$X\in\operatorname{MSAA}(F),\qquad X_{skip}=F.\tag{5}$$

F为PPA特征。该形式施加符合表示解耦原则的任务归纳偏置，使分支承担互补角色，语义细化不会覆盖位置敏感细节，对纹理丰富或杂乱环境尤其重要。

两流随后在FGCMamba中经SGCA整合：X作为查询，X_skip提供键和值，使解码器选择突出显著语义，同时将预测锚定在空间结构上，提高分割及边界定位。

### 3.3 频域引导上下文Mamba块

利用3.2节DSSG，我们引入FGCMamba块增强解码器，结合SGCA与FSAS，以增强多维特征建模。

第一项增强SGCA明确建模语义与空间交互：MSAA细化的X为查询，保留原编码细节的X_skip同时为键和值。

该配置计算反映高层语义模式与低层空间结构对齐的注意力。查询—键—值框架将语义投影到空间几何，实现跨表示域特征相关。与标准自注意力不同，此设计强制从语义抽象向空间锚定定向流动，符合跨模态注意力原则，支持解耦表示学习。

第二项改进用FSAS替代分离的空间和通道注意力，通过频率感知形式替代常规空间注意力。FSAS不直接在空间域计算注意图，而利用傅里叶变换更高效捕捉长依赖。如图3，给定$F\in\mathbb{R}^{H\times W\times C}$，先经1×1卷积产生三个投影：

$$F_q=\operatorname{Conv}_{1\times1}(F),\quad F_k=\operatorname{Conv}_{1\times1}(F),\quad F_v=\operatorname{Conv}_{1\times1}(F).\tag{6}$$

随后在频域计算，对F_q和F_k执行FFT，以逐元素复数乘法计算频谱相关：

$$A=\mathcal F^{-1}\bigl(\mathcal F(F_q)\cdot\overline{\mathcal F(F_k)}\bigr).\tag{7}$$

$\mathcal F$与$\mathcal F^{-1}$分别为傅里叶及逆变换，点乘为复数乘法，横线为共轭。归一化A并乘以F_v，最后经1×1卷积和残差得到：

$$Y=\operatorname{Conv}_{1\times1}\bigl(F_v\cdot\operatorname{LayerNorm}(A)\bigr)+X.\tag{8}$$

频域形式有效捕捉全局关系并降低成本，比传统空间注意力更好平衡表达与效率，改善细粒度感知。最后通过逐元素调制结合主路径及频率感知门控路径。用SGCA和FSAS产生的门控信号自适应加权解码图，选择强调信息丰富区域，平衡语义上下文和空间细节，显著提高分割精度。

**图3：**所提FSAS模块。Q（查询）、K（键）、V（值）由输入经1×1和3×3卷积生成；Q、K在频域处理以计算注意力，V用于聚合上下文。

### 3.4 损失函数

主要使用基础交叉熵和Dice，因为所有掩码只有单目标及背景两类。

$$L_{BCE+Dice}=\lambda_1L_{BCE}+\lambda_2L_{Dice},\tag{9}$$
$$L_{BCE}=-\frac1N\sum_{N}^{1}[y_i\log(\widehat y_i)+(1-y_i)\log(1-\widehat y_i)],\tag{10}$$
$$L_{Dice}=1-\frac{2|X\cap Y|}{|X|+|Y|}.\tag{11}$$

λ₁、λ₂为常数，通常默认(1,1)。**译注：**式(10)PDF可见求和上限1、下限N，按原式保留，未擅自改为通常的i=1到N。

## 4 实验

### 4.1 数据集

为全面评估不同应用领域的泛化和跨任务鲁棒性，选取特性明显不同的两个公开数据集，分别代表工业表面缺陷检测和遥感语义分割。二者在分辨率、形态复杂度、纹理和背景干扰上互补，可有效评估结构表示与泛化。

#### 4.1.1 MeiweiPCB数据集

使用已在[GitHub](https://github.com/youtang1993/MeiweiPCB)公开标注的MeiweiPCB分割数据集，采用2025年5月6日发布版本。由Meiwei Electronics开发，来自真实工业生产线，专为表面缺陷分割设计。图4所示，数据有926幅工业线扫相机图像，统一裁成220×220像素，包含断路、毛刺、漏孔、余铜等典型缺陷，每幅缺陷图都有高质量像素标签。复杂背景纹理、显著灰度变化、成像噪声、光照变化及遮挡，使边界建模和细节保存困难。统一零填充至256×256，按80%训练、20%测试划分。

**图4：**MeiweiPCB可视化：（a）表面污染；（b）划痕；（c）异物侵入；各组均依次为正常图、缺陷图、掩码。

#### 4.1.2 KWSD2水体分割数据集

使用Kaggle Water Segmentation Dataset 2（KWSD2），标注在[Kaggle](https://www.kaggle.com/datasets/kaoyuu/kaoyuuu/data)公开，访问日期2025年4月3日。KWSD2（KaoYuuu Water Segmentation Dataset）是遥感图像分析基准，专为高精度地表水提取设计。图5包含不同地区、时期的高分辨率遥感图，内容丰富、地貌变化大，覆盖湖泊、河流、水库和湿地。

**图5：**KWSD2可视化：（a）道路附近水体；（b）居民区附近水体；（c）城市小池塘。

图像涉及植被干扰、水面反射、云遮挡等复杂背景，以及剧烈形态变化和模糊边界，挑战上下文感知及边缘恢复。该集尤其适用于评估长依赖、空间保存和跨尺度整合。

原图492×492，四周各填10像素得到512×512。标签二值化：灰度大于127为水体，否则非水体。最终800幅训练、200幅测试，进行有监督学习及评估。

### 4.2 实验细节与评价指标

实验采用Ubuntu20.04、CUDA11.8、PyTorch2.0.1。优化器AdamW，初始学习率5×10⁻⁵，每步指数衰减。训练使用随机翻转、旋转增强，批大小8，最多100轮。评价采用精确率Precision、召回率Recall、F1及mIoU。

### 4.3 实验结果

CM-UNetv2在两数据集上表现优秀。表1显示MeiweiPCB mIoU0.8224、KWSD2 mIoU0.9516，优于现有主流方法。Precision与Recall接近，差值小，验证不同目标类型下的稳定性和鲁棒性。

**表1：**CM-UNetv2在两个数据集上的结果。

|数据集|Precision|Recall|F1|mIoU|
|---|---:|---:|---:|---:|
|MeiweiPCB|0.8970|0.8882|0.8925|0.8224|
|KWSD2|0.9690|0.9822|0.9750|0.9516|

图6展示代表结果：（a）准确分割过孔中心小缺陷，轮廓贴近真值；（b）准确识别分离两个独立缺陷；（c）在相似圆结构包围下，准确定位圆焊盘内边缘模糊缺陷；（d）强纹理干扰下也能识别图像边缘细长条缺陷。

**图6：**MeiweiPCB缺陷分割：（a）过孔中心小缺陷；（b）分布分离的两个独立缺陷；（c）边缘模糊、近似圆干扰的小缺陷；（d）图像边缘纵向条状缺陷。

图7展示KWSD2水体：（a）成功提取规则农田水体及小池塘；（b）水体与植被低对比时仍成功；（c）成功提取城市水体；（d）城乡结合部准确识别主要水体，尤其提取细长灌溉渠。

**图7：**KWSD2水体分割：（a）农业区，红框显示可清晰分割相邻水体；（b）城市区，环境复杂、边界模糊时仍准确描绘水缘并保持连续；（c）工业区，成功分割细长水体，轮廓平滑，形状保持强；（d）郊区，在杂乱背景中检测零散水斑与窄溪流，突出区域如图。

### 4.4 消融研究

设计两组消融从不同角度系统验证关键模块贡献。

第一组评估PPA、DSSG、解码器基于FSAS的FGCMamba，依次删除或替换模块，用mIoU、F1、Precision等分析上下文建模、空间细节和频域增强。

第二组评估不同嵌入维度与深度的性能效率，构建多种配置，在相同训练设置下比较并分析其影响。

两组结果验证模块独立贡献与协同，进一步表明架构在多类分割任务中的有效性和通用性。

#### 4.4.1 网络架构

表2、3显示模块A、B、C均明显改善mIoU和Precision，验证其提取、边界建模及细节恢复能力。

**表2：**MeiweiPCB网络结构消融。

|方法|Precision|Recall|F1|mIoU|
|---|---:|---:|---:|---:|
|CM-UNet|0.8640|0.8580|0.8615|0.7932|
|CM-UNet+A|0.8680|0.8650|0.8645|0.8061|
|CM-UNet+B|0.8750|0.8755|0.8752|0.8099|
|CM-UNet+C|0.8685|0.8605|0.8620|0.7978|
|CM-UNet+A+B|0.8905|0.8800|0.8851|0.8125|
|CM-UNet+A+C|0.8845|0.8855|0.8850|0.8168|
|CM-UNet+A+B+C|0.8970|0.8882|0.8925|0.8224|

**表3：**KWSD2网络结构消融。

|方法|Precision|Recall|F1|mIoU|
|---|---:|---:|---:|---:|
|CM-UNet|0.9489|0.9412|0.9450|0.9205|
|CM-UNet+A|0.9540|0.9621|0.9580|0.9298|
|CM-UNet+B|0.9572|0.9604|0.9588|0.9336|
|CM-UNet+C|0.9510|0.9645|0.9577|0.9250|
|CM-UNet+A+B|0.9615|0.9695|0.9655|0.9388|
|CM-UNet+A+C|0.9624|0.9705|0.9665|0.9405|
|CM-UNet+A+B+C|0.9632|0.9720|0.9676|0.9516|

模块A引入Patch-wise Patch Attention（PPA），增强多尺度上下文及细结构。KWSD2的mIoU增至0.9298、Precision0.9540；MeiweiPCB分别0.8061、0.8680。如图8a、9b，PPA改善细长河状结构和细节感知，重新连接部分断裂，但局部略过扩张。**译注：**PPA在正文不同位置分别称Parallelized Patch-Aware Attention和Patch-wise Patch Attention，保留原称并以PPA统一指代。

**图8：**MeiweiPCB结构消融：（a）PPA增强细长划痕感知，连接碎片并改善连续性；（b）DSSG保留细节、抑制噪声，改善杂乱区小缺陷检测；（c）三模块集成后最准确，高保真恢复长窄区域。

**图9：**KWSD2结构消融：（a）红框是相邻窄河，仅完整模型清晰分开，表明PPA对细结构区分重要；（b）红框存在边界歧义与干扰，DSSG细化轮廓、减少过分割；（c）FSAS改善长而碎的水体恢复，增强全局一致和整体完整性。

模块B（DSSG）双分支一路经MSAA增强语义，一路直传FGCMamba保留细节。此时FGCMamba无FSAS，仅用SGCA引导编码解码融合，准确建模边界。KWSD2和MeiweiPCB mIoU分别0.9336、0.8099。图8b和9c显示，在PPA上加DSSG可抑制冗余预测、锐化边缘、使结构更贴合真值。

模块C加入FSAS，以频域注意增强关键区域响应，抑制背景噪声，改善细节。KWSD2及MeiweiPCB mIoU分别0.9250、0.7978。如图8c、9a，FSAS增强长依赖，使边界恢复更准确、轮廓平滑精确。

三者同时使用时最佳，KWSD2 mIoU0.9516、MeiweiPCB0.8224。图8c、9a显示边界保存、细节恢复和抑噪更强。PPA增强缺陷敏感性，DSSG改善语义空间解耦并减少过分割，FSAS进一步优化全局一致性和边界。

#### 4.4.2 嵌入—深度架构

在两数据集上用三种CM-UNetv2配置研究嵌入维度和深度，KWSD2输入512×512，MeiweiPCB256×256。

表4中MeiweiPCB纹理更细、缺陷更多样，工业分割更复杂，整体稍低。各配置mIoU均大于0.81，96/2最佳，Precision0.8970、Recall0.8882、F1 0.8925、mIoU0.8224，说明噪声遮挡下细特征保持鲁棒。128/3稳定，分别0.8944、0.8850、0.8897、0.8199；64/4虽Precision0.8886、Recall0.8840接近，但mIoU较低0.8178，说明仅增深度无法补偿嵌入丰富度不足。

**表4：**MeiweiPCB嵌入—深度消融（列序照PDF表头）。

|Embed Dim|Depths|Precision|Recall|F1|mIoU|
|---:|---:|---:|---:|---:|---:|
|4|64|0.8886|0.8840|0.8863|0.8178|
|3|128|0.8944|0.8850|0.8897|0.8199|
|2|96|0.8970|0.8882|0.8925|0.8224|

表5的KWSD2三种都较强，仍有细微差异。高分辨率遥感图具有复杂结构、模糊边界和干扰，需强全局建模与细边界。嵌入96、深度2最佳，Precision0.9632、Recall0.9720、F1 0.9676、mIoU0.9516，平衡语义和空间。128/3也有竞争力，分别0.9618、0.9705、0.9661、0.9505，F1和mIoU略降。64/4虽Precision0.9602、Recall0.9690较高，mIoU最低0.9495，提示深结构、浅嵌入难以同时捕捉全局及细节。

**表5：**KWSD2嵌入—深度消融（列序照PDF表头）。

|Embed Dim|Depths|Precision|Recall|F1|mIoU|
|---:|---:|---:|---:|---:|---:|
|4|64|0.9602|0.9690|0.9645|0.9495|
|3|128|0.9618|0.9705|0.9661|0.9505|
|2|96|0.9632|0.9720|0.9676|0.9516|

> 译注：表4、5原表前两列数值与正文解释相反：表头先Embed Dim后Depths，但数值先4/3/2后64/128/96。此处照表保留，正文仍按原文解释。

总之96维、2层在两数据集泛化最佳，平衡简洁性与表达，兼顾大场景与细工业缺陷。这强调平衡嵌入丰富度及深度对准确、泛化和鲁棒性的重要性。

### 4.5 与其他方法比较

本节在MeiweiPCB与KWSD2上与其他语义分割网络比较，验证整体有效性及鲁棒性。

#### 4.5.1 MeiweiPCB比较

本小节在两个数据集比较以验证有效性。先进基线含DeepLabV3+[24]、SegFormer[40]、TransUNet[39]、SwinUNet[41]、CM-UNet[46]、CM-UNet++[48]。表6中CM-UNetv2的Recall、F1、mIoU最佳，TransUNet及CM-UNet++第二第三，均优于其他基线。图10中本方法多个缺陷场景均更好。与TransUNet均有良好整体恢复，但边界和细节更强。原文图8a的低对比细划痕边界模糊、纹理似背景，仅本方法完整连续分割；TransUNet和CM-UNet++虽检出主区，仍缺失断续；SegFormer和SwinUNet仅识别少量高对比段。图8b高对比划痕较容易，多数方法识别主体，但本文边界最佳且贴合真值；TransUNet和CM-UNet++主体完整但边缘缺细节；CM-UNet轻微粘连；DeepLabv3+、SwinUNet、SegFormer有漏检、覆盖不全。图8c位于密集过孔阵列，背景复杂，对抗干扰要求高，仅本文在强纹理中完整准确检测并保持清晰边界；TransUNet和CM-UNet++主体可见但边缘断裂偏移，CM-UNet部分不能与背景分开。各种挑战中本文最稳定全面，复杂背景和低对比下尤其鲁棒。**译注：**本段原文引用图8a—c，按上下文对应比较图10，保留原引用说明。

**表6：**MeiweiPCB方法比较。

|方法|Precision|Recall|F1|mIoU|
|---|---:|---:|---:|---:|
|Deeplabv3+|0.8712|0.8576|0.8643|0.7855|
|SegFormer|0.8560|0.8415|0.8487|0.7730|
|TransUnet|0.8890|0.8785|0.8837|0.8115|
|Swinunet|0.8615|0.8470|0.8542|0.7783|
|CM-UNet|0.8640|0.8580|0.8615|0.7932|
|CM-UNet++|0.8732|0.8610|0.8670|0.8084|
|CM-UNetv2|0.8970|0.8882|0.8925|0.8224|

**图10：**MeiweiPCB比较：（a）缺陷颜色接近背景、边缘模糊且纹理干扰明显，本方法细节识别更强；（b）清楚边界和高对比下，仍整体恢复最佳，划痕形状贴合真值；（c）复杂背景及密集结构遮掩细划痕时，凭抗干扰能力提取完整缺陷。

#### 4.5.2 KWSD2比较

为验证跨任务泛化，在KWSD2沿用4.5.1方法，除输入特征外全部超参数与MeiweiPCB相同。表7趋势类似表6，本文Recall、F1、IoU第一，mIoU比其他高超过1%，体现复杂遥感精度。Precision为0.9486排名第二，比DeepLabV3+的0.9402高0.84%，且优于其他，进一步确认综合有效性。图11a中水体形态复杂、边界不规则且多片相邻，仅本文完整提取所有水域；CM-UNet++紧随，多数准确、边缘较好，但漏掉小孤立区。图11b为建筑与水网混杂，水体多而窄、分布不规则、受建筑阴影干扰，难度高；本文主水体提取更完整、轮廓一致更好。图11c为农田和灌溉渠交错，水体窄、宽度变化大，与道路、田地、植被等低对比，本文结果最准确完整，断点少、边界贴合；CM-UNet++主体清楚但部分支流断；DeepLabV3+能恢复结构但边界不足；TransUNet和CM-UNet漏分断裂；SwinUNet和SegFormer窄渠明显不连续、结构差。

**表7：**KWSD2方法比较。

|方法|Precision|Recall|F1|mIoU|
|---|---:|---:|---:|---:|
|Deeplabv3+|0.9450|0.9410|0.9430|0.9320|
|SegFormer|0.9368|0.9491|0.9429|0.9158|
|TransUnet|0.9377|0.9432|0.9404|0.9223|
|Swinunet|0.9342|0.9481|0.9411|0.9186|
|CM-UNet|0.9489|0.9412|0.9450|0.9205|
|CM-UNet++|0.9549|0.9571|0.9560|0.9425|
|CM-UNetv2|0.9632|0.9720|0.9676|0.9516|

> 译注：本节正文Precision0.9486、DeepLab0.9402及排名说法与表7不符；表1的KWSD2三项指标也与表3、5、7不同。均按各自原值保留。

**图11：**KWSD2比较：（a）水与背景纹理差小、有多个相邻水域时，CM-UNet++可更准确提取结构；（b）城市多窄水体受遮挡时，CM-UNetv2分割优于其他，提取更完整；（c）灌溉渠与田地对比不明、水体窄时，CM-UNetv2保持结构完整性更好，接近真值。**译注：**图注(a)原写CM-UNet++，按原文保留。

#### 4.5.3 计算复杂度比较

表8汇总FLOPs、参数、FPS及mIoU。本文mIoU最高0.8224，同时成本21.63 GFLOPs、参数22.02M，具有强语义表示和边界定位。TransUNet参数93.23M，限制资源受限工业部署。CM-UNet++虽仅8.71M，但密集嵌套跳跃连接使成本71.65 GFLOPs、速度36.43 FPS。本文用更高效DSSG解耦语义细化及空间保留，空间特征绕过不必要变换，减少冗余和FLOPs同时保持细节。FSAS以轻量频域注意增强全局感知与边缘敏感，成本很低。合起来平衡准确和效率，适合严格性能及部署要求的工业分割。

**表8：**单块NVIDIA4090、256×256输入计算分析，mIoU在MeiweiPCB评估。

|方法|FLOPs(G)|参数(M)|FPS|mIoU|
|---|---:|---:|---:|---:|
|Deeplabv3+|82.95|60.99|58.32|0.7855|
|SegFormer|17.84|47.22|43.31|0.7730|
|TransUnet|37.23|93.23|53.47|0.8115|
|Swinunet|46.52|149.10|31.29|0.7783|
|CM-UNet|13.28|13.11|60.74|0.7932|
|CM-UNet++|71.65|8.71|36.43|0.8084|
|CM-UNetv2|21.63|22.02|47.32|0.8224|

为分析贡献，在256×256和512×512做内部消融。表9表明PPA是计算核心：256时移除它FLOPs降6.44G、参数降7.71M、速度升56.12 FPS。但4.4结构消融表明PPA对高精度及碎片重建不可缺，其图块注意赋予多尺度建模及结构感知，尽管成本高却不可替代。

**表9：**MeiweiPCB消融变体复杂度及推理速度（256×256，4090）。

|方法|FLOPs(G)|参数(M)|FPS|
|---|---:|---:|---:|
|CM-UNetv2|21.63|22.02|47.32|
|无DSSG|20.54|21.28|50.12|
|无PPA|15.19|14.31|56.12|
|无FSAS|20.31|20.70|49.88|

DSSG、FSAS则是轻量有效增强。DSSG双流同时丰富语义和保存空间，以低开销改善边界；FSAS以频域注意增强关键区域响应、抑制背景、恢复细节。表9、10显示移除任一仅略改FLOPs，确认轻量。它们与PPA共同在强语义基础上细化边界和细节。

**表10：**KWSD2消融变体复杂度及推理速度（原图注为256×256，4090）。

|方法|FLOPs(G)|参数(M)|FPS|
|---|---:|---:|---:|
|CM-UNetv2|86.53|22.02|36.91|
|无DSSG|83.15|21.28|39.04|
|无PPA|57.97|14.31|44.27|
|无FSAS|81.25|20.70|38.73|

> 译注：上下文说明同时测试256与512输入，KWSD2前文输入512×512，但表10图注原写256×256；此处不替换。

最后评估高分辨率可扩展性，各模块相对成本跨尺寸一致，验证设计稳定鲁棒。PPA构成表示核心，DSSG及FSAS提供关键高效细粒度增强，协作平衡准确、速度和成本，体现广泛工业部署潜力。

## 5 讨论

如4.5.1、4.5.2所示，CM-UNetv2在工业缺陷和遥感水体两代表任务中综合领先，主要得益于系统结构增强及MSAA双路径跳跃连接DSSG、PPA编码器、FGCMamba解码器、FSAS注意。

KWSD2中本文mIoU优于全部基线，Precision与Recall平衡。虽DeepLabV3+精确率略高，其mIoU及Recall低，体现细长水体、遮挡或模糊边界处理不足。根源是架构：DeepLabV3+依靠空洞卷积扩感受野，缺少有效局部细节聚合；本文DSSG一路增强语义、一路保留原编码特征，增强细结构和全图细节。**译注：**“DeepLab精确率略高”为原讨论陈述，与表7不一致，照译。

相比SegFormer和TransUNet，本文Recall及mIoU更稳定鲁棒。Transformer全局注意虽擅长长依赖，却难捕捉模糊边界或密集小目标的细结构，导致漏检。TransUNet在MeiweiPCB相对较好，但KWSD2明显下降，说明更适合清楚边界、目标集中场景。

CM-UNet和CM-UNet++都有跳跃连接，整体架构有优势。但本文用FGCMamba进一步增强引导，通过SGCA有效提取空间细节，融合语义与细空间，引导最相关编码特征参与重建，改善边界和纹理。FSAS整合空间与通道注意，在复杂纹理和噪声中突出关键区域、抑制冗余。

跨数据差异也反映结构与任务属性耦合。TransUNet遥感弱、工业强，表明更适于细节密集、纹理显著的近距离图；本文多尺度和全局局部协作在两任务均适应良好。

另一个优势是高效嵌入深度配置（96维、2层），两集均最佳。在复杂度可控时保证表达，提高训练稳定及推理效率。

总体上本文多项关键指标领先主流方法，以DSSG多尺度跳跃融合、PPA上下文感知、FGCMamba边界引导、FSAS关键区域表达，形成兼顾全局理解和局部建模的高性能系统，表现出可扩展、泛化和跨域适应。

## 6 结论

提出先进语义分割深网CM-UNetv2，改善边界、区域一致及小目标。为克服传统模型细结构与上下文依赖不足，加入PPA多尺度提取、DSSG语义空间解耦，解码期FGCMamba融合FSAS和SGCA，增强融合、轮廓和鲁棒性。

两数据集广泛实验表明关键指标更优，尤其边缘保存及细结构恢复。但也有局限：PPA虽显著提高准确，开销不可忽视，可能影响极受限场景实时部署；虽在两集泛化强，更多工业缺陷类型鲁棒性仍待验证。

未来将：（1）设计更轻PPA提高效率；（2）纳入真实工业生产线PCB样本增加多样性；（3）扩展多模态，结合光谱、纹理或时序线索，进一步改善复杂工业环境中的实用和适应性。

## 作者贡献及声明

**作者贡献：**Q.G.：构思、方法、调查、初稿；Y.C.：监督、项目管理、修订、经费；Y.Z.：数据整理、可视化；D.C.：形式分析、验证。所有作者阅读并同意发表版本。

**资助：**国家自然科学基金62273273，国家重点研发计划2023YFB3308803，陕西省重点研发计划2023-YBNY-203。

**机构审查委员会声明：**不适用。

**知情同意声明：**不适用。

**数据可用性：**研究原始数据可向通信作者索取。

**致谢：**感谢编辑及匿名审稿人提供有益建议以改善稿件。

**利益冲突：**作者声明无利益冲突。

## 参考文献

1. Mohd Amiruddin, A.A.A.; Zabiri, H.; Taqvi, S.A.A.; Tufa, L.D. 神经网络在故障诊断与检测中的应用：工程相关系统实现综述。Neural Comput. Appl. 2020, 32, 447–472. [CrossRef]

2. Zhang, H.; Jiang, L.; Li, C. CS-ResNet：用于PCB外观缺陷检测的代价敏感残差卷积神经网络。Expert Syst. Appl. 2021, 185, 115673. [CrossRef]

3. Gao, Y.; Li, X.; Wang, X.V.; Wang, L.; Gao, L. 面向工业智能的视觉缺陷识别最新进展综述。J. Manuf. Syst. 2022, 62, 753–766. [CrossRef]

4. Luo, Q.; Fang, X.; Liu, L.; Yang, C.; Sun, Y. 平面钢材表面自动视觉缺陷检测综述。IEEE Trans. Instrum. Meas. 2020, 69, 626–644. [CrossRef]

5. Maniat, M.; Camp, C.V.; Kashani, A.R. 利用谷歌街景图像进行基于深度学习的视觉裂纹检测。Neural Comput. Appl. 2021, 33, 14565–14582. [CrossRef]

6. Wei, X.; Yang, Z.; Liu, Y.; Wei, D.; Jia, L.; Li, Y. 基于图像处理与深度学习技术的铁路扣件缺陷检测：比较研究。Eng. Appl. Artif. Intell. 2019, 80, 66–81. [CrossRef]

7. Ming, W.; Shen, F.; Li, X.; Zhang, Z.; Du, J.; Chen, Z.; Cao, Y. 3C玻璃部件缺陷检测综合综述。Measurement 2020, 158, 107722. [CrossRef]

8. Tsai, D.M.; Molina, D.E.R. 具有圆形刀纹的机加工表面形态学缺陷检测。Measurement 2019, 134, 209–217. [CrossRef]

9. Zhu, M.; He, Y.; He, Q. 深度学习在遥感应用中的研究综述。Int. J. Geosci. 2019, 10, 1–11. [CrossRef]

10. Gryllias, K.C.; Antoniadis, I.A. 基于物理模型训练的支持向量机方法：工业环境滚动轴承故障检测。Eng. Appl. Artif. Intell. 2012, 25, 326–344. [CrossRef]

11. Xing, J.; Jia, M. 基于卷积神经网络的工件表面缺陷检测方法。Measurement 2021, 176, 109185. [CrossRef]

12. He, T.; Liu, Y.; Yu, Y.; Zhao, Q.; Hu, Z. 深度卷积神经网络在木材缺陷特征提取与检测中的应用。Measurement 2020, 152, 107357. [CrossRef]

13. Zhang, K.; Wang, W.; Lv, Z.; Fan, Y.; Song, Y. 采用注意力CNN的煤炭加工异物计算机视觉检测。Eng. Appl. Artif. Intell. 2021, 102, 104242. [CrossRef]

14. Chen, H.; Hu, Q.; Zhai, B.; Chen, H.; Liu, K. 用于表面缺陷检测的深度卷积网络鲁棒弱监督学习。Neural Comput. Appl. 2020, 32, 11229–11244. [CrossRef]

15. Zhang, C.; Shi, W.; Li, X.; Zhang, H.; Liu, H. 基于深度特征学习的改进裸PCB缺陷检测方法。J. Eng. 2018, 2018, 1415–1420. [CrossRef]

16. Liu, B.; Chen, D.; Qi, X. YOLO-pdd：利用序列图像深度表示的新型多尺度PCB缺陷检测方法。arXiv 2024, arXiv:2407.15427.

17. Li, D.; Fu, S.; Zhang, Q.; Mo, Y.; Liu, L.; Xu, C. 基于特征金字塔网络的改进PCB缺陷检测器。In Proceed- ings of the 2020 4th International Conference on Computer Science and Artificial Intelligence, Zhuhai, China, 11–13 December 2020; pp. 233–239.

18. Khalilian, S.; Hallaj, Y.; Balouchestani, A.; Karshenas, H.; Mohammadi, A. 利用去噪卷积自编码器进行PCB缺陷检测。In Proceedings of the 2020 International Conference on Machine Vision and Image Processing (MVIP), Qom, Iran, 8–20 February 2020; pp. 1–5.

19. Gustilo, R.C.; Ong, D.E.Y.; Chua, A.J.P. 使用卷积神经网络进行PCB故障检测。J. Southwest Jiaotong Univ. 2022, 57, 531–542. [CrossRef]

20. He, J.; Deng, Z.; Qiao, Y. 用于语义分割的动态多尺度滤波器。In Proceedings of the IEEE/CVF International Conference on Computer Vision, Seoul, Republic of Korea, 27 October–2 November 2019; pp. 3562–3572.

21. He, K.; Zhang, X.; Ren, S.; Sun, J. 用于视觉识别的深度卷积网络空间金字塔池化。IEEE Trans. Pattern Anal. Mach. Intell. 2015, 37, 1904–1916. [CrossRef]

22. Long, J.; Shelhamer, E.; Darrell, T. 用于语义分割的全卷积网络。In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Boston, MA, USA, 7–12 June 2015; pp. 3431–3440.

23. Ronneberger, O.; Fischer, P.; Brox, T. U-Net：用于生物医学图像分割的卷积网络。In Proceedings of the Medical Image Computing and Computer-Assisted Intervention–MICCAI, Munich, Germany, 5–9 October 2015; Volume 9351, pp. 234–241.

24. Chen, L.C.; Zhu, Y.; Papandreou, G.; Schroff, F.; Adam, H. 采用空洞可分离卷积的语义图像分割编码器—解码器。In Proceedings of the European Conference on Computer Vision (ECCV), Munich, Germany, 8–14 September 2018; Volume 11211, pp. 801–818.

25. Zhao, H.; Zhang, Y.; Liu, S.; Shi, J.; Loy, C.C.; Lin, D.; Jia, J. PSANet：用于场景解析的逐点空间注意力网络。In Proceedings of the European Conference on Computer Vision (ECCV), Munich, Germany, 8–14 September 2018; Volume 11208, pp. 267–283.

26. Chen, L.C.; Papandreou, G.; Schroff, F.; Adam, H. 重新思考用于语义图像分割的空洞卷积。arXiv 2017, arXiv:1706.05587.

27. Zhao, H.; Shi, J.; Qi, X.; Wang, X.; Jia, J. 金字塔场景解析网络。In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Honolulu, HI, USA, 21–26 July 2017; pp. 2881–2890.

28. Li, Z.; Liu, X. 基于改进UNet的PCB焊接缺陷分割方法。Appl. Sci. 2024, 14, 7370. [CrossRef]

29. Liu, Y.; Wu, H.; Xu, Y.; Liu, X.; Yu, X. 基于ControlNet与Swin Transformer的自动PCB样本生成及缺陷检测。Sensors 2024, 24, 3473. [CrossRef] [PubMed]

30. Feng, B.; Cai, J. 利用局部细节与全局依赖信息进行PCB缺陷检测。Sensors 2023, 23, 7755. [CrossRef] [PubMed]

31. Liu, B.; Luo, S.; Zhan, H.; Zhou, Y.; Huang, Z.; Chen, H. 用于复杂背景表面缺陷分割的变化感知孪生网络。Sci. Rep. 2025, 15, 11814. [CrossRef]

32. Lang, D.; Lv, Z. SEPDNet：简单有效的PCB表面缺陷检测方法。Sci. Rep. 2025, 15, 10919. [CrossRef] [PubMed]

33. Ulku, I.; Akagündüz, E. 二维图像语义分割深度学习架构综述。Appl. Artif. Intell. 2022, 36, 2032924. [CrossRef]

34. Zhou, L.; Zhang, C.; Wu, M. D-LinkNet：采用预训练编码器和空洞卷积的LinkNet，用于高分辨率卫星图像道路提取。In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Salt Lake City, UT, USA, 18–22 June 2018; pp. 182–186.

35. Zhao, X.; Wang, L.; Zhang, Y.; Han, X.; Deveci, M.; Parmar, M. 计算机视觉中的卷积神经网络综述。Artif. Intell. Rev. 2024, 57, 99. [CrossRef]

36. Zunair, H.; Hamza, A.B. 用于语义分割的掩码监督学习。arXiv 2022, arXiv:2210.00923.

37. Yang, Z.; Li, Z.; Shao, M.; Shi, D.; Yuan, Z.; Yuan, C. 掩码生成蒸馏。In European Conference on Computer Vision; Springer Nature: Cham, Switzerland, 2022; pp. 53–69.

38. Wu, F.; He, J.; Yin, Y.; Hao, Y.; Huang, G.; Cheng, L. 用于弱监督语义分割的掩码协同对比。In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, Waikoloa, HI, USA, 3–8 January 2024; pp. 862–871.

39. Chen, J.; Lu, Y.; Yu, Q.; Luo, X.; Adeli, E.; Wang, Y.; Lu, L.; Yuille, A.L.; Zhou, Y. TransUNet：Transformer是医学图像分割的强编码器。arXiv 2021, arXiv:2102.04306.

40. Xie, E.; Wang, W.; Yu, Z.; Anandkumar, A.; Alvarez, J.M.; Luo, P. SegFormer：采用Transformer的简单高效语义分割设计。In Proceedings of the Advances in Neural Information Processing Systems, Online, 6–14 December 2021; Volume 34, pp. 12077–12090.

41. Cao, H.; Wang, Y.; Chen, J.; Jiang, D.; Zhang, X.; Tian, Q.; Wang, M. Swin-Unet：用于医学图像分割的类U-Net纯Transformer。In Proceedings of the European Conference on Computer Vision (ECCV), Tel Aviv, Israel, 23–27 October 2022; pp. 205–218.

42. Gu, A.; Dao, T. Mamba：采用选择性状态空间的线性时间序列建模。arXiv 2023, arXiv:2312.00752.

43. Xu, R.; Yang, S.; Wang, Y.; Cai, Y.; Du, B.; Chen, H. 视觉Mamba：综述与新展望。arXiv 2024, arXiv:2404.18861.

44. Liu, Y.; Tian, Y.; Zhao, Y.; Yu, H.; Xie, L.; Wang, Y.; Ye, Q.; Jiao, J.; Liu, Y. VMamba：视觉状态空间模型。In Proceedings of the Advances in Neural Information Processing Systems, Vancouver, BC, Canada, 9–15 December 2024; Volume 37, pp. 103031–103063.

45. Zhao, S.; Chen, H.; Zhang, X.; Xiao, P.; Bai, L.; Ouyang, W. 用于大型遥感图像稠密预测的RS-Mamba。IEEE Trans. Geosci. Remote Sens. 2024, 62. [CrossRef]

46. Liu, M.; Dan, J.; Lu, Z.; Yu, Y.; Li, Y.; Li, X. CM-UNet：用于遥感图像语义分割的混合CNN-Mamba UNet。arXiv 2024, arXiv:2405.10530.

47. Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A.N.; Kaiser, L.; Polosukhin, I. 注意力就是你所需要的一切。In Proceedings of the Advances in Neural Information Processing Systems, Long Beach, CA, USA, 4–9 December 2017; Volume 30, pp. 5998–6008.

48. Cai, J.; Tao, L.; Li, Y. CM-UNet++：用于高分辨率遥感图像城市水体提取的多层信息优化网络。Remote Sens. 2025, 17, 980. [CrossRef]

**免责声明／出版者说明：**所有出版物中的陈述、观点及数据仅代表各作者和贡献者，不代表MDPI及／或编辑。MDPI及／或编辑不对因内容涉及的任何想法、方法、指令或产品造成的人身伤害或财产损失承担责任。
