# 检测真正重要的缺陷：面向制造业与零售物流的应用驱动异常检测基准（VAND 4.0 挑战赛）

**原文：**[Detecting Defects that Matter: An Application-Driven Benchmark for Anomaly Detection in Manufacturing and Retail Logistics (VAND 4.0 Challenge)](<18-VAND4-Application-Driven-Benchmark-2026.pdf>)。本译文对应本目录 14 页 PDF，arXiv:2610.04392v1，2026 年 10 月 3 日。翻译日期：2026-11-11。[arXiv 原文](https://arxiv.org/abs/2610.04392)。

**作者：**Lars Heckler-Kram¹˒²、Dorian Henning³、Ashwin Vaidya⁴、Jan-Hendrik Neudeck¹、Ulla Scheler¹、Anton Milan³、Samet Akcay⁴、Paula Ramos⁵、Sebastian Höfer³。

**单位：**¹ MVTec Software GmbH；² 慕尼黑工业大学；³ Amazon；⁴ Intel；⁵ NVIDIA。

**翻译说明：**按本地 PDF 翻译摘要、正文第 1—6 节、局限与未来工作、参考文献 [1]—[52] 和补充材料 A.1—A.5；覆盖图 1—4 图注、表 1—7 全部数值及公式 (1)—(22)。合并跨页续段，去除页眉页脚与页码；图形不重绘，图内可读文字补译。专名、数字及原文书目信息保留，参考文献另附中文题名；原文内部表述不一致以译注说明。“当前”等均指原文时点。原文称贡献有三方面，实际列出四项，本译文保留此情况。

## 摘要

现有异常检测基准已经饱和，并且常常不够真实。作为 VAND 4.0 挑战赛的一部分，我们提出一个隐藏测试集、由应用驱动的基准，覆盖两个部署关键领域：工业制造与零售物流。

工业赛道（MVTec AD 2）的结果表明，无监督异常分割仍然困难：常规设置最佳方法的像素 SegF₁ 仅约 57%，改进空间很大。零样本方法落后约 15 个 SegF₁ 点，确认针对任务利用正常数据训练，对精确缺陷定位仍至关重要。应对分布偏移的鲁棒性仍是关键开放挑战，DINOv3 骨干明显主导此赛道。

零售赛道（Kaputt 2）的结果表明：(1) 常见缺陷类型的有监督检测接近饱和；(2) 最佳现成 VLM 方法落后专用模型约 28 AP，说明当前 VLM 无法代替微调检测器；(3) 参考图像未能证明对最佳方法有帮助。稀有缺陷表现大幅下降：溢洒约 53 AP、缺件约 27 AP，有监督上限受数据可用性限制。为推动该领域进展，我们提供新低发生率零售异常检测数据集 Kaputt-Rare。

两个赛道均将计算效率作为首要指标，综合性能、吞吐量、内存和功耗。我们引入新的效率指标，发现最佳方法依赖沉重架构，而效率基本被忽视。总体上，研究界需要：(a) 开发更重视效率的方法；(b) 针对稀有缺陷与变化条件开发真正的异常检测方法。

挑战赛网站：[VAND 4.0 Challenge](https://sites.google.com/view/vand4-cvpr2026/challenge)。

## 1 引言

视觉异常检测（AD）是应用广泛的计算机视觉任务，从医疗诊断 [37]、自动驾驶 [19]、视频监控 [26]，到工业制造 [3] 和零售物流 [20]。这些任务的共同点是，此类设置中普遍存在长尾数据分布，导致异常事件稀少：按照定义，异常就是罕见的。

因此，AD 方法常在训练阶段建模正常数据分布，推理时识别偏离。已有多种方法遵循此范式 [2,34]，视觉语言模型（VLM）尤其提供令人振奋的新机会：仅需少量目标域正常数据，甚至完全不需要，就可解决视觉 AD [15]。但 AD 进展不仅依赖方法开发，也依赖合适基准数据集。尤其在工业领域，MVTec AD [3]、VisA [52] 促成突破性方法，却已实质饱和 [18,20]，AU-ROC 高达 99.9%。饱和妨碍进步：真正方法创新难与增量改进区分；对不受控照明、姿态等设置的鲁棒性仍未解决；无限制访问测试数据违背缺陷可能未知这一假设。此外，基于现有数据集的不同基准，在指标、正常训练数据量等协议上各异，实践者难以比较并选择适合应用的方法。因此学术与实际应用的鸿沟扩大，迫切需要新基准与稳健评估协议 [1]。

为此，我们提出面向制造与零售物流的应用驱动统一视觉 AD 基准，在 CVPR 2026 的视觉异常与新颖性检测挑战赛（VAND 4.0）中使用。众多学术和工业研究者贡献先进方法，按客观标准在留出测试集上评估。我们全面分析所有提交，最后讨论开放挑战和未来研究方向。贡献有三个方面：

- 提供工业制造与零售物流的综合真实世界视觉 AD 基准，并得出可行动的领域现状结论，包括常见缺陷性能饱和、VLM 当前局限与效率不足，以及稀有缺陷类别是关键未解决问题。
- 透明介绍挑战赛设计，支持清晰理解不同设置下的评估结果。
- 首次将算法效率作为 AD 基准关键标准；这常被学术研究忽视。提出衡量此高度部署相关属性的新指标，结果强调需要更重视效率的方法开发。
- 提出 Kaputt-Rare，以推动能解决零售物流 AD 的方法。约 1% 的缺陷发生率使其比 Kaputt [20] 更困难、更真实，促进从几乎无缺陷数据学习正常性与异常性的方法。

## 2 相关工作

### 2.1 异常检测方法

缺陷检测通常分为有监督、无监督、少样本、零样本几种范式。

本基准覆盖完整谱系。有监督方法利用有标注缺陷样本；实际中采集与标注昂贵、难以获得，但它们提供有价值的性能上界基线，用来衡量真正的 AD 方法。AD 又分为只需无异常参考样本的无监督方法，以及更困难的少样本、零样本设置，分别仅需极少或完全不需目标域参考数据。

工业检测尤其涌现大量无监督 AD 方法，可大体分为记忆库 [9,34]、重建 [10,16,27,49]、学生—教师 [2,35] 三类。三者遵循相同原则：显式嵌入正常数据（记忆库），或通过重建无异常数据、模仿已知教师的无缺陷嵌入来隐式学习。训练时常引入合成缺陷，进一步将特征空间约束于无异常数据 [42,50]。测试时，偏离该空间即判为异常。

CLIP [32] 等 VLM 出现后，研究广泛探索新 AD 范式 [23]：利用正常与异常状态文本，在语言与视觉共享语义空间匹配特征。由于预训练数据丰富，可能完全不需目标域参考数据，即零样本。类似地，Pixtral、GPT、Claude 等多模态大语言模型（MLLM）借助大规模训练，将日常物体及其可能正常/异常状态的深层语义知识直接编码于模型。这使其特别适合数据和物品变化极大的零售物流质量控制 [20]。然而计算成本仍高到难以实时部署。

VAND 4.0 同时评价工业制造与零售物流，为评估方法能否跨这些不同现实领域泛化提供独特机会。

### 2.2 异常检测数据集

已有多个 AD 基准被广泛采用。MVTec AD [3]、VisA [52] 推动通用方法开发，但性能已经饱和。一些近期数据集发表时就接近解决 [45]，不过仍通过扩展测试场景作出贡献。其他数据集将二维扩展到三维 [6,7]，[5] 则着重更深理解装配语义。除饱和降低可比性外，更基本的问题是：AD 数据集通常在评估中允许无限访问缺陷数据，违背 AD 本质。方法可能对某分辨率、特征提取器、训练设置隐式过拟合，损害其对真正未见缺陷的泛化，而这正是 AD 核心目标。因此，本挑战赛仅考虑限制测试访问的困难数据集，即 MVTec AD 2 [18] 和新近 Kaputt [20] 的扩展 Kaputt 2，以维护 AD 真正精神。

### 2.3 历届 VAND 挑战赛

年度视觉异常与新颖性检测挑战赛自 CVPR 2023 起举办，追随研究趋势，持续演进以弥合学术与工业鸿沟。首届聚焦工业零样本、少样本（≤10 样本）AD。2024 年纳入 VLM 进展，用 MVTec LOCO [5] 评估语义理解；同时引入聚焦照明、视点变化鲁棒性的第二赛道。2025 年 [17] 延续该赛道，但以真实图像 [18] 取代模拟。CVPR 2026 针对新兴且尚少探索的零售物流 [20] 增加 AD 任务，并首次明确将效率作为评价标准。各届均为研究者与实践者提供有关算法实际适用性的可行动见解。

## 3 VAND 4.0 挑战赛与基准设计

挑战赛设工业和零售两个赛道，除检测性能外也将模型效率作为关键评价标准。

### 3.1 赛道、设置与奖项

工业赛道基于 MVTec AD 2 [18]，按分割性能评估，需要将通常连续的预测二值化（第 3.3 节）。数据还可评估不同照明鲁棒性，变化可能来自不同设备或光源老化。赛道两种设置的训练数据访问不同：常规设置必须无监督，只用受评物体类别的正常图像训练；零样本设置允许任何公开数据训练，但禁止访问 MVTec AD 2。

零售赛道基于新近 Kaputt [20] 的扩展 Kaputt 2，是大规模零售物流 AD 数据集。因此着重评价适应多种物体类型的方法，并在图像级评估预测（第 3.3 节）。常规设置不约束方法，AD 与有监督均可，目标是建立新的任务性能上界基线。现成（off-the-shelf，OTS）设置则评价零/少样本 VLM/MLLM 现状，不允许微调或以其他方式修改核心模型权重。

两赛道奖项与实际场景一致，以激励应用驱动开发。除总体与零样本/OTS 最佳性能外，也按时间、内存、功耗奖励效率（第 3.3 节）。

### 3.2 数据集

数据集不仅源于真实困难场景，也通过隐藏真值和严格限制评估次数，约束在未见测试数据上的模型评估。

**工业赛道：MVTec AD 2。**[18] 为克服现有基准饱和而设计，含超过 8,000 幅高分辨率图像、8 类物体，如罐、织物、小瓶，分辨率 2.3—5 百万像素。训练/验证仅正常图像，多个测试集含不同照明下正常和异常样本。TEST_priv 照明与训练相同；TEST_priv,mix 显示相同场景但随机变化照明，可评估分布偏移鲁棒性。相比 MVTec AD [3,4]、VisA [52]、RealIAD [45]，类别较少却更复杂，具有较大类内变化、微小缺陷、边缘缺陷和多物体场景。

**零售赛道：Kaputt 2。**Kaputt [20] 是大规模零售物流视觉缺陷检测基准，产品类别高度多样，姿态变化显著，并有细粒度标注：严重程度、7 种缺陷、分割掩码和物品元数据。与挑战赛同步发布的 Kaputt 2 是来自同分布的 10,000 幅 RGB 留出评估图像，格式结构匹配原数据，缺陷发生率约 29%，类型分布相近，但隐藏标签以避免过拟合。

### 3.3 指标

**工业赛道：分割 F₁。**按像素 F₁（SegF₁）评价定位与分割，平衡精确率和召回率：

$$\mathrm{SegF}_1=2\frac{\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}.\tag{1}$$

SegF₁ 需要为连续异常图选阈值。科学界常尚未处理这一点，但实际部署必不可少。遵循标准 AD 协议，在对应测试集所有像素上计算精确率和召回率，而不是先逐图再平均。

工业最终排名 R_final 同时考虑总体性能与实际偏移鲁棒性，为模型在两个测试集按八类物体平均 SegF₁ 排名的均值：

$$R_{\mathrm{final}}=\tfrac12[R(\mathrm{SegF}_{1,\mathrm{priv}})+R(\mathrm{SegF}_{1,\mathrm{priv\ mix}})].\tag{2}$$

其中

$$\mathrm{SegF}_{1,t}=\frac18\sum_{c=1}^{8}\mathrm{SegF}_{1,c}(\mathrm{TEST}_t).\tag{3}$$

t 表示 priv 或 priv mix。R_final 相同时，以两个分数绝对差较小者优先。

**零售赛道：平均精确率。**用任意缺陷平均精确率 AP_any 评估真实零售物流检测。轻微、严重缺陷都视为正样本，评价所有可能阈值下，模型将缺陷物品排在正常物品之前的能力：

$$\mathrm{AP}_{\mathrm{any}}=\sum_n(R_n-R_{n-1})P_n.\tag{4}$$

Pₙ、Rₙ 为阈值 n 下精确率、召回率，Pₙ=TP/(TP+FP)，Rₙ=TP/(TP+FN)。TP 为正确识别的缺陷物品，FP 为被误报的正常物品，FN 为漏检缺陷物品。

与工业像素定位分割不同，零售聚焦图像级检测，主要衡量所有缺陷类型的 AP_any。它根据提交的 Kaputt 2 缺陷概率排序计算，避免评估时选择单个固定阈值，同时覆盖整条精确率—召回率曲线。最终排名由测试 AP_any 决定：

$$R_{\mathrm{final}}=R(\mathrm{AP}_{\mathrm{any}}).\tag{5}$$

AP 越高，整体检测越好。预测文件每幅测试图对应一个缺陷概率分数，越大表示更可能有缺陷。

**计算效率。**除预测性能外，评价效率以反映实际部署约束。效率综合检测质量、推理吞吐量、峰值内存、平均功耗。具体基于模型性能 q、归一化吞吐量 τ、内存 µ、功耗 π。q、τ 越高越好，内存功耗越低越好，因此用补量：

$$E=[q\tau(1-\mu)(1-\pi)]^{1/4}.\tag{6}$$

q 为任务分数：工业 q=SegF₁,avg=(SegF₁,priv+SegF₁,priv mix)/2；零售 q=AP_any。原始吞吐 t 按目标 t_max=30 fps 用边际收益递减函数 τ=t/(t+t_max) 归一化；峰值内存 m、平均功率 p 相对预算 m_max=64 GB、p_max=56 W 归一化：µ=m/m_max，π=p/p_max。

几何平均鼓励均衡模型，任一维严重不足都会降低最终分数，避免模型仅凭预测或吞吐高却消耗过多资源而得到高效率。目标操作点后吞吐归一化边际收益递减，防止纯速度主导评估。指标动机、归一化及敏感性分析见附录。

### 3.4 提交过程与评估协议

工业赛道使用 [MVTec Benchmark Server](https://benchmark.mvtec.com/)，零售使用 [Codabench](https://codabench.org/competitions/14156)。为保护隐藏测试集并防过度调参，工业每七天限三次提交，零售总计限四次。按第 3.3 节各赛道指标评价。

进入最终排行榜并取得获奖资格，必须提供可复现的开源代码库，含所有必要权重或下载说明，以及描述方法的公开技术报告。赛后在统一硬件上评估计算效率，细节见附录。

## 4 结果与分析

### 4.1 工业赛道

仅考虑同时有报告和代码的完整参赛方案，常规设置最终有 9 个，零样本有 5 个。较 2025 年 VAND [17]，参赛团队增加 40%。

**表 1：工业赛道（MVTec AD 2）常规设置排行榜。**

| 最终排名 | 方法 | SegF₁,priv | SegF₁,priv mix | 报告 | 代码 |
| --- | --- | --- | --- | --- | --- |
| 1 | SuperADD | 57.42 | 54.35 | ★ | GitHub |
| 2 | ISVL++ | 58.31 | 53.75 | ★ | GitHub |
| 3 | Forgis baseline | 49.02 | 45.80 | ★ | GitHub |
| 4 | MSAD | 48.54 | 45.50 | ★ | GitHub |
| 5 | DL-SubspaceAD | 46.88 | 45.36 | ★ | GitHub |
| 6 | vandseg 4 | 46.48 | 40.61 | ★ | GitHub |
| 7 | CAT | 44.26 | 39.49 | ★ | GitHub |
| 8 | ST | 41.53 | 33.97 | ★ | GitHub |
| 9 | GBE AD | 36.36 | 31.60 | ★ | GitHub |
| VAND 3.0 | ISVL [46] | 53.81 | 51.43 | ★ | GitHub |

**常规设置。**虽比 VAND 3.0 [17] 规则更严格，例如禁止按类别选择模型，去年冠军 ISVL [46] 的 SegF₁,priv 和 SegF₁,priv mix 仍分别被超越 2.5 个百分点以上、接近 3 个百分点（表 1）。冠军 SuperADD [33] 除改善预/后处理外，在去年的 SuperAD [51] 上加入 DINOv3 [38]，并改进构建正常特征记忆库时的子采样。第二名 ISVL++ [47] 同样将 ISVL 更新为 DINOv3，利用合成异常在 INP-Former [27] 上训练分割头并估计阈值。

**表 2：工业赛道（MVTec AD 2）零样本设置排行榜。**

| 最终排名 | 方法 | SegF₁,priv | SegF₁,priv mix | 报告 | 代码 |
| --- | --- | --- | --- | --- | --- |
| 1 | BALM | 42.64 | 41.45 | ★ | GitHub |
| 2 | ProtoFuse | 30.50 | 30.44 | ★ | GitHub |
| 3 | SPR (lin) | 23.96 | 24.05 | ★ | GitHub |
| 4 | sim2 | 24.55 | 22.84 | ★ | GitHub |
| 5 | EthMuSAM | 23.57 | 22.15 | ★ | GitHub |

**零样本设置。**BALM [12] 扩展 AnomalyVFM [11]，将阈值估计更深入融入训练，用 SAM [24] 精修二值分割图。与常规设置类似，通过更新特征提取器构建于近期方法之上。例如 ProtoFuse [28]、SPR [39] 等修改 VisualAD [21]，用 DINOv3 代替 DINOv2。

**分析。**工业结果显示出清晰模式，反映制造业视觉异常检测研究趋势。

**DINOv3 领先。**尤其常规设置强烈依赖 DINOv3 特征，提示从设计更复杂 AD 架构转向利用强大通用视觉表征，配合较简单评分机制。例如 DL-Subspace AD [22] 将基于 ViT 7B/16 的 DINOv3 骨干插入原 Subspace AD [25]，通过主成分分析压缩和重建实现异常检测。

**分辨率重要。**与 VAND 3.0 一样，切块与多尺度处理用于 MVTec AD 2 大图，反映高分辨率细微缺陷检测的难度。切块、多尺度或高分辨率均增加计算负担，可能严重损害效率（第 4.3 节）。

**形态学与深度学习二值化。**许多方案除 SAM 精修外，还用形态操作和连通分量过滤获得二值图。这既说明经典与深度学习可结合，也提出问题：能否将阈值选择/二值化进一步融入模型设计，免去无论何种形式的经验后处理？

**访问目标域反而降低鲁棒性。**照明偏移仍是中心难题。DL-Subspace AD 等较稳健，但常规设置多数方法两测试分数差明显。零样本中，已见与部分未见照明测试集差通常小于 1.5 个百分点。因而，能否为访问目标类别正常数据的方法引入更强鲁棒性？还需深入理解不同照明鲁棒或敏感的机制，或可揭示适用于其他采集变化所致偏移的措施。

### 4.2 零售赛道

74 位独立参赛者在 Kaputt 2 提交 286 份预测，其中 17 份为 OTS。

**表 3：零售赛道（Kaputt 2）常规设置排行榜。**

| 最终排名 | 团队 | AP_any | 报告 | 代码 |
| --- | --- | --- | --- | --- |
| 1 | dinovision | 90.84 | ★ | GitHub |
| 2 | sandeepsaluru | 89.30 | ★ | GitHub |
| 3 | xie233 | 84.32 | ★ | GitHub |

**常规设置。**表 3 显示包含报告、代码的可复现方案。最佳 AP_any 为 90.84，但前 30 名仅跨越不到 1.6 AP（89.27—90.84），见 [完整榜单](https://codabench.org/competitions/14156/#/results-tab)。冠军 dinovision [41] 集成 DINOv2-Large [30]、DINOv3-Large、DINOv3-Huge [38] 三个大 ViT 骨干，配合 56 视图测试时增强，推理成本约单模型的 3—5 倍。亚军 sandeepsaluru [36] 用两个 DINOv3-Large，成本约两倍。两者都在 Kaputt [20] 上有监督训练，利用 DINOv3 图块表征分类。几乎所有方法很少使用参考图像，暗示其对最佳方案帮助有限。

**表 4：零售赛道（Kaputt 2）现成（OTS）设置排行榜。**

| 最终排名 | 团队 | AP_any | 报告 | 代码 |
| --- | --- | --- | --- | --- |
| 1 | rahulvippala | 61.88 | ★ | GitHub |
| 2 | dinovision | 61.83 | ★ | GitHub |
| 3 | ant-hy | 54.87 | ★ | GitHub |
| 4 | amirparsa-sal | 50.49 | ★ | GitHub |
| 5 | sandeepsaluru | 46.75 | ★ | GitHub |

**现成 VLM 设置。**最佳 OTS AP_any=61.88，比最佳有监督约低 29 点 [43]，确认通用 VLM 当前无法代替微调分类器。使用模型包括 Gemini 3.1 Pro Gemini [13]、Gemma4-31B [14]、Qwen3.5-27B、Qwen3.5-9B [31]、InternVL [8]、GPT-4o [29]。相较朴素提示基线提高 19.6 AP 的关键创新为多提示集成 [44]、基于 logprob 的置信评分（由 token 对数概率提取连续缺陷概率，而非解析二元文本）[40]、结构化逐类别输出模式 [43]。最佳方案发现，刻意用不同措辞提问的多提示多样性，比多尺度或多模型多样性更有效 [36,43,44]。

> 译注：正文模型版本名与引用文献 [13] Gemini 2.5、[14] Gemma 3、[31] Qwen3.5-Omni 不完全对应，照原文保留，不据此推断版本关系。

**分析。**零售结果揭示物流视觉缺陷检测的若干发现。

**常见缺陷性能饱和。**前 30 名差在 1.6 AP_any 内，说明标签充分时，变形、穿孔等常见显著缺陷基本解决。此时边际收益依赖昂贵集成而非算法创新，继续开发这些类别的有监督方法回报递减。

**稀有缺陷失败。**逐类型分析（图 1）显示所有提交都难处理欠代表类别：溢洒仅约 53 AP_any，缺件约 27 AP_any。原因是训练数据不足，例如 Kaputt [20] 仅 705 个溢洒实例，以及这类缺陷视觉特点与结构损坏根本不同。提供同一物品无缺陷视图的参考图像也未弥合差距：Kaputt 2 每查询物品最多提供三张参考，但两个设置最佳方法都未将其作为主要信号。这与参考方法为标准的工业赛道不同，挑战了零售产品极多样就必须显式正常参考的假设。结果反而说明，当前方法已从大规模有监督训练获取充分正常先验，真正瓶颈是稀有缺陷样本不足，而非缺少参考数据。

**图 1：Kaputt 2 前三名方案各缺陷类型的 AP_any。**常见结构缺陷基本解决，稀有类别（溢洒、缺件）仍困难。图中横轴为缺陷类型，依次为功能动作异常（actuation）、拆解（deconstruction）、变形（deformation）、穿孔（penetration）、表面缺陷（superficial）、溢洒（spillage）、缺件（missing_unit）；纵轴为平均精确率，刻度为 0.0、0.2、0.4、0.6、0.8、1.0。图例为 dinovision、sandeepsaluru、xie233；柱形没有标注逐柱精确数值。

**VLM 尚无竞争力但进步迅速。**OTS 与有监督约 29 AP_any 差距说明 VLM 缺乏零售检测所需细粒度视觉辨别力。然而提示工程带来显著改进，随着视觉 grounding 能力改善可能越来越可用。多提示优于多模型集成表明，单模型重复判断比跨模型组合信号更有效。

**趋势。**架构与工业赛道类似：DINOv3 提取特征配任务分类头。但零售偏向聚合图块特征的全局图像分类，而工业依赖空间定位与局部评分。最佳方案普遍采用多视图测试增强以应对姿态变化。

### 4.3 效率

表 5 总结两赛道效率。仅对能够在硬件限制内完成端到端推理的方案测量，目标为配 Core Ultra Series 3 处理器、64 GB RAM 的工业边缘设备。零售唯一合格方案效率 0.3473，仍有很大改进空间。它将 DINOv2-ViTB/14 教师软目标蒸馏到 DINOv2-ViTS/14 学生，无沉重 VLM [48]。工业 SPR [39] 最高 0.1937，尽管其 q（平均 SegF₁）为所有被考虑方法最低，却在吞吐、内存、功耗上出色，可能因为只用 336×336 图像。总体说明提交主要追求高性能而忽视资源高效推理。

**表 5：两赛道按 E 得出的效率排名 R_eff。零售 q=AP_any；工业 q=SegF₁,avg。E 由归一化吞吐 τ、内存 µ、功耗 π 和性能 q 的几何平均得到（公式 6）。仅展示能在目标硬件运行的方法；原表各赛道各列最佳/次佳以粗体/下划线标注。**

| 赛道 | 效率排名 | 团队/方法 | E ↑ | τ ↑ | µ ↓ | π ↓ | q ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 零售 | 1 | xie233 | 0.3473 | 0.1093 | 0.7160 | 0.4442 | 0.8432 |
| 工业 | 1 | SPR (lin) | 0.1937 | 0.0993 | 0.8964 | 0.4300 | 0.2401 |
| 工业 | 2 | Forgis baseline | 0.1599 | 0.1185 | 0.9791 | 0.4441 | 0.4741 |
| 工业 | 3 | GBE AD | 0.1415 | 0.0221 | 0.9088 | 0.4144 | 0.3398 |
| 工业 | 4 | BALM | 0.1414 | 0.0250 | 0.9376 | 0.3899 | 0.4205 |
| 工业 | 5 | CAT | 0.1350 | 0.0631 | 0.9772 | 0.4477 | 0.4188 |
| 工业 | 6 | ISVL++ | 0.1219 | 0.0083 | 0.9142 | 0.4441 | 0.5603 |
| 工业 | 7 | MSAD | 0.1140 | 0.0355 | 0.9818 | 0.4446 | 0.4702 |
| 工业 | 8 | SuperADD | 0.1057 | 0.0041 | 0.9028 | 0.4435 | 0.5589 |

## 5 讨论

本挑战赛全面评估制造与零售两个不同应用领域的异常检测现状。

**零样本和 VLM 距实际部署仍远。**工业零样本落后约 15 SegF₁，零售 OTS 落后约 29 AP_any，确认通用模型尚不及任务专用微调。但两个赛道也表明提示工程和相关域迁移学习可显著缩小差距，指明有希望的方向。

**稀有缺陷和偏移是主要开放挑战。**零售稀有类型是真正失败模式，有监督方法没有额外数据就难解决；工业未见照明始终使性能下降。两者均反映现实 AD 核心属性：缺陷天然稀少，采集条件会变。

**效率值得更多关注。**性能冠军依赖昂贵集成（零售）或重骨干（工业），导致资源需求高、效率下降。实际部署要求未来开发优先考虑效率—性能权衡。轻量网络能否达到相近 AD 性能，也可能开启新方法空间。

## 6 结论

我们提出 VAND 4.0，作为工业制造与零售物流中应用驱动的视觉 AD 基准。两赛道考虑保持 AD 真正本质、即测试缺陷未知的数据集，确认：(1) 零售常见缺陷有监督检测接近饱和，稀有缺陷未解决；(2) OTS VLM/零样本落后专用模型，但受益于提示工程、集成、迁移学习；(3) 改善性能—效率权衡仍是开放问题，对大图细小微弱缺陷尤甚，而这在现实工业视觉检测十分常见。

### 局限

挑战赛未来仍有若干设计可改进。

**效率评估。**效率赛道为赛后评价而非实时竞争标准，限制参与。直接在提交流程融入效率约束可激励更重视资源的方法。

**区分评估协议。**零售最终排名未区分有无参考图像，难以隔离其贡献。工业零样本也需更严格界定允许哪些测试时适配、测试集归一化或跨图比较。明确规则既指导参赛者，也帮助研究界就允许预训练数据的零样本定义达成统一。

**受控特征提取。**除预定义评估协议，还可预定义特征提取器，或选轻量架构，改善方法可比性。也可要求骨干消融，以隔离各方法真正贡献。

### 未来工作：Kaputt-Rare

挑战赛表明零售常见缺陷有监督检测饱和，稀有类别却失败，而此处正需真正 AD。我们发布 Kaputt-Rare：Kaputt [20] 的重新划分版本，面向异常与新颖性检测，缺陷发生率约 1%，模拟质量受控物流环境中真实低发生率，要求从几乎无缺陷数据学习正常性与异常性。

表 6 给出初始基线建立参考性能，包括有监督分类器和无监督 AD，展示现实低发生率难度。Kaputt-Rare 以划分文件公开，将作为未来聚焦零售物流 AD 的 VAND 挑战赛基础。

**表 6：Kaputt-Rare 基线。ViT-S 在正常占多数、约 1% 缺陷的训练集微调，其余均无需训练（零/少样本）。ViT-S 用完整托盘图，CLIP、WinCLIP、PatchCore50 用物品裁剪。测试集 10067 个样本，其中 2089 轻微、1117 严重缺陷。数值均为百分比。**

| 方法 | AP_any | AP_major | AUROC |
| --- | --- | --- | --- |
| ViT-S [30] | 58.38 | 50.09 | 72.99 |
| CLIP [32] | 36.20 | 17.15 | 56.05 |
| WinCLIP-zero [23] | 33.87 | 19.11 | 52.30 |
| WinCLIP-few [23] | 34.05 | 19.29 | 52.41 |
| PatchCore50 [34] | 35.86 | 17.80 | 54.69 |

## 参考文献

[1] Aimira Baitieva, Yacine Bouaouni, Alexandre Briot, Dick Ameln, Souhaiel Khalfaoui, and Samet Akcay. Beyond academic benchmarks: Critical analysis and best practices for visual industrial anomaly detection. InProceedings of the Computer Vision and Pattern Recognition Conference, pages 4015–4025, 2025. 1

中文题名：超越学术基准：视觉工业异常检测的批判分析与最佳实践。

[2] Kilian Batzner, Lars Heckler, and Rebecca K ¨onig. EfficientAD: Accurate visual anomaly detection at millisecond-level latencies. InIEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 128–138, 2024. 1, 2

中文题名：EfficientAD：毫秒级延迟下的准确视觉异常检测。

[3] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. MVTec AD — A comprehensive real-world dataset for unsupervised anomaly detection. InIEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9584–9592, 2019. 1, 2, 3

中文题名：MVTec AD：用于无监督异常检测的综合真实世界数据集。

[4] Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger. The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.International Journal of Computer Vision, 129(4):1038–1059, 2021. 3

中文题名：MVTec 异常检测数据集：用于无监督异常检测的综合真实世界数据集。

[5] Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger. Beyond dents and scratches: Logical constraints in unsupervised anomaly detection and localization.International Journal of Computer Vision, 130 (4):947–969, 2022. 2, 3

中文题名：超越凹痕和划痕：无监督异常检测与定位中的逻辑约束。

[6] Paul Bergmann, Xin Jin, David Sattlegger, and Carsten Steger. The MVTec 3D-AD dataset for unsupervised 3D anomaly detection and localization. InProceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications — Volume 5: VISAPP, pages 202–213. SciTePress, 2022. 2

中文题名：用于无监督三维异常检测与定位的 MVTec 3D-AD 数据集。

[7] Luca Bonfiglioli, Marco Toschi, Davide Silvestri, Nicola Fioraio, and Daniele De Gregorio. The Eyecandies dataset for unsupervised multimodal anomaly detection and localization. InProceedings of the 16th Asian Conference on Computer Vision (ACCV), 2022. 2

中文题名：用于无监督多模态异常检测与定位的 Eyecandies 数据集。

[8] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling.arXiv preprint arXiv:2412.05271, 2024. 6

中文题名：通过模型、数据及测试时扩展拓宽开源多模态模型性能边界。

[9] Thomas Defard, Aleksandr Setkov, Angelique Loesch, and Romaric Audigier. PaDiM: A patch distribution modeling framework for anomaly detection and localization. InPattern Recognition. ICPR International Workshops and Challenges 2021, Proceedings, Part IV, pages 475–489. Springer, 2020. 2

中文题名：PaDiM：用于异常检测与定位的图块分布建模框架。

[10] Hanqiu Deng and Xingyu Li. Anomaly detection via reverse distillation from one-class embedding. InIEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9737–9746, 2022. 2

中文题名：通过单类嵌入的逆向蒸馏实现异常检测。

[11] Matic Fu ˇcka, Vitjan Zavrtanik, and Danijel Skoˇcaj. Anomalyvfm – transforming vision foundation models into zeroshot anomaly detectors. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 35555–35566, 2026. 5

中文题名：AnomalyVFM：将视觉基础模型转为零样本异常检测器。

[12] Zhi Gao, Shuying Xia, Yuxiang Zhao, Xinyi Fu, Mengjie Xie, Min Cao, and Qiyuan Zhang. Balm: Boundary-aligned logit margin learning for zero-shot industrial anomaly detection and segmentation.https : / / github . com / Serena - cooper / BALM / blob / main / BALM . pdf, 2026. Accessed: 2026-06-26. 5

中文题名：BALM：面向零样本工业异常检测与分割的边界对齐 logit 间隔学习。

[13] Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities.arXiv preprint arXiv:2507.06261, 2025. 6

中文题名：Gemini 2.5：以先进推理、多模态、长上下文及下一代智能体能力推进前沿。

[14] Gemma Team. Gemma 3 technical report.arXiv preprint arXiv:2503.19786, 2025. 6

中文题名：Gemma 3 技术报告。

[15] Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Ming Tang, and Jinqiao Wang. AnomalyGPT: Detecting industrial anomalies using large vision-language models.Proceedings of the AAAI Conference on Artificial Intelligence, 38(3):1932–1940, 2024. 1

中文题名：AnomalyGPT：利用大型视觉语言模型检测工业异常。

[16] Jia Guo, Shuai Lu, Weihang Zhang, Fang Chen, Huiqi Li, and Hongen Liao. Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20405–20415, 2025. 2

中文题名：Dinomaly：多类别无监督异常检测中少即是多的理念。

[17] Lars Heckler-Kram, Ashwin Vaidya, Jan-Hendrik Neudeck, Ulla Scheler, Dick Ameln, Samet Akcay, and Paula Ramos. From benchmarks to reality: Advancing visual anomaly detection by the vand 3.0 challenge.arXiv preprint arXiv:2509.17615, 2025. 3, 5

中文题名：从基准到现实：通过 VAND 3.0 挑战赛推进视觉异常检测。

[18] Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca K ¨onig, and Carsten Steger. The mvtec ad 2 dataset: Advanced scenarios for unsupervised anomaly detection.International Journal of Computer Vision, 134(4):175, 2026. 1, 3

中文题名：MVTec AD 2 数据集：面向无监督异常检测的进阶场景。

[19] Dan Hendrycks, Steven Basart, Mantas Mazeika, Mohammadreza Mostajabi, Jacob Steinhardt, and Dawn Song. A benchmark for anomaly segmentation. arXiv preprint arXiv:1911.11132v1, 2019. 1

中文题名：异常分割基准。

[20] Sebastian H ¨ofer, Dorian Henning, Artemij Amiranashvili, Douglas Morrison, Mariliza Tzes, Ingmar Posner, Marc Matvienko, Alessandro Rennola, and Anton Milan. Kaputt: A large-scale dataset for visual defect detection. InProceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Honolulu, Hawaii, USA, 2025. IEEE. 1, 2, 3, 6, 7, 8

中文题名：Kaputt：大规模视觉缺陷检测数据集。

[21] Yanning Hou, Peiyuan Li, Zirui Liu, Yitong Wang, Yanran Ruan, Jianfeng Qiu, and Ke Xu. Visualad: Language-free zero-shot anomaly detection via vision transformer. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21346–21356, 2026. 5

中文题名：VisualAD：通过视觉 Transformer 实现无语言零样本异常检测。

[22] Tenshi Ito. Dl-subspacead: Training-free anomaly segmentation via deep layer dinov3 features on full-resolution mvtec ad 2.https : / / github . com / riips / VAND4 . 0 _ IndustrialTrack _ Regular / blob / main/report.pdf, 2026. Accessed: 2026-06-26. 5, 6

中文题名：DL-SubspaceAD：在全分辨率 MVTec AD 2 上利用深层 DINOv3 特征实现免训练异常分割。

[23] Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, and Onkar Dabeer. Winclip: Zero- /few-shot anomaly classification and segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19606–19616, 2023. 2, 8

中文题名：WinCLIP：零/少样本异常分类与分割。

[24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll ´ar, and Ross Girshick. Segment anything.arXiv:2304.02643, 2023. 5, 6

中文题名：分割一切。

[25] Camile Lendering, Erkut Akdag, and Egor Bondarau. Subspacead: Training-free few-shot anomaly detection via subspace modeling. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28557–28566, 2026. 5

中文题名：SubspaceAD：通过子空间建模实现免训练少样本异常检测。

[26] Wei-Xin Li, Vijay Mahadevan, and Nuno Vasconcelos. Anomaly detection and localization in crowded scenes.IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(1):18–32, 2013. 1

中文题名：拥挤场景中的异常检测与定位。

[27] Wei Luo, Yunkang Cao, Haiming Yao, Xiaotian Zhang, Jianan Lou, Yuqi Cheng, Weiming Shen, and Wenyong Yu. Exploring intrinsic normal prototypes within a single image for universal anomaly detection. InIEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9974–9983, 2025. 2, 5

中文题名：探索单张图像内的内在正常原型以实现通用异常检测。

[28] Tomoyasu Nanaumi. Protofuse: Layer-aggregated and multi-scale prototype scoring for auxiliary-trained zero-shot industrial anomaly segmentation.https://github. com / riips / VAND4 . 0 _ Industrial _ Zeroshot _ ProtoFuse / blob / main / technical _ report . pdf, 2026. Accessed: 2026-06-26. 5

中文题名：ProtoFuse：辅助训练零样本工业异常分割的层聚合与多尺度原型评分。

[29] OpenAI. GPT-4o system card.arXiv preprint arXiv:2410.21276, 2024. 6

中文题名：GPT-4o 系统卡。

[30] Maxime Oquab, Timoth ´ee Darcet, Theo Moutakanni, Huy V . V o, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, ShangWen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision.arXiv preprint arXiv:2304.07193, 2023. 6, 8

中文题名：DINOv2：无监督学习稳健视觉特征。

[31] Qwen Team. Qwen3.5-Omni technical report.arXiv preprint arXiv:2604.15804, 2026. 6

中文题名：Qwen3.5-Omni 技术报告。

[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. InProceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 2, 8

中文题名：从自然语言监督学习可迁移视觉模型。

[33] Lukas Roming, Felix Lehnerer, Jonas V . Funk, Andreas Michel, Georg Maier, Thomas L ¨angle, and J ¨urgen Beyerer. Superadd: Training-free class-agnostic anomaly segmentation – cvpr 2026 vand 4.0 workshop challenge industrial track, 2026. 5

中文题名：SuperADD：免训练、类别无关异常分割——CVPR 2026 VAND 4.0 工作坊挑战赛工业赛道。

[34] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Sch¨olkopf, Thomas Brox, and Peter V . Gehler. Towards total recall in industrial anomaly detection. InIEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14298–14308, 2022. 1, 2, 8

中文题名：迈向工业异常检测的完全召回。

[35] Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn, and Bastian Wandt. Asymmetric student–teacher networks for industrial anomaly detection. InIEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2591–2601, 2023. 2

中文题名：用于工业异常检测的非对称学生—教师网络。

[36] Durga Sandeep Saluru. Diversity by input view: A twomember dinov3 vit-l ensemble for the kaputt 2 retail defect detection challenge.https : / / github . com / durga - sandeep / kaputt - defect - detection - cvpr2026 / blob / main / techical _ report . pdf, 2026. Accessed: 2026-06-26. 6

中文题名：通过输入视图获得多样性：Kaputt 2 零售缺陷检测挑战赛的双成员 DINOv3 ViT-L 集成。

[37] Philipp Seeb ¨ock, Jos´e Ignacio Orlando, Thomas Schlegl, Sebastian M. Waldstein, Hrvoje Bogunovi´c, Sophie Klimscha, Georg Langs, and Ursula Schmidt-Erfurth. Exploiting epistemic uncertainty of anatomy segmentation for anomaly detection in retinal OCT.IEEE Transactions on Medical Imaging, 39(1):87–98, 2020. 1

中文题名：利用解剖结构分割的认知不确定性检测视网膜 OCT 异常。

[38] Oriane Sim ´eoni, Huy V . V o, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha ¨el Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timoth´ee Darcet, Th´eo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herv ´e J ´egou, Patrick Labatut, and Piotr Bojanowski. DINOv3.arXiv preprint arXiv:2508.10104, 2025. 5, 6

中文题名：DINOv3。

[39] Xu Tan, Jie Guo, Haidong Gao, Yang Yu, Qi Jiang, and Ronghua Liang. A pure-vison framework with self-prototype reconstruction for zero-shot anomaly detection.https:// github.com/wwwlin0213/SPR/blob/main/SPR. pdf, 2026. Accessed: 2026-06-26. 5, 7

中文题名：具有自原型重建的纯视觉零样本异常检测框架。

[40] Xian Tao, Hengliang Luo, Mengqi Song, Zhen Qu, Dingrong Wang, Fei Shen, and Zhengtao Zhang. Off-the-shelf vlm ensemble with logprob-based confidence scoring for zeroshot retail anomaly detection.https://github.com/ dorianhenning/VAND2026_Kaputt_OTS/blob/ add - technical - report / technical _ report . pdf, 2026. Accessed: 2026-06-26. 6

中文题名：利用基于 logprob 的置信评分进行现成 VLM 集成，实现零样本零售异常检测。

[41] Xian Tao, Hengliang Luo, Mengqi Song, Zhen Qu, Dingrong Wang, Fei Shen, and Zhengtao Zhang. Multi-model ensemble with semi-supervised and multi-task learning for visual defect detection in kaputt dataset.https://github. com/dorianhenning/VAND2026_Kaputt/blob/ add - technical - report / technical _ report . pdf, 2026. Accessed: 2026-06-26. 6

中文题名：基于半监督与多任务学习的多模型集成，用于 Kaputt 视觉缺陷检测。

[42] Tran Dinh Tien, Anh Tuan Nguyen, Nguyen Hoang Tran, Ta Duc Huy, Soan T.M. Duong, Chanh D. Tr. Nguyen, and Steven Q. H. Truong. Revisiting reverse distillation for anomaly detection. InIEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24511– 24520, 2023. 2

中文题名：重新审视用于异常检测的逆向蒸馏。

[43] Rahul Vippala. Weighted per-category vlm ensembling for few-shot retail anomaly detection: A gemini-based offthe-shelf submission to vand 4.0 (kaputt2).https:// github . com / rvi876 / vand / blob / main / tech _ report.pdf, 2026. Accessed: 2026-06-26. 6

中文题名：用于少样本零售异常检测的逐类别加权 VLM 集成：面向 VAND 4.0（Kaputt2）的 Gemini 现成模型方案。

[44] Junhui Wan1, Hanyuan Chen, Fengliang Qi, Yuchen Hu, Qiuhui Shi, and Tao Fang. Prompt-driven visual anomaly detection for retail logistics via off-the-shelf vlms.https:// github.com/durga- sandeep/kaputt- defectdetection - cvpr2026 / blob / main / techical _ report.pdf, 2026. Accessed: 2026-06-26. 6

中文题名：通过现成 VLM 实现提示驱动的零售物流视觉异常检测。

[45] Chengjie Wang, Wenbing Zhu, Bin-Bin Gao, Zhenye Gan, Jianning Zhang, Zhihao Gu, Shuguang Qian, Mingang Chen, and Lizhuang Ma. Real-IAD: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection. arXiv preprint arXiv:2403.12580, 2024. 2, 3

中文题名：Real-IAD：用于多功能工业异常检测基准的真实世界多视角数据集。

[46] Xingao Wang, Shuying Xia, Zhaohong Liao, Mengjie Xie, Handa Wang, and Zhi Gao. Accurate anomaly localization in challenging industrial settings via a hybrid detection framework.https://github.com/ISVL119/isvl/ blob/main/vand3.0.pdf, 2025. Accessed: 2026-0622. 5

中文题名：通过混合检测框架在困难工业设置中准确定位异常。

[47] Xingao Wang, Zijun Zhou, Yibo Zhang, Zihe Xiao, Zifan Shen, Cheng Tao, Xiaonan Wang, and Tiancan Mei. Isvl++: Towards a unified framework for accurate anomaly localization in challenging industrial settings.https : //github.com/ISVL119/isvl-plusplus/blob/ main/isvl++.pdf, 2026. Accessed: 2026-06-22. 5

中文题名：ISVL++：迈向困难工业设置下准确异常定位的统一框架。

[48] Haonian Xie. Multi-task distilled dinov2 for retail visual defect detection.https : / / github . com / dorianhenning / CVPR2026 _ Retail _ Track _ Solution / blob / add - technical - report/technical_report.pdf, 2026. Accessed: 2026-06-26. 7

中文题名：用于零售视觉缺陷检测的多任务蒸馏 DINOv2。

[49] Vitjan Zavrtanik, Matej Kristan, and Danijel Sko ˇcaj. DRAEM — A discriminatively trained reconstruction embedding for surface anomaly detection. InIEEE/CVF International Conference on Computer Vision (ICCV), pages 8330–8339, 2021. 2

中文题名：DRAEM：用于表面异常检测的判别式训练重建嵌入。

[50] Vitjan Zavrtanik, Matej Kristan, and Danijel Skoˇcaj. DSR — A dual subspace re-projection network for surface anomaly detection. InComputer Vision – ECCV 2022, pages 539– 554, Cham, 2022. Springer Nature Switzerland. 2

中文题名：DSR：用于表面异常检测的双子空间重投影网络。

[51] Huaiyuan Zhang, Hang Chen, Yu Cheng, Shunyi Wu, Linghao Sun, Linao Han, Zeyu Shi, and Lei Qi. Superad: A training-free anomaly classification and segmentation method for cvpr 2025 vand 3.0 workshop challenge track 1: Adapt & detect, 2025. 5

中文题名：SuperAD：面向 CVPR 2025 VAND 3.0 工作坊挑战赛赛道 1“适应与检测”的免训练异常分类与分割方法。

[52] Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. Spot-the-difference self-supervised pretraining for anomaly detection and segmentation. InComputer Vision – ECCV 2022, pages 392–408, Cham, 2022. Springer Nature Switzerland. 1, 2, 3

中文题名：用于异常检测与分割的找不同自监督预训练。

## 补充材料 A：效率指标

图像 AD 模型通常单独报告 F₁ 等检测质量，而不结合计算成本。但实际部署中吞吐、内存、功耗同样决定性。下面定义单一标量，将性能与三种归一化效率量结合。

**设计目标：**

1. **可解释性。**分数处于 [0,1]，1 表示完美效率。
2. **均衡敏感性。**任何单一部分不应主导。任一维改进应反映于最终值，任一维严重不足应惩罚模型。
3. **吞吐收益递减。**超过例如 60 fps 的服务目标，额外速度价值递减。吞吐理论无上界，不像内存和功耗，因此需要饱和归一化。
4. **单调性。**性能、吞吐增加时分数单调上升，内存、功耗增加时单调下降。

### A.1 指标定义

原始测量均映射至 [0,1]，使四者可比较。

**性能。**SegF₁,avg、AP_any 等已经位于 [0,1]，本挑战无需再变换：

$$q_{\mathrm{industrial}}=\mathrm{SegF}_{1,\mathrm{avg}},\quad q_{\mathrm{industrial}}\in[0,1].\tag{7}$$

$$q_{\mathrm{retail}}=\mathrm{AP}_{\mathrm{any}},\quad q_{\mathrm{retail}}\in[0,1].\tag{8}$$

**吞吐。**原始 t 单位帧/秒，以目标 t_max 作收益递减归一化：

$$\tau=\frac{t}{t+t_{\max}},\quad t\ge0.\tag{9}$$

t_max 是例如 60 fps 的服务目标。此映射单调有界，自然软限制吞吐；已很快的模型翻倍，收益低于慢模型翻倍。

**内存。**m 为峰值 GPU 内存（MB），相对设备预算线性归一化：

$$\mu=\frac{m}{m_{\max}},\quad\mu\in[0,1].\tag{10}$$

越小越好，综合分数用 1−µ。超预算自然引发内存不足，故无法测 m。

**功耗。**p 为平均功率（瓦），相对预算归一化：

$$\pi=\frac{p}{p_{\max}},\quad\pi\in[0,1].\tag{11}$$

与内存一样，较低更优，用补量 1−π。

**综合效率。**四个分量以几何平均结合：

$$E=[q\tau(1-\mu)(1-\pi)]^{1/4}.\tag{12}$$

代入定义得：

$$E=\left[q\frac{t}{t+t_{\max}}\left(1-\frac{m}{m_{\max}}\right)\left(1-\frac{p}{p_{\max}}\right)\right]^{1/4}.\tag{13}$$

按此表达，满足吞吐服务级别协议的理想模型分数约为：

$$E_{\mathrm{ideal}}\approx0.841.\tag{14}$$

**选择几何平均的理由。**相对算术平均有三点。第一，强制零值湮灭：任何分量为零，例如 SegF₁,avg=0，或占满内存/功率预算，综合即为零；某一维完全失败确实可能让模型不可用。

第二，鼓励均衡权衡，惩罚不均衡。例如分量为

$$(0.9,0.9,0.9,0.9)\tag{15}$$

的模型，分数高于

$$(1.0,1.0,1.0,0.6),\tag{16}$$

即 0.9 对约 0.88，尽管算术平均均为 0.9。这样鼓励全面模型而非只擅长部分维度。

第三，保持输出有界。每因子都在 [0,1]，几何平均亦如此，无需 sigmoid 或 min–max 再缩放即可解释。

### A.2 参考工作点

表 7 所用参考最大值代表中档边缘部署场景。

**表 7：参考工作点参数。**

| 参数 | 数值 | 单位 |
| --- | --- | --- |
| t_max，目标吞吐 | 30 | fps |
| m_max，内存预算 | 62470.692864 | MB |
| p_max，功率预算 | 56.21 | W |

### A.3 实验分析

通过控制扫描分析指标：一分量独立变化，其余三个保持理想：

$$q=1,\quad\tau=1,\quad\mu=0,\quad\pi=0.\tag{17}$$

#### A.3.1 单分量敏感性

图 2 分别将四个分量由最差扫描至最佳。性能 q 体现四次方根增长：

$$\mathrm{EfficiencyScore}\propto q^{1/4}.\tag{18}$$

初始梯度陡，低性能端的小改进也能明显提高分数，恰当地反映 q=0.3 比 q=0.1 有用得多。

吞吐因公式 (9) 而饱和。对选定服务级别协议，最大可能吞吐贡献为：

$$0.5^{1/4}\approx0.84,\tag{19}$$

体现纯速度不能主导的设计意图。内存功耗则随归一化资源占用增加而单调降低，用满预算时为零。

**图 2：各分量单独变化时效率敏感性。**(a) F₁：四次方根增长，小幅改善也明显提高得分；(b) 吞吐：收益递减使曲线软限制于远低于 1；(c) 内存：用补量 1−µ，随占用增长下降；(d) 功耗：用补量 1−π，随功耗增长下降。

#### A.3.2 综合扫描

图 3 同时将四者由最差扫描到最佳：F₁、吞吐线性增加，内存与功耗线性下降。

**图 3：整个工作范围的综合效率。**近似线性响应确认没有单分量主导。

所得曲线近似线性，确认指标在四维间均匀分配敏感性。

#### A.3.3 F₁—吞吐联合曲面

图 4 为 F₁、吞吐共同变化，内存和功耗固定于各自预算 50%：

$$\mu=0.5,\quad\pi=0.5.\tag{20}$$

**图 4：中等资源使用时，F₁ 与吞吐上的效率曲面。**内存=50%、功耗=50%。凹曲面表明检测质量与推理速度之间的均衡权衡。

曲面表明高综合分数需要良好检测质量和合理吞吐；仅一轴出色不够。

### A.4 讨论

#### A.4.1 可配置参考点

t_max、m_max、p_max 特意设为用户可配置常数。云 GPU、边缘 CPU、FPGA 等部署目标可按约束设定，无需改变指标函数形式。

#### A.4.2 可扩展性

可加入参数量、延迟抖动等维度，将几何平均扩展为 n 因子、指数改为 1/n。对归一化 c₁,…,cₙ：

$$\mathrm{EfficiencyScore}=\left(\prod_{i=1}^n c_i\right)^{1/n}.\tag{21}$$

### A.5 局限

该指标有若干局限。首先，几何平均赋四者等权；若检测质量比效率更重要，可用加权几何平均：

$$E_w=\prod_{i=1}^{4}c_i^{w_i},\qquad\sum_{i=1}^{4}w_i=1.\tag{22}$$

cᵢ 为归一化分量，wᵢ 为用户权重。

其次，吞吐归一化在 0.5 后饱和。若应用极重速度，可采用对数等其他次线性归一化。

> 译注：附录有关吞吐“最大贡献 0.5¹⁄⁴”和“0.5 后饱和”的文字照原文保留。由公式 (9)，t=t_max 时 τ=0.5，而 t→∞ 时 τ→1；前述数值对应达到服务目标的点，不是公式在无限吞吐下的全局上限。正文 GPU 内存定义与实际边缘硬件说明亦按原文保留。
