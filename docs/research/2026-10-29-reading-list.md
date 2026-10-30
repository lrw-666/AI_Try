# 当前开题路线的阅读清单与实验产出

创建日期：2026-10-29；基础选读补充日期：2026-10-30。论文与资料入口按对应更新核对；本文整理阅读入口，不附本地 PDF。

配套文档：[研究方向与交接说明](./2026-10-29-research-direction-and-handoff.md)、[截至 11 月 16 日的开题工作计划](./2026-10-29-proposal-work-plan-to-nov16.md)。

**查阅分区：**第 1—6 节用于当前开题主线；第 7—9 节为补充说明与记录工具；文末[第 10 节：空闲时选读的基础论文](#optional-foundations)单独收录长期学习资料，不计入当前必读任务与时间预算。

## 1. 现在只围绕一个问题读

当前任务是：**在有像素标注的 PCB 图像上，完成缺陷/背景二值分割，并比较两种预训练表征路线。**

- 第一模型：预训练 SegFormer-B0，作为可以独立训练、评估和解释的有监督分割基线。
- 第二模型：冻结 DINOv3 ViT-S/16 编码器，接简单分割头，作为独立对照。是否做轻量适配，由误差和资源记录决定。
- 双编码器特征融合只是后续假设，当前不据此组织必读论文，也不把它写成已经证实有效的方案。
- 数据先考察 VisA 的 PCB1 与官方 2-class high-shot 协议；PCB2 或其他类别用于后续扩展。板级图像是否符合导师期待的“电子元器件”对象，仍需确认。

已经读完的 5 篇综述是背景积累。现在的主要缺口是数据协议、模型输入输出和完整实验闭环，不是再补一长串方法名称。目前尚无本路线完整训练结果，阅读不能替代结果，也不能据此宣称创新性或工业适用性。

结合在职非全日制、基础 PyTorch 实践、RTX 4060、Python 优先及毕业和求职目标，建议采用“**3 篇核心论文与官方资料 + 1 篇按需基础补课**”。11 月 15 日内部定稿、11 月 16 日材料截止、11 月 16—20 日答辩，不能等全部论文读完再开始实验。以下时间均为个人投入建议，不含长时间训练、环境排错和下载等待，也不保证在指定时间完成。

| 顺序 | 阅读包 | 本次定位 | 建议投入 | 读完后立即留下的东西 |
|---|---|---|---|---|
| 1 | VisA / SPot-the-Difference 论文与官方协议 | 核心：把任务、标签和划分说清楚 | 2—3 小时 | 数据协议表、样本与 mask 叠加图、待导师确认的问题 |
| 2 | SegFormer 论文、作者实现与 HF 文档 | 核心：建立第一模型 | 3—4 小时 | B0 结构草图、输入输出说明、基线配置草案 |
| 3 | ViT 基础 | 按需：token 与空间特征不清楚时再读 | 1—2 小时 | token→网格→像素预测的尺寸推导 |
| 4 | DINOv3 论文、官方模型卡、代码及许可 | 核心：设计第二模型独立对照 | 3—4 小时 | 权重可用性记录、冻结范围、简单头与公平比较清单 |

核心阅读约 8—11 小时；ViT 补课另计。拆成多个晚上，读到能推进当前小实验就先动手。未读完不是不能开始的理由。

这 3 篇是当前启动与理解必读，**不等于开题报告只引用 3 篇**。后续利用已读 5 篇综述回溯与本任务匹配的监督 PCB / 工业缺陷分割研究，补充 1—2 篇对口应用证据，并逐项核对对象、像素标注、划分和代码。异常检测论文或内部 X 射线检测研究，不能直接充当表面缺陷有监督分割的实验依据；现在无需为凑引用数量泛搜或恢复被删 PDF。

## 2. 第一包：VisA 数据与有监督任务协议

**准确书目信息：** Yang Zou 等，*SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*，ECCV 2022；arXiv 首次发布于 2022 年。VisA 是该论文发布的数据集，不能把论文方法 SPD、数据集 VisA 和自己的分割方法混为一项。[论文摘要与书目信息](https://arxiv.org/abs/2207.14315)

**阅读入口：**

- [论文 HTML 正文](https://arxiv.org/html/2207.14315v1)。
- [作者官方仓库 amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)：重点读 README 的 Data description、Data preparation、Metric computation。
- [官方 high-shot 划分文件](https://github.com/amazon-science/spot-diff/blob/main/split_csv/2cls_highshot.csv)及[官方数据准备脚本](https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py)。这是核对图像归属与 mask 处理的依据。

**为什么现在读：** 先确认“训练阶段有没有缺陷标注”和“模型输出是什么”，才能选择方法与评价。论文区分 1-class、2-class high-shot 和低样本设置；high-shot 对每类对象的正常与异常图像分别采用 60%/40% 训练/测试划分。本文选择带缺陷标注的有监督路线，不将其包装为仅用正常样本的异常检测。[论文第 4.2 节](https://arxiv.org/html/2207.14315v1#S4.SS2)

**必读范围：**

1. 摘要、引言中对工业缺陷任务和监督设置的说明。
2. 第 4 节 *Visual Anomaly (VisA) Dataset*，尤其 4.1 数据描述、4.2 协议与指标。
3. 第 5.1 节中 2-class high-shot 的实验设置；附录中指标讨论按需查看。
4. 官方 README、`2cls_highshot.csv` 和 `prepare_data.py`。第 3 节 SPD 自监督预训练机制先略读，不要求复现 SPD。

官方资料包含 PCB1—PCB4 复杂结构对象和像素标注；正常图像的 mask 不单独保存。准备脚本将缺陷 mask 中非零像素统一为 255。因此训练前要显式建立“背景=0、缺陷=1”的映射，并为正常图像生成全零 mask。**不能把文件中的 255 直接当作模型的 ignore label。** 这项标签转换是自己的实现责任。[官方数据说明](https://github.com/amazon-science/spot-diff#data-description)、[官方准备脚本](https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py)

**读后应该能回答：**

- 我做的是整图正常/异常分类、框检测，还是每像素缺陷分割？为什么选择二值分割？
- high-shot 中训练集是否同时含正常与异常图像？本实验的监督信息来自哪里？
- PCB1 图像展示的是完整 PCB、板上局部，还是元器件单体？它与导师对象的关系有哪些尚未确认？
- 官方 train/test 如何保留？验证集从哪里产生？正常图像、异常图像和像素前景不平衡如何记录？
- 为什么不能只看整体像素准确率或 AUROC，就声称小缺陷分得好？

**立刻产出，建议 2—3 小时：**

1. 一页任务与协议说明：对象、输入、输出、标注、官方划分来源、二值化规则、验证集方案。
2. 选取少量正常/异常样本，观察原图与 mask 对齐。后续实际拿到数据后，补全尺寸、mask 唯一值、缺陷像素比例、每个 split 的实际计数；本次文献核对不等于已完成本地数据审计。
3. 将验证集从官方训练部分固定划出并保存名单；这属于本实验增加的验证协议，要报告规则和随机种子。由同一原图产生的裁块必须保留在同一 split，避免泄漏。测试集不参与阈值、超参数或模型选择。
4. 写下当前评价计划：缺陷类 IoU、Dice/F1，补充 precision/recall 与像素级 PR 指标；说明聚合方式、空 mask 的处理和阈值选择。若使用 AP 或梯形积分 AUPRC，记清具体实现，不能仅因名称相近就与原文 AU-PR 数字直接比较。

## 3. 第二包：SegFormer-B0 与第一模型

**准确书目信息：** Enze Xie 等，*SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*，NeurIPS 2021；arXiv:2105.15203。[论文摘要与书目信息](https://arxiv.org/abs/2105.15203)

**阅读入口与实现归属：**

- [论文 HTML 正文](https://arxiv.org/html/2105.15203v3)。
- [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)：论文作者官方 PyTorch 实现，适合查结构、配置和原始实验设定。
- [Hugging Face Transformers 的 SegFormer 文档](https://huggingface.co/docs/transformers/en/model_doc/segformer)：HF 库维护的集成实现，**不是作者仓库本身**；更适合本次 Python 小实验的接口学习。
- [NVIDIA 组织下的 `nvidia/mit-b0` 模型页](https://huggingface.co/nvidia/mit-b0)：B0 编码器预训练权重候选；该页说明模型卡由 HF 团队撰写，不能把其中所有描述都当作作者原始说明。

**为什么现在读：** SegFormer 将分层 Transformer 编码器与轻量 MLP 解码器结合，可用于建立一个完整、独立的分割基线。阅读重点是把结构和自己的数据流对应起来，而不是背诵原论文榜单结果。B0 是当前资源约束下的起点；是否适合本地分辨率和批量，需实际记录显存、速度与精度。[论文第 3 节](https://arxiv.org/html/2105.15203v3)

**必读范围：**

1. 摘要与结构总图。
2. 3.1 *Hierarchical Transformer Encoder*：特征层级、空间尺寸变化、多尺度输出。
3. 3.2 *Lightweight All-MLP Decoder*：不同尺度怎样映射、对齐和输出分割 logits。
4. 4.1 实验设置、4.2 消融及附录 A 的 B0 配置。其他模型规模与鲁棒性实验先略读。
5. HF 文档的预处理说明、`SegformerForSemanticSegmentation`、输出 logits 和标签设置。重点核对版本对应的参数名，不把联网文档当成已安装库的版本证明。

**读后应该能回答：**

- 编码器和分割头分别负责什么？B0 中哪些参数来自预训练，哪些是本实验重新初始化？
- 多尺度特征怎样回到像素预测？logits 是否已经等于原图大小？
- 我的任务只有背景/缺陷两类，如何设置标签、输出通道和损失？
- 图像与 mask 的同步增强怎样实现？为什么 mask 缩放要避免产生虚假的中间类别？
- 使用 ImageNet 编码器预训练还是 ADE20K 分割预训练，会怎样影响实验解释？

**立刻产出，建议 3—4 小时：**

1. 手画一张“图像→MiT-B0 多尺度特征→分割头→上采样 logits→二值 mask”的结构图，用一次实际前向结果填写尺寸。
2. 建立基线配置表：预训练来源、输入尺寸、归一化、增强、损失、优化器、种子、验证规则及评价指标。建议先用两类 logits 明确背景/缺陷语义，再按证据调整。
3. 核对 `do_reduce_labels=False` 的适用性，确保背景 0 参与本任务学习；HF 文档中 ADE20K 的标签缩减示例不能原样套用到自己的二值 mask。[HF 预处理说明](https://huggingface.co/docs/transformers/en/model_doc/segformer)
4. 先做一个 batch 的前向、损失和反向检查，再做少量训练图像的过拟合检查，最后进入完整基线。检查应留下输入、标签、logits 尺寸和可视化，不以“脚本没报错”代替验证。

优先考虑从 MiT-B0 编码器预训练开始、重新训练二类分割头；若选择已有 ADE20K 分割 checkpoint，必须显式记录额外的分割预训练，并重置不匹配的分类头。两种初始化不能混写成同一次实验。`nvidia/mit-b0` 页面本身是分类编码器入口，不是已经训练好的 PCB 缺陷分割模型。[模型页](https://huggingface.co/nvidia/mit-b0)

## 4. 中间按需补课：ViT，不要求完整复现

**准确书目信息：** Alexey Dosovitskiy 等，*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*。arXiv 首次发布于 **2020 年**，发表于 **ICLR 2021**；arXiv:2010.11929。引用时避免把“首次预印本年份”和“会议年份”混为一谈。[论文记录](https://arxiv.org/abs/2010.11929)

**入口：** [论文 HTML](https://arxiv.org/html/2010.11929v2)、[作者官方仓库 google-research/vision_transformer](https://github.com/google-research/vision_transformer)。官方仓库以 JAX/Flax 为主，本次用于查概念，不要求因此切换技术栈。[官方实现说明](https://github.com/google-research/vision_transformer)

只有出现以下障碍再读：不清楚 patch/token、把 CLS token 当空间特征、不会把 patch tokens 还原网格，或者说不清 patch 大小与小缺陷之间的关系。

**必读范围：** 结构图与 3.1 *Vision Transformer (ViT)*；需要理解分辨率变化时读 3.2；注意力公式不清楚时查附录 A。不要求逐表复现大规模分类实验。[论文正文](https://arxiv.org/html/2010.11929v2)

**读后应该能回答：** 图像怎样变成 token 序列？全局 token 与 patch token 有什么区别？单尺度特征和 SegFormer 的多尺度特征有何不同？为什么特征网格还需要分割头和上采样？

**立刻产出，建议 1—2 小时：** 自己推导 512×512、patch=16 时的 32×32 空间网格，再用选定模型实测验证；写出 `B×N×C` 到 `B×C×H'×W'` 的转换，区分额外 token。这个练习不等于 DINOv3 采用与原始 ViT 完全相同的位置编码与 token 设计，具体接口以 DINOv3 实现为准。

## 5. 第三包：DINOv3 与冻结特征独立对照

**准确书目信息：** Oriane Siméoni 等，*DINOv3*，2025 年技术报告，arXiv:2508.10104。此处不自行补写未核实的会议名称。[论文记录](https://arxiv.org/abs/2508.10104)

**阅读入口与实现归属：**

- [论文 HTML 正文](https://arxiv.org/html/2508.10104v1)。
- [Meta 官方参考实现 facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)。
- [官方模型卡](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)。
- [Meta 在 HF 发布的 ViT-S/16 模型页](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)：目标是 `facebook/dinov3-vits16-pretrain-lvd1689m`，不要误选 S+、大模型或卫星权重。
- [官方 DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)：使用与再分发前阅读；开题材料据实注明权重来源与使用情况。

**为什么现在读：** 论文研究可迁移的视觉表征，包含密集特征的评估，为“冻结编码器后训练任务头”提供了可研究的起点。能否迁移到本任务，仍要通过 PCB 数据验证；它不是拿来即可输出 PCB 缺陷 mask 的现成分割器。论文的通用视觉基准结果不能直接写成自己的 PCB 结果。[论文摘要](https://arxiv.org/abs/2508.10104)

**必读范围：**

1. 引言与整体方法说明，先说清“自监督预训练”和“下游有监督分割”是两个阶段。
2. 第 4 节 *Gram: A Regularization for Dense Features*：了解要解决的密集特征问题与动机即可，当前不要求重训预训练目标。
3. 5.1 分辨率扩展、5.2 模型蒸馏，以及第 7 节模型家族中与 S/16 相关的内容。
4. 6.1.2 *Dense Linear Probing*、6.3.2 语义分割及附录 D.1：重点识别冻结范围、头部与评估协议，不把论文不同设置揉成一个结果。
5. 模型卡的输出、用途、限制与训练数据；README 的权重获取、预处理和加载入口；许可文件。[论文目录与正文](https://arxiv.org/html/2508.10104v1)

官方 S/16 是约 21M 参数的蒸馏模型，图像以 16×16 patch 表示；模型含全局、寄存器和 patch tokens，只有空间 patch tokens 能直接重排为图像网格。应根据具体 API 确认提取方式，不凭经验硬切 token。输入尺寸需与 patch 规则一致，避免边缘被隐式裁掉而造成标签错位。[官方模型家族](https://github.com/facebookresearch/dinov3#pretrained-models)、[模型卡](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)

**开放状态要分别记清：** 参考代码公开；官方权重需要申请或在 HF 登录同意访问条件；使用自定义 DINOv3 License；预训练数据来源在模型卡中有说明，但本轮未查到 LVD-1689M 完整数据的公开下载入口。不能由“代码可见、权重可申请”推导“全部预训练数据完全公开、自己能完整复现预训练”。本次也尚未验证个人账户的权重授权与本地加载。[权重入口](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)、[官方模型卡](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)、[许可](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)

**读后应该能回答：**

- 用的是哪个 checkpoint？来源、授权、代码版本和预处理在哪里记录？
- 输入图像得到什么特征？空间分辨率与原图有何关系？
- “冻结”具体指哪些参数不更新？训练过程中是否确实只有分割头得到梯度并更新？
- 最简单的分割头如何把 patch 特征变成二类 logits？上采样是否真的能恢复丢失的小缺陷信息？
- 与 SegFormer-B0 相比，哪些差异来自编码器、预训练、冻结策略、分割头或分辨率？现有对照能支持多强的结论？

**立刻产出，建议 3—4 小时：**

1. 一张可用性表：权重访问是否通过、下载来源、文件校验、依赖版本、一次前向能否完成。阅读阶段先记录“待验证”，不要填成已完成。
2. 一张结构图：图像→冻结 DINOv3-S/16→patch 特征网格→简单分割头→上采样二类 logits。先考虑 1×1 卷积/线性头，头的结构和参数量完整记录。
3. 做一次特征形状检查、梯度与参数更新检查，以及小样本头部训练，保留正常图误报和小缺陷漏检的例图。
4. 写公平比较清单：同一数据划分、同一测试集、明确的输入尺寸与增强、相同评价定义、同等清晰的模型选择规则；另报可训练参数、峰值显存和推理时间。预训练来源与训练策略不同，结论应是“两套路线的可行性比较”，不能仅凭一次差值断言某种预训练必然更好。

冻结特征没有优势也有研究价值：先排查标签、尺度、头部欠拟合与训练设置，再决定是否尝试轻量适配。增加适配或双分支融合必须有具体失败案例和对照理由，不能因为标题听起来更新就升为主线。

## 6. 阅读与小实验交替推进

本清单初建于 10 月 29 日深夜，执行计划从 10 月 30 日开始。具体日程以配套[开题工作计划](./2026-10-29-proposal-work-plan-to-nov16.md)为准，不因增加选读资料而推迟基线和材料写作。

| 阶段 | 阅读到什么程度就停 | 接着做什么 | 遇到问题回查哪里 |
|---|---|---|---|
| 数据先行 | 能解释任务、mask 与 high-shot | 样本/标注检查，固定划分 | VisA 第 4 节、CSV、准备脚本 |
| 第一模型 | 能解释 B0 数据流与标签设置 | 一个 batch→少量图过拟合→完整基线 | SegFormer 第 3 节、HF 接口 |
| 基础按需 | 能手算 patch 网格 | 检查 token 与空间尺寸 | ViT 第 3 节 |
| 第二模型 | 能解释冻结特征与简单头 | 前向与更新检查→独立训练评估 | DINOv3 模型卡、线性探测协议 |
| 写入开题 | 每项主张都有来源或实验记录 | 写可行性、局限与后续问题 | 回读对应原文，不继续泛搜 |

如果进度紧，先保证数据协议、SegFormer 基线和真实失败案例的证据链；DINOv3 未跑通就写明计划与当前状态，不编造对比结果。11 月 15 日前将已经取得的证据和未完成的工作分开写入材料。

## 7. 哪些资料现在不加为必读

| 资料或路线 | 什么时候才需要 | 当前处理 |
|---|---|---|
| U-Net | 编码器/解码器、跳跃连接等概念仍不清楚，或导师要求传统分割对照 | 按需补课；不要求先训练它才能做 SegFormer |
| 原始 DINO、DINOv2 | DINOv3 训练目标或模型演进阻碍理解 | 只回查相关段落，不要求沿发展史逐篇复现 |
| PatchCore、Dinomaly | 导师改为仅正常样本训练、无像素标注，或主任务重新界定 | 保留备选名称，不扩成必读分支 |
| 双编码器融合、复杂解码器 | 两个独立模型已有结果，误差说明互补性和实验价值 | 未来假设；开题前不强制实现 |
| 直接 PCB 应用论文 | 明确对象、像素标注、可用源码、划分与当前任务匹配之后 | 本清单暂不硬推荐未经核实的应用论文 |
| 更多综述、大规模预训练复现、C++ 部署 | 出现明确知识缺口，或毕业实验已稳固后 | 不占用眼前关键路径 |

## 8. 可复制的单篇笔记模板

每篇控制在一至两页。先凭自己理解填写，再让 AI 帮助查漏；公式、参数、图表数字与关键结论回到原文核对。看不懂的地方保留问号，不把 AI 的推测当作论文原话。

```markdown
# [论文简称]：与当前 PCB 二值分割的关系

- 阅读日期 / 用时：
- 准确标题、作者、年份、会议或技术报告：
- 论文链接 / 作者官方仓库 / 其他库实现：
- 实际使用的代码版本、checkpoint 与许可：

## 我自己的三句话
1. 作者解决什么问题？
2. 核心方法和证据是什么？
3. 本次项目直接使用哪一部分？

## 数据与监督
- 图像对象、标签、训练/验证/测试划分：
- 上游预训练信息与下游训练监督：
- 与我当前任务相同和不同之处：

## 数据流
- 输入形状与预处理：
- 关键中间特征形状：
- 输出、上采样、损失和可训练参数：
- 评价指标、聚合、阈值选择：

## 证据与边界
- 原文支持的主张（章节/图表）：
- 我的理解或假设：
- 尚未确认的问题：
- 不能从本文直接推出的结论：

## 马上做的一个小实验
- 要验证的问题：
- 固定条件与只改变的因素：
- 结果文件、日志或图像位置：
- 实际观察 / 失败原因 / 下一步：

## 可写入开题的两句话
- 已有文献依据：
- 本项目计划验证或已经验证的内容：
```

## 9. 本轮核对结果与仍待实测事项

已核对三篇核心论文与 ViT 的准确标题、首次发布年份及上述会议/报告身份；论文入口均使用摘要页或 HTML，不提供本地 PDF。作者官方仓库与 HF 库实现已分别标注；有效 SegFormer 文档路径包含 `/en/`。

本轮已从本地官方 ZIP 复核 PCB high-shot 的划分计数，具体数字见[研究方向与交接说明](./2026-10-29-research-direction-and-handoff.md)。这不等于完成全部图像解码、实际 mask 像素检查或训练集/验证集派生清单。

外部页面可访问只代表阅读入口有效，不代表依赖安装、权重授权、下载速度或本地实验已经验证。尤其需要后续补证：VisA 图像与 mask 审计、导师是否接受板级 PCB 对象、DINOv3 账户授权和前向运行、目标机器的实测显存与时长、两个模型的完整可重复训练结果。资源与资料可支撑开始实验，不能保证毕业评审结果或某个方法一定提升。

---

<a id="optional-foundations"></a>

## 10. 空闲时选读的基础论文：长期学习资料

补充日期：2026-10-30。以下是独立的课外选读书架，**没有开题前读完的要求，不增加当前三篇核心论文的必读范围，也不要求每篇复现**。可以按兴趣读摘要、结构图和核心方法，等实验遇到对应问题时再回查细节。

选择依据是它们能帮助理解视觉网络、分割训练和预训练模型，而非追求模型年份。表中的“用途”是结合当前任务与后续视觉开发作出的学习建议，不表示相关方法已在本项目验证有效。年份同时注明预印本与常见会议身份，避免引用时混淆。

### 10.1 CNN、分割与多尺度特征

| 论文与入口 | 年份 | 可以学到什么 | 与当前研究或后续实践的关系 |
| --- | --- | --- | --- |
| **ResNet**：[*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) | 预印本 2015；CVPR 2016 | 卷积骨干、残差连接、深层网络为什么需要更容易优化的路径；重点看残差块及恒等映射思路 | 理解常用视觉编码器，也为理解 Transformer 中的残差连接建立直觉。原论文的完整大规模训练不是入门要求 |
| **FCN**：[*Fully Convolutional Networks for Semantic Segmentation*](https://arxiv.org/abs/1411.4038) | 预印本 2014；CVPR 2015 | 从图像分类转向像素预测；全卷积、粗特征上采样、浅层与深层特征结合 | 建立“分类骨干＋分割输出”的基本认识，帮助理解自己的 logits、空间尺寸与像素标签 |
| **U-Net**：[*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597) | MICCAI 2015 | 编码器、解码器、跳跃连接、上下文与定位信息的结合；少量标注下的数据增强思路 | 对当前分割概念最直观的补课之一；原任务是生物医学图像，迁移到 PCB 的效果需要自己的实验 |
| **FPN**：[*Feature Pyramid Networks for Object Detection*](https://arxiv.org/abs/1612.03144) | 预印本 2016；CVPR 2017 | 自顶向下路径、横向连接、多分辨率特征及语义信息结合 | 理解多尺度、小目标/小缺陷和特征融合的常见思路。原文是目标检测，不把其成绩直接当作 PCB 分割依据 |

代码/图示参考：[U-Net 作者主页与原始实现](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)、[Torchvision ResNet 接口](https://docs.pytorch.org/vision/stable/models/resnet.html)。U-Net 原始发布包含 Caffe/Matlab 工具；Torchvision 是 PyTorch 官方库实现，不等同于论文作者逐行原始代码。当前主要用 Python 学概念即可，不必复刻老环境。

### 10.2 注意力机制与 Transformer

| 论文与入口 | 年份 | 可以学到什么 | 建议阅读重点 |
| --- | --- | --- | --- |
| **Bahdanau Attention**：[*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473) | 预印本 2014；ICLR 2015 | 根据当前需要，对不同输入位置分配权重，再汇总信息的基本思想 | 看对齐/注意力示意和加权求和。原任务是机器翻译，用于建立概念，不要求补完 RNN 翻译训练 |
| **Transformer**：[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) | NeurIPS 2017 | Q/K/V、缩放点积注意力、多头注意力、位置编码、残差与归一化 | 先读结构图和第 3 节，手算一组小张量尺寸；这是理解 ViT、SegFormer 和 DINOv3 的重要基础 |
| **SE / SENet**：[*Squeeze-and-Excitation Networks*](https://arxiv.org/abs/1709.01507) | 预印本 2017；CVPR 2018 | 通道注意力：根据图像内容调整各特征通道的权重 | 看 squeeze、excitation 和通道重标定；适合学习一个小模块怎样嵌入已有 CNN |
| **CBAM**：[*CBAM: Convolutional Block Attention Module*](https://arxiv.org/abs/1807.06521) | ECCV 2018 | 通道与空间注意力，以及怎样组织轻量特征重加权模块 | 看两种注意力的计算图与消融；理解设计动机，不把“加注意力模块”本身当成充分创新 |

**区分概念：**SE/CBAM 主要对通道或空间位置作特征重加权；Transformer 的自注意力通过 Q/K/V 建模 token 之间的关系。它们都属于注意力相关技术，但计算方式、作用位置和代价不同。ViT 是将 Transformer 用于图像的一类架构，不是与“注意力”并列且互不相关的技术。

### 10.3 视觉 Transformer 与自监督表征

ViT 的原论文链接、作者实现与尺寸练习已在第 4 节单列：[ViT 论文](https://arxiv.org/abs/2010.11929)。如果只想先了解视觉 Transformer 的来龙去脉，可以从那里开始，无需重复安排一项阅读任务。

| 论文与入口 | 年份 | 可以学到什么 | 与当前研究的连接 |
| --- | --- | --- | --- |
| **Swin Transformer**：[*Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*](https://arxiv.org/abs/2103.14030) | ICCV 2021 | 窗口注意力、移位窗口、层级特征；高分辨率视觉任务中的计算与表达权衡 | 对比标准 ViT、Swin 与 SegFormer 的结构差异。SegFormer 并不是直接采用 Swin，也不能将窗口机制当作所有视觉 Transformer 的共同设计 |
| **DINO**：[*Emerging Properties in Self-Supervised Vision Transformers*](https://arxiv.org/abs/2104.14294) | ICCV 2021 | 无人工类别标签的自蒸馏、教师—学生、不同视角一致性和视觉表征 | 回溯 DINOv3 的技术来源；先理解预训练如何获得特征，再区分自己的有监督下游训练 |
| **MAE**：[*Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377) | 预印本 2021；CVPR 2022 | 遮住大量图像 patch 后学习重建，非对称编码器/解码器，以及自监督预训练的另一条思路 | 与 DINO 的学习目标作对比；原始 MAE 主要重建像素，不能与 DINO 的表征/自蒸馏目标混为一谈 |

这组论文用于建立“结构怎样设计、特征怎样学出来”的区别。多层 ViT 特征通常仍在同一 patch 网格上，不能仅因抽取多个层，就称为 FPN/SegFormer 式的多分辨率金字塔。

### 10.4 训练、类别不平衡与轻量适配

| 论文与入口 | 年份 | 可以学到什么 | 在实践中怎样使用这些认识 |
| --- | --- | --- | --- |
| **Adam**：[*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980) | 预印本 2014；ICLR 2015 | 梯度一阶/二阶矩估计、自适应更新、学习率与优化器状态 | 看算法伪代码并对照 PyTorch 参数；先理解更新过程，收敛理论可按需深入 |
| **AdamW**：[*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) | 预印本 2017；ICLR 2019 | 解耦权重衰减，以及它与在自适应优化器中加入 L2 项的区别 | 帮助理解 Transformer 训练配置中的学习率、权重衰减和参数分组；不靠盲试所有超参数找最好测试分数 |
| **Focal Loss**：[*Focal Loss for Dense Object Detection*](https://arxiv.org/abs/1708.02002) | ICCV 2017 | 降低易分类样本的损失贡献；前景/背景不平衡与困难样本的关系 | 对缺陷像素稀少有启发；原文是 RetinaNet 检测实验，分割任务中是否优于交叉熵/Dice 需验证 |
| **LoRA**：[*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) | 预印本 2021；ICLR 2022 | 保留基础权重，通过可训练的低秩增量适配模型，理解参数高效微调 | 有助于以后使用基础模型。原论文面向语言模型，迁移到视觉需核对层和实现；与“完全冻结骨干、只训练头”是两种不同策略 |

接口参考：[PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)、[Torchvision sigmoid focal loss](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html)。后者面向匹配形状的二元 logits/标签，不能不检查输出与标签格式就直接替换当前两类 softmax 交叉熵。

### 10.5 未来兴趣拓展：多模态与扩散

| 论文与入口 | 年份 | 可以学到什么 | 阅读边界 |
| --- | --- | --- | --- |
| **CLIP**：[*Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020) | ICML 2021 | 图像—文本对比学习、共享表示空间、零样本分类与迁移 | 帮助理解图文检索和多模态视觉模型；它使用图文监督，与 DINO 的纯图像自监督设定不同。通用零样本识别不能直接保证工业瑕疵检出 |
| **DDPM**：[*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239) | NeurIPS 2020 | 前向加噪、学习去噪、反向生成，以及扩散模型的基本训练目标 | 适合了解生成式视觉的基础。先读流程和主要目标，再看推导；当前不新增扩散缺陷检测实验 |

### 10.6 怎样挑选，不把闲读变成负担

- **如果只先挑 4 篇：**U-Net → ResNet → Attention Is All You Need → DINO。这是结合当前分割与 DINOv3 路线的建议；ViT 概念不熟时，在 DINO 前回看第 4 节。
- **如果想理解融合与小缺陷：**FCN → FPN → Swin；关注空间尺寸、语义信息和计算成本，而不只是模型名称。
- **如果想改善训练理解：**Adam/AdamW → Focal Loss；先能解释当前优化器与损失，再做受控调整。
- **如果以后想接触基础模型应用：**MAE → LoRA → CLIP；对生成图像更有兴趣时再读 DDPM。这些分支按兴趣选择，不要求按顺序全部完成。

每次闲读可以只做 20—40 分钟的初览：读摘要、看主图，用自己的话写下“解决什么问题、核心操作是什么、与我任务有什么联系”三句话。这个时长仅用于建立印象，不代表能完整读懂论文。遇到关键概念再安排深入阅读或小张量练习，无需把每篇论文都变成一个新项目。
