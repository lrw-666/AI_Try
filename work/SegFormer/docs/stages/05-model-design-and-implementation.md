# 阶段 05：预训练权重与模型——设计与实现学习文档

| 字段 | 内容 |
| --- | --- |
| 状态 | 验收通过：固定来源、真实加载、人工张量 GPU 前后向与一次调试更新已验证 |
| 更新日期 | 2026-10-31 |
| 工程 / 资源 | `E:\桌面\AI\AI_Try\work\SegFormer` / `E:\桌面\AI\segformer_resources` |
| 前置 | 阶段 04 的 v2 分组清单、统一几何与配置 |
| 结论入口 | [第五阶段验收报告](../../reports/2026-10-31-stage05-model-loading-and-check.md) |
| 边界 | 没有 Dataset、真实图训练、正式损失或 IoU/Dice；本次不进入阶段 06 |

## 1. 本阶段要证明什么

目标不是“模型对象能建立”，而是逐层证明：拿到的是同一份预训练快照；encoder 真正继承它的权重；二类分割输出尺寸正确；有效区损失能反向传播到编码器及解码头；一次调试更新不污染磁盘快照；batch 2 能完成该范围内的 GPU 运算。

下载、加载和通路验证分为三个模块。先在 CPU 上校验来源并加载，移到 GPU 后先测 batch 1、再测 batch 2，最后做一次人工数据的调试更新。正式训练器仍留到后续阶段。

## 2. 技术栈与版本

| 技术 | 实际版本/配置 | 用途 |
| --- | --- | --- |
| Python | 3.12.16，独立 Conda prefix | CLI 与报告 |
| torch | 2.10.0+cu126 | 权重、张量、autograd、AdamW、CUDA 显存 |
| transformers | 4.57.6 | SegformerForSemanticSegmentation、SegformerImageProcessor |
| huggingface_hub | 0.36.2 | 固定 commit 下载 |
| GPU | RTX 4070 Laptop，8187.5 MiB | 实际前后向 |
| 当前探测精度 | FP32、AMP=False | 记录独立于正式训练 AMP 的测量口径 |
| unittest | 全项目 101 项通过 | 下载、加载、拒绝路径与既有阶段回归 |

沿用环境，未安装或升级任何包。快照只使用 `pytorch_model.bin`，以 `torch.load(weights_only=True)` 在 CPU 读取；safetensors 分支保留，但本次实际未触发。

## 3. 文件职责与数据流

| 文件 | 接口与职责 |
| --- | --- |
| `download_model.py` | `download` / `verify_snapshot` / `validate_source`；固定来源、下载、离线校验、既有目录保护及当前进程代理 |
| `modeling.py` | `load_all`；验证来源、仅本地加载 processor/模型、核对 encoder 张量与加载键、分类头和可训练参数 |
| `model_check.py` | `make_batch`、`measure_memory`、`runtime_checks`；人工标签、B1/B2 显存、前后向、调试更新和验收条件 |
| `tests/test_modeling.py` | 原有键分类、LFS、完整快照、人工标签测试；fixture 已升级为合法完整来源记录 |
| `tests/test_download_model.py` | 15 项新增离线回归：固定版本、污染拒绝、来源重建、代理不泄露等 |
| `tests/test_model_acceptance.py` | 10 项新增回归：精确权重、未知键、来源、形状/有限值门槛、256 几何与 CLI 尺寸拒绝 |
| `reports/model_source.json` | 原始 source.json 的小型副本，作为跨机来源记录 |
| `reports/model_download.json` | 原始真实下载结果，保留当时请求 main 及解析 commit |
| `reports/model_snapshot_check.json` | 本轮按仓库来源记录执行的真实离线校验 |
| `reports/model_loading.json` / `model_check.json` | 当前加载与 GPU 结果 |
| `reports/stage05-acceptance.json` | 验收、代码/配置/报告哈希及 v2 数据身份 |

```text
仓库来源记录：model ID + commit + 全文件大小/SHA
            ↓
下载到资源目录（既有固定目录不允许静默换版）
            ↓
来源/文件校验 → local_files_only 加载 → encoder 逐张量精确比较
            ↓
五项加载验收：权重 / 键 / processor / 二类头 / 可训练参数
            ↓
seed42 → GPU B1前后向峰值 → GPU B2前后向峰值
            ↓
人工 B2 → logits → 512上采样 → ignore CE vs 有效区CE
            ↓
抽查两处梯度 → AdamW一次内存更新 → 抽查参数位移
            ↓
全部快照文件及source前后SHA → 九项运行验收
```

## 4. 为什么这样加载

### 4.1 MiT-B0 checkpoint 与 SegFormer 的关系

`nvidia/mit-b0` 是 ImageNet 分类预训练的 MiT 编码器 checkpoint，包含 192 个 encoder 张量及两项原分类头。分割模型保留 encoder，另建 SegFormer 解码头和二类 1×1 卷积。

```text
原 checkpoint：encoder（3,319,392参数） + classifier（257,000参数）
分割模型：    encoder（沿用预训练）       + decode_head（395,266参数）
```

总模型为 3,714,658 参数，全部可训练。新解码头需要后续用分割数据训练，不能仅凭加载成功直接拿随机解码头评估缺陷效果。

### 4.2 三层证据缺一不可

1. **文件层：**固定 commit 与完整大小/SHA。空 files、没有登记的权重、额外 shadow 权重等都拒绝。
2. **框架加载层：**记录完整 missing/unexpected/mismatched。当前 16 项 missing 全是新 decode_head；unexpected 只有两个原 classifier；mismatched=0。
3. **张量层：**按名字、形状、dtype 和数值核对每个 encoder 张量。当前 192/192 FP32 精确相等，max_abs_diff=0，不接受原先 1e-6 的近似放行。

```python
problems = verify_snapshot(model_dir, source)
if problems:
    raise ValueError(...)
# 之后才读 processor 和模型。
```

然后比较 `segformer.encoder.x` 与 checkpoint 的 `encoder.x`，不按迭代顺序盲目对齐。未使用的 checkpoint 键完整列出，必须只有原 `classifier.weight` 和 `classifier.bias`，不能截断列表后隐藏异常。

source.json 是本地来源清单，SHA 证明文件与清单一致；它不等于第三方数字签名。跨机用仓库保存的 `--source-record` 再对照完整清单，不重新生成一份来源来掩盖损坏。

### 4.3 missing、unexpected、mismatched 的区别

- missing：目标模型有、checkpoint 没有；本次新解码头正常出现这种情况。
- unexpected：checkpoint 有、目标模型没有；本次原 ImageNet 分类头被丢弃。
- mismatched：同一键两边形状不同；本次没有。

`ignore_mismatched_sizes=True` 可能让部分形状不匹配参数随机初始化，不能当作权重验证。当前分类 checkpoint 没有分割头，不是“1000 类分割头改为 2 类”的形状冲突；旧文档关于 mismatched_other 一定报错的说法已修正。

## 5. Processor、几何与损失的边界

processor 当前配置：resize=False、reduce_labels=False、rescale=True、normalize=True，mean `[0.485,0.456,0.406]`、std `[0.229,0.224,0.225]`。

阶段 06 将把已经 letterbox 的 uint8 RGB 交给它，执行一次除以 255 和归一化。这里禁用 resize 是让几何只由项目管理，防止未来配置变更造成二次缩放。当前若输入已经是 512×512 画布，默认同尺寸 resize 并不会自动把有效区 512×390 单独拉伸；旧稿对此表述不准确。

本次主检查直接使用人工标准正态张量，不经过真实图 processor，因此不能说真实数据归一化已经验证。该验收属于阶段 06。

MiT 四个编码阶段的典型 stride 为 4/8/16/32；解码器融合后输出 stride 4，所以当前 `[B,3,512,512] → [B,2,128,128]`。再用 `F.interpolate(..., align_corners=False)` 明确上采样到标签尺寸后计算损失。

标签按当前几何生成：512 时上下 61 行为 255，区域内含 0/1；256 合成调试时改为上 30、下 31，缺陷坐标同步缩放。正式 CLI 必须与 baseline 的 512 一致。

两种 CE 相等说明忽略标签没有直接计入损失分子或分母。不能反过来说每一次 CE 不等就一定是 padding；也需检查类别权重、reduction、裁切位置及浮点误差。图像 padding 本身仍可能通过上下文影响有效区的预测。

## 6. 显存与优化器检查为什么分开

原实现的顺序是：B2 主调试更新 → B1/B2 probe，AdamW 的状态和部分主张量仍在，因此旧 570/1058 MiB 不能称为“无优化器状态的隔离前后向”。

现在先 B1 再 B2，在任何调试 optimizer 和主图创建之前运行；每次清除梯度/可释放缓存、重置峰值，记录 allocated 与 reserved。probe 不做优化器 step；之后才用人工 B2 检查实际梯度与更新。

| batch | 当前 allocated | 当前 reserved |
| ---: | ---: | ---: |
| 1 | 426.6 MiB | 504.0 MiB |
| 2 | 846.8 MiB | 1008.0 MiB |

allocated 为 Torch 张量实际分配；reserved 是其缓存分配器保留量，包含可复用空闲块。两者都不是整机真实占用，也不包括所有 CUDA 上下文/驱动开销；不能只凭总显存比例保证不会 OOM。

接受 batch=2 起步。后续训练还会有组合损失、AdamW 状态、AMP、保存/验证等额外路径，预算以阶段 08 的实际 20 步为准。DataLoader 的主要额外开销在主存／可能的页锁定内存，不能笼统算成 GPU 显存。

如果确需 batch=1、累积=2，应写新配置并记录有效每次更新图数、scheduler/optimizer更新计数；BatchNorm 与随机过程会使它不完全等同于物理 batch=2。

## 7. 实际执行与当前证据

| 验证 | 当前实测 |
| --- | --- |
| 快照离线校验 | 按 reports/model_source.json 全文件核对，problems=[]；本轮未重下权重 |
| 编码器 | 192/192 exact，max_abs_diff=0，unused仅两项classifier |
| 加载验收 | 5 项全通过 |
| 主前向 | logits `[2,2,128,128]`；标签0/1/255 |
| CE | 两路均0.6495407223701477，差0.0≤1e-6 |
| 反向抽样 | encoder norm 0.11584833；head norm 0.21285953；均有限非零 |
| 一次更新抽样 | 两参数 max_abs_delta约1.0000169e-4 |
| 快照完整性 | config、processor、权重及source SHA前后不变 |
| 运行验收 | 9 项全通过 |
| 项目回归 | 101 项通过 |

旧 CE 差5.96e-08、旧1058 MiB保存在 `reports/stage05_pre_acceptance/model_check.json`，不是当前验收结果。当前在初始化前固定seed42，且改变probe顺序，随机头/随机训练运算的序列变化会改变绝对loss；验收看形状、有限性和容差，而非追逐旧小数。

本阶段只抽查 encoder 与 decode_head 各一个参数的梯度及位移；报告明确scope，未声称整网逐参数审计。调试模型没有保存回预训练目录。

## 8. 操作指令与代理约定

每个新进程使用项目 `run_project.ps1`；交互终端先加载 `enter_project.ps1`，继续使用独立 Conda。

```powershell
Invoke-SegPython -m segformer_lab.download_model `
  --out "$SegResource\models\mit-b0" `
  --revision '80983a413c30d36a39c20203974ae7807835e2b4' `
  --source-record 'reports\model_source.json' --verify-only `
  --report 'reports\model_snapshot_check.json'

Invoke-SegPython -m segformer_lab.model_check --config 'configs\baseline.yaml' `
  --loading-report 'reports\model_loading.json' --check-report 'reports\model_check.json'
Invoke-SegPython -m unittest discover -v
```

本机有大小写冲突的旧代理，联网下载时显式传 `--proxy $SegProxy`，其中值为当时有效的 HTTP(S) 代理。参数只覆盖下载子进程，不更改全局配置；离线校验与训练不需要代理。其他工具应使用各自支持的配置，不能说所有联网命令必须有本参数。

换机下载的完整操作模板见[验收报告第 5 节](../../reports/2026-10-31-stage05-model-loading-and-check.md#5-代理与跨机器恢复约定)。模型资源留仓库外，Git 保存模型来源JSON、代码、配置、报告。当前快照损坏则先调查或换新目录，不覆盖来源绕过校验。

## 9. 学习自测与限制

1. **为什么192/192不能独自证明属于约定的commit？** 两份被同时替换的本地模型/检查点也可能相等，必须先验固定来源与文件哈希，再比较加载结果。
2. **为什么本次16个missing不是错误？** 分类预训练没有分割decode_head；新头按框架初始化，后续需要监督学习。
3. **为什么CE相等仍不能说padding无任何影响？** 这是标签归约层面的证据；填充图像仍参与特征上下文。
4. **为什么 batch1×累积2 不一定等于 batch2？** 梯度平均条件之外，BatchNorm、dropout及样本顺序可能改变行为。
5. **为什么保存source.json而不推权重？** 小型来源记录可精确指向commit和期望字节；权重可从该来源恢复并校验。

未验证真实图/掩码读取、同步增强、真实归一化、正式CE+Dice、原图逆变换与评价、AMP训练、DataLoader、训练恢复或最终性能。没有对未来硬件/库版本承诺逐位一致。

## 10. 下一阶段与修订记录

阶段 06 只实现 Dataset、同步水平翻转、同一letterbox、一次processor归一化和DataLoader；保留原图GT/几何/稳定ID，展示真实叠加，接一次eval/no_grad真实B2前向。不要在`__getitem__`加载模型或做CUDA；processor只构造一次。完整可复制指令在[验收报告第7节](../../reports/2026-10-31-stage05-model-loading-and-check.md#7-下一步只执行阶段-06)。

| 日期 | 修订 |
| --- | --- |
| 2026-10-31 初版 | 下载3个文件，建立加载与人工通路，76项回归 |
| 2026-10-31 验收 | 收紧来源/文件/未知键/精确权重门槛，固定种子；先B1/B2再调试更新，复测504/1008MiB |
| 2026-10-31 验收 | 新增portable source与恢复参数、保护既有固定目录及代理日志；101项回归，5项加载及9项运行条件通过 |
