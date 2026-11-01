# 阶段 07：损失与原图指标——定义、实现与数值核验

| 字段 | 当前记录 |
| --- | --- |
| 更新日期 | 2026-11-01 |
| 对应指南 | [主指南](../../../../docs/research/2026-10-31-segformer-baseline-end-to-end-guide.md)第9节 |
| 状态 | 已验证：统一IEEE策略完整检查42/42通过，fresh CPU单测186项通过；不是训练性能验收 |
| 前置依赖 | [阶段06](06-dataset-design-and-implementation.md)、v2分组清单、固定MiT-B0快照与Dataset契约 |
| 项目根 / 资源根 | `E:\桌面\AI\AI_Try\work\SegFormer` / `E:\桌面\AI\segformer_resources` |
| 解释器 / prefix | `E:\桌面\AI\segformer_resources\env\python.exe` / 同目录 |
| 代码身份 | 验收JSON的`provenance.files`、`files_after`及配置哈希；旧报告对应的代码不能代表当前修订 |
| 范围 | 损失、原图恢复、指标、人工夹具与随机头管线probe；未实现训练器或optimizer更新 |

## 1. 本阶段解决什么问题

输入是模型低分辨率logits、512标签、原图二值掩码和每张图的geometry；输出是可反传的损失、原图混淆量和验收证据。目标是在训练前固定“优化什么”和“如何计分”，用可手算的数组发现静默算错。

本阶段采用指南配方`CE + 0.5 × 正GT图前景Soft Dice`。它是实验配置，不是新算法，也不能据此断言CE-only一定学成全背景。随机初始化的解码头只用于检查管线，报告明确`not_a_baseline=true`，不能用其IoU或误报率评价模型能力。

## 2. 技术栈与隔离

| 组件 | 已有环境记录 | 本阶段用途 |
| --- | --- | --- |
| Python | 3.12.16 | 项目专用Conda prefix |
| PyTorch | 2.10.0+cu126 | 插值、CE、softmax、autograd与数值策略 |
| NumPy | 2.5.4 | 布尔混淆量、异常图宏平均、正常图百分位 |
| 平台 | Windows 11，RTX 4070 Laptop | CPU单测与独立GPU数值诊断 |
| 新增依赖 | 无 | 未安装或升级包 |

版本来源为[环境报告](../../reports/environment.json)；每次新验收还应核对JSON的`environment`。通过项目runner建立进程上下文，缓存位于资源根`cache`，临时文件位于`tmp`，大图位于`runs`；不会因此获得虚拟机级隔离。若需复核版本，使用runner执行`python -c`查询`sys.version`、`torch.__version__`、`numpy.__version__`，不要切到base环境。

## 3. 四个损失取舍

### 3.1 先放大logits，保留512标签

```text
[B,2,128,128] logits
 → bilinear至labels的[B,512,512]空间，align_corners=False
 → CE：非255像素的均值
 → FP32 softmax取通道1；valid=(labels!=255)，g=(labels==1)
 → Dice_i=(2·Σ(p·g·valid)+eps)/(Σ(p·valid)+Σ(g·valid)+eps)
 → L=ce_weight·CE+dice_weight·mean(1-Dice_i)，mean仅覆盖正GT图
```

把GT用nearest缩到128也能建立精确、可重复的定义，并非“标签归属有歧义”。本协议选择放大logits，是为了保留现有512标签和ignore区域，不再进一步丢失细小前景；代价是逐像素损失开销增加。改变此选择就改变实验身份。

### 3.2 只对有效GT有前景的图计算逐图Dice

每张图先独立求Dice，再对`Σ(g·valid)>0`的图取均值；不能把全batch像素拼在一起算一个Dice，也不能除以全batch图数。交集必须是`Σ(p·g)`，不是`(Σp)·(Σg)`。

正常图由CE监督，误报仍受惩罚。这是本次positive-only配方的明确选择；正常图也参与Dice、采用其他空GT约定等方案可以另行定义，并非数学上都错误。带eps的空GT公式也不是必然无定义，只是其优化行为与本配方不同。

### 3.3 全正常batch返回与图相连的0

`dice=prob.sum()*0.0`保留Dice分支的autograd连接，单独对它反传可得到存在且精确为0的梯度。若写成Python `0.0`再与CE相加，CE的梯度仍在，不能声称整个总损失失去梯度或GradScaler必然报错；本实现保留连接是为了Dice分支接口一致和可独立检查。

### 3.4 全ignore batch显式拒绝

若整批没有有效像素，CE均值没有有效分母，函数抛`ValueError`。若只是其中一张图全ignore而其他图有效，该图不贡献CE/Dice；`valid_images`只统计至少有一个有效像素的图。

## 4. 指标、NA与恢复顺序

### 4.1 原图恢复只有一个入口

```text
单图[2,h,w] logits → CPU FP32
 → bilinear至512×512，align_corners=False
 → 按pad_top/pad_left/new_h/new_w裁有效框
 → bilinear至orig_h×orig_w，align_corners=False
 → 两通道softmax → 缺陷通道概率>=0.5
```

先裁padding再恢复原图，且在logits恢复后才softmax；放大概率是另一种定义。geometry尺寸须为有限整数、padding非负、有效框不越界；提供bottom/right时还须与canvas总尺寸一致。

GT使用Dataset提供的原尺寸`original_mask`，不把512标签放大回去。阶段04记录的是**12张图中19个区域丢失，共39个原始前景像素**。单像素夹具中，原GT产生1个FN；被nearest跳过的512标签再放大后变成空GT，会把漏检变成NA。

Dataset的`original_mask`已经随增强同步翻转。默认拒绝`flipped=True`是baseline验证/测试不使用增强的策略约束，并非原mask未同步。`accept_flipped=True`是显式协议变体，调用方仍须确保logits与原尺寸mask处于相同坐标系。

### 4.2 先累计混淆量，再计算全局指标

```text
IoU=TP/(TP+FP+FN)        Dice=2TP/(2TP+FP+FN)
precision=TP/(TP+FP)     recall=TP/(TP+FN)
```

混淆量用int64累加，最后统一相除，不对batch IoU取均值冒充global。例：图A为TP1、FP0、FN0，图B为TP1、FP98、FN1；逐图均值为0.505，全局IoU为2/101。宏平均可能高于或低于全局值，差异方向取决于图像和错误分布；二者回答不同问题。

异常图宏平均只覆盖`label='anomaly'`且该指标有定义的图，报告`defined_images`和`na_images`。本项目的全局前景IoU也不能直接叫作按类别平均的mIoU。

### 4.3 空分母与正常图误报

- 指标不加平滑项。分母0输出JSON `null`及原因；逐图precision/recall也分别给出NA原因。
- GT有缺陷而预测全背景：IoU/Dice/recall为0，precision为NA。
- GT和预测均空：逐图IoU/Dice为NA，不按1分加入均值。
- 正常图误报率为`ΣFP_normal / Σ有效正常图像素`；另报逐图预测面积比例的中位数、P95、最大值。
- P95使用NumPy对逐图比例的线性插值，不是对全体像素算百分位。无正常图或无可定义异常指标时，分组统计输出NA及原因。

### 4.4 GT与输入的严格边界

评价mask只接受离散0/1/255：0背景、1缺陷、255忽略；原始多缺陷ID须先由Dataset合并。其他ID、小数、NaN均拒绝。255从TP/FP/FN/TN和面积分母中排除；全ignore样本拒绝，不能静默变成前景或空图。

`ImageScore.total_pixels`表示有效像素数，另有`ignored_pixels`。要求`TP+FP+FN+TN=有效像素数`；拒绝非法threshold、非有限logits/恢复概率及未知label，避免NaN比较后被误记为全背景。允许异常标签的GT为空，以保留缩小GT丢单像素的反例。

## 5. 文件、接口与关键代码

| 文件 | 职责与入口 |
| --- | --- |
| `segformer_lab/losses.py` | `segformer_loss`返回`LossTerms`；`loss_definition`记录实际权重/eps |
| `segformer_lab/metrics.py` | `restore_defect_probability`、`OriginalSpaceEvaluator`、评分定义 |
| `segformer_lab/evaluation.py` | `evaluation_precision(config, device=...)`提供统一评价精度上下文 |
| `segformer_lab/metrics_check.py` | 身份预检、人工夹具、同logits累计与两次真实前向对照、provenance |
| `configs/baseline.yaml` | 两类、512、255、损失参数、阈值0.5、正式eval batch=1、`eval.precision: ieee_fp32` |
| `tests/test_losses_metrics.py` | 原有公式、恢复和原GT反例 |
| `tests/test_loss_metric_acceptance.py` | half CE、输入边界、ignore、int64、NA与百分位回归 |
| `tests/test_metrics_check.py` | XOR、浮点容差、fixtures-only语义与预检拒绝 |
| `tests/test_evaluation.py` | 精度上下文与异常退出恢复；训练AMP配置独立性另见`test_metrics_check.py` |

`segformer_loss(logits, labels, *, ce_weight=1.0, dice_weight=0.5, eps=1e-6)`要求浮点logits与严格int64标签，固定两通道；eps有限且>0，权重有限非负且不能同时为0。`loss_definition(...)`的`runtime`和`LossTerms.as_dict()`记录实际参数，默认配方保持1/.5/1e-6。

实际实现片段（`losses.py: segformer_loss`）：

```python
ce = F.cross_entropy(logits_512.float(), labels, ignore_index=255, reduction="mean")
prob = torch.softmax(logits_512.float(), dim=1)[:, 1]
intersection = (prob * gt).sum(dim=(1, 2))
# 后续仅索引has_foreground子集，再逐图求Dice并取均值
```

修复前，CPU上half零logits `[1,2,128,128]`配全正常512标签，离开autocast调用可得到CE=NaN和零梯度；显式转FP32后CE≈ln2、梯度有限且非零。常规CUDA autocast内CE本已按FP32执行；这里修复的是函数脱离autocast后的边界，并非发现AMP的CE默认不稳定。Dice也显式FP32。[PyTorch AMP说明](https://docs.pytorch.org/docs/2.10/amp.html)列明CE的CUDA autocast精度策略。

评价统一用NumPy布尔数组计数（`metrics.py: add_sample`）：

```python
pred = prob.numpy() >= threshold
gt = mask == 1
valid = mask != 255
tp = int((pred & gt & valid).sum())
fp = int((pred & ~gt & valid).sum())
```

避免Torch与NumPy混合位运算的弃用告警；验证全部输入后才提交累计，失败样本不会污染已有计数。

## 6. 两种“batch一致性”必须分开

**固定logits换分组**只检查计分器：缓存同一份logits，改变遍历/分组后，混淆量、逐图记录和预测哈希必须严格相同。整数累计没有浮点近似的借口。

**B2与B1重新前向**检查模型浮点执行：除混淆量外，逐像素比较阈值预测、概率误差和变化像素距0.5的距离。旧“12像素差异”由净混淆量差推算，不能反映正反方向抵消，已撤回。

核心口径（`metrics_check.py: compare_forward_tensors`）：

```python
left = probability_left >= threshold
right = probability_right >= threshold
changed = left ^ right                  # 真正逐像素XOR，不是abs(TP差)+abs(FP差)
changed_pixels = int(changed.sum())
```

正式概率绝对容差固定`atol=1e-5, rtol=0`；还检查变化像素的阈值边距，不因“占全图比例小”就接受。独立诊断结果见[诊断记录](../../reports/stage07_diagnostics/README.md)：

| 诊断设置 | B2/B1真XOR | 最大概率绝对差 | 固定预算结论 |
| --- | --- | --- | --- |
| 默认cuDNN TF32开、matmul TF32关 | 621 / 9,013,680 | 2.855062484741211e-5 | 超过预算，41/42，不能验收 |
| 仅关闭TF32 | 1 / 9,013,680 | 1.4901161193847656e-7 | 该次诊断42/42 |

两种设置的B1重复均为XOR0；相同缓存logits改变累计分组也严格一致。关闭TF32是针对该精度差异的诊断，不等于已识别某个cuDNN kernel为唯一根因。

当前评估配置固定`eval.precision: ieee_fp32`。`evaluation_precision(config, device=...)`在上下文内禁用cuDNN/matmul TF32及外层autocast，在`finally`恢复先前flags与autocast状态；`train.amp`仍是独立训练选项。为适配锁定的PyTorch 2.10，统一使用legacy `allow_tf32`接口，不混搭新旧控制API。

保持正式eval batch=1，不放宽1e-5门槛；数值策略连同配置和生效状态留证。IEEE FP32也不承诺不同batch、设备或版本全平台bitwise一致；这符合[PyTorch数值精度说明](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)关于批处理和浮点执行的约束。

## 7. 复现命令与证据状态

以下命令使用项目runner；正式报告由本轮验收流程生成，日常复核写到资源根`tmp`，保留旧证据：

```powershell
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @(
  '-m','unittest','discover','-s','tests','-v')
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @(
  '-m','segformer_lab.metrics_check','--skip-model','--device','cpu',
  '--report','E:\桌面\AI\segformer_resources\tmp\stage07-fixtures\report.json',
  '--figure-dir','E:\桌面\AI\segformer_resources\tmp\stage07-fixtures\figures')
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @(
  '-m','segformer_lab.metrics_check','--config','configs/baseline.yaml','--device','cuda',
  '--report','E:\桌面\AI\segformer_resources\tmp\stage07-full\report.json',
  '--figure-dir','E:\桌面\AI\segformer_resources\tmp\stage07-full\figures')
```

`--skip-model`成功只说明夹具通过：模型相关checks是`null`，`fixtures_ok=true`，`ok/full_stage_verified=false`，不能写“阶段完整通过”。完整probe先核清单/快照身份及配置，再读固定val样本；保存模型、输入、原GT、logits/概率/预测及关键源文件的哈希证据，检查前后文件未变。

| 已有证据 | 能支持的结论 | 使用边界 |
| --- | --- | --- |
| CPU half CE及ignore/NA回归，含红→绿 | 对应缺陷已复现并修正；原有22项损失指标回归仍通过 | 无optimizer或真实模型训练 |
| `reports/stage07_diagnostics/*` | 上表两种数值设置的独立诊断 | 与最终统一策略的验收报告分开 |
| [正式JSON](../../reports/loss_metrics_check.json)、[执行日志](../../reports/stage07-loss-metrics-check.txt) | 2026-11-01T10:50:06+08:00；42/42 true，18.99s；`ok/full_stage_verified=true` | `not_a_baseline=true`，仅管线验收 |
| [完整回归日志](../../reports/stage07-unit-tests.txt) | 统一数值策略加入后186 tests，10.629s，OK | CPU单测，不包含训练性能结论 |

正式IEEE策略下，B2/B1真XOR为1/9,013,680，最大logit差3.725290298461914e-7、最大概率差1.4901161193847656e-7；B1重复的XOR、最大logit差和最大概率差均为0。同一缓存logits换累计分组严格相同。41份provenance文件前后未变，源代码、CSV和模型身份锚点检查均通过；完整明细保留于JSON，不复制随机头性能分数。

最终GPU检查和CPU回归均通过原项目runner执行；沙箱内曾出现DLL加载问题，沙箱外runner已验证正常，未为此改动环境、系统或依赖。历史32/32、156项或旧12px统计不能替代本轮证据。

恢复对照图中的**36.141 px**只属于该非对称合成夹具漏裁padding时相对理论质心的误差，不能推广为任意图像的固定偏移。

## 8. 修复与限制

本轮修复：half CE离开autocast的NaN；浮点标签被误放行；全ignore图数记错；mask255被当成前景；原ID/NaN/非法geometry静默进入评分；混合数组告警；NA原因缺失；用净混淆差代替预测XOR；fixtures-only与完整验收混淆。对应实现和回归见第5—7节。

仍须分清三个限制：双线性插值会混合边界附近的logits，因此“padding不计入loss”不意味着所有低分辨率padding邻近单元梯度必为0；eps影响优化用Soft Dice，不能搬到报告指标中；随机头probe只能证明管线行为，不能证明缺陷分割效果。

## 9. 学习自测

1. 全正常图为什么还能训练？为什么Dice-only反传应有零梯度，而总loss梯度通常非零？线索：第3节与`segformer_loss`。
2. 原GT有1个缺陷像素，512GT丢掉它，预测全背景：两种GT分别产生什么FN和IoU？线索：第4节及`RestoreTests`。
3. B2/B1的TP和FP净差都为0，能否证明预测相同？为什么IEEE策略下仍不要求跨平台bitwise一致？线索：第6节与逐像素XOR。

## 10. 下一依赖与修订记录

阶段08另行实现smoke训练，直接复用本阶段loss、恢复入口和评分定义。训练日志聚合时，CE按有效像素数加权，Dice按正GT图数加权，不能简单平均batch均值；若batch拆分改变两种分母，也不能假定任意拆分后的总loss天然相同。正式验证继续固定原GT、阈值、无增强和IEEE策略。

| 日期 | 修订与影响 | 复核证据 |
| --- | --- | --- |
| 2026-11-01 初稿 | 建立损失、恢复与指标；旧通过结论含不可靠差异统计 | 旧产物仅保留为历史 |
| 2026-11-01 复核 | 修正四取舍与梯度解释、GT255/NA、FP32 CE、XOR和TF32诊断；撤回12px解释，固定IEEE评价策略 | 本轮正式JSON 42/42、186项CPU回归通过；诊断历史单独保留 |
