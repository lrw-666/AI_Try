# 阶段 06：Dataset、同步变换与 DataLoader——设计与实现学习文档

| 字段 | 内容 |
| --- | --- |
| 状态 | 已通过第六阶段验收；39项数据契约、134项项目回归通过；真实B2 eval前向已运行 |
| 日期 | 2026-10-31 |
| 工程 / 资源 | `E:\桌面\AI\AI_Try\work\SegFormer` / `E:\桌面\AI\segformer_resources` |
| 前置 | v2分组清单、阶段04 geometry、阶段05固定MiT-B0与processor |
| 验收入口 | [第六阶段验收与下一步](../../reports/2026-10-31-stage06-acceptance.md) |
| 本阶段范围 | 数据读取、变换、batch、可视化及真实图eval/no_grad前向；没有损失/指标/训练器 |

## 1. 本阶段解决什么问题

把磁盘上的 VisA PCB1 图片和多ID掩码，变成模型可以直接接收、且训练与评价可以追溯的张量。此前已经证明原文件和快照可信；这里证明每次取数没有改错类别、图像与掩码没有各自翻转、processor没有重复归一化、padding仍是忽略标签。

检查不仅看shape，还要用非对称缺陷做强制镜像、手算归一化、检查每条geometry，并看真实图进入已加载模型后的实际输出。

## 2. 技术栈与版本

| 技术 | 实测版本/配置 | 本阶段作用 |
| --- | --- | --- |
| Python | 3.12.16，独立Conda | Dataset与脚本 |
| torch | 2.10.0+cu126 | Dataset/DataLoader、张量、真实eval前向 |
| transformers | 4.57.6 | SegformerImageProcessor；沿用阶段05模型加载 |
| NumPy / Pillow | 2.5.4 / 12.3.0 | 二值化、同步翻转、读取与缩放 |
| Matplotlib | 3.11.2，Agg | 9张五联图与原尺寸局部放大 |
| 运行配置 | batch2、workers0、seed42、pin_memory=False | 当前唯一已验收worker路径 |
| GPU | RTX4070 Laptop | 正常0011+异常087的B2 eval/no_grad前向 |

没有安装、升级包或下载新资源。真实原图、掩码和预训练快照未改；大图留资源目录，小报告和代码入Git。

## 3. 模块、接口与流向

| 文件 | 职责 |
| --- | --- |
| `segformer_lab/data.py` | PCB1Dataset、build_processor缓存、flip_decision、collate_pcb1、build_dataloader |
| `segformer_lab/dataset_check.py` | 身份预检、数据契约、独立归一化/翻转对照、固定验证图与真实前向、provenance |
| `tests/test_dataset.py` | 28项玩具数据与接口回归，含非对称图、配置ID、seed/epoch、processor和collate |
| `tests/test_dataset_check.py` | 5项验收回归：skip状态、源GT方向、每条geometry、身份预检等 |
| `reports/dataset_check.json` | 当前schema /2，39项结果及代码/源文件/PNG哈希 |
| `reports/stage06-acceptance.json` | 本轮独立验收、实际证据和原协议身份 |
| `runs/dataset_check` | 资源目录中的3张train图+6张固定val图 |

```text
清单 + 配置
 → 读取 RGB / 掩码（normal构造零）
 → 原始ID >0合并成1
 → 同一flip决策作用于RGB与二值mask
 → 共用geometry letterbox
      RGB：双线性 + pad(124,116,104)
      mask：最近邻 + pad255
 → processor一次 rescale/normalize（不再resize/减标签）
 → CPU pixel_values/labels + ID/geometry/flipped/original_mask
 → collate只堆叠固定尺寸张量
 → 调用方按需将模型输入搬GPU
```

## 4. Dataset 的字段与坐标契约

| 字段 | 单样本 | batch |
| --- | --- | --- |
| pixel_values | float32 `[3,512,512]`，CPU | `[B,3,512,512]` |
| labels | int64 `[512,512]`，0/1/255 | `[B,512,512]` |
| image_id / mask_id | 稳定相对路径；normal mask为空串 | list |
| geometry | 原W/H、scale、新W/H、四边padding、有效面积、ignore_index | list[dict] |
| flipped | 当前是否水平镜像 | list[bool] |
| original_mask（可选） | uint8 `[orig_h,orig_w]`，二值0/1 | list[np.ndarray]，不堆叠、不搬GPU |

**original_mask 的“original”指原尺寸。** train翻转后它也处于增强方向，与当前输入/标签对应；不是未翻转磁盘GT。val/test被限定flip=none，因此其原尺寸GT同时对应文件原图方向。评价使用val/test，不能把两个坐标系混用。

`prepared_sample()`额外暴露中间uint8图、512标签和原尺寸二值GT，用于诊断。`__getitem__()`只返回训练/评价需要的字段；它调用已经构造好的processor，不在此加载模型或做CUDA。

## 5. 实现细节与设计取舍

### 5.1 必须先合并，再padding

```python
binary = (raw_mask > 0).astype(np.uint8)
# 随后才做同步flip、最近邻缩放与255填充。
```

原始1..4是缺陷ID，而后加255是忽略标签。若在padding后做`>0`，会把忽略区错误变成缺陷。本次真实验证087的原始值为`{0,3}`，最终仍只有0/1/255，证明不只处理ID1。

正常图没有mask文件，不拼接空路径去读图；按真实图像大小构造全0。异常mask要求单通道且与原图尺寸相等。

### 5.2 同步翻转与epoch管理

```python
rng = np.random.default_rng([seed, epoch, index])
flipped = rng.random() < probability
# 同一次flipped判断同时作用于image和binary。
```

同一seed、epoch、index必然得到同一决策；数据顺序不变时重跑可复现，不依赖worker完成顺序。跨epoch改变模式是正常增强，本轮不改成每图永久固定。

调用方必须在每轮取迭代器前执行`train_dataset.set_epoch(epoch)`。DataLoader不会自动递增Dataset中的epoch；恢复训练时也要恢复并传入正确轮数。更改子集会改变index映射，应保存子集ID及SHA。

RGB HWC或2D标签的左右翻转与CHW张量的轴号不同：CHW图像翻dim=2，二维label翻dim=1。直接对CHW用`torch.fliplr`会翻高度，原交付已修过该错误，非对称渐变玩具图继续守住它。

val/test仅允许none，强制路径也拒绝；强制翻转只用于train诊断。seed/epoch须为非负整数，概率为有限的[0,1]；默认seed来自配置，loader默认继承Dataset seed。

### 5.3 训练子集不能被配置后又忽略

train在未显式传ids_file时读取已解析的`data.train_ids_file`，显式文件优先；每个ID都必须属于当前split，空/重复ID拒绝。val/test不自动继承训练子集。这能避免阶段08本想跑8图却误用全量训练集。

ID采用清单相对路径，不只用文件stem，避免Normal/0000和Anomaly/0000之类的同名冲突。相对路径规则以主配置已解析路径为准；手工API传ids_file时优先给明确路径。

### 5.4 Processor只做一次数值处理

Dataset把uint8 RGB画布交给processor，不提前除以255。四项开关固定为resize=False、reduce_labels=False、rescale=True、normalize=True，内置或手工注入实例都检查；mean/std必须是三个有限值，std>0。

独立参考路径是：

```python
x = image_512.astype(np.float32) / 255.0
chw = np.transpose(x, (2, 0, 1))
reference = (chw - mean[:, None, None]) / std[:, None, None]
```

另一侧是`__getitem__`的真实processor输出，不是同一张量自比；本次最大差0.0，验收容差1e-6。参考路径仍使用同一套已验快照常量；常量本身由来源文件SHA另外约束。

`lru_cache`共享Dataset processor，本轮清理cache后factory misses=1且各Dataset引用相同。统计范围仅数据工厂，不能把cache条目数或这一读数当作整个进程所有模型加载的构造次数。

### 5.5 original_mask 用list的理由与内存

每张1404×1070的uint8数组为1,502,280字节，B2 payload合计3,004,560字节（约3.00MB/2.87MiB），另有Python对象开销。保留list可支持不同原尺寸，避免为了batch强行pad/stack或额外复制。

CPU上的list和CPU堆叠都不会自动占用GPU；关键是只把pixel_values/labels送GPU。训练可`return_original_mask=False`，仍需读取原始GT来生成训练标签，但不在batch保留原尺寸数组；验证/测试计分须保留或明确重读。混合有/无original_mask的batch现在会报错，不再静默丢弃。

### 5.6 为什么现在继续workers0

Windows主进程直接读取最便于排错与控制epoch，当前已验证这条路径。没有吞吐基准，不说0最快；阶段08记录data_time、step_time及主机内存，再判断是否值得单独试2。

未来worker>0需验证spawn、顶层可导入类/collate、随机性、epoch传播、主机内存与进程退出；不能未经验证开启persistent_workers，否则主进程set_epoch未必传入已驻留worker。当前构造函数与取数都不在worker内做CUDA。

## 6. 验收脚本如何避免假通过

1. 读取样本前，复用阶段04清单/组源校验，并按仓库model_source离线校验固定快照；不联网。
2. 每条geometry独立由原W/H推导整数新尺寸/四边padding/有效区，检查每条label，不能只看第一条。
3. 检查图像有限性、shape/dtype，mask的uint8/0或1、原尺寸和所有metadata list。
4. val实际重读必须相等；val/test的random与force构造均实际拒绝。
5. 检查3张train诊断与固定6张val，选normal0011+anomaly087组成真实B2，`model.eval()`且`torch.no_grad()`。
6. 记录模型training=false、logits.requires_grad=false、实际输出有限，并存代码/配置/来源/源文件/PNG的SHA。

`--skip-model`只代表数据诊断：三个前向结果为null，status=skipped，data_checks_ok可为true，但ok与full_stage_verified为false。CLI诊断成功可exit0，不能将其误当完整阶段通过。

当前正式报告是scope=full_stage、full_stage_verified=true，39项全部执行并通过。

## 7. 可视化的两个坐标系

五联图前两列是未增强的source RGB与source GT，保持同一文件方向；后三列是增强后的512输入、label、overlay，局部放大展示同增强方向的原尺寸GT。

旧图错误是第一列未翻、第二列已翻却仍写original GT；数据本身的后三列正确，但展示会误导。现在第二列翻回source方向并写清标题，强制图的后三列仍镜像。验收比较004左侧弯针镜像到右侧时，红线是否跟着同一物理引脚。

[翻转对照小图](../../reports/figures/stage06-flip-comparison.png)已保存仓库；整批9张PNG在资源目录。全部9张本轮已实际看过，逐文件结论见[验收报告](../../reports/2026-10-31-stage06-acceptance.md)。0983的正常网格疑点延续阶段04记录，没有为了效果改标。

## 8. 真实执行结果与复现

| 项目 | 本轮结果 |
| --- | --- |
| 三清单规模 | 530 / 132 / 442；v2身份不变 |
| B2输入/标签 | float32[2,3,512,512] / int64[2,512,512]，CPU |
| 几何 | 512×390，上下61，B2 padding=124,928像素 |
| flip | uint8差0，归一化tensor差0.0，类别与原尺寸GT镜像exact |
| normalization | 公式最大差0.0 |
| 标签语义 | normal0011有效区0；anomaly087原ID0/3→0/1，512前景313px |
| 重读 | train/val输出与原尺寸GT均一致 |
| 真实前向 | val0011+087，eval/no_grad，float32[2,2,128,128]有限 |
| 数据契约/回归 | 39项通过 / 134项通过 |

```powershell
. '.\scripts\enter_project.ps1'
Invoke-SegPython -m segformer_lab.dataset_check `
  --config 'configs\baseline.yaml' --report 'reports\dataset_check.json'
Invoke-SegPython -m unittest discover -v
```

本轮已执行完整命令。关键日志为`reports/stage06-dataset-check.txt`与`stage06-unit-tests.txt`，证据摘要为`stage06-acceptance.json`。新机器需先恢复数据和模型；完整回归包含使用实际本地数据的CPU集成检查。

这里抽样实际读取9张唯一原图；未重新解码1104张，也未读取测试图片算成绩。阶段04已完成全量数据质量检查，两者范围不同。

## 9. 学习自测与限制

1. **为什么0/1/255不是三个训练类别？** 只有背景/缺陷两类，255是忽略标签，不参与正式损失。
2. **为什么原尺寸mask也可能已经镜像？** 原尺寸只说明H/W，方向由flipped决定；val/test不增强才对应文件坐标。
3. **为什么归一化误差0不代表最终精度高？** 它只证明输入数值路径正确，随机新分割头尚未学习缺陷。
4. **为什么一个成功exit0不一定是完整验收？** dataset_only模式故意跳过模型，需要检查scope和full_stage_verified。
5. **为什么每轮需要set_epoch？** Dataset不感知训练器轮次；相同epoch将继续给同样的确定性增强。

没有实现正式CE+Dice、原图logits还原与指标、训练器、AMP反向、恢复训练或worker>0；本阶段不提供模型质量结论。

## 10. 下一阶段与修订记录

下一步只做阶段07。损失使用512标签，主评价则恢复logits至原图并用未缩小的val/test original_mask累计混淆量。完整可复制指令在[验收报告第7节](../../reports/2026-10-31-stage06-acceptance.md#7-下一步指令只执行阶段-07)。阶段07通过后才写20步smoke。

| 日期 | 修订 |
| --- | --- |
| 2026-10-31 初版 | Dataset/collate/loader及27项检查、3张图；115项回归 |
| 2026-10-31 验收 | 修正skip状态与source GT展示，强化每样本契约、分组/来源身份和provenance，绘制固定6张val图 |
| 2026-10-31 验收 | 配置训练子集、seed/epoch/processor验证、评价禁强制翻转、collate混合拒绝；根默认discover导入修正后134项通过，完整39项及真实B2前向通过 |
