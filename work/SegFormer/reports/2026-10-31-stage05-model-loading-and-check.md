# 第五阶段验收：预训练快照、二类模型与前后向核验

日期：2026-10-31。工程：`E:\桌面\AI\AI_Try\work\SegFormer`。

**结论：完善后的第五阶段验收通过，可以进入阶段 06。** 本轮独立重跑了快照校验、GPU 前向/反向/一次内存调试更新和全套回归。这里没有 PCB1 训练成绩、IoU 或 Dice 指标。

## 1. 用户五项检查的结论

| 检查 | 本轮确认与决定 |
| --- | --- |
| encoder 192/192、差 0、只剩两项 classifier | 当前 `model_loading.json` 重新确认：192/192 精确相等，`max_abs_diff=0.0`，未使用键只有 `classifier.bias`、`classifier.weight` |
| forward 与 memory | logits 仍为 `[2,2,128,128]`；两路 CE 本次差 **0.0**；隔离后的 B2 峰值 **846.8 MiB allocated / 1008.0 MiB reserved**。不能继续把旧 1058 MiB 描述为不含优化器状态的隔离测量 |
| batch 2 起步 | **接受**，保持既定 batch=2、累积=1。阶段 08 用真实数据、完整损失与优化器跑 20 步再确认预算 |
| 显式代理 | 写入本机下载约定：需要联网且继承代理不可靠时，用该模块的 `--proxy`。不能推论为“所有工具均有这个参数，或不带一定 502”；离线加载、校验、训练无需代理 |
| 快照是否进 Git | 14.45 MB 权重快照继续留在资源目录；将小型完整来源记录保存为 `reports/model_source.json`，支持固定 commit + 全文件 SHA 的跨端重建 |

## 2. 本次修复了什么

| 问题 | 完善后的行为 |
| --- | --- |
| 原加载器只比较模型与本地 checkpoint，不查 source 声明的哈希 | `load_all` 在加载 processor/模型前验证 source、固定模型和 commit、全部记录文件大小及 SHA，再执行编码器逐张量比较 |
| 原快照校验可接受空清单、遗漏权重记录，额外 safetensors 还可能抢先加载 | 严格校验 schema、40 位 commit、唯一安全路径、元数据与选定权重条目；拒绝未登记文件、LFS 指针、哈希/大小变化和权重优先级冲突 |
| 既有快照损坏或请求另一版时，原下载器可能重新下载并重签 source | 已有目录发生污染、模型或显式 commit 不符就拒绝联网和覆盖；要换版本应新建目录 |
| 新机器只给模型名与 proxy 会跟随 main | 新目录默认固定本次 commit；推荐再传 `--source-record`，逐文件对照仓库来源清单 |
| 显存 probe 前已有 B2 调试更新，残留优化器状态与张量 | 改成 **B1 probe → B2 probe → B2 调试更新**；probe 运行时尚未创建 AdamW 和主调试张量，报告明确 FP32、AMP 关闭及范围 |
| 原通过条件遗漏 shape、有限 CE、processor、分类头等 | 加入显式 acceptance 条件；报告本身能区分成功与失败 |
| 模型头随机初始化前未固定种子 | 初始化前按配置设置 seed=42，报告记录 seed 和代码/配置/来源哈希 |
| 合成 256 标签仍固定 61 行 padding，且缺陷块越界 | 复用几何计算：256 时上 30、下 31，前景非空；正式 CLI 拒绝与配置 512 不一致的输入尺寸 |
| 控制台直接打印代理地址 | 只报告使用显式覆盖还是继承环境，不输出代理 URL 或凭据；显式代理仅作用于当前 Python 进程 |

修复采用先失败再通过的回归检查，最终全套 **101 项通过**。下载器新增的网络路径使用临时目录和离线替身测试，**本轮没有再次联网下载权重**；真实快照执行了离线校验和实际模型加载。

## 3. 快照身份与加载证据

固定模型：`nvidia/mit-b0@80983a413c30d36a39c20203974ae7807835e2b4`。
本地：`E:\桌面\AI\segformer_resources\models\mit-b0`。
原始下载记录保留在 `model_download.json`；本轮离线校验在 `model_snapshot_check.json`；可移植来源在 `model_source.json`。

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| config.json | 70043 | `e2378ac7c7a6981d6bdd1d9ccf29b611264c382afce89c561f39c9291164ff91` |
| preprocessor_config.json | 272 | `d96ab2cb112985c19d5ea32f1d1f2369104fc78b1379f5ce26e33ca8646202a7` |
| pytorch_model.bin | 14380029 | `4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc` |

合计 14,450,344 字节。既有原快照及 source.json 未改；调试更新只改变进程内存中的模型。

| 加载项 | 实测 |
| --- | --- |
| checkpoint 张量 / 参数量 | 194 / 3,576,392 |
| encoder 张量 / 参数量 | 192 / 3,319,392 |
| 精确相等 / 仅容差相等 | 192 / 0；现在验收要求 FP32 精确一致 |
| max_abs_diff | 0.0 |
| checkpoint 未使用 | 只有 classifier.bias、classifier.weight，原 ImageNet 分类头 |
| missing | 16 项，全部属于新建 decode_head |
| unexpected | 2 项，只有原分类头；未知多余键将被拒绝 |
| mismatched | 0 |
| 全模型 / 可训练参数 | 3,714,658 / 3,714,658 |
| 新分类头 | Conv2d，256→2，1×1；权重形状 `[2,256,1,1]` |
| processor | resize=False、rescale=True、normalize=True、reduce_labels=False |

这里的分类 checkpoint 没有分割解码头，因此解码头是“缺失后新初始化”，不是分类头形状冲突。本次 `ignore_mismatched_sizes=True` 没有被实际用于处理冲突；它本身不能证明 encoder 已加载。将来若有 decode_head 形状不符，仅看 `mismatched_other_keys` 也不够，因为头部冲突会归到 head；必须结合模型身份、完整键报告和 encoder 数值核验。

## 4. 本轮 GPU 实测结果

环境沿用独立 Conda：Python 3.12.16、torch 2.10.0+cu126、transformers 4.57.6、huggingface_hub 0.36.2；RTX 4070 Laptop，设备总显存 8187.5 MiB。没有安装或升级包。

### 4.1 人工张量与 CE

| 项目 | 本次结果 |
| --- | --- |
| 输入 | float32 `[2,3,512,512]` 人工张量，非真实 PCB 图 |
| 标签 | int64 `[2,512,512]`，值 `{0,1,255}` |
| padding | 上下各 61；有效行 `[61,451)` |
| logits | `[2,2,128,128]`，输出 stride 4 |
| 显式双线性上采样 | `[2,2,512,512]`，align_corners=False |
| 整幅 ignore CE | 0.6495407223701477 |
| 仅有效区 CE | 0.6495407223701477 |
| 差值 | **0.0**，阈值 ≤1e-6 |

旧运行差 `5.96e-08` 与本次 `0.0` 都符合容差。旧运行未固定模型初始化种子，修正后还改变了 probe/调试执行顺序，因此 CE 绝对值和头部随机统计可以改变；不能把“重现旧 loss 小数”当验收标准。seed 固定也不等于承诺不同硬件、库版本的逐位结果相同。

这证明 255 标签没有直接参与该 CE 的归约；不能声称图像 padding 对有效区预测完全没有影响，卷积和注意力的上下文仍可能受填充图像影响。

### 4.2 反向与更新

| 抽查参数 | 梯度范数 | 有限/非零 | 一步 max_abs_delta |
| --- | ---: | --- | ---: |
| segformer.encoder.patch_embeddings.0.proj.weight | 0.11584833 | 是/是 | 0.00010000169 |
| decode_head.linear_c.0.proj.weight | 0.21285953 | 是/是 | 0.00010000169 |

调试优化器为 AdamW(lr=1e-4, weight_decay=0)，只更新一次；不是正式训练配方。这里抽查每部分一个参数，不能据此宣称已经逐参数审计了整网梯度。

### 4.3 显存口径与 batch 决定

| batch | peak allocated | peak reserved | reserved / 总显存 |
| ---: | ---: | ---: | ---: |
| 1 | 426.6 MiB | 504.0 MiB | 6.16% |
| 2 | 846.8 MiB | 1008.0 MiB | 12.31% |

当前探测使用 FP32、AMP=False，范围为模型、输入、CE 前向与反向；尚未创建优化器及其状态。Torch allocator 的 allocated/reserved 也不等于 nvidia-smi 的整进程或整机显存占用，CUDA 上下文、其他应用和驱动另有开销。该次耗时包含初始化／内核预热等影响，不是吞吐基准。

**接受 batch=2 起步、梯度累积=1。**阶段 08 才用真实图、CE+Dice、AdamW、AMP 和连续 20 次更新确认运行预算。若实际 OOM，先检查显存占用与计算图保留，再改成 batch=1、累积=2并保存新配置。有效每次更新图数仍可为 2，但 BatchNorm、随机性等会使其不与原 batch=2 完全等价。

旧报告 1058 MiB 是原测量流程的结果，保存在 `stage05_pre_acceptance/model_check.json`，只供追溯。原代码在 probe 前已做调试更新，所以不能继续称其“不含优化器状态”；本次隔离结果为 1008 MiB。差额不能精确全部归因于某一种张量。

## 5. 代理与跨机器恢复约定

### 5.1 每次进入项目

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
. '.\scripts\enter_project.ps1'
```

本机历史上存在大小写重复且值冲突的代理环境变量。下载器在创建 Hub 会话前用显式 `--proxy` 覆盖当前子进程的 HTTP/HTTPS 大小写变量，清除旧 ALL_PROXY 干扰并同步 NO_PROXY 拼写。它不修改全局代理、shell profile、Conda base 或系统 PATH，也不把代理写进 source/config。

仅在需要联网下载时传**当时有效**的代理；端口可能变化，不把旧端口当永久约定。其他工具需用它们支持的代理配置，不能照搬本模块的参数。离线核验、模型加载及后续本地训练不需要 `--proxy`。

### 5.2 离线验证当前快照（本轮已执行）

```powershell
Invoke-SegPython -m segformer_lab.download_model `
  --out "$SegResource\models\mit-b0" `
  --revision '80983a413c30d36a39c20203974ae7807835e2b4' `
  --source-record 'reports\model_source.json' --verify-only `
  --report 'reports\model_snapshot_check.json'
```

### 5.3 换电脑重新下载（操作模板，未在第二台机器执行）

先拉 Git、按环境指南重建隔离环境，并调整该机资源根与 baseline.yaml。下面的代理地址由你替换为该机有效值；可直连时省略 `--proxy`。

```powershell
$SegProxy = Read-Host '输入当前有效的 HTTP(S) 代理地址'
Invoke-SegPython -m segformer_lab.download_model `
  --model-id 'nvidia/mit-b0' `
  --revision '80983a413c30d36a39c20203974ae7807835e2b4' `
  --source-record 'reports\model_source.json' `
  --out "$SegResource\models\mit-b0" --proxy $SegProxy
```

`--source-record` 不只固定 commit，还对照仓库保存的全文件大小和 SHA-256；下载后的字节不符就拒绝写成功 source。记录内的历史 `local_dir` 不作为新机器的目标路径，实际目标由 `--out` 决定。已有目录不一致时先调查或使用另一个新目录，不要为了通过校验覆盖来源记录。

快照不进入 Git；小型来源清单和复现代码已归档。若将来需要完全断网恢复，可额外备份整个快照到自己的备份盘，但不是本次 Git 提交内容。

## 6. 证据与验证范围

- 当前：`model_loading.json`（5 项加载条件通过）、`model_check.json`（9 项运行条件通过）、`model_snapshot_check.json`（问题列表为空）。
- 日志：`stage05-model-check.txt`、`stage05-snapshot-verification.txt`、`stage05-unit-tests.txt`（101 项通过）。
- 身份：`stage05-acceptance.json` 保存本轮代码、报告、来源及原 v2 数据清单哈希。
- 数据协议继续为 `segformer_b0_pcb1_v2_grouped`，0042/0043 均在 train；三份清单、组源及固定验证 ID 未改。

未执行真实 Dataset、正式组合损失、训练循环或验证集评价；阶段 06 起仍需逐步验收。现有显存余量不是毕业效果或最终精度的保证。

## 7. 下一步：只执行阶段 06

给 Agent 指定原工程目录，可以直接发送：

> 阅读 AGENTS.md、全程指南第 8 节、第四/五阶段验收及设计文档，只执行阶段 06。沿用独立 Conda 环境、当前 v2 清单和固定 MiT-B0 快照，先校验对应 SHA 与分组身份。实现可复用的 PCB1 Dataset、顶层 collate 和 DataLoader：正常图按原尺寸构造零掩码，异常原始 ID 在 padding 前以 >0 合并为 1；图像与掩码共用 geometry.py 的 letterbox。训练只做同步水平翻转，增加可强制翻转的核对路径；val/test 禁用随机增强。uint8 RGB 只交给 processor 一次 rescale/normalize，明确关闭 resize 和 reduce_labels，标签独立转 long。返回稳定相对路径 image_id、几何参数、翻转标志，验证原图 GT 作为逐样本列表保留。batch=2、workers=0、seed=42，验收 pixel_values float32[B,3,512,512]、labels int64[B,512,512]、0/1/255、有效区512×390与上下61行255；验证重复读取一致、同步翻转叠加对齐、归一化数值与公式一致、正常图有效区全0且异常前景非空。输出原图/增强叠加/输入与标签的可视化及小型JSON证据；接一次真实B2的eval/no_grad模型前向确认logits[B,2,128,128]，不做优化器更新。禁止在__getitem__中加载模型或操作CUDA，processor只初始化一次。输出中文docs/stages/06-dataset-design-and-implementation.md，更新README与阶段索引；大图、数据与张量放资源目录，小报告留仓库。不实现正式损失、指标或训练器，不加入DINOv3、裁块、额外类别权重或加权采样。

阶段 06 通过后是阶段 07 的 CE+Dice 与原图尺度指标；随后才能进入阶段 08 的 20 步冒烟。
