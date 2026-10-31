# 第六阶段验收：Dataset、同步变换与真实图片前向

日期：2026-10-31。工程：`E:\桌面\AI\AI_Try\work\SegFormer`。

**结论：完善后验收通过，可以进入阶段 07。** 本轮独立执行真实数据检查与 GPU eval 前向，39 项数据契约全通过；修复测试导入路径后，全项目 134 项回归通过。本阶段没有训练成绩、IoU 或 Dice。

## 1. 用户五项检查与决定

| 项目 | 本轮确认与决定 |
| --- | --- |
| flip 与 normalization 严格度 | 实测 uint8 镜像差 **0**、归一化公式差 **0.0**、归一化张量镜像差 **0.0**；类别标签和原尺寸 GT 镜像逐像素完全相等。保留浮点容差 ≤1e-6；RGB 缩放镜像允许 ≤1 LSB 舍入，不能把该容差用于类别标签 |
| 强制翻转图 | 新版 9 张图逐张看过；004 两根弯针从左移右，红色 GT 跟随同一物理引脚。修正了旧图前两列坐标不一致的问题，并增加原尺寸局部放大 |
| original_mask 保留 list | **接受**。B2 uint8 数组 payload 为 **3,004,560 B，约 3.00 MB / 2.87 MiB**，不含 Python 对象开销；训练可关闭保留，验证/测试原图计分需保留或明确重读 |
| num_workers=0 | **接受，继续使用 0。**这条路径已验证，先保障 Windows 可复现性。吞吐在阶段 08 实测后再决定是否另验 2；本轮不声称多 worker 已安全 |
| 按 index 和 epoch 决定翻转 | **保留当前策略**。同一 `(seed, epoch, index)` 可重复，跨 epoch 可改变翻法；每轮开始必须显式 `dataset.set_epoch(epoch)`。不改成每图永远固定方向 |

当前 tensor 镜像容差由 1 LSB、rescale_factor 与最小 std 推导（约 0.017508），用于涵盖 RGB 双线性缩放的最多 1 LSB 差异；本次实际差为 0.0。另有独立的归一化公式检查，阈值仍是 1e-6。零误差是本次结果，不是对所有未来库版本承诺逐位相同。

## 2. 本轮完善

| 原问题或边界 | 处理 |
| --- | --- |
| `--skip-model` 把未执行的前向记 True，可能整阶段假通过 | schema 升至 `/2`；数据诊断可返回 CLI exit 0，但 `scope=dataset_only`、`ok=false`、`full_stage_verified=false`，前向检查为 null、status=skipped |
| 强制图 source RGB 未翻转，旁边 GT 却已翻转 | 前两列统一显示文件方向的 source RGB/GT；后三列显示增强后的输入、标签、叠加，并明确坐标语义 |
| 只验证第一条 geometry，部分字段仅记录不判定 | 对每条几何、padding、有效区、像素有限性、原 mask dtype/list 及元数据列表增加门槛 |
| 三清单哈希检查漏掉已审核风险组 | 复用阶段 04 的校验器，核组源 SHA 与组不跨集；失败在读取样本前停止 |
| processor 使用前未按仓库来源清单验快照 | 先做固定 commit、来源清单及实际模型文件的离线核验，不联网 |
| 重读检查只测 train | 新增真实 val 重读；固定 6 张验证预览，选正常 0011 + 异常 087 做 B2 前向 |
| 训练子集配置可能被忽略 | 未显式传 ids_file 时读取配置 data.train_ids_file；空/重复/不属当前 split 的 ID 拒绝；val/test 不继承训练子集 |
| val/test 可通过 force 绕过增强约定 | 限定 val/test 只能 flip=none，random 和 force 都有实际拒绝检查 |
| 注入 processor 与种子约束不完整 | 内置/注入 processor 均验四开关及三通道有限 mean/std、std>0；默认 seed 来自配置，loader 继承 Dataset seed；校验 seed/epoch/翻转概率 |
| collate 混合有/无 original_mask 时可能静默丢弃 | 明确拒绝混合 batch |
| 证据没有绑定代码与源文件 | 记录配置、清单/meta/风险组、快照来源、代码、抽中图片/掩码和 PNG 的 SHA |
| 根目录 unittest discover 找不到测试辅助模块 | 将辅助导入修为 tests.test_dataset；按 README 的默认 discovery 重跑，134 项通过 |

## 3. 当前数据契约与坐标语义

```text
原图 RGB + 原始 mask
→ mask >0 合并为 0/1（正常图构造全零）
→ 同一个水平翻转决策作用于两者
→ 共用 geometry.py：图像双线性、标签最近邻
→ RGB pad(124,116,104)，label pad255
→ uint8 RGB 只做一次 processor rescale/normalize
→ float32 图像张量 + int64 标签
```

| 字段 | 契约 |
| --- | --- |
| pixel_values | CPU float32 `[B,3,512,512]`，有限 |
| labels | CPU int64 `[B,512,512]`，取值 0/1/255 |
| 有效区 | 512×390；上下各 61 行 255；B2 共 124,928 个 padding 像素 |
| image_id / mask_id | 清单相对路径列表；正常图 mask_id 空串 |
| geometry / flipped | 逐样本 list，几何与增强方向可追溯 |
| original_mask | list 中每项为 uint8 `[1070,1404]`，0/1；**原尺寸，但训练时方向随增强变化** |

`original_mask` 的“original”指分辨率，不保证训练样本仍朝磁盘文件方向。它与当前增强输入一致；val/test 因只允许不翻转，才同时对应磁盘原图坐标。下一阶段主评价用 val/test 原尺寸 GT，不能把训练翻转后的 mask 与未翻转源图混合计分。

list 的价值是保留可变原尺寸、避免无谓堆叠/复制和误搬上 GPU；不是“list 本身会自动省显存”。只有 pixel_values/labels 由调用方按需搬至 GPU，原 GT 与元数据保持 CPU。`return_original_mask=False` 只是不把原尺寸数组留在 batch，异常图训练仍需读取 GT 才能生成 512 标签。

## 4. 视觉验收记录

当前批量图位于 `E:\桌面\AI\segformer_resources\runs\dataset_check`。前三张为 train，后六张为固定 val；本轮均实际查看。

| 图片 | 观察 |
| --- | --- |
| train_normal_0000 | 原图倒置姿态保持一致；源 GT 全零，512 有效区黑、padding 灰，overlay 无红色 |
| train_anomaly_004 | 板左 T、右 R，左侧两根弯针；红线覆盖对应引脚，局部放大可见 |
| train_anomaly_forced_flip_004 | 前两列仍为源方向；后三列文字/T/R 镜像，弯针从左到右，红线同步；无明显错位 |
| val_normal_0011 | 正常零 GT，源图与输入一致 |
| val_normal_0455 | 正常零 GT，无额外翻转或拉伸 |
| val_normal_0983 | 正常零 GT；阶段 04 登记的网格外观疑点保留，不改标 |
| val_anomaly_087 | 板面斜划痕与红线对应，放大图位置一致 |
| val_anomaly_010 | 交叉/弯折引脚与红色 GT 对应 |
| val_anomaly_045 | 两个探头内圆异常区与 GT 对应 |

仓库保留少量便于跨端阅读的[未翻转/强制翻转对照图](./figures/stage06-flip-comparison.png)，由两张当前 PNG 的最后一列裁切并排，没有改变内容；裁切来源、坐标及 SHA 在同名 JSON。

旧版三张图已保存到资源目录 `runs/dataset_check/archive/pre_acceptance`，原 JSON 在 `reports/stage06_pre_acceptance`。它们用于追溯，不作为当前 39 项结果。

## 5. 真实执行结果与范围

- 当前报告 schema：`segformer-lab/dataset-check/2`；`scope=full_stage`、`data_checks_ok=true`、`full_stage_verified=true`、`ok=true`，39 项均为 True。
- 翻转探针为真实非对称 `Anomaly/004`；uint8 差0、归一化张量镜像差0.0，GT严格全等。
- 归一化另一条路径从 uint8 按 `(x/255−mean)/std` 手算，并非张量自比；本例最大差0.0。快照文件另按仓库来源记录校验。
- 验证正常图 0011 有效区全0；异常087原始取值 `{0,3}`，合并后0/1，512有效区前景313 px非空。
- 真正 GPU 前向为固定 val 正常0011 + 异常087：`model_training=false`、`logits_requires_grad=false`，输出 float32 `[2,2,128,128]` 且有限；没有优化器或反向。
- Dataset factory cache 清理后 misses=1，Dataset对象共享同一 processor；范围仅是数据工厂，模型加载器可能另建自己的 processor，不能称整个进程只有一次构造。
- 项目默认回归命令实际运行134项通过；39项数据检查实际运行通过。

本轮抽样实际读取9张唯一原图，并绘制9张图（004包含两种增强方向）；验证全部清单身份不等于重新解码1104张图。test仅构造Dataset、核清单/数量及增强拒绝，没有读取其图片或计算测试成绩。

原有 v2 train/val/test、风险组、固定 val IDs 与 MiT-B0 来源/权重身份均不变；未下载、安装或升级包。134 项回归中包含调用真实本地数据的 CPU 集成检查，因此在新机器上运行完整套件前需要先恢复数据和快照。

## 6. 日常操作与决策

每次新进程通过项目 runner；交互终端先进入工程并加载上下文：

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
. '.\scripts\enter_project.ps1'

# 完整第六阶段检查：真实 eval 前向、39项及9张图
Invoke-SegPython -m segformer_lab.dataset_check `
  --config 'configs\baseline.yaml' --report 'reports\dataset_check.json'
Invoke-SegPython -m unittest discover -v

# 仅数据诊断示例：未在本轮用此命令生成正式验收报告
Invoke-SegPython -m segformer_lab.dataset_check --skip-model `
  --report "$SegResource\runs\dataset_only_check.json" `
  --figure-dir "$SegResource\runs\dataset_only_preview"
```

仅数据诊断的成功退出不等于整阶段验收通过，须读 scope 和 full_stage_verified。离线检查和训练无需代理。

后续训练器每轮须：

```python
for epoch in range(start_epoch, epochs):
    train_dataset.set_epoch(epoch)
    for batch in train_loader:
        ...
```

这里说明调用约定，训练器尚未实现。相同配置、数据顺序、seed与epoch才能复现增强；过滤子集会改变index映射，需保存子集ID及其哈希。worker=0时此更新立即生效；未来若改多worker，需另验Windows spawn、epoch传递与主机内存，不能未经验证开启persistent_workers。

## 7. 下一步指令：只执行阶段 07

给 Agent 指定原工程目录，可直接发送：

> 阅读AGENTS.md、全程指南第9节及第六阶段验收，只执行阶段07。复用隔离环境、当前v2清单、固定快照和Dataset，先核身份，不重划数据。实现二通道CE(ignore_index=255) + 0.5×逐图前景Soft Dice；Dice用FP32 softmax，只平均GT有效区含缺陷的图，整个batch无正类时返回与图相连的零Dice项，正常图仍由CE惩罚误报；全忽略batch明确拒绝，避免NaN。损失先将logits上采样到512标签尺寸；padding不参与分子分母。实现单一原图恢复函数：低分辨率logits先双线性至512，按每条geometry去padding，再双线性至原H/W，align_corners=False，最后softmax取缺陷通道并以≥0.5二值化；GT直接用val/test的original_mask，不缩小GT计分，拒绝未经说明的flipped评价样本。以int64累计TP/FP/FN/TN后算全局前景IoU、Dice、precision、recall，不平均batch分数；分母零返回NA/原因，不加指标平滑项。另报正常图FP像素比例及逐图预测前景面积比例中位/P95，异常图宏平均单独列；正常GT且预测空的逐图IoU/Dice写NA。用手算2×2例子、全对/全背景/全前景、正常误报、忽略像素、全正常Dice可反向、padding概率变化不影响loss、不同batch划分累计一致，以及非对称原图恢复夹具验证。将一个原GT单像素放在512采样会消失的位置，确认原图计分仍记FN，不能用放大回去的缩小GT冒充原GT。输出中文docs/stages/07-loss-metrics-design-and-implementation.md、小型JSON与真实回归日志，更新README和阶段索引。大图/张量放资源目录。不实现训练器、优化器更新或20步smoke，不报告随机头在真实数据上的分数为基线性能。

阶段07通过后，阶段08才以batch2、workers0、AMP及固定训练8图完成20次optimizer更新冒烟；当前不要跳到正式训练。
