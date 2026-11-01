# 阶段 08：20 步 CUDA AMP 冒烟与保存重载核验

| 项目 | 值 |
| --- | --- |
| 创建/更新时间 | 2026-11-01；本轮独立验收修订 |
| 对应步骤 | 主指南第 10 节「步骤 8：20 步冒烟检查」 |
| 前置状态 | 阶段 01—07 已验收（07 为 42 项检查、186 项回归）；v2 清单与 MiT-B0 快照未变 |
| 本阶段范围 | 固定 8 图、AMP 训练 20 次成功更新、B1 FP32 通路评价、保存后新进程严格重载 |
| 明确不做 | 不评价泛化、不读 test 图像/掩码、不做测试推理、不启动 pilot 或正式训练 |
| 对应代码身份 | 以下原数值属于 `smoke_001`；原记录源码与提交的差异及当前代码身份，见本文末的独立验收补充和 `reports/stage09-acceptance.json` |

## 1. 本阶段解决什么问题

阶段 07 结束时，损失与指标已经用人工夹具证明"定义正确"，但**从未真正跑过一次优化器更新**。本阶段要证明的是整条通路：真实图片 → 512 letterbox → CUDA AMP 前向 → 组合损失 → `GradScaler` 反向 → 梯度裁剪 → AdamW 更新 → B1 FP32 评价 → 保存 → **新进程严格重载**。

20 次更新不是为了出成绩，而是为了在几秒内暴露尺寸、dtype、NaN、显存和保存格式的问题。这一阶段的分数不能称为基线。

## 2. 技术栈与版本

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | 3.12.16 | `segformer_resources\env` |
| PyTorch | 2.10.0+cu126 | `torch.amp.GradScaler('cuda')`、`torch.autocast` |
| transformers | 4.57.6 | MiT-B0 编码器 + 二类解码头，参数 3 714 658 全可训练 |
| 硬件 | RTX 4070 Laptop | 总显存 8187.5 MiB |
| 新增依赖 | **无** | 未安装、未升级任何包 |

## 3. 设计与取舍

### 3.1 固定 8 张训练图（两阶段共用）

`manifests/overfit_train_ids.txt` 由 `segformer_lab.subset` 一次生成，SHA-256 `2ac81b81…`，之后不再变动：

| 类型 | 图片 | 原图前景 | 占比 | 512 有效区前景 |
| --- | --- | ---: | ---: | ---: |
| normal | 0025 / 0121 / 0708 / 0812 | 0 | 0 % | 0 |
| anomaly | 065 | 1 150 px | 0.077 % | 155 |
| anomaly | 038 | 2 517 px | 0.168 % | 331 |
| anomaly | 031 | 3 409 px | 0.227 % | 443 |
| anomaly | 043 | 191 178 px | 12.726 % | 25 389 |

异常图按"原图前景像素排序后取 0 %/33 %/66 %/100 % 排名"选出，跨两个数量级，覆盖最小划痕与最大面积缺陷。正常图用 `random.Random(42).sample` 在按路径排序的正常清单中抽取。明细（含每张源图与掩码 SHA、清单与快照身份）在 `reports/overfit_subset.json`。

选择全程只用 train 清单，未读 val/test 图像。

### 3.2 关闭增强必须显式写

训练 `Dataset` 默认 `flip='random'`，所以两阶段都显式传 `flip='none'`，配置里也写 `train.flip: none` 与 `data.horizontal_flip_p: 0.0`。否则 8 张图在不同 epoch 会被随机镜像，"同一组图片"这个前提就不成立。

### 3.3 "成功更新"的定义

当前 AdamW 的 `GradScaler.step()` 返回值不能判断是否执行。原实现用**前后 scale 比较**辅助判定：

```python
scale_before = scaler.get_scale()
scaler.step(optimizer)
scaler.update()
scale_after = scaler.get_scale()
skipped = scale_after < scale_before
```

并在第一次成功更新时记录编码器与解码头的梯度范数、以及更新前后参数的 `max|Δ|`。计数上区分 `attempts`（尝试）与 `successful`（真正更新）。成功目标 20，尝试上限 30，多余尝试只允许容纳记录在案的 AMP 校准跳步。

独立验收已进一步改为 optimizer post-hook 观察真实执行，并核对 scaler 的异常决定、跳步时参数/优化器计数未变。总范数 inf 不必然等于 scaler 确认的溢出，不能仅据此写 skipped；实现详见[阶段 09 文档](./09-overfit-design-and-implementation.md#5-amp-跳步修复必须区分两类异常)。

### 3.4 评价路径复用阶段 07，不重写

`evaluate_dataset` 里：`model.eval()` → `torch.inference_mode()` → `with evaluation_precision(config, device='cuda')`，逐张 B1 前向；概率恢复与计分全部走 `metrics.py`。`precision_record` 同时记录 TF32 关闭前后状态（本机卷积默认 `cudnn.allow_tf32=True`）。

### 3.5 保存必须自包含

run 内保存 `model_config.json`、`processor/`、`checkpoints/model_state.pt`（仅模型张量）、`checkpoints/training_state.pt`（含优化器与 GradScaler 状态）、`resolved_config.yaml`。provenance 位于 `summary.json` 字段中，并非独立文件。重载入口 `load_run_model` 用 run 内结构与权重严格加载；本轮还确认 Dataset 实际使用保存的 processor。

## 4. 文件与接口

| 文件 | 作用 |
| --- | --- |
| `segformer_lab/training.py` | seed、AdamW 分组、AMP 训练循环、B1 FP32 评价、保存与严格重载 |
| `segformer_lab/smoke.py` | 阶段 08 训练入口 + `--verify-run` 新进程对照入口 |
| `segformer_lab/subset.py` | 生成并校验固定 8 图 ID 文件与明细 |
| `configs/smoke.yaml` | 更新预算、精度策略、固定 ID 文件、固定学习率 |
| `manifests/overfit_train_ids.txt` | 8 个 image 相对路径（阶段 09 复用同一文件） |
| `reports/overfit_subset.json` | 8 图明细与哈希 |
| `reports/stage08-gates.json` | 逐项门禁核对结果 |

run 结构（`E:\桌面\AI\segformer_resources\runs\smoke_001`）：

```text
summary.json / updates.jsonl / resolved_config.yaml / artifact_manifest.json
model_config.json / processor/ / checkpoints/{model_state,training_state}.pt
pre_save_probabilities.npz   # 6 张图全分辨率 FP32 概率，约 29 MB
eval_smoke/                  # 6 张三联图
reload_check.json            # 新进程重载对照
```

## 5. 关键实现讲解

### 5.1 训练步顺序

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits = model(pixel_values=pixels).logits
    terms = segformer_loss(logits, labels, ...)
scaler.scale(terms.total).backward()
scaler.unscale_(optimizer)
grad_norm = clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

`unscale_` 必须在裁剪之前，否则 `clip_grad_norm_` 裁剪的是放大了 65 536 倍的梯度，阈值 1.0 会完全失真。本次运行没有发生跳步，`scale` 始终 65 536。

### 5.2 优化器分组

AdamW 按名字分两组：`.bias` 与含 `norm` 的参数 `weight_decay=0`，其余 `0.01`。这是本实验明确采用的配方，不保证对所有任务都优于统一 decay。

### 5.3 显存与计时的口径

`torch.cuda.reset_peak_memory_stats()` 在训练前调用；每步前后 `cuda.synchronize()`；首步与稳态分开记录。首步可能包含首次训练内核、工作区和优化器状态准备，不能直接当稳态速度；CUDA 上下文也可能早在模型搬至 GPU 时已建立。

## 6. 如何操作与实际执行了什么

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
. '.\scripts\enter_project.ps1'

# 固定子集（一次性；--verify-only 可复核）
Invoke-SegPython -m segformer_lab.subset --config 'configs\baseline.yaml'

# 阶段 08 训练 + 保存
Invoke-SegPython -m segformer_lab.smoke --config 'configs\smoke.yaml' `
  --out 'E:\桌面\AI\segformer_resources\runs\smoke_001'

# 新进程：从保存物严格重载并复算
Invoke-SegPython -m segformer_lab.smoke --verify-run 'E:\桌面\AI\segformer_resources\runs\smoke_001' `
  --config 'configs\smoke.yaml'
```

实际执行：

| 次序 | 操作 | 结果 |
| --- | --- | --- |
| 1 | 生成固定子集 | 8 张（4N+4A），ids SHA `2ac81b81…` |
| 2 | 首次训练 | 20 步跑完，但在写 artifact manifest 时 `KeyError` 中断 |
| 3 | 修复路径分隔符后重跑 | **20/20 成功，0 跳步，21.1 s** |
| 4 | 新进程重载 | `strict` missing 0 / unexpected 0，概率差 0.0，XOR 0 |
| 5 | 门禁落盘 | 5/5 技术门禁通过（`reports/stage08-gates.json`） |

## 7. 真实验证证据

### 7.1 训练计数与梯度

| 项目 | 实测 |
| --- | --- |
| 成功更新 / 尝试 / 跳步 | **20 / 20 / 0** |
| 停止原因 | `target reached (20 successful updates)` |
| 编码器梯度范数（首步） | `segformer.encoder.patch_embeddings.0.proj.weight` = **0.0931** |
| 解码头梯度范数（首步） | `decode_head.linear_c.0.proj.weight` = **0.0253** |
| 参数真实变化 `max|Δ|` | 编码器 **6.014e-05**、头 **6.004e-05**（`parameters_changed=true`） |
| 学习率 | 6e-5 固定，无 warmup/scheduler |
| 编码器身份 | 192/192 张量与检查点完全一致，`max_abs_diff = 0.0` |

两处 `max|Δ|` 都约等于 `lr × 1`，符合 AdamW 一步更新幅度的预期，说明参数确实被改过，而不是"调用过 step 就算更新"。

### 7.2 损失曲线（20 步）

CE 从 **0.6350 → 0.2182**，总损失从 **1.1331 → 0.5937**；Dice 项在全正常 batch 上为 0（第 2、6、11、17 步），符合定义。

### 7.3 资源与计时

| 项目 | 实测 |
| --- | --- |
| 首步耗时 | 3 423.02 ms（含 CUDA 初始化） |
| 稳态中位 | **58.5 ms/update** |
| 20 步墙钟 | 5.701 s |
| 峰值 allocated / reserved | **806.5 / 934.0 MiB** |
| 训练集 | 8 张，B2，workers 0，累积 1 |
| 系统内存 | 总 15.22 GiB，可用 1.73 GiB，负载 88 % |

**这个 934 MiB 不能当作 50 轮训练的显存预算**：它没有包含 132 张验证集的循环、预测概率缓存与 TensorBoard。但它的确证明 B2/512/AMP 在本机没有 OOM。

### 7.4 评价（仅通路，不是基线）

6 张固定 val 预览（3 正常 + 3 异常），B1 FP32：

| 项目 | 值 |
| --- | --- |
| 累计像素 | 9 013 680 |
| TP / FP / FN / TN | **0 / 264 / 126 065 / 8 887 351** |
| 前景 IoU / Dice | 0.0 / 0.0 |
| 正常图误报像素 | 264（FP 像素比 5.86e-5） |
| TF32 | `before: cudnn=True` → `effective: False`；autocast 关闭 |
| 图 | `eval_smoke/` 6 张三联图（source / 原图 GT / 阈值叠加） |

随机头训练 20 步后基本预测背景，只有 264 个误报像素，这符合预期，**不构成失败，也不是性能结论**。

### 7.5 保存重载

| 项目 | 值 |
| --- | --- |
| 加载方式 | run 内 `model_config.json` + `model_state.pt`，`strict=True` |
| missing / unexpected | 0 / 0 |
| 6 张图最大概率差 | **0.0**（容差 atol 1e-5） |
| 二值预测 XOR | **0 / 9 013 680** |
| 混淆量与全局指标 | 完全一致 |
| 步数 | 保存记录 20，与重载一致 |

保存与重载是两个独立进程，中间没有共享内存。这条是"训练产物能被别人加载"的硬证据。

## 8. 遇到的问题与处理

| # | 现象 | 根因 | 处理 |
| --- | --- | --- | --- |
| 1 | 首次跑完 20 步后在保存阶段 `KeyError: 'checkpoints/model_state.pt'` | `path.relative_to(run_dir)` 在 Windows 上返回反斜杠路径，与 manifest 里的 POSIX key 不一致 | 改用 `.as_posix()`；失败 run 移入 `runs/archive/smoke_001_aborted_artifact_key` 保留，重跑得到当前有效 run |
| 2 | （预防）把尝试次数当更新次数 | `GradScaler.step()` 返回 `None` | 用前后 scale 比较判定跳步，并记录首步梯度与参数变化 |

以下是阶段 08 首次执行时的未验证项；本轮修正后的状态见第 11 节：

- 没有测过 AMP 跳步的真实路径（本次 0 跳步）；该分支有日志与"连续 3 次跳步即停"的保护，但没有实测触发。
- `workers=0` 仍是唯一验证过的设置。
- 没有做任何 val 全量或 test 评价。

## 9. 给用户的学习要点

1. **"跑通"与"学得好"要分开验收。** 本次 20 步的 IoU 是 0，短训效果不足本身不构成通路失败；不能断言所有随机头训练 20 步都会全背景。此阶段看梯度、真实更新和新进程重载。
2. **为什么必须记录首步与稳态两个耗时。** 首步 3.4 s、稳态 58 ms，差 58 倍。用首步推算 50 轮会得到荒谬的预算。
3. **为什么保存结构配置和 processor，不能只存 state_dict。** 没有 `model_config.json`，别人无法知道 2 类、ignore 255、dropout 设置；没有 processor，推理时的归一化常数就要靠猜。
4. **为什么跳步判定要看 scale。** `step()` 返回值在两种情况下都是 `None`，不检查 scale 就可能把"根本没更新"写成"第 N 步完成"。

## 10. 下一阶段与修订记录

阶段 09（已获授权，本次同轮串行）：另建 `overfit.yaml` 与新 run，从固定 MiT-B0 快照**重新初始化**，LR 1e-4 无衰减，同一 8 图无增强，更新 0/每 50 次评价，100/200 检查学习，最多 500 次尝试。不加载本 run 的权重或优化器。

| 日期 | 变更 | 说明 |
| --- | --- | --- |
| 2026-11-01 | 初版 | 20 步冒烟通过，5/5 技术门禁，新进程重载零差异 |
| 2026-11-01 | 独立验收 | 修保存 processor 注入与真实 step 判定；原 run 和新 smoke 分别新进程复验，最终完整回归 207 项通过 |

## 11. 本轮独立验收补充

第八、九阶段已通过[联合验收](../../reports/2026-11-01-stage08-09-acceptance.md)。原 `smoke_001` 的旧重载没有将保存 processor 传给 Dataset，本次 `run_check` 明确使用它，模型和训练状态全部核对后，6 张图概率差 0.0、XOR 0，原图指标一致。旧报告不能单独作为已验证预处理保存闭环的充分证据。

共享训练步修正后，另建 `smoke_003_stage09_acceptance`，真实 CUDA 回归仍为 20 成功/20 尝试/0 跳步，首步编码器与头均有梯度和参数变化，峰值 allocated/reserved 为 806.5/934 MiB；新进程重载同样概率差 0.0、XOR 0。该次训练步中位数 103.0 ms、墙钟 6.711 s，与旧速度分开记录，不声称新旧权重按位相同。

当前代码的 11 项训练 CPU 回归覆盖了真实 GradScaler 跳步、有限元素范数溢出、AMP 关闭后的非有限拒绝、连续跳步复位、参数有限性和模式恢复；完整项目回归为 207 项。新报告见 `reports/stage08-reload-check.json`、`stage09-smoke-reload-check.json`，小型运行原件在 `reports/stage08_09_evidence`。

原 `smoke_001` 记录的训练源码 SHA 为 `47551c0a…`，与阶段 08 Git 提交 `458be482…` 不同，准确历史源码未恢复。这是历史复现局限；本轮新回归有当前代码 SHA，但不将它伪装成原实验源码。完整 epoch resume、workers>0 与全量训练仍未验证。阶段 09 现也已验收，下一步仅[阶段 10](../../reports/2026-11-01-stage10-execution-handoff.md)。
