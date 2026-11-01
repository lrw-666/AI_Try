# 阶段 10：完整训练集三轮 pilot——设计与实现学习文档

| 项目 | 值 |
| --- | --- |
| 创建/更新时间 | 2026-11-01；独立验收修订 |
| 对应步骤 | 主指南第 12 节「步骤 10：完整训练集短程试跑」 |
| 前置状态 | 阶段 01—09 已通过独立验收（[联合验收](../../reports/2026-11-01-stage08-09-acceptance.md)；207 项回归） |
| 本阶段范围 | 530 张 train 全量、3 个完整 epoch、`flip: random` + 每轮 `set_epoch`、warmup+cosine、每轮 132 张 val 全量评价、last/best 保存、新进程复验 |
| 明确不做 | 不调超参、不加裁块/加权采样/后处理/DINO、不做 50 轮正式训练、不读 test 图像或掩码、不做测试推理、不实现 resume |
| 状态 | **已通过独立验收**；当前231项回归与完整CUDA重载通过，见[验收报告](../../reports/2026-11-01-stage10-acceptance.md) |
| 对应代码身份 | 记录运行使用 `pilot.py=52e3aae7…`、`training.py=91c829d2…`，已保存准确历史文本；当前验收修正及 SHA 见 `reports/stage10-acceptance.json`。配置 SHA `74f4c8f1…` 未变；记录运行 HEAD 为 `0c5f0bd` |
| run 目录 | `E:\桌面\AI\segformer_resources\runs\pilot_001`（约 134 MiB，不进 Git） |

## 1. 本阶段解决什么问题

阶段 09 证明的是"同一批 8 张图能拟合"。它故意关掉了增强、只用一个固定子集、只跑 200 次更新、评价也在同一批图上。**这些边界一旦拿掉，会有一整套新问题冒出来**：

- 530 张图一轮能不能被完整遍历？`drop_last=False` 下最后一批是 2 张，会不会被静默丢掉？
- 增强恢复成 `flip: random` 后，每轮的翻转图案会不会真的变？图像和掩码是不是同步翻的？
- 132 张 val 全量评价要多久、占多少内存？会不会因为想缓存全部原尺寸概率而把 1.7 GiB 可用内存打爆？
- warmup+cosine 在"按 epoch 计预算"时，步数怎么算、第几次更新到 base_lr、最后一次是不是 min_lr？
- 一个 epoch 边界上保存的 best/last，能不能被**另一个进程**严格加载并复算出同样的原图指标？

本阶段的产出就是回答这些。**它不是 baseline**：3 轮、总更新 795 次，模型刚把大缺陷找出来，12 张异常验证图里还有 7 张一个像素都没报（见 7.5）。任何"验证 IoU 0.608"都不该被当成论文成绩。

## 2. 技术栈与版本

| 组件 | 版本 | 本阶段作用 | 状态 |
| --- | --- | --- | --- |
| Python | 3.12.16 | `segformer_resources\env` | 已运行 |
| PyTorch | 2.10.0+cu126 | AMP + `GradScaler`、AdamW、`AdamW` 分组、`inference_mode` | 已运行 |
| transformers | 4.57.6 | SegFormer-B0 二类头、`SegformerImageProcessor` | 已运行 |
| matplotlib | 3.11.2 | 6 张 val 预览三联图（`Agg`） | 已运行 |
| NumPy / Pillow | 2.5.4 / 12.3.0 | 概率恢复、原图读取、加权聚合 | 已运行 |
| GPU | RTX 4070 Laptop 8 187.5 MiB | B2/512/AMP 训练 + B1 原图评价 | 已运行 |
| 新增依赖 | **无** | 未安装、未升级任何包 | — |

## 3. 设计与取舍

### 3.1 为什么不给 `train.py` 加一个 `if pilot:` 分支

`train.py` 的每条硬性检查都是阶段 09 的**定义**：必须是 8 张图、必须 `flip: none`、必须 `eval.precision: ieee_fp32`、必须按成功更新计数。为了跑 pilot 把这些检查放宽，等于把阶段 09 结论的前提拆掉——以后没人能说清 `overfit_002` 到底是在什么约束下跑出来的。

所以新增 `segformer_lab/pilot.py`，它**只拥有阶段 09 没有的那部分控制流**：epoch 循环、`set_epoch`、调度器、逐轮全量验证、last/best。所有数值语义仍然来自共享模块：

| 关注点 | 谁实现 | 阶段 10 是否重写 |
| --- | --- | --- |
| 数据、增强、letterbox、collate | `data.py`（阶段 06 验收） | 否 |
| 损失 CE + 0.5×逐图 Dice | `losses.py`（阶段 07 验收） | 否 |
| 原图恢复、混淆量、NA | `metrics.py`（阶段 07 验收） | 否 |
| IEEE FP32 评价上下文 | `evaluation.py`（阶段 07 验收） | 否 |
| 单步前向/损失/反传/裁剪/step | `training.train_step` + `step_optimizer` | **本轮从这里抽出来的共享函数** |
| 评价循环 | `training.evaluate_dataset`（B1、eval、inference_mode） | 否 |
| 预览图 | `smoke._render_samples` | 否 |
| epoch 控制流 / 调度 / 选 best / checkpoint | `pilot.py` | 新增（唯一新逻辑） |

为了让 08/09 与 10 真正共用一步，我把原来写在 `run_training` 循环体里的"前向 + 损失 + 反传 + 裁剪 + step"抽成 `train_step()`，`run_training` 改为调用它。阶段 09 的 `train.py` 一行没改，仍走 `run_training`。

### 3.2 调度器：先定义清楚第几次更新用哪个 LR

`WarmupCosineLR` 只有一个状态量 `completed`（已真正执行的 optimizer 更新数）：

```text
warmup_steps = max(1, round(warmup_ratio × total_steps)) = max(1, round(0.05 × 795)) = 40

completed <  40 :  lr = base_lr × (completed + 1) / 40
completed >= 40 :  progress = (completed − 40 + 1) / (795 − 40)
                   lr = min_lr + 0.5 × (base_lr − min_lr) × (1 + cos(π × progress))
```

off-by-one 说清楚：

- 第 1 次更新（`completed = 0`）用 `6e-5 × 1/40 = 1.5e-6`；
- 第 40 次更新（`completed = 39`）**正好等于** `6e-5`，warmup 期间任何一步都不会超过 base_lr；
- 第 41 次（`completed = 40`）进入 cosine，`progress = 1/755`，几乎还是 base_lr；
- 第 795 次（`completed = 794`）`progress = 755/755 = 1`，**正好等于** `min_lr = 1e-6`。

三条必须坚持的规则：

1. **只有 optimizer 真的执行了才 `step()`**。`advance_schedule()` 遇到 `skipped=True` 直接返回，不改 `completed`、不改 lr。AMP 跳步时权重没动，学习率却往前走，等于凭空加速。
2. **总步数按"计划预算"算，不按"实际成功数"算**。pilot 的计划是 3×265 = 795，这样第 3 轮末尾的 LR 一定落到 min_lr；若中途有跳步，成功更新数会小于 795，最后一次就不在 `progress = 1` 上——这是"调度不因跳步前进"的必然结果，如实记录，不去补偿。
3. **pilot 与正式训练是两套 schedule**。50 轮会重新算 `total_steps = 50 × 265`，warmup 也变成 662 步。两者曲线不同，不能写成"pilot 是正式训练的前 3 轮"。

### 3.3 按 epoch 完整遍历，不用"补到 795 次成功更新"

阶段 09 的预算单位是"成功更新次数"。到阶段 10 这样会出问题：如果第 1 轮末尾跳了几步，用"补步"的方式凑满 795，某些图就会被多看一遍、另一些少看一遍，"每轮 530 张各一次"的前提就没了。

所以循环写成 `for epoch in range(1, 4): for batch in loader:`，`drop_last=False`，并在每轮结束时**断言**：见过的 `image_id` 集合等于 train 清单集合、无重复、共 530 张。不满足就抛错停下，而不是继续训练。

### 3.4 增强：`set_epoch` 决定翻转图案

`data.flip_decision(mode, seed, epoch, index, p)` 用 `np.random.default_rng([seed, epoch, index])` 决定该样本是否翻转——同一个 `(seed, epoch, index)` 永远给同一个答案，换 epoch 就换图案。因此每轮开头必须 `train_ds.set_epoch(epoch)`，否则三轮拿到的是同一套增强。翻转同时作用于图像和掩码（`data.py` 里 `image.transpose(FLIP_LEFT_RIGHT)` 与 `binary[:, ::-1]` 成对出现），本轮用覆盖审计确认翻转率与图案变化（7.1）。

val 数据集固定 `flip="none"`，`metrics` 还会拒绝 `flipped=True` 的样本，从代码层面堵住"验证集被增强污染"。

### 3.5 损失聚合：不能用 batch 均值的均值

`segformer_loss` 返回的 `ce` 是**该批有效像素上的均值**，`dice` 是**该批正 GT 图上的均值**。直接把 265 个 batch 的 `ce` 求平均，等于给每批同样权重——而每批的有效像素数其实一样（都是 2×512×512 减去 24.98 % padding），所以 CE 这块差别不大；但 Dice 差别很大：全正常 batch 的 `dice` 是精确 0（图相连的零），它会被算进平均，把真实的 Dice 拉低。

本阶段按定义加权：

```text
epoch_CE   = Σ_batch (ce_batch × valid_pixels_batch) / Σ_batch valid_pixels_batch
epoch_Dice = Σ_batch (dice_batch × positive_images_batch) / Σ_batch positive_images_batch
epoch_total = 1.0 × epoch_CE + 0.5 × epoch_Dice
```

因为 CE 是"均值的加权平均"，上式严格等于"全部有效像素上的 mean"；Dice 同理严格等于"全部正 GT 图上的 mean"。每轮 `valid_pixels` 恒为 105 830 400（530 张 × 399 360），`positive_images` 恒为 48（train 的 48 张异常）。

### 3.6 只按 val 全局前景 IoU 选 best

`eval.selection_metric: foreground_iou_global`。每轮在 **132 张 val 全量**上算原图尺度指标，`iou > best_iou` 才算新 best（并列保留更早的 epoch），初始 best 为 -1.0，保证第一轮一定保存。**不用阶段 09 的 0.946 当达标线**——那是训练子集上的数，和留出验证没有可比性。

### 3.7 保存 last 与 best，但不假装支持 resume

`checkpoints/last.pt` 每轮覆盖写，`checkpoints/best.pt` 只在刷新 best 时写；两者都用"先写 `.tmp` 再 `os.replace`"的原子替换。payload 包含：模型张量、`model_config`、优化器状态、`GradScaler` 状态、调度器状态、完成的 epoch、成功更新/尝试/跳步数、best 记录、该轮验证指标、Python/NumPy/Torch/CUDA RNG 状态、身份信息。

**支持 resume 需要的不止这些**（还要能重建 loader 的采样顺序、验证过一个完整 epoch 边界、证明恢复后指标一致），本阶段没做，也没有验证，所以 `summary.json` 里直接写 `resume.implemented: false`。保存了 optimizer 不等于"断点续训已验证"。

### 3.8 冻结运行期的代码与配置

一次跑 3.5 分钟，最容易出的低级问题是"跑到一半改了代码"。所以运行开始与结束各写一份 `code_identity_{start,end}.json`：13 个源文件 SHA、配置 SHA、三份清单 SHA、`split_meta`/预览 ID SHA、Git HEAD、`git status --porcelain`、`git diff` 的全文与 SHA。结束时比较，`source_and_config_unchanged` 必须为 true（本轮为 true）。

## 4. 文件、接口与数据约定

| 文件/模块 | 新增或修改 | 职责 | 关键接口 |
| --- | --- | --- | --- |
| `segformer_lab/pilot.py` | 新增并补强独立复验 | epoch 循环、调度、逐轮全量验证、last/best、复验 | `--config --out [--device]`；`--verify-run <run> --config --report <独立JSON>` |
| `segformer_lab/training.py` | **修改** | 抽出 `train_step()`；`WarmupCosineLR`；`step_optimizer` 转公开名（保留 `_step_optimizer` 别名） | `train_step(model, batch, config, device, optimizer, scaler, grad_clip=, amp=, attempt=) -> (terms, evidence, step_ms)` |
| `configs/pilot.yaml` | **新增** 80 行 | 全量数据、3 轮、增强、调度、评价口径 | `train_ids_file: null`，`expected_train_images: 530` |
| `tests/test_pilot.py` | 新增并补齐验收回归 | 24项CPU回归：调度、加权聚合、配置、保存状态/概率/身份门槛 | `python -m unittest tests.test_pilot` |
| `reports/stage10-summary.json` | 小型JSON | 三轮覆盖/损失/指标/耗时/复验的小型证据 | 进 Git |
| `reports/stage10-unit-tests.txt` | 完整回归日志 | 当前231项，原219项日志另行归档 | 进 Git |
| `reports/figures/stage10-*.png` | **新增** 3 张 | 精选预览（漏检 / 命中 / 正常） | 进 Git |

run 结构（`segformer_resources\runs\pilot_001`，`artifact_manifest.json` 记录 30 个产物；加清单自身及之后写入的 `reload_check.json`，实际共 32 个文件。整体约 134 MiB）：

```text
summary.json                 三轮全部指标、耗时、内存、身份、局限
epochs.jsonl                 逐轮记录（含 132 张逐图分数）  219 KB
updates.jsonl                795 条逐次尝试记录（lr/scale/skip/batch_ids）
code_identity_start.json     运行前代码与配置指纹
code_identity_end.json       运行后指纹 + 是否被改过
resolved_config.yaml         绝对路径解析后的配置
model_config.json            best 对应的结构（JSON 可读）
processor/                   best 使用的预处理配置
checkpoints/best.pt          按 val IoU 选出（epoch 3）
checkpoints/last.pt          最后一轮
checkpoints/model_state.pt   仅权重镜像（严格加载入口）
pre_save_probabilities.npz   6 张固定预览的原尺寸概率（30 MB）
val_previews/epoch_0{1,2,3}/ 每轮 6 张三联图
reload_check.json            新进程复验结果
```

数据与张量约定（沿用阶段 06/07，未改）：训练批 `pixel_values float32[2,3,512,512]`、`labels int64[2,512,512]`（0/1/255，有效区 `[2,390,512]`，上下各 61 行 255）；评价 B1、原图 `1070×1404`、阈值 0.5、GT 用 `original_mask`。val 全量 198 300 960 像素，其中正常图 180 273 600、异常图 18 027 360。

## 5. 关键实现讲解

### 5.1 共享的一步

```python
# segformer_lab/training.py
def train_step(model, batch, config, device, optimizer, scaler, *, grad_clip, amp, attempt=None):
    where = "" if attempt is None else f" at attempt {attempt}"
    pixels = batch["pixel_values"].to(device, non_blocking=False)
    labels = batch["labels"].to(device, non_blocking=False)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(torch.cuda.current_device())
    tick = time.perf_counter()

    terms = compute_loss_terms(model, pixels, labels, config, amp=amp)
    if not _finite(terms.total):
        raise RuntimeError(f"non-finite loss{where}: {float(terms.total.detach())}")

    scaler.scale(terms.total).backward()
    evidence = step_optimizer(model, optimizer, scaler, grad_clip)
    torch.cuda.synchronize(torch.cuda.current_device())
    return terms, evidence, (time.perf_counter() - tick) * 1000.0
```

返回 `(terms, evidence, step_ms)`：`terms` 给日志聚合，`evidence` 里 `skipped / optimizer_step_executed / parameters_unchanged / optimizer_steps_unchanged / scale_before/after` 给"这一步到底有没有更新"作证。**计数、跳步判定、调度推进都留给调用方**——这正是 09（按更新数停）与 10（按 epoch 停）能共用算术、又不共用控制流的原因。

### 5.2 一次 attempt 的完整顺序

```python
terms, evidence, step_ms = train_step(...)
tally.add(ce, dice, terms.valid_pixels, terms.positive_images)   # 所有 attempt 都计入
if evidence["skipped"]:
    skipped_total += 1; consecutive_skips += 1
    if consecutive_skips >= 3: raise RuntimeError("three consecutive AMP skips ...")
    continue                                                     # 不推进调度、不算成功
consecutive_skips = 0
successful += 1
advance_schedule(schedule, evidence, optimizer)                  # 只有这里改 LR
```

三个细节：**跳步的那次也计入 loss 聚合**，因为它真实前向过、真实覆盖了那两张图，排除它会让"每轮覆盖 530 张"的账对不上；连续 3 次跳步直接抛错并写 `failure.json` 保留现场（本轮 0 次）；`lr_used` 在 `train_step` 之前取，所以日志里记的是"这一步用的 LR"，不是"下一步的 LR"。

### 5.3 每轮结束的覆盖审计

```python
coverage = {
    "images_seen": len(seen_ids), "unique_images_seen": len(seen_set),
    "missing_ids": sorted(expected - seen_set)[:5], "duplicate_visits": len(seen_ids) - len(seen_set),
    "batch_size_last": len(seen_ids) - batch_size * (epoch_attempts - 1),
    "dataset_epoch": train_ds.epoch, "flipped_images": len(flips),
    "flip_signature_sha256": sha256_file_text("\n".join(sorted(flips))),
}
if coverage["missing_ids"] or coverage["unexpected_ids"] or coverage["duplicate_visits"]:
    raise RuntimeError(...)
```

`flip_signature_sha256` 是"这一轮翻转了哪些图"的指纹，三轮指纹不同，结合epoch值及确定性重算可验证增强计划变化；它不等于逐张查看所有运行时像素；`batch_size_last` 证明最后一批没被 `drop_last` 吞掉。

### 5.4 加权聚合

```python
class _LossTally:
    def add(self, ce, dice, valid_pixels, positive_images):
        self.ce_sum += ce * valid_pixels;        self.ce_pixels += int(valid_pixels)
        self.dice_sum += dice * positive_images; self.dice_images += int(positive_images)
    def as_dict(self, *, ce_weight, dice_weight):
        ce = self.ce_sum / self.ce_pixels                      # 全像素口径
        dice = self.dice_sum / self.dice_images if self.dice_images else None
        return {"ce": ce, "dice": dice, "total": ce_weight * ce + dice_weight * (dice or 0.0), ...}
```

跨轮比较时这个口径很重要：第 1 轮 48 张正 GT 全部参与，第 3 轮也是 48 张，分母一致，曲线才可比。

## 6. 如何操作与实际执行了什么

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
. '.\scripts\enter_project.ps1'
$ErrorActionPreference = 'Continue'     # dot-source 之后必须重置，否则 conda 的 stderr 警告会拆掉管道

# 0) 接口核对（新增入口，先 --help，不把未跑过的接口写成结果）
Invoke-SegPython -m segformer_lab.pilot --help

# 1) 三轮 pilot
Invoke-SegPython -u -m segformer_lab.pilot --config 'configs\pilot.yaml' `
  --out 'E:\桌面\AI\segformer_resources\runs\pilot_001'

# 2) 新进程：严格加载 best.pt + run 内 processor，重算全量 val 与 6 张预览
Invoke-SegPython -u -m segformer_lab.pilot --verify-run 'E:\桌面\AI\segformer_resources\runs\pilot_001' `
  --config 'configs\pilot.yaml' --report 'reports\stage10-reload-check.json'

# 3) 项目回归
Invoke-SegPython -m unittest discover -s tests -t . -v
```

| 次序 | 操作 | 实际结果 | 证据 | 通过 |
| --- | --- | --- | --- | --- |
| 1 | `--help` | 打印 `--config/--out/--verify-run/--device`，退出 0 | 终端 | 是 |
| 2 | 单元回归（新增 12 项） | 12/12 通过 | `tests/test_pilot.py` | 是 |
| 3 | 全套回归 | **219 项，13.737 s，OK**（记录运行时） | `reports/stage10_evidence/original-unit-tests.txt` | 是 |
| 4 | 三轮 pilot | **795/795 更新，0 跳步，208.17 s，退出 0** | `runs\pilot_001\summary.json` | 是 |
| 5 | 新进程复验 | 严格加载 0 missing/0 unexpected；val 混淆量与全局指标逐位相同；6 张预览概率差 0.0、XOR 0 | `runs\pilot_001\reload_check.json` | 是 |

资源隔离：解释器 `...\env\python.exe`；缓存 `...\cache`；临时 `...\tmp`；run、checkpoint、预览、npz 全在 `...\runs\pilot_001`；仓库保留代码、配置、测试、小型汇总和日志，以及少量精选图；独立验收补充了原始小型证据。**未安装/升级任何包，未改 base、未改全局配置，未覆盖任何已有 run。**

## 7. 验收与当前状态

### 7.1 数据覆盖与增强（每轮）

| 项目 | 第 1 轮 | 第 2 轮 | 第 3 轮 |
| --- | ---: | ---: | ---: |
| 见过的图 / 去重后 | 530 / **530** | 530 / **530** | 530 / **530** |
| 重复访问 | 0 | 0 | 0 |
| 缺失 ID | 0 | 0 | 0 |
| 批次数（B2） | 265 | 265 | 265 |
| 最后一批张数 | 2 | 2 | 2 |
| `dataset.epoch` | 1 | 2 | 3 |
| 被翻转的图 | 248（46.79 %） | 265（50.00 %） | 265（50.00 %） |
| 翻转指纹与上一轮不同 | —（首轮） | 是 | 是 |

530 = 482 正常 + 48 异常，132 = 120 正常 + 12 异常，与 v2 清单一致（train `f1522eb9…`、val `60a6c6cb…`、test `4bdc5941…` 只做哈希）。

### 7.2 计数与学习率

| 项目 | 实测 |
| --- | --- |
| 计划更新（3 × 265） | 795 |
| 尝试 / 成功 / 跳步 | **795 / 795 / 0** |
| `scheduler.completed` | **795**（与成功更新数相等） |
| warmup | 40 步（`round(0.05 × 795)`） |
| LR 轨迹 | 1.5e-6 →（第 40 步）6e-5 →（第 795 步）**1e-6** |
| 每轮首/末 LR | 1.500e-6 → 4.799e-5 / 4.789e-5 → 1.719e-5 / 1.708e-5 → **1.000e-6** |
| 优化器 | AdamW，wd 0.01，betas (0.9, 0.999)；72 个衰减张量 + 133 个不衰减 |
| 参数 | 3 714 658（编码器 3 319 392 + 头 395 266），全量微调 |

`lr_last` 第 3 轮正好是 1e-6，说明 off-by-one 的定义和实现一致：最后一次更新落在 `progress = 1`。

### 7.3 训练损失（加权口径）

| 轮 | CE | Dice 项 | 总损失 | 有效像素 | 正 GT 图 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.118857 | 0.984199 | **0.610957** | 105 830 400 | 48 |
| 2 | 0.017636 | 0.962749 | **0.499010** | 105 830 400 | 48 |
| 3 | 0.012300 | 0.911114 | **0.467857** | 105 830 400 | 48 |

CE 掉了一个数量级，Dice 项只从 0.984 降到 0.911。这个组合要会读：CE 是**逐像素均值**，在前景少的情况下易受背景主导；Dice 只对含前景的图取平均。两项回答不同问题，不能只看CE下降。训练Dice下降较慢与验证仍有漏检的现象相容，但不能据训练损失直接判定验证失败的原因。

### 7.4 验证集全量指标（132 张，B1 IEEE FP32，原图尺度，阈值 0.5）

| 轮 | 全局 IoU | 全局 Dice | Precision | Recall | TP | FP | FN | 异常宏平均 IoU | 正常图 FP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.010309 | 0.020408 | 1.0000 | 0.0103 | 3 518 | 0 | 337 731 | 0.003082 | **0 px** |
| 2 | 0.237012 | 0.383200 | 1.0000 | 0.2370 | 80 880 | 0 | 260 369 | 0.065552 | **0 px** |
| 3 | **0.608100** | 0.756296 | 0.8610 | 0.6743 | 230 093 | 37 131 | 111 156 | **0.220685** | **0 px** |

- 累计像素 198 300 960（异常图 18 027 360），TN 从 197 959 711（第 1、2 轮）变成 197 922 580（第 3 轮），差值 37 131 = 全部 FP。
- 异常宏平均 Dice 同步 0.005945 → 0.098516 → 0.278437。
- **全局 NA 原因为空，不代表逐图没有 NA**：三轮全局指标均有定义；120 张正常图的 GT 和预测均空，逐图 IoU/Dice/precision/recall 为 null 并保留原因。异常图 precision 的 NA 数分别为 11/9/7；异常 GT 有缺陷时空预测的 IoU/Dice/recall 是0。
- 正常图误报比三轮都是 **0.0**，`images_with_any_false_positive = 0`。这不是"模型很准"，而是"模型还不敢报"——见 7.5。

### 7.5 逐图：0.608 是被三张大缺陷撑起来的

第 3 轮 12 张异常图的逐图结果（按 IoU 升序）：

| 图 | GT 前景 | 预测前景 | IoU |
| --- | ---: | ---: | ---: |
| 019 | 4 050 | **0** | 0.0000 |
| 033 | 2 393 | **0** | 0.0000 |
| 042 | 15 523 | **0** | 0.0000 |
| 051 | 4 335 | **0** | 0.0000 |
| 081 | 6 373 | **0** | 0.0000 |
| 087 | 2 328 | **0** | 0.0000 |
| 090 | 3 860 | **0** | 0.0000 |
| 010 | 5 153 | 9 046 | 0.3012 |
| 016 | 3 542 | 9 440 | 0.3465 |
| 059 | 79 992 | 35 286 | 0.4411 |
| 046 | 95 116 | 70 529 | 0.7415 |
| 045 | 118 584 | 142 923 | 0.8178 |

**12 张里 7 张一个缺陷像素都没报**（预测前景 0），第 1、2 轮分别是 11 张和 9 张。全局 IoU 0.608 主要来自 045/046/059 这三张（合计 293 692 px，占 val GT 前景的 86 %）。所以：

- 全局 IoU 上升是真的（能找大缺陷），但**不能**读成"小缺陷也能找"；
- 宏平均 0.2207 才是"每张图有一票"的读数；
- 正常图 0 误报 + 小缺陷全漏 = 模型此刻仍在"少报"一侧，当前分割效果不足；三轮不能判断最终收敛水平或将原因单独归于轮数。压缩的学习率预算可能有影响，但还需正式实验验证，不是需要立刻改损失或阈值的证据。

### 7.6 资源与耗时（实测，不是推算）

| 项目 | 第 1 轮 | 第 2 轮 | 第 3 轮 |
| --- | ---: | ---: | ---: |
| 训练墙钟 | 61.06 s | 45.20 s | 45.63 s |
| 其中单步计时合计 | 37.47 s | 28.93 s | 29.24 s |
| 其中读图/主机等待 | 23.59 s | 16.28 s | 16.39 s |
| 132 张 val 全量 | 14.30 s | 9.51 s | 9.71 s |
| 6 张预览 + 出图 | — | — | —（三轮合计 1.643 s） |
| 保存 | 1.92 s | 1.85 s | 1.66 s |
| 单步中位 | 124.30 ms | 105.39 ms | 105.36 ms |
| 峰值 allocated / reserved | — | — | **806.3 / 934.0 MiB** |

- 首轮慢是因为 CUDA 上下文与 cuDNN 选算法；**首轮的 61 s 不能用来估算长跑**。
- 初始化（含快照校验、两个 Dataset、processor、模型加载）3.593 s；脚本总计 **208.17 s**。
- 单步中位 105 ms 比阶段 08 的 58.5 ms 高，当前共享step包括有限性检查和真实执行确认，但没有控制变量计时，不能把全部速度差归因于这些操作——**测量对象不同，不能拿 58.5 ms 估正式训练**。
- 读图等待占训练墙钟约 36 %（16.3/45.2）。`workers=0` 时解码与计算串行，这是阶段 11 提高吞吐前最明确的一块。
- 主机 RAM：总 15.22 GiB，运行中可用 1.73 GiB，负载 88 %。全量 val 是**流式**评价（逐张前向、立即累计），没有把 132 张原尺寸概率留在内存；只有 6 张预览保留了概率（npz 30 MB 落盘）。

### 7.7 保存与复验

| 项目 | 实测 |
| --- | --- |
| best | epoch 3，`global_step` 795，`scheduler.completed` 795 |
| best.pt SHA-256 | `d1698b285430e07f125982300cca9d139df5d07324af9acfbbc5c41d1ea9b81c` |
| 严格加载 | `strict=True`，missing 0 / unexpected 0 |
| `model_state.pt` 与 `best.pt` 权重 | 逐张量相等（`mirror_matches_best = true`） |
| 新进程全量 val | 混淆量与全局指标**逐位相同**（IoU 0.6081003224） |
| 6 张预览概率 | 最大差 **0.0**，XOR **0 / 9 013 680 px** |
| 复验耗时 | 12.93 s（含模型构建、132 张 val、6 张预览） |

复验用的是 run 内 processor（不是重新从快照建），所以"训练产物自带预处理配置"这条也一起被验证了。

### 7.8 运行期代码冻结

| 项目 | 值 |
| --- | --- |
| git HEAD（运行前后一致） | `0c5f0bd1d6d1…`（`第九阶段验收`） |
| 运行开始时工作区 | `M segformer_lab/training.py` + 4 个未跟踪新文件 |
| `git diff` SHA（起/末） | `36226bb84e6c…` / 相同 |
| 13 个源文件 SHA | 起末一致 |
| `source_and_config_unchanged` | **true** |

也就是说：这次 run 的代码**不是** HEAD 的代码，而是 HEAD + 上述工作区改动（`91c829d2…` / `52e3aae7…`）。这一点必须写进结论，不能声称"用提交 `0c5f0bd` 训练"。

### 7.9 验收清单

| 验收条件 | 结果 | 证据 |
| --- | --- | --- |
| 全量 530 train / 132 val，非残留子集 | 通过（`train_ids_file: null`，每轮 530 张全见） | `summary.json`、`epochs.jsonl` |
| 训练增强恢复、每轮 `set_epoch` | 通过（翻转 248/265/265，指纹逐轮不同） | 7.1 |
| val 无增强 | 通过（`flip="none"`，metrics 拒绝 flipped） | Dataset无增强契约与真实前向门禁；原逐图分数字段未保存 flipped |
| 按 epoch 完整遍历、`drop_last=false` | 通过（265 批，末批 2 张） | 7.1 |
| attempt/success/skip 分开记录 | 通过（795/795/0） | `updates.jsonl` |
| scheduler 只随真更新前进 | 通过（`completed` 795 = 成功数；单测覆盖跳过分支） | 7.2、`tests/test_pilot.py` |
| warmup/cosine 实际步数明确 | 通过（40 + 755，首步 1.5e-6，末步 1e-6） | 7.2 |
| 每轮 132 张全量原图评价 | 通过（3 轮，全局指标有定义；逐图仍保留 NA） | 7.4 |
| 只按 val 全局 IoU 选 best | 通过（epoch 1/2/3 均刷新，best=3） | `summary.json.best` |
| 加权聚合而非 batch 均值 | 通过（公式 + 单测） | 5.4、`tests/test_pilot.py` |
| 只存 6 张预览、不全量缓存 | 通过（npz 只有 6 张，30 MB） | run 目录 |
| last + best 完整保存 | 通过（含 optimizer/scaler/scheduler/epoch/RNG） | 7.7 |
| 新进程严格加载并复算 | 通过（指标逐位相同、概率差 0） | `reload_check.json` |
| 代码/配置/清单身份固化 | 通过（起末一致） | 7.8 |
| resume | **未实现、未验证**，如实标注 | `summary.json.resume` |
| 不读 test、不做测试推理 | 通过（test 只算哈希，`data.test_images_read: false`） | `summary.json` |

## 8. 遇到的问题、修复与未解决项

### 8.1 本轮没有崩溃，但有三处必须记录的发现

| # | 发现 | 证据 | 影响与后续 |
| --- | --- | --- | --- |
| 1 | 单步中位从 58.5 ms 升到 105 ms | 7.6 | 两次测量的代码和运行状态不同，不能单凭这两个数字把全部差异归因于检查开销。正式训练继续保留已验收检查，不为提速默认删减；是否优化须有独立测量 |
| 2 | 读图等待占训练墙钟 36 % | 7.6（16.3/45.2 s） | `workers=0` 是唯一验证过的设置。可继续使用已验证的 workers=0 完成首个基线；workers=2/4 的正确性与收益可另作可选实验，不作为开跑前的必做门槛 |
| 3 | 7/12 异常图完全漏检、正常图 0 误报 | 7.5 | 模型仍在"少报"一侧。三轮结果不足以区分欠训练、细目标分辨率限制、数据不平衡等原因，不能仅据此确诊欠训练；**不能**据此改损失/阈值，也不该把全局 IoU 0.608 当成绩 |

### 8.2 工程上的两处修正

1. **把 `_step_optimizer` 转成公开的 `step_optimizer`，并保留旧别名**。阶段 09 的验收测试直接引用过内部名；改名 + 别名既让新入口不必调用私有函数，又不破坏既有回归。
2. **`train_step` 抽出后重跑全套回归**：219 项通过，其中 12 项是本轮新增（调度算术、跳步不推进、加权聚合、配置守卫）。抽共享函数是有回归风险的改动，靠这套测试兜住。

### 8.3 未解决/未验证

- **resume 未实现**：checkpoint 里虽然存了 optimizer/scaler/scheduler/epoch/RNG，但没有在 epoch 边界真正重跑验证过，所以不声称支持。阶段 11 若要 resume，必须实测"新进程载入 → 参数一致 → epoch/LR 正确 → 同一验证样本概率在容差内一致"。
- **`workers>0` 仍未验证**（从阶段 02 挂到现在）。
- **3 轮不足以判断收敛**：LR 已降到底，指标仍在上升；这说明"多跑几轮可能更好"，但本阶段不做这个结论，交给阶段 11。
- **best 选择只有 3 个候选 epoch**；正式训练的候选点更多，但验证集仍是同一 132 张图、12 张异常图，样本量不会因此增加。
- **调度被压缩**：3 轮的 cosine 只覆盖 755 步，与 50 轮总预算 13 250 步、其中 warmup 662/余下 cosine 12 588 步的轨迹不同。不要把 pilot 的 LR 曲线当成正式训练的前 3 轮。
- 日志编码：`Tee-Object` 写的是 UTF-16，`grep` 会当二进制。`reports/stage10-unit-tests.txt` 已转成 UTF-8；run 内 `updates.jsonl`/`epochs.jsonl` 本来就是 UTF-8。

## 9. 给用户的学习要点

1. **预算单位决定控制流**。"跑到 20 次成功更新"和"跑完 3 个 epoch"是两种不同的合同：前者按步数停，后者按数据覆盖停。用前者的循环去凑后者的账，会出现"某些图多看了几遍、某些少看一遍"，而任何按 epoch 的统计都会失真。
2. **调度器是"给权重用的"，不是"给时间用的"**。AMP 跳步时权重没动，如果 LR 照常前进，模型就在没更新的情况下被推到更小的学习率上——`advance_schedule()` 把这个条件写成了一个函数，单测里用 `{"skipped": True}` 就能验证。
3. **全局 IoU 与宏平均必须一起看**。本阶段全局 0.608、宏平均 0.221，差 2.75 倍；原因是一张 118 584 px 的缺陷抵得上 30 张 3 000 px 的小缺陷。本次只报全局会掩盖小目标失败；全局和宏平均在一般情况下没有固定大小关系。
4. **"没误报"可能是"没敢报"**。120 张正常图 0 误报看着漂亮，但同期 7 张异常图 0 预测——这两个数放在一起才是完整信息。
5. **保存了 optimizer 不等于支持 resume**。要不要写 `resume.implemented`，取决于有没有真的在新进程里从 epoch 边界恢复并核对指标，而不是取决于 checkpoint 里有没有那个字典。

自测问题：

1. 如果把 `advance_schedule` 改成无条件 `schedule.step()`，本轮 795 步里 LR 轨迹会怎么变？在什么情况下才会真正出问题？
   *线索*：7.2（`scheduler.completed` = 795）与 `tests/test_pilot.py::ScheduleAdvanceTests`；答案是"本轮 0 跳步，所以轨迹不变；一旦有跳步，LR 会领先于权重"。
2. 为什么第 3 轮的 TN 比前两轮少 37 131，而正常图误报仍是 0？
   *线索*：7.4 的 TN/FP 数字与 7.5 的逐图表；37 131 全部落在异常图上，是过报而不是正常图误报。
3. 如果要在阶段 11 把训练时间减半，最先该动哪一块？为什么不能直接改 `train.py` 里的 batch size？
   *线索*：7.6 的耗时分解（读图 36 %、单步 105 ms）；改 batch 不会自动修改配置中的学习率，但会改变每次更新的样本组成、每轮步数、调度预算及内存使用；需要明确配置和独立验证。

## 10. 下一阶段与修订记录

阶段 11（正式训练）依赖本阶段确认的这些东西，且必须保持一致：

- 复用 `pilot.py` 的 epoch 控制流与 `training.train_step/WarmupCosineLR`，只改预算：`epochs: 50`、`total_steps = 50 × 265 = 13 250`（warmup `round(0.05 × 13 250) = 662` 步，沿用 Python ties-to-even 规则；该取整必须写进配置解析结果）。**重新建 schedule，不沿用本阶段的 795 步计划。**
- `configs/baseline.yaml` 目前写的是 `epochs: 50`、`warmup_ratio: 0.05`、`min_learning_rate: 1e-6`，但还缺 `train.flip`、`train_ids_file` 之外的 pilot 守卫字段；正式训练入口要么复用 `pilot.py` 并放宽 `epochs`，要么另立 `train_full.py`，这一点需在阶段 11 开头定下来，不能一边跑一边改。
- 显存预算可参考 **934.0 MiB reserved**（B2/512/AMP），但本阶段已包含 best/last 与逐轮验证；正式训练若增加 TensorBoard 等功能需重新记录资源，轮数本身不意味着单步显存线性增长。
- 时间预算：按第 2、3 轮的 45.2/45.6 s 每轮，50 轮约 38 分钟训练 + 每轮 9.5 s 验证 ≈ 8 分钟，合计约 46 分钟——**这是本阶段的实测外推，不是承诺**；一旦开 `workers>0` 或加 TensorBoard 都要重测。
- 本阶段**未**启动正式训练，符合交接文档"三轮结束后停下"的要求。

| 日期 | 修改原因 | 改动与影响 | 修正后重新验证的证据 |
| --- | --- | --- | --- |
| 2026-11-01 | 初版 | 新增 `pilot.py` / `pilot.yaml` / `test_pilot.py`；`training.py` 抽出 `train_step` 并公开 `step_optimizer`；三轮 795/795/0、val IoU 0.0103→0.6081、宏平均 0.0031→0.2207；新进程复验概率差 0、指标逐位相同 | `runs\pilot_001\summary.json`、`reload_check.json`、`reports\stage10-summary.json`、`reports/stage10_evidence/original-unit-tests.txt`（原219项） |

## 11. 独立验收补充

最终完整回归231项通过（10.516s）；新进程CUDA复验13.7s，原30项inventory和完整best/last状态通过，两份205个优化器参数的step均795，132张原图的所有逐图/宏平均/正常统计/NA/global完全一致，六张预览9,013,680像素最大概率差0.0、XOR0。完整报告见[联合验证结果](../../reports/2026-11-01-stage10-acceptance.md)、[机器复验](../../reports/stage10-reload-check.json)与[审计](../../reports/stage10-evidence-audit.json)。原三轮219项测试及日志/源码另行归档，不用新代码冒充旧训练版本。

新增门槛覆盖inventory篡改、逐图统计变化、NPZ key/shape/范围/NaN/Inf、阈值XOR、完整优化器分组/状态、LR/计数/RNG结构与起止CSV/meta身份。原复验只比较global且未以XOR为门槛，存在假通过空间；本次收紧后真实保存物仍全部通过。

另实测Transformers配置构造器会原地添加attn_implementation字段；改用deepcopy构造后，保存元数据仍可完整核对。针对该副作用已先复现红测试，再验证修复。它不意味着原checkpoint损坏。

原run没有单独保存loaded.snapshot/source_sha/encoder_check的开始记录，当前文件核验不能反向补成当时的独立快照证据；阶段11须补。增强计划的计数和指纹可重算，实际像素同步沿用阶段06已验收的代码，未逐张目击所有训练增强图。当前resume仅保存结构，尚未验证，不能称支持断点续训。

下一步以[阶段11执行交接](../../reports/2026-11-01-stage11-execution-handoff.md)为准：保持固定配方和workers0，先补正式入口/恢复证据，再从原预训练新跑50轮，不自动测试。

| 日期 | 本轮修订 | 重新验证 |
| --- | --- | --- |
| 2026-11-01 | 完整保存物与概率门槛、配置构造器副作用、NA/来源/吞吐说明；原件不改 | 231项回归；CUDA原图复验和30项产物核验通过 |
