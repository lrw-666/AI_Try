# 阶段 03：引入 PCB1、固定划分与标签规则——设计与实现学习文档

> 历史版本说明（2026-10-31 第四阶段验收）：本文记录阶段 03 当时的 v1 文件级划分及其验证，不重写历史结果。当前基线已因 Normal/0042/0043 的近重复风险升级为 v2 分组协议；请使用[第四阶段验收报告](../../reports/2026-10-31-stage04-acceptance.md)和当前 README 中带 `--reviewed-groups` 的命令。本文 v1 清单及元数据已原样归档至 `manifests/archive/pcb1_v1`，不能用旧命令覆盖当前划分。

| 字段 | 填写内容 |
| --- | --- |
| 阶段编号与名称 | 步骤 3：VisA PCB1 数据准备、固定验证集划分、标签约定 |
| 状态 | 原始导入、固定划分与文件头检查已有验证记录；本轮安全与协议验收进行中，修复后的最终复核待完成 |
| 创建/更新时间 | 2026-10-31 17:00 +08:00 |
| 对应代码身份 | 仓库 `E:\桌面\AI\AI_Try` 分支 `master`，HEAD `dfe8fa1`；本阶段新增 `segformer_lab\prepare_data.py`（SHA-256 `8d856ccc...3171`，702 行）、`tests\test_prepare_data.py`（`3c3f963a...5c7f`，129 行）；`scripts\enter_project.ps1` 有修复（`19d5ef59...b92d8`），见 8.1 |
| 项目根目录 / 资源根目录 | `E:\桌面\AI\AI_Try\work\SegFormer` / `E:\桌面\AI\segformer_resources` |
| Python / 环境prefix | `...\segformer_resources\env\python.exe`，Python 3.12.16（沿用阶段 02，未重装任何包） |
| 依赖的上一阶段 | 01（入口脚本、`paths.py`）、02（可用环境）；两者验收记录见 `reports\2026-10-31-stage01-02-acceptance.md` |

**本轮验收补充：** [第三阶段验收与完善记录](../../reports/2026-10-31-stage03-acceptance.md)。修正覆盖权限、临时文件冲突、固定协议和失败发布保护后重新运行，数据与三份CSV身份不变；本轮测试数为23，原始命令输出作为首次运行历史记录保留。

## 1. 本阶段解决什么问题

模型训练前必须先把「哪些图用来学、哪些图用来挑模型、哪些图只用来交卷」固定下来。这一步做错，后面所有指标都没有意义：验证集里混进训练图，验证分数会虚高；测试集被反复用来调参，测试分数就不再是测试分数。

输入：

- `docs\dataset\VisA\VisA_20220922.tar`（1 929 840 640 B）——12 类工业图像数据集，PCB1 在 `pcb1/Data/...`；
- `docs\dataset\VisA\spot-diff-main.zip`——官方代码包，内含 `split_csv/2cls_highshot.csv`（官方 high-shot 划分表）。

输出（全部可在磁盘上核对）：

| 产物 | 位置 | 用途 |
| --- | --- | --- |
| 官方 CSV 副本 | `manifests\official_2cls_highshot.csv` | 与 ZIP 内文件逐字节相同，划分的唯一依据 |
| 训练清单 | `manifests\train.csv`（530 行） | 482 正常 + 48 异常 |
| 验证清单 | `manifests\val.csv`（132 行） | 120 正常 + 12 异常，固定配额 |
| 测试清单 | `manifests\test.csv`（442 行） | 402 正常 + 40 异常，等于官方 test |
| 划分元数据 | `manifests\split_meta.json` | 算法、种子、配额、来源哈希、全部检查结果 |
| 提取记录 | `segformer_resources\data\visa\extraction_manifest.json` | 1205 个文件的路径/大小/SHA-256 |
| 解压后的图像 | `segformer_resources\data\visa\pcb1\Data\{Images,Masks}` | 保留官方相对路径 |

边界：**不做可视化、不转标签、不训练、不下载权重**。本阶段确认路径、数量、集合交集，以及图像和掩码文件头能否解析、尺寸能否读取；完整解码和像素级审计（重复图、缺陷面积、缩放后是否丢小目标）属于阶段 04。

## 2. 技术栈与版本

| 技术/库 | 实际版本 | 本阶段作用 | 状态与证据 |
| --- | --- | --- | --- |
| Python | 3.12.16 | 运行数据准备模块 | 已运行 |
| `tarfile` / `zipfile` / `csv` / `hashlib` / `json` | 标准库（3.12.16） | 读归档、写清单、算哈希 | 已运行 |
| NumPy | 2.5.4 | 唯一的随机源 `default_rng(42)` | 已运行，版本写入 `split_meta.json` |
| Pillow | 12.3.0 | 读取图像/掩码文件头、模式和尺寸；未完整解码 | 已运行（1104 图 + 100 掩码） |
| `unittest` | 标准库 | 路径守卫与划分算法的回归 | 首版16项通过；本轮完善后全套23项通过（含阶段02的5项） |

没有引入 pandas、OpenCV、`datasets`：本阶段只需要顺序读 CSV 与文件头，多一个依赖就多一处版本漂移。图像统一用 Pillow，避免 OpenCV 的 BGR/通道顺序问题在阶段 04 才暴露。

## 3. 设计与取舍

### 3.1 数据流

```text
spot-diff-main.zip ──┐
  split_csv/2cls_highshot.csv
        │ 读原始字节 → SHA-256 比对 c3331...f09
        ├──► manifests/official_2cls_highshot.csv（逐字节副本）
        └──► 解析 10821 行，筛 object == pcb1 → 1104 行
                    │
VisA_20220922.tar ──┤  只取 pcb1/ 成员（1211 个，含 6 个目录）
  1.93 GB           │  路径守卫 → 逐成员写 .part → 校验 → 原子替换
                    └──► data\visa\pcb1\Data\{Images,Masks}（1205 个文件，283 MiB）

1104 行 ──► 官方 train 662 行 ──► 按 label 分组、按 image 排序
                                 ──► default_rng(42)：perm(normal) → perm(anomaly)
                                 ──► 前 120 正常 + 前 12 异常 = val
                                 ──► 其余 530 = train
           官方 test 442 行 ────► 原样 = test
                                 │
                                 ├──► train.csv / val.csv / test.csv（按 image 排序）
                                 ├──► split_meta.json（算法 + 计数 + 哈希 + 检查）
                                 └──► 控制台摘要 + 退出码
```

### 3.2 关键决策

| 决策点 | 选择 | 理由 | 未选方案 |
| --- | --- | --- | --- |
| 提取范围 | 只提取 `pcb1/`，保留 `pcb1/Data/...` 相对路径 | 1.93 GB 归档里有 12 类，本任务只用 PCB1；保留官方路径让清单可直接对照官方 CSV | 解压全部 12 类（多占数 GB、无用途）；解压后重排目录（清单与官方对不上） |
| 归档处理方式 | `tarfile` 以 `r:` 打开，逐成员取流、1 MiB 分块写盘 | 只读需要的成员，不把归档或图像整体读进内存 | 先整体流式解压再筛（多写 1.6 GB 临时数据） |
| 已有文件 | 先比大小再比 SHA-256，相同则跳过并记为 `verified-existing` | 「核验而不是无条件覆盖」；重复执行安全 | 无条件覆盖（可能覆盖掉用户手工修过的文件） |
| 正常图掩码 | `mask` 字段留空字符串，训练时按图像尺寸生成全零数组 | 不额外生成 1004 个全零 PNG（约 1 MB 但会与官方数据混淆），也不把空串拼成路径 | 生成零掩码文件；或用 `None` 让下游猜 |
| 划分随机源 | 单一 `numpy.random.default_rng(42)`，先 normal 后 anomaly | 指南规定的顺序；顺序不同结果就不同，必须写死 | 两次独立 `default_rng(42)`（会得到另一组 val）；`random.seed`（跨版本不稳定） |
| 清单排序 | 输出前按 `image` 排序 | 让文件内容与顺序和随机抽样的内部顺序解耦，重复执行可哈希比对 | 保留抽样顺序（同一划分但行序不同，哈希不稳定） |
| 标签转换 | 本阶段**不转**，只记录约定 | 官方掩码存的是 1..4 的多类 ID；转成 1 是训练标签语义，属 Dataset 阶段，写在这里容易与「存图用 255」的官方约定混淆 | 现在就重写掩码文件（破坏原始数据） |

### 3.3 风险与不确定点

| 风险 | 现状 | 后续验证 |
| --- | --- | --- |
| 同一实物出现在 train 与 test（近重复） | 本阶段未检测 | 阶段 04 做图像哈希/相似度审计，必要时另立分组划分协议 |
| 官方 test 与 train 是否同批次拍摄 | 未知 | 同上；若改变官方测试协议，必须另设实验名，不能冒充 high-shot |
| 掩码取值集合与二值化阈值 | 抽查 2 张：`{0,1}`、`{0,2}` | 阶段 04 统计全部 100 张的取值分布；本阶段只确认「非零即缺陷」的约定 |
| 512 缩放后小缺陷是否消失 | 未测 | 阶段 04 用最近邻缩放复算前景占比 |

## 4. 文件、接口与数据约定

| 文件/模块 | 新增或修改 | 职责 | 关键入口与输入输出 |
| --- | --- | --- | --- |
| `segformer_lab\prepare_data.py` | 新增 | 读官方 CSV、安全提取、固定划分、审计并写清单 | `python -m segformer_lab.prepare_data --archive ... --source-zip ... --data-root ... --category pcb1 --seed 42 --val-normal 120 --val-anomaly 12 --out manifests`；成功返回 0，任一检查失败返回 1 |
| `tests\test_prepare_data.py` | 新增 | 路径守卫（6 项）与划分算法（5 项）回归 | `python -m unittest discover -s tests` |
| `scripts\enter_project.ps1` | 修改 | 修复在「变量不存在」时被安全删除策略拦截的问题 | 见 8.1 |
| `manifests\*.csv` | 生成 | 三份派生清单 + 官方副本 | UTF-8 无 BOM、LF、按 `image` 升序 |
| `manifests\split_meta.json` | 生成 | 划分与来源的完整元数据 | 见 4.2 |
| `data\visa\extraction_manifest.json` | 生成 | 提取取证 | 见 4.3 |

### 4.1 清单列语义

```csv
object,split,label,image,mask
pcb1,train,normal,pcb1/Data/Images/Normal/0000.JPG,
pcb1,val,anomaly,pcb1/Data/Images/Anomaly/010.JPG,pcb1/Data/Masks/Anomaly/010.png
```

| 列 | 取值 | 说明 |
| --- | --- | --- |
| `object` | `pcb1` | 与官方 CSV 同名，便于回溯 |
| `split` | `train` / `val` / `test` | **派生后**的划分；官方原划分由官方 CSV 追溯 |
| `label` | `normal` / `anomaly` | 官方字段，未改写 |
| `image` | `pcb1/Data/Images/...` | 相对 `data\visa` 的路径，正斜杠 |
| `mask` | 异常：`pcb1/Data/Masks/...png`；正常：空串 | 正常图不指向任何文件 |

### 4.2 `split_meta.json` 关键字段

| 字段 | 本机实际值 | 作用 |
| --- | --- | --- |
| `algorithm.seed` / `numpy_version` | `42` / `2.5.4` | 复现划分所需的最小信息 |
| `algorithm.procedure` | 文字描述（分组、排序、两次 permutation） | 换机器重写代码时的唯一口径 |
| `counts` | 530 / 132 / 442 | 与指南表格对照 |
| `sources.archive` | size `1929840640`，sha256 `2eb8690c...f362` | 证明用的是哪份归档 |
| `sources.source_zip` | size `2518966`，sha256 `22155105...7738` | 同上 |
| `sources.official_csv` | sha256 `c3331eed...2f09` | 与指南记录的校验值一致 |
| `extraction.extraction_baseline` | 1205 文件 / 296 989 825 B / 16:52:56 | 真正写入字节的那次运行 |
| `manifests.{train,val,test}.sha256` | 见 6.5 | 清单身份 |
| `checks.*` | 全部 `true` | 见 6.4 |
| `protocol_counts` | expected = actual = 530/132/442 | 固定协议校验 |

`split_meta.json` 每次运行都会重写，`created_at` / `elapsed_seconds` 随之变化，因此**它的哈希不是稳定身份**；稳定身份是三份 CSV 的哈希。

### 4.3 `extraction_manifest.json`

1205 条记录，每条 `{path, size, sha256, state}`，`state` 为 `extracted` 或 `verified-existing`。另有 `members_matched`（1211，含 6 个目录）、`bytes_written`、`extraction_baseline`。这个文件是「磁盘上的图确实来自哪个归档成员」的直接证据。

## 5. 关键实现讲解

### 5.1 归档路径守卫：先判合法性，再判归属

```python
def safe_member_target(root: Path, member_name: str) -> Path:
    name = member_name.replace("\\", "/")
    if name.startswith("/") or _DRIVE.match(name):
        raise ValueError(f"absolute member path: {member_name!r}")
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError(f"parent traversal in member path: {member_name!r}")
    if any(":" in part for part in parts):
        raise ValueError(f"alternate data stream in member path: {member_name!r}")
    target = (root.resolve() / Path(*parts)).resolve()
    target.relative_to(root.resolve())   # 越界即抛 ValueError
    return target
```

为什么要四道检查而不是只用最后一行：

- 显式拒绝绝对路径、盘符和 `..`，先限定可接受的归档成员名；即使某个含 `..` 的路径最终仍落在根目录内，本模块也不接受这种写法。`resolve()` 与 `relative_to()` 随后检查解析后的目标是否仍在根目录内，包括已有符号链接或目录联接指向根外的情况。这些检查只计算路径，在通过前不写文件；
- `:` 检查针对 NTFS 备用数据流（`a.png:evil`），这类名字在资源管理器里看不出来；
- 反斜杠先统一成正斜杠，使成员名校验在不同平台上采用相同的分隔规则。Windows 本来就把反斜杠视为分隔符；部分其他平台会把它当作普通文件名字符，统一后可以识别其中的 `..` 分量。

真实归档里没有任何异常成员，所以这些分支没有在真实数据上触发过——它们由 `tests\test_prepare_data.py` 的 6 个用例覆盖，而不是靠「应该没问题」。

### 5.2 写入方式：临时文件 + 哈希 + 原子替换

```python
# 省略了异常清理细节；完整实现见 extract_category。
with tempfile.NamedTemporaryFile(
    mode="wb", dir=target.parent, prefix=f".{target.name}.",
    suffix=".part", delete=False,
) as out:
    part = Path(out.name)                 # 随机名称、独占创建
    # 逐个CHUNK读取并写入，同时累计size与SHA-256。
# 验证写入长度后再原子替换；finally只清理本次创建的临时文件。
part.replace(target)
```

三点考虑：

1. **不会留下半个文件**：中断时最多留下 `.part`，目标路径要么是旧的完整文件，要么是新的完整文件。
2. **哈希是边写边算的**：不需要为了记录校验值再读一遍磁盘。
3. **内存占用恒定**：无论 100 KB 还是 10 MB 的图，只占一个 1 MiB 缓冲。

同一段逻辑在「目标已存在」时改走另一条路径：先算已有文件的 SHA-256，再从归档流里算成员 SHA-256，两者一致才跳过；不一致默认报错并列出两个哈希；覆盖不同内容仅由显式 `--force-overwrite` 授权。`--rewrite-extraction` 不授予覆盖不同内容的权限。

### 5.3 划分：顺序即协议

```python
rng = np.random.default_rng(seed)
perm_normal = rng.permutation(len(train_normal))
perm_anomaly = rng.permutation(len(train_anomaly))
val_normal_rows = [train_normal[i] for i in perm_normal[:val_normal]]
val_anomaly_rows = [train_anomaly[i] for i in perm_anomaly[:val_anomaly]]
```

`default_rng(42)` 是同一个生成器对象，`permutation(normal)` 先消耗随机数，`permutation(anomaly)` 用剩下的状态。**把两行顺序调换，或换成两个独立生成器，得到的验证集就完全不同**，而且都「看起来合理」。所以：

- 算法写进 `split_meta.json` 的文字里；
- `test_matches_reference_permutation` 用一份独立实现复算，防止以后重构时悄悄改了顺序；
- 正式训练只读 CSV，不再抽样。

另一个细节：`train_normal` / `train_anomaly` 在抽样前按 `image` 排序。若不排序，顺序取决于 CSV 原始行序，而官方 CSV 的行序并不保证稳定。

### 5.4 审计在写清单之前完成

`audit()` 在发布 `train/val/test.csv` 之前运行；本轮修正后，审计失败只写 `preparation-failure.json` 并返回1，已有三份CSV和成功元数据保持不变。通过后才发布CSV与 `split_meta.json`。命令行在读写之前固定pcb1、seed42与120/12配额；协议同时核对总数和各标签数量。同名CSV若内容不同，会拒绝覆盖，避免悄悄改变已锁定的清单。

`checks.ok` 的组成：

```python
report["ok"] = all([
    report["counts_match_expected"],
    report["pairwise_disjoint"],
    report["test_equals_official_test_set"],
    report["duplicate_image_paths"] == 0,
    report["images_missing_count"] == 0,
    report["images_unreadable_count"] == 0,
    report["mask_missing_count"] == 0,
    report["mask_unreadable_count"] == 0,
    report["mask_size_mismatch_count"] == 0,
    report["normal_rows_zero_mask_constructible"],
])
```

## 6. 如何操作与实际执行了什么

全部在项目上下文中执行；每个新进程先 `$env:PSExecutionPolicyPreference='Bypass'`，再 dot-source 入口（或直接用 `run_project.ps1`）。**本阶段没有安装、升级或卸载任何包**。

### 6.1 主命令

```powershell
. '.\scripts\enter_project.ps1'
Invoke-SegPython -m segformer_lab.prepare_data `
  --archive '..\..\docs\dataset\VisA\VisA_20220922.tar' `
  --source-zip '..\..\docs\dataset\VisA\spot-diff-main.zip' `
  --data-root 'E:\桌面\AI\segformer_resources\data\visa' `
  --category pcb1 --seed 42 --val-normal 120 --val-anomaly 12 `
  --out 'manifests'
```

### 6.2 运行序列（同一台机器，2026-10-31）

| 次序 | 命令要点 | 实际结果 | 证据 |
| --- | --- | --- | --- |
| 1 | `--skip-extraction` 干跑 | 计数 530/132/442 正确；文件检查因未解压全部缺失（1104/100），`checks ok=False`，退出码 1 | `audit\stage03-dryrun.log` |
| 2 | 正式运行 | 提取 1211 成员 → 1205 文件 / 6 目录 / 296 989 825 B，提取 4.69 s，脚本内部 12.83 s；整条命令 15.3 s（含 1.93 GB 归档哈希） | `audit\stage03-prepare-data.log` |
| 3 | 重复运行 | `files=0 verified=1205`，三份 CSV 哈希与第 2 次完全相同；整条命令 7.6 s | `audit\stage03-rerun.log` |
| 4 | `--rewrite-extraction` 重建基线 | 重写 1205 文件（内容与归档一致），提取 16.96 s，建立 `extraction_baseline` | `audit\stage03-rewrite-baseline.log` |
| 5 | 最终运行 | `files=0 verified=1205`，提取 3.71 s，脚本内部 5.53 s，基线记录保留 | `audit\stage03-final-run.log` |
| 6 | 独立校验 + 单元测试 | 校验 `VERIFY: ok`；`Ran 16 tests ... OK` | `audit\stage03-verification.txt` |

另有 `audit\stage03-rebuild-baseline.log`：一次用 `--force-overwrite` 试图重建基线的尝试，因为「内容相同即跳过」而 `files=0`，随后才新增 `--rewrite-extraction`（见 8.3）。保留它是因为它记录了「为什么需要第二个开关」。

第 4 步的由来见 8.3：第 3 步把「首次写入 1205 个文件」的记录覆盖成了「校验 1205 个文件」，为避免元数据比事实更弱，重建了一次基线并修正了合并逻辑。

### 6.3 提取结果

```text
members_matched        : 1211   （1205 个文件 + 6 个目录）
files_written          : 0      （最终运行时为校验模式）
files_verified_existing: 1205
bytes (baseline)       : 296 989 825
data\visa 占用         : 287 MB（Images 286 MB + Masks 1 MB）
目录                   : data\visa\{extraction_manifest.json, pcb1\Data\{Images,Masks}}
```

只提取了 `pcb1`，其余 11 类没有落盘（`data root entries` 检查确认顶层只有 `extraction_manifest.json` 与 `pcb1`）。

### 6.4 审计结果（`split_meta.json` 的 `checks`）

| 检查 | 结果 |
| --- | --- |
| 计数等于官方派生值 | 通过（482/48/530、120/12/132、402/40/442） |
| 三集合两两交集 | `train&val=0`、`train&test=0`、`val&test=0` |
| test 与官方 test 完全一致 | 通过（缺失 0、多余 0） |
| 重复 image 路径 | 0 |
| 重复 mask 路径 | 0 |
| 图像存在 / 文件头可解析 | 1104 / 1104，缺失 0，文件头解析失败 0；未完整解码 |
| 图像尺寸分布 | `1404x1070`：1104 张（全部一致） |
| 异常掩码存在 / 文件头可解析 | 100 / 100，缺失 0，文件头解析失败 0；未全量完整解码 |
| 掩码与图像尺寸一致 | 100 / 100，不一致 0 |
| 掩码模式 | `L`（单通道）100 张 |
| 正常图 mask 字段为空 | 1004 行全部为空 → 可构造全零掩码 |

### 6.5 产物身份

```text
official_2cls_highshot.csv  c3331eede15f2da8a75b380d4fbfb24f75ee036ec9a25147eca898da6e972f09  (658 163 B)
train.csv                   b26077266a59efdf45306bd9de0ce726aca52db4d7d50ea3ae284eaa92130d53  (530 行)
val.csv                     7335d8c82b8997bffa06af59283f1ade7ac18949bfdec14973cd054f3161bb9c  (132 行)
test.csv                    4bdc594148f208c7067a66a8cb3b06fde53ebc1f4ab6fda84f08f80383061284  (442 行)
split_meta.json             b6ab5a420e04636d44d527132f7c94cf909c42569fac41cd7343f39416b2fcf3  （含时间戳，非稳定身份）
```

来源归档：

```text
VisA_20220922.tar    1 929 840 640 B   sha256 2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362
spot-diff-main.zip       2 518 966 B   sha256 22155105f72985b3bfa1295c24f0ed565977d3a77cf83a15e42f12d84ebc7738
```

### 6.6 独立复核（不使用 `prepare_data` 的代码）

`segformer_resources\audit\verify_stage03.py` 重新解析 ZIP、重新实现划分、重新算哈希，输出：

```text
official pcb1 rows: 1104 (train 662, test 442)
train.csv rows=530 normal=482 anomaly=48
val.csv rows=132 normal=120 anomaly=12
test.csv rows=442 normal=402 anomaly=40
totals: {'train': 530, 'val': 132, 'test': 442} (expected {'train': 530, 'val': 132, 'test': 442})
train+val == official train: True
test == official test      : True
val selection reproducible : True
missing files              : 0
data root entries          : ['extraction_manifest.json', 'pcb1']
files under pcb1/          : 1205
archive/zip/manifest hashes: checked against split_meta
spot image pcb1/Data/Images/Anomaly/010.JPG: mode=RGB size=(1404, 1070)
spot mask  pcb1/Data/Masks/Anomaly/010.png: mode=L size=(1404, 1070) values=[0, 1]
spot mask  pcb1/Data/Masks/Anomaly/016.png: mode=L size=(1404, 1070) values=[0, 2]
VERIFY: ok (independent checks passed)
```

抽查掩码出现 `{0,1}` 与 `{0,2}` 两种取值，直接印证了「官方掩码是多类 ID，不能直接当二类标签用」。

### 6.7 单元测试

```powershell
Invoke-SegPython -m unittest discover -s tests -v
# Ran 16 tests in 0.122s / OK    （5 项环境检查器 + 11 项本阶段）
```

本阶段 11 项覆盖：绝对路径、盘符、`..`、反斜杠 `..`、NTFS 数据流、正常成员；配额与互斥、确定性、与参考实现逐项一致、清单排序与正常行空掩码、配额超限报错。

## 7. 验收与当前状态

| 验收项（主指南 5.3） | 结果 | 证据 |
| --- | --- | --- |
| 三份计数与表格一致 | 满足：530 / 132 / 442 | `split_meta.json` `counts`、`protocol_counts.match=true` |
| 集合两两无交集 | 满足 | `checks.pairwise_image_overlap` 全 0 |
| 官方测试路径集合完全相等 | 满足 | `checks.test_equals_official_test_set=true` + 独立复核 `test == official test: True` |
| 异常图均有掩码、文件头可解析 | 满足：100/100 文件头可解析、尺寸一致；完整解码留阶段 04 | `checks.mask_*` |
| 正常图可构造零掩码 | 满足：1004 行 mask 为空 | `checks.normal_rows_zero_mask_constructible=true` |
| 官方 CSV 校验值与指南一致 | 满足：`c3331eed...2f09` | `sources.official_csv.sha256` |
| 重复执行不重划分 | 满足：三次运行 CSV 哈希完全相同 | 6.2 第 3、5 行 |
| 原 TAR/ZIP 保留在原处 | 满足：只读打开，大小与哈希与 `split_meta` 记录一致 | 独立复核 `archive/zip/manifest hashes` |
| 归档成员路径不越界 | 满足：1211 个成员全部通过守卫，无拒绝项 | `extraction_manifest.json`；守卫另有 6 项单元测试 |
| 已存在文件不被覆盖 | 满足：`force-overwrite` 仅在内容不同时生效，内容相同一律跳过 | 6.2 第 3/5 行 `verified=1205 files=0` |
| 逐条处理、不预加载 | 满足：`r:` 随机访问 + 1 MiB 分块；图像只读文件头 | 5.2；进程峰值内存未单独测量，见 8.6 |
| 阶段文档与索引 | 满足 | 本文件 + `docs/stages/README.md` |

**明确未做的**（防止误读）：

- 未做全量图像/掩码完整解码或像素级审计（重复图、缺陷面积、缩放后前景保留率）——阶段 04；第 6.6 节仅对少量掩码读取像素取值；
- 未把掩码 1..4 转成二类 1，也未生成任何二值掩码文件——转换发生在 Dataset 的 `__getitem__` 里，本阶段只固定约定；
- 未做可视化叠加图；
- 未检测 train/test 之间的近重复实物；
- 未测量进程峰值内存（只有「不整体读入」的代码结构证据，没有 RSS 数字）。

## 8. 遇到的问题、修复与未解决项

### 8.1 入口脚本在本机被安全删除策略拦截（阶段 01 代码的缺陷）

现象：在新的工具进程中执行 `run_project.ps1` 直接失败：

```text
[safe-delete][SAFE_DELETE_FAIL_CLOSED] {"target":"Env:PYTHONHOME","reason":"path-not-found"}
```

原因：`enter_project.ps1` 用 `Remove-Item -LiteralPath 'Env:PYTHONHOME' -ErrorAction SilentlyContinue` 清理继承变量。本机策略对「删除不存在的路径」是 fail-closed，`-ErrorAction` 不生效，于是脚本在第一步就中断。阶段 01 验收时那台进程里 `PYTHONHOME` 恰好存在，所以没暴露。

修复：删除前先判断变量是否存在。

```powershell
if ($null -ne [System.Environment]::GetEnvironmentVariable($name, 'Process')) {
    Remove-Item -LiteralPath ('Env:' + $name) -ErrorAction SilentlyContinue
}
```

`PIP_*` 的循环同样加了判断。修复后 `run_project.ps1`、`Invoke-SegPython` 恢复正常；`scripts\enter_project.ps1` 哈希随之变为 `19d5ef59...b92d8`。

### 8.2 通过 `run_project.ps1` 运行时看不到标准输出

现象：`& run_project.ps1 ... 2>&1 | Where-Object {...}` 只打印 `RESULT: FAILED` 的 `NativeCommandError`，Python 的几十行摘要全部不显示。

原因：包装器设置 `$ErrorActionPreference='Stop'`，`conda run` 把子进程 stderr 转发到自己的 stderr，PowerShell 把它包装成终止性错误，管道里只剩最后一条错误记录。

处置：本阶段改为「在同一进程内 dot-source 入口 + `& $SegConda run ... *> $log`」，把全部流重定向到日志文件后再读。既能拿到真实输出，也留下了 `audit\*.log` 证据。**没有**修改包装器本身——它的退出码语义是正确的，需要修的是调用侧的捕获方式。

补充：`*>` 在 PowerShell 5.1 写出的是 UTF-16，而且当时控制台的 GBK 解码把中文路径写成了 `Lhb—` 之类的乱码。最终改成 `*>&1 | ForEach-Object { $_.ToString() }` 收集，再用 `[IO.File]::WriteAllLines(..., UTF8Encoding($false))` 落盘，日志变为可读 UTF-8（`stage03-final-run.log` 即此法）。更早的几份日志已转码为 UTF-8，英文内容完整；其中中文路径段仍是当时解码错误的残留，**路径以 `split_meta.json` 为准**。

### 8.3 「首次写入」记录被后续校验运行覆盖

现象：第二次（纯校验）运行后，`extraction_manifest.json` 的 `files_written` 从 1205 变成 0，`split_meta.json` 里的提取块同步变成「写了 0 字节」，与事实不符。

原因：提取清单每次运行都整体重写，而第二次运行确实「没有写文件」。

处置：在 `prepare_data.py` 中加入 `extraction_baseline` 字段，优先保留历史中 `files_written > 0` 的那次记录；随后用新增的 `--rewrite-extraction`（唯一会重写「内容相同」文件的开关）重建了一次真实写入基线：

```text
baseline write: files=1205 bytes=296989825 at 2026-10-31 16:52:56 +0800
```

注意与 `--force-overwrite` 的区别：后者只覆盖**内容不同**的文件，`--rewrite-extraction` 才重写内容相同的文件，用于有意重写字节相同副本；内容不同的修复仍需显式force授权。日常核验无需这两个开关，也无需为重建写入数量而反复重写数据。

### 8.4 独立校验脚本自身有 bug

现象：`verify_stage03.py` 报 `KeyError: 'train.csv'`。
原因：`split_meta.json` 的 `manifests` 以 `train/val/test` 为键，我校验脚本按 `train.csv` 取。
处置：改正后通过。保留这个记录是因为它说明了一件事——**独立校验脚本也会错**，报错时要先确认是「被测对象错」还是「检查器错」，本次是后者。

### 8.5 掩码取值不止一种

抽查发现 `010.png` 是 `{0,1}`、`016.png` 是 `{0,2}`，与指南「合并前取值集合 `{0,1,2,3,4}`」一致。本阶段的处理：不修改文件，把「非零 → 训练标签 1、零 → 0」写进约定，留给 Dataset 阶段实现；阶段 04 统计全部 100 张的分布。

### 8.6 未解决

- `split_meta.json` 会因时间戳变化而哈希不同，只有三份 CSV 是稳定身份。文档里已区分，但若以后有人拿 `split_meta.json` 的哈希做一致性判断会产生误报。
- 进程峰值内存未测量。代码结构上不会整体加载（逐成员 1 MiB 分块、图像只读头），但没有 RSS 数字，属于「设计上成立、未测量」。
- 阶段 02 的验收记录提到可用内存会在 1.66—3.13 GiB 间波动，本阶段未受影响（提取过程未占满内存），但训练阶段仍需实测。

## 9. 给用户的学习要点

### 9.1 四个概念

**（1）官方 split 与派生 split 是两层。** 官方 CSV 只有 `train/test`；我们的 `train/val` 是从官方 `train` 里切出来的。所以清单里 `split=train` 的行**不等于**官方 train 行，追溯要看 `official_2cls_highshot.csv`。`test` 则是原样复制，一个都不许动。

**（2）固定配额的目的是消除「20% 怎么舍入」。** 602 的 20% 是 120.4，不同实现会给出 120 或 121；异常组 60 的 20% 是 12。写死「120 正常 + 12 异常」，跨机器、跨库版本都不会漂。

**（3）空 mask 字段是设计，不是缺失。** 正常图没有掩码文件。若把空串直接拼成路径，会得到 `data\visa\` 这样的目录路径，`Image.open` 报的是「不是文件」，很容易被误判成数据损坏。约定是：`mask` 为空 → 按图像尺寸生成全零数组。

**（4）掩码的「存储值」与「训练标签」是两套语义。** 官方把非零标注存成 1..4（甚至官方准备脚本会用 255 存图），我们训练用二类 CE，必须把 `>0` 映射为 1。这个映射放在读取时做，而不是改文件——改文件就再也回不到官方原始数据了。

### 9.2 三个自测问题

1. **`val.csv` 里有 `pcb1/Data/Images/Anomaly/010.JPG`，你能证明它不是随手挑的吗？**
   线索：5.3、`split_meta.json` 的 `algorithm.procedure`、`tests/test_prepare_data.py::test_matches_reference_permutation`。答案：它是 `default_rng(42)` 对「按 image 排序后的 60 张官方 train 异常图」做 `permutation` 后的前 12 个之一。任何一台机器按同样顺序重跑都得到同一批图，独立校验脚本重算后 `val selection reproducible: True`。

2. **如果明天发现某张训练图和某张测试图是同一块 PCB 的两次拍摄，应该改哪里？**
   线索：3.3 风险表、主指南「若在后续发现近重复…」。答案：不能只改 `test.csv` 了事。要么在阶段 04 建立分组划分（同一实物整组进同一 split），要么另立实验协议并改名，**不能继续叫 high-shot**，否则与官方基准不可比。

3. **`extraction_manifest.json` 里 1205 条记录的 `sha256` 有什么用？既然归档本身也有哈希。**
   线索：4.3、5.2。答案：归档哈希只能证明「用的哪份 tar」；单文件哈希证明「磁盘上这一张图确实来自那个成员」。前者在归档被替换时会变，后者能在只怀疑某几张图被改动时定位到具体文件；也是「已存在文件核验而非覆盖」的实现基础。

## 10. 下一阶段与修订记录

阶段 04（数据审计与可视化）会读取：

- `manifests\{train,val,test}.csv` 与 `data\visa\pcb1\...`（本阶段的产物，路径与列语义见 4.1）；
- `split_meta.json` 的 `checks`（作为验收基线；开始时仍核对清单身份，文件变化需重新验证）；
- 本阶段留下的约定：正常图空 mask、异常掩码非零即缺陷、test 不可改。

阶段 04 需要新增、本阶段未做的：图像解码审计（坏图/重复/近重复）、掩码取值分布统计、512 最近邻缩放后的前景保留率、叠加图人工检查。

必须保持不变的：三份 CSV 的哈希、官方 CSV 副本、`data\visa` 的目录结构、`val` 的 132 张具体名单。

| 日期 | 修改原因 | 改动与影响 | 修正后重新验证的证据 |
| --- | --- | --- | --- |
| 2026-10-31 | 首版建立 | 新增 `prepare_data.py`、`test_prepare_data.py`；生成三份清单、`split_meta.json`、`extraction_manifest.json` | 6.2 第 2 行；6.4—6.6 |
| 2026-10-31 | 入口脚本在变量不存在时被安全删除策略拦截 | `enter_project.ps1` 增加存在性判断（阶段 01 文件，哈希变更） | 后续所有 `run_project.ps1` 调用恢复正常 |
| 2026-10-31 | 纯校验运行覆盖了「首次写入」记录 | 新增 `extraction_baseline` 合并逻辑与 `--rewrite-extraction`；重建基线 | 6.2 第 4/5 行；`baseline write: files=1205 bytes=296989825` |
| 2026-10-31 | 独立校验脚本按键名取错 | 改为按 split 名索引 | 6.6 `VERIFY: ok` |
| 2026-10-31 | 本轮验收修正教学解释与证据边界 | 更正路径检查说明，区分文件头解析与完整解码；完整解码仍列为阶段 04 待做 | 已逐项对照路径守卫和 `Image.open` 调用；安全修复与最终复核进行中 |
| 2026-10-31 | 第三阶段独立验收发现覆盖权限、临时文件冲突、协议和发布保护问题 | 增加写入保护、原子发布、固定CLI协议及负向回归；修正文件头检查措辞 | 全套23项通过，正常复跑0文件写入/1205已核验，清单SHA不变；见验收记录 |
