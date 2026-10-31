# 阶段 04：数据审计与可视化——设计与实现学习文档

| 字段 | 内容 |
| --- | --- |
| 状态 | 已通过第四阶段验收；已完整重出 v2 分组划分及预览；未下载权重、未训练 |
| 日期 | 2026-10-31 |
| 工程 / 资源根 | `E:\桌面\AI\AI_Try\work\SegFormer` / `E:\桌面\AI\segformer_resources` |
| 前置阶段 | 03 的官方 CSV、已提取 PCB1、历史 v1 清单 |
| 验收入口 | [第四阶段验收与完善](../../reports/2026-10-31-stage04-acceptance.md) |
| 当前实验身份 | `segformer_b0_pcb1_v2_grouped`，以当前 `manifests/split_meta.json` 与 CSV SHA-256 为准 |

## 1. 本阶段解决什么问题

阶段 03 检查文件是否存在、文件头能否解析、路径划分是否正确。本阶段用 `Image.load()` 完整解码 1104 张图片与 100 张异常掩码，检查真实像素、标签取值、图像与 GT 对齐，并量化缩成 512 后的变化。

除了自动检查，还逐张查看原交付的 18 张五联图和近重复局部；新版清单生成后，15 张预览的 PNG 字节与已查看版本相同，另外 3 张新图实际查看。没有将机器检查通过当作标注一定正确。

本次验收发现两项需要完善：0042/0043 是高可信的同实物／固定场景近重复风险；原报告也误把“连通域数量减少”当作区域消失的充分证据。因此加入分组划分和逐原始区域身份追踪。

## 2. 技术栈与实际运行边界

| 技术 | 本机版本 | 作用 |
| --- | --- | --- |
| Python | 3.12.16，独立 Conda prefix | 审计与 CLI |
| Pillow | 12.3.0 | 完整解码、双线性图像缩放、最近邻标签缩放 |
| NumPy | 2.5.4 | 二值化、连通域、相关矩阵、可重复随机划分 |
| Matplotlib | 3.11.2，Agg | 五联图和总览图 |
| PyYAML | 6.0.3 | 读取并保存实际配置 |
| 标准库 | csv/json/hashlib/unittest | 清单、元数据、哈希和回归检查 |

本阶段没有安装或升级任何包，不需要新增 SciPy/OpenCV。环境、缓存、临时文件与批量图片继续留在资源根；小型审计、代码、配置、清单和学习文档在 Git 仓库中。

## 3. 设计与数据流

```text
官方 CSV + 已审风险组
  → v2 分组划分 → train / val / test / split_meta
                         ↓
配置 → 哈希及组约束校验 → 完整解码 → 原始 GT > 0 合并为 1
                                     ↓
                           原图 / 512 面积与原域追踪
                                     ↓
                           精确重复 / 全跨集近重复筛查
                                     ↓
                       JSON / CSV 摘要 + 固定五联图 + 人工记录
```

### 3.1 文件职责与接口

| 文件 | 接口与职责 |
| --- | --- |
| `configs/baseline.yaml` | 共用数据、模型、训练、评价、日志配置；阶段 04 只执行数据相关部分 |
| `segformer_lab/config.py` | `load_config` / `write_resolved_config`；相对路径按工程根解析，固定协议漂移则拒绝 |
| `segformer_lab/geometry.py` | `binarize_ids`、`compute_letterbox`、`letterbox_image`、`letterbox_label`、`count_components`、`component_retention`、dHash |
| `segformer_lab/prepare_data.py` | 保留 v1；显式 `--reviewed-groups` 运行 v2；组源验证、分组生成、独立组约束检查 |
| `segformer_lab/audit_data.py` | `verify_manifests` / `audit`；完整解码、统计、哈希、组约束和跨集重复筛查 |
| `segformer_lab/preview_data.py` | 固定选样、五联图、放大插图、index 与冻结 ID 保护 |
| `manifests/reviewed_groups.json` | 风险组成员、依据和处理决策；SHA 写入 split_meta |
| `manifests/val_preview_ids.txt` | 当前 v2 固定 3 正常 + 3 异常；本协议内不得静默改选 |
| `reports/data_audit` | 当前 v2 机器证据；历史 v1 小摘要另存 `data_audit_v1` |
| `tests/test_data_audit.py` | 几何、二值化、连通域身份、哈希、选样、配置 |
| `tests/test_group_split.py` | 分组合法性、可重复性、配额、v1/test 字节保持 |
| `tests/test_audit_groups.py` | 审计不只信任 metadata 的历史成功标志 |
| `tests/test_preview_ids.py` | 固定 ID 不能被默认参数或异常内容替换 |

### 3.2 数据与标签约定

1. 原图始终按 RGB；掩码保持原始 `L` 单通道文件，不覆盖写入。
2. 原始标签实测并集 `{0,1,2,3,4}`；先在内存中 `binary = (raw > 0).astype(np.uint8)`，得到 0/1。
3. 正常图无掩码文件，按原图尺寸构造 0 标签。
4. **先二值化，再缩放，再填充 255**。如果对填充后的标签再次 `>0`，255 会被误合并为缺陷。
5. 图像与标签共用一个 `Letterbox`。图像双线性、标签最近邻，防止标签插值出不存在的类别。
6. 255 是忽略区域，0 是正常背景。背景必须参与监督；损失与指标都不能把 255 当正常背景。

## 4. 关键实现与理解

### 4.1 同一个 letterbox 管理图像与掩码

```python
scale = min(input_size / width, input_size / height)
new_w = round(width * scale)
new_h = round(height * scale)
pad_left = (input_size - new_w) // 2
pad_top = (input_size - new_h) // 2
```

实际尺寸 1404×1070 → 512×390，左/右 0，上/下 61。有效面积为 512×390，不能用包含 padding 的 512² 冒充真实有效区面积。

画布的 23.828125% 是填充。如果误将其作为正常背景计入，前景占比会乘以 `390/512 = 0.76171875`，并非旧稿中的“稀释 4 倍”。CE 也必须按实际有效像素归约。

阶段 06 要直接复用这些几何函数；另写一套 resize 容易造成审计与 Dataset 不一致。此时尚未实现 Dataset 或逆变换指标，不能提前说原图评价已测试。

### 4.2 为什么必须追踪原始连通域

原始二值 GT 的每个 4 连通域分配一个 int32 ID，再对身份图用与标签完全一致的 Pillow NEAREST 采样。只要某个原始 ID 在输出有效区还有至少一个像素，就称该区域保留；完全没有则登记消失，并记录原始面积。

```python
original_ids, original_areas = _label_components(binary)
resized_ids = np.asarray(
    Image.fromarray(original_ids).resize((box.new_w, box.new_h), Image.NEAREST)
)
surviving_ids = set(int(v) for v in np.unique(resized_ids) if v > 0)
```

然后看原始 ID 与输出连通域的交叉对应：一个原域落入多个输出域是拆分；一个输出域包含多个原 ID 是合并。身份追踪不要求额外深度学习模型。

不能只比较前后数量：两个原域合并会减少数量，但都可能仍保留；一个原域消失、另一个拆成两块，又可能令总数不变。测试明确覆盖这些反例。

这些仍是**几何连通区域**，不是经过专家定义的真实缺陷实例；细划痕的断续标注也能形成多个域。

### 4.3 近重复为什么不能只用 dHash

同型号 PCB 的布局相似。初版 `dHash <= 8` 命中 91,218 对，无法用于可信过滤。现在对全部 362,564 对跨集图片计算 64×64 灰度 Pearson 相关，要求相关性 ≥0.985 且 MAD ≤6，dHash 只作为辅助字段。

所有阈值都是筛查规则，不能单独证明同一实物。0042/0043 的处理依据还有焊点、非规则网格及背景纤维的局部视觉重合。反过来，边缘像素存在差异也不能证明是不同板子。

v2 只把已审核风险组作为不可分单位，保留官方 test。未掌握原始物料 ID，所以不能声称已经证明全数据的实物级独立。

### 4.4 分组划分与固定预览的版本关系

v2 规则详见[验收报告第 3 节](../../reports/2026-10-31-stage04-acceptance.md#3-近重复处理与-v2-固定协议)。保持 seed 和配额相同，不代表划分就相同：随机排列的单位已经由图片变成组。因此将实验名改为 `segformer_b0_pcb1_v2_grouped`，重新保存整套清单与哈希。

旧版所有清单、元数据和预览 ID 先原样归档，再完整运行生成器；没有手工改一个 CSV 单元格。test 字节不变。新版中 0042/0043 都属于 train。

预览在本协议内冻结。CLI 默认计数为未指定：有固定 ID 就优先复用；显式要求数量不同，或文件含重复、空列表、不属当前 val 的图，直接报错，不静默重新选样。`write_ids` 也不能覆盖不同的既有 ID。

本轮协议升级使旧固定样例 0454 不再属于 val，因此有记录地归档旧 ID 并为 v2 重新生成；这是协议换版，不是模型训练中途换图。

### 4.5 五联图为什么需要局部插图

图像宽 1404，缺陷可能只占 0.1%。整体缩图上的红点不足以核对真实位置。对较小 GT 取外接框加边缘，生成原图与红色半透明掩码的放大插图；大圆形异常则看整图。

原始图片有些本来旋转 180°。只要原图、掩码、叠加及 512 输入方向一致，就不能把物体倒置误认成数据处理错误。

## 5. 当前真实结果

| 项目 | 实测 |
| --- | --- |
| 划分 | train 482 正常 +48 异常；val 120+12；test 402+40 |
| 完整解码 | 1104/1104 RGB 图、100/100 L 掩码成功；尺寸全为 1404×1070 |
| 正常标签检查 | 1004 张有效区全 0、填充全 255 |
| 异常前景 | 100 张均非空；中位 train 0.187648%、val 0.315787%、test 0.193939% |
| 512 前景面积比例保持率 | 中位 99.93862%，范围 94.67396%–104.26173% |
| 整张异常前景变空 | 0 张 |
| 原始连通域 | 328；保留 309、消失 19；消失涉及 12 张，原始面积合计 39 px、单域 1–8 px |
| 域数量净变化 | 减少 12 张、增加 14 张、不变 74 张 |
| 拆分 / 合并 | 37 个原始域拆分；20 个输出域包含多个原始 ID |
| 精确重复 | 0 组 |
| 已审核风险组 | 1 组、两成员都在 train，独立组检查通过 |
| v2 跨集近重复 | 本规则候选 0 对；不等于证明没有所有形式的泄漏 |
| 可视化 | 当前 train 12 + val 6 张五联图、两张总览、固定 ID；逐文件结论见验收报告 |
| 回归检查 | 62 项 unittest 通过；完整审计 13 项通过 |

本轮重生成再次校验 1205 个提取文件，写入 0 个文件；原始数据未改。当前报告中 test 统计只用于固定的数据质量审计，不能据此选择模型或调参。

## 6. 如何复现

每个新进程使用项目入口；不要用 PATH 上的裸 Python。新终端先执行：

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
. '.\scripts\enter_project.ps1'
```

当前 v2 的重复准备命令如下；已有不同内容的 CSV 会拒绝覆盖，不要自行删清单“修复”。

```powershell
Invoke-SegPython -m segformer_lab.prepare_data `
  --archive '..\..\docs\dataset\VisA\VisA_20220922.tar' `
  --source-zip '..\..\docs\dataset\VisA\spot-diff-main.zip' `
  --data-root "$SegResource\data\visa" `
  --category pcb1 --seed 42 --val-normal 120 --val-anomaly 12 `
  --reviewed-groups 'manifests\reviewed_groups.json' --out 'manifests'

Invoke-SegPython -m segformer_lab.audit_data `
  --config 'configs\baseline.yaml' --out 'reports\data_audit'
Invoke-SegPython -m segformer_lab.preview_data --split train --normal 6 --anomaly 6
Invoke-SegPython -m segformer_lab.preview_data --split val
Invoke-SegPython -m unittest discover -v
```

本轮已执行上述完整准备、审计、预览、回归；还实际执行了一次不带数量的 val 预览，依旧复用 6 张，ID 字节不变。再次准备会更新元数据时间等运行字段，三份 CSV 稳定身份仍应保持；已有训练 run 必须保存当时元数据快照。

## 7. 产物如何阅读

| 产物 | 看什么 |
| --- | --- |
| `summary.json`（schema /2） | 当前统计、检查、清单哈希、组验证和具体采样口径 |
| `summary.md` | 自动生成的简版表格；中文解释以本文及验收报告为准 |
| `per_image.csv` | 1104 行的文件 SHA、尺寸、原始值、前景、几何和逐图域统计 |
| `component_retention.json` | 100 张异常图的原域身份保留／消失及面积、拆分／合并 |
| `duplicates_exact.json` / `near_duplicates.json` | 当前 v2 的重复检测结果；历史候选在 `data_audit_v1` |
| `resolved_config.yaml` | 本次绝对路径与原始配置；换电脑需调整资源根，不能直接照搬 E 盘路径 |
| `stage04-acceptance.json` | 小型验收与代码/预览身份记录，便于跨端复核 |
| `stage04-unit-tests.txt` / `stage04-preview-reuse.txt` | 真实命令结果 |
| `runs/data_preview/train|val/index.json` | 当前预览名单、配置/清单/元数据哈希和图片位置；整批 PNG 留本地 |

`per_image.csv` 的 dHash 是十六进制字符串，Excel 可能误当数字；用代码或纯文本查看。

## 8. 风险、未做与后续判断

- **正常标注疑点：**登记 0196、0819、0983 三例；证据不足以改标，原数据保留，详见验收报告。
- **微小目标：**首轮仍是 512 全图、既定损失及普通采样；不预先加入裁块或加权采样。先完成基线并查看验证集的漏检和误检。
- **分辨率限制：**原图计分是必要的评价约定，不是图像细节恢复技术。若正式基线微小缺陷召回差，再建立独立的高分辨率／裁块对照配置，并记录训练成本。
- **模型阶段：**尚未下载 encoder 权重，尚未验证预训练加载；未实现真实 Dataset、损失、原图逆变换指标或训练器。
- **标注范围：**预览是抽样检查，不是 1104 张的全量专业复标。几何连通域不能直接当缺陷实例数。

## 9. 学习检查

1. **为什么总面积保留 99.94%，仍有原始区域消失？** 大区域主导面积，1–8 px 的碎小域可以全部丢失；面积与身份回答不同问题。
2. **为什么 `>0` 二值化必须在 padding 之前？** 原始非零 ID 是缺陷，后加 255 是忽略；顺序错会把忽略区改成缺陷。
3. **为什么 seed=42 没变，v2 仍改变划分？** 排列单位与输入长度改变，完整算法才定义实验身份；种子不是全部协议。
4. **为什么正常图不能仅用零掩码证明无漏标？** 零掩码来自 normal 标签，必须再看图像内容，存在疑点时保留记录。
5. **为什么先做预训练加载核验？** 下一阶段需要确认 encoder 真正继承权重；模型能前向不代表预训练加载正确。

## 10. 下一阶段与修订记录

下一步只执行阶段 05，具体可复制指令在[验收报告第 7 节](../../reports/2026-10-31-stage04-acceptance.md#7-下一步指令只执行阶段-05)。阶段 06 继续使用当前 v2 清单和几何实现；不得回到 v1 而不标记换版。

| 日期 | 修订与重新验证 |
| --- | --- |
| 2026-10-31 初版 | 完整解码、全跨集相似度筛查、18 张预览与初版测试 |
| 2026-10-31 验收 | 逐张视觉复核、登记正常图疑点；将 0042/0043 保守分组，完整生成并审核 v2 |
| 2026-10-31 验收 | 以原域身份追踪替代“净数量下降等于消失”的推断，重新审计量化 19 个微小域消失 |
| 2026-10-31 验收 | 固定 ID fail-closed、配置输出根与预览身份记录；62 项回归通过，v2 审计 13 项通过 |
