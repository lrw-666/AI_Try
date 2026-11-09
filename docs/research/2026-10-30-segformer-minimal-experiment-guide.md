# SegFormer 最小实验闭环：家庭电脑技术指南与 Agent 协作手册

> **2026-11-09状态说明：**本文保留10月30日的历史实施蓝图，教学与实验正文不变；现行入口为[主文档](./thesis-research-and-study-guide-zh.md)。当前题目《PCB外观缺陷检测方法研究》，工作一检测精度提升、工作二可解释性，适配和正常参照均为候选。PCB1 high-shot真实mask监督的SegFormer基线已有归档验收证据，DINO实验尚未开展。本轮仅采用当前E盘电脑核验：关键环境／数据／best工件已找到且身份一致，模型前向与val复算尚未在本轮执行；其他机器的检查不参与现状判断。下文“尚未创建”等属于原编写时状态。

> 编写日期：2026-10-30；复核修订：2026-10-30。适用场景：Windows 家庭电脑、NVIDIA GPU、Python/PyTorch，在 VisA PCB1 上完成有监督缺陷二值分割。
>
> **文档状态：实施指南，不是实验完成报告。** 本次交付只有本 Markdown；下文 `scripts/`、`src/`、`configs/` 是后续 Agent 应创建的工程接口，当前不能假定这些文件已经存在。安装、下载、解压和训练命令供你在家里电脑执行；本次没有在当前电脑完成这些操作，也没有实测显存、耗时或模型效果。
>
> **给执行 Agent：**先读本文件与[现行研究与学习主文档](./thesis-research-and-study-guide-zh.md)。按阶段交付，未通过当前阶段验收时，不启动后续完整训练。若环境提供 `superpowers:executing-plans` 或 `superpowers:subagent-driven-development`，按对应技能逐任务实施；本文件中的任务、接口和证据要求是项目约束。

**目标：**从真实图像和掩码出发，完成环境核验 → 数据审计与固定划分 → 预训练模型接入 → 训练/验证 → 最优权重保存 → 新进程加载 → 官方测试集评估 → 原图预测与失败分析。

**架构：**使用 Hugging Face Transformers 中的 SegFormer-B0 实现，全模型微调；将原 ADE20K 分类输出替换成背景/缺陷两类。数据处理、训练、评价和推理使用同一套公共模块，避免各脚本重复实现预处理和指标。

**技术栈：**Windows 10/11 x64、PowerShell、Python 3.11、PyTorch、Transformers、NumPy、Pillow；使用原生训练循环，不引入 Trainer、MMSegmentation、MMCV 或分布式训练。

**范围依据：**原研究路线与开题计划已合并至[现行研究与学习主文档](./thesis-research-and-study-guide-zh.md)。本实验提供开题可行性证据，不等同于最终论文完整实验。

**阅读导航：**第 3—7 节是电脑、权重和数据准备；第 8—9 节是给实现 Agent 的工程规格；第 10 节是实际运行顺序；第 14.4 节可直接复制给家里电脑的 Agent；第 17 节记录资料来源及本次实际检验范围。

---

## 1. 最终要得到什么

### 1.1 最小闭环的验收定义

- [ ] 在目标 GPU 上完成真实数据前向、有限 loss、反向传播和参数更新。
- [ ] 固定并保存训练、验证、测试清单，测试集保持官方 high-shot 测试成员不变。
- [ ] 完成一个预先记录训练预算的正式 run，遍历全部训练清单；每轮验证并保存最优模型。
- [ ] 在新 Python 进程中，仅从本地最优模型目录加载，完成推理和完整测试集计分。
- [ ] 生成缺陷类 IoU、Dice、precision、recall、正常图误报像素比例和可追溯预测图。
- [ ] 保存配置、依赖、源码版本、数据/权重来源、日志和实验摘要。

安装成功、输出一个 batch、训练 loss 下降、生成几张彩色图，分别只证明对应步骤可行。测试成绩很低也可以构成诚实的流程闭环，但全背景塌缩、标签错误、泄漏或指标错误必须记录并排查，不能作为有效方法结果解释。

### 1.2 三个级别分开命名

| 级别 | 内容 | 能说明什么 |
| --- | --- | --- |
| smoke | 1—2 个 batch，含保存和重载 | 环境、形状和梯度链路可用 |
| overfit | 固定 2 张异常训练图反复优化 | 数据、标签、优化和预测基本对应 |
| baseline | 全部派生训练集、逐轮验证、完整独立测试 | 在本配置和协议下的初步基线 |

不要将 smoke 或 overfit 写成 PCB1 完整基线，也不要将验证集分数改名为测试集分数。

## 2. 默认选择与约束

### 2.1 本手册的默认方案

| 项目 | 默认值 | 说明 |
| --- | --- | --- |
| 任务 | 两类语义分割 | `0=background`，`1=defect`，不是图像分类或异常分数建模 |
| 数据 | VisA `pcb1` | 官方 `split_csv/2cls_highshot.csv` |
| 预训练来源 | `nvidia/segformer-b0-finetuned-ade-512-512` | ImageNet 编码器预训练后经过 ADE20K 分割微调；不是只用 ImageNet 权重 |
| 更新方式 | 全模型微调 | 本轮不冻结编码器、不加 DINO、不融合 |
| 图像处理 | RGB，整图直接缩放到 `512×512` | 为最小闭环固定的起点；会改变长宽比，需记录这一局限 |
| 损失 | 两通道 logits＋未加权交叉熵 | 先保持简单；类别不平衡导致失败后再另开改进实验 |
| 验证选择 | 原图空间累计缺陷 IoU 最大 | 不根据测试集选 epoch、阈值或后处理 |
| 预测 | logits 上采样后 `argmax(dim=1)` | 不额外调阈值，无连通域筛选 |
| 训练起点 | batch=2，30 epochs，AdamW，lr=6e-5 | 本手册的工程起点，不是经验证的最优参数 |

模型来源和预训练阶段应写入报告。使用 ADE20K 微调权重便于接通整个分割网络，但之后与其他预训练方案比较时，不能把差异全部归因于网络结构。[NVIDIA 模型卡](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)

### 2.2 全局约束

1. 不改变官方测试成员；只从官方训练部分分层划出验证集，先分原图再增强或切块。
2. 正常图构造全零掩码，异常原始掩码非零值合并为 1；异常掩码缺失或为空必须报错。
3. `do_reduce_labels=False`，`num_labels=2`，标签为 `[B,H,W]` 的 `torch.long`，值域只能为 `{0,1}`。
4. 所有正式评价回到原图尺寸，用原始二值 GT；不能静默在缩小标签上计分。
5. 运行输出独立建目录，已有非空目录默认拒绝覆盖；调试与正式运行不得复用权重目录。
6. 原始数据和下载权重只读使用；不要提交数据、虚拟环境、token 或大量 checkpoint。
7. 未实测的版本组合、显存、耗时、性能、导师认可均标为待验证。

### 2.3 优先审查的五种失败

| 失败条件 | 必须得到的行为/证据 |
| --- | --- |
| 正常图没有掩码、异常图掩码缺失 | 前者生成零掩码；后者停止并列出图像 ID |
| 原始多 ID/255 掩码、缩放丢掉缺陷 | 二值化正确；最近邻缩放；统计前景消失比例并人工看图 |
| 同一原图泄漏到多个 split | 断言失败并给出重叠路径；保存所有清单哈希 |
| logits 只有输入的约 1/4 分辨率 | logits 先插值回目标尺寸，再作类别判断；不插值类别 ID |
| 重载丢失两类配置或只会在线加载 | 新进程离线加载成功；与保存前同一图 FP32 logits 在容差内一致 |

## 3. 家里电脑先做盘点

这里假设家里电脑是 Windows＋NVIDIA 显卡；此前口述为 RTX 4060，型号、显存和可用空间须在家中实测。若实际系统为 Linux，保留数据与算法协议，将环境命令转换为对应 shell，不混用两套命令。

### 3.1 目录与磁盘

建议数据和权重放在仓库外；系统盘空间紧张时，将缓存也放到数据盘。以下路径是示例，**首次执行前替换成家里真实路径**：

```text
D:/work/AI_Try/                         项目仓库
D:/datasets/VisA/                       解压后数据根目录
D:/models/segformer-b0-ade/              固定版本的原始权重快照
D:/cache/huggingface/                   下载缓存
```

先预留约 15—20 GB 作为环境、原始/解压数据、缓存与实验产物的工作余量；这是操作预算，不是数据精确占用或必要硬件规格。实际以剩余空间、tar 成员统计与 checkpoint 大小为准。

```powershell
# 在新开的 PowerShell 中执行；仓库应已复制/拉取到此目录。
$ProjectRoot = 'D:\work\AI_Try'
Set-Location -LiteralPath $ProjectRoot
$env:VISA_ROOT = 'D:\datasets\VisA'
$env:SEGFORMER_WEIGHTS = 'D:\models\segformer-b0-ade'
$env:HF_HOME = 'D:\cache\huggingface'
$env:PYTHONUTF8 = '1'
# 将后续 here-string 管道中的文本按 UTF-8 交给 Python；尤其适用于 Windows PowerShell 5.1。
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$ErrorActionPreference = 'Stop'
git status --short
nvidia-smi
py -0p
Get-PSDrive -PSProvider FileSystem
```

环境变量仅对当前终端及其子进程生效。换终端后需重新设置；配置中不要保存另一台电脑的绝对路径。若没有 Git 或 Python 3.11，先通过[Git 官方入口](https://git-scm.com/downloads)和[Python 官方 Windows 下载页](https://www.python.org/downloads/windows/)安装对应工具，并重开终端。Python 必须是 x64 的 3.11；这里的 `py -3.11` 不会自动安装该解释器。

### 3.2 跨电脑同步要核对

- 仓库有文档不代表数据也在；当前仓库将 VisA 图像 tar 设为忽略，Git 拉取不会带来它。
- 本机历史路径和 GPU 检查结果不代表家里环境。
- 当前电脑曾发现 `docs/dataset/VisA/VisA_20220922.tar` 和 `spot-diff-main.zip`；可自行搬运或按官方入口重新取得，搬运后要在家中校验。
- 不自动恢复用户删除的旧资料，不把 MVTec 异常检测路线接入本次工程。

## 4. 环境配置：固定版本，先验证 CUDA

### 4.1 推荐版本组合

这是为首次复现选择的固定版本起点，不代表 2026 年最新版本，也没有在家里电脑实装验证。PyTorch 的 torch/torchvision 配对来自官方历史版本安装表；Transformers 依赖边界见该版本源码。[PyTorch 安装表](https://pytorch.org/get-started/previous-versions/)、[Transformers v4.57.1 依赖](https://github.com/huggingface/transformers/blob/v4.57.1/setup.py)

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.x，记录实际补丁版本 |
| torch | 2.7.1，CUDA 12.6 wheel |
| torchvision | 0.22.1，与 torch 同源安装 |
| transformers | 4.57.1 |
| huggingface-hub | 0.36.0，保持 `<1.0` |
| numpy | 1.26.4 |
| Pillow | 11.3.0 |
| PyYAML / tqdm / pytest | 6.0.2 / 4.67.1 / 8.4.2 |

`torchvision` 用于保持完整视觉依赖配对；本轮图像操作以 Pillow 为主。不预装 TensorFlow、librosa、MMCV、OpenCV 或 Notebook 全家桶，减少无关依赖。

### 4.2 创建专用虚拟环境

```powershell
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 venv creation failed' }
# 不依赖 Activate.ps1，可避免 PowerShell 脚本执行策略问题。
$Py = (Resolve-Path -LiteralPath '.venv\Scripts\python.exe').Path
& $Py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& $Py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $Py -m pip install transformers==4.57.1 huggingface-hub==0.36.0 numpy==1.26.4 Pillow==11.3.0 PyYAML==6.0.2 tqdm==4.67.1 pytest==8.4.2
if ($LASTEXITCODE -ne 0) { throw 'Project dependencies installation failed' }
& $Py -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed' }
```

每条安装命令都应查看退出码；失败就先处理当前步骤，不把后续报错混成一组。`$ErrorActionPreference='Stop'` 不保证原生命令（如 python/git）失败时自动停止，因此上面显式检查 `$LASTEXITCODE`，其他阶段也遵循此规则。已有 `.venv` 时先检查它的解释器和包；不直接在不明环境上重复创建或删除环境。新终端中重新设置 `$Py`。`python`、`pip`、IDE 解释器都要指向同一个 `.venv`，不要把包装到全局 Python。

### 4.3 驱动与 CUDA 的关系

使用预编译 PyTorch wheel 时，通常无需另装完整 CUDA Toolkit 或 cuDNN；仍需要兼容的 NVIDIA 驱动。`nvidia-smi` 中的 CUDA Version 是驱动支持能力提示，`torch.version.cuda` 是该 PyTorch 的构建版本，两个数字不必相等。不要为了数字一致而同时重装驱动、Toolkit 和 torch；先实际执行 CUDA 运算。驱动版本要求应按 [NVIDIA CUDA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)核对。

```powershell
@'
import sys, torch, torchvision, transformers
print('python:', sys.version)
print('executable:', sys.executable)
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('transformers:', transformers.__version__)
print('build_cuda:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable: check driver and wheel'
print('gpu:', torch.cuda.get_device_name(0))
print('vram_gib:', torch.cuda.get_device_properties(0).total_memory / 2**30)
x = torch.randn(256, 256, device='cuda', requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print('CUDA forward/backward OK', float(loss.detach()))
'@ | & $Py -
```

验收：`pip check` 无冲突、CUDA 可用、真实 GPU 名称与显存已记录、前向和反传完成。此处仅为环境检查，不是 SegFormer 成功训练。

让 Agent 把本节版本写入 `requirements.txt`（常规依赖）和 `requirements-torch.txt`（torch 配对及其 CUDA index），另保存完整 `pip freeze` 和安装命令。只保存 `pip freeze` 不能完整说明 CUDA wheel 来源。

## 5. 源码与权重获取

### 5.1 三种源码来源要分清

| 来源 | 在本实验中的用途 |
| --- | --- |
| [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) | 作者原始实现、论文与结构参考；不是本手册运行依赖 |
| [Transformers v4.57.1 SegFormer](https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/segformer) | 本次实际安装并执行的模型实现 |
| 本项目 `src/segformer_lab/` 与 `scripts/` | 后续 Agent 创建的数据、训练、评价和推理代码 |

作者仓库基于旧版 MMSegmentation，并给出旧 torch/MMCV 示例。为减少 Windows 首次实验的编译成本，本手册直接使用 Transformers 实现，不执行作者仓库的旧安装命令，也不混装两个训练框架。[作者仓库说明](https://github.com/NVlabs/SegFormer#installation)

如需离线读源，可在**仓库外**任选一个参考目录执行：

```powershell
git clone --depth 1 https://github.com/NVlabs/SegFormer.git D:\reference\SegFormer
git clone --branch v4.57.1 --depth 1 https://github.com/huggingface/transformers.git D:\reference\transformers-v4.57.1
```

这些 clone 是可选参考，不需要 `pip install -e`。实际 import 的位置可检查：

```powershell
& $Py -c "import inspect; from transformers import SegformerForSemanticSegmentation as M; print(inspect.getfile(M))"
```

### 5.2 下载权重并固定 revision

首选模型：[nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)。先取得快照，再离线使用，避免训练中途联网。下列代码在首次下载前持久化 commit SHA；中断后重跑沿用同一 SHA，完成后重跑仅校验本地文件。使用 Hub 的按 commit 缓存，核对返回快照目录与必需文件，再复制到专用权重目录，避免将通用 `local_dir` 的旧内容误记成新版本。这是环境/权重准备代码，可在安装依赖后直接执行。

```powershell
@'
import hashlib, json, os, re, shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

def main():
    repo = 'nvidia/segformer-b0-finetuned-ade-512-512'
    required = ['config.json', 'preprocessor_config.json', 'model.safetensors']
    target = Path(os.environ['SEGFORMER_WEIGHTS']).resolve()
    target.mkdir(parents=True, exist_ok=True)
    source = target / 'source.json'

    def write_source(meta):
        tmp = target / 'source.json.tmp'
        tmp.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        tmp.replace(source)

    def hashes(directory):
        result = {}
        for name in required:
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f'Missing/empty file: {path}')
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    if source.exists():
        meta = json.loads(source.read_text(encoding='utf-8'))
        if meta.get('repo_id') != repo or not re.fullmatch(r'[0-9a-f]{40}', meta.get('revision', '')):
            raise RuntimeError('Unrecognized provenance; verify this directory manually')
        if meta.get('status') == 'complete':
            if hashes(target) != meta.get('sha256'):
                raise RuntimeError('Snapshot changed: do not silently overwrite it')
            print('Verified local snapshot:', target, meta['revision'])
            return
        if meta.get('status') != 'downloading':
            raise RuntimeError('Unknown/legacy source.json; inspect before reusing')
    else:
        if any(target.iterdir()):
            raise RuntimeError('Nonempty directory without provenance; use a new directory')
        sha = HfApi().model_info(repo).sha
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha):
            raise RuntimeError('Hub did not return a full commit SHA')
        meta = {'repo_id': repo, 'revision': sha, 'status': 'downloading'}
        write_source(meta)

    sha = meta['revision']
    snapshot = Path(snapshot_download(repo_id=repo, revision=sha, allow_patterns=required))
    if snapshot.name != sha:
        raise RuntimeError('Unexpected cache revision; refuse to label these files')
    expected = hashes(snapshot)
    for name in required:
        tmp = target / (name + '.tmp')
        shutil.copyfile(snapshot / name, tmp)
        tmp.replace(target / name)
    if hashes(target) != expected:
        raise RuntimeError('Snapshot copy verification failed')
    meta.update(status='complete', sha256=expected)
    write_source(meta)
    print('Saved verified copy:', target, sha)

main()
'@ | & $Py -
if ($LASTEXITCODE -ne 0) { throw 'Weight snapshot preparation failed' }
```

`source.json` 已保存三个文件的 SHA256，用于后续复用和搬运一致性校验；这些本地哈希不是独立的发布者校验公告。下载失败时保持 `status=downloading` 并沿用已存 revision，不改成另一个模型。完成后可搬运整个专用权重目录（含 `source.json`），在家里重跑本段做离线校验，不必搬缓存。旧版指南生成的不含 status/哈希的 `source.json` 应先人工核对，再升级元数据或换专用目录，不能自动视为完成。代码的缓存结构假设依据于 [Hub v0.36.0 下载实现](https://github.com/huggingface/huggingface_hub/blob/v0.36.0/src/huggingface_hub/_snapshot_download.py)，更换 Hub 版本时重新核查。模型、数据和代码分别有自己的许可，保留对应来源与条款记录。

### 5.3 两类模型的正确构建

下面是后续 `model.py` 应遵守的核心逻辑，不是独立训练脚本：

```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

processor = SegformerImageProcessor.from_pretrained(
    weights_dir, local_files_only=True,
    do_reduce_labels=False,
    size={"height": 512, "width": 512},
)
model, initial_info = SegformerForSemanticSegmentation.from_pretrained(
    weights_dir, local_files_only=True, use_safetensors=True,
    num_labels=2,
    id2label={0: "background", 1: "defect"},
    label2id={"background": 0, "defect": 1},
    ignore_mismatched_sizes=True,
    output_loading_info=True,
)
allowed = {"decode_head.classifier.weight", "decode_head.classifier.bias"}
for category in ("missing_keys", "mismatched_keys"):
    entries = initial_info.get(category) or []
    names = {item if isinstance(item, str) else item[0] for item in entries}
    if names - allowed:
        raise RuntimeError(f"Unexpected initialization: {category}={entries}")
if initial_info.get("unexpected_keys") or initial_info.get("error_msgs"):
    raise RuntimeError(f"Unexpected loading diagnostics: {initial_info}")
assert model.config.num_labels == 2
assert processor.do_reduce_labels is False
```

ADE20K 原输出为 150 类，改成两类时分类层形状不匹配并重新初始化是预期行为；检查加载告警只涉及预期分类层。编码器或大量其他层未载入则停止检查模型/版本。固定随机种子应发生在构建模型之前。

`ignore_mismatched_sizes=True` 只用于首次改变类别的初始化。初次加载也应启用 `output_loading_info=True`，只允许 `decode_head.classifier.weight/bias` 出现在 missing/mismatched 集合中，其他缺失、多余键或错误均停止检查。标签递减会把背景 0 转成忽略标签，本任务必须关闭。[预处理器实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/segformer/image_processing_segformer.py)、[模型实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/segformer/modeling_segformer.py)

重载已经训练的 `best/` 时，仅省略 `ignore_mismatched_sizes` **不等于检查了所有权重**：Transformers 对缺失/多余键可能只告警。后续 `load_trained_model()` 必须显式检查加载信息，避免丢失分类层时悄悄重新初始化。核心实现如下；放入 `model.py` 并补齐 import：

```python
def load_trained_model(checkpoint_dir):
    processor = SegformerImageProcessor.from_pretrained(
        checkpoint_dir, local_files_only=True,
    )
    model, info = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint_dir, local_files_only=True, use_safetensors=True,
        output_loading_info=True,
    )
    bad = {key: info.get(key) for key in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
           if info.get(key)}
    if bad:
        raise RuntimeError(f"Incomplete checkpoint: {bad}")
    if model.config.num_labels != 2 or model.config.id2label != {0: "background", 1: "defect"}:
        raise RuntimeError("Unexpected class configuration")
    if processor.do_reduce_labels is not False:
        raise RuntimeError("Unexpected label reduction in saved processor")
    return model, processor
```

此处依据 [Transformers v4.57.1 权重加载实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_utils.py)。训练后加载不得再传 `num_labels=2` 或新的分类映射来覆盖保存配置。

## 6. VisA 数据获取与解压

### 6.1 获取官方代码和图像

官方入口：[amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)，README 指向的[VisA 原始图像归档](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)。官方划分必须使用 `split_csv/2cls_highshot.csv`。不要下载仅含正常训练图的第三方 MVTec 风格重打包版本来代替。数据为 [CC BY 4.0](https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET)，引用和再分发应保留归属；仓库代码许可单独处理。

官方代码可放仓库外：

```powershell
git clone https://github.com/amazon-science/spot-diff.git D:\reference\spot-diff
git -C D:\reference\spot-diff rev-parse HEAD
```

如果用已有 `spot-diff-main.zip`，保存 ZIP 的 SHA256 和来源；如果不能确认 commit，就如实记 `commit=unknown`，不要编造 SHA。至少单独保存所用 CSV 的 SHA256。

检查准备搬运的归档，例如：

```powershell
# 换成图像 tar 的真实位置；这里只计算本地文件哈希。
Get-FileHash -LiteralPath 'D:\downloads\VisA_20220922.tar' -Algorithm SHA256
tar -tf 'D:\downloads\VisA_20220922.tar' | Select-Object -First 20
```

本地算出的哈希能核对搬运前后是否一致，若没有官方校验值，不能因此宣称与发布者文件完全一致。先查看归档成员层级，确认没有非预期路径，再解到专用目录，不覆盖已有整理数据：

```powershell
New-Item -ItemType Directory -Force -Path $env:VISA_ROOT | Out-Null
tar -xf 'D:\downloads\VisA_20220922.tar' -C $env:VISA_ROOT
if ($LASTEXITCODE -ne 0) { throw 'VisA extraction failed' }
```

若目标目录已有数据，先检查并复用，不重复解压覆盖。完整解压只是便利选择，训练只读 PCB1；空间不足时让 Agent 根据 CSV 精确提取 PCB1 成员。当前指南没有实际执行全量解压或解码。

### 6.2 数据根目录怎样判断

`VISA_ROOT` 应是 CSV 中相对图像路径的基准。例如 CSV 路径是 `pcb1/Data/Images/Normal/0000.JPG`，就应使 `VISA_ROOT / 该路径` 存在。不能凭文件夹名字猜根目录，也不要把 `Images` 或 `Normal` 当作根。

原始布局大致为：

```text
VisA/
  pcb1/
    Data/
      Images/
        Normal/...
        Anomaly/...
      Masks/
        Anomaly/...
```

实际大小写、扩展名和目录层级以所用 CSV 和归档成员为准，不手写“同名 PNG”规则猜掩码路径。

## 7. 数据审计、二值化与固定划分

### 7.1 官方 CSV 与预期数量

官方 CSV 表头为 `object,split,label,image,mask`；`object` 只选小写 `pcb1`，`split` 为 `train/test`，`label` 为 `normal/anomaly`，分别映射到 0/1。`image`、`mask` 是相对数据根的路径。读取后仍应打印表头和唯一值，异常字段立即报错，不把未知标签默认为正常。见 [high-shot CSV](https://github.com/amazon-science/spot-diff/blob/main/split_csv/2cls_highshot.csv)。

现有接续记录对所用官方 CSV 统计为：

| 官方部分 | 正常 | 异常 | 合计 |
| --- | ---: | ---: | ---: |
| train | 602 | 60 | 662 |
| test | 402 | 40 | 442 |

这些数字用于对照，不作为静默强行截取数据的依据。若下载版本不同，应报告源 CSV、哈希和差异；不得自行拼数据凑数。

### 7.2 派生验证集的确定规则

本手册采用可复现的明确划分：

1. 保留官方 442 张测试图不动。
2. 对官方训练部分，按正常/异常分别分层。
3. 每层先按完整相对图像路径排序；用 `numpy.random.default_rng(42)` 的同一个生成器，先处理正常层、再处理异常层，分别对层内索引 `permutation`。
4. 每层前 `ceil(该层数量×0.2)` 张为验证，其余为训练；保存各输出清单时再按图像路径排序。

如果源数据符合上表，最终应为：

| 派生部分 | 正常 | 异常 | 合计 |
| --- | ---: | ---: | ---: |
| train | 481 | 48 | 529 |
| val | 121 | 12 | 133 |
| test | 402 | 40 | 442 |

这是本项目自行派生的验证协议，不是 VisA 官方给出的第三个 split。12 张异常验证图样本较少，选择结果可能波动；先完成单种子流程，最终论文再按需要验证重复性。

若发现重复文件、近重复图或同一实物跨部分，先记录证据，按组划分的变更应单独命名协议，不能悄悄修改官方测试后仍声称原协议。

### 7.3 统一 manifest 格式

`manifests/pcb1_highshot_s42/{train,val,test}.csv` 使用 UTF-8、相对 `VISA_ROOT` 的 POSIX 风格路径，字段固定为：

```text
image_path,mask_path,image_label,official_split,split,width,height,foreground_pixels
```

- `image_label`：正常 0、异常 1；`official_split` 保留官方来源；`split` 为派生部分。
- 正常 `mask_path` 为空字符串，不是 `nan` 路径；每次读取时按原图 H/W 生成零掩码。
- `foreground_pixels` 为原图空间二值掩码的缺陷像素数；正常必须为 0，异常应大于 0。
- 保存源 CSV 副本、源/派生清单 SHA256、种子、分层规则和数量统计到 `split_metadata.json`。

### 7.4 掩码处理约定

```python
from PIL import Image
import numpy as np

if image_label not in (0, 1):
    raise ValueError(f"Unknown image label: {image_label}")
image = Image.open(image_path).convert("RGB")
if image_label == 0:
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
else:
    raw = np.asarray(Image.open(mask_path))
    assert raw.ndim == 2, "Inspect mask mode; do not silently guess RGB labels"
    mask = (raw > 0).astype(np.uint8)
    if not mask.any():
        raise ValueError(f"Empty anomalous mask: {mask_path}")
assert mask.shape == (image.height, image.width)
assert set(np.unique(mask)).issubset({0, 1})
```

原始掩码的非零值可能是多个缺陷 ID；先统计原始值，再合并为 1。不能把非零像素直接解释成多类训练标签，也不能把 255 自动当忽略区域。使用 `np.asarray` 保留调色板掩码的 ID，不先 `convert('L')` 改变标签含义。

若真实文件为多通道掩码，先查证通道编码及官方代码，再明确转换；遇到异常数据不要自动转成正常图。PCB1 的原始非零 ID 对应 Bent、Melt、Scratch、Missing，二值任务无需分别预测这些类别，见[官方 ID 定义](https://github.com/amazon-science/spot-diff/blob/main/utils/id2class.py)。

官方准备脚本实际位于 [`utils/prepare_data.py`](https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py)，其转换会将非零异常变成 255；若使用该脚本的重组数据，仍须将 255 转成 1。本手册直接读原始 CSV 和归档目录，**无需运行该重组脚本**，避免复制全数据及混淆两套目录结构。

### 7.5 必须生成的审计结果

让 Agent 全量解码 PCB1 的图像和异常掩码，并输出：

- 三个 split 正常/异常数量、尺寸分布、坏文件和重复路径列表。
- 跨 split 路径互斥断言；原图内容哈希重复项也列出供检查。
- 掩码 mode、原始像素 ID、尺寸对应、每图前景面积及占比。
- 对训练/验证部分，记录缩放到 512 后缺陷完全消失的异常图数量和 ID；同时保留原图面积。测试的同类分析在正式评估后展开。
- 从训练/验证部分导出至少 4 张正常、8 张异常的原图/GT/叠加图；覆盖较小缺陷，不只选清晰样本。

只因全局统计合理不能跳过人工看图。你应抽看图像与 GT 的位置是否对应，尤其是细小缺陷和异常图是否实际有非空标注。正式测试前，对测试数据只做预先定义的文件、尺寸与标签完整性审计；用来选择分辨率、损失、采样或其他开发配置的案例分析应限于训练/验证部分，测试错误分析放在配置固定、正式评估之后。

## 8. 后续 Agent 应创建的最小工程

以下目录都在项目仓库内；大文件仍在仓库外。不要提前创建网页、GUI、数据库或大型训练框架。

```text
AI_Try/
  requirements.txt
  requirements-torch.txt
  configs/segformer_b0_pcb1.yaml
  scripts/
    check_env.py
    prepare_visa.py
    inspect_data.py
    smoke.py
    train.py
    evaluate.py
    predict.py
  src/segformer_lab/
    __init__.py
    data.py
    model.py
    metrics.py
    engine.py
    artifacts.py
  tests/
    test_data_protocol.py
    test_metrics.py
    test_checkpoint.py
  manifests/pcb1_highshot_s42/
    source_2cls_highshot.csv
    train.csv
    val.csv
    test.csv
    split_metadata.json
  outputs/segformer/                    本地结果；不提交大文件
```

命令入口全部从仓库根目录运行。让 Agent 新建最小 `pyproject.toml`，统一通过 editable install 解决 `src` 导入；不能出现“IDE 可运行、PowerShell import 失败”。若项目届时已有打包配置，先合并而非覆盖。所有脚本要有 `if __name__ == '__main__': main()`。

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "segformer-lab"
version = "0.1.0"
requires-python = ">=3.11,<3.12"

[tool.setuptools.packages.find]
where = ["src"]
```

创建 `src/segformer_lab/__init__.py` 和上述打包文件后，执行 `& $Py -m pip install -e . --no-deps`；依赖仍由第 4 节版本文件管理。所有相对配置/输出路径以仓库根为基准，缺少必需环境变量时清楚报错。

### 8.1 公共接口契约

| 文件 | 必须实现的接口与责任 |
| --- | --- |
| `data.py` | `read_manifest(path) -> list[dict]`；`load_raw_sample(row, root) -> (PIL.Image, np.ndarray)`；`VisASegDataset(manifest, root, processor)` |
| `model.py` | `build_initial_model(weights_dir, seed) -> model`；`load_trained_model(checkpoint_dir) -> (model, processor)`；后者严格加载 |
| `metrics.py` | `BinaryCounts.update(pred, target)` 累计 int64 TP/FP/FN/TN；`compute()` 返回明确定义的指标 |
| `engine.py` | 单轮训练、验证和原图评价；验证/测试复用同一函数，差别仅在清单与输出路径 |
| `artifacts.py` | 配置/日志/哈希/来源保存；拒绝覆盖非空 run；保存模型与训练状态 |

训练 Dataset 每个样本返回 `pixel_values: float32[3,H,W]`、`labels: int64[H,W]`，批处理后为 `[B,3,H,W]` 和 `[B,H,W]`。评价需额外读取原图 GT 和原始尺寸，不把变尺寸 GT 用普通 `stack` 强行拼 batch。

### 8.2 配置文件的明确内容

下列是 `configs/segformer_b0_pcb1.yaml` 的规格，由 Agent 按此实现解析；路径以项目根为基准，`data_root` 与 `weights_dir` 从对应环境变量读取，不依赖 YAML 自动展开环境变量。

```yaml
experiment: segformer_b0_pcb1_highshot_s42
data_root_env: VISA_ROOT
weights_dir_env: SEGFORMER_WEIGHTS
manifest_dir: manifests/pcb1_highshot_s42
seed: 42
input_size: [512, 512]  # [height, width]
resize_policy: stretch
augmentation: none
num_labels: 2
do_reduce_labels: false
train:
  epochs: 30
  batch_size: 2
  num_workers: 0
  optimizer: adamw
  learning_rate: 0.00006
  weight_decay: 0.01
  scheduler: none
  loss: cross_entropy
  class_weights: null
  amp: true
  gradient_accumulation_steps: 1
  clip_grad_norm: 1.0
  early_stopping: false
  drop_last: false
evaluation:
  batch_size: 1
  amp: false
  space: original_image
  prediction: argmax
  checkpoint_metric: defect_iou_global
  postprocess: none
  empty_metric_value: null
output_root: outputs/segformer
```

这些设置旨在明确一个可解释的初始基线。无增强、恒定学习率和未加权 CE 不是最终性能方案；先看结果，再以独立 run 修改。分辨率会影响小缺陷可见性，调试可降低，但不能把 256 结果标成 512。

## 9. 预处理、训练与模型保存的实现要点

### 9.1 预处理只做一次

- 输入先转 RGB，交给 `SegformerImageProcessor`：图像缩放、`1/255` rescale 和预训练归一化。
- 调用时传入原始 PIL/uint8 图像；不要先 `ToTensor()/255` 再让 processor 除一次 255。
- 分割标签走 `segmentation_maps=mask`；由 processor 用最近邻缩放，输出 `labels`，并断言 0/1。
- 本轮无随机增强。后续若加几何增强，必须对图像/掩码同步，颜色增强只对图像做。

```python
batch = processor(images=image, segmentation_maps=mask, return_tensors="pt")
pixel_values = batch["pixel_values"].squeeze(0)
labels = batch["labels"].squeeze(0).long()
assert pixel_values.shape == (3, 512, 512)
assert labels.shape == (512, 512)
assert set(labels.unique().tolist()).issubset({0, 1})
```

执行 Agent 应核对指定版本的图像处理实现，而不是依赖从别的模型复制的默认参数。[Transformers 分割预处理源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/segformer/image_processing_segformer.py)

### 9.2 训练核心

传入 `labels` 后，所选实现会将较低分辨率 logits 插值到标签尺寸再计算交叉熵；不需要先将预测 `argmax`，也不额外对同一 logits 重复算一遍 loss。本轮训练全部可训练参数。[SegFormer forward 实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/segformer/modeling_segformer.py)

下面只示范一个训练 batch，Agent 仍须实现完整 epoch、验证、日志、异常处理和保存：

```python
model.train()
optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
    outputs = model(pixel_values=pixel_values.cuda(), labels=labels.cuda())
    loss = outputs.loss
if not torch.isfinite(loss):
    raise RuntimeError("Non-finite loss; save batch IDs and configuration")
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

先执行 `model.to('cuda')`，再创建 `torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)` 与 `torch.amp.GradScaler('cuda', enabled=use_amp)`，确保优化器引用目标设备上的参数。不手动把整个模型 `.half()`；autocast 只覆盖前向/损失计算，反传在上下文外。[PyTorch 2.7 AMP 文档](https://docs.pytorch.org/docs/2.7/amp.html)

每轮开始 `train()`；验证、测试、独立推理统一使用 FP32、`eval()`＋`torch.inference_mode()`，不沿用训练的 autocast，确保选择best与独立计分的精度设置一致。验证后下一轮恢复 `train()`。记录 mean train loss 时按实际图像数量加权，包含最后不足 batch 的样本。AMP训练同时记录scaler跳过更新的情况；loss有限不保证每步梯度都有限。

固定 Python、NumPy、torch、CUDA 随机种子，训练 loader 使用固定 `torch.Generator`、`shuffle=True`；验证/测试 `shuffle=False`。记录确定性设置，固定种子并不保证跨硬件逐位一致。

### 9.3 保存最佳模型与中断恢复

历史 best 初始化为 `None` 或负无穷。第一次有效验证即使 IoU 为 0 也保存 `best/`；之后仅当原图空间全局缺陷 IoU 严格大于历史最佳时更新，相同时保留更早 epoch。验证指标无效则停止并记录原因，不写入虚假的 best。至少保存：

```text
run/
  config.resolved.json
  environment.json
  requirements.freeze.txt
  provenance.json
  history.jsonl
  manifests/                     固化的源/派生清单及哈希
  best/                          供独立推理使用的最佳模型导出
    config.json
    model.safetensors
    preprocessor_config.json
    selection.json               最佳epoch及导出文件哈希，与完整代对应
  resume/
    epoch_0001/                  完整的一代恢复点，保留最近两代及最佳epoch
      model/                     模型、config与processor
      training_state.pt
      complete.json              全部文件完成后最后写入，含epoch和文件哈希
  latest.json                    指向最新完整恢复代；临时文件写完后replace
  run_summary.json
```

用 `model.save_pretrained(best_dir, safe_serialization=True)` 和 `processor.save_pretrained(best_dir)`。`best/` 用于推理；`resume/epoch_NNNN/` 中同一代的模型与状态用于恢复。`training_state.pt` 至少包含 epoch、global step、optimizer、AMP scaler、历史 best指标及 `best_epoch`、DataLoader generator 和随机数状态；只加载本项目自己生成的训练状态。

`save_pretrained` 不保存优化器，不能单凭它宣称支持原地续训。首次工程至少支持 epoch 边界恢复：中途崩溃从最近完成 epoch 重跑；不声称逐 batch 精确恢复。改变数据划分、分辨率、损失或 lr 后应建新 run，不作为原实验 resume。

**避免覆盖唯一恢复点：**每个 epoch 的模型和状态先写到新的一代目录，核验后最后写 `complete.json`，再以临时 JSON＋`Path.replace` 更新 `latest.json`。始终保留上一完整代；恢复只读有完成标记且哈希/epoch一致的一代，忽略未完成目录。不要先覆盖唯一的 last 模型再写训练状态，否则断电可能丢掉原本可恢复的上一轮。保留最近两代及最佳 epoch 即可，其余旧代的清理只针对已验证的本 run 目录。`best/` 导出也不得覆盖唯一好副本；导出损坏时可从保留的最佳完整代重新生成。

**best文件完整也可能滞后：**例如新一代已记录 `best_epoch=2`，但更新导出前中断，原 `best/` 仍是完整的epoch1。恢复与独立评估前，应将 `best/selection.json` 的epoch及文件哈希，与最新完整代状态所选的 `best_epoch` 对照；不一致时从保留的最佳完整代重新导出，核验后才计分。增加这一中断场景测试，不能仅以best目录存在且可加载作为验收。

PyTorch 2.7 的 `torch.load` 默认 `weights_only=True`；若训练状态直接含 NumPy 随机状态等对象，默认加载可能失败。本工程约定：完整训练状态只从本机本工程生成并确认来源的文件，以 `torch.load(path, map_location='cpu', weights_only=False)` 恢复；外部预训练模型仍使用 safetensors。先创建模型/优化器并恢复状态，最后恢复随机数和 DataLoader generator 状态，再迭代下一轮；否则模型初始化会消耗已恢复的随机序列。必须实际测试优化器、scaler 和随机状态恢复，不把仅权重加载称为续训。[PyTorch 2.7 torch.load](https://docs.pytorch.org/docs/2.7/generated/torch.load.html)

### 9.4 原图推理与计分

因为本轮直接整图拉伸，没有 padding/crop，可把 logits 直接插值回原图 H/W。若以后改 letterbox 或切块，这一步必须实现对应逆变换。

```python
import torch.nn.functional as F

model.eval()
with torch.inference_mode():
    logits = model(pixel_values=pixel_values.to(device)).logits
    logits_original = F.interpolate(
        logits.float(), size=(original_height, original_width),
        mode="bilinear", align_corners=False,
    )
    pred = logits_original.argmax(dim=1)[0].cpu().numpy().astype("uint8")
```

先上采样连续 logits 再判类别；不能先 argmax 再双线性缩放类别图。保存的预测 mask 像素值建议为 0/255 便于查看，但指标计算内部始终转换成布尔或 0/1。

## 10. 操作顺序：让 Agent 实现后逐关运行

**以下是自定义脚本的 CLI 契约，不是现成工具。** 执行 Agent 必须先实现、测试并通过 `--help`，再把准确可执行命令交给你。`--amp` 必须显式解析 `true/false` 字符串，不使用 `argparse` 的 `type=bool`（它会把字符串 `false` 误判为真）。新运行的优先级是 CLI 覆盖 YAML，再保存合并后的配置；**resume 不走这条覆盖规则，只恢复已有 run 的配置**。所有 CLI 出错必须非零退出；不得打印成功后悄悄跳过缺图、空标签或 GPU 错误。

| 模式 | 唯一训练停止预算 | 数据与保存规则 |
| --- | --- | --- |
| smoke | 固定 1—2 batch | 只用训练图；保存/重载临时模型 |
| overfit | `max_steps`，本例200次优化器更新 | 固定2张训练异常；忽略baseline epochs，不读val/test；保存最后模型和训练样本预测 |
| pilot/baseline | 约定的全部 epochs | 每轮完整训练＋验证，以验证IoU选best |
| resume | 既有run的剩余预算 | 从完整代恢复，不能再次执行已经完成的轮次 |

### 阶段 A：环境与源码入口

- [ ] 按第 4—5 节完成环境、CUDA 测试和权重快照。
- [ ] 创建最小工程、依赖文件及源码导入方式。
- [ ] 实现 `check_env.py`：输出解释器、torch/CUDA、GPU/显存、包版本、驱动、源码 Git HEAD 和 dirty 状态。

```powershell
New-Item -ItemType Directory -Force -Path outputs/segformer | Out-Null
& $Py scripts/check_env.py --output outputs/segformer/environment.json
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed' }
& $Py -m pip freeze | Set-Content -Encoding utf8 outputs/segformer/requirements.freeze.txt
```

验收：真实 GPU 前向/反传通过；明确包和模型来源。不要在这阶段安装 DINO、做模型融合或先开发可视化系统。

### 阶段 B：只处理数据

`prepare_visa.py` 消费官方 CSV 和原始 tar 解压目录，输出第 7 节 manifests；`inspect_data.py` 全量解码 PCB1、输出 JSON 审计和样例图。

数据脚本只需要 `VISA_ROOT`，不得因为尚未下载模型而要求 `SEGFORMER_WEIGHTS` 存在。已有 manifests 时比较源CSV、种子和派生清单哈希；一致则复用，不一致则停止并报告，不能静默重划覆盖。

```powershell
& $Py scripts/prepare_visa.py --data-root $env:VISA_ROOT --official-csv D:\reference\spot-diff\split_csv\2cls_highshot.csv --category pcb1 --val-ratio 0.2 --seed 42 --output manifests/pcb1_highshot_s42
& $Py scripts/inspect_data.py --config configs/segformer_b0_pcb1.yaml --output outputs/segformer/data_audit
& $Py -m pytest tests/test_data_protocol.py tests/test_metrics.py -q
```

验收：数量/互斥/标签断言通过、坏图为零或已明确处理、原图叠加人工核对通过。既有源文件不符合协议时停止，不要用跳过样本的方式“修好”。

### 阶段 C：真实小样本 smoke

实现 `smoke.py`：从派生训练集确定性取 1 张正常＋1 张异常，处理到指定尺寸，FP32 前向/反传，检查梯度和至少一个参数变化，保存、重载后在 eval 下比较同一图 logits。

```powershell
& $Py scripts/smoke.py --config configs/segformer_b0_pcb1.yaml --run-dir outputs/segformer/smoke_s42 --amp false
& $Py -m pytest tests/test_checkpoint.py -q
```

验收：loss 有限、标签和输出类别正确、有参数更新、本地重载可用；FP32 保存前后 eval logits 以 `rtol=1e-4, atol=1e-5` 比较。记录 GPU 峰值显存和实测单步时间，不把这个速度直接当完整训练耗时。

### 阶段 D：固定异常训练图过拟合

`train.py --mode overfit`：仅在派生训练集中筛选异常图，再按 `image_path` 字典序选前 2 张，保存选择 ID；batch=2、无增强、FP32、重复训练 200 个优化器更新步。该模式以 `--max-steps` 为唯一预算，忽略 YAML 中的 baseline `epochs=30`，不能跑完30次就提前结束。只读取这两张训练图，不进入验证/测试，也不套用验证IoU选best的逻辑；保存最后模型及这两图的预测。计分不作为泛化效果。

```powershell
& $Py scripts/train.py --config configs/segformer_b0_pcb1.yaml --mode overfit --overfit-images 2 --max-steps 200 --amp false --run-dir outputs/segformer/overfit_s42
```

验收：保存开始/结束的 loss、每图预测、每图前景 IoU；明确计分空间，建议与正式评价一样恢复原图。工程调试可把训练 IoU≥0.8 作为参考目标，但不保证 200 步必达，也不以0.8作为研究有效性门槛。未达时先排查标签、缩放、梯度和预测前景；若这些均正确且已有可见拟合进展，记录诊断后可进入一次pilot测量，不无限加步或调参追0.8。若仍全背景或梯度异常，则先处理该问题。这个结果只说明对这些训练样本的拟合情况。

### 阶段 E：短程完整数据试跑

```powershell
& $Py scripts/train.py --config configs/segformer_b0_pcb1.yaml --mode baseline --epochs 1 --run-dir outputs/segformer/pilot_s42
```

验收：全部派生训练图遍历一次、全部验证图完成计分、best导出与一整代resume恢复点成功保存，日志含样本数、训练/验证耗时、峰值显存和前景预测比例。

此时才能估计正式运行时间。粗略预算可用“实测单轮训练＋验证时长”乘轮数再留保存/启动余量，明确它是估计而非保证。不要把 pilot 权重直接继续成正式基线而未记录其额外训练轮数。

### 阶段 F：正式初始基线

```powershell
& $Py scripts/train.py --config configs/segformer_b0_pcb1.yaml --mode baseline --run-dir outputs/segformer/baseline_s42_v1
```

默认从原始预训练快照重新初始化，按 30 epochs 训练；根据全部验证集的全局缺陷 IoU 选择 best。本轮不加增强、加权采样、Dice/Focal 或后处理；如果 CE 导致塌缩，先把这次失败作为已记录基线，再另开 run 比较有依据的调整。

如果 pilot 显示预算不可承受，**在正式训练前**记录缩减配置，例如 10 epochs 或 batch=1；明确实际训练预算。完整训练指全部指定训练样本和约定预算都执行，不表示模型已充分收敛。

恢复示例，只有 Agent 实现并验证 epoch 边界恢复后才能使用：

```powershell
& $Py scripts/train.py --resume outputs/segformer/baseline_s42_v1
```

恢复必须读取 `run/config.resolved.json` 与固化清单的哈希，沿用原 mode、总epochs、batch、AMP及优化器配置。例如原run用 `--epochs 10 --amp false`，恢复后仍为10轮和FP32；不重新读默认30轮/AMP的YAML覆盖。若run已完成则报告已完成，不追加轮次。只允许通过当前 `VISA_ROOT` 重新定位相同数据，重新校验清单；改变训练参数或划分就创建新run。恢复和独立评估不需要原始预训练目录仍然存在，不能因此强制读取 `SEGFORMER_WEIGHTS`。

### 阶段 G：新进程独立测试与单图预测

先确定最佳 epoch 和推理规则，再首次正式测试；测试后不能因为成绩不理想反向选择阈值或 best epoch。`evaluate.py` 消费 run 中已解析配置，而不是允许测试时任意覆盖预处理。

```powershell
# 开启离线模式，证明推理不依赖重新下载。
$env:HF_HUB_OFFLINE = '1'
& $Py scripts/evaluate.py --run-dir outputs/segformer/baseline_s42_v1 --checkpoint best --split test --output outputs/segformer/baseline_s42_v1/test

# $ImagePath 替换成任意待检查的真实图像，不预设文件编号存在。
$ImagePath = 'D:\datasets\VisA\pcb1\Data\Images\Anomaly\REPLACE_WITH_REAL_FILE.JPG'
& $Py scripts/predict.py --checkpoint-dir outputs/segformer/baseline_s42_v1/best --image $ImagePath --output outputs/segformer/baseline_s42_v1/demo
```

`predict.py` 从 `best/preprocessor_config.json` 取得预处理和输入尺寸，不依赖 GT。若要保存带 GT 的展示图，由评价脚本处理。后续需要联网下载时在当前终端 `Remove-Item Env:HF_HUB_OFFLINE`。

验收：全部 442 张测试图均被计入，图像 ID 可追溯；输出原图尺寸 mask、叠加图和 JSON。独立预测不能要求调用训练脚本或重新初始化分类层。

## 11. 指标定义与可视化

### 11.1 主指标统一到原图空间

对全部测试图累计像素级 TP、FP、FN、TN（缺陷为正类）：

```text
defect_iou_global  = TP / (TP + FP + FN)
defect_dice_global = 2TP / (2TP + FP + FN)
defect_precision  = TP / (TP + FP)
defect_recall     = TP / (TP + FN)
normal_fp_pixel_ratio = 正常测试图上的预测缺陷像素总量 / 正常测试图像素总量
```

主指标是全测试集累计量，不是逐图平均；大图或大缺陷贡献更多，应在摘要说明。正常图虽没有前景，预测误报仍通过 FP 进入全局指标，同时单独报告正常图误报比例。不能只在 40 张异常图上计算然后标“完整测试”。

分母为零返回 JSON `null` 并记录原因，不输出非法 JSON 的 NaN。全背景预测时 precision 可为 null，但 defect recall、IoU 和 Dice 在有异常 GT 的完整测试中应为 0。逐图空 GT 且空预测的 IoU/Dice 也记 null，不给正常空图填 1 再抬高均值；正常图另用误报衡量。

逐图结果另存 `per_image.csv`，至少包括图像路径、图像标签、TP/FP/FN/TN、原图尺寸、GT/预测缺陷面积、IoU/Dice（可空）。图像级误报率需要定义像素图到图像判定规则，本轮不默认添加该指标。

### 11.2 必须保存的测试产物

```text
test/
  metrics.json
  per_image.csv
  predictions/            每图二值 mask，保留相对路径或唯一 ID，避免同名覆盖
  overlays/               原图＋GT＋预测＋错误图
  failures.md
```

至少展示正常误报、异常漏检、定位偏差和相对成功样本；不存在某类失败时如实写明。错误图明确区分 TP、FP、FN，并给图例。不要只展示最好的两三张。

### 11.3 成本记录

每次 run 保存总/可训练参数量、输入尺寸、batch、AMP 状态、轮数、训练/验证时长、`torch.cuda.max_memory_allocated()` 和 GPU 信息。峰值显存指 PyTorch 分配器口径，不等同于 `nvidia-smi` 的整卡占用。

推理计时可在 10 次预热后测至少 50 次，CUDA 计时前后同步；写明是否包含预处理、磁盘 IO、上采样和保存。不能把不含 IO 的模型前向时间称为整套系统延迟。

## 12. 最少但关键的验证

以下测试针对会破坏实验可信度的行为，不要求为普通配置读写堆大量测试。

| 测试文件 | 必须断言的行为 |
| --- | --- |
| `test_data_protocol.py` | 正常缺掩码→零图；异常缺掩码→报错；`[[0,2],[255,0]]`→`[[0,1],[1,0]]`；图/掩码几何一致；split 无交集，训练＋验证恰好覆盖官方训练 |
| `test_data_protocol.py` | 人工合成一块明显缺陷，resize 后位置正确且值域仍为 0/1；异常原始前景为空→报错；空 mask 字段不变成路径 `nan` |
| `test_metrics.py` | TP=1、FP=1、FN=1 时 IoU=1/3、Dice=1/2；正常图误报计入 FP；全背景预测的 recall=0；空分母为 null |
| `test_metrics.py` | 不等尺寸图按累计计数合并，不意外取逐图均值；预测/logits 恢复到原图 H/W 后才计分 |
| `test_checkpoint.py` | 保存与严格重载两类模型，eval FP32 logits 接近；关闭网络后可加载；不存在忽略形状不匹配的重载 |
| `test_checkpoint.py` | 首次验证 IoU=0 仍保存 best；更晚同分不覆盖；本机训练状态能恢复 optimizer/scaler/epoch，并恢复随机数序列 |
| `test_checkpoint.py` | 删除分类层参数后严格重载必须失败；模拟新一代保存中断时仍能回退上一完整代；原run为10轮/FP32，resume仍沿用该预算和精度 |

另外，用可替换的 Hub 客户端验证权重准备逻辑：下载中断后 revision 不变；完整快照可无网络复用；缺文件或哈希改变时报错；已有非空未知目录不得覆盖。用 stub 通过只证明控制逻辑，不代替真实 Hub 下载和模型加载。

数据测试优先用小型人工图/掩码，不依赖大数据下载；模型重载测试使用本地官方快照或已有本地模型，测试缺权重时必须清楚区分 skip 与 pass。GPU smoke 作为独立集成检查，不能用 CPU 单测通过代替 GPU 可用。

## 13. 常见问题及排查次序

| 现象 | 优先检查 | 处理方式 |
| --- | --- | --- |
| `CUDA unavailable` | 解释器、torch 是否 CPU wheel、驱动、GPU | 打印路径/版本；使用同一 `$Py`；先跑 CUDA 小运算，不立即重装全环境 |
| `operator torchvision::nms does not exist` | torch/torchvision 版本及 CUDA wheel 来源 | 在专用环境内按匹配配对修复；不要在训练代码里绕过错误 |
| HF 超时或 SSL 错误 | 官方模型页/网络/系统时间/代理 | 保存原始错误；另一电脑取得快照并校验；不关闭 TLS 验证 |
| model import 引出音频/其他依赖冲突 | 是否用了已有杂乱环境 | 回到专用 `.venv`；不要为了视觉任务追加全部音频依赖 |
| `Target ... out of bounds` / CUDA assert | 标签是否为 255、多 ID、float、三通道 | CPU 上先看 unique/shape；恢复 0/1 int64；修复后重启进程 |
| loss 为 NaN/Inf | 标签、归一化、lr、AMP | 记录图像 ID；关 AMP 用 FP32 复现；检查重复 rescale 与输入范围 |
| loss 下降但预测全背景 | 缺陷占比、缩放消失、输出通道、采样 | 先看固定异常图能否过拟合；排除错误后再试独立加权/损失实验 |
| 显存不足 | 实测分辨率、batch、残留进程 | batch 2→1，确认 AMP，关闭自己的无用进程；必要时另开较低分辨率 run |
| Windows DataLoader 卡住 | 多进程、入口保护、杀毒/IO | 初始 `num_workers=0`；稳定后再单独测试 2 个 worker |
| 保存后离线加载失败 | config/processor/权重是否在同一目录 | 补齐保存契约；加载训练 best 不得重新请求在线原模型 |
| 预测图错位 | H/W 顺序、长宽比、原图尺寸 | Pillow `.size` 为 W/H，插值 `size` 为 H/W；核对原图 GT |
| 测试分数异常高 | 标签泄漏、误读 GT、只算背景准确率 | 核查 split、实际 checkpoint、混淆量和输出图；不直接认定方法有效 |

OOM 后不要杀掉不属于本次任务的其他进程；先确认占用来源。batch=1 不等于有效 batch=2，除非明确实现梯度累积并记录。降到 256 能排障，但小缺陷可能消失，必须另标调试配置。

## 14. 人与 Agent 的协作方式

### 14.1 人负责的关键判断

你需要确认家中路径/硬件、数据和权重能访问，抽看原图/GT/预测，决定可接受运行预算，向导师确认对象边界。每阶段要求 Agent 展示具体文件、日志和图，不只接受“已经完成”。

至少自己能解释：标签 0/1 的含义、训练/验证/测试的分工、loss 与 IoU 的区别、为什么保存 best、为什么小缺陷可能在缩放后消失，以及实际改了哪些参数。

### 14.2 Agent 分工与依赖

| 工作包 | 可并行性 | 应交付什么 |
| --- | --- | --- |
| A 环境/权重 | 可与 B 并行 | 版本锁定、GPU 日志、权重 revision/哈希 |
| B 数据审计/划分 | 可与 A 并行 | manifests、审计 JSON、GT 叠加图、数据测试 |
| C 训练与保存 | 依赖 A/B 和共同接口 | smoke、overfit、pilot、正式 run 工程 |
| D 独立评估/复核 | 可提前实现人工指标测试；真实测试依赖 C | 严格重载、完整测试、预测图、实验卡 |

一个统筹 Agent 维护接口、配置和公共指标；并行 Agent 各自负责文件，避免同时改 `data.py`、配置格式或同一个 run。默认仅一个 GPU 训练任务；CPU 数据审计可并行，但不得争抢 GPU 或给验证计时增加不可解释负载。

### 14.3 每阶段交接格式

```text
阶段：A/B/C/...
已完成：列文件和对应验收证据
实际命令：完整命令、工作目录、退出码
配置：版本/种子/数据清单/输入大小/AMP/batch
产物：可打开的日志、图、模型或 JSON 路径
未完成：具体哪一步，没有证据就不要写完成
阻塞：原始报错、最小复现、已排除项
下一步：只列一个最优先动作
```

### 14.4 可直接发给家里电脑 Agent 的启动提示词

```text
请读取项目 AGENTS.md、docs/research/2026-10-30-segformer-minimal-experiment-guide.md
以及 docs/research/thesis-research-and-study-guide-zh.md。

我要在本机完成 SegFormer-B0 / VisA PCB1 high-shot 有监督二值分割最小闭环。
这份文档是实施规格，不表示其中自定义脚本已经存在。

先只读盘点 Git 状态、Python/GPU/显存、数据归档/解压目录和本地权重，
报告真实路径和缺项。保留用户已有修改和删除。按文档接口创建最小工程，
分阶段完成环境、数据审计、smoke、小样本过拟合、pilot、正式训练、独立测试。
安装和大文件下载前先核对已有可用资源；不要重复下载或覆盖数据。

只读已有官方 high-shot CSV，测试集不参与调参；正常掩码为0，异常非零合并为1，
关闭 do_reduce_labels，num_labels=2；评价恢复到原图空间。
主指标为完整测试集累计缺陷 IoU/Dice，同时报告正常图误报像素比例。
所有脚本统一配置、CLI和预处理，出错非零退出，非空 run 默认不得覆盖。
overfit严格执行max_steps，不让baseline epochs截断；resume只读原run的resolved配置，
不把原CLI覆盖重置成YAML默认值；重载必须检查loading_info，不能只省略ignore_mismatched_sizes。

当前范围不含 DINO、融合、异常检测备选、GUI或云租赁。
不要自动提交/推送，不提交数据、环境、下载权重和训练大文件。
有实际日志和产物才报告完成；遇到阻塞给出定位证据。
先完成A/B阶段及验收，再进入训练。完整训练启动前展示pilot实测资源和预估预算，
沿用我已授权的本地实验范围，预算明显超出本地条件时再说明需要我决定的变更。
```

## 15. 复现记录与完成报告模板

每个正式 run 结束创建 `experiment_card.md`，内容至少为：

```markdown
# Experiment card

- Status: complete / interrupted / failed
- Task: supervised binary PCB defect segmentation
- Data source / archive SHA256:
- Official CSV source / commit / SHA256:
- Derived manifest SHA256 / seed / counts:
- Pretrained model repo / revision / file SHA256:
- Source Git HEAD / dirty changes or patch:
- Python / torch / torchvision / transformers / CUDA / driver:
- GPU / VRAM:
- Input resize / normalization / labels:
- Epochs planned / completed / selected best epoch:
- Batch / optimizer / learning rate / weight decay / AMP:
- Training / validation time and peak allocated VRAM:
- Evaluation space / decision rule / empty-denominator convention:
- Test images evaluated / expected:
- TP / FP / FN / TN:
- Defect IoU / Dice / precision / recall:
- Normal-image false-positive pixel ratio:
- Output paths: config / logs / model / metrics / figures
- Main failure cases:
- Limitations and next smallest experiment:
```

源码存在未提交修改时，HEAD 不足以复现，另保存 diff 和新增源码文件快照。小型代码、配置、清单、测试、文档及摘要可纳入 Git；数据、缓存、`.venv`、weights、runs 默认本地保留。Agent 修改 `.gitignore` 时保留已有规则，针对新产物添加条目，不用全局忽略规则把本应保存的 manifests 或实验摘要一起隐藏。

最终报告要分别写“运行完成到哪一步”“得到什么实测结果”“有什么局限”。本轮只做 PCB1、单种子、固定缩放的初始实验；不能据此宣称工业部署可用、跨类别泛化或研究创新已经成立。

## 16. 开题前的收敛原则

原计划优先争取 11 月 3 日前出现可核验的首模型闭环，并在 11 月 8 日前收窄拟研究问题；这是10月30日的历史安排。现行进度与排查优先级以[研究与学习主文档](./thesis-research-and-study-guide-zh.md)及实际验收证据为准。

第一轮应回答的是：数据是否正确、这个监督分割方案能否在本机训练、主要漏检/误报是什么。小缺陷缩放损失、前景稀少或解码定位不足可以成为后续实验问题，但必须由样例和对照支持。先保住报告与 PPT，不在截止前为了追求成绩增加多个模型。

## 17. 官方资料索引与核查边界

以下入口于 2026-10-30 用于核对方案；安装/API按文中固定版本解释。网页与仓库未来可能改变，执行时保存实际 revision 和文件哈希。

1. [SegFormer 作者仓库](https://github.com/NVlabs/SegFormer)：论文、原始框架与旧安装依赖。
2. [Transformers v4.57.1 SegFormer 文档](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/segformer)：模型/处理器接口。
3. [该版本模型源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/segformer/modeling_segformer.py)：分类输出、损失和 logits。
4. [该版本预处理器源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/segformer/image_processing_segformer.py)：标签递减、归一化和掩码缩放。
5. [NVIDIA B0 模型卡](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)：权重来源与预训练背景。
6. [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)：torch/torchvision 配对与 CUDA wheel。
7. [PyTorch 2.7 AMP](https://docs.pytorch.org/docs/2.7/amp.html)：混合精度和梯度缩放。
8. [VisA 官方仓库](https://github.com/amazon-science/spot-diff)、[high-shot CSV](https://github.com/amazon-science/spot-diff/blob/main/split_csv/2cls_highshot.csv)、[utils/prepare_data.py](https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py)：下载、划分、标签转换。
9. [Hub 下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)：快照下载与 revision 固定。
10. [NVIDIA CUDA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)：驱动与运行时关系。
11. [Transformers v4.57.1 权重加载实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_utils.py)、[Hub v0.36.0 快照下载实现](https://github.com/huggingface/huggingface_hub/blob/v0.36.0/src/huggingface_hub/_snapshot_download.py)：加载告警、加载信息与按commit缓存行为。

### 编写机的只读抽查记录

编写过程中，辅助 Agent 对本机已有 ZIP/TAR 做了只读检查，未解压到磁盘、未下载图像或启动训练：所用官方 CSV 的 SHA256 为 `c3331eede15f2da8a75b380d4fbfb24f75ee036ec9a25147eca898da6e972f09`；TAR 含 PCB1 全部 1,104 个图像路径及 100 个异常掩码路径。100 张掩码均可解码，为 L 模式、1404×1070（宽×高），取值总集合 `{0,1,2,3,4}`；仅抽查 10 张图像解码，均为 RGB 和同尺寸，**不是全部图像已通过完整性验证**。这些结果只描述编写机的归档，家中仍须重新审计。

### 2026-10-30 复核与修订记录

本轮在原文档上复核数据协议、命令衔接和检查点规则；当前解释器为 Python 3.13.2，未安装目标 torch/torchvision/Transformers，所以没有执行目标环境中的GPU前后向或真实模型保存/恢复。

| 检验项 | 实际范围与结论 |
| --- | --- |
| 数据划分复算 | 按排序、default_rng(42)、分层ceil规则复算，得到529/133/442，与第7节一致 |
| 调试样本 | 派生训练集前两张异常为 `004.JPG`、`005.JPG`；原图前景2201/2792像素，512最近邻后383/486，约占0.146%/0.185% |
| 掩码与字节重复 | 100张异常mask均非空，缩到512后无整张前景消失；1104张图像文件字节SHA256无重复。字节无重复不排除视觉近重复或同一实物 |
| 下载重试复现 | 使用隔离stub复现旧逻辑中断后会重新解析main；修订为预存SHA、按commit缓存、文件哈希校验与离线复用 |
| 工程规格修正 | 补齐严格加载信息检查、overfit唯一停止预算、resume配置优先级与成套保存恢复点 |
| 代码静态检查 | 17段PowerShell解析通过；6段Python及2段PowerShell内嵌Python语法通过；TOML、围栏及相对文件链接检查通过。这些不是目标依赖运行测试 |
| 隔离逻辑检查 | 从文档提取代码，用stub完成6类下载行为、5类严格加载分支检查；用实际Pillow/NumPy完成人工掩码二值化/校验的4类检查。没有调用真实Hub或模型 |

上述数据检查使用已有本地归档，没有新增图像下载、全量解压或模型训练。数据完整性统计不证明模型可训练、200步过拟合可达0.8或当前配置效果充分；后续须按第10节在目标环境验收。

**仍须在家庭电脑验证：**操作系统与驱动、实际 GPU/显存、固定依赖能否安装导入、权重可达性、图像完整性、训练耗时、最佳输入分辨率和所有实测指标。本文给出可执行规格与核查路径，不替代这些实测。
