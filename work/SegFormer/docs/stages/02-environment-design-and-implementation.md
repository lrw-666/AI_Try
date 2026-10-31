# 阶段 02：独立 Conda 环境与 GPU 验证——设计与实现学习文档

| 字段 | 填写内容 |
| --- | --- |
| 阶段编号与名称 | 步骤 2：建立独立环境并验证 GPU |
| 状态 | 已验证（环境创建、依赖安装、导入、CUDA 前后向、随机权重 SegFormer 前后向、前后快照比对均实际完成） |
| 创建/更新时间 | 2026-10-31 12:52 +08:00 |
| 对应代码身份 | 仓库 HEAD `dfe8fa1`；本阶段新增 `segformer_lab\env_check.py`（SHA-256 `08104b6c...8759`，536 行）、`requirements-lock.txt`、`requirements-cu126.txt`、`environment-history.yml`；脚本沿用阶段 01 |
| 项目根目录 / 资源根目录 | `E:\桌面\AI\AI_Try\work\SegFormer` / `E:\桌面\AI\segformer_resources` |
| Python / 环境prefix | `E:\桌面\AI\segformer_resources\env\python.exe`，Python 3.12.16（Anaconda 打包，MSC v.1942） |
| 依赖的上一阶段 | `01-workspace-design-and-implementation.md`：入口脚本、快照脚本、`audit\before-*`、`paths.py` |

**独立验收更新：** 本阶段已完成复核与修正，详见[阶段01—02验收记录](../../reports/2026-10-31-stage01-02-acceptance.md)。原首次运行输出作为历史证据保留；当前状态以修正后的脚本、最新 `reports/environment.json` 与验收记录为准。

## 1. 本阶段解决什么问题

阶段 01 只搭了「外壳」，环境位置是空的。本阶段要回答三个问题，且必须有证据：

1. **解释器到底是谁？** 不是「应该是 3.12」，而是 `sys.executable` 指向哪、`site-packages` 在哪、用户 site 有没有被偷偷启用。
2. **GPU 真的能用吗？** 不是「`nvidia-smi` 能看到卡」，而是 PyTorch 能在卡上做前向、反向、拿到有限的梯度。
3. **搭建过程有没有污染 base 和全局配置？** 不是「我没动」，而是前后快照比对为「无差异」。

输入：阶段 01 的空资源根、`condarc.yml`、入口脚本。
输出：可用的独立环境、确定版本的训练栈、`reports\environment.json`、`requirements-lock.txt`、`environment-history.yml`、`audit\after-*` 与比对结论。

边界：**不下载模型权重、不训练、不处理数据**。本阶段的 SegFormer 检查用的是**随机初始化**的模型，只验证「代码路径能在 GPU 上跑通」，不产生任何精度结论。

## 2. 技术栈与版本

全部为实测值，查询方式见本节末尾。

| 技术/库 | 实际版本 | 本阶段作用 | 状态与证据 |
| --- | --- | --- | --- |
| Conda | 24.11.3 | 创建独立前缀环境 | 已运行 |
| Python | 3.12.16 | 项目解释器 | 已运行（`environment.json`） |
| pip | 26.2.2 | 安装训练栈 | 已运行 |
| PyTorch | 2.10.0+cu126 | 深度学习框架 | 已运行（前向+反向） |
| torchvision | 0.25.0+cu126 | 与 torch 成对，后续图像变换 | 已安装并导入 |
| CUDA runtime（wheel 内） | 12.6 | GPU 内核 | 已运行 |
| cuDNN | 由 torch 自带（`torch.backends.cudnn.version()` 记录） | 卷积加速 | 已查询 |
| Transformers | 4.57.6 | SegFormer 模型与 image processor | 已运行（随机权重前向） |
| tokenizers | 0.22.2 | Transformers 依赖 | 已安装 |
| safetensors | 0.8.0 | 权重读取格式 | 已安装 |
| huggingface_hub | 0.36.2 | 后续固定 revision 下载 | 已安装 |
| NumPy | 2.5.4 | 数组与后续数据管线 | 已安装 |
| Pillow | 12.3.0 | 图像读写（含中文路径） | 已安装 |
| Matplotlib | 3.11.2 | 后续可视化（Agg 后端） | 已安装 |
| PyYAML | 6.0.3 | 配置解析 | 已安装 |
| tqdm | 4.70.1 | 进度条 | 已安装 |
| TensorBoard | 2.21.0 | 后续训练曲线 | 已安装 |
| NVIDIA 驱动 | 617.42 | 支持 CUDA 12.6 运行时 | 已读取 |

版本与路径证据：

```powershell
# 在项目上下文中
Invoke-SegPython -c "import sys,torch,torchvision,transformers;print(sys.executable);print(torch.__version__,torchvision.__version__,transformers.__version__)"
Invoke-SegPip --version
```

关于「要不要用最新版」：本次 `pip index versions torch` 显示 cu126 索引已有到 **2.14.1**，本环境仍固定在 **2.10.0**——这是指南给定的可核对起点，不是「当时最新」。固定版本的目的是让本阶段的所有结论（显存占用、耗时、能否导入）与后续训练阶段可比；如果中途换版本，就必须重新跑本阶段的全部检查。

## 3. 设计与取舍

### 3.1 数据流

```text
snapshot_environment.ps1 -Label before
        │  audit\before-*（base 包、环境列表、condarc 哈希、系统/GPU）
        ▼
enter_project.ps1  ──►  进程内：路径变量 + 缓存变量 + condarc + TMPDIR + 控制台编码
        │
        ├─ Assert-SegCondaIsolation   ← 闸门：pkgs_dirs 必须只有一项
        ▼
conda create --prefix <env> --copy --no-shortcuts --no-default-packages
             --override-channels --channel repo.anaconda.com/pkgs/main
             --strict-channel-priority python=3.12 pip
        ▼
pip install（torch/torchvision 走 cu126 wheel；其余走 PyPI 镜像）
        ▼
env_check.py --probe full  ──►  reports\environment.json（版本/路径/GPU/前后向）
        ▼
pip freeze / conda env export --from-history / conda list --explicit
        ▼
snapshot_environment.ps1 -Label after  ──►  Compare-Object 比对
```

### 3.2 关键取舍

| 决策点 | 选择 | 理由 | 未选方案 |
| --- | --- | --- | --- |
| 环境形态 | Conda 前缀 `segformer_resources\env` | 绝对路径唯一、可写进脚本、仓库外 | 命名环境（依赖 `envs_dirs` 顺序）；venv（依赖 base 解释器） |
| 环境内是否再套 venv | 不套 | 多一层解释器只会增加「我到底用的是谁」的歧义 | venv 套娃 |
| Conda 装什么 | 只装 `python=3.12` + `pip` | Conda 只负责解释器，训练栈统一由 pip 管，避免两套包管理器互相覆盖 | Conda 装 pytorch（Windows 上 channel 版本滞后） |
| torch 来源 | 官方 cu126 wheel（本地文件安装） | 版本与 CUDA 构建可核对，SHA-256 可验证 | 直接未限定构建来源的 `pip install torch`（无法据此固定本项目的 cu126 构建） |
| 安装方式 | `--copy`、`--no-shortcuts`、`--no-default-packages` | 避免软链接跨盘、避免系统快捷方式、避免旧配置注入默认包 | 默认安装 |
| 环境登记 | `CONDA_REGISTER_ENVS=false` | 不往用户 `environments.txt` 写项目路径 | 默认登记 |
| 缓存归属 | 全部重定向到 `segformer_resources\cache` | 包、模型、编译缓存集中可查、可整体清理 | 用默认 `~/.cache`、`%TEMP%` |

### 3.3 风险与不确定点

| 风险 | 现状 | 后续验证方式 |
| --- | --- | --- |
| 8 GB 显存是否够 512 输入、batch 2 | 未验证训练，只验证了单次前向（峰值 848 MiB allocated） | 阶段 08 smoke 记录 `max_memory_allocated/reserved` |
| 系统可用内存偏低 | 本次实测 available 仅 **1.66 GiB**（总 15.22 GiB） | 训练前关闭无关应用；阶段 08 记录 DataLoader 稳定性 |
| 中文路径与某些依赖的兼容性 | 到本阶段为止未出现路径错误 | 后续每阶段首次运行即检查，出现即记录 |
| 代理链路对下载速度影响大 | PyPI 直连约 50 KB/s，镜像 90 秒装完；pytorch 直下 4—5 MB/s | 后续下载模型时优先记录实测速率 |

## 4. 文件、接口与数据约定

| 文件/模块 | 新增或修改 | 职责 | 关键入口与输入输出 |
| --- | --- | --- | --- |
| `segformer_lab\env_check.py` | 新增 | 采集真实运行时事实并可选做 GPU/模型探针 | `python -m segformer_lab.env_check --out reports\environment.json [--probe none|cuda|full] [--input-size 512]`；输出 JSON + 控制台摘要；失败返回 1 |
| `requirements-lock.txt` | 新增（生成） | `pip freeze` 原始结果，44 行 | 记录实际安装的每一个包 |
| `requirements-cu126.txt` | 新增 | 可移植的 torch 安装配方（含 wheel SHA-256） | `pip install -r requirements-cu126.txt` |
| `environment-history.yml` | 新增（生成） | `conda env export --from-history` | 只含 `python=3.12`、`pip` 与 channels |
| `audit\conda-explicit-win64.txt` | 新增（生成） | `conda list --explicit`，同平台重建用 | 25 行 |
| `audit\after-*` | 新增（生成） | 搭建后快照 | 与 `before-*` 同结构 |
| `audit\negative-control-env-check.json` | 新增（生成） | 故意破坏 TEMP 后的检查结果 | 证明检查器会失败 |

### 4.1 `environment.json` 的结构（后续阶段会读它）

| 顶层键 | 内容 | 为什么后续需要 |
| --- | --- | --- |
| `python` | `executable/version/prefix/base_prefix/site_packages_writable` | 实验卡要写「用的哪个解释器」 |
| `site` | `ENABLE_USER_SITE`、`suspect_sys_path` | 判断是否混入别的 Python 的包 |
| `hardware` / `disk` | CPU 逻辑核、总/可用内存、两个根所在盘剩余空间 | 解释 OOM 或「训练中断」的环境原因 |
| `gpu` | 型号、显存、compute capability、SM 数 | 与显存数字对应 |
| `packages` | 每个关键包的版本与文件路径 | 版本漂移排查 |
| `import_isolation` | 是否有包解析到前缀之外 | 隔离是否真的成立 |
| `effective_paths` | `tempfile/torch.hub/matplotlib/huggingface_hub` **实际**解析路径 + `outside_resource_root` | 变量设了不等于库会读，见 5.3 |
| `cuda_probe` / `segformer_probe` | 前向/反向是否成功、loss、梯度范数、峰值显存 | 「GPU 可用」的证据 |
| `status` / `errors` | `ok` 或 `failed` + 原因列表 | 可被脚本判断，不必人读 |

## 5. 关键实现讲解

### 5.1 「包在不在本项目」用路径判定，不用版本号

`env_check.py`：

```python
def _import_isolation(packages: dict[str, Any]) -> dict[str, Any]:
    prefix = Path(sys.prefix).resolve()
    outside: list[dict[str, str]] = []
    for name, entry in packages.items():
        file = entry.get("file")
        if not file:
            continue
        try:
            Path(file).resolve().relative_to(prefix)
        except ValueError:
            outside.append({"module": name, "file": file})
    return {"sys_prefix": str(prefix), "packages_checked": checked,
            "outside_prefix": outside, "ok": not outside}
```

为什么这样写：版本号相同不代表是同一份安装。本机 3.12.7 与 3.12.0rc3 都是 3.12，但包完全独立。用 `Path.relative_to(sys.prefix)` 判定「这个模块文件是不是在我这个前缀里」，是对「隔离」最直接的定义。`except ValueError` 是 `relative_to` 表示「不在下面」的方式，不是异常处理漏写。

### 5.2 探针必须和「真实训练」区分开

```python
def probe_segformer(input_size: int = 512) -> dict[str, Any]:
    """Forward/backward through HF SegFormer with RANDOM weights. ..."""
    config = SegformerConfig(num_labels=2, id2label={0: "background", 1: "defect"},
                             label2id={"background": 0, "defect": 1},
                             semantic_loss_ignore_index=255)
    model = SegformerForSemanticSegmentation(config).cuda()
    ...
    logits = model(pixel_values=pixel_values).logits
    upsampled = F.interpolate(logits, size=(input_size, input_size),
                              mode="bilinear", align_corners=False)
    loss = F.cross_entropy(upsampled, labels, ignore_index=255)
    loss.backward()
```

要点：

- 用 `SegformerConfig` 构造，**不调用 `from_pretrained`**，因此不触发下载、也不产生「预训练已加载」的错觉。函数 docstring 与 JSON 里的 `note` 都写明权重是随机的。
- 标签里故意放入 `255`（模拟 padding/忽略区），验证 `ignore_index` 真的生效，而不是等数据阶段才发现标签越界。
- 同时检查**编码器**和**解码头**各一个参数的梯度范数与「step 后是否变化」。只看 loss 有限不足以证明反传正确——如果只有解码头在动，loss 也会下降。
- 记录 `peak_allocated/reserved`，为后续 batch 选择留基线。

### 5.3 变量存在 ≠ 库会用它

这是本阶段最重要的一条经验。自检时发现 `TMPDIR` 没有被清除，`tempfile.gettempdir()` 回到了系统临时目录。所以 `env_check.py` 不看 `os.environ` 下结论，而是问库本身：

```python
effective["temp_dir"] = tempfile.gettempdir()
effective["torch_hub_dir"] = torch.hub.get_dir()
effective["matplotlib_cache_dir"] = matplotlib.get_cachedir()
from huggingface_hub import constants as hf_constants
effective["hf_hub_cache"] = hf_constants.HF_HUB_CACHE
```

并且把「是否落在资源根内」做成**硬性失败条件**：

```python
if report["effective_paths"]["outside_resource_root"]:
    keys = ", ".join(item["key"] for item in report["effective_paths"]["outside_resource_root"])
    errors.append(f"cache/temp paths outside the resource root: {keys}")
```

为了证明这个判断不是「永远返回 ok」，做了一次负向对照（见 6.7）：把 `TEMP/TMP/TMPDIR` 指回系统目录再跑，结果 `status=failed`、退出码 1。

### 5.4 安装流程为什么要「先验哈希再安装」

第一阶段直接 `pip install torch==2.10.0` 时，pip 在下载 2.59 GB 的 wheel 上卡了 11 分钟且磁盘无任何写入（详见 8.2）。改为：

1. 从官方 simple 索引取该 wheel 的 `#sha256=`；
2. `curl -C -` 断点续传下载到 `cache\wheels\`；
3. `sha256sum` 比对通过；
4. `pip install <本地 wheel>`。

好处：下载可续传、可测速、可校验；坏包在进入 `site-packages` 之前就被拦住。代价是多了一步手工操作，因此把可复现配方写进 `requirements-cu126.txt`。

## 6. 如何操作与实际执行了什么

以下命令均在项目上下文中执行（先 `$env:PSExecutionPolicyPreference='Bypass'` 再 dot-source 入口）。时间均为 2026-10-31 本地时间。

### 6.1 创建环境

```powershell
Assert-SegCondaIsolation
& $SegConda create --prefix $SegEnv --copy --no-shortcuts --no-default-packages `
    --override-channels --channel 'https://repo.anaconda.com/pkgs/main' `
    --strict-channel-priority 'python=3.12' pip -y
```

| 命令/检查 | 实际结果 | 证据路径 | 是否通过 |
| --- | --- | --- | --- |
| 第 1 次 `conda create` | 失败：`Collecting package metadata (repodata.json): ... failed`（元数据下载未完成，未产生环境目录） | 控制台输出（未落盘） | 失败（网络） |
| 第 2 次 `conda create` | 成功，退出码 0；提示 `conda activate E:\桌面\AI\segformer_resources\env` | `audit\conda-create.log`（该文件为第 2 次即成功那次的完整日志） | 通过 |
| `sys.executable` | `E:\桌面\AI\segformer_resources\env\python.exe` | 控制台输出 | 通过 |
| `sys.version` | `3.12.16 \| packaged by Anaconda, Inc. \| (main, Oct 26 2026, 20:13:05) [MSC v.1942 64 bit (AMD64)]` | `reports\environment.json` | 通过 |
| `site.ENABLE_USER_SITE` | `False` | `environment.json` `site` | 通过 |
| `pip --version` | `pip 26.2.2 from ...\env\Lib\site-packages\pip` | 控制台输出 | 通过 |
| `pip config list` | 只有 `:env:.cache-dir` 与 `:env:.config-file='nul'`（全局 `pip.ini` 的清华源被成功屏蔽） | 控制台输出 | 通过 |

### 6.2 安装 PyTorch 栈

```powershell
# 记录：pip index versions torch --index-url https://download.pytorch.org/whl/cu126
# 可用版本含 2.10.0+cu126（最新为 2.14.1+cu126，本次不用）
Invoke-SegPip install --no-user torch==2.10.0 torchvision==0.25.0 `
    --index-url https://download.pytorch.org/whl/cu126      # 首次尝试，卡住，见 8.2

# 最终采用的方式
curl -L -C - --proxy http://127.0.0.1:52108 -o cache\wheels\torch-2.10.0+cu126-cp312-cp312-win_amd64.whl <官方 URL>
sha256sum cache\wheels\torch-2.10.0+cu126-cp312-cp312-win_amd64.whl
Invoke-SegPip install --no-user --no-cache-dir `
    'E:\桌面\AI\segformer_resources\cache\wheels\torch-2.10.0+cu126-cp312-cp312-win_amd64.whl' `
    'E:\桌面\AI\segformer_resources\cache\wheels\torchvision-0.25.0+cu126-cp312-cp312-win_amd64.whl' `
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

wheel 身份（与官方索引 `#sha256=` 一致）：

```text
torch-2.10.0+cu126-cp312-cp312-win_amd64.whl
  size   2 589 881 452 bytes
  sha256 95d8409b8a15191de4c2958e86ca47f3ea8f9739b994ee4ca0e7586f37336413
torchvision-0.25.0+cu126-cp312-cp312-win_amd64.whl
  size   8 792 806 bytes
  sha256 57a8a103814c344d91c32425ba38a53daac4bf4aab074aaaea7b6bab4c22fb7b
```

| 命令/检查 | 实际结果 | 证据路径 | 是否通过 |
| --- | --- | --- | --- |
| 第 1 次 torch 安装 | 卡在下载（11 分 07 秒无进展，人工终止） | `audit\pip-torch-install.log`（后续被覆盖） | 失败（网络） |
| wheel 下载 | 完成，平均约 3.7 MB/s，用时 10 分 38 秒 | `cache\wheels\` | 通过 |
| SHA-256 校验 | torch / torchvision 均与官方一致 | `sha256sum` 输出 | 通过 |
| 第 1 次本地安装 | 失败：`[WinError 5] 拒绝访问 ... RECORD`（收尾阶段改名被系统占用） | 控制台输出 | 失败（文件锁） |
| 清理残留 | 删除 `site-packages\torch`、`torch-2.10.0+cu126.dist-info` 等未完成目录 | — | 通过 |
| 第 2 次本地安装 | 成功，`Successfully installed torch-2.10.0+cu126 torchvision-0.25.0+cu126`，用时 122.8 秒 | `audit\pip-torch-install.log` | 通过 |

### 6.3 安装其余依赖

```powershell
Invoke-SegPip install --no-user 'transformers==4.57.6' 'huggingface-hub>=0.34,<1' `
    'numpy>=1.26,<3' 'Pillow>=10,<13' 'matplotlib>=3.9,<4' 'PyYAML>=6,<7' `
    'tqdm>=4.66,<5' 'tensorboard>=2.18,<3' `
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60 --retries 3
```

| 命令/检查 | 实际结果 | 证据路径 | 是否通过 |
| --- | --- | --- | --- |
| 首次（PyPI 官方源） | 未完成：单包速度约 46—60 KB/s，12 MB 的 transformers 需 3 分 31 秒，进程中途终止 | `audit\pip-deps-install.log` | 失败（带宽） |
| 改用清华镜像 | 成功，用时 **90.0 秒**，34 个包 | `audit\pip-deps-install.log` | 通过 |
| `pip check` | `No broken requirements found.` | 控制台输出 | 通过 |

### 6.4 导入与 CUDA 前后向

```powershell
Invoke-SegPython -c "import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__); print('CUDA build:', torch.version.cuda); print('available:', torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(256,256,device='cuda',requires_grad=True); loss=(x@x.T).square().mean(); loss.backward(); torch.cuda.synchronize(); print('loss:', loss.item(), 'grad finite:', bool(torch.isfinite(x.grad).all()))"
```

实际输出：

```text
2.10.0+cu126 0.25.0+cu126 4.57.6
CUDA build: 12.6
available: True
NVIDIA GeForce RTX 4070 Laptop GPU
loss: 515.730224609375 grad finite: True
```

### 6.5 生成环境报告

```powershell
Invoke-SegPython -m segformer_lab.env_check --out 'reports\environment.json' --probe cuda
Invoke-SegPython -m segformer_lab.env_check --out 'reports\environment.json' --probe full
```

`--probe full` 的实际输出（`environment.json` 已被最终版覆盖）：

```text
SegFormer environment check
  python      : E:\桌面\AI\segformer_resources\env\python.exe
  version     : 3.12.16 | packaged by Anaconda, Inc. | (main, Oct 26 2026, 20:13:05) [MSC v.1942 64 bit (AMD64)]
  prefix      : E:\桌面\AI\segformer_resources\env
  user site   : False
  resource    : E:\桌面\AI\segformer_resources
  torch        : 2.10.0+cu126
  torchvision  : 0.25.0+cu126
  transformers : 4.57.6
  numpy        : 2.5.4
  PIL          : 12.3.0
  matplotlib   : 3.11.2
  yaml         : 6.0.3
  cuda build  : 12.6
  cuda avail  : True
  gpu         : NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB, cc 8.9)
  cuda probe  : ok=True loss=505.4765930175781 grad_finite=True
  segformer   : ok=True logits=[2, 2, 128, 128] params=3714658/3714658
                encoder_grad=0.145012 head_grad=0.015297 peak=848.1 MiB
  isolation   : outside_prefix=0 suspect_sys_path=0
  temp dir    : E:\桌面\AI\segformer_resources\tmp
  hf hub      : E:\桌面\AI\segformer_resources\cache\huggingface\hub
  torch hub   : E:\桌面\AI\segformer_resources\cache\torch\hub
  cache leak  : 0
  status      : ok
```

关键读数解释：

- `logits=[2, 2, 128, 128]`：512 输入在 SegFormer-B0 下输出 1/4 分辨率（`output_stride=4`）。后续训练必须把这个 logits 双线性插值回标签尺寸，否则 CE 形状不匹配。
- `params=3714658/3714658`：**全部参数可训练**。这是二类头的 B0，不是 1000 类分类模型；如果总参数明显偏大，说明加载了错误的结构。
- `encoder_grad=0.145012`、`head_grad=0.015297`：编码器与解码头都有非零梯度，说明没有误冻结编码器。
- 两个探针都在构造模型/张量前调用 `torch.manual_seed(42)`，因此重复运行数值可比（实测两次：`loss_ce` 完全相同，`encoder_grad` 末位 0.145012 / 0.145013 的差异来自 cuDNN 非确定性算法）。这也是「证据」与「随手截屏」的区别：数字必须能被别人复现。
- `peak=848.1 MiB`（reserved 1014.0 MiB）：这是 2×3×512×512 单次前向+反向的峰值，不含优化器状态与数据加载，**不能**当作训练显存预算。8 GB 卡上 batch 2 是否可行要等阶段 08 实测。
- `cuda probe ... elapsed_ms=219.93`、`segformer ... elapsed_ms=2204.0`：包含首次 CUDA 上下文初始化与 cuDNN 算法选择，不是稳态速度。
- `loss_ce=0.782416820526123`：随机权重、二类、含 255 忽略区的交叉熵。这个数字没有精度含义，只用于确认损失是可计算的有限值。

### 6.6 保存锁定文件

实际执行的写法（用 .NET 写 UTF-8 **无 BOM**；`Set-Content -Encoding utf8` 在 PowerShell 5.1 会写 BOM，`pip install -r` 读到时前面会多一个不可见字符）：

```powershell
$utf8 = New-Object System.Text.UTF8Encoding($false)
$freeze = & $SegConda run --no-capture-output --prefix $SegEnv python -m pip freeze 2>&1 |
          Where-Object { $_ -and $_.Trim() -ne '' }
[IO.File]::WriteAllLines((Join-Path $SegProject 'requirements-lock.txt'), [string[]]$freeze, $utf8)

$hist = & $SegConda env export --prefix $SegEnv --from-history 2>&1
[IO.File]::WriteAllLines((Join-Path $SegProject 'environment-history.yml'), [string[]]$hist, $utf8)

$explicit = & $SegConda list --prefix $SegEnv --explicit 2>&1
[IO.File]::WriteAllLines((Join-Path $SegAudit 'conda-explicit-win64.txt'), [string[]]$explicit, $utf8)

Invoke-SegPython -m segformer_lab.env_check --out 'reports\environment.json' --probe full
```

`Where-Object { $_ -ne '' }` 是必要的：`conda run` 会在输出里夹带空行，不过滤会污染锁定文件。

| 文件 | 实际结果 | 说明 |
| --- | --- | --- |
| `requirements-lock.txt` | 44 行 | 原始 freeze；torch/torchvision 因本地安装被记为 `file:///E:/%E6%A1%8C%E9%9D%A2/...whl#sha256=...` |
| `requirements-cu126.txt` | 手工新增 | 可移植配方，解决上面 `file://` 无法在别的机器复现的问题 |
| `environment-history.yml` | 14 行 | 只有 `python=3.12`、`pip`；**请注意它同时列出了本机用户 `.condarc` 合并进来的 7 个 channel**，这是 Conda 的行为，不代表本项目使用了它们 |
| `audit\conda-explicit-win64.txt` | 25 行 | 同平台重建用 |

### 6.7 隔离核查（本阶段的核心验收）

**（a）搭建后快照**

```powershell
& 'E:\桌面\AI\AI_Try\work\SegFormer\scripts\snapshot_environment.ps1' -Label after
# registered envs: E:\Anaconda   base packages: 510
```

**（b）base 包清单比对**

```powershell
Compare-Object (Get-Content ...\before-base-packages.txt) (Get-Content ...\after-base-packages.txt)
# 无输出 → IDENTICAL (name|version|build|channel, 510 entries)
```

**（c）全局配置哈希比对**

| 文件 | before | after | 结论 |
| --- | --- | --- | --- |
| `E:\Anaconda\.condarc` | `532994F9...70EAC` | `532994F9...70EAC` | 未变 |
| `C:\Users\Luorenwei\.condarc` | `0BD6F353...10AD0` | `0BD6F353...10AD0` | 未变 |
| `%APPDATA%\pip\pip.ini` | `582B0ADB...1DD87` | `582B0ADB...1DD87` | 未变 |
| `...\segformer_resources\config\condarc.yml` | 不存在 | `5035ACC0...E0222` | 新增（本项目自己的配置） |

**（d）base 目录是否被写入**

```text
E:\Anaconda\pkgs 中 11:55 之后被修改的条目数            = 0
E:\Anaconda\Lib\site-packages 中 11:55 之后被修改的条目数 = 0
```

**（e）环境登记表**

```text
%USERPROFILE%\.conda\environments.txt 内容：E:\Anaconda
```

项目环境**没有**被写入。这解释了为什么 `conda env list` 看不到它：

```powershell
& $SegConda list --prefix 'E:\桌面\AI\segformer_resources\env'   # 用 prefix 查询
```

**（f）负向对照——证明检查器会失败**

```powershell
# 故意把临时目录指回系统目录，绕过项目上下文
$env:TMPDIR = 'C:\Users\Luorenwei\AppData\Local\Temp'
& $SegPython -m segformer_lab.env_check --out audit\negative-control-env-check.json --probe none
```

实际输出：

```text
temp dir    : C:\Users\Luorenwei\AppData\Local\Temp
  cache leak  : 1 -> ['temp_dir']
  status      : failed
  error       : cache/temp paths outside the resource root: temp_dir
exit code: 1
```

说明：`status=ok` 不是「这个检查永远返回 ok」，而是「这次真的都落在项目目录里」。

### 6.8 资源占用实测

| 目录 | 大小 | 说明 |
| --- | ---: | --- |
| `env\` | 4 373 MB | 解释器 + 全部包 |
| `cache\wheels\` | 2 479 MB | torch/torchvision wheel，可删（重装时需重新下载） |
| `cache\conda_pkgs\` | 326 MB | Python 3.12 与 pip 的 conda 包 |
| `cache\pycache\` | 238 MB | 含 base 标准库镜像（见阶段 01 的 8.2） |
| `cache\pip\` | 125 MB | pip 的 http 缓存 |
| `cache\huggingface\` | 0 MB | **尚未下载任何模型**（符合本阶段边界） |
| `tmp\` / `runs\` / `audit\` | 1 / 0 / 1 MB | — |
| 合计 | 约 7.4 GB | E 盘可用空间由 163.52 GiB 降至 **152.24 GiB**（差额大于 7.4 GB，主要是安装过程中的临时解包与系统回收延迟） |

## 7. 验收与当前状态

| 验收项（主指南 4.4 与隔离规范 6 节） | 结果 | 证据 |
| --- | --- | --- |
| `pip check` 无依赖冲突 | 通过 | `No broken requirements found.` |
| GPU 运算完成 | 通过 | `cuda_probe.ok=True`，loss 有限，梯度有限 |
| 环境 JSON 存在 | 通过 | `reports\environment.json`（6.0 KB） |
| 锁定文件存在 | 通过 | `requirements-lock.txt`、`environment-history.yml`、`conda-explicit-win64.txt` |
| 包全部位于项目前缀 | 通过 | `import_isolation.outside_prefix = 0`，`suspect_sys_path = 0` |
| 缓存/临时路径位于资源根 | 通过 | `effective_paths.outside_resource_root = []`；负向对照会失败 |
| base 包清单前后一致 | 通过 | `Compare-Object` 无输出（510 条） |
| 全局 condarc / pip 配置未变 | 通过 | 三个文件哈希一致 |
| 项目环境未登记到用户表 | 通过 | `environments.txt` 只有 `E:\Anaconda` |
| 解释器路径在资源根内 | 通过 | `sys.executable`、`sys.prefix` 均在 `...\segformer_resources\env` |
| 用户 site 已禁用 | 通过 | `ENABLE_USER_SITE = False` |
| SegFormer 前后向完成 | 通过 | `logits=[2,2,128,128]`，编码器/头梯度非零且参数已更新 |
| 文档与索引更新 | 通过 | 本文件 + `docs/stages/README.md` |

**明确未做的**（防止误读）：

- 未下载 `nvidia/mit-b0` 预训练权重，`cache\huggingface\hub` 为 0 MB；因此**不存在**「编码器已加载预训练权重」的结论。
- 未验证 512×512、batch 2 的训练稳定性与显存上限；`peak=848.1 MiB` 只是单次前反向。
- 未安装 `torchaudio`、`datasets`、`accelerate`、MMCV、自定义 CUDA 算子（第一轮不需要）。
- 未做位级可复现性验证（cuDNN benchmark/deterministic 属训练阶段）。

## 8. 遇到的问题、修复与未解决项

### 8.1 第一次 `conda create` 元数据下载失败

现象：

```text
Collecting package metadata (repodata.json): ... failed
```

没有生成环境目录，退出码非 0。**未捕获到完整报错文本**（输出被 PowerShell 的 `NativeCommandError` 包装后截断），这是本次流程的一个取证缺陷。

处置：立即原样重试，成功。因此无法确定根因是代理抖动、TLS 握手还是 CDN 节点问题；只能说「首次尝试在元数据阶段失败，重试即成功，属瞬时网络问题」。第 2 次起把输出同时 `Tee-Object` 到 `audit\`，后续失败都有完整日志。

**教训**：安装类命令第一次就要落盘日志，否则失败后只能靠回忆。

### 8.2 pip 直接装 torch 卡死 11 分钟

现象：`pip install torch==2.10.0 ... --index-url https://download.pytorch.org/whl/cu126` 输出 `Downloading torch-2.10.0+cu126-...whl (2589.9 MB)` 后再无进展；11 分 07 秒后检查：`cache\pip` 停留在 5.1 MB，`tmp\` 为空，python 进程 CPU 5.7 秒、磁盘读 0 B/s——**在等一个不返回数据的 socket**。

定位证据：同一 URL 用 `curl -r 0-30000000` 走同一代理，得到 `http=206`、3.8 MB/s。说明网络路径本身可用，问题在 pip 的下载路径（代理下的长连接行为）。

处置：

1. 终止任务与残留进程；
2. 用 `curl -C -` 断点续传下载 wheel（实测 3.67 MB/s，10 分 38 秒完成）；
3. 校验 SHA-256；
4. `pip install <本地 wheel>`。

结果：成功。这个方案同时带来了「可校验」与「可续传」两个好处，已固化到 `requirements-cu126.txt`。

### 8.3 安装收尾报 `WinError 5 拒绝访问`（RECORD 改名失败）

现象：

```text
ERROR: Could not install packages due to an OSError: [WinError 5] 拒绝访问。
'E:\...\site-packages\torch-2.10.0+cu126.dist-info\RECORDo7an6hg3.tmp' -> '...\RECORD'
```

此时 `site-packages\torch` 已有 3.9 GB，`dist-info` 里同时存在 `RECORD`（11 756 行）与 `RECORDo7an6hg3.tmp`（13 960 行）——文件已复制完，只有最后一步改名被占用。

原因判断：pip 在收尾时对 `RECORD` 做临时文件+改名，Windows 上若有安全软件/索引服务正在扫描这个刚写入的 4 GB 目录，句柄未释放就会拒绝访问。未做进一步取证（无法在本机复现第二次）。

处置：

1. 删除未完成的 `site-packages\torch`、`torch-2.10.0+cu126.dist-info`、`functorch`、`torchgen`（都是本次安装的产物，不是用户数据）；
2. 带 `--no-cache-dir` 重新安装两个本地 wheel；
3. 122.8 秒后成功，`pip check` 通过，`RECORD` 完整。

注意：`RECORD` 是 pip 卸载/校验时的索引，若保留一个写坏的 `RECORD`，将来 `pip uninstall torch` 会漏删文件。因此这里选择「整目录重装」而不是「删掉 .tmp 了事」。

### 8.4 PyPI 直连过慢

现象：`--index-url https://pypi.org/simple` 下，transformers 12 MB 用了 3 分 31 秒（约 59.6 KB/s），numpy 12.6 MB 约 45.9 KB/s。

处置：改用清华镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`，同样 34 个包 **90 秒**装完。这是「显式指定索引」而不是修改全局 `pip.ini`——全局配置的哈希在前后比对中保持不变。

副作用：`pip freeze` 不记录来源，所以**镜像这件事只能靠文档记录**。这也是为什么 `requirements-cu126.txt` 顶部写了注释说明 wheel 来源与哈希。

### 8.5 `conda run` 依赖可写的 TEMP

做负向对照时把 `TMPDIR` 指到 `C:\Windows\Temp`，`conda run` 直接失败：

```text
Could Not Find C:\Windows\Temp\tmpcqc61cl6.bat
```

说明 `conda run` 会把自己的启动批处理写到 TEMP 再执行，若该目录不可写/不可访问，包装命令整体失效。这不影响正常运行（项目 `tmp\` 可写），但解释了一个坑：**以后若看到 `conda run` 报找不到 `.bat`，先查 TEMP 是否有效**。

### 8.6 未解决 / 待观察

- `cache\pycache\` 里有 base 标准库的字节码镜像（238 MB，含 transformers 编译产物）。保留，属设计副作用；若后续磁盘紧张可整体删除该目录，运行时会重新生成。
- `environment-history.yml` 把用户 `.condarc` 的 7 个 channel 一并导出，容易误导「本项目用了这些 channel」。已在 6.6 表格中标注；重建时以 `requirements-runtime.txt`（完整依赖）+ `condarc.yml` 和明确的 Conda 创建命令为准；`requirements-cu126.txt` 仅是两个 CUDA 包的辅助配方。
- 训练显存尚未验证。8 GB 卡 + 15.22 GiB 内存（可用仅 1.66 GiB）是后续阶段的主要风险，阶段 08 必须记录真实峰值。

## 9. 给用户的学习要点

### 9.1 四个概念

**（1）Conda 环境与「装了什么」是两件事。** Conda 只创建了 `python=3.12` 与 `pip`；后面 44 个包全是 pip 装的。所以 `conda env export --from-history` 只有两行依赖，**不能**当作完整重建依据——当前机取证依据是原始 `requirements-lock.txt` + CUDA wheel 来源；跨端完整依赖使用本轮补充的 `requirements-runtime.txt`，不能把仅含两个包的 `requirements-cu126.txt` 当作完整环境。

**（2）`+cu126` 是携带构建信息的本地版本标识。** `2.10.0` 是发布版本，`2.10.0+cu126` 进一步标明本项目选择的 CUDA 12.6 构建；安装源也要记录。不能只看一个不带构建来源的版本约束就断定安装的是 CPU 或某个 CUDA 版本，最终应检查 `torch.version.cuda`、设备可用性与真实 GPU 运算。

**（3）`torch.version.cuda` 与 `nvidia-smi` 的 CUDA 版本不是一回事。** 前者是 wheel 里打包的运行时版本（12.6），后者是驱动支持的最高版本（本次 `nvidia-smi` 报告 CUDA UMD 能力为 13.4）。用预编译 wheel 时不需要另装同版本 CUDA Toolkit；本机虽有 `CUDA_PATH=v12.6`，但本次完全没用到它。

**（4）`ENABLE_USER_SITE=False` 是隔离的一道锁。** 它保证 `pip install --user` 装的东西不会在这个解释器里被看到；配合 `--no-user`，双向都堵住。

### 9.2 三个自测问题

1. **`environment.json` 里 `logits=[2,2,128,128]`，为什么不是 `[2,2,512,512]`？这对后面的 loss 有什么影响？**
   线索：6.5 的读数解释、5.2 的 `F.interpolate`。答案：MiT 编码器是 1/4 下采样的层级结构，解码头输出保持 1/4 分辨率（stride 4）。CE 要求 logits 与标签同尺寸，所以必须先把 logits 双线性上采样到 512×512 再算损失；反过来把标签下采样到 128×128 会丢掉细缺陷，且与「原图尺度评价」的口径不一致。

2. **为什么 `segformer_probe` 要用随机权重，而不是直接下载 `nvidia/mit-b0`？**
   线索：5.2 的 docstring 与 `note` 字段。答案：本阶段只回答「这条代码路径在本机 GPU 上能不能跑」，随机权重足以验证结构、dtype、stride、梯度。下载权重属于阶段 05，还需要固定 revision 与记录 SHA；提前下载会让人误以为「预训练已就绪」。

3. **`cache leak: 0` 与负向对照的 `cache leak: 1`，两者差在哪一步？**
   线索：5.3 与 6.7(f)。答案：差别只在 `TEMP/TMP/TMPDIR` 是否指向项目 `tmp\`。前者三个变量都指向项目目录，四个库的实际解析路径都在资源根内；后者把 `TMPDIR` 改回系统目录，`tempfile.gettempdir()` 立刻逃逸，检查器把 `status` 置为 `failed` 并返回退出码 1。这说明检查是「真的在测」，不是打印常量。

## 10. 下一阶段与修订记录

阶段 03（数据准备）会用到：

- 解释器与环境：一律通过 `Invoke-SegPython` / `run_project.ps1`，不得直接双击 `env\python.exe`；
- `paths.py` 的 `DATA_DIR`（`segformer_resources\data`）与 `MANIFESTS_DIR`（仓库内 `manifests`）；
- 本文件的版本基线：若 `torch`/`transformers` 版本变化，`environment.json` 必须重新生成，且实验卡要更新。

必须保持不变的约定：`num_labels=2`、`id2label={0:background,1:defect}`、`ignore_index=255`、输入 512 下输出 stride 4。这三条在本阶段已由探针实测确认，后续改动必须作为新实验记录。

| 日期 | 修改原因 | 改动与影响 | 修正后重新验证的证据 |
| --- | --- | --- | --- |
| 2026-10-31 | 首版建立 | 创建环境、安装 torch 2.10.0+cu126 栈、实现 `env_check.py`、生成锁定文件与前后快照 | 6.1—6.8 的命令输出 |
| 2026-10-31 | 自检发现 `tempfile` 逃逸到系统临时目录 | `enter_project.ps1` 增加 `TMPDIR`；`env_check.py` 增加 `effective_paths` 硬性校验 | `temp dir = ...\segformer_resources\tmp`；负向对照 `status=failed` |
| 2026-10-31 | `pip` 直连下载卡死 | 改为 curl 续传 + SHA-256 校验 + 本地 wheel 安装 | 哈希一致，安装 122.8 秒成功 |
| 2026-10-31 | PyPI 带宽过低 | 显式切换到清华镜像（不改全局配置） | 90 秒装完 34 个包；`pip.ini` 哈希未变 |
| 2026-10-31 | 控制台中文乱码 | 入口脚本切换控制台输出编码 | 6.5 输出正常 |
| 2026-10-31 | 探针每次输出不同数值，无法作为可复现证据 | `probe_cuda` / `probe_segformer` 在构造张量/模型前 `torch.manual_seed(42)` | 连续两次运行 `loss_ce` 相同、`encoder_grad` 仅末位差 1 |

### 验收修订记录补充

| 日期 | 修改原因 | 改动与影响 | 修正后验证 |
| --- | --- | --- | --- |
| 2026-10-31 | 检查器可能漏报必要导入失败、错误prefix、用户site开启与缓存读取失败 | 增加明确失败条件，探针异常保留失败报告；必需模块扩为9项 | 5项回归由4项失败变为全部通过；真实full报告为ok且缓存越界0 |
| 2026-10-31 | 原始freeze包含本机wheel路径，CUDA双包配方不是完整环境 | 新增完整44项requirements-runtime.txt，保留原始lock取证，修正构建标签解释 | 与当前环境44项版本逐一匹配；pip check无冲突 |

当前复查：base的510项包及两份原Conda配置、用户pip.ini哈希均与搭建前一致。数据与模型目录仍无实际文件，未执行真实数据训练。
