# 阶段 01—02 成果验收与完善记录

日期：2026-10-31。结论：**经修复与独立复核，步骤 1—2 通过验收，可以进入步骤 3 的 PCB1 数据准备。**本次没有升级/重装训练依赖，没有删除 wheel 缓存，也没有开始导入数据或训练。

## 1. 实际核验结果

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 项目解释器 | 独立 prefix 内 Python 3.12.16，用户 site 禁用 | `reports/environment.json` |
| 固定训练栈 | torch 2.10.0+cu126、torchvision 0.25.0+cu126、transformers 4.57.6 | 同上与依赖清单 |
| 依赖一致性 | `pip check` 无冲突；可移植清单的 44 项版本均匹配实际环境 | 本次实际命令；`requirements-runtime.txt` |
| CUDA | 实际矩阵前后向成功，梯度有限 | `cuda_probe.ok=true` |
| SegFormer 环境探针 | 随机权重 B0，输入 `[2,3,512,512]`，logits `[2,2,128,128]`，编码器和头均有梯度并更新 | `segformer_probe.ok=true` |
| 探针显存 | 峰值 allocated 848.1 MiB | 同上；这是人工张量探针，不是正式训练峰值 |
| 包与缓存隔离 | 必需库从项目 prefix 导入；缓存/临时目录越界数为 0 | `import_isolation`、`effective_paths` |
| Conda 配置 | 包缓存仅项目目录，register_envs=False，notices=0 | 实际 `Assert-SegCondaIsolation` 输出 |
| base 包 | 当前 510 项包名/版本/构建/channel 与搭建前一致 | `segformer_resources/audit/acceptance-state.json` |
| 原有配置 | 两份原 `.condarc` 与原用户 `pip.ini` 的 SHA-256 均与搭建前一致 | 同上；仅做只读哈希，没有输出配置内容 |
| 环境检查器回归 | 5 项通过，包含 4 种不能误报成功的负向情形 | `tests/test_env_check.py` |
| 启动器回归 | PowerShell 7 与 Windows PowerShell 5.1 各 20 项通过 | `scripts/test_runtime_context.ps1`；对应 audit 日志 |

此次环境报告采样时可用物理内存为 3.13 GiB，总量为 15.22 GiB。截图中的 1.66 GiB 是先前时点读数。可用内存随应用变化，下一阶段采用流式解压/逐图读取；实际训练阶段仍按指南记录峰值，不能用随机权重探针证明训练效果。

## 2. 验收中发现并修正的问题

### 2.1 PowerShell 7 清变量后仍留下空字符串

首次独立复跑出现三条 Transformers 旧缓存警告，最终 `cache leak=1`、`status=failed`。证据保留在 `segformer_resources/audit/acceptance-environment-before-fixes.json`。

根因：本机 PowerShell 7/.NET 下，`SetEnvironmentVariable(name,$null,'Process')` 没有可靠地让键消失，而留下空值键。Transformers 4.57.6 检查变量是否存在，空值会覆盖正确的 HF 缓存设置。

修复：入口对需要清除的单个变量使用 `Remove-Item -LiteralPath Env:<name>` 真正移除；不枚举有大小写重名问题的整个 Env provider。修复后新进程导入不再出现这三条警告，库实际缓存路径在资源根内。

### 2.2 自定义资源根只传到 PowerShell，没有传给 Python

入口现在导出 `SEGFORMER_RESOURCE=$SegResource`，与 `paths.py` 的覆盖入口保持一致。包装器还修正了空参数传递、非零退出码、入口提前失败时的原始错误保留，以及调用者环境、工作目录和控制台编码恢复；两种 PowerShell 均有实际回归证据。

### 2.3 检查器可能将不完整环境判为通过

原检查器没有把必要库导入失败、用户 site 开启、解释器 prefix 错误、缓存读取失败全部纳入失败条件。验收先运行负向回归，5 项中 4 项失败，确认会误报；修复后 5 项全部通过。

`env_check.py` 现检查实际 prefix/executable、用户 site、9 个必需模块及其路径、缓存读取错误与 CUDA 状态。探针异常会保留为失败报告，不掩盖为成功。报告中的“通过”仍只覆盖这些明确检查，不表示完成了数据、预训练或训练验证。

### 2.4 跨端依赖重建说明不完整

- `requirements-lock.txt` 保留原始 `pip freeze` 和本机 wheel URL/SHA，作为安装取证记录。
- 新增 `requirements-runtime.txt`：完整 44 项版本约束，去掉仅本机可用的 wheel 文件路径，保留 `+cu126` 与官方 CUDA 索引；已与本机已装版本逐项核对。
- `requirements-cu126.txt` 只覆盖两个 CUDA 包，不再被说明为完整环境重建方案。
- `environment-history.yml` 保留实际 Conda 导出的原件；其中合并的旧 channel 不代表本次创建实际使用了这些源。重建仍使用指南中显式 `--override-channels` 的创建命令。
- 修正文档中“无后缀 torch 必然为 CPU”“构建标识不是版本信息”的过度简化；CUDA 构建须由来源、具体版本及实际运算共同核验。

## 3. 以后可复用的检查命令

在专用 PowerShell 中加载项目上下文后：

```powershell
. '.\scripts\enter_project.ps1'
Assert-SegCondaIsolation
Invoke-SegPip check
Invoke-SegPython -m unittest discover -v
Invoke-SegPython -m segformer_lab.env_check --out 'reports\environment.json' --probe full
& '.\scripts\test_runtime_context.ps1'
```

入口回归包含故意让 Python 返回 17 的负向检查，Conda 会打印相应 ERROR；以回归脚本最终全部检查通过及退出码 0 判断，不把预期的负向日志当成环境损坏。

跨端安装完整 Python 依赖的命令（**本次没有执行重装**）：

```powershell
Invoke-SegPip install --no-user -r 'requirements-runtime.txt' --index-url 'https://pypi.org/simple'
```

该配方针对 Windows x64、Python 3.12，需先建立正确的项目 Conda 环境与运行上下文；没有在第二台电脑实际重建，不能称为跨端已验证。

## 4. 下一步指令：只执行步骤 3

可将以下内容交给同一工作区中的 Agent：

```text
步骤1—2已完成验收。请先读阶段索引、01/02阶段文档和本验收记录，加载项目启动入口。
现在只执行主指南步骤3：VisA PCB1数据准备。

1. 复用现有Conda环境，不重装/升级包；每个新工具进程使用run_project.ps1或等效项目上下文。
2. 从现有docs/dataset/VisA/VisA_20220922.tar和spot-diff-main.zip读取数据与官方CSV。
3. 只安全提取pcb1至segformer_resources/data/visa，保留pcb1/Data/...相对路径，禁止归档越界与意外覆盖。
4. 使用官方2cls_highshot.csv，原test保持402正常+40异常。
5. 只从官方train按指南算法、种子42、固定配额120正常+12异常划出val。
6. 输出train.csv=530（482正常+48异常）、val.csv=132（120正常+12异常）、test.csv=442。
7. 保存官方CSV、split_meta.json、原图路径交集检查、异常mask路径检查和清单SHA-256。
8. 用逐条/流式方式处理，不将整个归档或全部图像解码后堆在内存里。
9. 只制定并检查后续标签约定：正常全0、异常非零ID合并为1；不修改原始TAR或原始mask。
10. 按模板完成docs/stages/03-data-preparation-design-and-implementation.md，更新阶段索引与项目README。

返回数据统计、可复现命令、审计结果和阶段文档链接。本次不进入阶段4可视化、模型权重下载或训练。
```

步骤3完成后再查收清单和数据审计，然后进入步骤4的图像/掩码叠加检查。当前 data/models 下均未发现实际数据/权重文件，阶段03—13仍保持未开始。
