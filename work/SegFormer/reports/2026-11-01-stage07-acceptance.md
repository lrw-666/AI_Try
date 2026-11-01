# 第七阶段验收与完善

验收日期：2026-11-01。范围是损失、原图恢复与指标正确性。本轮没有进行优化器更新，也没有产生训练后的基线成绩。最终验证状态见本文第 3 节及 [机器证据](./stage07-acceptance.json)。

## 1. 对五项检查的决定

| 用户检查 | 结论与执行约定 |
| --- | --- |
| 损失四项取舍 | 接受并固定：logits 先双线性上采样到 512 标签网格；Dice 仅对有效 GT 含正类的图片逐图计算后平均；全正常 batch 返回计算图相连的零 Dice，CE 继续监督正常图；全忽略 batch 显式抛错。配方为 `1.0×CE + 0.5×Dice`、`eps=1e-6`，改变时另存配置和实验身份 |
| NA 语义 | 接受：正常且预测空的逐图 IoU/Dice 为 `null`（展示为 NA）；precision 的预测正类分母为零也为 NA。保留原因，不以 1 或平滑项替代。正常图仍进入全局混淆量和正常图误报统计；异常图宏平均单列 |
| 恢复顺序 | 固定 `128 logits → 512 canvas → 按该图 geometry 裁 padding → 原尺寸 logits → softmax → >=0.5`。GT 使用原尺寸掩码，不能将 512 标签放大充当 GT |
| 旧“12 px”差异 | **该统计无效，撤回。**原实现用了净混淆量之差，既会抵消也会重复计数。现改成配对预测逐像素 XOR，并同时比较完整概率差、阈值距离、输入/GT/模型哈希和 B1 重复前向 |
| probe 能否引用 | 不能。始终保留 `not_a_baseline: true`。当前头是随机初始化的；真实图片 probe 只验证计算通路。正式评价必须在新进程加载训练后 checkpoint 的编码器与解码头，不能只修改标志或重新加载预训练编码器 |

“36 px 偏移”不是通用误差常数。本阶段特定合成夹具在漏裁 padding 时纵向质心偏差 36.141 px；正确路径为 0.366 px，包含插值影响。阶段 04 的小区域统计是 **19 个区域、共 39 个原图像素、涉及 12 张图**。

## 2. 验收中完善的实现

1. **损失数值边界。**CE 显式接收 `.float()` logits；独立回归证明，FP16 logits 在 autocast 之外对 512 标签求 CE 的原写法可能出现 NaN，修复后有限且可反传。正常 CUDA autocast 已会将 CE 放到 FP32，不能把此问题描述成所有 AMP 训练都会失败。Dice 仍以 FP32 求概率及累计。官方说明见 [PyTorch AMP](https://docs.pytorch.org/docs/2.10/amp.html)。
2. **参数与输入契约。**拒绝错误类别数、非 int64 标签、非法标签、非有限/非法权重与 epsilon；记录运行时真实损失参数。混合全忽略图时，`valid_images` 只计算有有效像素的图片。
3. **指标边界。**原图 GT 的 255 从混淆量与面积分母剔除，全忽略 GT 拒绝；非法小数、原始未合并的类别 ID、NaN、错误几何和阈值拒绝。NA 原因补全；消除 NumPy/Torch 混用的弃用警告。
4. **可信的 probe。**显式 seed 42；一个模型在 eval/no_grad 下执行 B2、B1、B1 重复；相同缓存 logits 单独改变累计分组。头/全模型参数在前后哈希一致；数据清单、分组协议、前序验收锚点和真实快照文件离线核对，身份错误时提前停止。
5. **防止跳过检查后假通过。**`--skip-model` 仅为夹具诊断，未执行的模型检查是 `null`；即使 CLI 退出码为 0，`ok`、`full_stage_verified` 仍为 false。完整验收必须实际执行真实 GPU probe。
6. **显式固定 FP32 精度。**默认 TF32 卷积开启时，真实 XOR 为 621，最大概率差 `2.855062484741211e-5`，超过预先规定的 `1e-5`。仅关闭本进程 TF32 后的对照为 XOR 1、最大概率差 `1.4901161193847656e-7`；B1 重复为零差异。详见 [诊断与原始日志](./stage07_diagnostics/README.md)。预算没有放宽；最终统一配置和重测结果以第 3 节为准。

FP32 精度配置只影响当前项目 Python 进程，不修改系统、驱动或其他环境。AMP FP16 训练仍可开启。批量与逐样本浮点计算不保证按位相同，参见 [PyTorch 数值精度说明](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)；本次对照不构成对某个具体 cuDNN kernel 的定位，也不保证其他硬件必有相同像素差。

## 3. 最终验证

**第七阶段通过验收，可以进入阶段 08。**统一精度策略后的独立重测结果如下：

| 验证 | 实际结果 | 证据 |
| --- | --- | --- |
| 人工夹具、身份、真实 CUDA probe | **42/42**；`ok=true`、`full_stage_verified=true`、`not_a_baseline=true`；18.99 s | [完整 JSON](./loss_metrics_check.json) |
| 全项目回归 | **186 项通过**；10.629 s，退出码 0 | [测试日志](./stage07-unit-tests.txt) |
| B2 与 B1 独立前向 | 真 XOR **1 / 9,013,680**；最大 logits 差 `3.725290298461914e-7`；最大概率差 `1.4901161193847656e-7`，保持 `atol=1e-5, rtol=0` | JSON 的 `two_forward_passes`，含逐图及阈值两侧距离 |
| 固定 B1 重复 | XOR **0**；logits/概率最大差均 **0.0** | JSON 的 `batch_1_repeat` |
| 相同缓存 logits 换累计分组 | 全局、逐图、输入/预测哈希严格一致 | `same_logits_layout_equal=true` |
| 输入、代码与快照身份 | 41 个记录文件在检查前后 SHA/大小一致；v2 三清单、风险组、预览 ID、阶段 05/06 锚点及真实模型快照均通过 | JSON 的 `identity`、`provenance` |
| 恢复图 | 已实际查看；正确质心线与理论线贴近，漏裁边路径明显偏移；归档 PNG 与本次生成图 SHA 相同 | [恢复对照图](./figures/stage07-restore-asymmetric.png) |

本机沙箱内 `conda.exe` 启动器出现系统 DLL 重定位错误；原 runner 在沙箱外只读启动验证成功。以上最终 GPU 检查与完整回归均使用原 runner 在沙箱外执行，没有修改启动脚本、包、系统或其他环境。不要将该工具沙箱问题误判成模型或 Conda 环境损坏。

## 4. 如何重新核验

在项目根目录打开 PowerShell。以下命令只使用已验证的独立环境，不安装或升级包：

```powershell
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @(
  '-m', 'segformer_lab.metrics_check',
  '--config', 'configs/baseline.yaml',
  '--report', 'reports/loss_metrics_check.json'
)
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @('-m', 'unittest', 'discover', '-v')
```

判断依据是完整 JSON 中 `ok=true`、`full_stage_verified=true`、所有检查为 true；仍应同时存在 `not_a_baseline=true`。`--skip-model` 的退出码不能代替这一判断。执行日志保存在 [检查日志](./stage07-loss-metrics-check.txt) 与 [回归日志](./stage07-unit-tests.txt)。恢复示意图为 [小型归档图](./figures/stage07-restore-asymmetric.png)。

## 5. 交付位置与已知边界

- [阶段 07 学习文档](../docs/stages/07-loss-metrics-design-and-implementation.md)：公式、代码接口、数据契约、实现细节与学习问题。
- [当前完整报告](./loss_metrics_check.json)：人工夹具、身份、随机头通路、逐像素差异和代码/输入哈希。
- [验收前历史说明](./stage07_pre_acceptance/README.md)：旧报告仅为追溯，不能继续引用其中的 12 px。
- [阶段索引](../docs/stages/README.md)：当前进度和后续依赖。

当前仍没有训练器、优化器更新、真实 AMP 训练显存预算或训练后质量结论。workers=0、训练 B2/累积 1、正式评价 B1 保持；不同 batch 含正类图片的数量会影响逐图正类 Dice 的组成，不把任意梯度累积宣称为与大 batch 数学等价。阶段 08 需验证实际训练，而不是复用前向显存当训练预算。

## 6. 下一步：可以合并执行阶段 08、09

可以在同一轮交给实施 Agent，完成后一起提交独立验收。执行顺序为 **08 内部门槛通过 → 09**，使用独立配置、run 目录和中文学习文档；08 出现未解释的失败就先停在 08。

完整可复制指令及交付清单见 [阶段 08—09 执行交接](./2026-11-01-stage08-09-execution-handoff.md)。本轮验收不代为启动这两个实验。阶段 09 结束后也不自动进入阶段 10 或正式训练。
