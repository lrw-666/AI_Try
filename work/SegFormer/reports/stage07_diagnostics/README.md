# 阶段 07 数值精度对照

这些是固化最终精度配置之前执行的诊断，不是训练实验，也不是正式性能结果。两次运行均使用同一固定数据、seed 42、预训练编码器和随机解码头，模型未更新。报告内保留当时的代码与输入哈希；后续代码修改后，这些哈希不应冒充当前版本。

| 设置 | cuDNN TF32 | matmul TF32 | B2/B1 真实 XOR 像素 | 最大概率绝对差 | 固定预算 `atol=1e-5, rtol=0` |
| --- | --- | --- | --- | --- | --- |
| 默认进程 | 开 | 关 | 621 / 9,013,680 | 2.855062484741211e-5 | 未通过：41/42 |
| 仅关闭 TF32 | 关 | 关 | 1 / 9,013,680 | 1.4901161193847656e-7 | 通过：42/42 |

两次运行的 B1 与 B1 重复结果均逐像素相同。相同缓存 logits 改变累计分组也严格相同。XOR 衡量阈值化预测是否不同；容差检验还检查完整概率差及变化像素在阈值两侧的距离，不能仅看差异比例。

- 默认报告：[default_tf32_loss_metrics_check.json](./default_tf32_loss_metrics_check.json)；[执行日志](./default_tf32_check.txt)。
- IEEE 诊断：[ieee_loss_metrics_check.json](./ieee_loss_metrics_check.json)；[执行日志](./ieee_check.txt)。
- `loss_metrics_resolved_config.yaml` 为该目录最后一次诊断的解析配置；当时 TF32 关闭操作由 Python 入口显式设置，尚未写入 YAML，所以应结合 JSON 的 `environment` 判断实际状态。

控制变量结果支持将 FP32 计算精度显式固定；本次没有分析具体 cuDNN kernel，不能声称已定位某个 kernel。PyTorch 也不保证逐样本与批量计算按位一致，见[官方数值精度说明](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)。正式验收保持原容差，不因结果不通过而放宽门槛。

这项设置仅作用于项目 Python 进程，不修改显卡驱动、系统配置或其他 Conda 环境。AMP FP16 训练仍可开启；后续 FP32 评价必须显式使用当前项目统一精度配置。
