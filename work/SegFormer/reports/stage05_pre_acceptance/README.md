# 第五阶段原测量记录

这里保留验收前的 model_check.json，用于追溯 CE 差值 5.96e-08 和 B2 reserved 1058 MiB 的原始读数。

原代码在显存 probe 前已经执行 B2 调试更新，仍保有优化器状态和部分张量，所以该读数不能解释为“不含优化器状态的隔离前反向”。修正后按 B1、B2、调试更新的顺序重新测量，当前结果与解释请看[第五阶段验收报告](../2026-10-31-stage05-model-loading-and-check.md)。

此目录不是当前运行验收入口，不用它替代 reports/model_check.json。
