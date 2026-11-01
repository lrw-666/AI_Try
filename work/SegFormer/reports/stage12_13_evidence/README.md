# 第12—13阶段验收证据说明

日期：2026-11-01。当前结论见[联合验收](../2026-11-01-stage12-13-acceptance.md)。这里保存小型统计、输入/输出身份与执行源码，不保存大权重、数据、全量预测或完整概率。

- `pre_acceptance_source`：提交7478742的四份源码准确字节。它们不是原首次计分源码的完整备份；`source-identity-scope.json`列出首次元数据SHA与其差异，原TF32修复前源码文本不可得。
- `original_eval`：首次eval001的元数据、门禁、规则、选择和产物清单原件。原完整统计的便携副本仍在上一级`stage12_evidence`。
- `original-stage12-summary.json`与`original_results`：验收前小型汇总/文本/CSV的原件；原五张图已存在Git提交7478742，保留SHA而不重复上传PNG。
- `accepted_eval`：本次完整132张val、442张test及单图的元数据、指标、规则、门禁、产物清单；逐图JSON/CSV无损gzip，解压后与独立验收输出字节完全相同。
- `accepted-eval-archive.json`：上述统计的原始与便携SHA/大小。预测PNG与完整浮点概率在资源目录，不随仓库上传。
- `regenerated-results-archive.json`：从新评价生成五张修正图与相关结果材料的原始SHA。仓库实验卡与results-summary仅额外添加验收状态/链接，其余科学数值与图像不改。
- `acceptance_driver.py.txt`：实际由项目runner执行的一次性验收脚本。val/test/predict分别在新进程执行，模型hook核对实际精度/设备；完整split的每张PNG与原始GT独立重算混淆量。脚本不训练、不选参，原训练与eval001目录只读。

实际命令（已执行，不能在同名输出目录重复运行）：

```powershell
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @('reports/stage12_13_evidence/acceptance_driver.py.txt','val')
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @('reports/stage12_13_evidence/acceptance_driver.py.txt','test')
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @('reports/stage12_13_evidence/acceptance_driver.py.txt','predict')
```

原始评价、验收技术复核均保留，不根据结果选优。标量审计不能替代像素真值检查，代码文字也不能替代实际前向观察；两类证据在上一级各自的报告中明确区分。没有全系统访问审计，不声称证明任何其他进程从未读取过test。

上一级`stage12-fix-tests.txt`保留原始失败/修复后通过的控制台字节，包含两处原生stderr尾随空格；仅对该日志设置Git whitespace例外，其余源码与报告继续检查，未改写日志来消除历史格式。
