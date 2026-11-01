# 第十一阶段原始小型证据

`formal` 保存原正式 run 的身份、配置、清单及原复验记录；完整 summary、50 轮逐图日志和 13,250 条更新日志分别无损 gzip 压缩。压缩头不含机器时间，解压内容已逐字节核对原件。没有复制权重、完整概率数组或批量图片。

`recorded_source` 保留原训练使用的 `train_full.py`、`pilot.py` 准确文本，以及旧恢复证据脚本。它们是历史资料，不是当前可导入模块。当前源码修复不会冒充产生原正式 run 的版本。

`resume_fixture_A/B/D/E` 和 `original-resume-report.json` 保留原恢复对照。旧报告存在以下范围错误，不能按“恢复全部通过”引用：

- 所有 fixture 的 best 都是 epoch 1，原 NPZ 来自 best。其概率对比不能代表续训结束的 epoch 3 last；D/E 的零差更不能证明继续更新后仍一致。
- 原脚本所谓 state_at_load 实际读取了已覆盖到 epoch 3 的 B last，而非初始 epoch 1 边界；对 Python/NumPy RNG 和真实 DataLoader generator 的比较也不完整。
- A/B 在恢复前的模型已不同，不能将最终偏差全部归因于恢复边界；旧报告对 GPU 唯一成因的解释超出了证据。
- B/D/E 恢复后的 artifact manifest 自身条目过期；当前实现已排除自引用及旧 summary，原件不改。

本轮另建有界的 `resume_acceptance_A/B_20261101`：A 连跑三轮，B 在一轮后保存并在 run 外冻结 epoch1-last.pt，先独立比较完整加载状态和真实下一轮数据顺序，再续完剩余两轮；最终比较实际 epoch 3 **last**，不再取 best。新结果与预声明 `atol=1e-5` 的范围以独立报告为准，不能因未通过而放宽门槛。

正式 50 轮未重跑，test 图像未读取。原件只读审计见上一级 `stage11-evidence-audit.json`；当前完整验收见[第十一阶段验收](../2026-11-01-stage11-acceptance.md)。

新对照的实际结论：边界状态及下一轮顺序通过；真正epoch3 last的概率最大差为0.00015032291412353516、XOR33，未通过原定门槛。原件中的旧结论保留用于审计，不代表当前认可。`original-unit-tests.txt`保留历史242项测试的原始控制台字节（含尾随空格），仅对此文件设置Git whitespace例外；当前263项回归日志在上一级。

解压示例（只写资源目录，不覆盖训练 run）：

```powershell
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @('-c', "import gzip,pathlib; p=pathlib.Path('reports/stage11_evidence/formal/epochs.jsonl.gz'); o=pathlib.Path('E:/桌面/AI/segformer_resources/tmp/stage11-epochs.jsonl'); o.write_bytes(gzip.decompress(p.read_bytes()))")
```
