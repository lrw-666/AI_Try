# 第八、九阶段验收与完善

验收日期：2026-11-01。**两阶段通过验收，可以进入阶段 10 的三轮完整数据试跑。**本轮没有启动 pilot 或正式训练，也没有进行 test 图像推理。

## 1. 结论及指标口径

| 阶段 | 实际证据 | 结论 |
| --- | --- | --- |
| 08：完整训练通路 | 原 smoke 20 成功/20 尝试/0 跳步；本轮新进程使用保存 processor，概率最大差 0.0、预测 XOR 0；保存状态全部核对 | 训练、保存、重载通路通过 |
| 09：固定八图学习 | 200 成功/201 尝试/1 跳步；训练子集全局 IoU 0.008250→0.945710、异常图宏平均 IoU 0.014414→0.760745；正常图 FP 873,511→85 | 已证明可以拟合这八张训练图 |
| 修正后真实 smoke 回归 | 新 run `smoke_003_stage09_acceptance`：20/20/0；编码器/头均有梯度与参数变化；新进程概率差 0.0、XOR 0 | 共享训练代码修正后 CUDA 路径通过 |
| 完整项目回归 | **207 项通过，10.281 s，退出码 0** | 包含新增训练与复验边界测试 |

**0.945710 是八张已参与训练图片上的数值，不是留出验证、测试或论文基线成绩。**043 的缺陷占子集 GT 前景约 96.43%，必须同时看宏平均和小缺陷逐图分数。最终 065/038/031/043 的 IoU 分别为 0.6932/0.8239/0.5665/0.9593；031 仍有明显多报与漏报。

原阶段 08 已有提交 `eb7c3ec`，本轮连同第 09 阶段成果及独立验收修正保存。数据、权重、环境、完整概率数组及批量输出留仓库外，只收代码、配置、中文学习文档、小型证据与精选图片。

## 2. 独立核验了哪些证据

- 三个原始有效 run 的 **47 个产物** bytes/SHA 与 manifest 一致；固定 8 图及 4 掩码共 12 个源文件哈希、三划分清单、固定快照与元数据均核对。
- 对更新 0/50/100/150/200 的逐图 TP/FP/FN/TN 独立求和，重算全局 IoU/Dice、异常宏平均和正常误报；与原曲线、摘要一致，每次 12,018,240 像素。
- 新进程严格加载第 09 阶段保存的 **208 个模型状态张量**，与 training_state 全部相等且有限；**205 个参数的优化器 step 均为 200**，计数为 200/201/1，GradScaler scale 为 32768。
- 原图评价使用 run 内 processor、B1、FP32、IEEE 精度上下文及无增强。重新计算的全部逐图混淆量、指标和 NA 字段与原结果一致；历史 split 元数据错误单独说明并校正展示。
- 原 smoke 与新 smoke 的旧概率数组实际可用，因此可验证概率差 0.0 与 XOR 0。**overfit 原 run 没存旧概率图**，该项明确为 `not_available/null`，不伪造“概率完全相同”；通过结论依据严格模型状态、实际前向及全部原图分数复算。

证据入口：[原始数值审计](./stage08-09-evidence-audit.json)、[原 smoke 重载](./stage08-reload-check.json)、[第 09 阶段重载](./stage09-reload-check.json)、[新 smoke 重载](./stage09-smoke-reload-check.json)、[完整测试日志](./stage09-unit-tests.txt)。

## 3. 已修复的问题

| 问题 | 修正及验证 |
| --- | --- |
| 梯度范数 inf 被直接视为 AMP 跳步 | 区分逐元素非有限、scaler 缓存决定和有限元素的范数归约溢出。正常路径先验证范数再裁剪；不满足真实跳步条件时，在 optimizer step 前停止 |
| 仅靠 scale/范数推断成功更新 | optimizer post-hook 观察真实执行；跳步时检查参数及优化器 step 未变；执行后参数必须有限，首步编码器与头有真实梯度和变化 |
| 连续/累计跳步混淆 | 成功时重置连续计数；3 次连续跳步停止；真实 CPU GradScaler 回归覆盖间断与连续两种情况 |
| 训练图被标为 val | `evaluate_dataset` 使用实际 `dataset.split`；原历史文件不改写，新报告证明这 8 个 ID 属于 train，并显式登记元数据纠正 |
| 评价尾部无条件 `model.train()` | 正常及异常退出都恢复调用方原 train/eval 模式 |
| 原重载没有实际使用保存 processor | 将保存的 processor 传给 Dataset；新复验器验证实例实际被使用，核配置及清单身份；两个 smoke 保存前后概率一致 |
| “局部放大”仅有全图和 ROI 框 | 修为上排全图、下排实际 ROI 的 source/GT/概率/预测；从已保存权重生成，不重训八图 |
| 更新尝试预算少算 3 次 | 三次中断都进入 attempt 76，应为 `3×76+201=429/500`，而非 426 |
| 正常图初始化 FP 写错 | 正确为 873,511；1,965,699 是全部八图 FP。已改摘要、README、索引与学习文档 |
| 原来源记录漏掉训练入口 | 当前扩展代码身份范围；保存准确的历史训练源码，原入口由当时 HEAD 恢复，并保留证据局限 |

额外发现：checkpoint 中 `model_config.id2label` 使用整数键，写 JSON 后变为字符串键。复验器先核实差异，再按 JSON 表示规范化后完整比较；标签值或类别数改变仍会失败。这是序列化表示差异，不能误判成模型结构损坏。

## 4. 图片检查与结果解释

本轮实际查看了 0/50/100/150/200 五张总览、旧“放大图”及修正后的 065/031 局部图。早期主要先学到 043 的大缺陷，之后小缺陷改善，与宏平均曲线一致；未见新增图像/掩码整体错位。

- [初始化八图总览](./figures/stage09-initial.png)与[保存权重重载后的最终总览](./figures/stage09-final.png)。
- [065 真正的局部放大](./figures/stage09-zoom-065.png)：主要瑕疵被定位，细小区域仍有漏检，不能据 0.693 的逐图值宣称完整检出。
- [031 真正的局部放大](./figures/stage09-zoom-031.png)：左侧连接区域预测偏粗，右侧细结构有遗漏，与 FP/FN 记录相符，保留为失败案例。

只把上述少量学习图片纳入仓库；整批历史图仍在资源目录，原图未覆盖。

## 5. 资源、历史和已知限制

当前代码的新 smoke 峰值 allocated/reserved 为 **806.5/934 MiB**，训练步中位 **103.0 ms**、训练循环墙钟 **6.711 s**；原 overfit 训练步约 58.3 ms、循环墙钟 48.773 s 含评价回调。测量对象和代码不同，不用首步或旧中位速度推算完整训练时长。主机 RAM 可用约 1.78 GiB，完整验证需流式处理，避免保存全部原图概率到内存。

原 `smoke_001` 记录的训练源码 SHA 与阶段 08 Git 提交不同，准确历史执行源码未恢复；本次不宣称其与提交完整一致。新 smoke 有当前代码身份且闭环通过。overfit 训练源码准确快照已归档，入口从已记录 HEAD 推定，但原 run 未自动保存入口 SHA。这些局限见[原件归档说明](./stage08_09_evidence/README.md)。

本轮保存的 checkpoint 能严格加载，但**未验证完整 epoch 断点续训**；原格式不含全部 scheduler/RNG/epoch 状态，不能因为有 optimizer/scaler 就声称 resume 完整。workers>0、长期 AMP 跳步率、完整数据泛化和正式训练仍待后续实验。训练步读取 scaler 的私有接口受当前锁定 PyTorch 2.10 限制，升级前须重验。

## 6. 复现与学习文档

原始 run 只读复验可执行：

```powershell
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try\work\SegFormer'
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @(
  '-m','segformer_lab.run_check',
  '--run','E:\桌面\AI\segformer_resources\runs\overfit_002',
  '--config','configs/overfit.yaml',
  '--out','E:\桌面\AI\segformer_resources\tmp\stage09-recheck.json',
  '--device','cuda'
)
& '.\scripts\run_project.ps1' -Quiet -PythonArgs @('-m','unittest','discover','-v')
```

若工具沙箱内 Conda 启动器再次出现 DLL 重定位错误，本轮已验证原 runner 在沙箱外可运行；没有为此安装包、重建环境或修改系统。不要将启动沙箱的问题当成模型失败。

[第 08 阶段学习文档](../docs/stages/08-smoke-design-and-implementation.md)已补验收修正；[第 09 阶段学习文档](../docs/stages/09-overfit-design-and-implementation.md)解释真实 step、AMP 边界、宏平均、保存预处理和复验可证明的范围。机器验收摘要为 [stage09-acceptance.json](./stage09-acceptance.json)。

## 7. 下一步

仅执行[阶段 10 完整训练集三轮 pilot](./2026-11-01-stage10-execution-handoff.md)：530 train、132 val，移除八图子集，恢复随机增强和每轮 `set_epoch`，从原固定快照重新初始化。当前 `train.py` 仍是专用八图入口，需按交接实现完整 epoch 路径，不能直接套配置。

三轮完成后再次验收，再决定正式训练。此处没有启动第 10 或第 11 阶段。
