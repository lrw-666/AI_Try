# PCB1 SegFormer-B0 — 结果摘要

> 本文所有数字来自同一条 run `pcb1_b0_v2_seed42` 与它的评价目录，**没有重新训练、没有重新选 best、没有第二套指标实现**。
> val 用于选模；test 原始一次性评价与验收技术复核分开留档，未据复核分数择优。两者分列，不混用。

## 1. 模型与选模

| 项目 | 值 |
| --- | --- |
| best epoch / step | **39** / **10330** 次累计成功更新（best checkpoint 时刻） |
| 选择规则 | highest val global foreground IoU; ties keep the earlier epoch |
| best 权重 SHA-256 | `192c827bef00607f4258e5fadf0f1d97970ca5cc727a0add749be0fba5997485` |
| 训练预算 | 50 epoch × 265 batch = 13 250 次尝试；13243 次成功、7 次 AMP 跳步 |

## 2. 验证集（132 = 120 正常 + 12 异常）

| 指标 | 值 |
| --- | --- |
| 全局前景 IoU / Dice | **0.7775** / 0.8748 |
| Precision / Recall | 0.9398 / 0.8183 |
| TP / FP / FN / TN | 279,246 / 17,894 / 62,003 / 197,941,817 |
| 异常 12 张 macro IoU / Dice | **0.5396** / 0.6203 |
| 正常图 FP | 1,495 px（11/120 张有任意 FP），比例 8.293e-06 |

## 3. 测试集（442 = 402 正常 + 40 异常，冻结协议）

| 指标 | 值 |
| --- | --- |
| 全局前景 IoU / Dice | **0.6833** / 0.8119 |
| Precision / Recall | 0.9008 / 0.7390 |
| TP / FP / FN / TN | 252,227 / 27,779 / 89,101 / 663,638,653 |
| 异常 40 张 macro IoU / Dice | **0.5249** / 0.6351 |
| macro P / R（有定义张数） | 0.8071（38/40） / 0.5832 |
| 正常图 FP | 3,689 px（32/402 张），比例 6.108e-06 |
| 每图预测前景比（正常） | 中位 0.0e+00，P95 2.19e-05，最大 3.83e-04 |

## 4. 怎么读这两个数

- **全局与 macro 权重及样本范围不同**：全局先累加全部图的 TP/FP/FN，再除一次；可视为按各图并集像素数加权。异常 macro 则对 40 张异常图等权，正常图 FP 另报。0.6833 与 0.5249 的差距提示逐图表现不均，尚不能证明小缺陷一定弱于大缺陷；需要按 GT 面积/宽度分层验证。
- **val 与 test 必须分列**：val 参与选模，这会影响与 test 的比较；本次 test 更低的具体原因未经对照验证。不能用其中一组分数替换另一组。
- **正常图只有像素级统计**：32/402 张"有任意误报像素"不等于"图像级假阳性率"，本项目没有定义图像级判定规则。

## 5. 图表

| 文件 | 内容 |
| --- | --- |
| `figures/training-curves.png` | 50 轮 loss / LR / val IoU、Dice、macro IoU / 正常 FP（symlog 保留零），标注 best epoch 39 |
| `figures/anomaly-fixed-6.png` | 固定 6 张异常：原图 / GT / 概率（0–1 固定色标）/ 二值 / TP-FP-FN |
| `figures/anomaly-worst-cases.png` | IoU 最低与 recall 最低的并集（按规则排序） |
| `figures/normal-fixed-6.png` | 固定 6 张正常：原图 / 全零 GT / 概率 / 误报（红） |
| `figures/normal-worst-fp.png` | 正常图误报像素最多的 5 张 |
| `curves.csv` | 逐轮数值（epoch / attempt / successful / LR / loss / val 指标） |
| `failure-cases.csv` | 入选图 ID、入选理由、逐图计数与重建 XOR |

色标说明：概率图统一 `vmin=0, vmax=1`，没有按图拉伸；TP 绿、FP 红、FN 蓝，图内有图例。
来源：run `E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42`；evaluation `E:\桌面\AI\segformer_resources\runs\pcb1_b0_v2_seed42_eval_acceptance_20261101`。生成器在推理与写图前核验评价文件清单、best/processor/config/日志身份与展示选择；完整 SHA 见 `results-summary.json.source_validation`。
