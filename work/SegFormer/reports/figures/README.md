# 第四阶段的少量近重复复核图

仅保存 0042/0043 的三个局部并排图（合计约 0.93 MB），支持[第四阶段验收报告](../2026-10-31-stage04-acceptance.md)中的分组决策；没有上传数据集或整批五联图。

来源为 VisA PCB1 原图 `pcb1/Data/Images/Normal/0042.JPG` 与 `0043.JPG`。每张左侧为 0042、右侧为 0043；原图像素裁切后加 30 px 白色标题条，未对齐配准、修图或增强。

| 文件后缀 | 原图裁切框 `(left, top, right, bottom)` |
| --- | --- |
| header | `(470, 140, 850, 430)` |
| mesh | `(210, 365, 540, 730)` |
| background | `(860, 0, 1250, 300)` |

用 Pillow 的 `Image.crop(box)` 分别裁切两图，再水平拼接即可复现。源图 SHA 在 `reports/data_audit/per_image.csv`，本目录 PNG 的 SHA 在 `reports/stage04-acceptance.json`。
