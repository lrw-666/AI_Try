# Proposal reading PDFs - 261107

Ordered collection of the papers in Sections 7.1 and 7.2 of the [current research guide](../../research/thesis-research-and-study-guide-zh.md). Folder numbers give the recommended reading sequence. This collection does not mean that all 14 papers must be read in full before the proposal.

- **1-5: read first.** Understand the task, the implemented baseline, the proposed feature backbone and the transfer/adaptation motivation.
- **6-10: targeted reading for the two proposed works.** Focus on the mechanism, required inputs, supervision and evidence checks.
- **11-14: consult as needed.** Broader precedents and optional foundations. If token/patch concepts are unclear, consult paper 14 before paper 3.

The four PDFs already stored under `261029` were copied unchanged for a self-contained reading folder. Ten other PDFs were downloaded from publishers, proceedings or arXiv. Older source folders and translations remain unchanged.

## Reading order

| Order | Paper / local PDF | Version | Pages |
| ---: | --- | --- | ---: |
| 1 | [SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation](<01. VisA-SPD/SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation.pdf>) | Existing project PDF; copied unchanged | 25 |
| 2 | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](<02. SegFormer/SegFormer- Simple and Efficient Design for Semantic Segmentation with Transformers.pdf>) | Existing project PDF; copied unchanged | 18 |
| 3 | [DINOv3](<03. DINOv3/DINOv3.pdf>) | Existing project PDF; copied unchanged | 67 |
| 4 | [Rethinking Transfer Learning for Industrial Inspection: DINOv3 vs. ImageNet Pretraining Across RGB and X-ray Tasks](<04. Industrial-Transfer-Learning/Rethinking Transfer Learning for Industrial Inspection- DINOv3 vs. ImageNet Pretraining Across RGB and X-ray Tasks.pdf>) | arXiv v1; accepted CVPR 2026 VISION workshop | 9 |
| 5 | [SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection](<05. SuperSimpleNet/SuperSimpleNet- Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection.pdf>) | arXiv v2; accepted ICPR 2024 | 20 |
| 6 | [S³DNet: Support-Driven Spatial–Semantic Decoupling for Few-Shot Strip-Steel Surface Defect Segmentation](<06. S3DNet/S³DNet- Support-Driven Spatial-Semantic Decoupling for Few-Shot Strip-Steel Surface Defect Segmentation.pdf>) | Mathematics 14(20), 3730 (2026), publisher PDF | 26 |
| 7 | [Towards Total Recall in Industrial Anomaly Detection](<07. PatchCore/Towards Total Recall in Industrial Anomaly Detection.pdf>) | CVPR 2022 proceedings PDF | 11 |
| 8 | [Siamese network with change awareness for surface defect segmentation in complex backgrounds](<08. CADNet/Siamese network with change awareness for surface defect segmentation in complex backgrounds.pdf>) | Scientific Reports 15, 11814 (2025), publisher PDF | 19 |
| 9 | [Explainable AI reveals Clever Hans effects in unsupervised learning models](<09. Clever-Hans-Effects/Explainable AI reveals Clever Hans effects in unsupervised learning models.pdf>) | Nature Machine Intelligence 7, 412-422 (2025), publisher PDF | 14 |
| 10 | [Sanity Checks for Saliency Maps](<10. Sanity-Checks/Sanity Checks for Saliency Maps.pdf>) | NeurIPS 2018 proceedings PDF | 11 |
| 11 | [Explainable Visual Anomaly Detection via Concept Bottleneck Models](<11. CONVAD/Explainable Visual Anomaly Detection via Concept Bottleneck Models.pdf>) | arXiv v3, revised 2026-10-12; cite as preprint | 27 |
| 12 | [AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2](<12. AnomalyDINO/AnomalyDINO- Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2.pdf>) | WACV 2025 proceedings PDF | 11 |
| 13 | [Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection](<13. Dinomaly/Dinomaly- The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection.pdf>) | CVPR 2025 proceedings PDF | 11 |
| 14 | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](<14. ViT/An Image is Worth 16x16 Words- Transformers for Image Recognition at Scale.pdf>) | Existing project PDF; copied unchanged | 22 |

## What to look for

**1. VisA-SPD** - Task, VisA supervision protocols and sensitivity to local defects. Read introduction, dataset/protocol and supervised experiments.

**2. SegFormer** - Understand the existing baseline: Figure 2, encoder/decoder and Section 3; connect the paper to actual tensor shapes.

**3. DINOv3** - Dense features, patch/token geometry and downstream interfaces. Read selectively; no need to reproduce pretraining.

**4. Industrial-Transfer-Learning** - Evidence that transfer depends on target modality and adaptation. Its distilled ConvNeXt configuration is not the planned ViT configuration.

**5. SuperSimpleNet** - Frozen features, adaptation and supervision. The VisA experiment uses normal-only training, so its scores are not direct supervised PCB1 comparisons.

**6. S3DNet** - Closest local-detail/frozen-feature mechanism to check before claiming novelty for work 1. Strip-steel episodic few-shot is a different protocol.

**7. PatchCore** - Normal patch memory and nearest-neighbour distance scoring. Retrieval itself is established and does not explain an unrelated prediction branch.

**8. CADNet** - PCB reference-difference segmentation. Check normal-reference input requirements before transferring the idea to unpaired VisA images.

**9. Clever-Hans-Effects** - Motivation to examine actual industrial-model evidence. It does not prove that the current PCB model has the same shortcut.

**10. Sanity-Checks** - Why appealing saliency maps are insufficient; understand randomization checks and their limits.

**11. CONVAD** - Optional explanation-object, intervention and branch-faithfulness reference. Concept bottlenecks are not a mandatory additional project task.

**12. AnomalyDINO** - Optional frozen DINO feature and nearest-neighbour industrial localization precedent; uses DINOv2 and a different supervision protocol.

**13. Dinomaly** - Optional reconstruction-based DINO industrial anomaly-detection precedent; not the current supervised PCB segmentation protocol.

**14. ViT** - Conditional foundation only: if patch/token concepts are unclear, consult before paper 3. A full reading is not required before the proposal.

## Sources and checks

The [download manifest](download-manifest.json) records each full title, source, version, page count, file size and SHA-256. First-page titles, PDF structure and page geometry were checked. Full published text is stored without translation, rewriting, or page edits. Filenames follow the existing convention: a colon becomes a hyphen, Windows-invalid filename characters are replaced, and typographic dashes are normalized to ASCII hyphens.

Use the paper-specific supervision and input conditions when discussing results. CONVAD is recorded as the verified arXiv v3 preprint; it is not labelled as a confirmed conference publication. Proceedings PDFs for AnomalyDINO and Dinomaly are the formal papers, not later code ports.

| Order | Primary landing page | PDF source / local reuse |
| ---: | --- | --- |
| 1 | [Paper](https://arxiv.org/abs/2207.14315) | [Existing project PDF](<../261029/1. VisA/SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation.pdf>) |
| 2 | [Paper](https://arxiv.org/abs/2105.15203) | [Existing project PDF](<../261029/2. SegFormer/SegFormer- Simple and Efficient Design for Semantic Segmentation with Transformers.pdf>) |
| 3 | [Paper](https://arxiv.org/abs/2508.10104) | [Existing project PDF](<../261029/4. DINOv3/DINOv3.pdf>) |
| 4 | [Paper](https://arxiv.org/abs/2605.23472) | [Downloaded PDF](https://arxiv.org/pdf/2605.23472v1) |
| 5 | [Paper](https://arxiv.org/abs/2408.03143) | [Downloaded PDF](https://arxiv.org/pdf/2408.03143v2) |
| 6 | [Paper](https://www.mdpi.com/2227-7390/14/20/3730) | [Downloaded PDF](https://www.mdpi.com/2227-7390/14/20/3730/pdf) |
| 7 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) | [Downloaded PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf) |
| 8 | [Paper](https://www.nature.com/articles/s41598-025-94733-4) | [Downloaded PDF](https://www.nature.com/articles/s41598-025-94733-4.pdf) |
| 9 | [Paper](https://www.nature.com/articles/s42256-025-01000-2) | [Downloaded PDF](https://www.nature.com/articles/s42256-025-01000-2.pdf) |
| 10 | [Paper](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) | [Downloaded PDF](https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf) |
| 11 | [Paper](https://arxiv.org/abs/2511.20088) | [Downloaded PDF](https://arxiv.org/pdf/2511.20088v3) |
| 12 | [Paper](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html) | [Downloaded PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf) |
| 13 | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html) | [Downloaded PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.pdf) |
| 14 | [Paper](https://arxiv.org/abs/2010.11929) | [Existing project PDF](<../261029/3. ViT/An Image is Worth 16x16 Words- Transformers for Image Recognition at Scale.pdf>) |
