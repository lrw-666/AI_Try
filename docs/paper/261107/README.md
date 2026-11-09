# Proposal reading PDFs - 261107

Updated: 2026-11-09. This is the numbered source collection for the [current research guide](../../research/thesis-research-and-study-guide-zh.md) and the [industrial/PCB field review](../../research/2026-11-09-industrial-vision-pcb-research-status-and-roadmap-zh.md).

The original 14 papers are retained and five focused references have been added, giving **19 original PDFs**. Folder numbers and PDF filename prefixes now match. The original PDF bytes are unchanged; the ten existing Chinese translations have moved with their papers, with only their source-PDF links updated. No new full translations are implied for the five additions. The older `261029` sources and translations remain unchanged.

The collection supports a PCB application and task-adaptation thesis. Existing components can be reused when the PCB problem, necessary adaptation, fair comparisons, gains and limits are demonstrated. A dataset change alone does not establish methodological novelty. Local-detail branches and position-restricted references remain candidates, not established contributions or compulsory architectures.

## Read selectively

| Purpose | Orders | Question to answer |
| --- | --- | --- |
| First pass | 01, 02, 04, 05, 06 | What is the supervised task, what has been implemented, and why might representation adaptation help? Read relevant sections, not every page. |
| Supervised application foundation | 03 | How can limited defect samples, annotation precision and a specific industrial problem guide a segmentation study? |
| Work 1 candidate and PCB context | 07, 08 | What already exists in frozen-DINO/local-detail methods and single-image PCB segmentation? What needs adaptation and evidence here? |
| Work 2 candidate and direct predecessors | 09, 10, 11 | How do normal memory, position filtering, feature distance and paired-image differences work? Which assumptions hold for unpaired VisA? |
| Verifiable decision evidence | 12, 13, 14 | Which computation is explained, and how can correspondence, interventions and branch coverage be tested? |
| Adjacent approaches | 15, 16 | What can normal-only DINO methods teach us without mixing supervision protocols or scores? |
| Evaluation and application limits | 17, 18 | Why do thresholds, lighting, hidden tests and runtime matter? These readings do not add two benchmark projects. |
| Conditional foundation | 19 | If patch/token concepts are unclear, consult this before 04; a full reading is optional. |

## Reading order and local originals

| Order | Paper / local PDF | Stored version | Pages |
| ---: | --- | --- | ---: |
| 01 | [SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation](<01. VisA-SPD/01-VisA-SPD-2022.pdf>) | Existing project PDF; copied unchanged | 25 |
| 02 | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](<02. SegFormer/02-SegFormer-2021.pdf>) | Existing project PDF; copied unchanged | 18 |
| 03 | [Segmentation-Based Deep-Learning Approach for Surface-Defect Detection](<03. Supervised-Surface-Defect-Segmentation/03-Tabernik-Supervised-Surface-Defect-Segmentation-2019.pdf>) | arXiv v3 (2019-06-11); J. Intelligent Manufacturing online 2019 / volume 31 (2020); this file is the author preprint | 17 |
| 04 | [DINOv3](<04. DINOv3/04-DINOv3-2025.pdf>) | Existing project PDF; copied unchanged | 67 |
| 05 | [Rethinking Transfer Learning for Industrial Inspection: DINOv3 vs. ImageNet Pretraining Across RGB and X-ray Tasks](<05. Industrial-Transfer-Learning/05-DINOv3-vs-ImageNet-Industrial-Transfer-2026.pdf>) | arXiv v1; accepted CVPR 2026 VISION workshop | 9 |
| 06 | [SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection](<06. SuperSimpleNet/06-SuperSimpleNet-2024.pdf>) | arXiv v2; accepted ICPR 2024 | 20 |
| 07 | [S³DNet: Support-Driven Spatial–Semantic Decoupling for Few-Shot Strip-Steel Surface Defect Segmentation](<07. S3DNet/07-S3DNet-Few-Shot-Strip-Steel-Segmentation-2026.pdf>) | Mathematics 14(20), 3730 (2026), publisher PDF | 26 |
| 08 | [CM-UNetv2: An Enhanced Semantic Segmentation Model for Precise PCB Defect Detection and Boundary Restoration](<08. CM-UNetv2/08-CM-UNetv2-PCB-Defect-Segmentation-2025.pdf>) | Sensors 25(16), 4919 (2025), original open-access publisher PDF | 25 |
| 09 | [Towards Total Recall in Industrial Anomaly Detection](<09. PatchCore/09-PatchCore-2022.pdf>) | CVPR 2022 proceedings PDF | 11 |
| 10 | [Rethinking Continual Anomaly Detection on the Edge: Benchmarking Under Realistic Industrial Conditions](<10. DINOSaur/10-DINOSaur-Continual-Anomaly-Detection-2026.pdf>) | arXiv v2 (2026-06-29); stored author preprint; formal publication status not independently verified | 20 |
| 11 | [Siamese network with change awareness for surface defect segmentation in complex backgrounds](<11. CADNet/11-CADNet-Reference-Defect-Segmentation-2025.pdf>) | Scientific Reports 15, 11814 (2025), publisher PDF | 19 |
| 12 | [Explainable AI reveals Clever Hans effects in unsupervised learning models](<12. Clever-Hans-Effects/12-Clever-Hans-Effects-2025.pdf>) | Nature Machine Intelligence 7, 412-422 (2025), publisher PDF | 14 |
| 13 | [Sanity Checks for Saliency Maps](<13. Sanity-Checks/13-Sanity-Checks-for-Saliency-Maps-2018.pdf>) | NeurIPS 2018 proceedings PDF | 11 |
| 14 | [Explainable Visual Anomaly Detection via Concept Bottleneck Models](<14. CONVAD/14-CONVAD-2025-preprint-v3-2026.pdf>) | arXiv v3, revised 2026-10-12; cite as preprint | 27 |
| 15 | [AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2](<15. AnomalyDINO/15-AnomalyDINO-2025.pdf>) | WACV 2025 proceedings PDF | 11 |
| 16 | [Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection](<16. Dinomaly/16-Dinomaly-2025.pdf>) | CVPR 2025 proceedings PDF | 11 |
| 17 | [The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection](<17. MVTec-AD2/17-MVTec-AD2-Advanced-Anomaly-Detection-2025.pdf>) | arXiv v1, submitted 2025-03-27; preserved as the 2025 preprint, not the 2026 journal version | 21 |
| 18 | [Detecting Defects that Matter: An Application-Driven Benchmark for Anomaly Detection in Manufacturing and Retail Logistics (VAND 4.0 Challenge)](<18. VAND4-Application-Benchmark/18-VAND4-Application-Driven-Benchmark-2026.pdf>) | arXiv v1, submitted 2026-10-03 | 14 |
| 19 | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](<19. ViT/19-ViT-2020.pdf>) | Existing project PDF; copied unchanged | 22 |

## Why these five additions

| Order | Addition | Specific gap addressed |
| ---: | --- | --- |
| 03 | Tabernik et al., supervised surface-defect segmentation | Adds a direct industrial application foundation beyond backbone papers: sample budget, annotation precision and segmentation-to-decision design. It does not establish PCB performance. |
| 08 | CM-UNetv2 | Adds single-image, mask-supervised PCB context alongside CADNet's paired-reference input. Its patch dataset and reported mIoU are not the local PCB1 protocol. |
| 10 | DINOSaur | Adds the closest position-indexed normal-memory and neighborhood-distance precedent. Position proximity is not structural semantics; its normal-only continual/image-level evaluation does not validate PCB mask segmentation or explanation. |
| 17 | MVTec AD 2 | Adds subtle defects, hidden ground truth and lighting-shift evaluation; read AU-PRO integration limits separately from deployment thresholds. |
| 18 | VAND4 organizer benchmark | Adds a recent application-driven hidden-test assessment. The industrial challenge ranks thresholded SegF1 under two lighting conditions, not the same protocol as AU-PRO or local PCB1 Dice. |

## Mechanism and claim boundaries

- **07 S3DNet:** uses support images and masks for episodic steel-defect segmentation. The frozen encoder/local-spatial idea is a close precedent, but removing its support mechanism changes the task and must be disclosed.
- **10 DINOSaur:** normal patches are stored by position; the query's coordinate neighborhood filters candidates; minimum feature-space L2 distance gives a patch score. The stored v2 PDF was checked at Sections 4.3-4.4, equations (6)-(9). Its v2 abstract qualifies the result under geometric drift; do not substitute the older landing-page claim of superiority on all five protocols. Publication status is not independently verified; the conference-like layout is not evidence of acceptance.
- **11 CADNet:** requires a corresponding defect-free reference. VisA does not provide a paired golden board for every test image. Its decoder uses change-aware processing; it should not be described as generic cross-attention without checking the actual computation.
- **12-14 explanations:** a task mask, a retrieved reference and a physical manufacturing cause are different outputs. Save the references actually used in scoring and test their relevance; plausible heatmaps alone do not establish faithfulness.
- **15-16:** the stored AnomalyDINO and Dinomaly PDFs are the formal WACV/CVPR papers. Their referenced supplementary appendices are not silently added to these files. Later code ports are not the original paper backbone.
- **17:** the local AD2 file is explicitly the 2025 arXiv v1 used for protocol checks, not the 2026 journal version. **18** is the 2026 organizer preprint. Scores from different versions, tasks and supervision settings must not be combined into one ranking.

## Sources and checks

The [download manifest](download-manifest.json) records every original title, source, stored version, purpose, page count, size and SHA-256, and the previous order/path for the 14 renamed originals. All five additions passed PDF signature/EOF checks, full-page parsing, first-page title/version checks and visual inspection of the first/final pages. Original text and page content were not edited.

The new PDFs have concise ASCII filenames beginning with their reading number. Existing translations retain `full_translation_zh.md`; their original translation dates and text remain intact. The table below separates primary landing pages from the actual PDF acquisition source.

| Order | Primary landing page | PDF source / local reuse |
| ---: | --- | --- |
| 01 | [Paper](https://arxiv.org/abs/2207.14315) | [Existing project PDF](<../261029/1. VisA/SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation.pdf>) |
| 02 | [Paper](https://arxiv.org/abs/2105.15203) | [Existing project PDF](<../261029/2. SegFormer/SegFormer- Simple and Efficient Design for Semantic Segmentation with Transformers.pdf>) |
| 03 | [Paper](https://arxiv.org/abs/1903.08536v3) | [Original PDF](https://arxiv.org/pdf/1903.08536v3) |
| 04 | [Paper](https://arxiv.org/abs/2508.10104) | [Existing project PDF](<../261029/4. DINOv3/DINOv3.pdf>) |
| 05 | [Paper](https://arxiv.org/abs/2605.23472) | [Original PDF](https://arxiv.org/pdf/2605.23472v1) |
| 06 | [Paper](https://arxiv.org/abs/2408.03143) | [Original PDF](https://arxiv.org/pdf/2408.03143v2) |
| 07 | [Paper](https://www.mdpi.com/2227-7390/14/20/3730) | [Original PDF](https://www.mdpi.com/2227-7390/14/20/3730/pdf) |
| 08 | [Paper](https://www.mdpi.com/1424-8220/25/16/4919) | [Original PDF](https://www.mdpi.com/1424-8220/25/16/4919/pdf) |
| 09 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) | [Original PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf) |
| 10 | [Paper](https://arxiv.org/abs/2605.24251v2) | [Original PDF](https://arxiv.org/pdf/2605.24251v2) |
| 11 | [Paper](https://www.nature.com/articles/s41598-025-94733-4) | [Original PDF](https://www.nature.com/articles/s41598-025-94733-4.pdf) |
| 12 | [Paper](https://www.nature.com/articles/s42256-025-01000-2) | [Original PDF](https://www.nature.com/articles/s42256-025-01000-2.pdf) |
| 13 | [Paper](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) | [Original PDF](https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf) |
| 14 | [Paper](https://arxiv.org/abs/2511.20088) | [Original PDF](https://arxiv.org/pdf/2511.20088v3) |
| 15 | [Paper](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html) | [Original PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf) |
| 16 | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html) | [Original PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.pdf) |
| 17 | [Paper](https://arxiv.org/abs/2503.21622v1) | [Original PDF](https://arxiv.org/pdf/2503.21622v1) |
| 18 | [Paper](https://arxiv.org/abs/2610.04392v1) | [Original PDF](https://arxiv.org/pdf/2610.04392v1) |
| 19 | [Paper](https://arxiv.org/abs/2010.11929) | [Existing project PDF](<../261029/3. ViT/An Image is Worth 16x16 Words- Transformers for Image Recognition at Scale.pdf>) |
