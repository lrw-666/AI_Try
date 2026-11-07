# Literature grounding and thesis research assessment

Assessment date: 2026-11-07 (Asia/Shanghai).

Current research and study entry: [Chinese master guide](thesis-research-and-study-guide-zh.md). This file is a detailed literature appendix. The latest two-work design draft and study priorities are maintained in that guide; compute limits are implementation conditions rather than a core research objective.

This is an evidence review and a set of research recommendations, not an approved implementation plan. Existing experiments and plans have not been changed. The review uses representative primary literature; it does not establish an exhaustive novelty search. Chinese discussion is provided in the accompanying chat; this durable Markdown record follows the repository's English-language rule.

## 1. Assessment

The direction is academically grounded: pretrained visual representations have an established role in industrial defect detection and segmentation. The defensible question is how to adapt those representations to a specified PCB inspection task, with measurable localization, false-alarm, resource, and evidence-quality constraints. Simply replacing the encoder with DINOv3 is an experimental baseline rather than a demonstrated contribution.

The advisor's latest reply supports industrial visual inspection, proposes the title “PCB exterior defect detection methods,” suggests accuracy and explainability, and prefers multiple substantive works. It does not reject segmentation or a particular backbone. It also does not establish that the previous plan already satisfies thesis requirements.

Recommended task boundary: supervised binary defect segmentation of board-level RGB exterior images, initially using the VisA PCB high-shot protocol. Explainability concerns evidence used by the predictor; it does not establish the physical manufacturing cause of a defect.

## 2. Local experimental grounding

The verified SegFormer-B0 baseline uses PCB1, one seed, and train/validation/test sets of 530/132/442 images. Test contains 402 normal and 40 anomalous images. Test global defect IoU is 0.68334; pixel recall is 0.73896; anomaly-image macro IoU is 0.52486. Three anomalous images have zero true-positive pixels, including two empty predictions. Thirty-two normal images have at least one false-positive pixel.

Sources: [acceptance record](../../work/SegFormer/reports/2026-11-01-stage12-13-acceptance.md), [experiment card](../../work/SegFormer/reports/pcb1_b0_v2_seed42/experiment-card.md), and [current research and study guide](thesis-research-and-study-guide-zh.md).

These results establish a reproducible baseline and unresolved errors. They do not establish that small defects are the principal cause, that resizing caused the misses, that DINO will help, or that the predictor meets a production acceptance specification. DINO training, proposed adaptations, and explanation experiments remain unexecuted.

## 3. Closest supervised literature

| Primary reference | Actual task and evidence | Relevance and limits |
| --- | --- | --- |
| Tabernik et al., Segmentation-based Deep-learning Approach for Surface-defect Detection, Journal of Intelligent Manufacturing, online 2019 / volume 31 (2020), 759-776. [Publisher](https://link.springer.com/article/10.1007/s10845-019-01476-x), [full text](https://arxiv.org/html/1903.08536v2) | Pixel-supervised industrial surface segmentation followed by an image decision network; evaluates limited defect samples and spatial-resolution choices on KolektorSDD. | Establishes the supervised segmentation-to-inspection route and annotation constraints. The proposed network is randomly initialized; a pretrained DeepLab comparator must not be confused with their method. Pretraining is a candidate advantage, not a universal guarantee. |
| Zou et al., SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation, ECCV 2022. [Paper](https://arxiv.org/html/2207.14315v1), [official data/protocol](https://github.com/amazon-science/spot-diff) | Studies pretrained representations for defect recognition and localization; introduces VisA and normal-only, high-shot supervised, and few-shot supervised protocols. Its motivation explicitly distinguishes sensitivity to local defects from invariance to global changes. | A direct foundation for the present task, rather than merely a dataset citation. The downstream supervised segmentation uses masks. Its SPD pretraining procedure is different from using an existing DINO checkpoint. Historical novelty claims in its introduction describe 2022, not the current field. |
| Rolih, Fucka and Skocaj, SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection, ICPR 2024. [Paper](https://arxiv.org/abs/2408.03143) | Uses pretrained features with trainable adaptation and defect prediction, accommodating available abnormal labels. | Demonstrates that pretrained features plus a trainable adaptor/head is already established. Its supervised evaluation is on KSDD2 and SensumSODF; its VisA experiment uses a different supervision regime from the user's current baseline. |
| Rolih et al., No Label Left Behind: A Unified Surface Defect Detection Model for all Supervision Regimes, Journal of Intelligent Manufacturing. [Publisher](https://link.springer.com/article/10.1007/s10845-025-02680-8), [preprint](https://arxiv.org/abs/2508.19060) | Extends the SuperSimpleNet study to multiple annotation regimes. | Published online in 2025, assigned to the 2026 volume. Useful when selecting a supervised industrial comparator, subject to reproducing an appropriate mode on the same split. Do not copy published results from another dataset or regime as a direct baseline score. |
| Gharbage et al., Rethinking Transfer Learning for Industrial Inspection: DINOv3 vs. ImageNet Pretraining Across RGB and X-ray Tasks, CVPR 2026 VISION Workshop. [Full text](https://arxiv.org/html/2605.23472v1), [conference record](https://openaccess.thecvf.com/content/CVPR2026W/VISION26/html/Gharbage_Rethinking_Transfer_Learning_for_Industrial_Inspection_DINOv3_vs._ImageNet_Pretraining_CVPRW_2026_paper.html) | Compares supervised ImageNet and DINOv3-distilled ConvNeXt under frozen and fully finetuned transfer. Frozen features do not show a consistent advantage; RGB finetuning is more favorable, while X-ray transfer differs. | The closest recent motivation for asking when adaptation matters. It studies ConvNeXt, not the proposed ViT-S/16. Results use validation sets and one seed; RarePlanes is a localization proxy. Avoid copying prose gain magnitudes without checking Table 2. This is a workshop paper, not a main-conference paper. |

## 4. Adjacent foundation-model and anomaly-detection literature

| Primary reference | Actual protocol and mechanism | How it informs this thesis |
| --- | --- | --- |
| Roth et al., Towards Total Recall in Industrial Anomaly Detection (PatchCore), CVPR 2022. [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) | Normal-only patch-feature memory and nearest-normal comparison. | Establishes local reference evidence and pretrained feature reuse. It does not validate mask-supervised training or explain an unrelated segmentation branch. |
| Damm et al., AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2, WACV 2025. [Paper](https://arxiv.org/html/2405.14529v3), [conference PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf) | Frozen DINOv2 with few normal reference images and patch nearest neighbors; no downstream parameter training. | Supports DINO features for industrial localization and examining input resolution. Its few-shot setting counts normal references, unlike a supervised defect-mask budget. Formal publication is WACV 2025; the initial preprint is from 2024. |
| Guo et al., Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection, CVPR 2025. [Conference paper](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html), [full text](https://arxiv.org/html/2405.14325v5) | Normal-only training with frozen DINOv2-register features, a bottleneck, and reconstruction decoder. | China-affiliated work from Tsinghua, Beijing Institute of Technology and Shanghai Jiao Tong. Confirms domestic foundation-feature research; it is not supervised PCB segmentation. A later repository DINOv3 port must not be presented as the CVPR paper's original backbone. |
| Yuan et al., AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration, 2025 preprint. [Full text](https://arxiv.org/html/2509.14084v2) | DINOv3 and CLIP text representations with lightweight cross-modal adaptation and anomaly-aware calibration. Source-domain training includes anomaly masks; the target dataset is unseen. | Sun Yat-sen University affiliation is shown in the preprint. Demonstrates that DINOv3 industrial adaptation already exists. Target-domain zero-shot does not mean no anomaly labels anywhere in training. Conference publication was not independently verified in this review. |
| S3DNet: Support-Driven Spatial-Semantic Decoupling for Few-Shot Strip-Steel Surface Defect Segmentation, Mathematics 14(20):3730, 2026. [Publisher](https://www.mdpi.com/2227-7390/14/20/3730) | Publisher-indexed text describes frozen DINOv3 semantics and lightweight support-conditioned/spatial decoding for episodic 1/5-shot steel-defect segmentation. | A close novelty-check reference for frozen DINOv3 plus lightweight supervised adaptation. Different product, dataset, and episodic supervision. Direct publisher fetch was rate-limited; indexed primary text was available. Do not treat its numerical results as a mandatory reproducible baseline without code and protocol inspection. |

These publications show a shared international research trajectory. A useful literature review groups methods by supervision, representation, adaptation, and evaluation; geography alone does not explain methodological differences.

China-affiliated supervised work also includes Yao et al., Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection (BGAD), CVPR 2023. [Official paper page](https://openaccess.thecvf.com/content/CVPR2023/html/Yao_Explicit_Boundary_Guided_Semi-Push-Pull_Contrastive_Learning_for_Supervised_Anomaly_Detection_CVPR_2023_paper.html). It uses limited known anomalies to improve decision discrimination while addressing bias toward seen anomalies. This supports studying the value and risks of available defect labels, but its seen/unseen anomaly objective must not be conflated with closed-set segmentation.

A more directly PCB-related domestic example is Liu et al., Siamese Network with Change Awareness for Surface Defect Segmentation in Complex Backgrounds, Scientific Reports 15:11814 (2025), from the University of Electronic Science and Technology of China, South China Normal University, and Sichuan Changhong. [Primary PDF](https://www.nature.com/articles/s41598-025-94733-4.pdf), [author code](https://github.com/HATFormer/CADNet). It uses a tested image and a corresponding defect-free reference, including supervised PKU-Market-PCB experiments. It motivates reference differences and complex-background handling, but requires a different input condition from the current single-image task. Pretrained initialization was not verified in this review; do not label it established pretrained adaptation evidence.

Guo et al., CM-UNetv2: An Enhanced Semantic Segmentation Model for Precise PCB Defect Detection and Boundary Restoration, Sensors 25(16):4919 (2025), from Xi'an University of Technology, studies supervised MeiweiPCB segmentation with spatial/semantic and boundary modules. [Publisher](https://www.mdpi.com/1424-8220/25/16/4919). It supports the relevance of PCB detail and background problems. Its other dataset KWSD2 is remote-sensing water segmentation, not the industrial KSDD2 dataset. Pretrained initialization was not established here.

## 5. Why investigating decision evidence is substantive

Kauffmann et al., Explainable AI Reveals Clever Hans Effects in Unsupervised Learning Models, Nature Machine Intelligence 7, 412-422 (2025), investigates spurious evidence in representation learning and industrial anomaly models. Its industrial experiments connect learned evidence and preprocessing sensitivity to performance changes under plausible shifts. [Primary article](https://www.nature.com/articles/s42256-025-01000-2). This supports auditing prediction strategies; it does not establish that the user's SegFormer or DINO model has the same failure.

Stropeni et al., Explainable Visual Anomaly Detection via Concept Bottleneck Models, first posted in 2025 and revised in October 2026, proposes CONVAD with concept-level and localization branches, interventions, and a user study. [Full text](https://arxiv.org/html/2511.20088v3). It provides direct precedent for separating spatial localization from semantic evidence. Its limitations explicitly discuss branch-specific faithfulness and fusion. Treat it here as a verified preprint; this assessment does not assert an independently verified conference status.

Adebayo et al., Sanity Checks for Saliency Maps, NeurIPS 2018, shows why visual plausibility alone is insufficient for explanation quality. [Paper](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html). Randomization is one diagnostic, not a proof of explanation correctness or a sufficient ranking criterion.

Three distinctions matter: a defect mask is a task output; an explanation concerns the predictor's actual evidence; a physical fault diagnosis concerns the manufacturing process. None automatically supplies the others. Confidence calibration and uncertainty are also distinct from explainability.

## 6. Practical need and cautious problem formulation

The practical reasoning is: defect annotations are limited and costly; pretrained features offer reusable information; their original objectives and spatial granularity may mismatch fine defect localization; adaptation might improve the relevant information; evidence analysis can test whether improvements rely on suitable signals.

This chain is supported by prior work but does not prove a new method. The local experiment establishes errors requiring investigation, not their causes. A defensible proposed question is:

> Under a fixed supervised PCB protocol, how can pretrained features be adapted to reduce localization misses and normal-structure false alarms, while exposing and testing the evidence used in defect decisions?

The current high-shot protocol should remain clearly identified. Claiming improved label efficiency requires a controlled experiment with varied training-label budgets, a fixed evaluation population, and repeated sampling. Forty-eight anomalous optimization images do not automatically make the experiment an official few-shot benchmark.

## 7. Two related candidate works

Work A: error-driven feature adaptation. Establish frozen DINO plus a simple head, then test one justified change in spatial-detail preservation, feature selection, or lightweight parameter adaptation. Use train/validation diagnostics to choose the principal bottleneck. Include a conventional adaptation baseline so the contribution is not merely the presence of an adapter. SegFormer remains a useful system baseline, but cross-backbone differences cannot isolate the causal effect of self-supervised pretraining.

Work B: reference-grounded defect evidence. A possible direction is position/structure-aware normal-reference comparison that directly contributes to the final defect score. Investigate whether it reduces confusion between repeated normal PCB structures and defects while producing inspectable reference evidence. Compare against ordinary unrestricted reference retrieval and the base predictor. This is a proposed hypothesis, not an established research gap or a selected design.

If retrieval is attached only after an independent segmentor has predicted, the retrieved image explains that retrieval operation, not automatically the segmentor. If multiple scoring branches are fused, measure and disclose each branch's contribution and the portion of the final decision the explanation covers. Existing nearest-neighbor, prototype, fusion, adapter, and attention mechanisms cannot themselves be claimed as new.

The two works may share data, features and evaluation. They need distinct questions and evidence, not two unrelated large models. Whether they meet the advisor's workload and novelty expectations must be assessed using concrete designs.

## 8. Evidence required to make the proposed contributions persuasive

1. Preserve original-image/group split integrity. Train and tune on train/validation only. Historical PCB1 test results and failure cases have already been viewed; disclose this rather than calling future use completely blind. Use an additional untouched evaluation set when practical.
2. Separate system-level comparison from mechanism attribution. For the latter, keep the DINO backbone, data, input resolution, head, and optimization conditions controlled while varying the proposed mechanism. Full finetuning is a useful reference if affordable, not an unmeasured mandatory promise.
3. Report original-resolution defect IoU/Dice, per-image performance, and defined defect-region recall. Define image/board decisions before reporting board misses or false rejection. Select thresholds and postprocessing on validation data.
4. Diagnose scale, contrast and structure-related errors explicitly. Define grouping rules before evaluating a new method. An input-resolution improvement alone does not prove a new architecture or adaptation mechanism.
5. For evidence evaluation, compare matched normal-reference replacement of high-evidence regions against equal-area random/normal-region controls. Include normal-to-normal replacement controls for paste artifacts. Such tests assess model responses to interventions, not manufacturing causality.
6. Test plausible nuisance changes only where labels remain valid. Measure both prediction and evidence stability. Synthetic perturbation robustness does not establish real factory distribution coverage.
7. Repeat key comparisons and report variation, resource use and end-to-end latency. A separately trained second PCB category demonstrates wider evaluation, not automatic cross-board generalization.

Real-IAD, CVPR 2024, provides a China-affiliated large-scale multi-view industrial benchmark. [Primary paper](https://arxiv.org/abs/2403.12580). MVTec AD 2 explicitly includes lighting-condition shifts. [Primary paper](https://arxiv.org/abs/2503.21622). These motivate evaluation beyond a single clean split; they are not evidence that the proposed method has passed those conditions, nor an instruction to expand this thesis to every benchmark.

## 9. Claims to avoid and replacements

| Unsupported claim | Defensible replacement |
| --- | --- |
| DINOv3 is newer, so it will improve PCB detection. | Test whether a particular representation and adaptation improves specified errors under controlled conditions. |
| Pretrained representation transfer to industrial defects is an unexplored field. | Prior work is substantial; the proposed scope is a specific supervised PCB limitation and a validated response. |
| SegFormer versus DINO proves supervised versus self-supervised pretraining. | This is a comparison of two complete transfer configurations with multiple differing factors. |
| A heatmap or retrieved normal example explains the entire model. | Establish which computation the evidence belongs to and test its relation to the actual prediction. |
| A higher public-benchmark score demonstrates industrial compliance. | Limit conclusions to measured operating conditions; actual acceptance requires product and process specifications and appropriate data. |

## 10. Focused reading sequence

First read SPD for the direct task motivation and protocol, Gharbage et al. for the limits of frozen transfer, and SuperSimpleNet for an established supervised feature-adaptation pipeline. Then read AnomalyDINO and Dinomaly for DINO-based industrial feature use, carefully preserving their normal-only distinction. Read Kauffmann et al. and CONVAD before specifying the second work. Use SegFormer and DINOv3 original papers for architecture and feature interfaces rather than as sole evidence of industrial need.

The current next deliverable is a 6-8 minute proposal presentation following the already-established direction. The Chinese master guide contains the two-work design draft and a ten-slide outline; another advisor discussion is not a prerequisite. Results, novelty, industrial readiness, and final method effectiveness remain subject to evidence.
