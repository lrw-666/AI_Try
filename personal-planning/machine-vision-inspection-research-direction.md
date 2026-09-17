# Industrial Machine Vision Inspection Research Direction

Date: 2026-09-15

## Decision Context

The active graduate research direction is industrial machine vision inspection. This decision follows discussion with my advisor and replaces the earlier Video Agent and long-video retrieval route.

The direction addresses a practical manufacturing problem: components and finished products may contain visible defects, missing or incorrect parts, assembly deviations, or abnormal appearances, and a visual system should detect these conditions reliably.

The immediate objective is not to lock in a thesis title. It is to build the foundations needed to identify a feasible inspection problem, reproduce credible baselines, and discuss a narrower research question with the advisor using experimental evidence.

## Problem Scope

Potential inspection targets include:

- Surface defects such as scratches, cracks, stains, dents, pits, bubbles, discoloration, and texture irregularities.
- Missing, extra, reversed, misplaced, or incorrectly assembled components.
- Positional, angular, spacing, alignment, shape, or dimensional deviations.
- Appearance anomalies that are difficult to enumerate exhaustively in advance.
- Small defects whose visibility depends strongly on resolution, lighting, contrast, or viewpoint.

This list defines the domain but does not select the final thesis target. The final target should be chosen according to data access, annotation cost, defect frequency, industrial value, experimental feasibility, and advisor guidance.

## Complete Vision Solution

An industrial vision system is not only a neural network. The complete chain is:

`Defect definition -> lighting and optics -> image acquisition -> calibration and preprocessing -> region of interest -> algorithm -> decision rule -> evaluation -> deployment and monitoring`

Errors can originate at any point in this chain. Poor lighting or insufficient resolution cannot always be repaired by a larger model. A useful study must therefore record imaging conditions and data quality, not only model parameters.

## Method Families

### Classical Image Processing

- Grayscale and color-space conversion.
- Denoising, smoothing, sharpening, and contrast enhancement.
- Global, adaptive, and color-based thresholding.
- Edge detection and morphological operations.
- Contours, connected components, geometric features, and blob analysis.
- Template matching, feature matching, registration, and difference imaging.
- Camera calibration, perspective correction, and pixel-to-physical measurement.

Classical methods should be the first baseline when the imaging environment is controlled and the defect has clear geometric, color, or intensity characteristics.

### Supervised Deep Learning

- Image classification when a whole image or crop has one label.
- Object detection when the defect requires a bounding box or the component position matters.
- Semantic or instance segmentation when the defect shape and area matter.
- Transfer learning when available labeled data is limited.

### Industrial Anomaly Detection

- One-class or normal-only learning when defect types are rare or incomplete.
- Feature-memory, reconstruction, or distillation approaches as baselines.
- Image-level and pixel-level anomaly evaluation.

Anomaly detection is promising for open-set defects, but it should not be treated as the default solution before checking whether supervised labels and simple rules are already sufficient.

## Research Question Template

Every candidate topic should answer these questions:

1. What object, component, or product is inspected?
2. What counts as a defect, and what does not?
3. What imaging setup makes the defect visible?
4. What data and annotations are available?
5. What is the simplest credible baseline?
6. Which failure mode remains after the baseline?
7. What change is proposed, and why should it address that failure?
8. Which metrics reflect the real cost of false alarms and missed defects?
9. Does the improvement hold across batches, lighting changes, viewpoints, and defect subtypes?
10. Can another person reproduce the result from the recorded code, data split, parameters, and environment?

## Evaluation Principles

Use metrics according to the task:

- Classification: precision, recall, F1, confusion matrix, ROC-AUC where appropriate, and per-class results.
- Detection: precision-recall curves, mAP, small-object performance, and missed-defect analysis.
- Segmentation: IoU, Dice, pixel-level precision and recall, and defect-area sensitivity.
- Anomaly detection: image-level and pixel-level AUROC or AUPRO where supported, plus threshold-dependent false-positive and false-negative rates.
- System performance: latency, throughput, memory use, image-acquisition stability, and robustness under controlled disturbances.

Always preserve qualitative failure cases. A high aggregate score can hide systematic misses on small, low-contrast, or rare defects.

## Thesis Boundaries

- Do not train a large foundation model from scratch.
- Do not create a large industrial dataset without confirmed access and annotation support.
- Do not combine classification, detection, segmentation, anomaly detection, 3D vision, and deployment into one thesis unless the scope is demonstrably manageable.
- Do not describe ordinary model replacement or hyperparameter tuning as research novelty.
- Do not assume a public-dataset gain will transfer directly to a production line.
- Do not fix the final product, defect type, dataset, model, or novelty claim before baseline experiments and advisor confirmation.

## Candidate Starting Experiments

These are investigation exercises rather than committed thesis topics:

- A controlled surface-defect pipeline using grayscale, filtering, thresholding, morphology, and connected components.
- A missing-component or position-deviation task using template matching, registration, contours, and geometric rules.
- A supervised defect baseline on a small public dataset using transfer learning.
- A normal-only anomaly-detection baseline on an industrial benchmark such as MVTec AD or VisA.
- A comparison of a classical pipeline and a learning-based method under lighting, noise, blur, and small-defect changes.

## Outputs for the Next Advisor Discussion

- A table comparing several industrial inspection problem types.
- A dataset survey with licenses, sizes, labels, defect types, and limitations.
- At least one runnable classical baseline with visualized intermediate stages.
- At least one simple learning-based baseline with a fixed data split and metrics.
- A failure-case summary explaining what the baselines cannot handle.
- Two or three narrow candidate thesis questions grounded in those failures.
