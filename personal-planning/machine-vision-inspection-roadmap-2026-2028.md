# Industrial Machine Vision Inspection Roadmap 2026 to 2028

Date: 2026-09-15

## Goal

By graduation in 2028, I should be able to design, implement, evaluate, and explain an industrial visual inspection solution. The expected portfolio is a connected body of work rather than isolated tutorials:

`Image acquisition -> preprocessing -> defect representation -> baseline -> evaluation -> failure analysis -> improvement -> deployment`

The thesis should emerge from repeated experiments and advisor feedback. It should not be selected only from paper titles or current model popularity.

## Working Constraints

- Approximately two study hours per workday and six hours per weekend day.
- Part-time graduate study must remain compatible with full-time C++ work.
- Practical output and visible feedback are important for sustained motivation.
- Expensive model training, large-scale annotation, and premature infrastructure should be avoided.
- Python and OpenCV may be used for fast experiments; C++ becomes important when performance, integration, or deployment is part of the question.

## Phase 1 Image Processing and OpenCV Foundations

Target period: September to November 2026.

### Learning Goals

- Understand image representation, channels, bit depth, histograms, sampling, interpolation, and coordinate systems.
- Practice filtering, enhancement, thresholding, edges, morphology, contours, connected components, and geometric transforms.
- Learn to visualize each intermediate stage and explain why a parameter changes the result.
- Establish repeatable experiment and note-taking habits.

### Required Artifacts

- A small image-processing exercise repository.
- At least eight focused experiments, each with input, code, intermediate images, output, and observations.
- A classical defect-detection mini-project using a controlled set of images.
- A parameter-sensitivity note covering lighting, threshold, kernel size, noise, and resolution.

### Completion Gate

Given a simple visible defect, I can propose a processing chain, implement it, display intermediate results, choose a threshold from evidence, and explain its failure cases.

## Phase 2 Classical Industrial Vision Inspection

Target period: December 2026 to February 2027.

### Learning Goals

- Understand lighting geometry, lens and sensor considerations, exposure, depth of field, calibration, and resolution requirements.
- Practice template matching, image registration, difference imaging, feature matching, dimensional measurement, and rule-based decisions.
- Learn how fixtures, regions of interest, and controlled imaging simplify the algorithm.

### Required Artifacts

- A lighting and image-quality experiment with controlled changes.
- A missing-part, alignment, or dimensional-deviation prototype.
- A short inspection specification defining acceptable and defective samples.
- A false-positive and false-negative analysis rather than only successful screenshots.

### Completion Gate

I can distinguish an imaging problem from an algorithm problem and can build a rule-based baseline whose operating conditions and failure boundaries are explicit.

## Phase 3 Deep Learning Foundations for Inspection

Target period: March to June 2027.

### Learning Goals

- Understand training, validation, test splits, loss, optimization, overfitting, regularization, transfer learning, and class imbalance.
- Build simple classification, detection, or segmentation baselines according to the label type.
- Learn dataset inspection, annotation quality checks, augmentation, metric selection, and reproducible configuration.

### Required Artifacts

- One transfer-learning classification experiment.
- One localization experiment using detection or segmentation.
- A reproducible training configuration and environment record.
- Per-class metrics, confusion analysis, qualitative results, and a failure-case collection.

### Completion Gate

I can train and evaluate a baseline without data leakage, interpret the main metrics, reproduce the run, and explain whether the problem comes from data, labels, optimization, or task formulation.

## Phase 4 Industrial Defect Datasets and Baseline Research

Target period: July to December 2027.

### Learning Goals

- Compare supervised inspection and normal-only anomaly detection.
- Understand small defects, low contrast, domain shift, few-shot conditions, open-set anomalies, and threshold calibration.
- Build a stable benchmark around one narrow problem rather than collecting disconnected demos.

### Required Artifacts

- A documented dataset survey and selection decision.
- At least two credible baselines using the same split and evaluation protocol.
- Controlled experiments for one or two important factors such as lighting, resolution, augmentation, feature scale, or threshold strategy.
- A research memo that converts observed failures into candidate research questions.

### Completion Gate

The selected problem has sufficient data, a reproducible baseline, a meaningful unresolved failure mode, and an evaluation protocol that the advisor accepts as a basis for thesis work.

## Phase 5 Thesis Convergence and Engineering Deployment

Target period: January to June 2028, adjusted to the actual graduate schedule.

### Learning Goals

- Narrow the thesis to one inspection problem, one primary method contribution, and a controlled evaluation.
- Run ablation, robustness, error, and cost analysis.
- Package the method into a reproducible inference pipeline.
- Evaluate whether C++, ONNX Runtime, OpenVINO, TensorRT, or an edge platform is relevant to the final problem.

### Required Artifacts

- Advisor-approved thesis question and evidence-backed proposal.
- Versioned data split, experiment configurations, and result tables.
- Baseline and proposed-method comparisons.
- Failure-case taxonomy and limitations.
- A runnable demonstration and engineering documentation.
- Thesis chapters that accurately separate evidence, interpretation, and future work.

### Completion Gate

The work is reproducible, the contribution is narrower than the entire system, improvements are supported by controlled evidence, limitations are explicit, and the demonstration reflects the thesis claims.

## Core Ability Stack

### Imaging

Lighting, optics, camera settings, calibration, resolution, field of view, depth of field, and acquisition stability.

### Algorithms

Classical image processing, geometry, classification, detection, segmentation, and anomaly detection.

### Experimentation

Dataset inspection, labeling, splits, metrics, ablations, robustness testing, reproducibility, and failure analysis.

### Engineering

Python prototyping, OpenCV, PyTorch, version control, configuration management, visualization, C++ integration, and deployment profiling.

### Research Communication

Paper reading, evidence extraction, advisor discussion notes, experiment reports, technical writing, and honest limitation statements.

## Monthly Execution System

Each month should contain:

- One primary technical question.
- One runnable artifact.
- One metric or controlled comparison.
- One failure-case review.
- One concise research note.
- One decision about what to continue, change, or stop.

Reading without an experiment does not count as completion when the topic is experimentally testable.

## Scope Control

- Prefer one complete small pipeline to several incomplete frameworks.
- Establish a simple baseline before adding architectural complexity.
- Change one major experimental factor at a time.
- Keep the data split fixed when comparing methods.
- Stop adding methods when the current failure has not been clearly defined.
- Ask whether better imaging or data would solve the problem before increasing model complexity.
- Use C++ when it adds engineering value, not as a constraint on early algorithm learning.

## Monthly Review Questions

- What did I run rather than only read?
- Which intermediate result helped me understand the algorithm?
- Which defects were missed, and under what conditions?
- Which false alarms would be unacceptable in a real inspection line?
- Did the result depend on an accidental data split or imaging condition?
- What is the simplest next experiment that can distinguish two explanations?
- What evidence should I bring to the next advisor discussion?
