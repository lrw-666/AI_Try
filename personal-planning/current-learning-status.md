# Current Learning Status and Twelve Week Plan

Date: 2026-09-15

## Current State

I am beginning a new research route in industrial machine vision inspection after discussion with my advisor. I am not a complete programming beginner, but I need to rebuild my computer-vision knowledge from practical foundations.

My previous Video Agent and long-video research plans have been archived. The current priority is to understand basic visual algorithms, imaging conditions, experiment design, and defect-detection workflows before pursuing a narrow thesis contribution.

## Existing Foundation

- Computer Science undergraduate background.
- Previous machine-learning and deep-learning coursework.
- C++ application-development experience.
- Earlier embedded and FreeRTOS engineering experience.
- Current work involving video plugins and visual data paths.
- Awareness of software architecture, code quality, documentation, and testing.

## Main Gaps

- Insufficient hands-on practice with basic image-processing algorithms.
- Limited understanding of industrial lighting, optics, cameras, calibration, and measurement.
- Little experience building and debugging a complete defect-detection pipeline.
- Limited dataset, annotation, training, evaluation, and failure-analysis practice.
- No current baseline that can be used to discuss a narrow thesis question with the advisor.

## Time Budget

- Workdays: approximately two hours per day.
- Weekend days: approximately six hours per day.
- Nominal weekly capacity: approximately twenty-two hours.
- Planning rule: reserve at least twenty percent of the time for review, debugging, and unfinished work.

## Twelve Week Objective

Complete a first evidence-based machine vision learning loop:

`Understand an operation -> implement it -> visualize its effect -> apply it to a simple defect -> measure the result -> record failures`

The cycle should end with one classical inspection baseline, one public-dataset learning baseline, and a comparison that can support the next advisor discussion.

## Weeks 1 to 4 Image Processing Foundations

### Topics

- Image loading, channels, data types, histograms, cropping, resizing, and interpolation.
- Grayscale and color spaces.
- Noise, Gaussian and median filtering, sharpening, and contrast enhancement.
- Global, Otsu, and adaptive thresholding.
- Sobel, Scharr, Laplacian, and Canny edges.
- Erosion, dilation, opening, closing, and morphological gradients.
- Contours, connected components, bounding rectangles, area, perimeter, and shape features.

### Outputs

- Eight small OpenCV exercises with saved intermediate images.
- One note explaining when each operation helps and when it damages defect evidence.
- One reusable script structure for input, processing stages, visualization, parameters, and output.

## Weeks 5 to 8 Classical Inspection Baseline

### Topics

- Lighting direction, diffuse versus directional lighting, reflections, exposure, blur, and resolution.
- Regions of interest, background normalization, alignment, and image registration.
- Template matching, difference imaging, geometric measurement, and rule-based decisions.
- Threshold calibration and false-positive or false-negative analysis.

### Outputs

- A small controlled image set, preferably self-collected or clearly licensed.
- One missing-part, position-deviation, surface-defect, or dimensional inspection baseline.
- Visual output that shows every major processing stage.
- A test table containing normal samples, defect samples, decisions, errors, and observations.

## Weeks 9 to 12 Learning Based Baseline

### Topics

- Survey one public industrial defect dataset and confirm its task and license.
- Inspect class distribution, image resolution, annotations, and common data-quality issues.
- Establish one simple transfer-learning, detection, segmentation, or anomaly-detection baseline according to the labels.
- Use a fixed data split and record the environment, configuration, seed, and metrics.
- Compare the learning-based result with the classical baseline on at least one shared failure dimension.

### Outputs

- Dataset card and split description.
- Runnable training or inference configuration.
- Metric table and qualitative result samples.
- Failure cases grouped by defect size, contrast, location, imaging condition, or class.
- A short comparison of classical and learning-based methods.
- Two or three questions for the next advisor discussion.

## Weekly Rhythm

- Monday: review the current question and define the week's smallest deliverable.
- Tuesday to Thursday: implement or run experiments and save intermediate results.
- Friday: organize metrics, failures, and unresolved questions.
- Saturday: complete the main experiment and improve reproducibility.
- Sunday: write the weekly note, clean the artifact, and choose the next step.

## Progress Tracking

| Period | Planned Evidence | Current Status | Completion Evidence |
|---|---|---|---|
| Weeks 1-4 | Eight OpenCV exercises and operation notes | Not started | Code, images, and notes |
| Weeks 5-8 | Classical inspection baseline | Not started | Dataset, pipeline, decisions, and error table |
| Weeks 9-12 | Public-dataset learning baseline | Not started | Configuration, metrics, qualitative results, and failures |
| End of cycle | Advisor discussion package | Not started | Comparison memo and candidate research questions |

## Risk Controls

- If a week falls behind, reduce the number of algorithms but complete one full experiment.
- If a model cannot run on available hardware, use a smaller baseline before considering new compute.
- If results are unstable, inspect data and imaging before adding model complexity.
- If a public dataset does not match the intended industrial problem, treat it as a learning exercise rather than a thesis commitment.
- If reading starts replacing experiments, require a runnable artifact before adding another paper.

## End of Cycle Review

- Which visual operations can I now explain from their effect on an image?
- Can I distinguish imaging failures from algorithm failures?
- Did I complete a reproducible classical baseline?
- Did I complete a reproducible learning-based baseline?
- Which failure mode is technically interesting and practically relevant?
- What evidence supports the next thesis discussion with my advisor?
