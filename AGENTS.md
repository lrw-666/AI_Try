# Project Agent Context

This workspace supports personal development planning, graduate research, academic paper analysis, technical experiments, and long-term learning records.

## Personal Background

The user is a computer science professional who graduated three years ago.

The user's undergraduate degree was in Computer Science from an ordinary second-tier university in China. After graduation, the user worked in Shenzhen for more than one year as an embedded software development engineer, focusing on FreeRTOS engine development.

The user later enrolled in a part-time Computer Science master's program. The academic advisor works in computer vision and digital media. To balance work and graduate study, the user moved to Beijing and took a C++ application-development role involving front-end and back-end design for video plugins.

## Current Research Direction

The current research and learning entry is `docs/research/thesis-research-and-study-guide-zh.md` (updated 2026-11-09). The thesis title is `《PCB外观缺陷检测方法研究》`. The two research goals are improving detection accuracy and explainability. PCB feature adaptation and reference scoring are candidate means, not the goals themselves; the work must address PCB inspection rather than stop at dataset transfer comparisons or retrieval optimization. The explainability work should evaluate explanation reliability and its cost to the detection task, without requiring an IoU increase by definition. Use the guide for the latest scope, candidate mechanisms, learning priorities, and legacy-document migration. The dated 2026-11-07 literature and defense-Q&A files and the 2026-11-09 industrial/PCB review are supporting appendices. PCB-specific adaptation may support an application contribution when the problem, comparisons, gains and limits are demonstrated; changing datasets alone does not establish methodological novelty. Local-detail branches and spatially constrained references remain candidates, with S3DNet and DINOSaur as direct precedents. Retain the supervised high-shot task and PCB1 baseline; preselect another PCB category for later validation. Preserve all SegFormer experiment artifacts. Historical guides may receive current-entry notes and link repairs without changing their experimental or instructional content. The user's 2026-11-09 request authorizes renumbering/renaming the `docs/paper/261107` collection and adding verified original PDFs while preserving existing PDF bytes and translation bodies.

Runtime status is authoritative only for checks performed on this current E-drive machine; checks on another computer do not establish its state. On 2026-11-09, `E:\桌面\AI\segformer_resources` and its project interpreter were present, and the baseline best checkpoint, model/processor configurations, split CSVs, and resource manifests matched the archived acceptance identities. This verifies the listed files and their identities; it does not claim a fresh model forward or full validation rerun. Before making a substantive choice that changes research objectives, supervision protocol, technical route, or workload, explain the viable options and tradeoffs and ask the user to choose. The current pass establishes the broad direction without fixing modules or an implementation plan.

As of 2026-09-15, the active graduate research direction is industrial machine vision inspection. The practical goal is to detect defects in components and manufactured products through visual methods.

Future research and learning work should begin with image-processing and computer-vision foundations, then progress through classical inspection algorithms, deep-learning baselines, industrial datasets, controlled evaluation, and deployment.

The former Video Agent, Video-RAG, and long-video research route is preserved under `archive/2026-09-15-video-agent-route/`. Treat it as historical material rather than an active plan unless the user explicitly reopens it.

## Working Scope

Future work in this project may include:

- Personal career and growth planning related to machine vision and applied computer vision.
- Image-processing, OpenCV, industrial imaging, defect detection, classification, detection, segmentation, and anomaly-detection study.
- Academic paper notes, literature reviews, dataset surveys, and research summaries.
- Reproducible experiments with code, visualized intermediate results, metrics, and failure cases.
- C++ integration, inference optimization, edge deployment, and engineering documentation when relevant.
- Thesis planning based on experimental evidence and advisor feedback.
- Reflection notes about work, study, technical depth, and long-term positioning.

## Language Rule

All future Markdown records in this project should be written in English unless the user explicitly requests otherwise for a specific file or task.

Unless the user explicitly requests otherwise, all future folders and files should use English names.

## Collaboration Rule

Help the user understand and solve problems from the user's own perspective, including background, temperament, constraints, motivations, and long-term goals.

Prefer concrete experiments and visible outputs over abstract plans. Separate confirmed evidence from assumptions, and do not claim research novelty or industrial applicability without appropriate evaluation.

## Document and Git Storage

The Git repository is `AI_Try`. Keep project source, research and design Markdown, stage learning documents, configurations, split manifests, and small portable acceptance summaries inside this repository. Do not keep the only copy of a working document in a sibling resource, output, or temporary directory.

For SegFormer, `../segformer_resources` is the local runtime area for the Conda environment, extracted datasets, downloaded models, caches, temporary files, and bulk run output. Keep those large or machine-specific resources out of Git. Preserve small reproducible summaries, inventories, and selected learning figures under `work/SegFormer/reports` when useful.

Do not relocate an existing Conda prefix or runtime directory merely to place it under the repository. Treat a resource migration as separate work that requires updating paths and verifying the environment. Repository folder placement and Git tracking are separate decisions; inspect the staged file list and sizes before each requested push.
