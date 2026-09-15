# Machine Vision Research Route Restructure Design

Date: 2026-09-15

## Purpose

The active research direction will change from video stream and AI Agent research to industrial machine vision inspection. The new direction follows the advisor's guidance and focuses on detecting defects in components and manufactured products through visual methods.

This is a deliberate shift toward a more concrete and foundational research path. It should help build durable skills in image processing, computer vision experiments, defect detection, engineering evaluation, and deployment. The previous route remains useful as historical context, but it must no longer appear as the current plan.

## Chosen Organization Strategy

Use a complete archive and rebuild the active personal planning records.

All previous Video Agent research files and route-specific personal planning files will be preserved under:

`archive/2026-09-15-video-agent-route/`

Archived files will remain unchanged whenever possible. The active planning area will contain only the new machine vision inspection direction and the personal background that is still relevant.

## Archive Scope

Move the entire existing `video_agent_lab/` directory into the archive.

Move these route-specific personal planning records into `archive/2026-09-15-video-agent-route/personal-planning/`:

- `personal-planning/2026-06/`
- `personal-planning/2026-summer/`
- `personal-planning/career-development-plan/`
- `personal-planning/career-path-reports/`
- `personal-planning/thesis-feasibility-analysis/`
- `personal-planning/learning-status-2026-06.md`
- The current pre-transition version of `personal-planning/current-positioning.md`

Preserve a snapshot of the pre-transition `personal-planning/AGENTS.md` in the archive before rewriting the active copy.

Create an archive README that records:

- The archive date.
- The reason for the research-direction change.
- The scope of archived material.
- A clear statement that the material is retained for historical reference and is no longer the active plan.

The `travel-planning/` directory will not be changed.

## Active Personal Planning Files

After archiving, rebuild the active `personal-planning/` area around five files.

### AGENTS.md

Retain the durable personal background, working constraints, advisor context, and collaboration principles. Remove active instructions that prioritize long-video understanding, Video-RAG, event indexing, multi-Agent systems, or the previous Topic A through Topic E thesis candidates.

Set industrial machine vision inspection as the current research direction. Future planning should connect practical visual inspection problems with the user's C++, engineering, and graduate-study background.

### current-positioning.md

Record the new current positioning as an applied machine vision and industrial visual inspection engineer in development.

The document will explain that the transition is based on advisor discussion and is considered beneficial because it starts from foundational algorithms, creates concrete experiment loops, and builds transferable computer vision ability.

It will retain relevant strengths such as C++, embedded systems, video engineering, system thinking, and engineering discipline. It will remove Agent engineering and long-video retrieval as active positioning themes.

### machine-vision-inspection-research-direction.md

Define the research domain, initial problem boundaries, and thesis exploration principles.

The initial scope will cover:

- Visual inspection of components and manufactured products.
- Surface defects, missing parts, incorrect assembly, dimensional or positional deviation, and appearance anomalies.
- Classical image-processing baselines and deep-learning baselines.
- Classification, object detection, segmentation, and anomaly detection where appropriate.
- Data quality, lighting, optics, imaging setup, annotation, evaluation, robustness, and deployment constraints.

The document will avoid assuming a final defect type, dataset, or model before practical investigation and advisor confirmation.

### machine-vision-inspection-roadmap-2026-2028.md

Define a staged learning and research route:

1. Image processing and OpenCV foundations.
2. Classical machine vision inspection methods.
3. Deep learning foundations for visual inspection.
4. Industrial defect detection experiments and dataset practice.
5. Thesis topic convergence, controlled evaluation, and engineering deployment.

Each stage will contain concrete outputs rather than reading-only goals. Outputs will include notebooks or programs, visualized intermediate results, small datasets or annotations, baseline comparisons, failure-case records, and periodic research notes.

### current-learning-status.md

Record the transition date, current foundation, current gaps, available study time, and the next twelve-week execution focus.

The first execution cycle will emphasize foundational algorithms and small reproducible experiments instead of immediately pursuing architectural novelty or a complex thesis system.

## Content Removal Rules

Remove these ideas from active planning documents:

- Long-video semantic event indexing as the primary thesis direction.
- Video-RAG and long-video question answering as the main research route.
- Frame-log diagnostic Agents as a thesis candidate.
- Multi-Agent orchestration as a near-term research objective.
- Topic A through Topic E from the previous thesis-space analysis.
- June and summer execution schedules that have expired.
- Career comparisons that no longer represent the selected path.

Do not erase this information from archived files.

## New Research Route Principles

The active route will use the following principles:

- Start with image formation and basic algorithms before model selection.
- Treat lighting, optics, camera configuration, and data quality as part of the vision solution.
- Establish simple and explainable baselines before adding deep models.
- Select models according to the actual defect form and available supervision.
- Measure precision, recall, false-positive rate, false-negative rate, localization or segmentation quality, latency, and robustness where relevant.
- Keep experiments small, reproducible, and connected to visible outputs.
- Align thesis scope with the advisor before committing to a specific product, component, defect type, or novelty claim.
- Preserve C++ and deployment knowledge as an engineering advantage, without forcing every early learning task into C++.

## File Operation Safety

- Use moves rather than deletions for previous route material.
- Resolve and verify all source and destination paths before recursive moves.
- Keep every recursive move inside the workspace.
- Check hashes for archived PDF and DOCX files before and after moving them.
- Do not modify archived DOCX or PDF contents.
- Do not alter travel records.
- Remove the tracked Word temporary lock file from the active project if it is present; it is not a meaningful archive artifact.

## Verification

The restructure is complete when:

- The previous `video_agent_lab/` exists only inside the dated archive.
- All previous route-specific personal planning artifacts are present in the archive.
- The active `personal-planning/` directory contains the five current files defined above.
- Active planning documents consistently identify industrial machine vision inspection as the current direction.
- Active documents do not present Video Agent research as a current goal.
- Archived binary files match their pre-move hashes.
- `travel-planning/` is unchanged.
- Git shows only the intended moves, rewritten context files, new active planning files, and removal of the meaningless Word lock file.

## Expected Outcome

The project will clearly distinguish two periods:

- A preserved historical Video Agent exploration period.
- A current industrial machine vision inspection period focused on foundational learning, reproducible experiments, and advisor-aligned thesis development.

This organization keeps earlier thinking available without allowing it to dilute the new direction.
