# Machine Vision Research Route Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive the previous Video Agent research route without altering its meaningful artifacts, then rebuild the active personal planning records around advisor-aligned industrial machine vision inspection.

**Architecture:** Treat the dated archive as an immutable historical snapshot and `personal-planning/` as the single active source of truth. Perform exact-path moves for old folders, preserve hashes for binary artifacts, then create focused English Markdown records for positioning, research scope, roadmap, and current execution status.

**Tech Stack:** Git, PowerShell, Markdown, SHA-256 file hashes, repository-local `AGENTS.md` instructions.

**Spec:** `docs/superpowers/specs/2026-09-15-machine-vision-route-restructure-design.md`

## Global Constraints

- Archive root: `archive/2026-09-15-video-agent-route/`.
- Use English names for all new files and folders.
- Write all new Markdown records in English.
- Move old route material instead of deleting it.
- Do not alter the contents of archived PDF or DOCX files.
- Remove the meaningless tracked Word lock file instead of archiving it.
- Keep all recursive move sources and destinations inside the workspace.
- Do not modify `travel-planning/`.
- Do not present Video Agent, Video-RAG, long-video retrieval, multi-Agent orchestration, or the old Topic A through Topic E set as current plans.
- Do not select a final defect type, product, dataset, model, or thesis novelty before experiments and advisor confirmation.

## Target File Structure

```text
archive/
  2026-09-15-video-agent-route/
    README.md
    binary-sha256.csv
    video_agent_lab/
    personal-planning/
      AGENTS.md
      current-positioning.md
      learning-status-2026-06.md
      2026-06/
      2026-summer/
      career-development-plan/
      career-path-reports/
      thesis-feasibility-analysis/

personal-planning/
  AGENTS.md
  current-positioning.md
  machine-vision-inspection-research-direction.md
  machine-vision-inspection-roadmap-2026-2028.md
  current-learning-status.md
```

### Task 1: Create the Archive and Preserve the Previous Route

**Files:**
- Create: `archive/2026-09-15-video-agent-route/README.md`
- Create: `archive/2026-09-15-video-agent-route/binary-sha256.csv`
- Move: `video_agent_lab/`
- Move: `personal-planning/2026-06/`
- Move: `personal-planning/2026-summer/`
- Move: `personal-planning/career-development-plan/`
- Move: `personal-planning/career-path-reports/`
- Move: `personal-planning/thesis-feasibility-analysis/`
- Move: `personal-planning/learning-status-2026-06.md`
- Move: `personal-planning/current-positioning.md`
- Copy: `personal-planning/AGENTS.md`
- Remove: `personal-planning/thesis-feasibility-analysis/~$dustrial-vision-agent-alignment-and-feasibility-report.docx`

**Interfaces:**
- Consumes: The current repository tree and the approved restructure spec.
- Produces: A dated, indexed archive with integrity records and no active Video Agent workspace.

- [ ] **Step 1: Verify the workspace and all move targets**

Run a PowerShell read-only check that resolves the workspace root, verifies each source exists, and confirms every resolved source begins with the workspace path followed by a directory separator. Verify the proposed archive root resolves beneath the same workspace before creating it.

Run:

```powershell
$workspaceRoot = [IO.Path]::GetFullPath((Get-Location).Path)
$sources = @(
  'video_agent_lab',
  'personal-planning/2026-06',
  'personal-planning/2026-summer',
  'personal-planning/career-development-plan',
  'personal-planning/career-path-reports',
  'personal-planning/thesis-feasibility-analysis',
  'personal-planning/learning-status-2026-06.md',
  'personal-planning/current-positioning.md',
  'personal-planning/AGENTS.md'
)
$sources | ForEach-Object {
  $resolved = (Resolve-Path -LiteralPath $_).Path
  if (-not $resolved.StartsWith($workspaceRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Source outside workspace: $resolved"
  }
  [pscustomobject]@{ Source = $_; Resolved = $resolved }
}
$archiveRoot = [IO.Path]::GetFullPath((Join-Path $workspaceRoot 'archive/2026-09-15-video-agent-route'))
if (-not $archiveRoot.StartsWith($workspaceRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Archive outside workspace: $archiveRoot"
}
```

Expected: Nine verified sources and one archive path beneath the workspace.

- [ ] **Step 2: Remove the meaningless Word lock file**

Resolve the exact lock-file path, verify it is inside the workspace, confirm its size is 162 bytes, then remove only that file with `Remove-Item -LiteralPath`. Do not use a wildcard.

Expected: The lock file no longer exists; the real industrial vision feasibility report remains present.

- [ ] **Step 3: Create the archive directories and integrity manifest**

Create the archive root and its `personal-planning/` child. Before moving binaries, enumerate every `.docx` and `.pdf` under the move sources, calculate SHA-256 hashes, store workspace-relative original paths, byte lengths, and hashes in `binary-sha256.csv`.

Expected CSV columns:

```text
OriginalPath,Length,SHA256
```

- [ ] **Step 4: Write the archive README**

Create the archive README in English with these exact sections:

```markdown
# Video Agent Research Route Archive

## Archive Decision

This archive preserves the research and career-planning route used before 2026-09-15. After discussion with the graduate advisor, the active research direction changed to industrial machine vision inspection for detecting defects in components and manufactured products.

## Why the Direction Changed

The new route is more concrete, starts from foundational visual algorithms, supports reproducible experiments, and aligns directly with advisor guidance. It is expected to build durable computer vision and engineering ability.

## Archived Scope

This archive contains the complete former `video_agent_lab` and the personal planning documents that treated Video Agent, long-video understanding, Video-RAG, or Agent engineering as the active research route.

## Status

These files are historical references. They are not the current research plan and should not guide future work unless explicitly revisited.

## Integrity

`binary-sha256.csv` records the original paths, sizes, and SHA-256 hashes of archived PDF and DOCX files.
```

- [ ] **Step 5: Move the archived material**

Move `video_agent_lab/` to the archive root. Move the seven listed personal-planning folders and files to the archive's `personal-planning/` directory. Copy the old `personal-planning/AGENTS.md` into the archive snapshot, leaving the active copy available for Task 3.

Expected: No listed old route folder remains in active `personal-planning/`; `video_agent_lab/` no longer exists at repository root.

- [ ] **Step 6: Verify archive integrity**

For every row in `binary-sha256.csv`, map `video_agent_lab/...` to `archive/2026-09-15-video-agent-route/video_agent_lab/...` and `personal-planning/...` to `archive/2026-09-15-video-agent-route/personal-planning/...`. Recalculate length and SHA-256 and require an exact match.

Expected: Zero missing files, zero length mismatches, and zero hash mismatches.

- [ ] **Step 7: Commit the archive move**

```powershell
git add -- archive video_agent_lab personal-planning
git commit -m "docs: archive previous video agent research route"
```

### Task 2: Create the Active Machine Vision Planning Records

**Files:**
- Create: `personal-planning/current-positioning.md`
- Create: `personal-planning/machine-vision-inspection-research-direction.md`
- Create: `personal-planning/machine-vision-inspection-roadmap-2026-2028.md`
- Create: `personal-planning/current-learning-status.md`

**Interfaces:**
- Consumes: The approved spec, durable personal background from the archived planning context, and the advisor's new direction.
- Produces: Four focused active planning records with non-overlapping responsibilities.

- [ ] **Step 1: Write the new current positioning**

Create `current-positioning.md` with these sections and assertions:

- `Current Decision`: record the 2026-09-15 transition after advisor discussion.
- `Why This Direction Fits`: explain the value of starting from foundations and building practical experimental ability.
- `Transferable Strengths`: retain C++, embedded systems, video engineering, system thinking, and engineering discipline.
- `Current Positioning`: define the developing role as applied machine vision and industrial visual inspection engineer.
- `Boundaries`: state that Video Agent and long-video retrieval are archived, not active priorities.
- `Near-Term Focus`: prioritize OpenCV, imaging fundamentals, classical algorithms, deep-learning baselines, datasets, evaluation, and defect-focused experiments.

- [ ] **Step 2: Write the research-direction record**

Create `machine-vision-inspection-research-direction.md` with:

- Decision context and advisor alignment.
- Problem scope covering surface defects, missing parts, incorrect assembly, positional or dimensional deviation, and appearance anomalies.
- Method families: classical image processing, classification, detection, segmentation, and anomaly detection.
- System factors: lighting, optics, camera setup, image quality, annotation, robustness, latency, and deployment.
- Research-question template: defect definition, available supervision, baseline, proposed improvement, evaluation, and failure analysis.
- Thesis boundaries: no fixed dataset, product, model, or novelty claim before experiments and advisor confirmation.
- First investigation outputs: dataset survey, baseline comparison, experiment template, and advisor discussion notes.

- [ ] **Step 3: Write the 2026-2028 roadmap**

Create `machine-vision-inspection-roadmap-2026-2028.md` with five phases:

1. Image processing and OpenCV foundations.
2. Classical industrial vision inspection.
3. Deep-learning foundations for visual inspection.
4. Industrial defect datasets and baseline experiments.
5. Thesis convergence and deployable engineering work.

For every phase, specify learning goals, concrete artifacts, completion criteria, and scope-control rules. Include recurring monthly review questions and require code, visual outputs, metrics, and failure cases rather than reading-only completion.

- [ ] **Step 4: Write the current learning status and twelve-week cycle**

Create `current-learning-status.md` with:

- Existing foundation and transferable experience.
- Gaps in image processing, optics, industrial imaging, dataset work, model training, and evaluation.
- Time constraint of approximately two hours per workday and six hours per weekend day.
- Weeks 1-4: image operations, filtering, thresholding, edges, morphology, contours, connected components, and small OpenCV exercises.
- Weeks 5-8: lighting and imaging experiments, template matching, geometric measurement, and one classical defect-detection baseline.
- Weeks 9-12: one public industrial defect dataset, a simple deep-learning baseline, metric reporting, and a classical-versus-learning comparison.
- End-of-cycle outputs: repository code, sample results, experiment log, failure cases, and questions for the advisor.

- [ ] **Step 5: Validate active-document consistency**

Run:

```powershell
rg -n -i "Video-RAG|long-video|long video|multi-agent|Topic A|Topic B|Topic C|Topic D|Topic E" personal-planning
```

Expected: Matches appear only where a current document explicitly states that the former route is archived, never as an active recommendation or milestone.

- [ ] **Step 6: Commit the active planning records**

```powershell
git add -- personal-planning/current-positioning.md personal-planning/machine-vision-inspection-research-direction.md personal-planning/machine-vision-inspection-roadmap-2026-2028.md personal-planning/current-learning-status.md
git commit -m "docs: establish machine vision inspection research route"
```

### Task 3: Update Project and Personal Agent Context

**Files:**
- Modify: `AGENTS.md`
- Modify: `personal-planning/AGENTS.md`

**Interfaces:**
- Consumes: The four active planning records from Task 2.
- Produces: Future-agent instructions that consistently treat machine vision inspection as the active research direction.

- [ ] **Step 1: Update the root project context**

Preserve the user background and English Markdown naming rule. Update the working scope to prioritize industrial machine vision, defect detection, image processing, classical vision, deep-learning visual inspection, C++, deployment, paper analysis, experiments, and learning records. State that the previous Video Agent route is archived and should not be treated as active unless the user explicitly reopens it.

- [ ] **Step 2: Rewrite the personal planning context**

Retain durable background, personality patterns, work-study constraints, advisor relationship, and planning principles. Remove the old active sections centered on Video Agent research, long-video event indexing, Agent engineering, and Topic A through Topic E.

Add current-direction sections that require future advice to:

- Start from machine vision and image-processing foundations.
- Connect algorithms to concrete inspection problems and imaging conditions.
- Prefer small reproducible experiments over premature novelty.
- Compare classical and deep-learning baselines.
- Track metrics, failure modes, and deployment constraints.
- Preserve C++ and engineering experience as a later deployment advantage.
- Keep thesis selection aligned with advisor feedback.

- [ ] **Step 3: Check instruction consistency**

Run searches across active context files for the previous route terms and manually inspect every match. Confirm the archive is described only as history and the new direction is unambiguous.

- [ ] **Step 4: Commit the context update**

```powershell
git add -- AGENTS.md personal-planning/AGENTS.md
git commit -m "docs: update context for industrial machine vision research"
```

### Task 4: Perform Final Repository Verification

**Files:**
- Verify: `archive/2026-09-15-video-agent-route/**`
- Verify: `personal-planning/**`
- Verify unchanged: `travel-planning/**`

**Interfaces:**
- Consumes: Completed archive and active planning structure.
- Produces: Evidence that the restructure is complete, internally consistent, and lossless for meaningful historical artifacts.

- [ ] **Step 1: Verify the final file structure**

Run `rg --files archive personal-planning travel-planning` and compare the result with the target structure. Confirm all archived material is present and only the five active personal-planning files remain.

- [ ] **Step 2: Re-run binary integrity verification**

Recalculate every archived binary hash from `binary-sha256.csv`. Require exact equality for path mapping, length, and SHA-256.

- [ ] **Step 3: Verify travel records are unchanged**

Run:

```powershell
git diff HEAD~3 -- travel-planning
```

Expected: No output.

- [ ] **Step 4: Run formatting and consistency checks**

Run:

```powershell
git diff --check HEAD~3..HEAD
rg -n -i "current.*Video Agent|primary.*Video-RAG|Topic [A-E]" AGENTS.md personal-planning
git status --short
```

Expected: No whitespace errors; no active old-route assertions; clean working tree.

- [ ] **Step 5: Review the final history**

Run:

```powershell
git log -4 --oneline
```

Expected commits, newest first:

1. `docs: update context for industrial machine vision research`
2. `docs: establish machine vision inspection research route`
3. `docs: archive previous video agent research route`
4. `docs: plan machine vision route restructure`

The already committed design specification remains immediately before the implementation-plan commit.
