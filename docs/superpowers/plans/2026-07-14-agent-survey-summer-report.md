# Agent Survey Summer Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a visually verified Chinese Design Report that turns the July 14-August 31, 2026 summer period into an intensive Agent survey-paper program and ends with a defensible two-year thesis/work research-topic decision process.

**Architecture:** Clone the retained Design Report DOCX, distill its exact visual contract, and populate a copied document with structured Chinese content backed by workspace context and verified paper metadata. Keep content, sources, document construction, and validation separable so the report can be regenerated and audited without changing the retained template.

**Tech Stack:** Bundled Codex Python 3, `python-docx`, OOXML package inspection, Documents plugin render/audit scripts, LibreOffice headless rendering, JSON source catalog.

## Global Constraints

- The report body is Chinese; persistent Markdown files and all file/folder names are English.
- The retained Design Report reference remains unchanged and controls the visual system.
- The schedule begins on July 14, 2026 and ends on August 31, 2026.
- The first week is shorter; full weeks may use the user's available 22 hours without assuming full-time study.
- The report must lead to one primary topic and one backup topic, not merely a reading list.
- Candidate topics are evaluated rather than predetermined.
- Technical claims use original paper or publisher pages; peer-reviewed and preprint status are distinguished where verified.
- Do not invent performance results, industrial outcomes, or exact costs.
- Do not modify or revert unrelated dirty-worktree files.
- Deliver only the final DOCX to the user; render outputs and template evidence are internal QA artifacts.

## File Structure

- Create: `personal-planning/2026-summer/agent-survey-sources.json` - verified literature metadata and reading priority.
- Create: `personal-planning/2026-summer/report_content.py` - structured Chinese report data, schedule, matrices, templates, and references.
- Create: `personal-planning/2026-summer/build_agent_survey_report.py` - copies the retained template and builds the final DOCX.
- Create: `personal-planning/2026-summer/validate_agent_survey_report.py` - content, date, placeholder, source, and structural checks.
- Create: `personal-planning/2026-summer/agent-survey-and-research-topic-report.docx` - final deliverable.
- Create: `personal-planning/2026-summer/_report-work/artifact.md` - task-local template contract.
- Create: `personal-planning/2026-summer/_report-work/reference-render/` - reference page renders.
- Create: `personal-planning/2026-summer/_report-work/final-render-<N>/` - unique final render iteration directories.

---

### Task 1: Distill the Retained Design Report Template

**Files:**
- Create: `personal-planning/2026-summer/_report-work/artifact.md`
- Create: `personal-planning/2026-summer/_report-work/reference-render/`
- Create: `personal-planning/2026-summer/_report-work/template-style-evidence.json`

**Interfaces:**
- Consumes: retained reference `C:/Users/10746/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-design-report/assets/reference.docx`
- Produces: an `artifact.md` contract recording SHA-256, page/section system, styles, slot map, package-preservation rules, and permitted cloned body patterns.

- [ ] **Step 1: Load the bundled document runtime and define exact paths**

Use the dependency loader and record:

```text
PYTHON_BIN=C:/Users/10746/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe
SKILL_DIR=C:/Users/10746/.codex/plugins/cache/openai-primary-runtime/documents/26.709.11516/skills/documents
REFERENCE_DOCX=C:/Users/10746/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-design-report/assets/reference.docx
TMP_DIR=C:/Users/10746/Desktop/专业书籍/study/AI_Try/personal-planning/2026-summer/_report-work
```

- [ ] **Step 2: Render and inspect every reference page**

Run:

```powershell
& $PYTHON_BIN "$SKILL_DIR/render_docx.py" $REFERENCE_DOCX --output_dir "$TMP_DIR/reference-render"
```

Expected: one PNG per reference page; cover, contents/body, tables, headers, and footers are visually accounted for.

- [ ] **Step 3: Collect structural and style evidence**

Run:

```powershell
& $PYTHON_BIN "$SKILL_DIR/scripts/section_audit.py" $REFERENCE_DOCX
& $PYTHON_BIN "$SKILL_DIR/scripts/style_lint.py" $REFERENCE_DOCX --json "$TMP_DIR/template-style-evidence.json"
& $PYTHON_BIN "$SKILL_DIR/scripts/heading_audit.py" $REFERENCE_DOCX
& $PYTHON_BIN "$SKILL_DIR/scripts/images_audit.py" $REFERENCE_DOCX
& $PYTHON_BIN "$SKILL_DIR/scripts/fields_report.py" $REFERENCE_DOCX
```

Expected baseline: 8.5 x 11 inch portrait pages, two sections, 1 inch margins, one cover image, one cover metadata table, one evidence table, and reusable Title/Heading 1/Heading 2/normal patterns.

- [ ] **Step 4: Write the exact template contract**

Record in `artifact.md`:

```markdown
# Design Report Template Contract

## Reference
- Absolute path
- SHA-256
- Render page count and evidence paths

## Page System
- Exact section sizes, margins, break types, and header/footer distances

## Typography and Components
- Title, metadata, contents label, Heading 1, Heading 2, body, callout, evidence table, source list, header, and footer properties

## Slot Map
- Cover title
- Cover subtitle
- Cover author/date
- Body start at the source "Contents" slot
- Reusable evidence-table pattern
- Appendix source-list pattern

## Preservation and Extension
- Preserve cover image, page geometry, styles, theme, numbering, relationships, and recurring chrome
- Remove source placeholder prose only from editable body slots
- Permit cloned Heading 1, Heading 2, normal, callout, and table patterns for additional body pages
```

- [ ] **Step 5: Verify the retained template is unchanged**

Recompute SHA-256 and compare it with `artifact.md`. Expected: exact match.

---

### Task 2: Verify and Structure the Literature Source Catalog

**Files:**
- Create: `personal-planning/2026-summer/agent-survey-sources.json`
- Create: `personal-planning/2026-summer/validate_agent_survey_report.py`

**Interfaces:**
- Consumes: original arXiv, OpenReview, ACM, Springer, or publisher pages.
- Produces: JSON array with `id`, `title`, `authors`, `year`, `status`, `url`, `layer`, `priority`, `reading_mode`, and `purpose`.

- [ ] **Step 1: Verify the core source set using original pages**

Include at minimum:

```text
2308.11432  A Survey on Large Language Model based Autonomous Agents
2302.07842  Augmented Language Models: a Survey
2402.02716  Understanding the Planning of LLM Agents: A Survey
2405.17935  Tool Learning with Large Language Models: A Survey
2404.13501  A Survey on the Memory Mechanism of Large Language Model based Agents
2503.16416  Survey on Evaluation of LLM-based Agents
2505.00753  A Survey on Large Language Model based Human-Agent Systems
2402.15116  Large Multimodal Agents: A Survey
2312.17432  Video Understanding with Large Language Models: A Survey
2502.08826  Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation
2201.08071  Temporal Sentence Grounding in Videos: A Survey and Future Directions
2402.01680  Large Language Model based Multi-Agents: A Survey of Progress and Challenges
2406.00252  Towards Rationality in Language and Multimodal Agents: A Survey
2510.17491  Empowering Real-World: A Survey on the Technology, Practice, and Evaluation of LLM-driven Industry Agents
2508.17692  LLM-based Agentic Reasoning Frameworks: A Survey from Methods to Scenarios
10.1145/3794858  Generalizability of Large Language Model-Based Agents: A Comprehensive Survey
```

- [ ] **Step 2: Add a selective representative-paper queue**

Add method/benchmark papers only where they help test a survey taxonomy, including Agent foundations, tool use, long-video understanding, Video-RAG, temporal grounding, evidence retrieval, and industrial visual diagnosis. Mark each as `selective` rather than `deep` reading.

- [ ] **Step 3: Write catalog validation**

Implement:

```python
def validate_sources(path: Path) -> list[str]:
    """Return validation errors for missing fields, duplicate IDs/URLs, invalid priorities, or unsupported status labels."""
```

Allowed values:

```python
ALLOWED_STATUS = {"peer-reviewed", "accepted", "preprint", "status-unverified"}
ALLOWED_PRIORITY = {"core", "targeted", "reference"}
ALLOWED_READING_MODE = {"deep", "selective", "lookup"}
```

- [ ] **Step 4: Run the source validator**

Run:

```powershell
& $PYTHON_BIN personal-planning/2026-summer/validate_agent_survey_report.py --sources personal-planning/2026-summer/agent-survey-sources.json
```

Expected: `PASS: source catalog is structurally valid`.

---

### Task 3: Author the Structured Chinese Report Content

**Files:**
- Create: `personal-planning/2026-summer/report_content.py`
- Modify: `personal-planning/2026-summer/validate_agent_survey_report.py`

**Interfaces:**
- Consumes: workspace planning context and `agent-survey-sources.json`.
- Produces: `REPORT_META`, `SECTIONS`, `WEEKLY_PLAN`, `TOPIC_CRITERIA`, `TOPIC_FAMILIES`, `RISKS`, `APPENDICES`, and `REFERENCES`.

- [ ] **Step 1: Define report metadata and required section keys**

```python
REPORT_META = {
    "title": "Agent 综述学习与两年研究选题设计报告",
    "subtitle": "2026 年 7 月 14 日至 8 月 31 日暑期强化计划",
    "author": "个人学习与研究规划",
    "date": "2026 年 7 月",
}

REQUIRED_SECTIONS = [
    "executive_summary",
    "background",
    "key_findings",
    "knowledge_architecture",
    "summer_program",
    "reading_method",
    "topic_funnel",
    "two_year_roadmap",
    "risks",
    "recommendations",
    "appendix",
]
```

- [ ] **Step 2: Write the main narrative**

Cover every approved design point: career insight, practical AI-assisted learning, Agent versus model/workflow, innovation and commercial value, industrial video landing, personal prototype cost structure, and the user's applied AI/video systems positioning.

- [ ] **Step 3: Write the exact seven-week plan**

Each weekly record must include:

```python
{
    "dates": "2026-07-20 to 2026-07-26",
    "theme": "Planning, tool learning, reflection, and memory",
    "deep_reading": [...],
    "selective_reading": [...],
    "concept_questions": [...],
    "practice": [...],
    "deliverables": [...],
    "topic_funnel_action": "...",
    "completion_gate": "...",
}
```

Week 1 uses July 14-19; Weeks 2-7 use the approved date ranges; August 31 is a separate finalization record.

- [ ] **Step 4: Write the topic decision system**

Use a 100-point weighted score:

```python
TOPIC_CRITERIA = [
    ("research_clarity", 15),
    ("data_and_baselines", 15),
    ("personal_resource_feasibility", 15),
    ("advisor_cv_alignment", 10),
    ("cpp_video_work_alignment", 15),
    ("evaluation_quality", 10),
    ("industrial_value", 10),
    ("two_year_staging", 5),
    ("model_change_resilience", 5),
]
```

Add hard gates: a topic is rejected if it lacks obtainable data, a reproducible baseline, a measurable evaluation protocol, or a feasible first prototype within three months.

- [ ] **Step 5: Add appendices and reusable templates**

Include the survey catalog, selective queue, paper-note template, weekly review checklist, candidate-topic brief, scorecard, advisor-discussion brief, and glossary.

- [ ] **Step 6: Extend and run content validation**

Validate:

```python
def validate_content() -> list[str]:
    """Check required sections, exact dates, weekly outputs, 100-point criteria, hard gates, source IDs, and forbidden placeholders."""
```

Forbidden text includes `Lorem`, `TBD`, `TODO`, `[Author]`, `[Month YYYY]`, and tool citation tokens.

Expected: `PASS: report content satisfies the approved specification`.

---

### Task 4: Build the DOCX From the Retained Template

**Files:**
- Create: `personal-planning/2026-summer/build_agent_survey_report.py`
- Create: `personal-planning/2026-summer/agent-survey-and-research-topic-report.docx`

**Interfaces:**
- Consumes: retained reference, `artifact.md`, `report_content.py`, and `agent-survey-sources.json`.
- Produces: final DOCX while preserving source-derived page setup, styles, cover image, tables, headers, footers, and recurring elements.

- [ ] **Step 1: Implement reference verification and working-copy creation**

```python
def verify_reference(reference: Path, expected_sha256: str) -> None:
    """Raise ValueError when the retained reference differs from artifact.md."""

def make_working_copy(reference: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference, output)
```

- [ ] **Step 2: Implement slot replacement and source-pattern reuse**

```python
def replace_cover_slots(doc: Document, meta: dict[str, str]) -> None: ...
def clear_editable_template_body(doc: Document) -> None: ...
def add_heading(doc: Document, text: str, level: int) -> Paragraph: ...
def add_body(doc: Document, text: str) -> Paragraph: ...
def add_callout(doc: Document, label: str, text: str) -> Paragraph: ...
def add_comparison_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[int]) -> Table: ...
def add_page_break(doc: Document) -> None: ...
```

`clear_editable_template_body` removes the source `Contents` through `Template note` placeholder content while retaining the cover section, cover image, section properties, styles, relationships, and recurring chrome. New body elements must clone documented source patterns; do not apply a generic preset.

- [ ] **Step 3: Compose report sections in Design Report order**

Use this top-level sequence:

```text
Executive summary
At a glance
Background and updated judgment
Key findings
Implications for the user's research and career
Agent knowledge architecture
Recommendations: intensive summer program
Topic funnel and two-year roadmap
Conclusion
Appendix
References
```

- [ ] **Step 4: Build tables with stable geometry**

Use the template evidence table pattern and the packaged table-geometry helper. Repeat header rows for multi-page tables. Do not allow a row to split when the row contains one week's complete plan.

- [ ] **Step 5: Build the first DOCX iteration**

Run:

```powershell
& $PYTHON_BIN personal-planning/2026-summer/build_agent_survey_report.py
```

Expected: final DOCX exists, is non-empty, opens with two sections, and preserves the reference cover image and page geometry.

---

### Task 5: Run Structural and Content Verification

**Files:**
- Modify: `personal-planning/2026-summer/validate_agent_survey_report.py`
- Verify: `personal-planning/2026-summer/agent-survey-and-research-topic-report.docx`

**Interfaces:**
- Consumes: final DOCX, source catalog, and approved specification.
- Produces: a zero-exit validation result or actionable errors.

- [ ] **Step 1: Implement complete visible-text extraction**

```python
def extract_visible_text(docx_path: Path) -> str:
    """Collect body paragraphs, table cells, headers, and footers."""
```

- [ ] **Step 2: Add final-document assertions**

Assert:

```text
The exact report title is present.
All seven weekly date ranges and August 31 are present.
All required knowledge layers are present.
The topic criteria sum to 100.
At least four candidate topic families are presented as candidates.
One-primary/one-backup selection instructions are present without a predetermined winner.
Every cited source ID exists in agent-survey-sources.json.
No template placeholder, Lorem text, TODO, or tool token remains.
The cover image and both sections remain present.
Page size and margins match artifact.md.
```

- [ ] **Step 3: Run validation**

Run:

```powershell
& $PYTHON_BIN personal-planning/2026-summer/validate_agent_survey_report.py --docx personal-planning/2026-summer/agent-survey-and-research-topic-report.docx
```

Expected: `PASS: final DOCX is structurally and textually valid`.

---

### Task 6: Render, Inspect, and Iterate Until Clean

**Files:**
- Create: `personal-planning/2026-summer/_report-work/final-render-1/`
- Create as needed: `personal-planning/2026-summer/_report-work/final-render-2/`, etc.
- Modify as needed: `report_content.py` and `build_agent_survey_report.py`

**Interfaces:**
- Consumes: validated final DOCX and retained reference.
- Produces: clean page PNGs and fidelity/structure evidence.

- [ ] **Step 1: Render the final DOCX to a fresh directory**

```powershell
& $PYTHON_BIN "$SKILL_DIR/render_docx.py" personal-planning/2026-summer/agent-survey-and-research-topic-report.docx --output_dir personal-planning/2026-summer/_report-work/final-render-1 --emit_pdf
```

Expected: one PNG per page and a non-empty PDF used only for QA.

- [ ] **Step 2: Inspect every page at 100% zoom**

Check cover fidelity, Chinese glyphs, heading hierarchy, paragraph rhythm, repeated table headers, row splits, citations, page breaks, footer alignment, clipping, overlaps, and excessive blank space.

- [ ] **Step 3: Run template fidelity and feature audits**

```powershell
& $PYTHON_BIN "$SKILL_DIR/scripts/render_and_diff.py" $REFERENCE_DOCX personal-planning/2026-summer/agent-survey-and-research-topic-report.docx --outdir personal-planning/2026-summer/_report-work/fidelity-diff-1
& $PYTHON_BIN "$SKILL_DIR/scripts/section_audit.py" personal-planning/2026-summer/agent-survey-and-research-topic-report.docx
& $PYTHON_BIN "$SKILL_DIR/scripts/style_lint.py" personal-planning/2026-summer/agent-survey-and-research-topic-report.docx --json personal-planning/2026-summer/_report-work/final-style-evidence.json
& $PYTHON_BIN "$SKILL_DIR/scripts/a11y_audit.py" personal-planning/2026-summer/agent-survey-and-research-topic-report.docx
```

Expected: intended body differences only; no unexplained page-geometry, cover, style, relationship, or recurring-chrome loss.

- [ ] **Step 4: Fix defects and rerender to a new iteration directory**

Never reuse an old render directory. Repeat build, structural validation, render, and every-page inspection until all pages are clean.

- [ ] **Step 5: Run the final gate**

Run the validator one final time and verify the retained template SHA-256 again. Expected: all checks pass and the reference is unchanged.

---

### Task 7: Final Review and Delivery

**Files:**
- Final: `personal-planning/2026-summer/agent-survey-and-research-topic-report.docx`

**Interfaces:**
- Consumes: latest passing DOCX and latest inspected page renders.
- Produces: one user-facing link to the final DOCX.

- [ ] **Step 1: Review the report against the acceptance criteria**

Confirm the report is systematic, comprehensive, targeted to the user, executable through August 31, and explicitly optimized for a two-year thesis/work topic decision.

- [ ] **Step 2: Check workspace scope**

Run:

```powershell
git status --short
```

Confirm no unrelated user changes were modified or reverted.

- [ ] **Step 3: Deliver only the final DOCX**

Return exactly one standalone Markdown link to the final DOCX and a short note that template fidelity, structural validation, and every-page render inspection passed.
