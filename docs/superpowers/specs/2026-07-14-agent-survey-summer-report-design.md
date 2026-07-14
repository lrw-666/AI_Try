# Agent Survey Summer Report Design

## Purpose

Create a Chinese Design Report that consolidates the preceding career and research discussion and replaces the prior summer emphasis with a systematic Agent survey-paper learning program. The program runs from July 14, 2026 through August 31, 2026 and is designed to end with one primary and one backup research topic that can be completed within two years and can benefit both the user's master's thesis and current or future work.

## User Context

The report must be grounded in the user's actual profile:

- Early-career computer science professional with embedded, FreeRTOS, C++, and video plugin experience.
- Part-time computer science master's student whose advisor works in computer vision.
- Current bridge area: video systems, applied computer vision, and Agent-enabled workflows.
- Available learning time: about two hours per workday and six hours per weekend day.
- Limited personal data, compute, annotation, and API budgets.
- Better suited to applied systems research than large-scale foundation-model training or math-heavy model innovation.
- Motivated by practical exploration, visible artifacts, and paper ideas that can become working prototypes.

## Core Positioning

The report must not frame "Agent" as a permanent job title or as a goal by itself. It must frame Agent as a systems method for connecting perception, memory, planning, tools, evidence, evaluation, and human oversight to solve domain workflows.

The summer program uses a funnel structure:

1. Build a broad and stable conceptual map of Agent systems.
2. Study mechanisms that remain useful when models and frameworks change.
3. Narrow into multimodal, video, retrieval, temporal grounding, and industrial applications.
4. Generate candidate research questions at the intersection of thesis feasibility and work value.
5. Compare candidates using explicit evidence, constraints, baselines, metrics, and a two-year execution path.

## Deliverable

- Format: Chinese DOCX.
- Template: the retained `Design Report` reference selected by the user.
- Final path: `personal-planning/2026-summer/agent-survey-and-research-topic-report.docx`.
- Expected length: approximately 18 to 25 pages after template formatting.
- Intended use: personal execution, advisor discussion, and later thesis-topic refinement.

## Report Structure

### 1. Executive Summary

Explain the revised summer priority, the reason for using survey papers, the funnel strategy, and the final August 31 decision target.

### 2. Background and Updated Judgment

Consolidate the prior discussion into a coherent argument:

- The friend's career outcome shows the value of accumulated project experience, domain exposure, timing, and experience packaging, not a need to copy a specific embodied-intelligence role.
- AI assistance lowers the practical barrier to reading, reproducing, and testing research ideas, but does not remove the need for fundamentals, verification, or evaluation.
- Studying one current Agent product has limited durability; studying planning, tool use, memory, retrieval, evaluation, reliability, and human oversight has longer-term value.
- General multimodal models do not eliminate domain Agent work because real systems still require private data handling, long-video processing, evidence localization, cost control, logs, tool integration, and auditable workflows.
- A resource-constrained student should avoid frontier-model training and large benchmark races, and should pursue lightweight system-method research with strong baselines and measurable outcomes.

### 3. Key Findings About Agent Research and Industrial Value

Summarize the earlier conclusions about:

- Where Agent innovation can exist.
- What counts as weak versus strong technical content.
- Why commercial value comes from workflow improvement rather than the Agent label.
- How Agent systems can land in industrial video review, anomaly triage, and video-log joint diagnosis.
- Where personal prototype costs arise and how local GPU inference, selective API use, caching, and staged processing control cost.

### 4. Agent Knowledge Architecture

Organize the literature into seven layers:

1. Agent foundations and augmented language models.
2. Planning, reasoning, reflection, and task decomposition.
3. Tool learning, execution, error handling, and permissions.
4. Memory, RAG, state, and experience reuse.
5. Evaluation, reliability, safety, and human-Agent collaboration.
6. Multimodal Agents, video understanding, multimodal RAG, and temporal grounding.
7. Industry Agents, workflow integration, deployment, observability, privacy, and business value.

For each layer, the report must explain definitions, major taxonomies, representative survey papers, questions to answer while reading, and how the layer connects to the user's possible research space.

### 5. Intensive Summer Program

Use exact dates and a heavier but realistic workload.

#### Week 1: July 14-19

Focus: Agent foundations, autonomous Agents, and augmented language models.

Required output: Agent/LLM/workflow comparison, core architecture map, glossary, and one video-diagnosis example mapped to perception, memory, planning, action, and feedback.

#### Week 2: July 20-26

Focus: planning, tool learning, reflection, and memory.

Required output: mechanism comparison table, video-tool registry, failure and recovery flow, and three early research-question cards.

#### Week 3: July 27-August 2

Focus: Agent evaluation, reliability, human-Agent systems, and cost-aware evaluation.

Required output: multi-level evaluation matrix, error taxonomy, human intervention design, and a draft 30-50 task evaluation set.

#### Week 4: August 3-9

Focus: large multimodal Agents and video understanding with LLMs.

Required output: local-model/API division of labor, three candidate multimodal architectures, and a lightweight video-to-event experiment design.

#### Week 5: August 10-16

Focus: multimodal RAG, long-video understanding, temporal grounding, and evidence retrieval.

Required output: video evidence pipeline, keyword/vector/hybrid retrieval baselines, a small timestamp-grounded question set, and an initial innovation-gap list.

#### Week 6: August 17-23

Focus: industry Agents, industrial video workflows, video-log joint diagnosis, deployment constraints, and commercial value.

Required output: at least ten industrial pain points, four candidate thesis topics, a data and compute feasibility estimate for each, and a thesis/work benefit statement for each.

#### Week 7: August 24-30

Focus: candidate-topic comparison, targeted reading of representative method papers, and lightweight feasibility checks.

Required output: evidence-backed topic scorecard, one-page briefs for the top three candidates, risk register, and two-year milestone draft for the leading candidates.

#### Finalization: August 31

Required output: one primary topic, one backup topic, a concise advisor-discussion brief, and a two-year execution roadmap.

Each full week should normally include two to three survey papers, three to five representative method papers for selective reading, one structured synthesis artifact, one small analysis or experiment, and one topic-funnel update. Week 1 is shorter and should prioritize the two foundation surveys and the architecture map.

### 6. Reading and Note-Taking Method

The report must specify a repeatable paper workflow:

1. First-pass triage of title, abstract, figures, taxonomy, conclusion, and limitations.
2. Extraction of problem definition, system components, datasets, baselines, metrics, cost, failure modes, and open questions.
3. Comparison against at least one non-Agent or simpler workflow baseline.
4. Translation of paper ideas into tools, schemas, evaluation cases, or small experiments.
5. Weekly synthesis that updates the research-topic funnel.

The report must distinguish survey-paper deep reading from representative-paper selective reading so the workload remains achievable.

### 7. Topic Funnel and Decision Framework

Candidate topics must be scored on:

- Fit with the user's C++ and video-system experience.
- Fit with the advisor's computer-vision environment.
- Clear research question and defensible novelty.
- Availability of data and reproducible baselines.
- Personal compute, API, annotation, and time feasibility.
- Measurable evaluation protocol.
- Direct or plausible work value.
- Potential to become a portfolio-quality prototype.
- Ability to produce staged results within two years.
- Resilience to rapid changes in foundation models and Agent frameworks.

The report should use a weighted matrix and explain that a topic cannot win only because it is fashionable. It must pass feasibility, evaluation, and work-value gates.

Likely topic families to evaluate include:

- Low-cost, evidence-grounded event retrieval for long industrial video.
- Joint diagnosis using video evidence and system logs.
- Cost-aware hierarchical perception and model routing for video analysis.
- Human-in-the-loop abnormal-event review and report generation.
- Evaluation methods for traceable video Agent workflows.

These are candidate families, not predetermined conclusions.

### 8. Two-Year Roadmap Requirement

For the selected topic, the report must define four stages:

1. Foundation and reproducible baseline.
2. Focused method contribution and controlled experiments.
3. Industrial-style prototype and robustness evaluation.
4. Thesis writing, work transfer, portfolio packaging, and advisor communication.

The roadmap must allow the research question to survive changes in specific models and frameworks.

### 9. Risks and Controls

Cover at least these risks:

- Reading without synthesis.
- Framework chasing.
- Treating an API integration as research innovation.
- Topic scope becoming too broad.
- Excessive token, GPU, or annotation cost.
- Lack of realistic data.
- Weak or circular evaluation.
- Overcommitting to video before confirming a meaningful problem.
- Neglecting current C++ and video engineering advantages.

Each risk must have a concrete weekly or stage-level control.

### 10. References and Appendices

Include:

- A curated survey-paper list grouped by knowledge layer.
- A representative-paper reading queue.
- A reusable paper-note template.
- A weekly review checklist.
- A candidate-topic one-page brief template.
- A topic scorecard template.
- A glossary of key Agent terms.

## Evidence and Citation Policy

- Use the existing workspace planning records as personal-context sources.
- Use original paper pages, publisher pages, or maintained paper repositories for technical claims.
- Distinguish peer-reviewed publications from arXiv preprints where known.
- Do not invent industrial results, performance numbers, or costs.
- Present cost ranges as planning estimates with explicit assumptions.
- Use normal human-readable references in the DOCX; do not expose tool citation tokens.

## Template and Layout Requirements

- Clone the retained Design Report reference and preserve its page setup, sections, styles, headers, footers, tables, and recurring elements.
- Use the template's executive-summary, key-findings, implications, recommendations, and appendix structure as the visual authority.
- Use tables only where comparison or scheduling benefits from them.
- Keep dense literature lists and reusable forms in appendices so the main narrative remains readable.
- Use exact dates, meaningful section headings, and consistent Chinese typography.
- Do not replace the template with a generic document preset.

## Quality and Verification

Before delivery:

1. Check that every date falls between July 14 and August 31, 2026.
2. Check that the workload reflects the shorter first week and the user's available study time.
3. Check that the report leads to a topic decision rather than only a reading list.
4. Check that every candidate-topic criterion is measurable or operationalized.
5. Check that the report does not assume the final topic in advance.
6. Check for placeholders, contradictions, unsupported factual claims, and duplicated sections.
7. Render the DOCX to page images and visually inspect every page.
8. Fix clipping, overflow, broken tables, missing glyphs, and awkward page breaks, then rerender until clean.

## Acceptance Criteria

The finished report is successful when it gives the user:

- A coherent explanation of why survey learning is the summer priority.
- A comprehensive but navigable Agent knowledge map.
- An executable July 14-August 31 schedule with concrete weekly outputs.
- A disciplined mechanism for turning reading into research questions.
- A transparent method for choosing a two-year topic that benefits thesis, work, and career development.
- A visually verified Chinese DOCX that faithfully uses the selected Design Report template.
