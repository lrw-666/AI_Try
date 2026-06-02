# Personal Planning Agent Context

This folder is dedicated to the user's personal career planning, long-term development strategy, self-analysis, graduate study planning, research direction exploration, and periodic reflection.

All Markdown files in this folder should be written in English unless the user explicitly requests another language for a specific task. File and folder names should also use English by default.

## User Background Snapshot

The user is a computer science professional who graduated three years ago.

The user studied Computer Science as an undergraduate at an ordinary second-tier university in China. During university, the user developed a strong early interest in computer technology because of its logical rigor and practical power. A supportive mentor opened up broader areas of computing, including computer vision and robotics. The user participated in many competitions, achieved strong academic results, and won a national scholarship.

After graduation, the user missed the main campus recruitment window and went to Shenzhen to seek more opportunities. The user joined a strong large-scale embedded systems company as an embedded software development engineer, working on FreeRTOS engine development. The company had mature processes and highly professional colleagues, which gave the user a solid early professional foundation. However, the work rhythm was slow, procedures were heavy, compensation was limited, and long-term development felt constrained.

The user later enrolled in a part-time Computer Science master's program at a 211 university. The user's advisor is an expert in computer vision. To balance work and study, the user moved to Beijing and took a C++ application development role involving front-end and back-end design for video plugins. This role has a faster development pace and less overtime, making weekend classes easier to manage. However, the team has weaker engineering discipline, unclear architecture, insufficient documentation, and less rigorous coding practices.

The user's advisor has suggested exploring research directions related to industrial video.

## Personality And Behavioral Patterns

The user identifies as INTP. This should be treated as a useful self-description rather than a rigid label.

The user tends to be curiosity-driven, concept-oriented, and attracted to systems that reveal structure, logic, and hidden mechanisms. The user responds well to open exploration, technical breadth, real projects, and meaningful mentorship.

The user performs well when the learning environment contains novelty, feedback, autonomy, and concrete creation. The user's university experience shows strong potential under supportive conditions: competition work, mentor guidance, practical technical exposure, and academic recognition.

The user is less suited to long cycles of repetitive preparation, rote memorization, and high-stakes standardized exams. Motivation tends to decay when the task becomes repetitive, externally imposed, and disconnected from visible progress or personal meaning.

The user appears to have stronger practical engineering sensitivity than exam endurance. The user notices process quality, code rigor, architecture clarity, documentation discipline, and team professionalism. This suggests that the user should not define ability only through mathematics, algorithms, or exam outcomes.

The user may underestimate personal strengths because recent transitions involved uncertainty, missed recruitment timing, exam setbacks, and imperfect job environments. Future planning should separate real skill limits from confidence damage caused by mismatched evaluation systems.

## Planning Principles

Career planning should be analyzed from the user's own perspective, not from a generic software engineer template.

The user should avoid paths that depend mainly on long-term rote exam preparation, abstract algorithmic research, or pure mathematical intensity unless there is a strong external reason and a clear support system.

The user should prioritize paths where C++, video systems, embedded systems, computer vision, industrial scenarios, and AI-enabled engineering can overlap.

The user's part-time master's program should be used as a bridge, not merely a credential. Research topics should ideally connect the advisor's vision expertise, the user's video engineering work, and practical industrial problems.

The user should build a visible portfolio of applied work: paper notes, system design notes, video-processing experiments, industrial video prototypes, C++ engineering writeups, and small AI-assisted tools.

Planning should balance three dimensions:

- Market value: whether the direction has durable demand in the AI era.
- Personal fit: whether the work matches curiosity, autonomy, and engineering instincts.
- Execution cost: whether the path is realistic alongside work and part-time graduate study.

## Updated Preference Signals

The user is especially interested in future-facing fields such as robotics, 3D vision, AR/VR, spatial computing, AI-native software, and agent development.

The user has relatively low intrinsic interest in traditional monitoring, operations, and routine surveillance-style work, but may still consider such paths if they provide strong stability or compensation.

The user is open to a major career transition, including AI product, technical product, public-sector, public-institution, or stable institutional paths, if the tradeoff is rational.

The user prefers Chengdu as a medium-high priority city after graduation. A Chengdu technical role around 20k RMB per month would be satisfying. A Chengdu public-sector or public-institution role around 10k RMB per month can be acceptable if the stability and quality of life are strong.

The user can accept short-term overtime for real project needs, but dislikes chronic structural overtime caused by poor leadership, weak management, or company dysfunction.

The user strongly dislikes strong-sales roles and should avoid career paths where sales pressure is the core daily responsibility.

The user's available study capacity is approximately 2 hours per workday and 6 hours per weekend day under current conditions.

## Suggested Long-Term Positioning

The user should explore becoming an applied AI/video systems engineer rather than competing directly as a pure algorithm researcher or generic CRUD application developer.

Promising positioning themes include:

- Industrial video systems with AI-assisted analysis.
- C++ video processing and real-time media infrastructure.
- Computer vision application engineering for inspection, monitoring, robotics, or industrial automation.
- Edge AI and embedded vision systems.
- AI-assisted engineering tools for video workflows, plugin systems, testing, documentation, and data analysis.
- AI agent development for workflow automation, research assistance, code/document analysis, and domain-specific engineering tools.

## Current Main Planning Direction

As of 2026-05-22, the user's most promising near-term planning thread is C++ video systems plus AI agent engineering.

The key insight is that the user can avoid diving too deeply into the hardest XR or robotics technology stacks while still staying close to those future-facing fields. Video streams are a shared infrastructure layer for XR, robotics, spatial computing, digital twins, intelligent monitoring, and multimodal agent workflows.

The user's current video-plugin work should be expanded from edge business logic toward video stream ingestion, push/pull streaming, frame extraction, event metadata, semantic indexing, multimodal retrieval, and agent-based workflow automation.

The recommended thesis bridge is an applied research direction around long video stream semantic event indexing and agent-based retrieval/question answering. This can connect the advisor's computer vision background, the user's C++ video work, and the user's interest in AI-native agent development without requiring pure algorithm research or heavy mathematical novelty.

## Recent Planning Decisions

As of 2026-05-25, the planning focus has evolved from one fixed thesis title into a broader "video stream + agent" thesis and career topic space.

The important conclusion is that the user should not lock onto a single title too early. The current best strategy is to explore several adjacent video stream + agent topics from June to August 2026, then choose the strongest one for the September 2026 thesis proposal.

The field frontier should be understood in several layers:

- Long video understanding: models and benchmarks are improving, but long context remains expensive and evidence localization is still difficult.
- Video-RAG: recent work converts video into auxiliary text, events, documents, keyframes, or graph structures before retrieval and reasoning.
- Multi-agent video understanding: agents are used for task decomposition, retrieval, evidence aggregation, reflection, and anomaly explanation.
- Online or streaming video understanding: research is moving toward incremental memory, sliding-window reasoning, low-latency understanding, and proactive responses.
- Video library question answering: systems are expanding from single-video QA toward multi-video retrieval, entity discovery, schema generation, and cross-video search.

For the user's part-time master's background and limited learning time, the feasible research strategy is not to train a new video foundation model or reproduce large-scale benchmark results. The feasible strategy is to build a lightweight, reproducible, system-level method using existing models and tools.

The recommended innovation level is system-method innovation rather than model-parameter innovation:

- Design event schemas for video streams.
- Combine temporal indexing, keyword retrieval, vector retrieval, and event-type filtering.
- Build agent toolchains with explicit retrieval, evidence localization, verification, and reporting tools.
- Return traceable evidence such as timestamps, keyframes, event clips, confidence notes, and source fields.
- Evaluate with small but controlled datasets using Recall@K, temporal localization error, evidence hit rate, answer accuracy, latency, and cost.

Five thesis topic candidates have been identified:

- Topic A: Semantic event indexing and agent-based retrieval QA for long video streams.
- Topic B: Incremental event memory and proactive summarization agents for real-time video streams.
- Topic C: Multimodal entity/event graph construction and agent querying for video libraries.
- Topic D: Frame-log joint diagnostic agents for video system engineering.
- Topic E: Lightweight Video-RAG agent event explanation for a vertical scenario such as campus, meeting, teaching, robot first-person view, or equipment operation videos.

The current recommendation is:

- Primary candidate: Topic A, because it is the most balanced across thesis feasibility, job relevance, reproducibility, and limited training requirements.
- Strong secondary candidate: Topic D, because it is closest to the user's current video-plugin work and can become a strong engineering portfolio project.
- Frontier enhancement module: Topic B, because online/streaming video understanding is more forward-looking but should be implemented as a simplified low-frequency sampling and sliding-window memory module rather than a full streaming VLM.
- Optional applied variant: Topic E, if the advisor prefers a clearer application scenario.
- Lower priority for first proposal: Topic C, because it has strong innovation potential but higher data, schema, and evaluation cost.

The June to September 2026 validation strategy should be:

- June 2026: Build small prototypes for Topic A and Topic D. Topic A should include video input, frame extraction, auxiliary text/event generation, indexing, and simple QA. Topic D should sketch frame-log joint diagnosis using playback logs, push/pull stream state, and visual evidence.
- July 2026: Read and summarize core papers such as DrVideo, Video-RAG, VideoRAG, VideoStir, LongVideoBench, Video-MME, HM-RAG, RAVEN, and online video understanding papers.
- August 2026: Build a demonstrable system with video input, event extraction, indexing, agent tool calls, timestamp/keyframe evidence return, and basic evaluation.
- September 2026: Propose Topic A as the main thesis direction, optionally adding Topic B or Topic D as a distinctive enhancement module.

Important implementation boundaries:

- Avoid training a new video large model.
- Avoid large-scale dataset construction.
- Avoid deep SLAM, 3D reconstruction, or robotics control as the thesis core.
- Avoid making the thesis only a chatbot or shallow API wrapper.
- Avoid traditional surveillance operations as the main narrative unless it is reframed as video stream intelligence, event indexing, or system diagnostics.

The thesis should be framed as applied computer vision plus AI engineering, not merely software engineering. When discussing with the advisor, emphasize video understanding, temporal evidence localization, multimodal retrieval, event indexing, and evaluation metrics. Agent engineering should be presented as the orchestration layer that makes the vision system usable and explainable.

## Discussion Style For Future Planning

When analyzing problems in this folder, help the user:

- Clarify the real constraint behind confusion.
- Separate external pressure from personal preference.
- Convert vague anxiety into concrete options and experiments.
- Compare paths by fit, risk, opportunity, and execution cost.
- Produce practical next actions rather than only abstract advice.
- Preserve English written records while allowing Chinese discussion when useful.
