# June 2026 Learning Status And Goals

Date: 2026-05-25

## Current State

I am starting June 2026 as an applied AI/video systems beginner rather than a complete programming beginner.

My current foundation:

- I have taken machine learning and deep learning courses, so I have conceptual exposure to supervised learning, neural networks, model training, and evaluation.
- I have practical C++ and embedded software experience, including earlier exposure to lower-level engineering and FreeRTOS-style work.
- My current work involves C++ application development and video plugin-related business, but I have not yet systematically mastered the main video streaming pipeline.
- I have very limited hands-on experience reading papers, reproducing research ideas, building AI/CV experiments, and turning papers into runnable prototypes.
- My advisor's background is computer vision, and my current planning direction is video stream + agent, especially semantic event indexing and agent-based retrieval/question answering for long video streams.

## Current Constraints

- Available study time is about 2 hours per workday and about 6 hours per weekend day.
- I am a part-time master's student, so the plan must be realistic alongside work and classes.
- I should avoid choosing a thesis path that requires training a new large video model, building a large-scale dataset, or doing heavy mathematical novelty from scratch.
- I need visible progress and concrete artifacts to stay motivated.

## Current Advantages

- I have engineering taste and can recognize architecture, documentation, code quality, and process problems.
- I have a bridge between C++ video engineering, applied computer vision, and AI agent workflows.
- My current job gives me practical contact with video-related systems, even if the team environment is not ideal.
- My long-term positioning can be stronger if I become an applied AI/video systems engineer rather than a pure algorithm researcher or generic application developer.

## June 2026 Main Goal

Build the first minimal learning and practice loop for the video stream + agent direction:

Video input -> frame extraction -> event records -> simple retrieval -> evidence-based answer.

The purpose of June is not to finish a thesis prototype. The purpose is to prove that I can read relevant papers, extract implementable ideas, run basic video processing experiments, and create a small evidence-based video event retrieval workflow.

## Baseline Goals

- Read and summarize at least 2 papers or benchmark papers related to long video understanding / Video-RAG.
- Run basic FFmpeg experiments: inspect video metadata, extract frames, and cut clips.
- Create a simple event schema for video records.
- Generate 10-30 event records for at least one short video.
- Implement or simulate simple keyword retrieval over those event records.
- Write a short end-of-month comparison between Topic A and Topic D.

## Stretch Goals

- Read 4 papers: LongVideoBench or Video-MME, Video-RAG, DrVideo, and one agentic video/RAG paper such as HM-RAG or RAVEN.
- Build a minimal Topic A prototype that can answer simple questions with event timestamps and evidence frames.
- Add vector retrieval or hybrid retrieval on top of keyword search.
- Prepare a small demo and one-page summary for advisor discussion.

## June Direction Decision

Primary exploration direction:

- Topic A: Semantic event indexing and agent-based retrieval QA for long video streams.

Secondary exploration direction:

- Topic D: Frame/log joint diagnostic agent for video system engineering.

The June strategy is to validate both lightly, then decide which one is more suitable for July-August deepening and September thesis proposal preparation.

## Progress Log

| Week | Focus | Planned Output | Actual Output | Notes |
|---|---|---|---|---|
| Week 1 | Learning map and environment | Paper note 1, FFmpeg command notes, extracted frames |  |  |
| Week 2 | Video-RAG to event schema | Paper note 2, event schema v0, 10 events |  |  |
| Week 3 | Retrieval and evidence | Search script, 5 test questions, evidence table |  |  |
| Week 4 | Minimal answer workflow | answer_with_evidence demo, Topic A/D comparison |  |  |

## Monthly Review Questions

- What did I actually run this month?
- Which paper ideas became code, schema, or evaluation metrics?
- Which part felt most natural: video engineering, event indexing, retrieval, Agent workflow, or diagnosis?
- What should be the main July focus?
- What should I discuss with my advisor before the September proposal?
