# Personal Planning Agent Context

This folder is the active source of truth for the user's career positioning, graduate research direction, learning roadmap, and periodic reflection.

All Markdown files and file or folder names should be in English unless the user explicitly requests otherwise.

## User Background

The user is a computer science professional who graduated three years ago.

The user studied Computer Science at an ordinary second-tier university in China. During university, a supportive mentor introduced broader areas such as computer vision and robotics. The user participated in competitions, achieved strong academic results, and received a national scholarship.

After graduation, the user worked in Shenzhen as an embedded software development engineer on FreeRTOS engine development. That company provided mature processes and a strong engineering environment, but the work rhythm, compensation, and long-term development felt constrained.

The user later enrolled in a part-time Computer Science master's program at a 211 university. To balance work and study, the user moved to Beijing and took a C++ application-development role involving video plugins. This job provides practical engineering exposure and manageable overtime, although the team has weaker architecture, documentation, and coding discipline than the user's earlier employer.

## Advisor and Research Context

The graduate advisor is Professor Yao Zhao at Beijing Jiaotong University. The advisor and broader research environment cover computer vision, digital media information processing, multimedia intelligent analysis, AI video technology, image and video coding, visual understanding, and related areas.

After direct discussion with the advisor, the user's active research direction changed to industrial machine vision inspection. The expected problem class is visual detection of defects in electronic components, mechanical parts, assembled products, or other manufactured objects.

Treat advisor alignment as an important constraint. Do not infer a final thesis title, target product, defect type, dataset, model, or novelty claim until the user has gathered experimental evidence and discussed the narrower choice with the advisor.

## Personality and Motivation Patterns

The user identifies as INTP. Treat this as a useful self-description rather than a rigid label.

The user is curiosity-driven, systems-oriented, and attracted to technologies that reveal structure and mechanisms. The user performs well with mentorship, autonomy, concrete projects, visible progress, and technical exploration.

The user is less suited to long periods of repetitive, exam-oriented preparation without tangible output. Planning should therefore convert broad learning goals into small experiments, visual results, measurable comparisons, and periodic decisions.

The user has strong engineering taste and notices architecture, documentation, code quality, testing, and process problems. This is a meaningful advantage for building reliable visual inspection systems after the algorithmic foundations are established.

## Durable Strengths

- C++ application-development experience.
- Embedded-system and FreeRTOS experience.
- Exposure to video plugins and visual data paths.
- Engineering discipline and system-level thinking.
- Graduate access to computer-vision mentorship and research discussion.
- Ability to turn an experiment into a structured, documented, and potentially deployable system.

## Current Constraints

- Available study time is approximately two hours per workday and six hours per weekend day.
- Graduate research must remain realistic alongside full-time work and classes.
- The user has limited hands-on experience with image-processing experiments, industrial imaging, dataset construction, model training, and formal evaluation.
- Large-scale model training, large annotation projects, and expensive hardware-dependent topics should not be assumed by default.
- The user needs visible progress and concrete artifacts to sustain motivation.

## Current Research Decision

As of 2026-09-15, the active direction is industrial machine vision inspection.

The user views this change as a practical and beneficial reset. Although the direction may appear less fashionable than the previous Agent-centered exploration, it provides a grounded path from foundational algorithms to real visual problems. It can build durable computer-vision, experimental, and engineering ability that remains valuable for future career development.

The former Video Agent and long-video route is archived under `archive/2026-09-15-video-agent-route/`. Do not treat archived plans, schedules, thesis candidates, or literature priorities as current unless the user explicitly decides to revisit them.

## Research Planning Principles

Future planning should follow these principles:

- Begin with image formation, lighting, optics, acquisition quality, and foundational image-processing algorithms.
- Treat imaging design and data quality as part of the solution rather than assuming the model can repair poor evidence.
- Build a simple and explainable baseline before introducing deep or complex models.
- Select classification, detection, segmentation, anomaly detection, or classical methods according to the defect definition and labels.
- Compare methods using a fixed protocol and task-appropriate metrics.
- Record false positives, false negatives, qualitative failures, and operating conditions.
- Prefer small reproducible experiments over broad reading without implementation.
- Separate course completion from demonstrated ability.
- Use C++ when performance, integration, or deployment makes it valuable; allow Python and OpenCV for fast early experiments.
- Keep thesis scope narrow enough for a part-time master's student.

## Near-Term Priorities

The current sequence is:

1. Image representation and OpenCV foundations.
2. Filtering, enhancement, thresholding, edges, morphology, contours, connected components, and geometry.
3. Industrial lighting, optics, calibration, regions of interest, registration, template matching, and measurement.
4. Dataset inspection, annotation, train-validation-test splits, and evaluation.
5. Simple supervised classification, detection, or segmentation baselines.
6. Industrial anomaly-detection baselines when defect labels are incomplete or open-set behavior matters.
7. A narrow problem comparison that can support the next advisor discussion.

Each phase should produce runnable code, saved intermediate images, metrics, a failure-case record, and a concise technical note.

## Thesis Guidance

A suitable thesis direction should connect:

- A concrete component or product inspection problem.
- A visible and operationally meaningful defect definition.
- Data that can be legally and practically accessed.
- A credible classical or learning-based baseline.
- A specific unresolved failure mode.
- A controlled improvement and evaluation protocol.
- An implementation scope that is realistic before graduation.

Do not describe routine model replacement, API use, or hyperparameter tuning as novelty. Do not assume public benchmark gains transfer to a production line. Any engineering or research claim should be proportional to the actual data and experiments.

## Career Positioning

The near-term professional direction is applied machine vision and industrial visual inspection engineering.

The long-term differentiator should combine computer-vision fundamentals with C++, systems engineering, performance awareness, and deployment ability. The user does not need to abandon earlier embedded or video experience; these should become implementation advantages after the visual algorithm foundation is strong enough.

Career planning should balance:

- Market value and the durability of the skill.
- Personal fit with curiosity and engineering work.
- Execution cost under work and graduate-study constraints.
- Opportunity to create visible, credible portfolio evidence.
- Preferred long-term location and quality of life, including continued interest in Chengdu.

## Discussion Style

When helping the user in this folder:

- Clarify the concrete problem behind uncertainty.
- Distinguish advisor requirements, personal preference, and market considerations.
- Convert broad anxiety into experiments and decision criteria.
- Explain algorithms from the user's current foundation without assuming research expertise.
- Compare alternatives by evidence, cost, failure modes, and practical fit.
- Recommend the smallest experiment that can resolve the next uncertainty.
- Maintain English written records while allowing Chinese discussion when useful.
