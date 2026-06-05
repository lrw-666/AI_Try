# Weekly Development And Technology Report Automation

## Objective

Every Monday, generate two Word-format weekly reports. The reports should have strong factual accuracy, clear context, reliable sourcing, and useful analysis rather than a simple headline digest. There is no fixed page-count requirement; cover the most important developments with enough depth to explain why they matter.

The two reports are:

1. Personal development trends weekly: read the personal-planning Markdown files in this project, first summarize what the user should complete during the current week, then track work, research, career, learning, and industry developments that may affect or benefit the user's development. This report is a practical weekly planning and direction-filtering document. It should connect the user's current constraints, thesis/career direction, learning capacity, project work, and useful external signals into concrete next actions. The scope may include AI, agents, video, C++, computer vision, industrial systems, productivity tools, content production, automation tools, job-market signals, open-source ecosystems, policy changes, and other relevant developments. Do not force the report to be strongly tied to robotics; include any development that is meaningfully useful for the user's learning, positioning, research, or career planning.
2. Technology frontier weekly: summarize important technology-frontier developments from the previous week, including but not limited to AI, robotics, biotechnology, medicine, chips, energy technology, scientific computing, space technology, materials science, security, developer tools, and other high-impact research or industry changes. This report is for broad knowledge growth and does not need to be directly related to the user's personal development direction. It should help the user build technical taste, frontier awareness, and English technical vocabulary. Prefer reliable sources such as major journals, preprint servers, reputable research institutions, official company announcements, standards bodies, regulators, and high-quality technology media.

Do not generate separate international-news-weekly or china-news-weekly documents. International or China-related events may still be mentioned inside the two remaining reports when they directly affect personal development or represent important technology-frontier developments.

Output folders and final DOCX filenames must use English names. Report body content should be written in Chinese-English bilingual form by default, with Chinese as the main explanatory language and English used as a learning aid. Use bilingual section titles, bilingual summaries, or paired Chinese/English paragraphs where helpful. Project folders, filenames, and Markdown records should use English names unless the user explicitly requests otherwise.

## Schedule And Coverage

- Frequency: once every Monday.
- Timezone: Asia/Shanghai.
- Coverage window: the previous complete natural week, from Monday 00:00 to Sunday 23:59.
- Weekly folder name: create a folder named with the Monday date in `YYYY-MM-DD` format.

## Output Directory And Filenames

Weekly output directory:

```text
automation-work/YYYY-MM-DD/
```

The folder must contain two final Word documents:

```text
YYYY-MM-DD_personal-development-trends-weekly.docx
YYYY-MM-DD_technology-frontier-weekly.docx
```

If intermediate material is useful, create `sources/` or `notes/` under the weekly folder. The two `.docx` files remain the primary deliverables.

## Quality Requirements

- There is no fixed page-count target. Do not pad mechanically; prioritize information density, structure, readability, and analysis quality.
- Each report should include a clear title, date range, executive summary or opening summary, main analysis, impact assessment, follow-up watchlist or next actions, and references.
- Each report should cover the most important few developments rather than attempting exhaustive coverage.
- Each report should use Chinese-English bilingual writing by default. Chinese should carry the main explanation; English should help the user learn vocabulary, concepts, and professional phrasing.
- Important facts must be cross-checked. Prefer official releases, reputable media, research institutions, company announcements, papers, policy documents, and other reliable sources.
- Do not merely list titles. Explain background, key actors, timeline, direct causes, deeper causes, possible next steps, and uncertainty.
- Impact assessment should distinguish short-term, medium-term, and long-term effects when useful.
- For disputed or uncertain developments, present multiple perspectives and avoid unsupported conclusions.
- References should be written as human-readable source lists. Do not leak internal tool citation markers into the documents.

## Suggested Structure

### 1. Personal Development Trends Weekly

First read and summarize the personal-planning Markdown files. Current priority files:

- `personal-planning/agent.md` if present
- `personal-planning/current-positioning.md`
- `personal-planning/learning-status-2026-06.md`

If new Markdown files are added under `personal-planning/`, include them in the scan.

Use the personal plan to decide which external developments matter most, but do not restrict the report to robotics or any single narrow domain. Suggested priority:

1. Developments directly related to long-term positioning, career path, research interests, current work constraints, or current monthly learning goals.
2. AI agents, video generation or video understanding, C++/systems engineering, computer vision, industrial applications, automation workflows, and personal productivity tools.
3. Open-source projects, papers, product releases, company strategy changes, funding signals, job-market trends, policy changes, and tools that may improve the user's learning, research, portfolio work, or career positioning.

The document should include:

- Cover information: title, coverage period, generation date.
- Opening section: `This Week's Work Plan / 本周任务安排`. This section must appear before the external trend analysis. It should translate the user's personal-planning files into concrete tasks for the current week, including study tasks, project/prototype tasks, paper-reading tasks, work-reflection tasks, advisor-communication tasks, and portfolio-output tasks that matter.
- Weekly task prioritization: distinguish `Must Do / 必须完成`, `Should Do / 应该推进`, and `Optional / 可选探索` items. Keep the plan realistic for the user's available time: about 2 hours per workday and 6 hours per weekend day.
- Personal-plan relevance summary: explain which parts of the user's long-term positioning are being reinforced or challenged.
- External trend analysis: summarize only the most relevant few papers, products, companies, open-source projects, job-market signals, policy changes, or industry events.
- Implications: translate each important external signal into concrete implications for learning, portfolio work, research topics, job positioning, or advisor discussion.
- Suggested actions for the next week.
- References.

### 2. Technology Frontier Weekly

- Cover information: title, coverage period, generation date.
- Executive summary: the most important technology-frontier judgments from the previous week.
- Major technology developments overview: prioritize impact, novelty, reliability of evidence, and likely medium-term consequences. Cover only the most important few developments with real depth.
- Deep dives: explain the technical background, what changed this week, why it matters, who may be affected, what uncertainty remains, and what English technical terms are worth learning.
- Suggested coverage areas: AI and foundation models, robotics, biotechnology and medicine, chips and computing infrastructure, energy technology, aerospace, materials science, security, developer tools, and major scientific publications.
- Source preference: Nature, Science, Cell, The Lancet, NEJM, arXiv where appropriate, official university or lab releases, reputable company announcements, standards bodies, regulators, and high-quality technology journalism.
- Impact assessment: scientific significance, industrial value, policy and ethics, talent and skill implications, and possible long-term direction shifts.
- Learning value section: briefly explain what the user can learn from these developments even when they are not directly connected to the user's current career plan.
- Next-week watchlist.
- References.

## Word Generation Requirements

- Use the Documents plugin capability to generate `.docx` files.
- For new Word documents, use a professional weekly-report or research-brief style with consistent heading hierarchy, page headers and footers, tables, and reference formatting.
- Before delivery, render each DOCX into page images and visually inspect for overlapping text, table overflow, bad page breaks, missing glyphs, and header/footer problems.
- If the rendering environment is unavailable, complete a structural check and report why visual render QA was not completed.

## Automation Steps

1. Compute the Monday date for the current run and create `automation-work/YYYY-MM-DD/`.
2. Determine the previous complete natural-week coverage window.
3. Read this task file and `personal-planning/**/*.md`.
4. Search and cross-check sources for the two report categories: personal-development trends and technology-frontier developments.
5. Draft the outline, core judgments, and source list for each report. For the personal-development report, draft the current week's work plan before drafting the external trend sections.
6. Generate two `.docx` files with Chinese-English bilingual content.
7. Render and visually inspect the two DOCX files; revise and re-render if needed.
8. Save final DOCX files to the weekly folder.
9. Report the output directory, filenames, coverage window, content overview, and QA status.

## Success Criteria

- The weekly directory exists and uses the correct English path.
- The two `.docx` files exist and use the correct English filenames.
- Each report is complete, well-structured, reliably sourced, and analytical.
- Each report uses Chinese-English bilingual writing by default and has no obvious visual defects.
- Each report covers the most important few developments with useful depth rather than chasing a fixed page count.
- The personal-development report clearly reads and responds to the personal-planning Markdown files, and begins with a realistic current-week work plan.
- The technology-frontier report broadens the user's knowledge and is not limited to the user's personal development direction.
