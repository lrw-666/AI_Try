# Weekly News Report Automation

## Objective

Every Monday, generate four Word-format weekly news reports. The reports should have strong factual accuracy, clear context, reliable sourcing, and useful analysis rather than a simple headline digest. There is no fixed page-count requirement; cover the most important events with enough depth to explain why they matter.

The four reports are:

1. International news weekly: summarize major international events from the previous week, explain causes and consequences, and assess the impact on the international community.
2. China news weekly: summarize major China-related events from the previous week, explain causes and consequences, and assess the impact on Chinese society.
3. Technology frontier weekly: summarize important technology-frontier developments from the previous week, including but not limited to AI, robotics, biotechnology, medicine, chips, energy technology, scientific computing, space technology, and other high-impact research or industry changes. This report is for broad knowledge growth and does not need to be directly related to the user's personal development direction. Prefer reliable sources such as major journals, preprint servers, reputable research institutions, official company announcements, and high-quality technology media.
4. Personal development trends weekly: read the personal-planning Markdown files in this project, then track work, research, career, learning, and industry developments that may affect or benefit the user's development. The scope may include AI, agents, video, C++, computer vision, industrial systems, productivity tools, content production, automation tools, job-market signals, open-source ecosystems, policy changes, and other relevant developments. Do not force the report to be strongly tied to robotics; include any development that is meaningfully useful for the user's learning, positioning, research, or career planning.

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

The folder must contain four final Word documents:

```text
YYYY-MM-DD_international-news-weekly.docx
YYYY-MM-DD_china-news-weekly.docx
YYYY-MM-DD_technology-frontier-weekly.docx
YYYY-MM-DD_personal-development-trends-weekly.docx
```

If intermediate material is useful, create `sources/` or `notes/` under the weekly folder. The four `.docx` files remain the primary deliverables.

## Quality Requirements

- There is no fixed page-count target. Do not pad mechanically; prioritize information density, structure, readability, and analysis quality.
- Each report should include a clear title, date range, executive summary, main analysis, impact assessment, follow-up watchlist, and references.
- Each report should cover the most important few events rather than attempting exhaustive coverage.
- Each report should use Chinese-English bilingual writing by default. Chinese should carry the main explanation; English should help the user learn vocabulary, concepts, and professional phrasing.
- Important facts must be cross-checked. Prefer official releases, reputable media, research institutions, company announcements, papers, policy documents, and other reliable sources.
- Do not merely list news titles. Explain background, key actors, timeline, direct causes, deeper causes, possible next steps, and uncertainty.
- Impact assessment should distinguish short-term, medium-term, and long-term effects when useful.
- For disputed events, present multiple perspectives and avoid unsupported conclusions.
- References should be written as human-readable source lists. Do not leak internal tool citation markers into the documents.

## Suggested Structure

### 1. International News Weekly

- Cover information: title, coverage period, generation date.
- Executive summary: roughly one page covering the 5 to 8 most important judgments.
- Major international events overview: sorted by importance rather than only by time.
- Deep dives: background, sequence of events, key nodes, stakeholder positions, and impact.
- Impact assessment: geopolitics, economy and trade, technology industry, energy and climate, financial markets, and global governance.
- Next-week watchlist.
- References.

### 2. China News Weekly

- Cover information: title, coverage period, generation date.
- Executive summary: key judgments about Chinese society, economy, technology, policy, and public life.
- China-related event overview: domestic policy, industrial technology, macroeconomics, society and livelihood, diplomacy, and international relations.
- Deep dives: policy background, real constraints, public response, industry impact, and regional impact where relevant.
- Impact assessment: governance, economic structure, employment and education, technology industry, social sentiment, and international relations.
- Next-week watchlist.
- References.

### 3. Personal Development Trends Weekly

First read and summarize the personal-planning Markdown files. Current priority files:

- `personal-planning/agent.md`
- `personal-planning/current-positioning.md`
- `personal-planning/learning-status-2026-06.md`

If new Markdown files are added under `personal-planning/`, include them in the scan.

Use the personal plan to decide which external developments matter most, but do not restrict the report to robotics or any single narrow domain. Suggested priority:

1. Developments directly related to long-term positioning, career path, and research interests.
2. AI agents, video generation or video understanding, C++/systems engineering, computer vision, industrial applications, automation workflows, and personal productivity tools.
3. Open-source projects, papers, product releases, company strategy changes, funding signals, job-market trends, policy changes, and tools that may improve the user's learning, research, portfolio work, or career positioning.

The document should include:

- Personal-plan relevance summary.
- The most important direction changes this week.
- Key papers, products, companies, open-source projects, or industry events.
- Concrete implications for learning, portfolio work, research topics, and career opportunities.
- Suggested actions for the next week.
- References.

### 4. Technology Frontier Weekly

- Cover information: title, coverage period, generation date.
- Executive summary: the most important technology-frontier judgments from the previous week.
- Major technology developments overview: prioritize impact, novelty, reliability of evidence, and likely medium-term consequences.
- Deep dives: explain the technical background, what changed this week, why it matters, who may be affected, and what uncertainty remains.
- Suggested coverage areas: AI and foundation models, robotics, biotechnology and medicine, chips and computing infrastructure, energy technology, aerospace, materials science, security, developer tools, and major scientific publications.
- Source preference: Nature, Science, Cell, The Lancet, NEJM, arXiv where appropriate, official university or lab releases, reputable company announcements, standards bodies, regulators, and high-quality technology journalism.
- Impact assessment: scientific significance, industrial value, policy and ethics, talent and skill implications, and possible long-term direction shifts.
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
4. Search and cross-check sources for the four report categories.
5. Draft the outline, core judgments, and source list for each report.
6. Generate four `.docx` files with Chinese-English bilingual content.
7. Render and visually inspect the four DOCX files; revise and re-render if needed.
8. Save final DOCX files to the weekly folder.
9. Report the output directory, filenames, coverage window, content overview, and QA status.

## Success Criteria

- The weekly directory exists and uses the correct English path.
- The four `.docx` files exist and use the correct English filenames.
- Each report is complete, well-structured, reliably sourced, and analytical.
- Each report uses Chinese-English bilingual writing by default and has no obvious visual defects.
- Each report covers the most important few events with useful depth rather than chasing a fixed page count.
- The personal-development report clearly reads and responds to the personal-planning Markdown files.
- The technology-frontier report broadens the user's knowledge and is not limited to the user's personal development direction.
