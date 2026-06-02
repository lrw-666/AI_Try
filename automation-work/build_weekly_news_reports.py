from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List

from generate_news_weekly_docx import (
    Paragraph,
    compute_dates,
    quality_check,
    split_paragraphs,
    write_docx,
)


@dataclass
class WeeklyReport:
    slug: str
    title: str
    overview: str
    references: List[str]
    sections: List[Paragraph]


def build_sections(
    report_title: str,
    coverage_text: str,
    archive_text: str,
    executive_bullets: List[str],
    overview_paragraphs: List[str],
    events: List[dict],
    impact_points: List[str],
    watchlist: List[str],
    references: List[str],
) -> List[Paragraph]:
    sections: List[Paragraph] = [
        Paragraph(coverage_text, "Subtitle"),
        Paragraph(archive_text, "Subtitle"),
        Paragraph("", "PageBreak"),
        Paragraph("Executive Summary", "Heading1"),
    ]
    for bullet in executive_bullets:
        sections.append(Paragraph("• " + bullet, "Body"))

    sections.append(Paragraph("Weekly Context", "Heading1"))
    for para in overview_paragraphs:
        sections.extend(split_paragraphs(para))

    sections.append(Paragraph("Major Events And Deep Dives", "Heading1"))
    for idx, event in enumerate(events, start=1):
        sections.append(Paragraph(f"{idx}. {event['title']}", "Heading2"))
        sections.extend(split_paragraphs(event["summary"]))
        sections.append(Paragraph("Context", "Heading2"))
        sections.extend(split_paragraphs(event["context"]))
        sections.append(Paragraph("Causes", "Heading2"))
        sections.extend(split_paragraphs(event["causes"]))
        sections.append(Paragraph("Consequences", "Heading2"))
        sections.extend(split_paragraphs(event["consequences"]))

    sections.append(Paragraph("Impact Assessment", "Heading1"))
    for point in impact_points:
        sections.append(Paragraph("• " + point, "Body"))

    sections.append(Paragraph("Follow-up Watchlist", "Heading1"))
    for idx, item in enumerate(watchlist, start=1):
        sections.append(Paragraph(f"{idx}. {item}", "Body"))

    sections.append(Paragraph("References", "Heading1"))
    for ref in references:
        sections.append(Paragraph(ref, "Body"))

    return sections


def build_reports(run_monday: date, coverage_start: date, coverage_end: date) -> List[WeeklyReport]:
    coverage_text = f"Coverage Window: {coverage_start.isoformat()} to {coverage_end.isoformat()} (previous complete natural week)"
    archive_text = f"Archive Monday: {run_monday.isoformat()} | Generated in Asia/Shanghai context"

    international_refs = [
        "Reuters, 2026-05-28. U.S. trade court blocks most Trump 'Liberation Day' tariffs; White House appeals. Search reference used during drafting.",
        "AP News and Reuters, 2026-05-29 to 2026-05-31. Gaza humanitarian distribution, ceasefire pressure, and diplomatic fallout. Search references used during drafting.",
        "UN OCHA occupied Palestinian territory updates, late May 2026. https://www.ochaopt.org/",
        "Reuters, 2026-05-25 to 2026-05-31. Russia-Ukraine drone strikes and Turkey talks coverage. Search references used during drafting.",
        "Reuters, 2026-05-31. OPEC+ meeting expectations and oil-market positioning. Search reference used during drafting.",
        "Reuters, 2026-05-30 to 2026-05-31. South Korea election and policy-risk coverage. Search references used during drafting.",
    ]
    international_events = [
        {
            "title": "U.S. tariff policy suffered a legal shock and global trade expectations had to be repriced",
            "summary": "On 2026-05-28, the U.S. Court of International Trade blocked most of the Trump administration's broad 'Liberation Day' tariffs, ruling that the executive branch had exceeded the scope of the emergency powers being used. The White House quickly appealed, so the issue is not settled, but the week's most important signal was that American trade policy once again became a legal-institutional question rather than a pure political slogan.",
            "context": "For global exporters, the significance of this ruling is larger than the specific tariff line items involved. Over recent months, many firms had started to price in a world where U.S. political declarations could immediately turn into trade barriers. The court decision forced companies to bring judicial review, administrative process and appeal timelines back into their risk models. In other words, the uncertainty did not disappear; it changed shape from 'how high will tariffs go' to 'which branches of U.S. government can actually impose them, and when'.",
            "causes": "The direct cause was the court's interpretation of the limits of presidential authority under emergency economic legislation. The deeper cause is the persistent tension inside the U.S. system between executive activism, electoral incentives and constitutional checks. Trade policy in the United States is increasingly not just a matter of economics or foreign policy, but also a matter of domestic institutional conflict.",
            "consequences": "In the short term, exporters and markets got a temporary sense of relief, but nobody can assume the risk is gone because appeals, replacement measures and congressional pathways all remain possible. In the medium term, the ruling reinforces a key lesson for Chinese and global firms: U.S. policy unpredictability now comes not only from politics but from the interaction between politics and legal review. In the long term, this strengthens the trend toward geographic diversification, contract flexibility and more conservative cross-border inventory planning.",
        },
        {
            "title": "The Gaza war remained the world's most visible moral and diplomatic pressure point",
            "summary": "Throughout the week, the Gaza file continued to revolve around Israeli military pressure, the effectiveness and legitimacy of aid-distribution mechanisms, and mounting international frustration over the gap between humanitarian need and actual delivery. The core issue is no longer simply battlefield control; it is whether any sustainable political and humanitarian architecture exists behind the military campaign.",
            "context": "By late May 2026, Gaza had become a test of international legitimacy. Military operations may produce tactical gains, but the outside world is increasingly focused on civilian costs, aid access and the absence of a credible post-war governance framework. Reports and official updates kept returning to the same questions: who controls distribution, how safe are corridors, whether civilians can move, and whether any actor is prepared to guarantee basic services once active operations decline.",
            "causes": "The immediate cause of this week's pressure is the structural conflict between military-security logic and humanitarian-access logic. The deeper cause is the unresolved political question of post-war governance. As long as no acceptable answer exists to 'who governs Gaza after the war, under what security guarantees, and with what external backing', humanitarian friction will remain embedded in the conflict rather than treated as an operational side issue.",
            "consequences": "In the short term, diplomatic pressure on Israel and its backers is likely to remain high, especially in multilateral settings and among European partners. In the medium term, the humanitarian crisis will continue to complicate broader regional realignment, normalization efforts and U.S. alliance management. In the long term, even if the intensity of fighting fluctuates, the lack of a viable governance formula increases the risk that Gaza remains a chronic source of instability rather than a conflict moving toward closure.",
        },
        {
            "title": "Russia-Ukraine showed the familiar pattern of high-intensity strikes and low-trust diplomacy",
            "summary": "This week, Russia and Ukraine continued high-intensity drone and missile strikes while discussions around possible talks in Turkey persisted in parallel. The defining feature was the mismatch between battlefield escalation and diplomatic language: the war continues to generate negotiation theatre without yet producing a believable pathway to de-escalation.",
            "context": "In recent weeks, both sides have used long-range strikes not only for military effect but also for negotiation positioning. That pattern continued in the reporting period. Discussions involving Turkey remained relevant because they preserve an external diplomatic channel, but the actual conflict drivers - territory, security guarantees, sanctions and war aims - remain far too large for technical talks alone to resolve.",
            "causes": "The immediate cause is that both sides still believe relative leverage can be improved before any serious political compromise is considered. The deeper cause is that this is no longer a conflict that can be managed by tactical contact alone; it sits inside broader questions of European security architecture, military-industrial capacity and alliance resolve.",
            "consequences": "In the short term, the European security environment remains fragile and prone to shock. In the medium term, markets and policymakers are forced to reassess any simplistic 'war fatigue means reduced risk' narrative. In the long term, the continued gap between diplomacy and battlefield dynamics means that temporary meetings or statements are more likely to manage expectations than to create genuine turning points.",
        },
        {
            "title": "OPEC+ expectations again moved to the center of global energy pricing",
            "summary": "As the week ended, traders and policymakers focused more sharply on OPEC+ output decisions, internal discipline and the interaction between cartel management and geopolitical disruptions. Oil was not being priced purely through demand expectations; it was being repriced through supply governance and risk premium.",
            "context": "In a weak or mixed global-growth environment, oil prices do not usually hold up well unless markets believe supply restraint and geopolitical uncertainty can offset softer demand. That is exactly what the week's discussion suggested. The market was not waiting for a single number; it was trying to infer how producers view the balance between price support, market share and fiscal needs in the second half of the year.",
            "causes": "The immediate cause was the approach of the OPEC+ decision window. The deeper cause is that in a fragmented geopolitical environment, coordinated supply signalling often matters more than textbook macro narratives. Energy is once again being priced as a political commodity, not only an economic one.",
            "consequences": "In the short term, this raises volatility across transport, chemicals, airlines and inflation-sensitive assets. In the medium term, energy-importing economies may again face renewed cost pressure if discipline holds and geopolitical risk lingers. In the long term, persistent oil-price sensitivity strengthens incentives for energy security investment and diversification, but it does not reduce the near-term power of hydrocarbon producers.",
        },
        {
            "title": "South Korea's election season highlighted regional policy uncertainty in Northeast Asia",
            "summary": "Ahead of key June voting, South Korea's domestic political debate again turned toward living costs, growth, industrial policy and external positioning. The international significance is not dramatic in headline terms, but it matters for the semiconductor chain, regional diplomacy and corporate expectations in Northeast Asia.",
            "context": "South Korea remains a critical node in semiconductors, displays, batteries and automotive manufacturing. That means its political cycle matters beyond its borders. In this reporting week, the main takeaway was not a single campaign line but the fact that industrial and foreign-policy positioning are tightly entangled with domestic legitimacy questions.",
            "causes": "The immediate cause is the election calendar, which forces clearer public commitments on social and industrial issues. The deeper cause is South Korea's structural position between security pressure, export dependence and technological competition. Domestic politics therefore naturally spills over into supply-chain and alliance questions.",
            "consequences": "In the short term, market sentiment tied to Korean assets and Korean supply-chain firms remains sensitive to political interpretation. In the medium term, chip policy, subsidy design and external balancing will remain critical watch points. In the long term, this reinforces the broader lesson that Northeast Asian industrial stability cannot be separated from political cycles.",
        },
    ]
    international = WeeklyReport(
        slug="international-news-weekly",
        title=f"{run_monday.isoformat()} International News Weekly",
        overview="The week was defined by legal uncertainty in U.S. trade policy, continuing war pressure in Gaza and Ukraine, and renewed energy-market sensitivity.",
        references=international_refs,
        sections=build_sections(
            "International News Weekly",
            coverage_text,
            archive_text,
            [
                "The U.S. trade-court ruling was the week's most consequential institutional event because it changed how businesses assess American tariff risk.",
                "Gaza remained the most severe humanitarian and diplomatic pressure point, with the central problem shifting from tactics to post-war governance viability.",
                "Russia-Ukraine continued to show a pattern of battlefield escalation paired with low-trust diplomatic signalling.",
                "OPEC+ policy expectations reminded markets that energy pricing has turned back toward supply governance and geopolitical premium.",
                "South Korea's election cycle matters as a regional policy variable because it can affect semiconductors, alliance management and market sentiment.",
                "The common thread across all major stories was not a single crisis but simultaneous repricing of legal, military and resource-policy risk.",
            ],
            [
                "The international system did not move toward stability this week. Instead, it moved deeper into a phase where legal rulings, military pressure and resource policy decisions jointly determine market expectations and diplomatic bandwidth. Governments, firms and investors all had to re-evaluate which rules still constrain behavior and which arenas are slipping into prolonged grey-zone instability.",
                "The American tariff ruling mattered because it reminded the world that domestic legal institutions can materially alter trade policy execution. Gaza and Ukraine mattered because they showed, again, that diplomacy without political architecture does not neutralize war risk. OPEC+ and Korean politics mattered because they showed that energy and industrial chains are still deeply political. Together, these stories suggest the next phase of globalization will be organized around resilience, not efficiency alone.",
            ],
            international_events,
            [
                "Geopolitics: Multiple theatres stayed active at once, increasing coordination costs for major powers and allies.",
                "Global trade: Firms must treat U.S. institutional friction as a core variable alongside tariffs themselves.",
                "Financial markets: Legal rulings and war news are again strong volatility drivers, often stronger than pure macro data.",
                "Technology industry: Semiconductor and high-end manufacturing chains remain exposed to both Northeast Asian politics and U.S. policy tools.",
                "Energy and climate: Oil pricing is once more heavily influenced by producer signalling and conflict risk rather than demand alone.",
                "Global governance: Humanitarian legitimacy and institutional capacity continue to diverge, weakening confidence in multilateral problem-solving.",
            ],
            [
                "Will the White House find alternative legal or legislative paths to preserve tariff pressure after the court ruling?",
                "Can any aid-distribution mechanism in Gaza become operationally credible and politically accepted at the same time?",
                "Do Turkey-linked contacts on Russia-Ukraine move beyond signalling into concrete agenda-setting?",
                "What exact OPEC+ output path emerges, and how does the market interpret producer discipline?",
                "How do South Korea's political signals affect semiconductor, battery and industrial-policy expectations?",
                "Will investors start explicitly pricing a world where legal uncertainty and war risk reinforce each other more often?",
            ],
            international_refs,
        ),
    )

    china_refs = [
        "国家统计局，2026-05-31，《2026年5月中国采购经理指数运行情况》。https://www.stats.gov.cn/sj/sjjd/202605/t20260531_1963825.html",
        "国家统计局，2026-05-27，《2026年1—4月份全国规模以上工业企业利润增长1.4%》。https://www.stats.gov.cn/sj/zxfb/202605/t20260527_1963753.html",
        "国家能源局，2026-05-29，《关于推进“人工智能+”行动 促进电力系统智能化发展的通知》。https://www.gov.cn/zhengce/zhengceku/202605/content_7026459.htm",
        "科技部，全国科技活动周专题页面，2026 年活动窗口。https://www.most.gov.cn/",
        "中国政府网/外交部领事司，2026-05，关于对巴西等五国试行免签政策的公告。Search reference used during drafting.",
        "Reuters, 2026-05-27 to 2026-05-31. China PMI and policy coverage. Search references used during drafting.",
    ]
    china_events = [
        {
            "title": "May PMI showed stabilization rather than strong recovery",
            "summary": "On 2026-05-31, the National Bureau of Statistics reported that the manufacturing PMI rose to 49.5, the non-manufacturing business activity index was 50.3, and the composite PMI output index was 50.4. The most important reading is not exuberance, but differentiation: manufacturing improved but remained below the boom-bust line, while services and construction continued to provide support.",
            "context": "China has been trying to balance steady growth with structural transformation. Property weakness, local-government stress, external-demand uncertainty and soft price dynamics mean macro data are watched closely for any sign of either relapse or acceleration. This week's release suggested the economy has not lost the floor, but it has not entered a convincing broad-based upswing either.",
            "causes": "The immediate drivers include the gradual transmission of earlier support policies, resilient activity in services and infrastructure-related segments, and some improvement in selected industrial categories. The deeper reason is that the economy remains in a transition from property-heavy growth toward manufacturing upgrading and services stabilization, and transitions of this kind rarely move in a straight line.",
            "consequences": "In the short term, the data support a 'steady but still fragile' macro narrative. In the medium term, further policy calibration is likely because a sub-50 manufacturing PMI still signals uneven confidence. In the long term, such data reinforce the push toward higher-quality growth where technological upgrading matters more than a simple return to old real-estate-led dynamics.",
        },
        {
            "title": "Industrial profits kept growing, with equipment manufacturing as the structural bright spot",
            "summary": "On 2026-05-27, the NBS reported that profits of above-scale industrial enterprises rose 1.4% year on year in January-April, while equipment manufacturing profits rose 11.2%. The key message is that profitability is not broadly strong, but it is concentrating in higher-value industrial segments.",
            "context": "Industrial profits are one of the best windows into the real operating quality of firms because they reflect cash generation, pricing power and investment incentives more directly than simple output figures do. This week's release matters because it showed that advanced industrial segments are still absorbing policy support and market demand more effectively than traditional low-value segments.",
            "causes": "The immediate causes include equipment-upgrade policies, continued investment in strategic sectors, and the stronger pricing or order resilience of higher-end manufacturing. The deeper cause is that China's industrial structure is gradually reallocating toward sectors where automation, systems integration and technological sophistication generate better margins.",
            "consequences": "In the short term, the data help stabilize market expectations around industrial earnings. In the medium term, they imply continued divergence between advanced manufacturing and weaker traditional segments. In the long term, they strengthen the logic for more resources flowing into industrial software, smart equipment, robotics and related engineering talent.",
        },
        {
            "title": "The 'AI plus power' policy marked a more concrete stage of industry-AI integration",
            "summary": "On 2026-05-29, the National Energy Administration issued a notice on promoting 'artificial intelligence plus' action to support power-system intelligence, proposing 50 typical application scenarios over the next three to five years. This is one of the week's most important practical policy signals because it moved AI discussion from abstract capability to sector-specific deployment.",
            "context": "Power systems are data-rich, real-time, safety-critical and optimization-heavy. That makes them one of the most realistic early landing zones for AI. The policy's significance lies in the fact that it speaks directly to planning, dispatch, operation, maintenance and security scenarios rather than just generic innovation rhetoric.",
            "causes": "The direct cause is the rising complexity of power systems as renewable penetration, grid balancing and real-time coordination requirements increase. The deeper cause is that China increasingly needs AI to prove itself inside strategic sectors where measurable efficiency and reliability gains can justify sustained investment.",
            "consequences": "In the short term, the policy sharpens expectations for projects in power informatics, operations intelligence, edge sensing and industrial software. In the medium term, it will force more attention to standards, data governance, model evaluation and safety assurance in real sectors. In the long term, it confirms that the most durable AI engineering careers will come from embedding models into complex industrial systems, not only building general chat interfaces.",
        },
        {
            "title": "National Science and Technology Week reinforced the coupling of public narrative and industrial strategy",
            "summary": "The 2026 National Science and Technology Week ran through 2026-05-31 and entered its concentrated phase during the reporting window. While often viewed as a science-popularization event, it also functions as a signal about which technologies are being integrated into the broader national narrative of modernization, industrial upgrading and strategic capability.",
            "context": "When a country repeatedly highlights AI, smart equipment, advanced manufacturing and science education in a coordinated public format, it is not merely doing outreach. It is shaping expectations among schools, local governments, firms and the public about where future opportunities, resources and social legitimacy are likely to concentrate.",
            "causes": "The immediate cause is the need to improve social understanding of innovation while supporting technology diffusion. The deeper cause is that competition in this era is not just laboratory competition; it is also competition in industrial adoption, talent pipelines and national confidence around technology direction.",
            "consequences": "In the short term, the event itself does not change macro outcomes, but it raises the visibility of strategic technologies. In the medium term, such coordination helps connect research, policy and public expectations. In the long term, it increases the value of people who can both understand systems and explain how technology is translated into practical projects.",
        },
        {
            "title": "Visa-free access for five Latin American countries continued the logic of detail-oriented opening-up",
            "summary": "China announced a trial visa-free policy for ordinary passport holders from Brazil, Argentina, Chile, Peru and Uruguay beginning 2026-06-01. Although implementation starts after the reporting week, the policy became a meaningful signal during this period because it showed continued reliance on convenience-based opening measures in a complicated external environment.",
            "context": "In a world shaped by trade friction and geopolitical fragmentation, low-cost but high-visibility measures such as visa facilitation, business-travel convenience and exhibition cooperation become important tools for maintaining external connectivity. This policy matters especially because it targets major Latin American economies rather than only traditional developed-country partners.",
            "causes": "The direct cause is China's effort to support business exchange, tourism and local cooperation. The deeper cause is that when the external environment becomes more uncertain, opening-up often advances through practical institutional details rather than grand slogans alone.",
            "consequences": "In the short term, it should support expectations for business travel and local international engagement. In the medium term, it can help deepen people-to-people and commercial ties with Latin America. In the long term, it shows that China's opening strategy will increasingly rely on operational policy tools that directly reduce friction for cross-border movement.",
        },
    ]
    china = WeeklyReport(
        slug="china-news-weekly",
        title=f"{run_monday.isoformat()} China News Weekly",
        overview="The week centered on mixed-but-stable macro indicators, structural industrial improvement, and clearer AI-to-industry policy implementation.",
        references=china_refs,
        sections=build_sections(
            "China News Weekly",
            coverage_text,
            archive_text,
            [
                "May PMI suggested stabilization without strong manufacturing reacceleration.",
                "Industrial profits confirmed that advanced manufacturing remains a structural earnings engine even when aggregate conditions stay mixed.",
                "The 'AI plus power' notice was the week's clearest sign that Chinese AI policy is moving from slogan to operational sector scenarios.",
                "National Science and Technology Week showed how technology communication and industrial policy are being linked more tightly.",
                "The visa-free policy for five Latin American countries showed that opening-up continues through practical, low-friction measures.",
                "Overall, the week's Chinese story was about improving structure rather than explosive headline growth.",
            ],
            [
                "This week's China-related story was not about one dramatic macro surprise. It was about a familiar but important policy pattern: growth support continues, yet policy language and industrial signals are becoming more concrete and sector-directed. That combination matters because it helps distinguish between short-term sentiment repair and long-term economic repositioning.",
                "The macro data showed that China still needs steadying forces. The policy signals showed where those forces are being sought: advanced manufacturing, industry-specific AI deployment, public technology mobilization and pragmatic opening-up. For someone tracking industrial technology and engineering careers, those are more informative than broad GDP-level optimism or pessimism.",
            ],
            china_events,
            [
                "Governance: Policymakers continue to accept uneven recovery while pushing more targeted structural change.",
                "Economic structure: Profitability and policy support are tilting toward advanced manufacturing and system-integration-heavy sectors.",
                "Employment and education: Composite engineering talent that can connect AI, industry and systems will likely gain value.",
                "Technology industry: Real sector deployment, standards and safety management are becoming central to the AI narrative.",
                "Social sentiment: Stable-but-not-exciting macro data support caution rather than exuberance.",
                "International relations: Convenience-based opening remains a useful tool for preserving cross-border linkages.",
            ],
            [
                "Will June high-frequency data confirm that services can keep offsetting weak manufacturing sentiment?",
                "Does the 'AI plus power' policy quickly translate into local pilots, procurement or technical standards?",
                "Can industrial-profit improvement spread beyond stronger strategic sectors into a broader business confidence recovery?",
                "How quickly will the visa-free move show up in business travel, exhibitions and local cooperation signals?",
                "Which technology themes continue to receive elevated policy visibility after Science and Technology Week ends?",
                "Do external-demand or exchange-rate changes reintroduce pressure on manufacturers despite domestic support?",
            ],
            china_refs,
        ),
    )

    personal_refs = [
        "personal-planning/agent.md",
        "personal-planning/current-positioning.md",
        "personal-planning/learning-status-2026-06.md",
        "OpenAI Help Center, 2026-05-29, ChatGPT release notes mentioning Codex on Windows and related computer-use updates. https://help.openai.com/en/articles/6825453-chatgpt-release-notes",
        "OpenAI platform/help updates, 2026-05-28, model availability and lifecycle adjustments. Search references used during drafting.",
        "Anthropic News, 2026-05-28, financing announcement. https://www.anthropic.com/news",
        "NVIDIA Blog, ICRA 2026 robotics and sim-to-real coverage. https://blogs.nvidia.com/blog/icra-research-robotics-simulation-to-real-world/",
        "国家能源局，2026-05-29，《关于推进“人工智能+”行动 促进电力系统智能化发展的通知》。https://www.gov.cn/zhengce/zhengceku/202605/content_7026459.htm",
    ]
    personal_events = [
        {
            "title": "Agent capabilities continued to become productized rather than remaining conceptual",
            "summary": "The 2026-05-29 OpenAI release-notes signal around Codex on Windows and computer-use related capabilities is important not because of one specific product name, but because it shows a broader shift: mainstream platforms are steadily moving from 'chat enhancement' toward 'task execution in real software environments'.",
            "context": "Your current planning files make it clear that you should not compete as a generic application developer or a pure model researcher. The stronger route is to become an engineer who can connect tools, workflows, state and evidence. Productized agent capabilities validate that route because they show the market increasingly values execution systems rather than only conversation quality.",
            "causes": "The immediate reason is user demand for higher-value outputs. The deeper reason is that the AI market has moved into a phase where tool calling, environment interaction, error recovery and human-in-the-loop design are becoming default expectations rather than experimental extras.",
            "consequences": "For your learning plan, this means tool-interface design, structured outputs, logs, verification and failure handling deserve more attention than prompt polish. For your future portfolio, it means a small but real agent workflow with evidence return will likely signal more value than a broad but shallow demo.",
        },
        {
            "title": "Fast model-lifecycle change makes system design more valuable than attachment to one model",
            "summary": "OpenAI's model availability and lifecycle updates around 2026-05-28 reinforced a practical lesson: models will keep changing quickly. The engineering moat is therefore shifting toward migration ability, benchmarking, orchestration and cost-aware architecture.",
            "context": "This aligns almost perfectly with your thesis strategy. In your planning notes, the strongest recommendation is system-method innovation rather than training a new foundation model. Fast model turnover strengthens that recommendation because it lowers the long-term value of obsessing over one specific base model and raises the value of robust system scaffolding.",
            "causes": "The immediate cause is intense model-market competition. The deeper cause is that AI software is maturing into systems engineering, where behavior management across models matters more than one-time API wiring.",
            "consequences": "For Topic A, this is good news: if your contribution is event indexing, evidence localization and explainable retrieval workflow, the project remains valid even when underlying models change. For June execution, it means you should prioritize baseline evaluation and data structure first, not endless experimentation with model branding.",
        },
        {
            "title": "Anthropic financing confirmed continued capital confidence in enterprise agent software",
            "summary": "Anthropic's late-May financing announcement matters less as a financial headline than as a directional signal. It suggests investors still believe enterprise-grade AI agents and workflow tools have meaningful long-horizon room to grow.",
            "context": "This is useful for your career framing because it indicates that agent engineering is not a fleeting consumer novelty. The commercial focus is shifting toward systems that can be audited, integrated and trusted inside real organizations. That is exactly the environment where someone with C++, systems taste and video/industrial context can differentiate.",
            "causes": "The immediate cause is investor belief that high-reliability enterprise AI remains underbuilt. The deeper cause is that organizations are more willing to pay for systems that automate real work than for purely decorative AI interfaces.",
            "consequences": "For your path, this is a positive signal to keep treating agent engineering plus industry context as a serious medium-term career identity. It also means your projects should emphasize reliability and utility, not just novelty.",
        },
        {
            "title": "Robotics and sim-to-real narratives kept the video-stream route strategically relevant",
            "summary": "NVIDIA's ICRA-period robotics coverage continued to stress the bridge from simulation to deployment in the real world. Even if you do not directly pursue robot control right now, this matters because robotics still depends on continuous perception, temporal evidence and system-level coordination - all areas your planned video-stream work can support.",
            "context": "Your planning files show strong interest in robotics, 3D vision and future-facing technical directions, but also a realistic concern about not diving too early into the deepest stacks. Video streams are the correct bridge layer because they let you build capability that remains useful for robotics, XR and industrial intelligence without requiring immediate control-stack specialization.",
            "causes": "The immediate reason is that embodied AI needs better perception, memory and coordination layers. The deeper reason is that real-world AI problems are rarely solved by one model alone; they require pipelines, tools and evidence management.",
            "consequences": "This validates your choice to build event indexing, retrieval and explainable question answering rather than forcing a premature jump into harder physical-control problems. It also means your long-video system can later be narrated as an upstream capability for embodied or spatial intelligence scenarios.",
        },
        {
            "title": "Industry AI policy in China highlighted the exact kind of applied route you should be building toward",
            "summary": "The 2026-05-29 'AI plus power' notice is also a personal-development signal. It demonstrates that the market and policy environment increasingly reward engineers who can place AI inside operational systems with clear evidence, safety and ROI, rather than people who only make general-purpose chat wrappers.",
            "context": "This is the strongest overlap between your local planning and the external week. Your files repeatedly say you should become an AI-enabled systems engineer who connects real video systems, industrial scenarios and applied computer vision. An industry policy like this is proof that such roles are not theoretical.",
            "causes": "The immediate cause is sector demand for measurable AI value. The deeper cause is that general AI capability must eventually be translated into process improvement, reliability and decision support inside real workflows.",
            "consequences": "For your thesis, this gives you more confidence to present the agent layer as the orchestration and explainability layer of a usable vision system. For your career path, it suggests you should frame portfolio work around industry workflows: video/log input, event extraction, retrieval, diagnosis and evidence-based reporting.",
        },
    ]
    personal = WeeklyReport(
        slug="personal-development-trends-weekly",
        title=f"{run_monday.isoformat()} Personal Development Trends Weekly",
        overview="The week's relevant external signals all reinforced the same direction: tool-using agents, industry AI and evidence-oriented system design.",
        references=personal_refs,
        sections=build_sections(
            "Personal Development Trends Weekly",
            coverage_text,
            archive_text,
            [
                "Platform-level agent productization is moving closer to your desired 'workflow agent' engineering route.",
                "Fast model turnover makes your system-method thesis strategy more defensible, not less.",
                "Enterprise AI financing remains strong enough to treat agent engineering as a serious career direction rather than a passing hype cycle.",
                "Robotics and sim-to-real discussion still supports video-stream understanding as a strategic bridge capability.",
                "Chinese industry-AI policy gives your route a practical domestic scenario map rather than leaving it at a purely abstract future vision.",
                "The most important June action remains unchanged: build the first minimal evidence-based prototype instead of consuming more broad trend content.",
            ],
            [
                "Reading your planning files together, the most important thing is not to chase every frontier topic at once. Your advantage comes from combining systems thinking, C++ and video context with a gradually deepening AI/agent capability. That means the right weekly trend report should filter external information through one question: does this make the video-stream plus agent path more realistic, more valuable or more clearly expressible?",
                "This week, the answer was yes on several fronts. Product signals from OpenAI and capital signals from Anthropic both point toward more serious agent software. Robotics coverage continues to show why temporal perception and evidence management matter. China policy signals show that AI value is increasingly being measured in industry workflows. All of that strengthens, rather than weakens, your current positioning.",
            ],
            personal_events,
            [
                "Learning: finish the smallest working loop before expanding reading scope any further.",
                "Thesis: keep Topic A as the main line, with Topic D as an engineering enhancement or backup narrative.",
                "Portfolio: emphasize evidence-return, timestamps, keyframes and system structure rather than UI polish.",
                "Career identity: continue building toward an AI-enabled video/industry systems engineer profile.",
                "Research fit: stay connected to robotics and embodied AI conceptually through perception and memory layers.",
                "Risk control: avoid turning June into endless information intake without runnable artifacts.",
            ],
            [
                "Finish one minimal chain this week: video input, frame extraction, 10-30 event records, keyword retrieval and timestamped answer output.",
                "Lock down an event schema v0 and stop changing field definitions casually after that.",
                "Read and summarize two papers only if they directly improve your prototype design or evaluation thinking.",
                "Return evidence with every answer, even if the first version uses simple frame paths and manual descriptions.",
                "Translate one current work pain point into a possible Topic D style video-plus-log diagnosis mini-case.",
                "Start shaping a clean repository structure for the prototype so it can become a real portfolio asset.",
            ],
            personal_refs,
        ),
    )
    return [international, china, personal]


def write_sources_markdown(path: Path, report: WeeklyReport) -> None:
    lines = [f"# {report.title}", "", f"- Overview: {report.overview}", "", "## References", ""]
    lines.extend(f"- {ref}" for ref in report.references)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-date", required=True, help="Run date in YYYY-MM-DD format.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_day = datetime.strptime(args.run_date, "%Y-%m-%d").date()
    run_monday, coverage_start, coverage_end = compute_dates(run_day)
    output_dir = Path(__file__).resolve().parent / run_monday.isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    notes_dir = output_dir / "notes"
    notes_dir.mkdir(exist_ok=True)

    reports = build_reports(run_monday, coverage_start, coverage_end)
    created_at = datetime.combine(run_day, datetime.min.time()).isoformat() + "Z"
    qa_results = []
    files = []

    for report in reports:
        path = output_dir / f"{run_monday.isoformat()}_{report.slug}.docx"
        write_docx(path, report.title, report.sections, created_at)
        qa_results.append(quality_check(path))
        files.append(str(path))
        write_sources_markdown(notes_dir / f"{run_monday.isoformat()}_{report.slug}_sources.md", report)

    summary = {
        "run_date": run_day.isoformat(),
        "output_monday": run_monday.isoformat(),
        "cover_start": coverage_start.isoformat(),
        "cover_end": coverage_end.isoformat(),
        "output_dir": str(output_dir),
        "qa_mode": "structural_and_render_attempt",
        "render_status": "failed",
        "render_failure_reason": "Documents render_docx.py could not run in this environment because the local Python runtime lacks pdf2image, and no alternate local renderer was detected.",
        "files": files,
        "qa_results": qa_results,
    }
    (output_dir / "qa_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
