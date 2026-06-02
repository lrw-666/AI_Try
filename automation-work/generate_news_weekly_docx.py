from __future__ import annotations

import argparse
import json
import os
import textwrap
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List
from xml.sax.saxutils import escape


BODY_FONT = "Calibri"
EAST_ASIA_FONT = "Microsoft YaHei"


@dataclass
class Paragraph:
    text: str
    style: str = "Body"


@dataclass
class Report:
    slug: str
    title: str
    sections: List[Paragraph]
    references: List[str]
    overview: str


def compute_dates(run_day: date) -> tuple[date, date, date]:
    monday = run_day - timedelta(days=run_day.weekday())
    cover_start = monday - timedelta(days=7)
    cover_end = monday - timedelta(days=1)
    return monday, cover_start, cover_end


def normalize_block(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return textwrap.dedent(stripped).strip()


def split_paragraphs(text: str, style: str = "Body") -> List[Paragraph]:
    parts = [normalize_block(part) for part in text.split("\n\n")]
    return [Paragraph(part, style) for part in parts if part]


def xml_paragraph(text: str, style: str) -> str:
    lines = [escape(line) for line in text.split("\n")]
    runs = []
    for index, line in enumerate(lines):
        if line:
            runs.append(
                "<w:r><w:rPr><w:rFonts w:ascii=\"{0}\" w:hAnsi=\"{0}\" "
                "w:eastAsia=\"{1}\"/></w:rPr><w:t xml:space=\"preserve\">{2}</w:t></w:r>".format(
                    BODY_FONT, EAST_ASIA_FONT, line
                )
            )
        if index < len(lines) - 1:
            runs.append("<w:r><w:br/></w:r>")
    if not runs:
        runs.append("<w:r/>")
    return (
        "<w:p>"
        f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>"
        + "".join(runs)
        + "</w:p>"
    )


def xml_page_break() -> str:
    return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"


def make_document_xml(title: str, paragraphs: Iterable[Paragraph]) -> str:
    body = []
    body.append(xml_paragraph(title, "Title"))
    for paragraph in paragraphs:
        if paragraph.style == "PageBreak":
            body.append(xml_page_break())
        else:
            body.append(xml_paragraph(paragraph.text, paragraph.style))
    sect = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" "
        "w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "<w:cols w:space=\"708\"/>"
        "<w:docGrid w:linePitch=\"360\"/>"
        "</w:sectPr>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:w15=\"http://schemas.microsoft.com/office/word/2012/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 w15 wp14\">"
        "<w:body>"
        + "".join(body)
        + sect
        + "</w:body></w:document>"
    )


def make_styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="{BODY_FONT}" w:hAnsi="{BODY_FONT}" w:eastAsia="{EAST_ASIA_FONT}"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
        <w:lang w:val="zh-CN" w:eastAsia="zh-CN"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="160" w:line="276" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Body"/>
    <w:uiPriority w:val="10"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="0" w:after="120" w:line="300" w:lineRule="auto"/>
      <w:jc w:val="center"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="{BODY_FONT}" w:hAnsi="{BODY_FONT}" w:eastAsia="{EAST_ASIA_FONT}"/>
      <w:b/>
      <w:sz w:val="36"/>
      <w:szCs w:val="36"/>
      <w:color w:val="1F1F1F"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Body"/>
    <w:pPr>
      <w:spacing w:before="0" w:after="180" w:line="276" w:lineRule="auto"/>
      <w:jc w:val="center"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="{BODY_FONT}" w:hAnsi="{BODY_FONT}" w:eastAsia="{EAST_ASIA_FONT}"/>
      <w:sz w:val="22"/>
      <w:szCs w:val="22"/>
      <w:color w:val="555555"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Body"/>
    <w:uiPriority w:val="9"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="360" w:after="160"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="{BODY_FONT}" w:hAnsi="{BODY_FONT}" w:eastAsia="{EAST_ASIA_FONT}"/>
      <w:b/>
      <w:sz w:val="30"/>
      <w:szCs w:val="30"/>
      <w:color w:val="2E4A62"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Body"/>
    <w:uiPriority w:val="9"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="260" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="{BODY_FONT}" w:hAnsi="{BODY_FONT}" w:eastAsia="{EAST_ASIA_FONT}"/>
      <w:b/>
      <w:sz w:val="26"/>
      <w:szCs w:val="26"/>
      <w:color w:val="334E68"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Body">
    <w:name w:val="Body"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:after="160" w:line="276" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="{BODY_FONT}" w:hAnsi="{BODY_FONT}" w:eastAsia="{EAST_ASIA_FONT}"/>
      <w:sz w:val="22"/>
      <w:szCs w:val="22"/>
      <w:color w:val="1F1F1F"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Quote">
    <w:name w:val="Quote"/>
    <w:basedOn w:val="Body"/>
    <w:pPr>
      <w:ind w:left="420" w:right="180"/>
      <w:spacing w:after="140" w:line="276" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:color w:val="444444"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def make_core_xml(title: str, created_at: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties
    xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:dcterms="http://purl.org/dc/terms/"
    xmlns:dcmitype="http://purl.org/dc/dcmitype/"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(title)}</dc:title>
  <dc:creator>Codex Automation</dc:creator>
  <cp:lastModifiedBy>Codex Automation</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created_at}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created_at}</dcterms:modified>
</cp:coreProperties>
"""


def make_app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex Automation</Application>
</Properties>
"""


def make_settings_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="420"/>
</w:settings>
"""


def write_docx(path: Path, title: str, paragraphs: Iterable[Paragraph], created_at: str) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>
"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", package_rels)
        zf.writestr("word/document.xml", make_document_xml(title, paragraphs))
        zf.writestr("word/styles.xml", make_styles_xml())
        zf.writestr("word/settings.xml", make_settings_xml())
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("docProps/core.xml", make_core_xml(title, created_at))
        zf.writestr("docProps/app.xml", make_app_xml())


def build_reports(monday: date, cover_start: date, cover_end: date, generated_on: date) -> List[Report]:
    period_text = f"统计周期：{cover_start.isoformat()} 至 {cover_end.isoformat()}（Asia/Shanghai）"
    generated_text = f"生成日期：{generated_on.isoformat()}    归档周目录：{monday.isoformat()}"

    international_sections: List[Paragraph] = [
        Paragraph(period_text, "Subtitle"),
        Paragraph(generated_text, "Subtitle"),
        Paragraph("", "PageBreak"),
        Paragraph("目录", "Heading1"),
        Paragraph("1. 执行摘要\n2. 本周国际大事总览\n3. 重点事件一：伊朗谈判与核不扩散体系承压\n4. 重点事件二：加沙停火脆弱与人道瓶颈\n5. 重点事件三：G7 财长会议与全球经济安全议程\n6. 重点事件四：第79届世界卫生大会与全球治理\n7. 影响评估\n8. 下周观察清单\n9. 参考来源", "Quote"),
        Paragraph("执行摘要", "Heading1"),
        *split_paragraphs(
            """
            本周国际局势的主轴不是单一战场，而是几条相互交织的安全与治理压力线同时抬升。第一条线是伊朗方向：5月22日至24日，围绕停火、核问题和制裁安排的谈判继续推进，但联合国《不扩散核武器条约》审议大会未能达成一致，说明即便战场强度暂缓，核治理的制度性裂缝仍在扩大。第二条线是加沙方向：5月21日联合国安理会讨论显示，停火并未自动转化为稳定治理，谁来治理战后加沙、如何恢复基本公共服务以及如何保障稳定的人道通道，仍是悬而未决的核心问题。

            第三条线是全球经济安全议程。5月19日 G7 财长和央行行长公报继续把全球失衡、供应链安全、对乌支持和经济胁迫放在同一框架下讨论，这意味着地缘政治正在更深地嵌入财政、金融、产业和贸易政策。第四条线是全球卫生治理。5月18日至23日召开的第79届世界卫生大会，一方面延续了疫情后对卫生系统韧性的追问，另一方面也把气候、药物警戒和财政约束带入全球公共产品讨论。

            对中国观察者而言，本周国际环境的关键判断有三点。其一，伊朗与加沙问题都进入“军事烈度下降但政治难题上升”的阶段，短期看能源和航运风险可能边际缓和，但中期更依赖政治协议能否落地。其二，G7 与多边卫生议程都显示，未来国际合作会越来越表现为“安全化的合作”而不是传统意义上的自由扩张型合作。其三，全球治理并未坍塌，但其运行方式正在从规则扩展转向风险管理，这会深刻影响企业的跨境布局、技术合作和资金定价。
            """
        ),
        Paragraph("本周国际大事总览", "Heading1"),
        *split_paragraphs(
            """
            1. 伊朗方向的停火与谈判进入敏感博弈期。美国政府释放“协议大体框架已谈妥”的信号，但美国国内强硬派和伊朗方面都对后续安排保持高度警惕。与之并行的，是联合国核不扩散审议机制在5月23日无果而终，反映出大国与地区安全议题已难在传统军控平台内顺利吸纳。

            2. 加沙问题仍是全球舆论与外交焦点。安理会5月21日讨论强调停火脆弱、治理悬空、援助受阻以及西岸局势恶化等问题。OCHA 近期通报则持续显示，供水、卫生、发电与跨境通行约束仍在削弱援助效率。

            3. G7 财长会议把全球经济、对乌融资、供应链韧性、经济胁迫和绿色转型整合到统一政策语言中。这说明发达经济体并未回到单纯增长导向，而是继续用安全与价值链重塑框架看待全球化。

            4. 第79届世界卫生大会延续“卫生系统韧性 + 气候健康 + 药物警戒”的议程组合。全球卫生已不再只是公共卫生官僚体系内部的话题，而是与财政约束、国际合作信任和危机治理能力直接相连。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点事件一：伊朗谈判与核不扩散体系承压", "Heading1"),
        Paragraph("背景脉络", "Heading2"),
        *split_paragraphs(
            """
            伊朗问题在本周呈现出“军事后果未清、政治谈判先行”的典型特征。AP 在5月22日至24日连续报道中提到，美国国务卿鲁比奥称谈判取得“轻微进展”，特朗普则在5月23日表示有关伊朗和霍尔木兹海峡的安排“已大体谈妥”，但美国国内共和党强硬派迅速质疑谈判可能放松对伊朗的结构性约束。与此同时，伊朗方面把外部军事打击与恢复谈判并置处理，显示其策略重心是争取安全缓冲与制裁空间。

            更值得注意的是，5月23日结束的联合国《不扩散核武器条约》审议大会未能形成最终共识。表面上看，停火谈判与审议大会分属不同轨道；实质上，两者揭示的是同一个问题：地区核风险越来越难被传统国际制度单独消化，必须嵌入大国交易、地区安全与能源通道安排中。
            """
        ),
        Paragraph("本周进展与前因后果", "Heading2"),
        *split_paragraphs(
            """
            从时间线看，5月22日是观察窗口的起点。鲁比奥对外释放“轻微进展”，意味着谈判没有破裂，但也远未形成稳定安排。5月23日，特朗普公开称协议框架已大体成型，强调霍尔木兹海峡重开与局势降温的重要性。到5月24日，美国国内批评声浪上升，表明即便白宫想以“有限成果”结束一轮危机，其国内政治成本也不低。

            从因果链条看，当前伊朗议题至少有四层逻辑。第一层是战场与海运风险，尤其是霍尔木兹海峡对全球能源运输的敏感性。第二层是核能力与核查安排，这是所有停火之后无法回避的实质问题。第三层是美国国内政治与盟友协调，如果协议被视为“只换来短暂平静”，后续执行会持续受掣肘。第四层是制度层面，当 NPT 审议大会无法达成一致时，意味着国际社会缺乏一个可以普遍接受的承压阀门。
            """
        ),
        Paragraph("影响评估", "Heading2"),
        *split_paragraphs(
            """
            短期影响是市场会继续围绕海峡通行、能源价格和停火文本的可信度交易情绪。只要霍尔木兹海峡未再出现大规模中断，油价风险溢价可能边际下降，但不会消失。

            中期影响在于核问题是否被纳入一个可执行、可核查、可被主要行为体接受的框架。如果没有，地区局势很可能从高烈度冲突转为低烈度对抗，反复消耗国际注意力和各方资源。

            长期影响则是全球核治理制度信誉进一步削弱。NPT 审议大会再次无果，会促使更多国家在现实安全困境下转向双边威慑、灰色地带行动和不完全透明的能力保留，而不是依赖统一的制度性安排。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点事件二：加沙停火脆弱与人道瓶颈", "Heading1"),
        Paragraph("背景脉络", "Heading2"),
        *split_paragraphs(
            """
            5月21日联合国安理会围绕加沙未来的讨论表明，当前问题已经从“如何停火”扩展为“停火之后谁治理、谁保障基本秩序、谁承担重建责任”。联合国新闻稿明确指出，停火仍然脆弱，西岸局势恶化、平民伤亡、人道需求攀升和治理真空同时存在。

            OCHA 在5月初和5月下旬的连续情况通报则从执行层面补充了这一点：即便局部暴力下降，发电机、燃油、零部件、发动机油和通行许可等瓶颈仍然导致供水、卫生和医疗系统处于脆弱状态。也就是说，加沙问题的难点已从单纯“是否有援助”转变为“援助系统是否具备最低运转条件”。
            """
        ),
        Paragraph("本周进展与前因后果", "Heading2"),
        *split_paragraphs(
            """
            本周安理会辩论的核心不在于新的突破，而在于各方承认现实：战后加沙的治理安排没有自然答案。联合国在5月21日的会议叙述中提到，讨论将聚焦“谁治理加沙以及如何治理”，这说明武装、地方行政、外部安全保障和援助协调之间仍未形成稳定的政治结构。

            从因果角度看，加沙困局之所以久拖不决，不仅因为停火脆弱，更因为三类约束叠加。第一类是通行和物流约束，导致援助供给效率低、系统性服务难恢复。第二类是治理约束，没有清晰的政治主体承担持续秩序维护与公共管理。第三类是信任约束，各方都担心临时安排被对手利用，因此更愿意保留否决权而非前推执行。
            """
        ),
        Paragraph("影响评估", "Heading2"),
        *split_paragraphs(
            """
            短期看，加沙问题继续作为中东风险的重要变量存在，但市场对其定价更多依赖是否外溢到海运、能源与周边国家。

            中期看，如果治理架构迟迟不能形成，援助疲劳、设施损耗和地区政治极化会同步加深。届时即使停火维持，也难以转化为稳定恢复。

            长期看，加沙问题会继续侵蚀国际秩序中的人道规范信誉，并放大联合国体系在执行层面的约束。它还会持续影响阿拉伯国家、欧美国家与地区力量之间的政策协调。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点事件三：G7 财长会议与全球经济安全议程", "Heading1"),
        Paragraph("背景脉络", "Heading2"),
        *split_paragraphs(
            """
            5月19日发布的 G7 财长和央行行长公报，是理解当前西方主要发达经济体如何把“宏观经济治理”与“国家安全治理”合并观察的关键文本。公报的参与方不仅包括 G7 自身，还包括 IMF、世界银行、OECD、FSB、IEA、ADB 和 AfDB 等机构，说明议程设计本身就具有明显的体系整合作用。

            与传统的财长会议相比，2026年的公报更像是一份“风险协调声明”。它不只讨论通胀和增长，还延伸到对乌融资、经济安全、供应链韧性、发展融资和绿色转型，说明全球主要经济体已将宏观政策空间视为地缘竞争的一部分。
            """
        ),
        Paragraph("本周进展与前因后果", "Heading2"),
        *split_paragraphs(
            """
            本周公报延续了几项清晰信号。其一，继续支持乌克兰及相关融资安排，表明欧洲战场仍是 G7 财金合作的重要粘合剂。其二，围绕全球失衡与非市场政策工具的讨论没有降温，意味着未来贸易与产业摩擦不会显著回落。其三，发展中国家被纳入对话范围，但方式更偏向“规则内嵌式参与”，而非重新定义框架。

            造成这一局面的根源，在于发达经济体越来越难把效率、韧性与安全分开处理。疫情、战争、能源波动与技术竞争之后，财政和央行部门也必须正面应对供应链、制裁、关键矿产和基础设施韧性问题。换言之，金融官员不再只是危机后的“善后者”，而正在成为地缘经济规则的直接塑造者。
            """
        ),
        Paragraph("影响评估", "Heading2"),
        *split_paragraphs(
            """
            短期内，G7 公报本身不会改变全球增长轨迹，但它会继续影响市场对产业补贴、对外投资审查、跨境税制和能源融资的政策预期。

            中期内，经济安全逻辑会加快全球产业链区域化和友岸化布局，跨国企业将更频繁地在效率与政治可接受性之间做平衡。

            长期看，全球化不会终止，但将更多体现为“分层互联”：高敏感领域趋向阵营化，低敏感领域保持交易性开放，国际机构更多承担风险协调而非强制整合功能。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点事件四：第79届世界卫生大会与全球治理", "Heading1"),
        Paragraph("背景脉络", "Heading2"),
        *split_paragraphs(
            """
            5月18日至23日，第79届世界卫生大会在日内瓦召开。WHO 总干事在开幕致辞中强调当前全球卫生体系面临财政紧张、系统韧性不足以及新风险交织的现实；同期气候与健康边会、药物警戒决议等安排则显示，全球卫生议程正在从疫情后恢复转向“复杂风险治理”。

            这一议程的重要性在于，它既折射各国对多边合作仍有需求，也暴露多边治理资源与政治意愿之间的落差。与传统认知不同，卫生治理现在同时承载气候适应、公共财政、供应链稳定和社会韧性的含义。
            """
        ),
        Paragraph("本周进展与前因后果", "Heading2"),
        *split_paragraphs(
            """
            本周大会有两个值得关注的方向。第一，是把卫生系统韧性与气候风险连接起来。WHO 关于健康与气候的活动明确提到，要在多边场域推进连续性的政治承诺，这说明公共卫生不再被视作孤立部门政策。第二，是5月24日通过加强药物警戒的决议，这反映出全球药品与疫苗体系经历冲击后，各国更加重视不良反应监测、监管协同和安全信任。

            从因果逻辑看，疫情后各国卫生系统虽然摆脱了急性危机，但并未回到稳定常态。财政紧张限制了扩容能力，气候变化带来新的健康风险，国际合作信任不足又削弱了共享效率，因此大会讨论越来越从“扩张愿景”转向“韧性底线”。
            """
        ),
        Paragraph("影响评估", "Heading2"),
        *split_paragraphs(
            """
            短期影响主要体现在公共卫生与医药监管政策继续趋向审慎和标准化，相关企业要更加重视合规、追踪和国际协调。

            中期影响在于，多边卫生合作会更多围绕气候健康、药品安全、供应韧性和能力建设展开，而不是回到单一疾病治理框架。

            长期影响是全球治理议题的“交叉化”进一步加深。卫生、气候、产业和金融政策之间的边界会越来越模糊，国际组织的价值更多体现在风险协调与议程对接能力。
            """
        ),
        Paragraph("综合影响评估", "Heading1"),
        *split_paragraphs(
            """
            地缘政治维度上，本周最重要的变化不是哪一方获得了决定性优势，而是多条危机线都进入“高不确定、低确定性成果”的区间。伊朗、加沙和乌克兰等议题都显示，危机管理比危机解决更接近现实。

            经济贸易维度上，G7 继续推进经济安全化议程，意味着全球企业将面对更复杂的关税、补贴、投资审查和供应链冗余要求。能源与金融市场维度上，只要霍尔木兹海峡和中东局势不再显著升级，风险溢价可能下行，但脆弱性仍高。全球治理维度上，卫生与核不扩散两条线都说明，多边机制仍然必要，但其效能越来越依赖大国协调与执行条件，而不是仅靠制度文本本身。
            """
        ),
        Paragraph("下周观察清单", "Heading1"),
        *split_paragraphs(
            """
            1. 伊朗停火谈判是否出现可验证的文本化安排，尤其是核问题与海峡通行安排是否被纳入。

            2. 加沙方向是否出现新的治理框架提案，以及跨境通道和关键物资准入能否获得实质改善。

            3. G7 公报后的执行动作，特别是对俄融资安排、关键供应链和经济安全工具是否有进一步政策细化。

            4. 世界卫生大会之后，各国是否把卫生韧性与气候适应、药物警戒和财政承诺转化为可执行政策。
            """
        ),
    ]
    international_refs = [
        "AP News, Rubio reports 'slight progress' in Iran talks as Pakistan renews efforts to mediate a peace deal, 2026-05-22, https://apnews.com/article/c4be639e938fa57533f28f9fd62fb43b",
        "AP News, Trump says a deal with Iran and opening of Strait of Hormuz are 'largely negotiated', 2026-05-23, https://apnews.com/article/1c283f26d037102cc5e6f798546d0e59",
        "AP News, Republicans who have drawn a hard line on Iran pan Trump's emerging proposal to end the war, 2026-05-24, https://apnews.com/article/7894b2f0e6459cddbcdaaaef5d5f1850",
        "AP News, Conference at UN to review nuclear nonproliferation treaty fails to reach agreement, 2026-05-23, https://apnews.com/article/c500d22a0d91972546acafdfbe358b72",
        "United Nations, Security Council LIVE: Ambassadors debate future of Gaza amid stalled progress on lasting peace, 2026-05-21, https://www.un.org/en/security-council-live-ambassadors-debate-future-gaza-amid-stalled-progress-lasting-peace",
        "OCHA / UNISPAL, Humanitarian Situation Report - 1 May 2026, published 2026-05-02, https://www.un.org/unispal/document/ocha-humanitarian-situation-report-1-may-2026/",
        "UN Secretary-General noon briefing, Gaza supply restrictions update, 2026-05-26, https://www.un.org/sg/en/content/highlight/2026-05-26.html",
        "Council of the European Union, G7 Finance Ministers' and Central Bank Governors’ communiqué, 2026-05-19, https://www.consilium.europa.eu/en/press/press-releases/2026/05/19/g7-finance-ministers-and-central-bank-governors-communique-19-may-2026/",
        "WHO, Director-General's opening remarks at the 79th World Health Assembly, 2026-05-18, https://www.who.int/news-room/speeches/item/who-director-general-s-opening-remarks-at-the-79th-world-health-assembly-high-level-welcome---18-may-2026",
        "WHO, Health, climate change, air quality and energy at the 79th World Health Assembly, 2026-05-18, https://www.who.int/news-room/events/detail/2026/05/18/default-calendar/health-and-climate-change-at-the-79th-world-health-assembly",
        "WHO, 79th World Health Assembly adopts resolution to strengthen pharmacovigilance, 2026-05-24, https://www.who.int/news/item/24-05-2026-79th-world-health-assembly-adopts-resolution-to-strengthen-pharmacovigilance",
    ]
    international_sections.append(Paragraph("参考来源", "Heading1"))
    international_sections.extend(Paragraph(ref, "Body") for ref in international_refs)

    china_sections: List[Paragraph] = [
        Paragraph(period_text, "Subtitle"),
        Paragraph(generated_text, "Subtitle"),
        Paragraph("", "PageBreak"),
        Paragraph("目录", "Heading1"),
        Paragraph("1. 执行摘要\n2. 本周中国相关大事总览\n3. 重点事件一：4月经济数据与增长结构\n4. 重点事件二：5月LPR不变与政策节奏\n5. 重点事件三：中美经贸关系在峰会后进入执行观察期\n6. 重点事件四：APEC贸易部长会与区域合作议程\n7. 重点事件五：中俄元首会晤与外交平衡\n8. 影响评估\n9. 下周观察清单\n10. 参考来源", "Quote"),
        Paragraph("执行摘要", "Heading1"),
        *split_paragraphs(
            """
            本周中国相关议题的核心，不是单一刺激政策，而是“增长韧性、政策克制与外部环境管理”三者之间的再平衡。5月18日国家统计局发布的1至4月和4月单月数据表明，中国经济仍保持稳中有进，但内需偏弱、外部变化加深和部分行业经营困难等问题没有消失。装备制造、高技术制造、信息服务和机器人相关产出继续较快增长，显示增长结构的“新动能”特征更加突出。

            货币政策方面，5月20日新一期 LPR 继续维持 1 年期 3.00%、5 年期以上 3.50% 不变。市场层面通常把这理解为“宽松取向未变，但政策暂不急于再加码”。这意味着决策层更希望先观察前期政策、财政投放、地方执行与外部冲击之间的传导效果，而不是连续用价格工具回应所有压力。

            外部关系方面，本周中国面对的是两个不同方向的经贸与外交窗口。其一，是特朗普 5 月中旬访华后，中美经贸共识进入执行观察期；农业采购与论坛机制有助于暂稳关系，但结构矛盾并未消失。其二，是 5 月 22 日至 23 日在苏州举行的 APEC 贸易部长会议以及 5 月 20 日至 21 日的中俄元首北京会晤，前者凸显中国希望继续维持亚太多边合作平台，后者则强调在大国博弈中维持战略纵深与外交回旋空间。
            """
        ),
        Paragraph("本周中国相关大事总览", "Heading1"),
        *split_paragraphs(
            """
            1. 4月宏观数据总体平稳，但结构性差异明显。统计局指出外部环境变化影响加深、供强需弱矛盾仍突出，但高技术制造、装备制造、信息服务和机器人相关指标继续走强。

            2. 5月 LPR 按兵不动。央行层面没有释放急迫降息信号，显示政策重心更偏向观察流动性、财政配合与实体传导，而不是频繁调整基准利率。

            3. 中美关系进入峰会后执行期。农业采购承诺、经贸论坛等有利于缓和局部压力，但稀土、科技、台湾和全球地缘冲突仍可能影响关系稳定性。

            4. APEC 贸易部长会议在苏州举行，中国商务部强调在复杂环境下寻求最大公约数，说明中国仍把区域多边合作视为对冲全球经济碎片化的重要平台。

            5. 中俄元首会晤延续“战略协作 + 外交平衡”的基调。对中国而言，其意义不只是双边关系，而是如何在中美摩擦、中东风险和亚太竞争中维持多线外交能力。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点事件一：4月经济数据与增长结构", "Heading1"),
        Paragraph("背景脉络", "Heading2"),
        *split_paragraphs(
            """
            5月18日，国家统计局同时发布“1—4月份国民经济保持稳中有进发展态势”和新闻发言人答记者问。官方口径一方面强调经济展现出韧性和活力，另一方面明确承认外部环境变化影响加深、国内供强需弱矛盾突出、部分行业和企业仍较困难。这种“双向表达”值得重视，因为它说明当前政策评估逻辑并非单纯追求短期高增速，而是在增长、结构优化和风险控制之间做平衡。

            数据层面最有代表性的信号，是增长动能继续向高技术制造、装备制造、信息服务和服务消费倾斜。统计局披露，1—4 月高技术制造业增加值同比增长 12.6%，计算机通信和其他电子设备制造业增加值增长 14%，机器人减速器、工业机器人产量分别增长 73.3% 和 25.7%。这意味着“新动能”不再只是口号，而是在工业链条上出现更清晰的量化体现。
            """
        ),
        Paragraph("本周进展与前因后果", "Heading2"),
        *split_paragraphs(
            """
            如果把这组数据放回更大背景中看，可以发现当前中国经济的主要矛盾正在从“总量是否企稳”转向“结构性修复是否足够快”。就业和物价总体稳定，为政策保持耐心提供了条件；但供给扩张快于有效需求修复、部分行业盈利承压，以及外部不确定性提高，又限制了经济向上斜率。

            前因是过去一年稳增长政策、制造业升级、出口韧性和新型消费共同支撑了基本盘；后果则是政策层面更有可能继续采用“财政更积极、货币更灵活但不急转”的组合。也就是说，当前宏观政策并不需要证明经济已经没有问题，而是要证明在复杂环境下可以维持可持续的修复路径。
            """
        ),
        Paragraph("影响评估", "Heading2"),
        *split_paragraphs(
            """
            短期看，这组数据有助于稳定市场对增长下限的判断，尤其是制造业升级和服务消费仍在提供支撑。

            中期看，真正决定修复质量的仍是内需能否持续改善、房地产与地方财政约束是否继续缓解，以及新动能是否能带动更广泛的就业与收入预期。

            长期看，统计局数据再次印证中国经济的竞争力正更多来自制造业升级、数字化和工程化能力，但这也意味着未来政策必须更重视如何把产业升级转化为居民部门的广泛获得感。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点事件二：5月LPR不变与政策节奏", "Heading1"),
        Paragraph("背景脉络", "Heading2"),
        *split_paragraphs(
            """
            5月20日，新一期 LPR 公布，1 年期和 5 年期以上品种均维持不变。新华社直接给出“均未调整”的权威表述，路透调查与市场报道则说明，市场事前已普遍预期本月维持不变。这意味着政策制定者并不认为当前流动性和融资成本需要通过立刻下调基准来再度强化。

            这并不等于货币政策转向收紧。相反，统计局在 5 月 18 日发布会上仍把当前政策概括为“更加积极的财政政策和适度宽松的货币政策”。问题不在方向，而在节奏：当银行体系流动性相对充裕、前期稳增长政策仍在传导时，利率工具的边际收益未必足以覆盖其对汇率、银行利润和资产定价的副作用。
            """
        ),
        Paragraph("本周进展与前因后果", "Heading2"),
        *split_paragraphs(
            """
            从政策机制上看，LPR 持平反映了两个判断。第一，政策层希望继续保留操作空间，以便在外部冲击或内需回落超预期时再出手。第二，监管层并不愿意把“稳增长”简化为单一的降息叙事，而更重视财政、信贷结构、产业政策和消费支持的协同。

            这一安排的前因，是一季度以来增长仍有韧性、就业物价尚稳以及市场利率环境并不紧张；后果则是未来一段时间内，市场将更关注专项债、消费支持、房地产去库存安排以及科技和制造业定向支持，而不是单独押注基准利率调整。
            """
        ),
        Paragraph("影响评估", "Heading2"),
        *split_paragraphs(
            """
            短期影响是市场对宽松预期边际降温，但对政策总量取向不会出现明显逆转判断。

            中期影响在于，如果内需修复偏慢、外需受压或地缘风险抬升，利率工具仍可能被重新启用；但在那之前，政策层更可能优先运用结构性和财政性工具。

            长期看，LPR 多月持平也说明中国宏观调控更强调“组合拳”的稳定性，而非每月通过利率信号塑造市场情绪。这对企业和居民意味着，应更关注政策组合的方向，而不是单一价格变量。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点事件三：中美经贸关系在峰会后进入执行观察期", "Heading1"),
        Paragraph("背景脉络", "Heading2"),
        *split_paragraphs(
            """
            虽然特朗普访华主要发生在 5 月 14 日至 15 日，但其后续影响持续落在本周。AP 5 月 18 日报道称，中国同意提高对美国牛肉和禽类等农产品采购，白宫将其表述为峰会后释放给美国农业州的重要成果。新华社 5 月 15 日则强调，两国元首达成一系列新的共同认识，提出建设具有战略稳定性的中美关系。

            需要注意的是，这种“阶段性共识”不应被误解为结构矛盾已经缓解。路透在峰会前后报道中反复提到，经贸之外的伊朗、台湾、稀土与技术限制仍是议程中的敏感变量。也就是说，本周真正值得观察的不是峰会语言本身，而是这些共识能否转化为低摩擦执行期。
            """
        ),
        Paragraph("本周进展与前因后果", "Heading2"),
        *split_paragraphs(
            """
            从前因看，中美双方都有稳定局部关系的现实动机。美国需要在农业、通胀与国际危机应对上争取可展示成果；中国则希望在外部不确定性上升时减少双边关系的额外扰动，为内需修复和多边布局争取时间。

            但从后果看，峰会后的共识只能带来“执行观察窗口”，难以自动带来“关系重置”。农业采购是相对容易落地的低敏感成果，经贸论坛机制有助于提升沟通频率，但涉及技术封锁、地缘安全、产业补贴和地区热点的深层矛盾，仍可能在后续数周重新抬头。
            """
        ),
        Paragraph("影响评估", "Heading2"),
        *split_paragraphs(
            """
            短期看，中美关系边际缓和有助于压低市场对极端关税升级和供应链突发中断的预期。

            中期看，若执行期内缺乏更多可量化成果，贸易与科技摩擦可能再次主导舆论。尤其在美国选举政治和亚太安全议题交织时，关系稳定性依旧有限。

            长期看，中美关系更可能维持“竞争中的有限管理”状态。对中国政策与企业而言，关键不是押注关系全面改善，而是利用窗口期做好多元化布局和风险冗余。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点事件四：APEC贸易部长会与区域合作议程", "Heading1"),
        Paragraph("背景脉络", "Heading2"),
        *split_paragraphs(
            """
            5月22日至23日，2026年 APEC 贸易部长会议在苏州举行。新华社英文报道引述商务部长王文涛的表态称，会议积极寻求各方参与经济和贸易合作的最大公约数。这一措辞本身就很能说明问题：在全球贸易碎片化、供应链安全化和大国竞争强化背景下，中国仍希望维持亚太区域合作的平台稳定性。

            APEC 的意义不在于其能立即解决所有分歧，而在于它提供了一个相对包容、议题面广且便于“先达成部分共识”的多边场域。对中国而言，主办“APEC中国年”相关活动，也是在用区域合作叙事对冲全球地缘政治叙事。
            """
        ),
        Paragraph("本周进展与前因后果", "Heading2"),
        *split_paragraphs(
            """
            会议本周释放的主要信号，是中国在复杂国际环境下仍试图把对外经济政策放在“合作可持续、分歧可管理”的框架中推进。相比双边博弈，多边平台更容易让中国强调规则、发展和开放议程。

            从前因看，全球经济正在经历安全逻辑强化与区域化重组；从后果看，中国会更积极地利用 APEC、RCEP 以及其他亚太合作平台维持贸易与投资流动性。这既有经济含义，也有外交含义：多边平台越有生命力，中国在处理中美竞争时的回旋空间就越大。
            """
        ),
        Paragraph("影响评估", "Heading2"),
        *split_paragraphs(
            """
            短期看，APEC 平台有助于稳定市场对亚太经贸合作仍可持续的判断。

            中期看，如果中国能够把会议成果与数字贸易、绿色转型、供应链韧性等议题衔接起来，将有助于增强区域规则塑造能力。

            长期看，亚太多边机制能否继续有效，将成为判断中国外部经济环境韧性的关键变量之一。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点事件五：中俄元首会晤与外交平衡", "Heading1"),
        Paragraph("背景脉络", "Heading2"),
        *split_paragraphs(
            """
            5月20日至21日，普京访华并与习近平举行会晤。路透在会前报道中强调，此次“茶叙外交”会被放在特朗普刚结束访华的大背景下审视。对中国而言，这次会晤不是简单的礼节性互动，而是对外传递“在处理中美关系的同时，中国仍保有独立而多线的战略协作能力”。

            中俄关系的实际内容仍以经贸、能源、安全协作和国际议题协调为主。但从国际感知角度看，它的象征意义更大：当全球主要力量关系同步重组时，北京需要证明自己既能与华盛顿管理竞争，也不会放弃与莫斯科维持战略协作。
            """
        ),
        Paragraph("本周进展与前因后果", "Heading2"),
        *split_paragraphs(
            """
            前因是中美峰会刚结束、伊朗与中东局势仍未完全稳定、俄乌战场问题继续外溢到全球能源与金融议程。后果则是中俄会晤被外界自然解读为中国在大国关系之间进行平衡布局的一部分。

            对中国来说，这种平衡布局的意义至少有三层：一是稳住北方战略纵深与能源合作预期；二是在联合国与全球治理议题上保留协调空间；三是在面对西方经济安全议程时，尽量避免自身被单线压缩。
            """
        ),
        Paragraph("影响评估", "Heading2"),
        *split_paragraphs(
            """
            短期看，会晤有助于稳定外界对中俄关系延续性的判断。

            中期看，中国仍需处理好与美国缓和局部关系、与俄罗斯保持合作、同时避免被卷入更深地缘对抗之间的张力。

            长期看，这类会晤的真正价值不在于单次成果，而在于它构成中国多线外交和战略自主的一部分制度化节奏。
            """
        ),
        Paragraph("综合影响评估", "Heading1"),
        *split_paragraphs(
            """
            政府治理维度上，本周最鲜明的特点是政策没有大开大合，而是以稳住预期、观察传导为主。经济结构维度上，新动能继续增强，但如何把制造与技术升级转化为更广泛的需求改善，仍是下一步关键。就业与教育维度上，若高技术制造、信息服务和机器人产业继续扩张，将为青年技术人才与工程人才创造更多机会，但前提是行业景气能够持续外溢。国际关系维度上，中国正在同时经营双边缓和、多边合作和大国平衡三条线，这会成为未来几个月宏观环境的重要外生变量。
            """
        ),
        Paragraph("下周观察清单", "Heading1"),
        *split_paragraphs(
            """
            1. 4月经济数据发布后，是否出现更细化的消费、投资和产业支持政策。

            2. LPR 不变之后，财政、结构性工具和房地产相关政策是否承担更多托底角色。

            3. 中美峰会后执行期能否继续释放可量化成果，还是再次被高敏感议题打断。

            4. APEC 与其他区域平台的后续安排是否能形成更明确的亚太合作议程。

            5. 中俄高层互动后，是否在能源、经贸或国际议题协调上出现新的公开信号。
            """
        ),
    ]
    china_refs = [
        "国家统计局，1—4月份国民经济保持稳中有进发展态势，2026-05-18，https://www.stats.gov.cn/sj/zxfb/202605/t20260518_1963732.html",
        "国家统计局，新闻发言人就2026年4月份国民经济运行情况答记者问，2026-05-18，https://www.stats.gov.cn/sj/sjjd/202605/t20260518_1963741.html",
        "国家统计局，2026年4月份居民消费价格同比上涨1.2%，2026-05-11，https://www.stats.gov.cn/sj/zxfb/202605/t20260511_1963659.html",
        "新华社，新一期贷款市场报价利率（LPR）5月20日出炉，均未调整，2026-05-20，https://www.news.cn/20260520/2b9cf295a0204d41b876dc8133f75eb5/c.html",
        "Reuters via MarketScreener, China leaves lending benchmarks unchanged for 12th month in May, 2026-05-20, https://www.marketscreener.com/news/china-leaves-lending-benchmarks-unchanged-for-12th-month-in-may-ce7f5ad8dc80f026",
        "AP News, China agrees to boost trade for US beef and poultry following Trump-Xi summit, 2026-05-18, https://apnews.com/article/832bafb5ca0be21e4a1d149c5db56b58",
        "新华社，习近平同特朗普达成一系列新的共同认识，2026-05-15，https://english.news.cn/20260515/ac69347774b1476dae10605206fb11fb/c.html",
        "Reuters via Investing.com, Trump, Xi set for Beijing talks with trade truce, Iran war at stake, 2026-05-14, https://www.investing.com/news/stock-market-news/trump-xi-set-for-beijing-talks-with-trade-truce-iran-war-at-stake-4686875",
        "新华社，APEC ministers' meeting seeks broadest common ground on economic, trade cooperation: MOC, 2026-05-23, https://english.news.cn/20260524/08d67fcb7d9c4f79a8926b1c9093a0e9/c.html",
        "新华社，APEC trade ministers' meeting opens in east China's Suzhou, 2026-05-22, https://english.news.cn/20260522/cd7b86dc7ec6473ca53d04a129fb276d/c.html",
        "Reuters via Investing.com, Xi, Putin to meet in Beijing for tea diplomacy after Trump visit, 2026-05-20, https://www.investing.com/news/commodities-news/xi-putin-to-meet-in-beijing-for-tea-diplomacy-after-trump-visit-4699947",
    ]
    china_sections.append(Paragraph("参考来源", "Heading1"))
    china_sections.extend(Paragraph(ref, "Body") for ref in china_refs)

    personal_sections: List[Paragraph] = [
        Paragraph(period_text, "Subtitle"),
        Paragraph(generated_text, "Subtitle"),
        Paragraph("", "PageBreak"),
        Paragraph("目录", "Heading1"),
        Paragraph("1. 个人规划关联摘要\n2. 执行摘要\n3. 本周方向变化总览\n4. 重点进展一：Google I/O 2026 与 Agent 平台化\n5. 重点进展二：Gemini Robotics-ER 1.6 与具身推理\n6. 重点进展三：LongVidSearch / Event-Causal RAG / SVFSearch\n7. 重点进展四：Reachy Mini 应用商店与开源物理智能体\n8. 对个人学习与研究的启发\n9. 下周建议行动清单\n10. 参考来源", "Quote"),
        Paragraph("个人规划关联摘要", "Heading1"),
        *split_paragraphs(
            """
            根据 `personal-planning/agent.md`、`personal-planning/current-positioning.md` 和 `personal-planning/learning-status-2026-06.md`，当前最清晰的主线已经不是泛泛地“学 AI”，而是把 C++ 视频系统、长视频理解、事件检索、Agent 工作流和部分具身智能前沿连接成一个可执行的能力组合。你最值得押注的不是从零去和纯算法研究者比数学强度，而是做“视频流 + Agent + 工程系统”的应用型研究与产品化原型。

            近期规划里已经明确：主候选论文方向是 Topic A“长视频流语义事件索引与 Agent 检索问答”，强相关的工程型候选是 Topic D“视频系统的帧-日志联合诊断 Agent”。这决定了外部观察的优先级应该是：第一，看长视频检索、证据定位、Video-RAG 和 Agent 规划是否有更可实现的方法；第二，看平台侧是否正在提供更低门槛的 Agent 编排能力；第三，看机器人与具身智能前沿中哪些能力能为“物理世界中的 Agent 推理”提供范式参考，而不是直接把自己拖进高成本机器人控制栈。
            """
        ),
        Paragraph("执行摘要", "Heading1"),
        *split_paragraphs(
            """
            本周最重要的变化，是“Agent 从功能点走向平台层”的趋势继续强化。Google I/O 2026 不再把 AI 简单包装成辅助编码，而是明确提出 Antigravity 2.0 和 Antigravity CLI 这样的 agent-first 平台。对你的规划来说，这意味着未来做视频检索、证据回溯和系统诊断时，难点会更多落在任务拆解、工具编排、证据校验和成本控制，而不是只调用一个大模型接口。

            第二个关键变化，是 Google DeepMind 把 Gemini Robotics-ER 1.6 明确定位为高层推理模型，并强调多视角理解、空间推理、任务规划、成功检测、仪表读数以及调用外部工具的能力。这一信息非常重要，因为它印证了你当前的 thesis framing 是合理的：未来高价值系统不一定靠训练新模型取胜，而很可能靠“现成模型 + 可解释工具链 + 证据结构”的系统方法取胜。

            第三个关键变化，是长视频与视频检索研究正在更明显地转向“代理式检索规划”和“事件化记忆组织”。LongVidSearch 提供了多跳证据检索规划基准，Event-Causal RAG 强调事件因果链和双向检索，SVFSearch 则说明即使在短视频场景，真正困难的也不是单帧识别，而是面向任务的检索与知识调用。这三者都直接支持你六月的主目标：做一个视频输入 -> 事件记录 -> 简单检索 -> 基于证据回答的最小闭环。
            """
        ),
        Paragraph("本周方向变化总览", "Heading1"),
        *split_paragraphs(
            """
            1. Agent 工程正在从“嵌入一个助手”升级为“搭建一层工作流操作系统”。这与你做视频检索与系统诊断工具链高度相关。

            2. 具身智能前沿正在把“推理模型”和“执行模型”分层。对你来说，这提供了一个有价值的设计启发：论文和项目完全可以把推理层与执行层解耦，而不必一开始就端到端做重模型。

            3. 视频研究前沿越来越强调证据访问接口、检索规划和事件结构，而不是单次问答准确率。这与 Topic A 的系统方法创新方向高度一致。

            4. 开源机器人生态正在探索“应用商店 + 代理生成 + 浏览器模拟器”模式，说明物理智能体的门槛正在下降，但最有价值的仍是任务封装与交互工作流，而不是单一硬件本体。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点进展一：Google I/O 2026 与 Agent 平台化", "Heading1"),
        Paragraph("本周信号", "Heading2"),
        *split_paragraphs(
            """
            Google Developers Blog 在 5 月 19 日的 I/O 2026 Developer keynote 总结中明确表示，行业已经从“AI simply assists you”转向“agents that can independently navigate complex tasks across your entire workflow”。核心发布点不是某个单一模型，而是 Gemini 3.5 系列与 Antigravity 2.0、Antigravity CLI 等 agent-first 开发平台升级。

            这类表述的意义，在于大型平台厂商开始把 Agent 视为开发基础设施的一层，而不是插件功能。对于你的规划，这意味着未来真正稀缺的能力，不是会不会写一个 prompt，而是能否把复杂任务拆成可调用的工具节点、状态节点、验证节点和输出节点。
            """
        ),
        Paragraph("对个人路线的启发", "Heading2"),
        *split_paragraphs(
            """
            你正在规划的视频流语义索引系统，本质上也应该按平台思路来设计：不要把它做成“一个巨大的问答入口”，而要拆成若干工具，例如视频切片、帧抽取、ASR/字幕抽取、事件记录器、索引检索器、证据定位器、回答合成器、结果验证器。这样既符合当下 Agent 平台的主流方向，也更适合硕士阶段用有限时间做出可解释、可调试、可演示的系统。

            另一个启发是 CLI 化。你当前工作背景和学习节奏决定了，可自动化、可脚本化的流程最适合积累作品。相比追求一个看上去很“智能”的前端，优先把视频处理、事件抽取和检索问答做成稳定的命令行闭环，更容易形成论文原型和工程资产。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点进展二：Gemini Robotics-ER 1.6 与具身推理", "Heading1"),
        Paragraph("本周信号", "Heading2"),
        *split_paragraphs(
            """
            Google DeepMind 在 4 月 14 日发布、并在 5 月持续传播的 Gemini Robotics-ER 1.6，有两个细节非常值得你关注。第一，它被明确描述为“高层推理模型”，能够进行视觉与空间理解、任务规划和成功检测，并能够原生调用 Google Search、VLA 模型或第三方函数。第二，它把“仪表读数”作为一个新能力单独强调，这本质上是在说明机器人场景里的价值不只来自运动控制，也来自对真实世界证据的结构化读取与解释。

            这与你的个人路线之间有很强的映射关系。你并不一定要做机器人，但你完全可以把“视频系统中的 Agent”理解为一种弱具身智能：系统需要读取外部世界的时序证据、调用工具、判断任务是否完成、并给出可追溯解释。换句话说，视频流检索问答和具身推理在“系统方法论”上并不远。
            """
        ),
        Paragraph("对个人路线的启发", "Heading2"),
        *split_paragraphs(
            """
            你可以从中提炼出一个重要 thesis principle：不要只做回答，要做“成功检测”和“证据确认”。比如，当 Agent 回答“某个事件是否发生过”时，系统不仅给答案，还应返回时间戳、关键帧、相关字幕片段、置信度说明以及为何排除了其他候选事件。这种设计会让你的工作更接近“应用计算机视觉 + AI 工程”，而不是浅层聊天机器人。

            此外，Gemini Robotics-ER 1.6 所体现的“推理层调用执行层”的双层结构，也很适合你当前有限时间与有限算力条件。你的原型不必训练新模型，而应优先证明：一个通用多模态/语言模型，能否通过工具编排，在视频系统中完成足够好的事件检索、证据回溯和解释。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点进展三：LongVidSearch / Event-Causal RAG / SVFSearch", "Heading1"),
        Paragraph("本周信号", "Heading2"),
        *split_paragraphs(
            """
            近期最契合你主线的研究信号，来自三个方向。LongVidSearch 提出长视频多跳证据检索规划基准，重点不是最终答案，而是要求 Agent 通过统一工具接口进行迭代检索与规划，从而把“检索失败”和“推理失败”分离开。Event-Causal RAG 则把长视频复杂场景的推理建立在事件记忆与因果链条之上，强调双向检索和关联视频证据。SVFSearch 则把任务压到一个更具体的短视频垂域：即便是一帧搜索，真正难点依然是知识密集型检索和规划，而不是单纯视觉识别。

            三个工作指向同一个结论：未来视频智能系统的核心竞争力，不在于把整段视频一次性塞给模型，而在于先把视频变成“可检索、可规划、可证据化”的中间表示。这个结论与你六月计划中的 event schema、10-30 条事件记录和 evidence-based answer 完全一致。
            """
        ),
        Paragraph("对个人路线的启发", "Heading2"),
        *split_paragraphs(
            """
            你接下来完全可以把 Topic A 的最小系统做成三层。第一层是视频到事件：用 FFmpeg、ASR、关键帧和人工规则先生成轻量事件记录。第二层是事件到检索：先做关键词与规则检索，再考虑向量或混合检索。第三层是检索到回答：让 Agent 显式调用“查找候选事件”“定位片段”“返回证据”“生成答案”几个工具，而不是一步到位回答。

            从论文表述看，这也让你的创新点更清晰。你不是声称训练出更强的视频大模型，而是在有限资源下提出一个证据可追踪、评估可拆解、成本可控制的长视频事件检索问答框架。这比“又一个视频问答 demo”更有说服力。
            """
        ),
        Paragraph("", "PageBreak"),
        Paragraph("重点进展四：Reachy Mini 应用商店与开源物理智能体", "Heading1"),
        Paragraph("本周信号", "Heading2"),
        *split_paragraphs(
            """
            Hugging Face 在 5 月初发布 Reachy Mini agentic robotics appstore 文章，核心信息不是某个单一机器人能力，而是把“用自然语言描述需求 -> AI agent 编写和测试代码 -> 发布到机器人”做成一个开放默认的生态闭环。再结合其文档中“应用商店 powered by Hugging Face Spaces”的设计，可以看到一个很有意思的趋势：物理智能体正在复制软件生态里的模板化分发与社区迭代模式。

            对你来说，这条线不一定是近期主战场，但它非常适合作为灵感来源。它说明“智能体价值”越来越来自应用封装、接口设计和工作流体验，而不是只有模型本身。这与你讨厌低质量重复工作、偏好系统化与可见成果的个人模式是吻合的。
            """
        ),
        Paragraph("对个人路线的启发", "Heading2"),
        *split_paragraphs(
            """
            你后续做视频 Agent 或工程诊断 Agent 时，可以借鉴这种生态思路：把不同能力做成一组可以复用的小工具或小应用。例如“抽帧工具”“日志对齐工具”“片段回放工具”“事件导出工具”“错误归因工具”。长远看，这比一次性写死一个大而全系统更容易演化成作品集和职业资产。

            同时，这也提醒你保持“弱具身、强工作流”的策略。你可以持续关注机器人和物理世界智能体，但不必现在就跳入高成本硬件路线。更现实的做法，是先在视频和软件世界里把 Agent 工具链练扎实，再决定是否把能力外延到真实硬件场景。
            """
        ),
        Paragraph("对个人学习、作品集、研究选题和职业机会的具体启发", "Heading1"),
        *split_paragraphs(
            """
            学习层面，你六月最该做的不是继续泛读更多方向，而是形成第一个最小闭环：选一个短视频，做元数据分析、抽帧、事件记录、简单检索和带证据回答。作品集层面，你应优先积累“过程可见”的小系统，而不是只有笔记没有运行结果。研究选题层面，Topic A 仍是当前最优主线，Topic D 可以作为工程增强模块或备用方案。职业机会层面，本周这些信号都说明“会用 Agent 构建工具链的工程师”价值在上升，尤其是能把 AI 接到真实视频、日志、工业流程和研发流程中的人。

            如果把中期路线再压缩一句话：你要把自己从“会一点视频业务代码的人”升级成“能把视频流、检索、证据系统和 Agent 工作流接起来的人”。这条路既比纯算法研究更适合你，也比泛化的 CRUD 开发更抗同质化。
            """
        ),
        Paragraph("下周建议行动清单", "Heading1"),
        *split_paragraphs(
            """
            1. 用一个 3 至 10 分钟的视频完成 FFmpeg 命令清单：查看元数据、抽帧、切片、导出关键片段。

            2. 设计 event schema v0。至少包括：事件 ID、时间戳范围、主体、动作、证据类型、文本摘要、置信度、来源字段。

            3. 先不用向量库，先把 10 至 30 条事件记录做成 JSON，并实现关键词检索 + 人工可读证据输出。

            4. 阅读并做笔记：LongVidSearch、Event-Causal RAG。阅读目标不是“全懂”，而是提炼接口、记忆结构和评估指标。

            5. 为 Topic D 画一个最小架构草图：视频帧、播放日志、推流/拉流状态、错误截图与诊断报告如何联动。

            6. 准备一页给导师的沟通材料，核心不是技术堆砌，而是说明你要解决的“长视频证据检索与可解释问答”问题以及为何可行。
            """
        ),
    ]
    personal_refs = [
        "项目内个人规划，personal-planning/agent.md，最近更新 2026-05-25",
        "项目内个人规划，personal-planning/current-positioning.md，日期 2026-05-22",
        "项目内个人规划，personal-planning/learning-status-2026-06.md，日期 2026-05-25",
        "Google Developers Blog, All the news from the Google I/O 2026 Developer keynote, 2026-05-19, https://developers.googleblog.com/all-the-news-from-the-google-io-2026-developer-keynote/",
        "Google DeepMind, Gemini Robotics-ER 1.6: Powering real-world robotics tasks through enhanced embodied reasoning, 2026-04-14, https://deepmind.google/blog/gemini-robotics-er-1-6/",
        "Google DeepMind, Gemini Robotics-ER 1.6 model card, published 2026-04-20, https://deepmind.google/models/model-cards/gemini-robotics-er-1-6/",
        "arXiv, LongVidSearch: An Agentic Benchmark for Multi-hop Evidence Retrieval Planning in Long Videos, 2026-03-15, https://arxiv.org/abs/2603.14468",
        "arXiv, Event-Causal RAG: A Retrieval-Augmented Generation Framework for Long Video Reasoning in Complex Scenarios, 2026-05-07, https://arxiv.org/abs/2605.06185",
        "arXiv, SVFSearch: A Multimodal Knowledge-Intensive Benchmark for Short-Video Frame Search in the Gaming Vertical Domain, 2026-05-18, https://arxiv.org/abs/2605.17946",
        "Hugging Face Blog, Introducing the agentic robotics appstore for 10,000 Reachy Minis, 2026-05-06, https://huggingface.co/blog/clem/reachymini-appstore",
        "Hugging Face Docs, Reachy Mini documentation and app ecosystem, accessed 2026-05-27, https://huggingface.co/docs/reachy_mini/v1.6.1/en",
    ]
    personal_sections.append(Paragraph("参考来源", "Heading1"))
    personal_sections.extend(Paragraph(ref, "Body") for ref in personal_refs)

    return [
        Report(
            slug="international-news-weekly",
            title=f"{monday.isoformat()} 国际新闻周报",
            sections=international_sections,
            references=international_refs,
            overview="Iran talks, the Gaza humanitarian situation, the G7 economic security agenda, and the WHO World Health Assembly.",
        ),
        Report(
            slug="china-news-weekly",
            title=f"{monday.isoformat()} 中国新闻周报",
            sections=china_sections,
            references=china_refs,
            overview="April macroeconomic data, the May LPR decision, post-summit US-China implementation, the APEC trade ministers' meeting, and China-Russia meetings.",
        ),
        Report(
            slug="personal-development-trends-weekly",
            title=f"{monday.isoformat()} 个人发展相关方向进展周报",
            sections=personal_sections,
            references=personal_refs,
            overview="Tracks agent platforms, embodied reasoning, long-video retrieval, and open robotics ecosystems around the video-stream plus agent direction.",
        ),
    ]


def quality_check(docx_path: Path) -> dict:
    result = {
        "file": str(docx_path),
        "zip_valid": zipfile.is_zipfile(docx_path),
        "required_parts": {},
        "paragraph_count_estimate": 0,
        "qa_mode": "structural_only",
    }
    required_parts = [
        "[Content_Types].xml",
        "_rels/.rels",
        "word/document.xml",
        "word/styles.xml",
        "word/settings.xml",
        "word/_rels/document.xml.rels",
        "docProps/core.xml",
        "docProps/app.xml",
    ]
    if result["zip_valid"]:
        with zipfile.ZipFile(docx_path, "r") as zf:
            names = set(zf.namelist())
            for part in required_parts:
                result["required_parts"][part] = part in names
            document_xml = zf.read("word/document.xml").decode("utf-8")
            result["paragraph_count_estimate"] = document_xml.count("<w:p>")
    return result


def write_sources_markdown(path: Path, report: Report) -> None:
    source_title = report.slug.replace("-", " ").title()
    content = [f"# {source_title}", "", f"- Overview: {report.overview}", "", "## References", ""]
    content.extend(f"- {ref}" for ref in report.references)
    path.write_text("\n".join(content), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-date", default=None, help="YYYY-MM-DD; defaults to today")
    parser.add_argument("--base-dir", default="automation-work", help="Output base directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_day = (
        datetime.strptime(args.run_date, "%Y-%m-%d").date()
        if args.run_date
        else datetime.now().date()
    )
    monday, cover_start, cover_end = compute_dates(run_day)
    base_dir = Path(args.base_dir)
    output_dir = base_dir / monday.isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    notes_dir = output_dir / "notes"
    notes_dir.mkdir(exist_ok=True)

    reports = build_reports(monday, cover_start, cover_end, run_day)
    created_at = datetime.combine(run_day, datetime.min.time()).isoformat() + "Z"

    qa_results = []
    for report in reports:
        filename = f"{monday.isoformat()}_{report.slug}.docx"
        docx_path = output_dir / filename
        write_docx(docx_path, report.title, report.sections, created_at)
        qa_results.append(quality_check(docx_path))
        write_sources_markdown(notes_dir / f"{monday.isoformat()}_{report.slug}_sources.md", report)

    summary = {
        "run_date": run_day.isoformat(),
        "output_monday": monday.isoformat(),
        "cover_start": cover_start.isoformat(),
        "cover_end": cover_end.isoformat(),
        "output_dir": str(output_dir),
        "qa_mode": "structural_only",
        "visual_qa": {
            "status": "not_completed",
            "reason": "LibreOffice/soffice unavailable and Word COM automation could not be used reliably in this environment."
        },
        "files": [result["file"] for result in qa_results],
        "qa_results": qa_results,
    }
    (output_dir / "qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
