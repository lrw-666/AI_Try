# Design Report Template Contract

## Reference

- Retained DOCX: `C:/Users/10746/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-design-report/assets/reference.docx`
- SHA-256: `ba1e11258ff52659318a321462a5e598c7bed33cf991329eb91150788fcd1a7b`
- Rendered page count: 6
- Section count: 2
- Reference renders: `_report-work/reference-render/page-1.png` through `page-6.png`
- Style evidence: `_report-work/template-style-evidence.json`
- Package inventory: `_report-work/package-inventory.json` (18 ZIP parts with byte sizes and SHA-256 values)
- Rendering note: LibreOffice is not installed. Reference pages were exported read-only with hidden Microsoft Word COM and rasterized with the bundled Poppler runtime.

## Page System

- Page size: US Letter portrait, 8.50 x 11.00 inches in both sections.
- Margins: 1.00 inch left, right, top, and bottom in both sections.
- Header distance: 0.50 inch.
- Footer distance: 0.50 inch.
- Section 1: cover, `NEW_PAGE`, no visible header or footer content.
- Section 2: body, `NEW_PAGE`, independent header and footer.
- Body header: thin gray horizontal rule, report title on the left, date on the right.
- Body footer: right-aligned PAGE field.
- Odd/even and first-page variants: disabled.

## Typography

- Visual character: black hierarchy, white background, restrained gray rules and fills, no decorative accent color.
- Latin font family: Helvetica Neue.
- Chinese extension: use Microsoft YaHei for East Asian glyphs while keeping Latin font mappings source-derived.
- Title: source `Title` style, 40 pt, bold, 1.158 line spacing, 5 pt after, keep with next.
- Heading 1: Helvetica Neue 24 pt, regular, black, 18 pt before, 10 pt after, keep with next and keep together. Source examples begin a new page.
- Heading 2: Helvetica Neue 15 pt, regular, black, 14 pt before, 6 pt after, keep with next and keep together.
- Heading 3: Helvetica Neue 12 pt, bold, black, 10 pt before, 4 pt after, keep with next and keep together.
- Body: source normal role resolving to Helvetica Neue 11 pt, black, approximately 1.15 line spacing.
- Table text: Helvetica Neue 9 pt; headers bold; body regular.
- Source list: normal role, 11 pt, using the source bullet numbering definition.
- Chinese typography deviation: East Asian font mapping is an explicit content-language requirement; size, weight, spacing, and hierarchy remain source-derived.

## Lists and Numbering

- The reference does not expose named `List Bullet` or `List Number` paragraph styles.
- Bullet and numbered paragraphs use real `w:numPr` numbering on normal paragraphs.
- Numbered recommendation pattern: 0.375 inch left indent, -0.188 inch hanging indent, 9 pt after, 1.15 line spacing.
- Reuse the existing bullet and numbered `numId` definitions by cloning source paragraph properties; do not type bullet symbols or manual number prefixes.

## Tables

### Cover Metadata Table

- One row, three columns.
- Grid widths: 4305, 901.67, and 4153.33 DXA as stored by the source.
- Left cell: subtitle, up to two short lines.
- Middle cell: intentional spacer.
- Right cell: preparer label and month/year on two lines.
- No visible borders.

### Evidence Table

- Four source rows, three columns; reusable by cloning.
- Grid widths: 1800, 3300, and 4020 DXA.
- Header row: 9 pt bold text with light gray fill.
- Body rows: 9 pt regular text, fine gray grid, vertically centered.
- Use only for genuinely comparable records. Long narrative belongs in prose or appendices.
- New multi-page tables may reuse the same fill, border, text, and padding pattern with deliberate column widths totaling the 9360 DXA body width.
- Header rows must repeat; rows must expand naturally and must not use fixed heights.

## Components

### Cover

- Preserve `word/media/image1.png`, its relationship, crop, anchor, and 6.60 x 6.37 inch treatment.
- Replace the first Title-style paragraph only.
- Replace text in the existing cover metadata table only.

### Contents

- Source TOC is inside the first structured document tag in `word/document.xml`.
- Preserve the TOC field instruction and update field results with Microsoft Word after authoring.
- The visible `Contents` paragraph uses the 24 pt Heading 1 visual role with a subtle rule/fill treatment and 18 pt before/after.

### Body Header

- `word/header1.xml` contains structured document tag `goog_rdk_0` with the report-title and date slots.
- Patch only the visible text nodes, preserving the content control, rule, paragraph properties, relationships, and header geometry.

### Footer

- Preserve the PAGE field in `word/footer1.xml` and its right alignment.

### Lead Callout

- Source normal paragraph with 0.18 inch left indent, gray fill, left vertical rule, 8 pt before, 10 pt after, and 1.12 line spacing.
- First run is bold label; subsequent run is regular explanation.
- Reuse for decisions, caveats, completion gates, and key takeaways only.

## Content Flow

1. Cover.
2. Contents page.
3. Executive summary and at-a-glance decisions.
4. Background and updated judgment.
5. Key findings and implications.
6. Agent knowledge architecture.
7. Recommendations and seven-week intensive plan.
8. Topic funnel and two-year roadmap.
9. Conclusion.
10. Appendix and references.

## Slot Map

| Location | Stable locator | Purpose | Action |
|---|---|---|---|
| Cover title | `word/document.xml`, first `Title` paragraph after the drawing paragraph | Report title | Rewrite in place |
| Cover subtitle | First table, row 1, cell 1 | Two-line scope statement | Rewrite in place |
| Cover preparer/date | First table, row 1, cell 3 | Authoring context and month | Rewrite in place |
| Contents label | First body paragraph after section break, source style/direct-format pattern | Front-matter label | Rewrite in place |
| TOC | First `w:sdt` in `word/document.xml` | Navigation | Preserve field; refresh with Word |
| Body header | `word/header1.xml`, `w:sdt` tag `goog_rdk_0` | Short title and date | Patch visible text nodes only |
| Body placeholder content | Body nodes from source Executive summary through Template note | Main report | Remove placeholder nodes and insert source-pattern clones |
| Evidence table | Second source table | Comparable findings | Clone pattern, then populate |
| Footer page number | `word/footer1.xml`, PAGE field | Navigation | Preserve |

## Package Preservation

- Preserve-only unless an explicit slot above requires change: `[Content_Types].xml` relationships, theme, font table, numbering definitions, image relationship, `word/media/image1.png`, footer field structure, web settings, and core drawing parts.
- Editable: `word/document.xml`, visible body content, cover metadata cell text, and body-header visible text.
- Styles and numbering may be read and referenced but must not be replaced with a generic preset.
- `word/settings.xml` may change only as part of a Word field refresh or `w:updateFields` setting.
- The final package may add no unexplained media, comments, tracked changes, or content controls.
- Compare the final package to `_report-work/package-inventory.json`; any missing preserve-only part or relationship is a fidelity failure.

## Fidelity Gates

- Retained reference SHA-256 remains unchanged.
- Final section count remains 2; page size, margins, and header/footer distances remain exact.
- Cover image remains visually identical and in the same first-page position.
- Body pages retain the source header rule, title/date slots, and right-aligned page number.
- Title, Heading 1, Heading 2, Heading 3, body, callout, list, and table patterns remain recognizably source-derived.
- TOC and PAGE fields remain structurally present and display correct final values after Word refresh.
- Every final page is exported and visually inspected at 100 percent zoom.
- No clipping, overlap, missing Chinese glyphs, broken tables, boundary-hugging cell text, unexplained blank pages, or awkward page breaks.
