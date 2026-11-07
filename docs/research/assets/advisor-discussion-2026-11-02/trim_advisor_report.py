"""Save pages 1-7 separately and update the PDF page-count labels."""
from pathlib import Path
import hashlib
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, TextStringObject

repo = Path(__file__).resolve().parents[4]
source = repo / 'output/pdf/2026-11-02-advisor-discussion-report.pdf'
target = repo / 'output/pdf/论文选题报告.pdf'
before = hashlib.sha256(source.read_bytes()).hexdigest()
reader = PdfReader(source)
assert len(reader.pages) == 10
writer = PdfWriter()
writer.append(reader, pages=(0, 7), import_outline=True)
for i, page in enumerate(writer.pages):
    stream = ContentStream(page.get_contents(), writer)
    replacements = 0
    for operands, operator in stream.operations:
        if operator == b'Tj' and operands[0] == f'{i+1:02d} / 10':
            operands[0] = TextStringObject(f'{i+1:02d} / 07')
            replacements += 1
    assert replacements == 1, (i + 1, replacements)
    page.replace_contents(stream)
writer.add_metadata({'/Title': '论文选题报告', '/Subject': 'PCB外观缺陷分割：论文选题与研究计划', '/Author': '研究计划整理'})
with target.open('wb') as output:
    writer.write(output)
result = PdfReader(target)
assert len(result.pages) == 7
assert len(result.outline) == 7
for i, page in enumerate(result.pages):
    original_text = reader.pages[i].extract_text().replace(f'{i+1:02d} / 10', f'{i+1:02d} / 07')
    assert page.extract_text() == original_text
assert hashlib.sha256(source.read_bytes()).hexdigest() == before
print(f'Verified: {len(result.pages)} pages, {len(result.outline)} bookmarks, page labels /07; original unchanged.')
print(f'Output size: {target.stat().st_size} bytes')
