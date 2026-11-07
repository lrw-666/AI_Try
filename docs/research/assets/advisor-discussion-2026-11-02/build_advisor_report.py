"""Build an evidence-based Chinese advisor report; no inference or training.

Run with the bundled Codex Python. All visualizations use saved RGB/GT/predictions.
Portable selected evidence and the Markdown source are kept beside research docs.
"""
from pathlib import Path
import csv, hashlib, json, shutil, re
import numpy as np
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, Color, white
from reportlab.lib.utils import ImageReader
from reportlab.lib.enums import TA_LEFT

ASSET = Path(__file__).resolve().parent
REPO = ASSET.parents[3]
ROOT = REPO.parent
DATA = ROOT / 'segformer_resources/data/visa'
EVAL = ROOT / 'segformer_resources/runs/pcb1_b0_v2_seed42_eval_acceptance_20261101/test'
REPORT = REPO / 'work/SegFormer/reports/pcb1_b0_v2_seed42'
OUT = REPO / 'output/pdf/2026-11-02-advisor-discussion-report.pdf'
SOURCE = REPO / 'docs/research/2026-11-02-advisor-discussion-report.md'
TMP = ROOT / 'tmp/pdfs/advisor-report'
for folder in (ASSET, OUT.parent, TMP): folder.mkdir(parents=True, exist_ok=True)

W, H = 595.276, 841.89
M, CW = 42, 511.276
NAVY, TEAL, BLUE, AMBER = '#14334A', '#008674', '#3269AD', '#A66B12'
INK, MUTED, LINE, PALE = '#243746', '#647583', '#DAE3E9', '#F3F6F8'
pdfmetrics.registerFont(TTFont('YaHei', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('YaHeiB', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily('YaHei', normal='YaHei', bold='YaHeiB', italic='YaHei', boldItalic='YaHeiB')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows = {r['image_id']:r for r in csv.DictReader((EVAL/'per_image.csv').open(encoding='utf-8-sig'))}
failrows = {r['image_id']:r for r in csv.DictReader((REPORT/'failure-cases.csv').open(encoding='utf-8-sig'))}
manifest = {'description':'Saved baseline evidence only. Display crops/overlays do not alter evaluation.',
            'source_run':'pcb1_b0_v2_seed42', 'source_evaluation':str(EVAL), 'samples':[]}

def sample(kind, ident, roi=None):
    key = f'pcb1/Data/Images/{kind}/{ident}.JPG'
    src, predpath = DATA/key, EVAL/'predictions'/Path(key).with_suffix('.png')
    rgb = Image.open(src).convert('RGB')
    pred = np.asarray(Image.open(predpath)) > 0
    maskpath = DATA/f'pcb1/Data/Masks/Anomaly/{ident}.png' if kind=='Anomaly' else None
    gt = np.asarray(Image.open(maskpath)) > 0 if maskpath else np.zeros(pred.shape, bool)
    actual = dict(tp=int((pred&gt).sum()),fp=int((pred&~gt).sum()),fn=int((~pred&gt).sum()))
    for k,v in actual.items(): assert v==int(rows[key][k]), (key,k,v,rows[key][k])
    assert sha(src)==failrows[key]['source_sha256']
    name = ('anomaly-' if kind=='Anomaly' else 'normal-')+ident
    shutil.copy2(src, ASSET/f'{name}-rgb.jpg')
    a = np.asarray(rgb).copy()
    overlay = a.copy()
    overlay[gt] = (0.40*a[gt]+0.60*np.array([0,195,154])).astype('uint8')
    error = (a*0.62+255*0.38).astype('uint8')
    error[pred&gt]=[0,173,116]
    error[pred&~gt]=[235,72,72]
    error[~pred&gt]=[49,127,231]
    imgs={'gt':Image.fromarray((gt*255).astype('uint8')),
          'pred':Image.fromarray((pred*255).astype('uint8')),
          'overlay':Image.fromarray(overlay), 'errors':Image.fromarray(error)}
    for suffix,im in imgs.items(): im.save(ASSET/f'{name}-{suffix}.png')
    if roi:
        for suffix,im in {'rgb':rgb,**imgs}.items(): im.crop(roi).save(ASSET/f'{name}-{suffix}-roi.png')
    entry={'image_id':key,'rgb_source':str(src),'rgb_sha256':sha(src),
           'prediction_source':str(predpath),'prediction_sha256':sha(predpath),
           'mask_source':str(maskpath) if maskpath else 'all-zero normal mask',
           'mask_sha256':sha(maskpath) if maskpath else None,'display_crop_xyxy':roi,
           'verified_counts':actual,'archived_per_image':rows[key]}
    manifest['samples'].append(entry)
    return name

sample('Normal','0001')
sample('Anomaly','049',(490,600,810,900))
g = np.asarray(Image.open(DATA/'pcb1/Data/Masks/Anomaly/003.png'))>0
gy,gx=np.where(g)
gcx,gcy=int((gx.min()+gx.max())//2),int((gy.min()+gy.max())//2)
sample('Anomaly','003',(max(0,gcx-175),max(0,gcy-165),min(1404,gcx+175),min(1070,gcy+165)))
# ROI for the normal FP is calculated from the saved prediction, with fixed context.
p = np.asarray(Image.open(EVAL/'predictions/pcb1/Data/Images/Normal/0333.png'))>0
ys,xs=np.where(p)
cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2
roi=(max(0,int(cx)-200),max(0,int(cy)-165),min(1404,int(cx)+200),min(1070,int(cy)+165))
sample('Normal','0333',roi)
shutil.copy2(REPORT/'figures/training-curves.png', ASSET/'training-curves.png')
(ASSET/'evidence-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

cv = canvas.Canvas(str(OUT), pagesize=(W,H), pageCompression=1)
cv.setTitle('毕业论文选题与研究计划 | 导师沟通报告')
cv.setAuthor('研究计划整理 | 2026-11-02')
cv.setSubject('PCB有监督缺陷分割：已完成SegFormer基线、DINOv3独立对照与证据驱动适配')
md=['# 毕业论文选题与研究计划：导师沟通报告\n', '更新日期：2026-11-02（Asia/Shanghai）。以本次核对的既有文档与保存结果为依据；本次未重新训练或评价。\n']
layout=[]
page=0

def rr(x,y,w,h,fill=PALE,stroke=None,r=6):
    cv.setFillColor(HexColor(fill)); cv.setStrokeColor(HexColor(stroke or fill))
    cv.roundRect(x,H-y-h,w,h,r,fill=1,stroke=bool(stroke))

def para(s,x,y,w=CW,size=10.2,leading=16,color=INK,bold=False,record=True):
    style=ParagraphStyle('p',fontName='YaHeiB' if bold else 'YaHei',fontSize=size,
        leading=leading,textColor=HexColor(color),wordWrap='CJK',splitLongWords=True,
        spaceAfter=0,allowWidows=0,allowOrphans=0)
    ob=Paragraph(s,style); _,hh=ob.wrap(w,1000)
    ob.drawOn(cv,x,H-y-hh)
    layout.append({'page':page,'top':y,'bottom':y+hh,'text':re.sub('<[^>]+>','',s)[:70]})
    if record: md.append(re.sub(r'<br\s*/?>','\n',re.sub('</?b>','',s))+'\n')
    return hh

def line(s,x,y,size=10,color=INK,bold=False):
    cv.setFont('YaHeiB' if bold else 'YaHei',size);cv.setFillColor(HexColor(color));cv.drawString(x,H-y-size,s)

def start(k,title,subtitle,sources):
    global page
    page+=1
    cv.bookmarkPage('page'+str(page));cv.addOutlineEntry(title,'page'+str(page),0)
    line('研究计划 / 导师沟通',M,25,8.5,MUTED)
    cv.setStrokeColor(HexColor(LINE));cv.line(M,H-47,W-M,H-47)
    line(k,M,64,9,TEAL,True)
    para(title,M,83,CW,21,28,NAVY,True,False)
    para(subtitle,M,121,CW,9.5,15,MUTED,False,False)
    cv.setStrokeColor(HexColor(LINE));cv.line(M,54,W-M,54)
    line('2026-11-02  |  讨论稿，题目与范围待导师确认',M,799,7.5,MUTED)
    cv.setFont('YaHei',8);cv.drawRightString(W-M, H-807, f'{page:02d} / 10')
    para(sources,M,768,CW,7.2,10,MUTED,False,False)
    md.append('\n---\n\n## '+str(page)+'. '+title+'\n\n'+subtitle+'\n')

def end(): cv.showPage()

def label(s,x,y,color=TEAL):
    para(s,x,y,CW,11,17,color,True)

def box(title,text,x,y,w,h,color=TEAL,fill=PALE):
    rr(x,y,w,h,fill)
    para(title,x+13,y+10,w-26,11,17,color,True)
    hh=para(text,x+13,y+35,w-26,10,15)
    assert hh<=h-42,(page,title,hh,h)

def table(headers,data,x,y,widths,font=9.2,leading=14):
    sty=ParagraphStyle('t',fontName='YaHei',fontSize=font,leading=leading,wordWrap='CJK',textColor=HexColor(INK))
    hsty=ParagraphStyle('th',parent=sty,fontName='YaHeiB',textColor=white)
    vals=[[Paragraph(str(t),hsty) for t in headers]]+[[Paragraph(str(t),sty) for t in row] for row in data]
    ob=Table(vals,colWidths=widths,hAlign='LEFT')
    ob.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor(NAVY)),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor(PALE)]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
        ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('LINEBELOW',(0,-1),(-1,-1),0.5,HexColor(LINE))]))
    _,hh=ob.wrap(sum(widths),1000);ob.drawOn(cv,x,H-y-hh)
    layout.append({'page':page,'top':y,'bottom':y+hh,'text':'TABLE '+str(headers)})
    md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |')
    for row in data: md.append('| '+' | '.join(str(t).replace('<br/>',' / ') for t in row)+' |')
    md.append('')
    return hh

def img(name,x,y,w,h,caption=None):
    p=ASSET/name
    im=Image.open(p);iw,ih=im.size;sc=min(w/iw,h/ih);dw,dh=iw*sc,ih*sc
    cv.drawImage(ImageReader(im),x+(w-dw)/2,H-y-dh,width=dw,height=dh,mask='auto')
    if caption: para(caption,x,y+h+4,w,8.5,13,MUTED)
    md.append(f'![{caption or name}](assets/advisor-discussion-2026-11-02/{name})\n')

def arrow(x1,y1,x2,y2,color=MUTED,dash=False):
    cv.setStrokeColor(HexColor(color));cv.setFillColor(HexColor(color));cv.setLineWidth(1.3)
    cv.setDash(4,3) if dash else cv.setDash()
    cv.line(x1,H-y1,x2,H-y2);cv.setDash()
    import math
    a=math.atan2(y2-y1,x2-x1)
    p=cv.beginPath();p.moveTo(x2,H-y2)
    p.lineTo(x2-6*math.cos(a-.5),H-(y2-6*math.sin(a-.5)))
    p.lineTo(x2-6*math.cos(a+.5),H-(y2-6*math.sin(a+.5)));p.close();cv.drawPath(p,fill=1,stroke=0)

# 1. One-page memo.
start('01 / 一页备忘录','论文选题：预训练视觉表征的迁移与适配','建议先读第1-3页：任务与基础 → 真实样例 → 整篇研究路线。后文为讨论依据。','依据：[L1] 开题与导师沟通指南；[L2-L3] 已验收实验卡与结果。数值为既有记录，本次未重跑。')
rr(M,159,CW,83,'#EAF5F2')
para('暂定题目',M+14,170,CW-28,9,14,TEAL,True)
para('基于预训练视觉表征的PCB外观缺陷分割方法研究',M+14,194,CW-28,15,22,NAVY,True)
para('希望研究：在有限缺陷标注与本机GPU条件下，怎样兼顾缺陷定位、正常区域误报和实现成本。',M,259,CW,11,18,INK,True)
label('任务与对象',M,313)
para('以公开VisA的PCB板级RGB外观图像为对象，使用已有像素标注进行背景/缺陷二值分割。输入为图像，输出为对应原图的缺陷掩码；当前先以PCB1建立可比较的实验协议。',M,337)
label('已完成的前期成果',M,397)
para('已完成SegFormer-B0的数据处理、迁移训练、独立评价、单图预测和可视化，形成13步工程闭环。PCB1单seed：test 442张，全局前景IoU <b>0.6833</b>、Dice <b>0.8119</b>，40张异常图macro IoU <b>0.5249</b>。仍有漏检与正常结构误报。',M,421)
label('整篇拟研究路线',M,497)
para('<b>A 已完成：</b>SegFormer独立基线 → <b>B 待实施：</b>冻结DINOv3-S/16＋简单可训练分割头独立对照 → <b>适配待选：</b>基于train/val证据选择一项改动并做必要消融。融合、预测集成与蒸馏仅作条件候选。',M,521)
label('现实约束与本次希望明确的决定',M,597)
para('在职非全日制；研究以Python/PyTorch为主，控制难度和扩展数量。本机RTX 4070 Laptop约8GB显存；DINO访问与资源可行性尚待实测。希望导师确认PCB对象、题目与研究深度、最低必要对照/消融，以及资源受阻时的替代范围。',M,621)
rr(M,699,CW,48,PALE)
para('<b>时间边界：</b>已知材料截止11月16日，答辩窗口11月16-20日；具体流程与个人场次待确认。开题前以真实基线和完整计划沟通，DINO结果不作为当前计划的前置条件。',M+11,707,CW-22,9,14)
end()

# 2. Real object examples.
start('02 / 一页样例','先看研究对象：板级RGB图像与缺陷像素','下列均为现有PCB1测试图及已保存预测。正常图与异常图为不同样本，不是成对的前后对照。','图源：VisA [R1]；样本Normal/0001、Anomaly/049；GT与预测来源及校验见[L4]和本报告evidence-manifest.json。')
img('normal-0001-rgb.jpg',M,166,264,202)
para('正常样本 / 0001.JPG',M,373,264,10,15,TEAL,True)
box('这类图像里有什么？','一块电路板及其多个电子元件、连接结构和引脚。\n研究对象是板级可见外观。\n正常图的GT为全零，本例预测也为空。',325,170,228,191,TEAL)
para('异常样本 / 049.JPG：GT标注了下方引脚区域，基线未预测出任何缺陷像素。',M,412,CW,10.5,17,NAVY,True)
xs=[M,M+174,M+348]
for x,name,title in zip(xs,['anomaly-049-rgb.jpg','anomaly-049-gt.png','anomaly-049-pred.png'],['原始RGB外观','异常GT：白色为缺陷','预测：全黑，未检出']):
    img(name,x,446,163,125,title)
img('anomaly-049-overlay-roi.png',M,601,142,133)
para('GT区域局部放大',M,736,142,8,12,TEAL)
para('<b>本例的可核查事实</b><br/>GT 10,459像素；预测0像素；FN（漏检）10,459像素；IoU与recall均为0。左图绿色为GT叠加，放大仅用于看清标注。',205,608,348,10,16)
para('像素定位便于查看缺陷的位置与范围；任务不自动输出缺陷类型。当前对象为板级外观，未评价内部、电气功能或产线合格判定。',205,693,348,9.4,15,MUTED)
end()

# 3. Whole research roadmap.
start('03 / 整体路线图','两条独立路线先比较，再决定一项适配','状态说明：绿色=已完成；蓝色=待实施；琥珀色=待证据选择；虚线=有条件才进入。','依据：[L1] §4-5、§11；[L7] 原研究方向。A/B独立训练与输出，图中汇合表示比较分析。')
box('共同任务与数据协议','PCB1固定原图划分 / 0-1标签 / 原图评价 / train-val开发与选模',M,162,CW,76,NAVY)
arrow(298,238,298,257);arrow(298,257,167,257);arrow(298,257,429,257)
arrow(167,257,167,275,TEAL);arrow(429,257,429,275,BLUE)
box('A / SegFormer-B0 · 已完成','MiT-B0预训练编码器\n＋整个新建二类解码头\n全网络微调，已有结果与失败图',M,275,249,116,TEAL,'#EAF5F2')
box('B / DINOv3-S/16 · 待实施','冻结预训练骨干\n＋简单可训练分割头\n先做可行性，再做完整独立对照',M+263,275,248,116,BLUE,'#EEF3FB')
arrow(167,391,167,411);arrow(429,391,429,411);arrow(167,411,298,411);arrow(429,411,298,411);arrow(298,411,298,428)
box('共同评价 / 定位、正常误报、逐图失败与成本','两个模型各自用val选配置和权重；方案冻结后评价。比较差异，分析何种条件下各有价值。',M,428,CW,85,NAVY)
arrow(298,513,298,535,AMBER)
box('一项适度适配 / 待train-val证据选择','从尺度或切块、轻量头或选层、监督或采样等候选中选一个主要改动；做原方案对照与必要消融，记录收益、成本及无收益条件。',M,535,CW,94,AMBER,'#FCF5E8')
arrow(298,629,298,650)
box('论文分析 / 形成可追溯的结论与适用边界','A基线＋B独立对照＋一项适配及验证，是当前讨论的主体工作量。是否满足学位深度和最低要求，请导师判断。',M,650,CW,86,NAVY)
cv.setDash(4,3);cv.setStrokeColor(HexColor(AMBER));cv.roundRect(M,H-759,CW,18,4,fill=0,stroke=1);cv.setDash()
para('条件候选：互补证据＋预算允许＋导师认可 → 融合 / 集成 / 蒸馏',M+9,743,CW-18,8.1,12,AMBER,True)
md.append('条件候选进入条件：train/val出现互补证据、预算允许且导师认可其优先级；不预设为最终方法。\n')
end()

# 4. Baseline quantitative evidence.
start('04 / 当前基线','已有结果能支撑后续研究，也暴露定位缺口','统一展示val与test。所有指标为前景二值分割口径；数值保留四位小数。','依据：[L2] 实验卡；[L3] results.md与results-summary.json；[L5] 联合验收。单类别PCB1、seed42。')
hh=table(['原图划分','正常 / 异常','用途'],[['train：530','482 / 48','训练模型参数'],['val：132','120 / 12','选择配置与best权重'],['test：442','402 / 40','方案冻结后的评价记录']],M,160,[117,119,275])
label('训练与评价配置',M,160+hh+17)
para('划分来源为官方2-class high-shot，train/val采用v2分组派生。MiT-B0预训练编码器＋新建整个二类头，共3,714,658参数，全网络微调。512×512 letterbox；CE＋0.5前景Soft Dice；50轮训练，val选第39轮best（10,330次成功更新）。恢复1404×1070原图计分，阈值0.5。',M,160+hh+42,CW,9.8,15)
y=160+hh+118
mh=table(['指标','val：132张','test：442张'],[
['全局前景IoU','0.7775','<b>0.6833</b>'],['全局Dice','0.8748','<b>0.8119</b>'],
['全局precision / recall','0.9398 / 0.8183','0.9008 / 0.7390'],
['异常图macro IoU','0.5396（12张）','<b>0.5249（40张）</b>'],
['正常图FP像素','1,495','3,689'],['有任意FP像素的正常图','11 / 120','32 / 402']],M,y,[223,144,144])
label('这些结果的实际意义',M,y+mh+17)
para('已建立从图像到缺陷掩码、再到可复核指标和失败案例的完整基准，后续模型可以在相同问题上比较。precision高于recall，提示当前协议下仍有较多缺陷像素未被找回；尚无第二模型，不能据此判断相对优劣。',M,y+mh+42,CW,9.7,15)
para('global先累计TP/FP/FN；异常macro对异常图等权。两者差值不能单独证明小缺陷更差。32/402仅指含任意FP像素的正常图，不能直接解释为产品误剔率。',M,y+mh+108,CW,8.7,13,MUTED)
end()

# 5. Visual successes / failures.
start('05 / 可视化证据','基线能定位部分缺陷，但存在漏检与误报','以下为手工选取的说明性样例，不代表整体分布；局部裁剪只用于展示，不参与重新计分。','依据：[L3-L4]；Anomaly/003、049与Normal/0333均在既有样例清单中。颜色：TP绿 / FP红 / FN蓝。')
cols=[M,M+131,M+262,M+393]
def visual_row(name,y,heading,description):
    para(heading,M,y,CW,10.6,16,NAVY,True)
    for x,suf,title in zip(cols,['rgb','gt','pred','errors'],['RGB局部','GT / 白色为缺陷','预测 / 阈值0.5','TP / FP / FN']):
        img(name+'-'+suf+'-roi.png',x,y+28,118,109,title)
    para(description,M,y+159,CW,9.2,14)
visual_row('anomaly-003',160,'01 / 部分有效：异常003，IoU 0.8278','GT 1,618像素；TP 1,457，FP 142，FN 161。此例说明当前模型已能学到有用的缺陷定位。')
visual_row('anomaly-049',357,'02 / 完全漏检：异常049，IoU 0','GT 10,459像素，预测为空。失败并非只有少量边缘误差；原因需在train/val上进一步检验。')
visual_row('normal-0333',554,'03 / 正常误报：正常0333，FP 575像素','本图GT全零，红色全部为误报。正常结构也可能被当成缺陷，因此后续不能只追求异常图分数。')
end()

# 6. Technical stack and models.
start('06 / 技术栈与学习路径','从像素数据流到表征迁移与受控实验','技术栈围绕论文问题展开。当前A已实现，B为后续计划；C++/部署仅在导师明确要求时另定边界。','依据：[L2] 实际环境；[L6] 基线学习与代码指南；模型原理：[R2-R3]。')
box('A：任务驱动微调 · 已实现','RGB → 层级MiT多尺度特征 → 轻量解码头 → 二类logits → 原图缺陷掩码。预训练编码器与新建头一起更新。',M,159,CW,91,TEAL,'#EAF5F2')
box('B：冻结表征迁移 · 待实施','RGB → 冻结DINOv3-S/16 → patch tokens恢复空间网格 → 简单可训练头 → 上采样与原图还原。下游仍使用缺陷mask监督。',M,263,CW,91,BLUE,'#EEF3FB')
y=374
hh=table(['层次','主要技术与学习内容','研究中的作用'],[
['数据与几何','Python、NumPy、Pillow；mask语义、同步变换、letterbox与原图还原','保证缺陷与像素位置一致'],
['训练与迁移','PyTorch、Transformers；预训练初始化、梯度、CE/Dice、AMP、checkpoint','理解哪些参数在学习、如何选模'],
['冻结视觉表征','ViT patch/token、空间网格、冻结与可训练头、特征接口','比较两种迁移路线，验证密集特征可用性'],
['评价与实验','IoU/Dice、precision/recall、macro、正常FP、消融、成本统计','把模型变化与效果、代价对应起来'],
['复现与沟通','Conda、CUDA、Git、配置/清单/SHA、CSV/JSON、图表与CLI','保留证据链，形成论文实验与图表']],M,y,[90,274,147],8.9,14)
para('<b>现有运行环境：</b>Python 3.12.16 / PyTorch 2.10.0+cu126 / Transformers 4.57.6 / CUDA 12.6；RTX 4070 Laptop约8GB。版本为已有实验记录，不是后续环境必须机械照搬的要求。',M,y+hh+18,CW,9.2,14)
para('S/16在512×512输入下对应32×32空间网格；需正确处理额外tokens。冻结可减少梯度开销，前向仍有成本，不能预先保证更快。A/B预训练、结构、规模和更新范围均有差异，不将结果简单归因为监督方式。',M,y+hh+71,CW,8.8,13.5,MUTED)
end()

# 7. Experiment design.
start('07 / 后续研究如何验证','用三个问题组织实验，适配由证据决定','后续方法开发、头部与超参数选择仅使用train/val；冻结方案后再评价并留档。','依据：[L1] §4-5；[L7] 研究方向。以下为拟实施方案，DINO与适配均无当前实验成绩。')
hh=table(['研究问题','拟实施实验','预期证据'],[
['RQ1：两种迁移路线\n效果与成本有何差异？','A与B使用共同原图划分和标签口径；各自用val合理选模，记录尝试范围与预算。','global / 异常macro / 正常FP；参数、显存、时间与逐图失败'],
['RQ2：一项适配\n能否缓解明确瓶颈？','先分析train/val错误，选一个主要改动；与未适配方案及必要组件消融比较。','改动对应的错误是否减少；其他指标是否恶化；收益能否重复'],
['RQ3：收益的代价\n与适用边界是什么？','在同硬件、精度与计时范围下记录成本；资源允许时有目的补seed或类别。','效果-成本取舍；负结果、失败条件及结论范围']],M,159,[135,217,159],9,14)
y=159+hh+20
label('适配候选：先观察，再选其中一项',M,y)
th=table(['待验证的失败证据','可选改动','最小必要对照'],[
['缩放后细节受损','输入分辨率或切块','原方案 / 调整后；尺度分层与成本'],
['特征有信号、简单头定位不足','一处轻量头或选层调整','简单头 / 适配头；必要组件消融'],
['前景稀少与错误有关联','一项损失或采样调整','固定其他条件，同时报告正常FP']],M,y+27,[163,162,186],8.8,13)
y2=y+27+th+18
box('条件分支：融合、预测集成或蒸馏','仅在两模型train/val表现出互补、预算允许且导师认可时进入。需同时保留两个单模型对照，并证明新增收益值得成本；这些设计尚未选定。',M,y2,CW,87,AMBER,'#FCF5E8')
para('<b>评价边界：</b>test分数和失败图已查看，后续披露该接触记录，不称完全盲测。阈值沿用0.5或按预先约定的val规则选择；新方案单独建run，不覆盖已有基线。普通切块、换头或loss组合的创新性还需文献与实验支持。',M,y2+101,CW,9.1,14)
para('<b>DINO先通过的小规模检查：</b>核权重访问、版本与SHA；检查空间token及原图几何；确认骨干参数不更新、头部有梯度且确实更新；实测显存与时间后再决定完整训练预算。',M,y2+170,CW,9.1,14)
end()

# 8. Milestones and constraints.
start('08 / 两段时间表','开题前交清楚的材料，开题后交可验证的研究','开题材料已知截止2026年11月16日；最终论文日期、中期与送审节点需按正式要求倒排。','依据：[L1] §10-12。下列日期为现有计划节点，非导师已确认承诺；工作时长为可调整假设。')
label('开题前 / 11月2日-16日',M,159)
hh=table(['节点','优先任务','可见交付'],[
['11/2-7','核对象与要求、读懂基线、阅读两条路线、联系导师','讨论包、样例、相关工作与问题清单'],
['11/8','收敛对象、题目、内容与优先级','范围与研究问题草稿'],
['11/11','形成完整报告初稿并交反馈','覆盖技术路线、实验、可行性和日程'],
['11/13-15','PPT与讲稿、计时预演、流程检查、内部定稿','可播放材料与提交备份'],
['11/16','按正式时刻提交、核回执与个人场次','提交记录；答辩窗口11/16-20']],M,187,[84,220,207],8.8,13.5)
y=187+hh+18
label('开题后 / 按依赖推进，日期待学校节点确认',M,y)
hh2=table(['阶段','工作与意义','进入下一步的交付'],[
['M1 可行性','核DINO访问、几何/token、冻结/梯度和资源；先验证能否可靠研究','真实可行性报告与预算'],
['M2 独立对照','完整train/val训练与选模，冻结后评价；回答RQ1','A/B指标、失败与成本对照'],
['M3 一项适配','train/val证据选改动，完成对照及必要消融；回答RQ2-3','效果、代价、负结果与边界'],
['M4-M5 补充与写作','资源允许再补seed/类别；章节写作并行，研究冻结后定稿','导师确认范围的完整论文']],M,y+26,[91,249,171],8.6,13)
yy=y+26+hh2+17
para('<b>预算与取舍：</b>现有计划假设工作日每天60-90分钟、周末每天2-3小时，11/2-15约18-27小时主动投入。开题前优先理解、文献、沟通与材料；开题后先以M1实测锁定训练预算。资源不足先削可选扩展，再带瓶颈与备选和导师讨论。',M,yy,CW,9.1,14)
end()

# 9. Advisor decisions and academic contribution.
start('09 / 导师讨论清单','希望本次讨论明确什么，论文如何落地','建议按“对象与题目 → 研究深度 → 必要实验 → 资源备选 → 正式节点”的顺序讨论。','依据：[L1] §2、§4.5、§7-8；[L7] 原研究方向。下表为空间与优先级建议，未视为导师已认可。')
hh=table(['需要导师决定','当前建议 / 讨论起点'],[
['1. 对象与题目','确认PCB板级外观、有缺陷mask监督是否符合方向；暂用“预训练视觉表征的PCB外观缺陷分割”。'],
['2. 论文主体与深度','A基线＋B冻结表征独立对照＋一项适配与必要消融；明确工作量与算法创新的正式要求。'],
['3. 必要实验与优先级','哪些对照/消融为必做，是否必须多seed、其他PCB类别；明确停止扩展与进入写作的条件。'],
['4. 访问或资源受阻备选','DINO权重获取、显存或时间受限时，允许哪些替代配置/模型或缩范围；先说明改变的研究问题。'],
['5. 系统与个人掌握','是否有GUI、部署等硬性要求；重点掌握数据流、梯度、选模、评价与研究决策，并如实说明辅助范围。'],
['6. 流程与时间边界','核11/16具体时刻、模板、签字审阅、个人答辩场次，以及中期、送审、最终毕业节点。']],M,159,[132,379],9.1,14)
y=159+hh+20
label('预期论文组织与研究价值',M,y)
para('绪论与相关研究 → 任务、数据和共同协议 → 预训练模型独立迁移对照 → 问题驱动适配与消融 → 综合实验与局限 → 总结。章节数量可合并，适配章的具体方法待证据决定。',M,y+27,CW,10,16)
para('已有价值是可追溯的实验基础；拟形成的价值是两种迁移路线的实证比较、一个明确问题的适配验证，以及定位-误报-成本的适用边界。研究贡献仍需文献、受控对照和消融支持。',M,y+91,CW,10,16)
rr(M,y+160,CW,69,'#EAF5F2')
para('<b>建议会后记录：</b>对象/题目____；最低必要内容____；必做对照与消融____；可选项____；资源备选及触发条件____；正式节点____；下次反馈时间____。',M+13,y+173,CW-26,9.3,15)
end()

# 10. Evidence / reading references and limitations.
start('10 / 依据与阅读入口','数字、图像与研究计划均可追溯','内部文件路径以AI_Try仓库根目录为起点；完整实验身份与小数精度以原始记录为准。','本报告整理日期：2026-11-02（Asia/Shanghai）。外部一手资料同日核对；未启动下载、训练或重新评价。')
label('内部证据与计划来源',M,158)
sources=[
('[L1] 开题、整篇论文规划与导师沟通指南','docs/research/2026-11-02-proposal-preparation-and-advisor-discussion-guide.md'),
('[L2] 当前训练、硬件与选模事实','work/SegFormer/reports/pcb1_b0_v2_seed42/experiment-card.md'),
('[L3] 指标、数学口径与结果摘要','work/SegFormer/reports/pcb1_b0_v2_seed42/results.md；同目录results-summary.json'),
('[L4] 样例选择与逐图失败统计','work/SegFormer/reports/pcb1_b0_v2_seed42/failure-cases.csv；同目录figures/'),
('[L5] 工程闭环验收记录','work/SegFormer/reports/2026-11-01-stage12-13-acceptance.md'),
('[L6] 技术栈与个人学习入口','docs/research/2026-11-02-segformer-baseline-learning-and-code-guide.md'),
('[L7] 原研究方向与衔接','docs/research/2026-10-29-research-direction-and-handoff.md')]
y=186
for title,path in sources:
    para(title,M,y,CW,9.1,14,NAVY,True)
    ph=para(path,M,y+17,CW,7.6,11,MUTED);y+=ph+25
label('外部一手资料',M,y+8)
refs=[
('[R1] Zou等，SPot-the-Difference，ECCV 2022；VisA官方数据说明','https://github.com/amazon-science/spot-diff'),
('[R2] Xie等，SegFormer，NeurIPS 2021','https://arxiv.org/abs/2105.15203'),
('[R3] Siméoni等，DINOv3，2025技术报告；官方实现与模型卡','https://arxiv.org/abs/2508.10104')]
y+=37
for title,url in refs:
    para(title,M,y,CW,9.1,14,NAVY)
    para(f'<link href="{url}" color="{BLUE}">{url}</link>',M,y+17,CW,8,12,BLUE)
    y+=43
para('<link href="https://github.com/facebookresearch/dinov3" color="#3269AD">DINOv3官方代码与模型卡：github.com/facebookresearch/dinov3</link>',M,y,CW,8,12,BLUE)
y+=28
para('<b>图像与统计来源：</b>原图/GT来自本地VisA；预测来自固定best39的已保存评价PNG。配套evidence-manifest.json记录所用样本、源路径、SHA-256、展示裁剪框，以及与per_image.csv核对的TP/FP/FN。绿色GT叠加与错误着色为显示处理，未修改预测或重新选模。',M,y,CW,8.7,13)
para('<b>当前结论边界：</b>单PCB类别、单seed，test仅40张异常；未做置信区间、稳定推理延迟、图像级判定或工业现场验证。13步闭环和已有验收支撑实现可信度，不能替代论文创新、完整工作量或学生本人理解。',M,y+65,CW,8.7,13)
end()

cv.save()
SOURCE.write_text('\n'.join(md),encoding='utf-8')
(TMP/'layout-log.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2),encoding='utf-8')
overflow=[r for r in layout if r['bottom']>758 and r['top']<768]
print(json.dumps({'pdf':str(OUT),'source':str(SOURCE),'pages':page,'overflow':overflow,'bytes':OUT.stat().st_size},ensure_ascii=False))
assert page==10
assert not overflow, overflow
