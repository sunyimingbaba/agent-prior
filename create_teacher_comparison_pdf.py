from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

OUT = Path("cup_comparison_tables_for_teacher.pdf")
pdfmetrics.registerFont(TTFont("Heiti", "/System/Library/Fonts/STHeiti Light.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("Heiti-Bold", "/System/Library/Fonts/STHeiti Medium.ttc", subfontIndex=0))
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCN", fontName="Heiti-Bold", fontSize=17, leading=22, textColor=colors.HexColor("#16324F"), spaceAfter=5))
styles.add(ParagraphStyle(name="H2CN", fontName="Heiti-Bold", fontSize=12, leading=16, textColor=colors.HexColor("#16324F"), spaceBefore=4, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyCN", fontName="Heiti", fontSize=8.5, leading=12, textColor=colors.HexColor("#253447"), spaceAfter=4))
styles.add(ParagraphStyle(name="Cell", fontName="Heiti", fontSize=8.0, leading=10, textColor=colors.HexColor("#1F2937")))
styles.add(ParagraphStyle(name="CellB", fontName="Heiti-Bold", fontSize=8.0, leading=10, textColor=colors.HexColor("#16324F")))
styles.add(ParagraphStyle(name="CellTitle", fontName="Heiti-Bold", fontSize=10, leading=12, textColor=colors.white))

def P(text, bold=False, title=False):
    return Paragraph(str(text), styles["CellTitle" if title else ("CellB" if bold else "Cell")])

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Heiti", 6.5); canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(10 * mm, 7 * mm, "CUP 对比实验表 · † 为论文报告结果，Ours 为本机结果")
    canvas.drawRightString(287 * mm, 7 * mm, f"第 {doc.page} 页"); canvas.restoreState()

def comp_table(title, rows):
    headers = ["方法", "I2T R@1", "I2T R@5", "I2T R@10", "T2I R@1", "T2I R@5", "T2I R@10", "mR", "来源 / 协议"]
    data = [[P(title, title=True)] + [P("", True)] * 8, [P(x, True) for x in headers]]
    for row in rows:
        data.append([P(x, bold=(x.startswith("Ours"))) for x in row])
    widths = [58*mm, 16*mm, 16*mm, 17*mm, 16*mm, 16*mm, 17*mm, 15*mm, 54*mm]
    t = Table(data, colWidths=widths, repeatRows=2, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("SPAN", (0,0), (-1,0)), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16324F")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#DCEAF7")),
        ("GRID", (0,0), (-1,-1), .3, colors.HexColor("#CBD5E1")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 3), ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ] + [("BACKGROUND", (0,r), (-1,r), colors.HexColor("#F8FAFC")) for r in range(2, len(data), 2)]))
    return t

DATA = {
"UCM": [
    ["LW-MCR-b", "12.38", "43.81", "59.52", "12.00", "46.38", "72.48", "41.10", "† CUP Table V · TGRS 2022"],
    ["LW-MCR-d", "15.24", "51.90", "62.86", "11.90", "50.95", "75.24", "44.68", "† CUP Table V · TGRS 2022"],
    ["LW-MCR-u", "18.10", "47.14", "63.81", "13.14", "50.38", "79.52", "45.35", "† CUP Table V · TGRS 2022"],
    ["SMLGN", "12.86", "49.52", "75.71", "14.29", "52.76", "84.67", "48.30", "† 转引第三方对比表 · GNTGRS 2024"],
    ["MSITA", "16.86", "49.33", "73.33", "14.29", "57.16", "91.58", "50.43", "† 转引第三方对比表 · GNTGRS 2024"],
    ["SSJDN", "17.86", "53.57", "72.02", "20.54", "62.56", "82.98", "51.59", "† CUP Table V · ACM TOMM 2023"],
    ["PGRN", "16.19", "51.43", "75.24", "13.62", "60.86", "95.05", "52.06", "† JSTARS 2025"],
    ["MSA", "13.33", "59.05", "77.14", "13.14", "57.52", "92.48", "52.11", "† 转引第三方对比表 · TGRS 2024"],
    ["RGAL", "17.62", "58.86", "87.24", "18.10", "61.10", "86.43", "54.89", "† 转引第三方对比表 · TGRS 2025"],
    ["PE-RSITR", "18.82", "62.84", "93.72", "22.71", "55.81", "80.33", "55.71", "† 转引第三方表 · TGRS 2023"],
    ["CLIP zero-shot", "11.90", "45.24", "72.86", "10.75", "43.48", "71.04", "42.55", "本机 · ICML 2021"],
    ["CLIP linear probe", "9.52", "46.67", "80.48", "12.54", "49.65", "80.30", "46.53", "本机 · ICML 2021"],
    ["CLIP-Adapter", "9.83", "42.52", "66.79", "14.61", "51.03", "84.14", "44.82", "† CUP Table V · arXiv 2021"],
    ["VL-Prompt", "7.96", "45.48", "75.96", "12.51", "51.57", "91.52", "47.50", "† CUP Table V · arXiv 2022"],
    ["CoOp", "7.19", "42.76", "74.19", "9.78", "48.34", "87.41", "44.95", "† CUP Table V · IJCV 2022"],
    ["VPT", "13.81", "49.05", "75.24", "14.73", "58.01", "92.44", "50.54", "† CUP Table V · ECCV 2022"],
    ["MaPLE", "13.97", "49.68", "75.24", "12.21", "52.80", "84.81", "48.12", "† CUP Table V · CVPR 2023"],
    ["CUP (CCB)", "14.92", "54.13", "80.95", "14.86", "59.80", "93.04", "52.95", "† TNNLS 2025"],
    ["PDAR-RSITR", "22.94", "60.01", "84.39", "19.56", "63.62", "95.61", "57.69", "† IJCAI 2026 · 六项均值"],
    ["Ours (Agent + tool + GRPO)", "20.00", "64.29", "88.10", "17.91", "63.98", "97.01", "58.55", "本机 · ViT-L/14 · seed=0"],
],
"RSICD": [
    ["LW-MCR-b", "4.57", "13.71", "20.11", "4.02", "16.47", "28.23", "14.52", "† CUP Table IV · TGRS 2022"],
    ["LW-MCR-d", "3.29", "12.52", "19.93", "4.66", "17.51", "30.02", "14.66", "† CUP Table IV · TGRS 2022"],
    ["LW-MCR-u", "4.39", "13.35", "20.29", "4.30", "18.85", "32.34", "15.59", "† CUP Table IV · TGRS 2022"],
    ["GaLR w/o MR", "6.50", "18.91", "29.70", "5.11", "19.57", "31.92", "18.62", "† CUP Table IV · TGRS 2022"],
    ["GaLR with MR", "6.59", "19.85", "31.04", "4.69", "19.48", "32.13", "18.96", "† CUP Table IV · TGRS 2022"],
    ["PIR", "9.88", "27.26", "39.16", "6.97", "24.56", "38.92", "24.46", "† ACM MM 2023"],
    ["DOVE", "8.66", "22.35", "34.95", "6.04", "23.95", "40.35", "22.72", "† TGRS 2024"],
    ["CLIP zero-shot", "6.68", "21.50", "33.76", "5.50", "19.43", "30.17", "19.51", "本机 · ICML 2021"],
    ["CLIP linear probe", "9.33", "31.75", "45.11", "7.46", "24.92", "38.77", "26.22", "本机 · ICML 2021"],
    ["GLISA", "19.52", "40.44", "52.28", "14.75", "39.50", "55.46", "36.99", "† TGRS 2024"],
    ["IERR", "17.29", "35.41", "48.58", "13.07", "35.32", "51.31", "33.50", "† TGRS 2024"],
    ["HarMA", "16.36", "34.48", "47.74", "12.92", "37.17", "53.07", "33.62", "† ICLR-W 2024"],
    ["VPT", "7.23", "19.40", "30.77", "6.94", "25.26", "40.73", "21.72", "† ECCV 2022"],
    ["MaPLE", "7.93", "26.78", "42.42", "6.88", "24.47", "40.12", "24.77", "† CVPR 2023"],
    ["DFIM", "23.42", "45.09", "62.63", "17.70", "46.61", "63.42", "43.14", "† IJCAI 2025 · ViT-B/32"],
    ["PDAR-RSITR", "18.76", "39.79", "54.24", "15.54", "42.46", "58.57", "38.23", "† IJCAI 2026 · 六项均值"],
    ["RRSITR (0% noise)", "15.86", "35.50", "48.54", "12.05", "34.68", "50.41", "32.84", "† CVPR 2026 · ViT-B/32 · 5 seeds"],
    ["CUP (CCB)", "13.14", "36.14", "51.36", "12.17", "35.36", "52.45", "33.43", "† TNNLS 2025"],
    ["EAPA", "20.31", "40.81", "53.34", "14.84", "39.51", "55.90", "37.45", "† ACM MM 2025"],
    ["CMER", "15.55", "35.49", "49.03", "10.66", "33.41", "51.49", "32.61", "† JAG 2026 · ViT-L/14 + BERT"],
    ["Ours (Agent + tool + GRPO)", "21.50", "51.88", "67.70", "15.72", "42.03", "59.88", "43.12", "本机 · ViT-L/14 · seed=0"],
],
"RSITMD": [
    ["LW-MCR-b", "9.07", "22.79", "38.05", "6.11", "27.74", "49.56", "25.55", "† CUP Table III · TGRS 2022"],
    ["LW-MCR-d", "10.18", "28.98", "39.82", "7.79", "30.18", "49.78", "27.79", "† CUP Table III · TGRS 2022"],
    ["LW-MCR-u", "9.73", "26.77", "37.61", "9.25", "34.07", "54.03", "28.58", "† CUP Table III · TGRS 2022"],
    ["GaLR w/o MR", "13.05", "30.09", "42.70", "10.47", "36.34", "53.35", "31.00", "† CUP Table III · TGRS 2022"],
    ["GaLR with MR", "14.82", "31.64", "42.48", "11.15", "36.68", "51.68", "31.41", "† CUP Table III · TGRS 2022"],
    ["PIR", "18.14", "41.15", "52.88", "12.17", "41.68", "63.41", "38.24", "† ACM MM 2023"],
    ["DOVE", "16.81", "36.80", "49.93", "12.20", "44.13", "66.50", "37.73", "† TGRS 2024"],
    ["CLIP zero-shot", "10.84", "28.10", "37.39", "10.23", "30.96", "45.90", "27.24", "本机 · ICML 2021"],
    ["CLIP linear probe", "18.14", "36.95", "50.44", "12.82", "39.77", "56.79", "35.82", "本机 · ICML 2021"],
    ["GLISA", "28.21", "50.94", "62.50", "20.50", "54.80", "72.19", "48.41", "† EAPA Table 1 · TGRS 2024"],
    ["IERR", "27.66", "48.89", "61.06", "24.43", "56.59", "72.66", "48.55", "† EAPA Table 1 · TGRS 2024"],
    ["HarMA", "25.81", "48.37", "60.61", "19.92", "53.27", "71.21", "46.53", "† EAPA Table 1 · ICLR-W 2024"],
    ["VPT", "14.98", "32.05", "40.15", "15.97", "41.35", "60.35", "34.14", "† ECCV 2022"],
    ["MaPLE", "16.96", "39.97", "53.02", "13.59", "43.25", "61.92", "38.12", "† CVPR 2023"],
    ["DFIM", "34.97", "57.36", "71.25", "28.72", "59.83", "77.92", "55.01", "† IJCAI 2025 · ViT-B/32"],
    ["PDAR-RSITR", "28.54", "52.34", "66.05", "26.16", "59.07", "75.78", "51.32", "† IJCAI 2026 · 六项均值"],
    ["RRSITR (0% noise)", "25.44", "46.82", "58.98", "20.88", "53.59", "70.90", "46.10", "† CVPR 2026 · ViT-B/32 · 5 seeds"],
    ["CUP (CCB)", "23.23", "45.80", "60.84", "19.84", "52.83", "71.11", "45.61", "† TNNLS 2025"],
    ["EAPA", "29.87", "53.32", "62.83", "24.16", "57.48", "71.81", "49.91", "† ACM MM 2025"],
    ["CMER", "23.67", "49.77", "63.93", "17.25", "56.41", "81.15", "48.70", "† JAG 2026 · ViT-L/14 + BERT"],
    ["Ours (Agent + tool + GRPO)", "36.50", "65.93", "75.88", "32.19", "64.04", "77.24", "58.63", "本机 · ViT-L/14 · seed=0"],
],
}

doc = SimpleDocTemplate(str(OUT), pagesize=landscape(A4), leftMargin=9*mm, rightMargin=9*mm, topMargin=10*mm, bottomMargin=13*mm)
story = [Paragraph("CUP 对比实验表", styles["TitleCN"]), Paragraph("三数据集 · 七项检索指标 · 论文报告结果与本机结果分开标注", styles["BodyCN"]), Spacer(1, 2)]
for idx, (dataset, rows) in enumerate(DATA.items()):
    story.append(comp_table(dataset, rows))
    if dataset != "RSITMD": story.append(PageBreak())
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
