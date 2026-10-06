from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle, PageBreak, SimpleDocTemplate
)


OUT = Path("output/pdf/cup_experiment_comparison_report.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)

font_path = "/System/Library/Fonts/STHeiti Light.ttc"
bold_path = "/System/Library/Fonts/STHeiti Medium.ttc"
pdfmetrics.registerFont(TTFont("Heiti", font_path, subfontIndex=0))
pdfmetrics.registerFont(TTFont("Heiti-Bold", bold_path, subfontIndex=0))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleCN", parent=styles["Title"], fontName="Heiti-Bold", fontSize=20,
    leading=26, alignment=TA_CENTER, textColor=colors.HexColor("#16324F"),
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="H2CN", parent=styles["Heading2"], fontName="Heiti-Bold", fontSize=13,
    leading=18, textColor=colors.HexColor("#16324F"), spaceBefore=8, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="BodyCN", parent=styles["BodyText"], fontName="Heiti", fontSize=9.2,
    leading=15, textColor=colors.HexColor("#253447"), spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="SmallCN", parent=styles["BodyText"], fontName="Heiti", fontSize=7.6,
    leading=11, textColor=colors.HexColor("#4B5563"),
))
styles.add(ParagraphStyle(
    name="CellCN", parent=styles["BodyText"], fontName="Heiti", fontSize=7.1,
    leading=9, textColor=colors.HexColor("#1F2937"),
))
styles.add(ParagraphStyle(
    name="CellBoldCN", parent=styles["BodyText"], fontName="Heiti-Bold", fontSize=7.1,
    leading=9, textColor=colors.HexColor("#16324F"),
))


def P(text, style="CellCN"):
    return Paragraph(text, styles[style])


def table(data, widths, header_rows=1):
    t = Table(data, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    cmds = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#DCEAF7")),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.HexColor("#16324F")),
        ("FONTNAME", (0, 0), (-1, header_rows - 1), "Heiti-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row in range(header_rows, len(data)):
        if row % 2 == 0:
            cmds.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F8FAFC")))
    t.setStyle(TableStyle(cmds))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Heiti", 7)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 10 * mm, "CUP 实验对比摘要 · 论文报告结果与本机结果分开标注")
    canvas.drawRightString(192 * mm, 10 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUT), pagesize=A4, rightMargin=14 * mm, leftMargin=14 * mm,
    topMargin=14 * mm, bottomMargin=17 * mm,
)
story = []
story += [
    Paragraph("CUP 遥感图文检索实验对比摘要", styles["TitleCN"]),
    Paragraph("给导师的快速阅读版 · 2026-09-21", styles["SmallCN"]),
    Spacer(1, 5),
    Paragraph(
        "这份摘要把当前看板中的本机实验、原版 CUP 复现，以及新增的顶会论文结果放在同一份材料中。论文数字只作为外部参照，不与本机 ViT-L/14、单 seed 结果做无条件的严格优劣结论。",
        styles["BodyCN"]),
]

story += [Paragraph("1. 主方法与本机实验", styles["H2CN"])]
main_rows = [
    [P("方法", "CellBoldCN"), P("UCM mR", "CellBoldCN"), P("RSICD mR", "CellBoldCN"), P("RSITMD mR", "CellBoldCN"), P("来源", "CellBoldCN")],
    [P("原版 CUP（CCB 先验）"), P("50.85"), P("31.15"), P("40.25"), P("本机复现，ViT-L/14，seed=0")],
    [P("纯推理 Agent（无工具、无 RL）"), P("59.49"), P("45.69"), P("68.17"), P("本机消融，上界参考")],
    [P("Ours：域内工具 + Agent + GRPO 冷启动", "CellBoldCN"), P("58.55", "CellBoldCN"), P("43.12", "CellBoldCN"), P("58.63", "CellBoldCN"), P("本机主方法，ViT-L/14，seed=0")],
]
story.append(table(main_rows, [56 * mm, 22 * mm, 22 * mm, 22 * mm, 54 * mm]))
story += [Spacer(1, 5), Paragraph("解读：本机结果的主结论是 Agent 先验带来明显增益；域内工具与 GRPO 的作用应结合注入率、奖励消融和多 seed 结果一起解释，不能只看单个 mR。", styles["BodyCN"])]

story += [Paragraph("2. 新增顶会论文结果（已写入 experiment_board.html）", styles["H2CN"])]
paper_rows = [
    [P("论文 / 会议", "CellBoldCN"), P("RSICD mR", "CellBoldCN"), P("RSITMD mR", "CellBoldCN"), P("UCM mR", "CellBoldCN"), P("使用说明", "CellBoldCN")],
    [P("DFIM · IJCAI 2025"), P("43.14"), P("55.01"), P("—"), P("ViT-B/32；原论文 Table 1，直接报告六项均值")],
    [P("PDAR-RSITR · IJCAI 2026"), P("38.23"), P("51.32"), P("57.69"), P("原论文 Table 1/2；六项均值由 I2T/T2I 六个 R@K 计算")],
    [P("RRSITR · CVPR 2026（0% 噪声）"), P("32.84"), P("46.10"), P("—"), P("ViT-B/32，5 seeds；补充材料 Table 1")],
]
story.append(table(paper_rows, [48 * mm, 22 * mm, 22 * mm, 20 * mm, 64 * mm]))
story += [Spacer(1, 4), Paragraph("出处：DFIM <link href='https://www.ijcai.org/proceedings/2025/0647.pdf' color='#2563EB'>IJCAI 原文 PDF</link>；PDAR-RSITR <link href='https://www.ijcai.org/proceedings/2026/0661.pdf' color='#2563EB'>IJCAI 原文 PDF</link>；RRSITR <link href='https://openaccess.thecvf.com/content/CVPR2026/html/Song_Robust_Remote_Sensing_Image-Text_Retrieval_with_Noisy_Correspondence_CVPR_2026_paper.html' color='#2563EB'>CVPR 官方页面</link>。", styles["SmallCN"])]

story += [Paragraph("3. 噪声实验应如何对齐", styles["H2CN"])]
noise_rows = [
    [P("项目", "CellBoldCN"), P("当前状态", "CellBoldCN"), P("注意事项", "CellBoldCN")],
    [P("看板已有曲线"), P("word-drop：0 / 20 / 40%"), P("这些是现有实验数字；60 / 80% 尚未补跑，不应外推")],
    [P("RRSITR 顶会参照"), P("训练集 noisy correspondence：0 / 20 / 40 / 60 / 80%"), P("这是图文配对错配噪声，不等同于 query word-drop；可作为鲁棒性协议参照，不能直接合并曲线")],
]
story.append(table(noise_rows, [38 * mm, 55 * mm, 83 * mm]))
story += [Spacer(1, 4), Paragraph("RRSITR 的 CVPR 2026 补充材料报告了 RSITMD、RSICD 和 NWPU 的 0% 噪声结果，并专门研究高噪声训练。看板已把它单独列为噪声对应参照；后续若补跑本项目 60 / 80% word-drop，应使用单独图例。", styles["BodyCN"])]

story += [Paragraph("4. 给导师看的结论与限制", styles["H2CN"])]
for text in [
    "(1) 主结论：在本机同骨干、同数据划分、同 seed 的条件下，Agent 先验相对 CCB 有明显提升；定性图中也展示了同一 query 下 GT 被拉回 Top-5 的案例。",
    "(2) 新增对比：DFIM（IJCAI 2025）和 PDAR-RSITR（IJCAI 2026）是直接面向遥感图文检索的顶会结果；RRSITR（CVPR 2026）用于噪声鲁棒性参照。",
    "(3) 限制：论文结果的 backbone、训练协议和 seed 不一定与本机一致；表中 † 只表示论文报告数字，不表示严格可比。主论文应优先报告全七指标、多 seed 均值和协议注释。",
]:
    story.append(Paragraph(text, styles["BodyCN"]))

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
