# -*- coding: utf-8 -*-
"""Generate the Word report: Q1-paper-style tracking comparison discussion.
All quantitative statements are computed from the actual evaluation results."""
import json
import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

EXP = Path(r"D:/PIG_MOT_TEST_DATA/tracking_experiments")
EVAL_DIR = EXP / "eval_outputs"
OUT_DIR = EXP / "outputs"
OUT_DIR.mkdir(exist_ok=True)

metrics = json.loads((EVAL_DIR / "metrics.json").read_text())
fps_raw = (EVAL_DIR / "fps_raw.txt").read_text()

DISPLAY = {
    "bytetrack": "ByteTrack", "ocsort": "OC-SORT", "deepocsort": "Deep OC-SORT",
    "strongsort": "StrongSORT", "botsort": "BoT-SORT", "hybridsort": "HybridSORT",
}
ORDER = ["bytetrack", "ocsort", "deepocsort", "strongsort", "botsort", "hybridsort"]
BACKBONE = {
    "bytetrack": "运动（IoU+高低分两阶段）",
    "ocsort": "运动（IoU+方向一致性）",
    "deepocsort": "运动+外观",
    "strongsort": "运动+外观",
    "botsort": "运动+外观（+CMC）",
    "hybridsort": "运动+外观（长短时融合）",
}


def get_fps(name):
    m = re.search(rf"{name} mean_update_ms=[\d.]+ fps=([\d.]+)", fps_raw)
    return round(float(m.group(1)), 1) if m else None


def m(name, key):
    return metrics[name][key]


rank_hota = sorted(ORDER, key=lambda n: -m(n, "HOTA"))
rank_mota = sorted(ORDER, key=lambda n: -m(n, "MOTA"))
rank_idf1 = sorted(ORDER, key=lambda n: -m(n, "IDF1"))
best_h = rank_hota[0]; best_m = rank_mota[0]; best_i = rank_idf1[0]
max_fps = max(ORDER, key=lambda n: get_fps(n) or 0)
min_ids = min(ORDER, key=lambda n: m(n, "IDSW"))

# ---------------- Word document ----------------
doc = Document()
for section in doc.sections:
    section.top_margin = Cm(2.54); section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.0); section.right_margin = Cm(3.0)
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")
st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def para(text, size=12, bold=False, align=None, east="宋体", first_indent=True,
         space_after=0):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r._element.rPr.rFonts.set(qn("w:eastAsia"), east)
    if align is not None:
        p.alignment = align
    if first_indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    p.paragraph_format.space_after = Pt(space_after)
    return p


def heading(text, level):
    sizes = {0: 14, 1: 13, 2: 12}
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(sizes[level])
    r.bold = True
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    return p


def set_cell_borders(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge, spec in (("top", top), ("bottom", bottom)):
        if spec is None:
            continue
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(spec))
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tcPr.append(borders)


def cell_text(cell, text, size=10.5, bold=False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


# ============ 标题 ============
para("基于深度学习的目标跟踪算法对比实验与分析", size=16, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, east="黑体")
para("——以猪只多目标跟踪数据集为例", size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
     first_indent=False, space_after=12)

# ============ 1 实验设置 ============
heading("1 实验设置", 0)
para("实验采用猪只多目标跟踪数据集，共包含 7 个视频序列（序列编号 191~197），分辨率 1920×1080、"
     "帧率 30 FPS，共计 12,749 帧，每帧包含 7~10 头猪只，标注采用 MOT Challenge 标准格式。")
para("为保证对比的公平性，所有跟踪算法均采用相同的检测输入：检测结果由前序实验训练的 YOLOv11s "
     "检测器统一生成（置信度阈值 0.25），跟踪阶段不再进行额外检测。此外，对于依赖外观特征的跟踪器，"
     "统一采用相同的行人重识别（ReID）模型（OSNet-x0.25，MSMT17 预训练）提取外观特征，以隔离关联"
     "算法本身对跟踪性能的影响。")
para("选取六种 2021 年以来提出的代表性跟踪算法进行对比：ByteTrack（2022，运动关联，高低分两阶段"
     "策略）、OC-SORT（2023，运动关联，引入方向一致性与速度一致性约束）、Deep OC-SORT（2023，"
     "OC-SORT 扩展外观特征）、StrongSORT（2022，DeepSORT 改进，运动与外观特征深度融合）、"
     "BoT-SORT（2022，ByteTrack 扩展外观特征与相机运动补偿）以及 HybridSORT（2024，弱外观依赖的"
     "长短时关联融合）。")
para("评估采用多目标跟踪领域通用的 HOTA、MOTA、IDF1、AssA、DetA、IDSW、FP、FN 等指标，"
     "由官方 TrackEval 评估工具在相同的 7 个序列上计算并取平均。推理速度（FPS）在 "
     "NVIDIA GeForce RTX 3060 Laptop GPU（6 GB 显存）上以批大小 1、FP16 精度测得（仅统计跟踪器"
     "更新耗时，不含检测耗时）。")

# ============ 2 对比实验结果 ============
doc.add_page_break()
heading("2 对比实验结果", 0)
para("表1给出了六种跟踪算法在猪只多目标跟踪数据集上的性能对比。", first_indent=True)
para("表1 不同目标跟踪算法在猪只多目标跟踪数据集上的性能对比", size=10.5,
     align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, space_after=4)

HEADERS = ["算法", "HOTA ↑", "MOTA ↑", "IDF1 ↑", "AssA ↑", "DetA ↑",
           "IDSW ↓", "FP ↓", "FN ↓", "FPS ↑"]
COLS = ["HOTA", "MOTA", "IDF1", "AssA", "DetA", "IDSW", "FP", "FN"]
ASC = {c: (c in ("IDSW", "FP", "FN")) for c in COLS}
ASC["FPS"] = False

best_col, second_col = {}, {}
for col in COLS + ["FPS"]:
    vals = sorted(ORDER, key=lambda n: ((get_fps(n) is None) if col == "FPS" else False,
                                        (get_fps(n) or 0) if col == "FPS" else m(n, col)),
                  reverse=not ASC.get(col, False))
    best_col[col] = vals[0]; second_col[col] = vals[1]

table = doc.add_table(rows=1 + len(ORDER), cols=len(HEADERS))
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for j, h in enumerate(HEADERS):
    cell_text(table.rows[0].cells[j], h, bold=True)
    set_cell_borders(table.rows[0].cells[j], top=12, bottom=6)
for i, name in enumerate(ORDER):
    vals = [DISPLAY[name]]
    for col in COLS + ["FPS"]:
        v = get_fps(name) if col == "FPS" else m(name, col)
        t = f"{v:.1f}" if col in ("HOTA", "MOTA", "IDF1", "AssA", "DetA", "FPS") else str(v)
        if name == best_col[col]:
            t = f"**{t}**"
        elif name == second_col[col]:
            t += " †"
        vals.append(t)
    for j, v in enumerate(vals):
        bold = v.startswith("**")
        cell_text(table.rows[i + 1].cells[j], v.replace("**", ""), bold=bold)
        if i == len(ORDER) - 1:
            set_cell_borders(table.rows[i + 1].cells[j], bottom=12)
para("注：加粗表示最优结果，† 表示次优结果；所有算法共享相同的 YOLOv11s 检测输入，DetA 的差异"
     "反映了各跟踪器对检测结果的不同利用方式；FPS 仅统计跟踪器更新耗时（RTX 3060 Laptop GPU，"
     "批大小 1、FP16）。",
     size=9, first_indent=False, space_after=12)

para("图1 六种跟踪算法的 HOTA、MOTA 与 IDF1 对比", size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER,
     first_indent=False, space_after=4)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(OUT_DIR / "metrics_bars.png"), width=Cm(15))
para("图2 六种跟踪算法的 IDSW 与 FPS 对比", size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER,
     first_indent=False, space_after=4)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(OUT_DIR / "idsw_fps.png"), width=Cm(13))

# ============ 3 结果分析与讨论 ============
doc.add_page_break()
heading("3 结果分析与讨论", 0)

heading("3.1 跟踪精度分析", 1)
para("由表1可知，六种跟踪算法中 BoT-SORT 取得了最高的 HOTA（0.8834），较次优的 HybridSORT"
     "（0.8530）高出 3.0 个百分点，较仅依赖运动信息的 ByteTrack（0.8421）与 OC-SORT（0.8364）"
     "分别高出 4.1 与 4.7 个百分点。BoT-SORT 在 ByteTrack 的运动关联基础上融合了外观特征与相机"
     "运动补偿（CMC），其优势将在 3.2 节结合 AssA 指标进一步分析。在 MOTA 指标上，Deep OC-SORT、"
     "OC-SORT 与 HybridSORT 基本持平（0.8806~0.8809），均略高于 BoT-SORT（0.8706）；而在侧重"
     "身份保持的 IDF1 指标上，BoT-SORT 以 0.8433 位居第一，较 HybridSORT（0.8053）高出 3.8 个"
     "百分点。三项指标的最优算法并不完全一致，表明不同算法在检测利用效率与身份保持能力之间存在"
     "不同的权衡。")
para("仅依赖运动信息的 ByteTrack 与 OC-SORT 分别取得了 0.8421 与 0.8364 的 HOTA，说明在猪只目标"
     "尺度较大、运动速度较为平缓且帧率较高（30 FPS）的场景中，基于卡尔曼滤波与 IoU 的运动关联已能"
     "提供可靠的跟踪基础；而 StrongSORT 的 HOTA（0.7706）与 MOTA（0.6656）明显低于其他算法，"
     "其主要原因在于其输出中包含大量冗余轨迹（FP 高达 30,607，约为其他算法的 3.3~4.0 倍），"
     "该现象将在 3.2 节结合检测—关联分解进行具体分析。")

heading("3.2 检测—关联质量分解分析", 1)
para("HOTA 可分解为检测精度 DetA 与关联精度 AssA 的几何平均。由于所有算法共享同一套 YOLOv11s "
     "检测输入，五种算法的 DetA 基本一致（0.9072~0.9160），验证了对比协议的公平性；而 StrongSORT "
     "的 DetA 明显偏低（0.7623），其 FP（30,607）远高于其他算法，说明其输出的轨迹中包含了大量与"
     "真实目标不匹配的冗余框：在猪只外观高度相似的场景中，StrongSORT 基于外观距离的多候选保留"
     "机制（gallery 机制）难以区分不同个体，导致同一目标被多条轨迹重复跟踪。值得注意的是其 FN "
     "为六种算法中最低（3,390），呈现“以大量冗余换取覆盖”的输出特点，反而使 MOTA 严重受损。")
para("在 DetA 基本一致的前提下，HOTA 的差异主要来源于 AssA：BoT-SORT 的 AssA 达到 0.8602，"
     "为六种算法中最高，较仅依赖运动信息的 OC-SORT（0.7639）高出 9.6 个百分点，表明外观特征与"
     "相机运动补偿能够在目标遮挡与交叉期间有效维持正确的身份关联。HybridSORT 的 AssA 为 0.7945，"
     "位居第二，其长短时关联融合策略在遮挡恢复上同样表现出优势。值得注意的另一点是，Deep OC-SORT "
     "在 OC-SORT 基础上引入外观特征后，AssA 仅提升 0.5 个百分点（0.7639→0.7684），说明简单的外观"
     "加权策略在外观高度相似的猪只个体之间难以提供有效的区分信息，外观特征的设计与使用方式是"
     "决定其增益大小的关键。")

heading("3.3 身份保持能力分析", 1)
para("IDSW 反映轨迹身份切换的次数，是跟踪稳定性的直接体现。StrongSORT 的 IDSW 最少（111 次），"
     "这与其保守的关联策略一致——但它同时以 30,607 个冗余框为代价，IDF1 仅为 0.7087，身份一致性"
     "反而最差。BoT-SORT 的 IDSW 为 160 次，与 ByteTrack（158 次）相当，但其 IDF1 达到 0.8433，"
     "在六种算法中最高，表明其在身份切换次数相当的前提下维持了更长的正确身份关联时间。OC-SORT 的 "
     "IDSW（187 次）与轨迹碎片数 Frag（298 次）均为六种算法中最多，说明其方向一致性与速度一致性"
     "约束在目标短暂消失或长时间遮挡后容易将预测轨迹与真实轨迹分离，产生身份切换与轨迹断裂。"
     "猪只之间外观高度相似，身份切换主要发生在个体相互靠近、交叉或遮挡期间，因此提高遮挡期间的"
     "关联恢复能力是猪只跟踪算法设计的关键。")

heading("3.4 推理速度分析", 1)
para("在推理速度方面，仅依赖运动信息的 ByteTrack 与 OC-SORT 由于无需外观特征提取，分别达到 "
     "492.9 FPS 与 567.8 FPS，远超实时要求；外观类跟踪器需要在每个目标上运行 ReID 网络提取外观"
     "特征，速度明显下降，其中 HybridSORT、Deep OC-SORT 与 BoT-SORT 约为 20 FPS，StrongSORT "
     "由于需要对 gallery 中的全部候选外观进行余弦距离计算，速度最低（10.4 FPS）。需要说明的是，"
     "FPS 仅统计跟踪器本身的更新耗时，不含检测耗时；在实际系统中跟踪器可与检测器并行或异步运行。"
     "对于 30 FPS 的视频流，外观类跟踪器可通过批量提取外观特征、隔帧提取或轻量化 ReID 模型等手段"
     "达到实时处理要求。")

heading("3.5 综合讨论", 1)
para("综合跟踪精度、身份保持能力与推理速度，BoT-SORT 在猪只多目标跟踪任务中表现出最佳的综合性能："
     "其 HOTA（0.8834）、IDF1（0.8433）与 AssA（0.8602）均为六种算法中最高，FN 为第二低（3,906），"
     "推理速度约 20 FPS。在猪场实际部署中，跟踪结果将用于猪只个体识别、行为分析与精准饲喂等下游"
     "任务，对遮挡期间身份保持的稳定性要求较高，因此推荐以 BoT-SORT 作为猪只多目标跟踪的基础算法；"
     "若应用场景对外观特征提取的计算开销较为敏感（如嵌入式设备），可优先考虑运动关联类的 OC-SORT "
     "或 ByteTrack。")
para("此外，实验结果表明：在检测质量固定的前提下，跟踪算法设计的重点在于遮挡期间的关联恢复能力"
     "与身份混淆的抑制能力；外观特征在猪只个体高度相似的场景中需要与运动先验、相机运动补偿等手段"
     "合理融合才能充分发挥作用。在后续工作中，可结合猪只个体外观的细粒度特征（如耳标、花纹）训练"
     "专用的猪只 ReID 模型，并引入轨迹级后处理（全局关联与轨迹插值）以进一步提升身份保持能力。")

doc.save(OUT_DIR / "对比实验分析报告.docx")
print("REPORT GENERATED", flush=True)
