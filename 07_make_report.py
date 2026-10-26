# -*- coding: utf-8 -*-
"""Generate the Word report: Q1-paper-style comparative-experiment discussion.
Table values are read from the actual results; the analysis text is written
against the measured numbers."""
import json
import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

EXP = Path(r"D:/pig-objection detection/experiments")
EVAL_DIR = EXP / "eval_results"
OUT_DIR = EXP / "outputs"
OUT_DIR.mkdir(exist_ok=True)

metrics = json.loads((EVAL_DIR / "combined_metrics.json").read_text())
bench = json.loads((EVAL_DIR / "benchmark.json").read_text())

MODELS = ["frcnn", "ssd", "yolov5s", "yolov8s", "yolo11s", "rtdetr-l"]
DISPLAY = {
    "frcnn": "Faster R-CNN", "ssd": "SSD300", "yolov5s": "YOLOv5s",
    "yolov8s": "YOLOv8s", "yolo11s": "YOLOv11s", "rtdetr-l": "RT-DETR-L",
}


def m(name, key):
    return metrics[name][key]


def b(name, key):
    return bench[name][key]


def train_time(name):
    if name in ("ssd", "frcnn"):
        log = EXP / "eval_results" / f"{name}_train_log.csv"
        if not log.exists():
            return None
        return sum(float(l.split(",")[5]) for l in log.read_text().splitlines()[1:])
    txt = (EXP / "yolo_train.log").read_text(encoding="utf-8", errors="ignore")
    match = re.search(rf">>> {name} done in ([\d.]+) min", txt)
    return float(match.group(1)) if match else None


def total_instances():
    n = 0
    for split in ["train", "val"]:
        for p in (EXP / "dataset" / "labels" / split).glob("*.txt"):
            n += len([l for l in p.read_text().splitlines() if l.strip()])
    return n


N_IMAGES = len(json.loads((EVAL_DIR / "gt_val_coco.json").read_text())["images"])

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
para("基于深度学习的目标检测算法对比实验与分析", size=16, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, east="黑体")
para("——以猪只目标检测数据集为例", size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
     first_indent=False, space_after=12)

# ============ 1 实验设置 ============
heading("1 实验设置", 0)
para(f"实验采用猪只目标检测数据集，共包含 1015 张标注图像（分辨率 1280×720），标注类别为"
     f"猪只（pig），共 {total_instances()} 个标注实例。按照 8:2 的比例将数据集随机划分为训练集"
     f"（812 张）与验证集（{N_IMAGES} 张），所有算法在相同的数据划分下进行训练与评估，以保证对比的公平性。")
para("选取六种具有代表性的目标检测算法进行对比：两阶段的 Faster R-CNN（ResNet-50-FPN 骨干网络）、"
     "单阶段基于锚框的 SSD300（VGG-16 骨干网络）、单阶段基于锚框的 YOLOv5s、单阶段无锚框的 YOLOv8s 与 "
     "YOLOv11s，以及基于 Transformer 的端到端检测器 RT-DETR-L。所有模型均采用 COCO 预训练权重进行微调，"
     "其中 YOLO 系列与 RT-DETR-L 训练 100 个轮次，SSD300 训练 60 个轮次，Faster R-CNN 训练 50 个轮次。")
para("采用以下指标评估算法性能：精确率（Precision，P）、召回率（Recall，R）、IoU 阈值为 0.5 的平均精度"
     "（mAP@0.5）、IoU 阈值 0.5~0.95 的平均精度（mAP@0.5:0.95）、参数量（Params）、浮点运算量（FLOPs）"
     "以及推理速度（FPS，批大小为 1、FP16 精度、含 NMS）。所有模型的检测精度指标均使用统一的 COCO 评估器"
     "在相同的验证集上计算。实验在 NVIDIA GeForce RTX 3060 Laptop GPU（6 GB 显存）上完成。")

# ============ 2 对比实验结果 ============
doc.add_page_break()
heading("2 对比实验结果", 0)
para("表1给出了六种算法在猪只检测数据集上的性能对比。", first_indent=True)
para("表1 不同目标检测算法在猪只检测数据集上的性能对比", size=10.5,
     align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, space_after=4)

HEADERS = ["算法", "骨干网络", "Params\n(M) ↓", "FLOPs\n(G) ↓", "P ↑", "R ↑",
           "mAP@0.5 ↑", "mAP@0.5:0.95 ↑", "FPS ↑"]
COLS = ["params_M", "gflops", "precision", "recall", "ap50", "ap50_95", "fps"]
ASC = {"params_M": True, "gflops": True, "fps": True,
       "precision": False, "recall": False, "ap50": False, "ap50_95": False}
BACKBONE = {
    "frcnn": "ResNet-50-FPN", "ssd": "VGG-16", "yolov5s": "CSPDarknet53",
    "yolov8s": "C2f-Darknet", "yolo11s": "C3k2-Darknet", "rtdetr-l": "HGNetv2",
}
best_col, second_col = {}, {}
for col in COLS:
    vals = sorted(MODELS, key=lambda n: (b(n, col) if col in ("params_M", "gflops", "fps")
                                         else m(n, col)), reverse=not ASC[col])
    best_col[col] = vals[0]; second_col[col] = vals[1]

table = doc.add_table(rows=1 + len(MODELS), cols=len(HEADERS))
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for j, h in enumerate(HEADERS):
    cell_text(table.rows[0].cells[j], h, bold=True)
    set_cell_borders(table.rows[0].cells[j], top=12, bottom=6)
for i, name in enumerate(MODELS):
    vals = [DISPLAY[name], BACKBONE[name]]
    for col in COLS:
        v = b(name, col) if col in ("params_M", "gflops", "fps") else m(name, col)
        t = f"{v:.2f}" if col == "params_M" else (f"{v:.1f}" if col in ("gflops", "fps") else f"{v:.4f}")
        if name == best_col[col]:
            t = f"**{t}**"
        elif name == second_col[col]:
            t += " †"
        vals.append(t)
    for j, v in enumerate(vals):
        bold = v.startswith("**")
        cell_text(table.rows[i + 1].cells[j], v.replace("**", ""), bold=bold)
        if i == len(MODELS) - 1:
            set_cell_borders(table.rows[i + 1].cells[j], bottom=12)
para("注：加粗表示最优结果，† 表示次优结果；FLOPs 在 640×640 输入下计算（SSD300 为 300×300）；"
     "FPS 在 RTX 3060 Laptop GPU 上以批大小 1、FP16 精度测得（含 NMS）。",
     size=9, first_indent=False, space_after=12)

para("图1 六种算法的 P-R 曲线（IoU = 0.5）", size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER,
     first_indent=False, space_after=4)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(OUT_DIR / "pr_curves.png"), width=Cm(11))
para("图2 六种算法的 mAP 与 FPS 对比", size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER,
     first_indent=False, space_after=4)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(OUT_DIR / "metrics_bars.png"), width=Cm(15))

# ============ 3 结果分析与讨论 ============
doc.add_page_break()
heading("3 结果分析与讨论", 0)

heading("3.1 检测精度分析", 1)
para("由表1可知，六种算法在猪只检测数据集上均取得了较高的检测精度，mAP@0.5 全部超过 0.97，"
     "其中 YOLOv11s 取得最高的 mAP@0.5（0.9894），较最低的 SSD300（0.9791）高出 1.0 个百分点。"
     "然而，当 IoU 阈值收紧至 0.5~0.95 区间时，各算法之间的性能差距明显拉大：RT-DETR-L 的 "
     "mAP@0.5:0.95 达到 0.8114，位居第一，较次优的 YOLOv11s（0.8057）高出 0.6 个百分点，"
     "较 YOLOv5s 与 YOLOv8s 分别高出 1.2 与 1.3 个百分点；而 Faster R-CNN 与 SSD300 的 "
     "mAP@0.5:0.95 分别仅为 0.7391 与 0.6861，与最优算法相差 7.2 与 12.5 个百分点。"
     "这表明在低 IoU 阈值下各算法均可较准确地检出猪只，检测性能的差异主要体现在高 IoU 阈值下的"
     "边界框定位精度上。")
para("作为唯一的两阶段检测器，Faster R-CNN 通过区域提议网络（RPN）生成候选区域并对其进行"
     "二次分类与回归，其 mAP@0.5 达到 0.9864，与最优算法仅相差 0.3 个百分点；但其 mAP@0.5:0.95 "
     "（0.7391）与单阶段及端到端算法存在明显差距。这说明在目标尺度相对一致、场景相对固定的猪只"
     "检测任务中，两阶段结构对候选框精修的优势未能充分转化为定位精度的提升，反而引入了额外的计算开销。")
para("单阶段算法中，SSD300 的 mAP@0.5 与 mAP@0.5:0.95 分别为 0.9791 与 0.6861，在六种算法中"
     "均为最低。SSD300 将输入分辨率固定为 300×300，在降低计算量的同时使图像细节信息在下采样过程中"
     "严重丢失；其基于人工设计默认锚框的检测方式对目标尺度的适应性也相对有限，导致召回率（0.9644）"
     "为六种算法中最低。相比之下，YOLO 系列以 640×640 分辨率作为输入并采用多尺度特征融合结构，"
     "YOLOv5s、YOLOv8s 与 YOLOv11s 的 mAP@0.5:0.95 分别达到 0.7997、0.7985 与 0.8057，"
     "较 SSD300 高出 11.3~12.0 个百分点。在 YOLO 系列内部，采用无锚框检测头与解耦头结构的 "
     "YOLOv8s、YOLOv11s 在 mAP@0.5 上与基于锚框的 YOLOv5s 基本持平；其中 YOLOv11s 通过 C3k2 "
     "轻量级特征提取模块与 C2PSA 注意力机制，以六种算法中最低的计算量（21.7 GFLOPs）将 "
     "mAP@0.5:0.95 较 YOLOv5s 进一步提升了 0.6 个百分点，表明无锚框范式结合注意力机制有效"
     "避免了锚框超参数对检测性能的制约。")
para("基于 Transformer 的 RT-DETR-L 取得了最优的 mAP@0.5:0.95（0.8114）与最高的召回率"
     "（0.9934）。该算法利用自注意力机制对全局上下文进行建模，并通过匈牙利匹配实现端到端的"
     "集合预测（无需 NMS 后处理），对大尺度、密集分布及部分遮挡的猪只具有更强的关联能力，"
     "验证了 Transformer 结构在猪只检测场景中的有效性。")

heading("3.2 精确率—召回率权衡分析", 1)
para("从 P、R 两项指标来看，SSD300 取得了最高的精确率（0.9865），但其召回率最低（0.9644），"
     "呈现“高精确、低召回”的特点，说明其检测结果误检较少但漏检相对较多；RT-DETR-L 的召回率最高"
     "（0.9934）而精确率相对较低（0.9606），表明其在输出较多候选框以减少漏检的同时引入了一定数量的"
     "误检框；Faster R-CNN 的精确率为六种算法中最低（0.8724），而召回率仍达到 0.9888，同样呈现"
     "“高召回、低精确”的特点。相比之下，YOLO 系列的精确率与召回率均保持在 0.98 以上，其中 YOLOv11s "
     "的精确率与召回率之差仅为 0.5 个百分点，P-R 均衡性最佳。在实际猪场应用中，既要避免漏检以保证"
     "猪只计数的准确性，又要控制误检以减少误报，因此 YOLO 系列在 P-R 均衡性上的表现更具实用价值。")

heading("3.3 定位精度分析", 1)
para("mAP@0.5 与 mAP@0.5:0.95 之间的差距反映了 IoU 阈值提高时检测精度的衰减幅度，即边界框的"
     "定位质量。RT-DETR-L 的两项指标差距最小（17.7 个百分点），说明其预测框与真实框的重合度整体"
     "最高、定位最为精确；YOLOv11s 次之（18.4 个百分点）。SSD300 的衰减幅度最大（29.3 个百分点），"
     "结合其最低的召回率，进一步印证了低分辨率输入对边界框回归质量的制约。Faster R-CNN 的衰减幅度"
     "为 24.7 个百分点，尽管两阶段结构对候选框进行了二次精修，其定位精度仍弱于 YOLO 系列与 "
     "RT-DETR-L，这可能与其候选框数量较多、回归分支在有限训练轮次内未充分收敛有关。")

heading("3.4 模型复杂度与推理速度分析", 1)
para(f"从模型复杂度来看，YOLOv5s 的参数量最小（{b('yolov5s','params_M'):.2f} M），Faster R-CNN "
     f"由于采用两阶段结构参数量最大（{b('frcnn','params_M'):.2f} M），RT-DETR-L 因包含 Transformer "
     f"编码器—解码器结构，参数量达到 {b('rtdetr-l','params_M'):.2f} M。在计算量方面，YOLOv11s 得益于 "
     f"C3k2 轻量级模块与深度可分离卷积，FLOPs 仅为 {b('yolo11s','gflops'):.1f} G，为六种算法中最低；"
     f"Faster R-CNN 的 FLOPs 最高（{b('frcnn','gflops'):.1f} G）。")
para(f"在推理速度方面，YOLOv8s 达到了最高的 {b('yolov8s','fps'):.1f} FPS，YOLOv5s（{b('yolov5s','fps'):.1f} FPS）"
     f"与 YOLOv11s（{b('yolo11s','fps'):.1f} FPS）次之，三者均满足实时检测（≥30 FPS）的需求；"
     f"SSD300 的推理速度为 {b('ssd','fps'):.1f} FPS，亦达到实时水平。RT-DETR-L（{b('rtdetr-l','fps'):.1f} FPS）"
     f"与 Faster R-CNN（{b('frcnn','fps'):.1f} FPS）低于实时阈值，难以直接满足实时视频流的处理要求。")

heading("3.5 综合讨论", 1)
para(f"综合检测精度、模型复杂度与推理速度三项指标，YOLOv11s 在猪只检测任务中表现出最佳的综合性能："
     f"其 mAP@0.5 为 {m('yolo11s','ap50'):.4f}、mAP@0.5:0.95 为 {m('yolo11s','ap50_95'):.4f}，"
     f"均为满足实时性要求的算法中最优；计算量为六种算法中最低（{b('yolo11s','gflops'):.1f} GFLOPs），"
     f"推理速度达 {b('yolo11s','fps'):.1f} FPS。RT-DETR-L 虽然取得了最高的 mAP@0.5:0.95，"
     f"但其推理速度低于实时阈值，更适用于对精度要求高、允许离线批处理的场景。"
     f"在猪场实际部署中，检测模型需部署于嵌入式或边缘计算设备并配合机器人/监控系统运行，"
     f"对精度与实时性均有较高要求，因此推荐以 YOLOv11s 作为猪只检测的基础模型。")
para("此外，实验结果表明不同检测范式各有优劣：两阶段算法检测流程完备但计算开销大、速度受限；"
     "单阶段无锚框算法在精度与速度之间取得了更好的平衡；Transformer 端到端算法凭借全局建模能力"
     "在定位精度上具有明显优势，但参数量与计算量较大。在后续工作中，可考虑通过剪枝、量化与知识蒸馏等"
     "模型轻量化手段进一步压缩 YOLOv11s 的参数量与计算量，以满足更严格的边缘部署需求，"
     "并探索高分辨率输入与多尺度训练策略以进一步提升对小目标猪只的检测能力。")

doc.save(OUT_DIR / "对比实验分析报告.docx")
print("REPORT GENERATED", flush=True)
