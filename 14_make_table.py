# -*- coding: utf-8 -*-
"""Generate the Q1-style MOT comparison table (png/md/csv/xlsx) + bar charts."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.styles import Font as XLFont

EXP = Path(r"D:/PIG_MOT_TEST_DATA/tracking_experiments")
EVAL_DIR = EXP / "eval_outputs"
OUT_DIR = EXP / "outputs"
OUT_DIR.mkdir(exist_ok=True)

metrics = json.loads((EVAL_DIR / "metrics.json").read_text())

DISPLAY = {
    "bytetrack": "ByteTrack",
    "ocsort": "OC-SORT",
    "deepocsort": "Deep OC-SORT",
    "strongsort": "StrongSORT",
    "botsort": "BoT-SORT",
    "hybridsort": "HybridSORT",
}
ORDER = ["bytetrack", "ocsort", "deepocsort", "strongsort", "botsort", "hybridsort"]
COLOR = {
    "bytetrack": "#2a78d6", "ocsort": "#eb6834", "deepocsort": "#1baf7a",
    "strongsort": "#eda100", "botsort": "#e87ba4", "hybridsort": "#008300",
}
INK = "#0b0b0b"; MUTED = "#898781"; SURFACE = "#fcfcfb"; GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 300,
})

fps_raw = (EVAL_DIR / "fps_raw.txt").read_text() if (EVAL_DIR / "fps_raw.txt").exists() else ""


def get_fps(name):
    m = re.search(rf"{name} mean_update_ms=[\d.]+ fps=([\d.]+)", fps_raw)
    return round(float(m.group(1)), 1) if m else None


HEADERS = ["Tracker", "HOTA ↑", "MOTA ↑", "IDF1 ↑", "AssA ↑", "DetA ↑",
           "IDSW ↓", "FP ↓", "FN ↓", "FPS ↑"]
COLS = ["HOTA", "MOTA", "IDF1", "AssA", "DetA", "IDSW", "FP", "FN"]
ASC = {c: (c in ("IDSW", "FP", "FN")) for c in COLS}
ASC["FPS"] = False

rows = []
for name in ORDER:
    m = metrics[name]
    rows.append({
        "name": name, "display": DISPLAY[name],
        "HOTA": round(m["HOTA"], 2), "MOTA": round(m["MOTA"], 2),
        "IDF1": round(m["IDF1"], 2), "AssA": round(m["AssA"], 2),
        "DetA": round(m["DetA"], 2), "IDSW": m["IDSW"], "FP": m["FP"], "FN": m["FN"],
        "FPS": get_fps(name),
    })

best, second = {}, {}
for col in COLS + ["FPS"]:
    vals = sorted(rows, key=lambda r: (r[col] is None, r[col] or 0),
                  reverse=not ASC.get(col, False))
    best[col] = vals[0]["name"]; second[col] = vals[1]["name"]

def fmt(r, col):
    v = r[col]
    if col in ("HOTA", "MOTA", "IDF1", "AssA", "DetA", "FPS"):
        return f"{v:.1f}" if v is not None else "—"
    return str(v)

def cell(r, col):
    t = fmt(r, col)
    if r["name"] == best[col]:
        return t, True
    if r["name"] == second[col]:
        return t + " †", False
    return t, False

# markdown
md = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
for r in rows:
    cells = []
    for col in COLS + ["FPS"]:
        t = fmt(r, col)
        if r["name"] == best[col]:
            t = f"**{t}**"
        elif r["name"] == second[col]:
            t += " †"
        cells.append(t)
    md.append("| " + f" | ".join([r["display"]] + cells) + " |")
md.append("")
md.append("The best results are in **bold**, the second-best are marked with †. All trackers share the "
          "same YOLOv11s detection input; DetA differences reflect how each tracker exploits the "
          "detections (confidence filtering, redundant tracks). ReID-based trackers share the same "
          "OSNet-x0.25 (MSMT17) appearance model. Metrics averaged over 7 sequences (12,749 frames, "
          "1920×1080@30fps) with TrackEval. FPS measured on an RTX 3060 Laptop GPU (batch 1, FP16).")
(OUT_DIR / "comparison_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

import csv
with open(OUT_DIR / "comparison_table.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(HEADERS)
    for r in rows:
        w.writerow([r["display"]] + [fmt(r, c) for c in COLS + ["FPS"]])

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "comparison"
ws.append(HEADERS)
for r in rows:
    ws.append([r["display"]] + [fmt(r, c) for c in COLS + ["FPS"]])
for j, col in enumerate(COLS + ["FPS"], start=1):
    for i, r in enumerate(rows, start=2):
        cell_ = ws.cell(row=i, column=j)
        if col in COLS + ["FPS"] and r["name"] == best[col]:
            cell_.font = XLFont(bold=True)
        elif col in COLS + ["FPS"] and r["name"] == second[col]:
            cell_.value += " †"
wb.save(OUT_DIR / "comparison_table.xlsx")

# ---------------- publication-style PNG table ----------------
ncols = len(HEADERS); nrows = len(rows) + 1
fig, ax = plt.subplots(figsize=(10.5, 2.8))
ax.axis("off")
cw = [1.9] + [0.85] * 9
xs = np.concatenate([[0], np.cumsum(cw)]) / sum(cw)
y_head, y_row = 0.88, 0.74
for j, h in enumerate(HEADERS):
    ax.text((xs[j] + xs[j + 1]) / 2, y_head, h, ha="center", va="center",
            fontsize=10.5, weight="bold")
for i, r in enumerate(rows):
    y = y_row - i * (0.74 / nrows)
    vals = [(r["display"], False)] + [cell(r, c) for c in COLS + ["FPS"]]
    for j, (t, bold) in enumerate(vals):
        ax.text((xs[j] + xs[j + 1]) / 2, y, t, ha="center", va="center",
                fontsize=10, weight="bold" if bold else "normal")
for y in [y_head + 0.045, y_row - (nrows - 1) * (0.74 / nrows) - 0.035]:
    ax.plot([0, 1], [y, y], color=INK, lw=1.4)
ax.plot([0, 1], [y_row + 0.03, y_row + 0.03], color=INK, lw=0.8)
ax.text(0, -0.14, "The best results are in bold; † denotes the second-best. All trackers share the same YOLOv11s "
        "detection input; DetA differences reflect how each tracker exploits the detections. ReID-based trackers "
        "share the same OSNet-x0.25 appearance model. Metrics averaged over 7 sequences with TrackEval; "
        "FPS on an RTX 3060 Laptop GPU (batch 1, FP16).",
        fontsize=7.5, color=MUTED, va="top")
ax.set_xlim(0, 1); ax.set_ylim(-0.2, 1.0)
fig.savefig(OUT_DIR / "table_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---------------- bar charts ----------------
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
panels = [("HOTA", "HOTA", 1.0), ("MOTA", "MOTA", 1.0), ("IDF1", "IDF1", 1.0)]
labels = [DISPLAY[n] for n in ORDER]
for ax, (title, col, ymax) in zip(axes, panels):
    vals = [r[col] for r in rows]
    bars = ax.bar(range(len(ORDER)), vals, color=[COLOR[n] for n in ORDER],
                  width=0.62, zorder=3)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "metrics_bars.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, (title, col) in zip(axes, [("IDSW", "IDSW"), ("FPS", "FPS")]):
    vals = [r[col] or 0 for r in rows]
    bars = ax.bar(range(len(ORDER)), vals, color=[COLOR[n] for n in ORDER],
                  width=0.62, zorder=3)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + max(vals) * 0.02,
                f"{v:.1f}" if col == "FPS" else str(int(v)),
                ha="center", va="bottom", fontsize=8.5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "idsw_fps.png", bbox_inches="tight")
plt.close(fig)

print("ALL FIGURES/TABLES GENERATED", flush=True)
