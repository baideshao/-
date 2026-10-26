# -*- coding: utf-8 -*-
"""Build TrackEval layout: seqinfo.ini + GT copy + seqmap for the 7 pig MOT sequences."""
import shutil
from pathlib import Path

ROOT = Path(r"D:/PIG_MOT_TEST_DATA")
EXP = ROOT / "tracking_experiments"
SEQS = [str(s) for s in range(191, 198)]
FPS = 30
W, H = 1920, 1080

GT_ROOT = EXP / "data" / "gt" / "mot_challenge" / "pig-train"
SEQMAP_DIR = GT_ROOT / "seqmaps"
SEQMAP_DIR.mkdir(parents=True, exist_ok=True)

seqmap_lines = []
for seq in SEQS:
    seq_dir = GT_ROOT / seq / "gt"
    seq_dir.mkdir(parents=True, exist_ok=True)
    # copy gt
    shutil.copy(ROOT / seq / "gt" / "gt.txt", seq_dir / "gt.txt")
    shutil.copy(ROOT / seq / "gt" / "labels.txt", seq_dir / "labels.txt")
    # seqLength = max frame id in gt
    frames = {int(l.split(",")[0]) for l in (seq_dir / "gt.txt").read_text().splitlines() if l.strip()}
    seq_len = max(frames)
    # seqinfo.ini
    (GT_ROOT / seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq}\nimDir=img1\nframeRate={FPS}\nseqLength={seq_len}\n"
        f"imWidth={W}\nimHeight={H}\nimExt=.jpg\n")
    seqmap_lines.append(f"{seq}\n")
(SEQMAP_DIR / "pig-train.txt").write_text("".join(seqmap_lines))
print("prepared", len(SEQS), "sequences")
