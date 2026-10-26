# -*- coding: utf-8 -*-
"""Generate detections for all 7 sequences with the trained YOLOv11s
(from the detection experiment) -> det.txt per sequence (MOT format)."""
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(r"D:/PIG_MOT_TEST_DATA")
EXP = ROOT / "tracking_experiments"
DET_DIR = EXP / "detections"
DET_DIR.mkdir(exist_ok=True)
SEQS = [str(s) for s in range(191, 198)]
CONF = 0.25

model = YOLO(r"D:/pig-objection detection/experiments/runs_yolo/yolo11s/weights/best.pt")

for seq in SEQS:
    video = ROOT / f"MyVideo_{seq}.mp4"
    cap = cv2.VideoCapture(str(video))
    out_lines = []
    frame_idx = 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=1280, conf=CONF, iou=0.6, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(float, b[:4])
                conf = float(b[4]) if len(b) > 4 else 1.0
                out_lines.append(f"{frame_idx},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},-1,-1\n")
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"[{seq}] {frame_idx}/{total}", flush=True)
    cap.release()
    (DET_DIR / f"{seq}.txt").write_text("".join(out_lines))
    print(f"[{seq}] done: {frame_idx - 1} frames, {len(out_lines)} detections", flush=True)
print("ALL DETECTIONS DONE", flush=True)
