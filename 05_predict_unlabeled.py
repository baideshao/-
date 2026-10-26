# -*- coding: utf-8 -*-
"""Predict on the 1000 unlabeled images with the best model.
Saves: predictions/labels/*.txt (YOLO format + conf), counts.csv, sample_detections.png."""
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.models.detection import (
    ssd300_vgg16, SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
)
from ultralytics import YOLO

ROOT = Path(r"D:/pig-objection detection")
EXP = ROOT / "experiments"
CKPT_DIR = EXP / "torch_models"
EVAL_DIR = EXP / "eval_results"
PRED_DIR = EXP / "predictions"
(PRED_DIR / "labels").mkdir(parents=True, exist_ok=True)

CONF = 0.25


def get_unlabeled():
    labeled = {s.split()[0] for s in (EXP / "split.txt").read_text().splitlines()}
    return sorted(p.stem for p in (ROOT / "data").glob("*.jpg") if p.stem not in labeled)


def best_model():
    m = json.loads((EVAL_DIR / "combined_metrics.json").read_text())
    return max(m.items(), key=lambda kv: kv[1]["ap50_95"])[0]


# ---------------- YOLO inference ----------------
def predict_yolo(name, stems):
    best = EXP / "runs_yolo" / name / "weights" / "best.pt"
    model = YOLO(str(best))
    out = {}
    for stem in stems:
        r = model.predict(str(ROOT / "data" / f"{stem}.jpg"), imgsz=640,
                          conf=CONF, iou=0.6, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            out[stem] = np.zeros((0, 5), dtype=np.float32)
        else:
            out[stem] = np.hstack([r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy()[:, None]]).astype(np.float32)
    return out


# ---------------- torchvision inference ----------------
def predict_tv(name, stems):
    if name == "ssd":
        model = ssd300_vgg16(weights=None, num_classes=2)
        transform = SSD300_VGG16_Weights.COCO_V1.transforms()
        size = 300
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2,
                                        min_size=640, max_size=640)
        transform = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
        size = 640
    ckpt = torch.load(CKPT_DIR / f"{name}_best.pth", map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    out = {}
    batch, stems_b = [], []
    for stem in stems:
        img = Image.open(ROOT / "data" / f"{stem}.jpg").convert("RGB")
        img_t = transform(tv_tensors.Image(img))
        batch.append(img_t); stems_b.append(stem)
        if len(batch) == 8:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                outs = model([im.cuda() for im in batch])
            for o, s in zip(outs, stems_b):
                boxes = o["boxes"].cpu().numpy(); scores = o["scores"].cpu().numpy()
                keep = scores >= CONF
                out[s] = np.hstack([boxes[keep], scores[keep, None]]).astype(np.float32)
            batch, stems_b = [], []
    if batch:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            outs = model([im.cuda() for im in batch])
        for o, s in zip(outs, stems_b):
            boxes = o["boxes"].cpu().numpy(); scores = o["scores"].cpu().numpy()
            keep = scores >= CONF
            out[s] = np.hstack([boxes[keep], scores[keep, None]]).astype(np.float32)
    return out


def save_results(stems, preds):
    counts = []
    for stem in stems:
        boxes = preds[stem]
        H, W = cv2.imread(str(ROOT / "data" / f"{stem}.jpg")).shape[:2]
        lines = []
        for b in boxes:
            x1, y1, x2, y2, s = b
            cx = (x1 + x2) / 2 / W; cy = (y1 + y2) / 2 / H
            w = (x2 - x1) / W; h = (y2 - y1) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {s:.4f}")
        (PRED_DIR / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        counts.append({"image": f"{stem}.jpg", "num_pigs": len(boxes)})
    with open(PRED_DIR / "counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "num_pigs"])
        w.writeheader()
        w.writerows(counts)
    total = sum(c["num_pigs"] for c in counts)
    (PRED_DIR / "total.txt").write_text(f"images={len(stems)} total_pigs={total}\n")
    return counts


def make_montage(stems, preds, n=20, cols=5):
    random.seed(42)
    sample = random.sample(stems, min(n, len(stems)))
    cells = []
    for stem in sample:
        img = cv2.imread(str(ROOT / "data" / f"{stem}.jpg"))
        for b in preds[stem]:
            x1, y1, x2, y2, s = b
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{s:.2f}", (int(x1), max(int(y1) - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cells.append(img)
    h, w = cells[0].shape[:2]
    rows = (len(cells) + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(cells):
        r_, c_ = divmod(i, cols)
        canvas[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = im
    cv2.imwrite(str(PRED_DIR / "sample_detections.png"), canvas)
    print(f"montage saved with {len(cells)} images", flush=True)


if __name__ == "__main__":
    stems = get_unlabeled()
    print(f"unlabeled images: {len(stems)}", flush=True)
    name = best_model()
    print(f"best model: {name}", flush=True)
    if name.startswith("yolo") or name.startswith("rtdetr"):
        preds = predict_yolo(name, stems)
    else:
        preds = predict_tv(name, stems)
    counts = save_results(stems, preds)
    print(f"total pigs detected: {sum(c['num_pigs'] for c in counts)}", flush=True)
    make_montage(stems, preds)
    print("PREDICTION DONE", flush=True)
