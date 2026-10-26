# -*- coding: utf-8 -*-
"""Train 4 ultralytics models sequentially: YOLOv5s, YOLOv8s, YOLOv11s, RT-DETR-L."""
import sys
import time
import torch
from pathlib import Path
from ultralytics import YOLO

EXP = Path(r"D:/pig-objection detection/experiments")
DATA = str(EXP / "dataset.yaml")
PROJECT = str(EXP / "runs_yolo")
WEIGHTS_DIR = EXP / "weights"

JOBS = [
    ("yolov5su.pt", "yolov5s", 16),
    ("yolov8s.pt", "yolov8s", 16),
    ("yolo11s.pt", "yolo11s", 16),
    ("rtdetr-l.pt", "rtdetr-l", 4),
]


def train_one(weights, name, batch, epochs=100):
    weights = str(WEIGHTS_DIR / weights)  # use local file (github blocked)
    print(f"\n{'='*60}\n>>> {name}: {weights} batch={batch}\n{'='*60}", flush=True)
    model = YOLO(weights)
    kwargs = dict(
        data=DATA, imgsz=640, epochs=epochs, batch=batch, device=0,
        workers=4, project=PROJECT, name=name, exist_ok=True,
        seed=42, pretrained=True,
    )
    t0 = time.time()
    try:
        model.train(**kwargs)
    except Exception as e:
        if "out of memory" in str(e).lower() and batch >= 2:
            print(f"[OOM] retry {name} with batch={batch // 2}", flush=True)
            torch.cuda.empty_cache()
            model = YOLO(weights)
            model.train(**{**kwargs, "batch": batch // 2})
        else:
            raise
    print(f">>> {name} done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for weights, name, batch in JOBS:
        if only and only not in name:
            continue
        train_one(weights, name, batch)
    print("ALL YOLO TRAINING DONE", flush=True)
