# -*- coding: utf-8 -*-
"""Params / GFLOPs / FPS for all 6 trained models -> eval_results/benchmark.json.
FPS measured end-to-end (incl. NMS), batch 1, fp16, on the same RTX 3060 Laptop GPU."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:/pig-objection detection/experiments/scripts")  # for -I isolated mode

import numpy as np
import torch
from PIL import Image
from torchinfo import summary as torchinfo_summary
from torchvision import tv_tensors
from torchvision.models.detection import (
    ssd300_vgg16, SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
)
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
import eval_utils

ROOT = Path(r"D:/pig-objection detection")
EXP = ROOT / "experiments"
CKPT_DIR = EXP / "torch_models"
EVAL_DIR = EXP / "eval_results"
BENCH_IMGS = eval_utils.get_split_stems("val")[:30]


def yolo_params_flops(best_pt, imgsz=640):
    model = YOLO(str(best_pt))
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    flops = get_flops(model.model, imgsz=imgsz)  # GFLOPs
    return params, flops


def yolo_fps(best_pt, imgsz=640, n=30):
    model = YOLO(str(best_pt))
    src = [str(ROOT / "data" / f"{s}.jpg") for s in BENCH_IMGS]
    model.predict(src[0], imgsz=imgsz, verbose=False)  # warmup
    times = []
    for s in src[:n]:
        r = model.predict(s, imgsz=imgsz, verbose=False)[0]
        times.append(r.speed["inference"] + r.speed["postprocess"])
    ms = float(np.mean(times))
    return 1000.0 / ms


def tv_model(name):
    if name == "ssd":
        model = ssd300_vgg16(weights=None, num_classes=2)
        transform = SSD300_VGG16_Weights.COCO_V1.transforms()
        size = (3, 300, 300)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2,
                                        min_size=640, max_size=640)
        transform = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
        size = (3, 640, 640)
    ckpt = torch.load(CKPT_DIR / f"{name}_best.pth", map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    return model, transform, size


def tv_params_flops(name):
    model, _, size = tv_model(name)
    info = torchinfo_summary(model, input_size=(1, *size), verbose=0)
    params = info.total_params / 1e6
    flops = info.total_mult_adds * 2 / 1e9  # MACs -> FLOPs
    return params, flops


@torch.no_grad()
def tv_fps(name, n=100):
    model, transform, size = tv_model(name)
    def run_once():
        stem = BENCH_IMGS[i % len(BENCH_IMGS)]
        img = Image.open(ROOT / "data" / f"{stem}.jpg").convert("RGB")
        img_t = transform(tv_tensors.Image(img))
        x = img_t.unsqueeze(0).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            return model(x)
    for i in range(20):  # warmup
        run_once()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        run_once()
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


if __name__ == "__main__":
    out = {}
    for name in ["yolov5s", "yolov8s", "yolo11s", "rtdetr-l"]:
        best = EXP / "runs_yolo" / name / "weights" / "best.pt"
        print(f">>> benchmark {name} ...", flush=True)
        try:
            p, g = yolo_params_flops(best)
            fps = yolo_fps(best)
            out[name] = {"params_M": round(p, 2), "gflops": round(g, 1),
                         "fps": round(fps, 1), "imgsz": 640}
            print(f"    {name}: {p:.2f}M {g:.1f}G {fps:.1f} FPS", flush=True)
        except Exception as e:
            print(f"    {name} FAILED: {e}", flush=True)
    for name, size in [("ssd", 300), ("frcnn", 640)]:
        print(f">>> benchmark {name} ...", flush=True)
        try:
            p, g = tv_params_flops(name)
            fps = tv_fps(name)
            out[name] = {"params_M": round(p, 2), "gflops": round(g, 1),
                         "fps": round(fps, 1), "imgsz": size}
            print(f"    {name}: {p:.2f}M {g:.1f}G {fps:.1f} FPS", flush=True)
        except Exception as e:
            print(f"    {name} FAILED: {e}", flush=True)
    (EVAL_DIR / "benchmark.json").write_text(json.dumps(out, indent=2))
    print("BENCHMARK DONE", flush=True)
