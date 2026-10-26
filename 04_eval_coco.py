# -*- coding: utf-8 -*-
"""Unified evaluation of all 6 models with the SAME pycocotools evaluator.
For every model: COCO-style AP@.5:.95, AP@.5, AP@.75 + P/R at conf=0.25/IoU=0.5 + PR curve.
Outputs: eval_results/combined_metrics.json, eval_results/pr_{name}.npz"""
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, r"D:/pig-objection detection/experiments/scripts")  # for -I isolated mode

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.models.detection import (
    ssd300_vgg16, SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
)
from ultralytics import YOLO
import eval_utils

ROOT = Path(r"D:/pig-objection detection")
EXP = ROOT / "experiments"
CKPT_DIR = EXP / "torch_models"
EVAL_DIR = EXP / "eval_results"
EVAL_DIR.mkdir(exist_ok=True)
VAL_STEMS = sorted(eval_utils.get_split_stems("val"))
ID_MAP = eval_utils.get_image_id_map("val")


# ---------------- GT boxes (xyxy) per image ----------------
def load_gt_boxes():
    gt = {}
    for stem in VAL_STEMS:
        root = ET.parse(ROOT / "label" / f"{stem}.xml").getroot()
        boxes = []
        for obj in root.iter("object"):
            if obj.find("name").text.strip() != "pig":
                continue
            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text); ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text); ymax = float(bb.find("ymax").text)
            xmin, xmax = sorted([xmin, xmax]); ymin, ymax = sorted([ymin, ymax])
            boxes.append([xmin, ymin, xmax, ymax])
        gt[stem] = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    return gt


def iou_matrix(a, b):
    """a: [N,4], b: [M,4] -> [N,M] IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def compute_pr(gt, preds, conf_thr=0.25, iou_thr=0.5):
    """preds: {stem: [N,5] xyxy+score}. Greedy matching."""
    tp = fp = fn = 0
    for stem in VAL_STEMS:
        g = gt[stem]
        fn += len(g)
        p = preds.get(stem)
        if p is None or len(p) == 0:
            continue
        p = p[p[:, 4] >= conf_thr]
        if len(p) == 0:
            continue
        p = p[np.argsort(-p[:, 4])]
        ious = iou_matrix(p[:, :4], g)
        matched_gt = set()
        for i in range(len(p)):
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= iou_thr and j not in matched_gt:
                tp += 1
                matched_gt.add(j)
            else:
                fp += 1
        fn -= len(matched_gt)
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    return prec, rec


# ---------------- inference: YOLO family ----------------
def infer_yolo(weights_path):
    model = YOLO(str(weights_path))
    preds = {}
    for stem in VAL_STEMS:
        r = model.predict(str(ROOT / "data" / f"{stem}.jpg"), imgsz=640,
                          conf=0.001, iou=0.6, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            preds[stem] = np.zeros((0, 5), dtype=np.float32)
        else:
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()[:, None]
            preds[stem] = np.hstack([b, s]).astype(np.float32)
    return preds


# ---------------- inference: torchvision family ----------------
def build_tv_model(name):
    if name == "ssd":
        model = ssd300_vgg16(weights=None, num_classes=2)
        transform = SSD300_VGG16_Weights.COCO_V1.transforms()
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2,
                                        min_size=640, max_size=640)
        transform = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    ckpt = torch.load(CKPT_DIR / f"{name}_best.pth", map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    return model, transform


@torch.no_grad()
def infer_tv(name):
    model, transform = build_tv_model(name)
    preds = {}
    batch_imgs, batch_stems = [], []
    for stem in VAL_STEMS:
        img = Image.open(ROOT / "data" / f"{stem}.jpg").convert("RGB")
        img_t = transform(tv_tensors.Image(img))
        batch_imgs.append(img_t)
        batch_stems.append(stem)
        if len(batch_imgs) == 8:
            with torch.autocast("cuda", dtype=torch.float16):
                outs = model([im.cuda() for im in batch_imgs])
            for out, s in zip(outs, batch_stems):
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                keep = scores >= 0.01
                preds[s] = np.hstack([boxes[keep], scores[keep, None]]).astype(np.float32)
            batch_imgs, batch_stems = [], []
    if batch_imgs:
        with torch.autocast("cuda", dtype=torch.float16):
            outs = model([im.cuda() for im in batch_imgs])
        for out, s in zip(outs, batch_stems):
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            keep = scores >= 0.01
            preds[s] = np.hstack([boxes[keep], scores[keep, None]]).astype(np.float32)
    return preds


def topk(preds, k=100):
    out = {}
    for stem, p in preds.items():
        if len(p) > k:
            p = p[np.argsort(-p[:, 4])[:k]]
        out[stem] = p
    return out


# ---------------- main ----------------
def evaluate_model(name, preds_fn):
    preds = topk(preds_fn(), k=100)
    # COCO detections
    results = []
    for stem in VAL_STEMS:
        p = preds[stem]
        boxes, scores = p[:, :4], p[:, 4]
        results.append((ID_MAP[stem], boxes, scores, np.zeros(len(p), dtype=int)))
    dets = eval_utils.dets_from_xyxy(results, ID_MAP)
    stats, (rec, pr), _ = eval_utils.evaluate_dets(dets, split="val")
    prec, rec_p = compute_pr(gt_boxes, preds)
    np.savez(EVAL_DIR / f"pr_{name}.npz", recall=rec, precision=pr)
    return {
        "name": name,
        "ap50_95": float(stats[0]), "ap50": float(stats[1]), "ap75": float(stats[2]),
        "precision": float(prec), "recall": float(rec_p),
    }


if __name__ == "__main__":
    gt_boxes = load_gt_boxes()
    jobs = {
        "yolov5s": lambda: infer_yolo(EXP / "runs_yolo" / "yolov5s" / "weights" / "best.pt"),
        "yolov8s": lambda: infer_yolo(EXP / "runs_yolo" / "yolov8s" / "weights" / "best.pt"),
        "yolo11s": lambda: infer_yolo(EXP / "runs_yolo" / "yolo11s" / "weights" / "best.pt"),
        "rtdetr-l": lambda: infer_yolo(EXP / "runs_yolo" / "rtdetr-l" / "weights" / "best.pt"),
        "ssd": lambda: infer_tv("ssd"),
        "frcnn": lambda: infer_tv("frcnn"),
    }
    metrics = {}
    for name, fn in jobs.items():
        print(f">>> evaluating {name} ...", flush=True)
        try:
            m = evaluate_model(name, fn)
            metrics[name] = m
            print(f"    {name}: AP50={m['ap50']:.4f} AP50-95={m['ap50_95']:.4f} "
                  f"P={m['precision']:.4f} R={m['recall']:.4f}", flush=True)
        except Exception as e:
            print(f"    {name} FAILED: {e}", flush=True)
    (EVAL_DIR / "combined_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("ALL EVALUATIONS DONE", flush=True)
