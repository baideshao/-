# -*- coding: utf-8 -*-
"""Shared evaluation utilities: COCO GT build + pycocotools eval (same evaluator for all models)."""
import json
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(r"D:/pig-objection detection")
EXP = ROOT / "experiments"
EVAL_DIR = EXP / "eval_results"
EVAL_DIR.mkdir(exist_ok=True)

MAX_DET = 100


def get_split_stems(split):
    out = []
    for line in (EXP / "split.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == split:
            out.append(parts[0])
    return out


def get_image_id_map(split="val"):
    """{stem: coco_image_id} — MUST match the ids in build_gt_coco (sorted stems, 1-based)."""
    return {stem: i for i, stem in enumerate(sorted(get_split_stems(split)), start=1)}


def build_gt_coco(split="val"):
    """Build COCO-format GT json from VOC XMLs of a split (saved once)."""
    gt_path = EVAL_DIR / f"gt_{split}_coco.json"
    if gt_path.exists():
        return gt_path
    images, anns = [], []
    ann_id = 0
    for i, stem in enumerate(sorted(get_split_stems(split)), start=1):
        root = ET.parse(ROOT / "label" / f"{stem}.xml").getroot()
        size = root.find("size")
        W = int(size.find("width").text)
        H = int(size.find("height").text)
        images.append({"id": i, "file_name": f"{stem}.jpg", "width": W, "height": H})
        for obj in root.iter("object"):
            if obj.find("name").text.strip() != "pig":
                continue
            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text); ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text); ymax = float(bb.find("ymax").text)
            xmin, xmax = sorted([xmin, xmax]); ymin, ymax = sorted([ymin, ymax])
            w = max(xmax - xmin, 0); h = max(ymax - ymin, 0)
            ann_id += 1
            anns.append({"id": ann_id, "image_id": i, "category_id": 1,
                         "bbox": [xmin, ymin, w, h], "area": w * h, "iscrowd": 0})
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "pig"}]}
    gt_path.write_text(json.dumps(coco))
    return gt_path


def evaluate_dets(dets, gt_path=None, split="val"):
    """dets: list of {image_id, category_id, bbox, score}. Returns (stats, per_class_pr, eval_obj)."""
    gt_path = gt_path or build_gt_coco(split)
    coco_gt = COCO(str(gt_path))
    if len(dets) == 0:
        rec = np.linspace(0, 1, 101)
        return np.zeros(6), (rec, np.zeros(101)), None
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.maxDets = [1, 10, MAX_DET]
    ev.params.iouThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats  # [AP@.5:.95, AP@.5, AP@.75, APs, APm, APl]
    # per-class PR curve at IoU=0.5: precision[T=10,R=101,K,A,M]
    prec = ev.eval["precision"]
    rec = np.linspace(0, 1, prec.shape[1])
    pr = prec[0, :, 0, 0, 2]  # IoU idx 0 = 0.5, class 0, area all, maxDet idx 2
    return stats, (rec, pr), ev


def dets_from_xyxy(results, image_id_map):
    """Convert per-image xyxy predictions to COCO detections list.
    results: list of (image_id, boxes[N,4], scores[N], labels[N]).
    image_id_map: {stem: coco_image_id}."""
    dets = []
    for img_id, boxes, scores, labels in results:
        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(float, b)
            dets.append({"image_id": img_id, "category_id": int(l) + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "score": float(s)})
    return dets
