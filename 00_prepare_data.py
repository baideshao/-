# -*- coding: utf-8 -*-
"""VOC XML -> YOLO format, 80/20 train/val split (seed=42)."""
import os
import shutil
import xml.etree.ElementTree as ET
import random
from pathlib import Path

ROOT = Path(r"D:/pig-objection detection")
EXP = ROOT / "experiments"
DATA_DIR = ROOT / "data"
LABEL_DIR = ROOT / "label"

OUT = EXP / "dataset"
for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

# class name -> id
CLASSES = ["pig"]
name2id = {n: i for i, n in enumerate(CLASSES)}

def convert_one(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)
    lines = []
    for obj in root.iter("object"):
        name = obj.find("name").text.strip()
        if name not in name2id:
            print(f"skip unknown class {name} in {xml_path.name}")
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text); ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text); ymax = float(bb.find("ymax").text)
        xmin, xmax = sorted([xmin, xmax])
        ymin, ymax = sorted([ymin, ymax])
        # clip to image
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(W, xmax); ymax = min(H, ymax)
        if xmax - xmin < 2 or ymax - ymin < 2:
            continue
        cx = (xmin + xmax) / 2 / W
        cy = (ymin + ymax) / 2 / H
        w = (xmax - xmin) / W
        h = (ymax - ymin) / H
        lines.append(f"{name2id[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines

xmls = sorted(p.name for p in LABEL_DIR.glob("*.xml"))
random.seed(42)
random.shuffle(xmls)
n_val = int(round(len(xmls) * 0.2))
val_set = set(xmls[:n_val])

stats = {"train": 0, "val": 0}
for xml_name in xmls:
    stem = xml_name[:-4]
    img_src = DATA_DIR / f"{stem}.jpg"
    if not img_src.exists():
        print(f"missing image {stem}, skip")
        continue
    split = "val" if xml_name in val_set else "train"
    lines = convert_one(LABEL_DIR / xml_name)
    if not lines:
        print(f"no valid boxes in {xml_name}, skip")
        continue
    shutil.copy(img_src, OUT / "images" / split / f"{stem}.jpg")
    (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    stats[split] += 1

# dataset.yaml
(EXP / "dataset.yaml").write_text(
    f"path: {OUT.as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n  0: pig\n"
)
# split record for torchvision loader
(EXP / "split.txt").write_text("\n".join(
    f"{xml_name[:-4]} {split}" for xml_name in xmls
    if (DATA_DIR / f"{xml_name[:-4]}.jpg").exists()
) + "\n")
print(stats)
print("done")
