# -*- coding: utf-8 -*-
"""Train SSD300-VGG16 and Faster R-CNN R50-FPN (torchvision, COCO-pretrained) on the pig dataset.
Usage: 02_train_torchvision.py --model ssd|frcnn [--batch N]"""
import argparse
import csv
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, r"D:/pig-objection detection/experiments/scripts")  # for -I isolated mode

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.models.detection import (
    ssd300_vgg16, SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import eval_utils

ROOT = Path(r"D:/pig-objection detection")
EXP = ROOT / "experiments"
CKPT_DIR = EXP / "torch_models"
CKPT_DIR.mkdir(exist_ok=True)
SEED = 42


# ---------------- dataset ----------------
class PigDataset(Dataset):
    def __init__(self, split, transform, only_valid=True):
        self.stems = eval_utils.get_split_stems(split)
        self.transform = transform
        self.img_dir = ROOT / "data"
        self.xml_dir = ROOT / "label"
        # for training, pre-filter images whose boxes would vanish after 300px resize
        if only_valid and split == "train":
            valid = []
            for stem in self.stems:
                root = ET.parse(self.xml_dir / f"{stem}.xml").getroot()
                for obj in root.iter("object"):
                    if obj.find("name").text.strip() != "pig":
                        continue
                    bb = obj.find("bndbox")
                    w = float(bb.find("xmax").text) - float(bb.find("xmin").text)
                    h = float(bb.find("ymax").text) - float(bb.find("ymin").text)
                    if abs(w) >= 5 and abs(h) >= 5:
                        valid.append(stem)
                        break
            self.stems = valid
        random.Random(SEED).shuffle(self.stems)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        W, H = img.size
        root = ET.parse(self.xml_dir / f"{stem}.xml").getroot()
        boxes, labels = [], []
        for obj in root.iter("object"):
            if obj.find("name").text.strip() != "pig":
                continue
            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text); ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text); ymax = float(bb.find("ymax").text)
            xmin, xmax = sorted([xmin, xmax]); ymin, ymax = sorted([ymin, ymax])
            xmin = max(0, xmin); ymin = max(0, ymin)
            xmax = min(W, xmax); ymax = min(H, ymax)
            if xmax - xmin >= 2 and ymax - ymin >= 2:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # torchvision: 0 = background
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=(H, W), dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        img_t = tv_tensors.Image(img)
        img_t = self.transform(img_t)  # dtype only; resize+normalize happen inside the model
        return img_t, target, stem


def collate(batch):
    return tuple(zip(*batch))


# ---------------- evaluate ----------------
@torch.no_grad()
def evaluate(model, val_loader, id_map, device):
    model.eval()
    results = []
    for images, _, stems in val_loader:
        images = [im.to(device) for im in images]
        with torch.autocast("cuda", dtype=torch.float16):
            outs = model(images)
        for out, stem in zip(outs, stems):
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy() - 1  # torchvision labels are 1-indexed; make 0-indexed
            keep = scores >= 0.01
            results.append((id_map[stem], boxes[keep], scores[keep], labels[keep]))
    dets = eval_utils.dets_from_xyxy(results, id_map)
    stats, pr, _ = eval_utils.evaluate_dets(dets, split="val")
    return stats  # [AP50-95, AP50, AP75, APs, APm, APl, ...]


# ---------------- train loop ----------------
def train(model, train_loader, val_loader, device, args, id_map):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    if args.model == "ssd":
        milestones = [45]
    else:
        milestones = [max(1, args.epochs - 10)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")

    log_path = EXP / "eval_results" / f"{args.model}_train_log.csv"
    f = open(log_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["epoch", "ap50_95", "ap50", "ap75", "lr", "time_min"])

    best_ap = -1
    t_start = time.time()
    for epoch in range(args.epochs):
        # Faster R-CNN: freeze backbone for epoch 0, then unfreeze (official recipe)
        if args.model == "frcnn" and epoch == 1:
            for p in model.parameters():
                p.requires_grad = True
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.1)

        model.train()
        t0 = time.time()
        for images, targets, _ in train_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(images, targets)
                if isinstance(out, tuple):  # SSD returns (loss_dict, detections)
                    out = out[0]
                losses = sum(v for v in out.values())
            scaler.scale(losses).backward()
            if args.model == "frcnn":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        scheduler.step()

        stats = evaluate(model, val_loader, id_map, device)
        ap5095, ap50 = stats[0], stats[1]
        writer.writerow([epoch, f"{ap5095:.4f}", f"{ap50:.4f}", f"{stats[2]:.4f}",
                         f"{optimizer.param_groups[0]['lr']:.2e}", f"{(time.time()-t0)/60:.1f}"])
        f.flush()
        print(f"[{args.model}] ep{epoch} AP50-95={ap5095:.4f} AP50={ap50:.4f} "
              f"({(time.time()-t0)/60:.1f} min/epoch, total {(time.time()-t_start)/60:.1f} min)", flush=True)
        if ap5095 > best_ap:
            best_ap = ap5095
            torch.save({"model": model.state_dict(), "epoch": epoch, "ap5095": ap5095, "ap50": ap50},
                       CKPT_DIR / f"{args.model}_best.pth")
            print(f"    saved best {args.model}_best.pth AP50-95={ap5095:.4f}", flush=True)
    f.close()
    return best_ap


def build_model(name):
    if name == "ssd":
        # load COCO weights, then swap the classification head for 1 class (+bg)
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, 2)
        transform = SSD300_VGG16_Weights.COCO_V1.transforms()
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            min_size=640, max_size=640)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        for p in model.backbone.parameters():  # frozen for epoch 0
            p.requires_grad = False
        transform = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    return model, transform


def run(batch, model_name, epochs):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda")
    model, transform = build_model(model_name)
    ds_train = PigDataset("train", transform, only_valid=True)
    ds_val = PigDataset("val", transform, only_valid=False)
    id_map = eval_utils.get_image_id_map("val")
    train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch, shuffle=False,
                            collate_fn=collate, num_workers=2, pin_memory=True)
    model.to(device)
    train(model, train_loader, val_loader, device,
          argparse.Namespace(model=model_name, epochs=epochs), id_map)
    print(f"[{model_name}] TRAINING DONE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ssd", "frcnn"])
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()
    epochs = 60 if args.model == "ssd" else 50
    batch = args.batch or (16 if args.model == "ssd" else 4)
    try:
        run(batch, args.model, epochs)
    except Exception as e:
        if "out of memory" in str(e).lower() and batch >= 2:
            print(f"[OOM] retry with batch={batch // 2}", flush=True)
            torch.cuda.empty_cache()
            run(batch // 2, args.model, epochs)
        else:
            raise


if __name__ == "__main__":
    main()
