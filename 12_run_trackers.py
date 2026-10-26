# -*- coding: utf-8 -*-
"""Run 6 famous trackers (all 2021+) with the SAME YOLOv11s detections, one pass per sequence.
Results -> TrackEval layout: results/trackers/mot_challenge/pig-train/{tracker}/data/{seq}.txt
Also measures per-tracker update time (-> FPS)."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"D:/PIG_MOT_TEST_DATA/tracking_experiments/scripts")

ROOT = Path(r"D:/PIG_MOT_TEST_DATA")
EXP = ROOT / "tracking_experiments"
DET_DIR = EXP / "detections"
RES_DIR = EXP / "results" / "trackers" / "mot_challenge" / "pig-train"
SEQS = [str(s) for s in range(191, 198)]

from boxmot import BoTSORT, StrongSORT, DeepOCSORT, OCSORT, BYTETracker, HybridSORT

# runtime patch: boxmot 10.0.79 hybridsort uses a 9-dim state with 5-dim
# measurement z [x,y,s,score,r], but KalmanFilterXYSR.unfreeze() unpacks only 4
# dims -> crash. Generalize unfreeze for dim_z==5; keep original for dim_z==4.
from collections import deque
from copy import deepcopy as _deepcopy
from boxmot.motion.kalman_filters.xysr_kf import KalmanFilterXYSR as _KF

_orig_unfreeze = _KF.unfreeze


def _unfreeze_patched(self):
    if self.dim_z != 5:
        return _orig_unfreeze(self)
    if self.attr_saved is None:
        return
    new_history = _deepcopy(list(self.history_obs))
    self.__dict__ = self.attr_saved
    self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)
    occur = [int(d is None) for d in new_history]
    indices = np.where(np.array(occur) == 0)[0]
    index1, index2 = indices[-2], indices[-1]
    z1 = new_history[index1].flatten()
    z2 = new_history[index2].flatten()
    time_gap = index2 - index1
    dz = (z2 - z1) / time_gap
    for i in range(index2 - index1):
        new_box = (z1 + (i + 1) * dz).reshape((self.dim_z, 1))
        self.update(new_box)
        if not i == (index2 - index1 - 1):
            self.predict()
            self.history_obs.pop()
    self.history_obs.pop()


_KF.unfreeze = _unfreeze_patched

TRACKERS = {
    "bytetrack": BYTETracker,
    "ocsort": OCSORT,
    "deepocsort": DeepOCSORT,
    "strongsort": StrongSORT,
    "botsort": BoTSORT,
    "hybridsort": HybridSORT,
}
WEIGHTS = EXP / "weights"  # pre-downloaded ReID weights (avoid blocked github)

T0 = time.time()


def load_dets(seq):
    dets = {}
    for line in (DET_DIR / f"{seq}.txt").read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        frame, conf = int(p[0]), float(p[6])
        x, y, w, h = map(float, p[2:6])
        dets.setdefault(frame, []).append([x, y, x + w, y + h, conf, 0.0])
    return {f: np.array(v, dtype=np.float32) for f, v in dets.items()}


def make_tracker(name):
    cls = TRACKERS[name]
    # all ReID-based trackers share the SAME appearance model (fair comparison,
    # isolates the association algorithm); filename must contain the model type
    from pathlib import Path as _P
    reid = _P(WEIGHTS / "osnet_x0_25_msmt17.pt")
    if name == "botsort":
        return cls(model_weights=reid, device="cuda:0", fp16=True)
    if name == "strongsort":
        return cls(model_weights=reid, device="cuda:0", fp16=True)
    if name == "deepocsort":
        return cls(model_weights=reid, device="cuda:0", fp16=True)
    if name == "hybridsort":
        return cls(reid_weights=reid, device="cuda:0", half=True, det_thresh=0.3)
    return cls()  # bytetrack / ocsort (motion-only, no ReID)


def main():
    fps_stats = {name: [] for name in TRACKERS}
    for seq in SEQS:
        trackers = {name: make_tracker(name) for name in TRACKERS}
        outs = {name: [] for name in TRACKERS}
        dets = load_dets(seq)
        cap = cv2.VideoCapture(str(ROOT / f"MyVideo_{seq}.mp4"))
        frame_idx = 1
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        t_seq = time.time()
        while True:
            ok, img = cap.read()
            if not ok:
                break
            d = dets.get(frame_idx)
            d = d if d is not None and len(d) else np.zeros((0, 6), dtype=np.float32)
            for name, tracker in trackers.items():
                t0 = time.perf_counter()
                tracks = tracker.update(d, img)
                dt = time.perf_counter() - t0
                if frame_idx <= 500:
                    fps_stats[name].append(dt)
                for t in tracks:  # x1,y1,x2,y2,id,conf,cls,(ind)
                    x1, y1, x2, y2, tid = map(float, t[:5])
                    conf = float(t[5]) if len(t) > 5 else -1.0
                    outs[name].append(
                        f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},-1,-1\n")
            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"[{seq}] {frame_idx}/{total} ({time.time()-t_seq:.0f}s)", flush=True)
        cap.release()
        for name in TRACKERS:
            out_dir = RES_DIR / name / "data"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{seq}.txt").write_text("".join(outs[name]))
        print(f"[{seq}] done in {(time.time()-t_seq)/60:.1f} min", flush=True)
    # FPS summary (per-tracker mean update time over first 500 frames of the first seqs)
    with open(EXP / "eval_outputs" / "fps_raw.txt", "w") as f:
        for name in TRACKERS:
            ts = fps_stats[name]
            mean_ms = float(np.mean(ts)) * 1000
            f.write(f"{name} mean_update_ms={mean_ms:.2f} fps={1000 / mean_ms:.1f} n={len(ts)}\n")
            print(f"{name}: {1000 / mean_ms:.1f} FPS (update {mean_ms:.2f} ms)", flush=True)
    print(f"ALL TRACKING DONE in {(time.time()-T0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
