| Tracker | HOTA ↑ | MOTA ↑ | IDF1 ↑ | AssA ↑ | DetA ↑ | IDSW ↓ | FP ↓ | FN ↓ | FPS ↑ |
|---|---|---|---|---|---|---|---|---|---|
| ByteTrack | 0.8 | 0.9 | 0.8 | 0.8 | 0.9 | 158 † | 9227 | 4012 | 492.9 † |
| OC-SORT | 0.8 | **0.9** | 0.8 | 0.8 | **0.9** | 187 | 7628 † | 4366 | **567.8** |
| Deep OC-SORT | 0.8 | 0.9 † | 0.8 | 0.8 | 0.9 † | 178 | **7626** | 4346 | 19.7 |
| StrongSORT | 0.8 | 0.7 | 0.7 | 0.8 | 0.8 | **111** | 30607 | **3390** | 10.4 |
| BoT-SORT | **0.9** | 0.9 | **0.8** | **0.9** | 0.9 | 160 | 9136 | 3906 † | 19.7 |
| HybridSORT | 0.8 † | 0.9 | 0.8 † | 0.8 † | 0.9 | 167 | 7727 | 4286 | 20.1 |

The best results are in **bold**, the second-best are marked with †. All trackers share the same YOLOv11s detection input; DetA differences reflect how each tracker exploits the detections (confidence filtering, redundant tracks). ReID-based trackers share the same OSNet-x0.25 (MSMT17) appearance model. Metrics averaged over 7 sequences (12,749 frames, 1920×1080@30fps) with TrackEval. FPS measured on an RTX 3060 Laptop GPU (batch 1, FP16).
