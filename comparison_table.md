| Method | Backbone | Params (M) ↓ | FLOPs (G) ↓ | P ↑ | R ↑ | mAP@0.5 ↑ | mAP@0.5:0.95 ↑ | FPS ↑ |
|---|---|---|---|---|---|---|---|---|
| Faster R-CNN | ResNet-50-FPN | 41.30 | 181.5 | 0.8724 | 0.9888 | 0.9864 | 0.7391 | 20.5 |
| SSD300 | VGG-16 | 23.75 | 60.9 | **0.9865** | 0.9644 | 0.9791 | 0.6861 | 39.6 |
| YOLOv5s | CSPDarknet53 | **9.12** | 24.0 † | 0.9818 | 0.9908 | 0.9893 † | 0.7997 | 83.7 † |
| YOLOv8s | C2f-Darknet | 11.14 | 28.6 | 0.9823 | 0.9919 † | 0.9893 | 0.7985 | **87.4** |
| YOLOv11s | C3k2-Darknet | 9.43 † | **21.7** | 0.9853 † | 0.9903 | **0.9894** | 0.8057 † | 63.4 |
| RT-DETR-L | HGNetv2 | 32.81 | 109.9 | 0.9606 | **0.9934** | 0.9887 | **0.8114** | 26.9 |

The best results are in **bold**, the second-best are marked with †. All models were fine-tuned from COCO-pretrained weights (100 epochs for YOLO/RT-DETR; 60 for SSD300; 50 for Faster R-CNN, batch sizes 4-16 on an RTX 3060 Laptop 6GB). FLOPs are computed at 640×640 input except SSD300 (300×300). FPS measured end-to-end (including NMS), batch size 1, FP16, on the same GPU. All metrics are evaluated on the same validation set (203 images) with the same COCO evaluator.
