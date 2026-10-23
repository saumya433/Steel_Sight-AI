# models/

Put your trained weights here (not tracked by the training scripts, committed to git on purpose so the Space/Cloud app can load them):

| file | what | how |
|---|---|---|
| `best_v1.pt` | Stage 2 detector (YOLOv8n, 30 epochs, 73.3% mAP@50) | copy `runs/detect/train-13/weights/best.pt` from Colab |
| `anomaly_bank.pt` | Stage 1 memory bank + calibrated threshold | `python scripts/train_anomaly.py` |
| `v2_s100.pt` | optional v2 (YOLOv8s, strong aug) | `python scripts/train.py --model yolov8s.pt --epochs 100 --name v2_s100` |

The dashboard reads `models/best_v1.pt` by default; override with `STEELSIGHT_WEIGHTS=models/v2_s100.pt`.
