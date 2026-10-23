#!/usr/bin/env python
"""Train the Stage 2 detector.

    # v1 baseline (what the deck reports)
    python scripts/train.py --model yolov8n.pt --epochs 30 --name v1_n30

    # v2: bigger model, longer, stronger augmentation
    python scripts/train.py --model yolov8s.pt --epochs 100 --name v2_s100 --aug strong

    # resume from your own weights (fine-tune on real line images)
    python scripts/train.py --model models/best_v1.pt --data config/jindal.yaml --epochs 60 --name jindal_ft

The best weights are copied to models/<name>.pt when training finishes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

AUG = {
    "default": {},
    # Steel texture has no "up" and lighting varies along a line, so vertical
    # flips and brightness jitter are valid; mosaic + mixup regularise the
    # small dataset; close_mosaic turns mosaic off for the final epochs.
    "strong": dict(hsv_v=0.5, hsv_s=0.3, degrees=10, translate=0.15, scale=0.5,
                   flipud=0.5, fliplr=0.5, mosaic=1.0, mixup=0.1, close_mosaic=10),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--data", default="config/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="v2_s100")
    ap.add_argument("--aug", choices=list(AUG), default="strong")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=args.batch,
        device=args.device, name=args.name, exist_ok=True, **AUG[args.aug],
    )
    # Ask the trainer where it saved (Ultralytics versions differ on the default run directory).
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    out = Path("models") / f"{args.name}.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy(best, out)
    print(f"run dir: {model.trainer.save_dir}\nbest weights -> {out}")


if __name__ == "__main__":
    main()
