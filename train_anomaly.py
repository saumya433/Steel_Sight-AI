#!/usr/bin/env python
"""Train and calibrate the Stage 1 anomaly screen.

    python scripts/train_anomaly.py --data YOLO --out models/anomaly_bank.pt

On NEU-DET every image contains a defect, so "normal texture" is taken from
the regions OUTSIDE the annotated boxes (with a margin). On a real line,
point --normal-dir at a folder of defect-free frames instead and no labels
are needed at all.

Outputs
    models/anomaly_bank.pt      memory bank + threshold (loaded by the pipeline)
    reports/anomaly.json        calibration numbers you can quote:
                                  - patch-level AUROC (defect patch vs normal patch)
                                  - image-level recall at the chosen false-alarm budget
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steelsight.anomaly import AnomalyConfig, AnomalyScreen, _downsample_mask, auroc, boxes_to_mask  # noqa: E402
from steelsight.data import label_path_for, list_images, read_yolo_labels  # noqa: E402


def load_split(folder: Path, limit: int | None):
    imgs, masks = [], []
    for p in list_images(folder / "images")[:limit]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        boxes = [b[1:] for b in read_yolo_labels(label_path_for(p), w, h)]
        imgs.append(im)
        masks.append(boxes_to_mask(boxes, h, w) if boxes else None)
    return imgs, masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="YOLO", help="YOLO layout root with train/ and val/")
    ap.add_argument("--normal-dir", default=None, help="folder of defect-free frames (real line); overrides --data for fitting")
    ap.add_argument("--out", default="models/anomaly_bank.pt")
    ap.add_argument("--limit", type=int, default=None, help="use at most N train images (speed)")
    ap.add_argument("--percentile", type=float, default=99.5, help="false-alarm budget on normal patches")
    ap.add_argument("--backbone", default="resnet18",
                    help='"resnet18" (ImageNet, PatchCore default) or "yolo:models/best_v1.pt" to share the detector backbone')
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = AnomalyConfig(backbone=args.backbone)
    if args.device:
        cfg.device = args.device
    screen = AnomalyScreen(cfg)

    if args.normal_dir:
        imgs = [cv2.imread(str(p)) for p in list_images(args.normal_dir)[: args.limit]]
        imgs = [im for im in imgs if im is not None]
        print(f"fitting on {len(imgs)} defect-free frames from {args.normal_dir}")
        screen.fit(imgs, masks=None)
    else:
        imgs, masks = load_split(Path(args.data) / "train", args.limit)
        print(f"fitting on background regions of {len(imgs)} train images")
        screen.fit(imgs, masks=masks)
    print("bank:", screen.calibration)

    # ---- calibrate + measure on the validation split
    vimgs, vmasks = load_split(Path(args.data) / "val", None)
    scores = screen.patch_scores(vimgs)                       # (n, gh, gw)
    gh, gw = scores.shape[1:]
    pos, neg, img_max = [], [], []
    for s, m in zip(scores, vmasks):
        img_max.append(float(s.max()))
        if m is None:
            neg.append(s.ravel())
            continue
        cell = _downsample_mask(m, gh, gw)
        pos.append(s[cell == 1])
        neg.append(s[cell == 0])
    pos_a = np.concatenate(pos) if pos else np.array([])
    neg_a = np.concatenate(neg) if neg else np.array([])
    thr = screen.calibrate(neg_a, percentile=args.percentile)
    report = {
        **screen.calibration,
        "val_images": len(vimgs),
        "patch_auroc_defect_vs_normal": round(auroc(pos_a, neg_a), 4) if len(pos_a) and len(neg_a) else None,
        "image_recall_at_budget": round(float(np.mean(np.array(img_max) >= thr)), 4),
        "normal_patch_false_alarm_budget": f"{100 - args.percentile:.2f}%",
        "note": "image_recall = fraction of defective val images whose max patch score >= threshold",
    }
    screen.save(args.out)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/anomaly.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
