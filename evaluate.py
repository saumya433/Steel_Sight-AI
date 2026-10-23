#!/usr/bin/env python
"""Evaluate a detector like production: per-class metrics + per-class thresholds.

    python scripts/evaluate.py --weights models/best_v1.pt --data config/data.yaml

Writes
    reports/eval.json           mAP, per-class AP50/P/R/F1, speed, thresholds
    reports/*.png               confusion matrix and P/R/F1 curves from Ultralytics
    config/thresholds.json      per-class REVIEW floor (best F1) and AUTO floor
                                (first confidence where precision >= --auto-precision)

The threshold sweep is done here, independently of Ultralytics internals, by
matching predictions to ground truth at IoU >= 0.5 and sweeping confidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steelsight.data import label_path_for, list_images, load_classes, read_yolo_labels  # noqa: E402
from steelsight.gate import GateConfig  # noqa: E402


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def sweep(model, val_images: list[Path], classes: list[str], iou_thr: float = 0.5, imgsz: int = 640):
    """Return per-class {conf: [...], tp: [...], n_gt: int} after greedy IoU matching."""
    per = {c: {"conf": [], "tp": [], "n_gt": 0} for c in classes}
    for p in val_images:
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        gts = read_yolo_labels(label_path_for(p), w, h)
        for g in gts:
            per[classes[g[0]]]["n_gt"] += 1
        r = model.predict(im, conf=0.001, imgsz=imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy(); conf = r.boxes.conf.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int)
        order = np.argsort(-conf)
        matched = [False] * len(gts)
        for i in order:
            name = classes[cls[i]]
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if matched[j] or g[0] != cls[i]:
                    continue
                v = iou(xyxy[i], g[1:])
                if v > best:
                    best, bj = v, j
            tp = best >= iou_thr
            if tp:
                matched[bj] = True
            per[name]["conf"].append(float(conf[i]))
            per[name]["tp"].append(bool(tp))
    return per


def pr_at(conf: np.ndarray, tp: np.ndarray, n_gt: int, t: float) -> tuple[float, float, float]:
    keep = conf >= t
    tps = int(tp[keep].sum()); fps = int((~tp[keep]).sum())
    p = tps / (tps + fps) if tps + fps else 0.0
    r = tps / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="config/data.yaml")
    ap.add_argument("--val-images", default=None, help="override: folder of val images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--auto-precision", type=float, default=0.90, help="precision target for the AUTO_FLAG floor")
    ap.add_argument("--min-review", type=float, default=0.10, help="never set a review floor below this")
    ap.add_argument("--anomaly-bank", default="models/anomaly_bank.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    import yaml

    classes = load_classes(args.data)
    model = YOLO(args.weights)
    Path("reports").mkdir(exist_ok=True)

    # 1) Ultralytics val: mAP numbers + plots
    kw = {"data": args.data, "imgsz": args.imgsz, "plots": True, "verbose": False, "name": "eval", "exist_ok": True}
    if args.device:
        kw["device"] = args.device
    m = model.val(**kw)
    per_class_ap = {classes[int(i)]: round(float(ap50), 4) for i, ap50 in zip(m.box.ap_class_index, m.box.ap50)}
    for png in Path(m.save_dir).glob("*.png"):
        shutil.copy(png, Path("reports") / png.name)

    # 2) independent threshold sweep
    if args.val_images:
        val_images = list_images(args.val_images)
    else:
        d = yaml.safe_load(Path(args.data).read_text())
        val_images = list_images(Path(d["path"]) / d["val"])
    per = sweep(model, val_images, classes, imgsz=args.imgsz)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    review, auto, table = {}, {}, {}
    for c in classes:
        conf = np.array(per[c]["conf"]); tp = np.array(per[c]["tp"], dtype=bool); n_gt = per[c]["n_gt"]
        if len(conf) == 0 or n_gt == 0:
            review[c], auto[c] = 0.25, 0.60
            table[c] = {"n_gt": n_gt, "note": "no predictions or no ground truth"}
            continue
        f1s = [pr_at(conf, tp, n_gt, t) for t in grid]
        best_i = int(np.argmax([f for _, _, f in f1s]))
        t_rev = max(args.min_review, float(grid[best_i]))
        t_auto = None
        for t, (p, r, f) in zip(grid, f1s):
            if t >= t_rev and p >= args.auto_precision and (conf >= t).sum() >= 5:
                t_auto = float(t); break
        if t_auto is None:
            t_auto = max(t_rev, 0.60)
        review[c], auto[c] = round(t_rev, 2), round(t_auto, 2)
        p, r, f = f1s[best_i]
        pa, ra, _ = pr_at(conf, tp, n_gt, t_auto)
        table[c] = {"n_gt": n_gt, "ap50": per_class_ap.get(c), "review_floor": review[c], "precision_at_review": round(p, 3),
                    "recall_at_review": round(r, 3), "f1_at_review": round(f, 3), "auto_floor": auto[c],
                    "precision_at_auto": round(pa, 3), "recall_at_auto": round(ra, 3)}

    cfg = GateConfig(review_min=0.25, auto_flag=0.60, per_class_review=review, per_class_auto=auto)
    if Path(args.anomaly_bank).exists():
        import torch
        blob = torch.load(args.anomaly_bank, map_location="cpu", weights_only=False)
        cfg.anomaly_threshold = blob.get("threshold")
    cfg.save("config/thresholds.json")

    report = {
        "weights": args.weights,
        "val_images": len(val_images),
        "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
        "speed_ms": {k: round(float(v), 2) for k, v in m.speed.items()},
        "per_class": table,
        "gate": json.loads(Path("config/thresholds.json").read_text()),
    }
    Path("reports/eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2))
    print("\nper-class thresholds (review / auto):")
    for c in classes:
        print(f"  {c:<16} {review[c]:.2f} / {auto[c]:.2f}   AP50 {per_class_ap.get(c, float('nan')):.3f}")
    print("\nwrote reports/eval.json, reports/*.png, config/thresholds.json")


if __name__ == "__main__":
    main()
