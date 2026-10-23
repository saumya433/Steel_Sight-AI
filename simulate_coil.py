#!/usr/bin/env python
"""Run the full pipeline over a folder of frames as if a coil were passing the camera.

    python scripts/simulate_coil.py --images YOLO/val/images --coil COIL-2026-1042 --speed 300 --fps 20

Logs every frame to the results store (data/steelsight.db), builds the coil
defect map and runs the roll diagnosis. The dashboard's Coil tab reads the
same store, so this is also how you pre-load demo data before a presentation.

--demo-periodic N plants a synthetic repeating scratch every N metres on top
of the real events, to SHOW the roll-diagnosis mechanism. It prints a clear
"synthetic" marker; never present it as a real measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steelsight.coil import DEFAULT_ROLL_TABLE_MM, CoilTracker, best_diagnosis, diagnose_by_class  # noqa: E402
from steelsight.data import list_images  # noqa: E402
from steelsight.drift import DriftMonitor  # noqa: E402
from steelsight.gate import GateConfig  # noqa: E402
from steelsight.pipeline import InspectionPipeline  # noqa: E402
from steelsight.store import ResultStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="YOLO/val/images")
    ap.add_argument("--weights", default="models/best_v1.pt")
    ap.add_argument("--anomaly-bank", default="models/anomaly_bank.pt")
    ap.add_argument("--thresholds", default="config/thresholds.json")
    ap.add_argument("--coil", default="COIL-DEMO-001")
    ap.add_argument("--speed", type=float, default=300.0, help="line speed m/min")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--db", default="data/steelsight.db")
    ap.add_argument("--device", default=None)
    ap.add_argument("--demo-periodic", type=float, default=None, help="plant a synthetic repeating defect every N metres")
    args = ap.parse_args()

    gate = GateConfig.load(args.thresholds) if Path(args.thresholds).exists() else GateConfig()
    pipe = InspectionPipeline(args.weights, gate, anomaly_bank=args.anomaly_bank, device=args.device)
    store = ResultStore(args.db)
    tracker = CoilTracker(args.coil, line_speed_m_per_min=args.speed, fps=args.fps)
    drift = DriftMonitor(baseline_n=40, window=20)

    images = list_images(args.images)[: args.limit]
    print(f"inspecting {len(images)} frames from {args.images} as {args.coil} at {args.speed} m/min")
    for i, p in enumerate(images):
        im = cv2.imread(str(p))
        res = pipe.inspect(im, draw_heatmap=False)
        pos = tracker.add_frame(i, res.decision.value, res.gate.kept, image_h=im.shape[0])
        store.log_inspection(res.to_record(), coil_id=args.coil, frame=i, position_m=pos, image_path=str(p))
        drift.update(anomaly=res.anomaly_score, confidence=res.gate.max_conf or None)
        if i % 20 == 0:
            print(f"  frame {i:4d}  pos {pos:7.2f} m  {res.decision.value:<9} {res.gate.reason[:60]}  ({res.timings_ms['total_ms']:.0f} ms)")

    print("\ncoil summary:")
    print(json.dumps(tracker.summary(), indent=2))

    planted = None
    if args.demo_periodic:
        planted = [k * args.demo_periodic for k in range(1, int(tracker.length_m / args.demo_periodic) + 1)]
        print(f"\n[SYNTHETIC] planted {len(planted)} repeating events at {args.demo_periodic} m pitch")
    diags = diagnose_by_class(tracker.events, DEFAULT_ROLL_TABLE_MM, extra_positions=planted, n_frames=tracker.frames_seen)
    print("\nroll diagnosis (per class, AUTO_FLAG events only):")
    for name, d in diags.items():
        print(f"  {name:<16} {d.message}")
    best = best_diagnosis(diags)
    if best:
        print(f"  -> strongest periodic signal: {best[0]}: {best[1].message}")
    for metric in ("anomaly", "confidence"):
        s = drift.status(metric)
        if s:
            print("drift:", s.message)
    print(f"\nlogged to {args.db}; open the dashboard Coil tab to see the map.")


if __name__ == "__main__":
    main()
