#!/usr/bin/env python
"""Export the detector for deployment and measure speed.

    python scripts/export.py --weights models/best_v1.pt --format onnx
    python scripts/export.py --weights models/best_v1.pt --format engine --half      # TensorRT FP16 (needs a GPU + TensorRT)
    python scripts/export.py --weights models/best_v1.pt --format engine --int8 --data config/data.yaml

Why: on the line the model runs on an edge GPU (Jetson / small RTX). ONNX is
the portable intermediate; TensorRT FP16/INT8 is what gets you from ~10 ms to
<15 ms END-TO-END at 640 px, or several hundred FPS at 320 px.

After export, the pipeline loads the engine the same way as .pt:
    InspectionPipeline("models/best_v1.engine", ...)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def bench(weights: str, imgsz: int, n: int = 50, device=None) -> dict:
    from ultralytics import YOLO
    model = YOLO(weights)
    dummy = (np.random.rand(imgsz, imgsz, 3) * 255).astype(np.uint8)
    kw = {"imgsz": imgsz, "verbose": False}
    if device is not None:
        kw["device"] = device
    for _ in range(5):
        model.predict(dummy, **kw)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(dummy, **kw)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"weights": weights, "imgsz": imgsz, "ms_per_image": round(ms, 2), "fps": round(1000 / ms, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "torchscript"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--data", default="config/data.yaml", help="calibration data for INT8")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bench", action="store_true", help="benchmark before and after")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.bench:
        print("before:", bench(args.weights, args.imgsz, device=args.device))
    model = YOLO(args.weights)
    kw = {"format": args.format, "imgsz": args.imgsz, "half": args.half, "simplify": True}
    if args.int8:
        kw.update({"int8": True, "data": args.data})
    if args.device is not None:
        kw["device"] = args.device
    out = model.export(**kw)
    print("exported:", out)
    if args.bench:
        print("after:", bench(str(out), args.imgsz, device=args.device))


if __name__ == "__main__":
    main()
