"""Stage 2: the YOLO defect detector (thin wrapper around Ultralytics).

Keeps the rest of the system independent of the Ultralytics API, so the
detector can later be swapped for an ONNX / TensorRT engine without touching
the gate, the coil tracker or the dashboard.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclasses.dataclass
class Detection:
    cls_id: int
    name: str
    conf: float
    xyxy: tuple[float, float, float, float]   # pixel coordinates in the input image

    @property
    def area_frac(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, (x2 - x1) * (y2 - y1))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class Detector:
    """Wraps a trained Ultralytics model (.pt, .onnx or .engine)."""

    def __init__(self, weights: str | Path, imgsz: int = 640, device: Optional[str] = None):
        from ultralytics import YOLO

        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.imgsz = imgsz
        self.device = device
        self.names: dict[int, str] = dict(self.model.names)
        self.last_speed_ms: dict[str, float] = {}

    def predict(self, image_bgr: np.ndarray, conf_floor: float = 0.05) -> list[Detection]:
        """Run the detector with a LOW floor; the gate applies real thresholds later.

        Using a low floor here and thresholding per class in the gate is what
        lets one model serve classes with very different confidence profiles
        (e.g. crazing peaks near 0.2, patches near 0.5).
        """
        t0 = time.perf_counter()
        kw = {"conf": conf_floor, "imgsz": self.imgsz, "verbose": False}
        if self.device is not None:
            kw["device"] = self.device
        r = self.model.predict(image_bgr, **kw)[0]
        self.last_speed_ms = {k: float(v) for k, v in r.speed.items()}
        self.last_speed_ms["wall"] = (time.perf_counter() - t0) * 1000.0
        dets: list[Detection] = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for b, c, k in zip(xyxy, conf, cls):
            dets.append(Detection(int(k), self.names.get(int(k), str(k)), float(c), tuple(float(v) for v in b)))
        return dets

    @staticmethod
    def draw(image_bgr: np.ndarray, dets: Sequence[Detection], colour_for=None, max_boxes: int = 10,
             max_labels: int = 4) -> np.ndarray:
        """Annotate the most confident detections on a copy of the image (BGR).

        Small NEU-DET frames (200 px) are upscaled first so labels stay legible;
        only the top ``max_labels`` boxes get a text label to keep the frame readable.
        """
        import cv2

        out = image_bgr.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        scale = 1.0
        if max(out.shape[:2]) < 480:
            scale = 480.0 / max(out.shape[:2])
            out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        fs = max(0.4, out.shape[1] / 1000.0)
        thick = max(1, int(round(out.shape[1] / 300)))
        ordered = sorted(dets, key=lambda d: -d.conf)[:max_boxes]
        for i, d in enumerate(ordered):
            x1, y1, x2, y2 = (int(round(v * scale)) for v in d.xyxy)
            colour = colour_for(d) if colour_for else (0, 165, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick)
            if i >= max_labels:
                continue
            label = f"{d.name} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ty = y1 if y1 - th - 6 >= 0 else y2 + th + 6
            cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty), colour, -1)
            cv2.putText(out, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        if len(dets) > max_boxes:
            cv2.putText(out, f"+{len(dets) - max_boxes} more boxes", (6, out.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        return out
