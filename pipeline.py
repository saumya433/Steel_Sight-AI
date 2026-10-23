"""End-to-end inspection of one frame: Stage 1 -> Stage 2 -> gate -> annotated image."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .anomaly import AnomalyScreen
from .detector import Detection, Detector
from .gate import DECISION_COLOURS_BGR, Decision, GateConfig, GateResult, decide


@dataclasses.dataclass
class InspectionResult:
    decision: Decision
    gate: GateResult
    detections: list[Detection]          # everything the detector returned (low floor)
    anomaly_score: Optional[float]
    heatmap: Optional[np.ndarray]        # HxW float32, Stage 1 patch distances
    annotated: np.ndarray                # BGR image with boxes + decision banner
    timings_ms: dict[str, float]

    def to_record(self) -> dict:
        return {
            "decision": self.decision.value,
            "anomaly_score": self.anomaly_score,
            "anomaly_fired": self.gate.anomaly_fired,
            "n_detections": len(self.gate.kept),
            "max_conf": self.gate.max_conf,
            "classes": sorted({d.name for d in self.gate.kept}),
            "detections": [d.to_dict() for d in self.gate.kept],
            "reason": self.gate.reason,
            "timings_ms": self.timings_ms,
        }


class InspectionPipeline:
    def __init__(
        self,
        detector_weights: str | Path,
        gate_config: GateConfig,
        anomaly_bank: Optional[str | Path] = None,
        imgsz: int = 640,
        device: Optional[str] = None,
    ):
        self.detector = Detector(detector_weights, imgsz=imgsz, device=device)
        self.gate_cfg = gate_config
        self.anomaly: Optional[AnomalyScreen] = None
        if anomaly_bank and Path(anomaly_bank).exists():
            self.anomaly = AnomalyScreen.load(anomaly_bank, device=device)
            if self.gate_cfg.anomaly_threshold is None and self.anomaly.threshold is not None:
                self.gate_cfg.anomaly_threshold = self.anomaly.threshold

    def inspect(self, image_bgr: np.ndarray, draw_heatmap: bool = True) -> InspectionResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        score, heat = None, None
        if self.anomaly is not None:
            score, heat = self.anomaly.score(image_bgr)
            timings["stage1_ms"] = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        dets = self.detector.predict(image_bgr)
        timings["stage2_ms"] = (time.perf_counter() - t1) * 1000.0
        timings["stage2_inference_ms"] = self.detector.last_speed_ms.get("inference", float("nan"))

        g = decide(dets, self.gate_cfg, anomaly_score=score, image_hw=image_bgr.shape[:2])
        timings["total_ms"] = (time.perf_counter() - t0) * 1000.0

        annotated = self._annotate(image_bgr, g, heat if draw_heatmap else None)
        return InspectionResult(g.decision, g, dets, score, heat, annotated, timings)

    def _annotate(self, image_bgr: np.ndarray, g: GateResult, heat: Optional[np.ndarray]) -> np.ndarray:
        import cv2

        base = image_bgr if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        out = base.copy()
        if heat is not None and g.decision == Decision.UNKNOWN:
            # show WHERE the surface looks abnormal when the detector is silent
            h = heat - heat.min()
            h = (255 * h / (h.max() + 1e-6)).astype(np.uint8)
            cmap = cv2.applyColorMap(h, cv2.COLORMAP_INFERNO)
            out = cv2.addWeighted(out, 0.55, cmap, 0.45, 0)
        colour = DECISION_COLOURS_BGR[g.decision]
        auto_ids = {id(d) for d in g.auto}
        out = Detector.draw(out, g.kept, colour_for=lambda d: DECISION_COLOURS_BGR[Decision.AUTO_FLAG] if id(d) in auto_ids else DECISION_COLOURS_BGR[Decision.REVIEW])
        banner = f"{g.decision.value}"
        if g.anomaly_score is not None:
            banner += f" | anomaly {g.anomaly_score:.2f}"
        fs = max(0.45, out.shape[1] / 900.0)
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.rectangle(out, (0, 0), (tw + 12, th + 12), colour, -1)
        cv2.putText(out, banner, (6, th + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
        return out
