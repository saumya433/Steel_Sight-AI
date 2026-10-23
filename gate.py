"""The decision layer: turns raw model outputs into an action an operator trusts.

Four outcomes:
    PASS       nothing abnormal at or above the review floor
    REVIEW     detector found something at medium confidence -> operator decides
    AUTO_FLAG  detector is confident -> flag automatically, no human needed
    UNKNOWN    Stage 1 says "abnormal" but Stage 2 cannot name it -> engineer review

Thresholds are PER CLASS because the classes have very different confidence
profiles (on NEU-DET v1, crazing F1 peaks near 0.2 while patches peaks near
0.5). ``scripts/evaluate.py`` writes them to config/thresholds.json from the
validation F1 curves; Phase 2 on a real line recalibrates them against QA
verdicts.
"""

from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path
from typing import Optional, Sequence

from .detector import Detection


class Decision(str, enum.Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    AUTO_FLAG = "AUTO_FLAG"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass
class GateConfig:
    review_min: float = 0.25                                   # default floor: below this a box is ignored
    auto_flag: float = 0.60                                    # default: at/above this a box auto-flags
    per_class_review: dict[str, float] = dataclasses.field(default_factory=dict)
    per_class_auto: dict[str, float] = dataclasses.field(default_factory=dict)
    anomaly_threshold: Optional[float] = None                  # Stage 1 threshold (None = Stage 1 off)
    min_box_frac: float = 0.0                                  # ignore boxes smaller than this fraction of the image

    def review_floor(self, name: str) -> float:
        return float(self.per_class_review.get(name, self.review_min))

    def auto_floor(self, name: str) -> float:
        return float(self.per_class_auto.get(name, self.auto_flag))

    @classmethod
    def load(cls, path: str | Path) -> "GateConfig":
        d = json.loads(Path(path).read_text())
        return cls(
            review_min=d.get("review_min", 0.25),
            auto_flag=d.get("auto_flag", 0.60),
            per_class_review=d.get("per_class_review", {}),
            per_class_auto=d.get("per_class_auto", {}),
            anomaly_threshold=d.get("anomaly_threshold"),
            min_box_frac=d.get("min_box_frac", 0.0),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2))


@dataclasses.dataclass
class GateResult:
    decision: Decision
    kept: list[Detection]            # detections at/above their class review floor
    auto: list[Detection]            # subset at/above their class auto floor
    anomaly_score: Optional[float]
    anomaly_fired: bool
    reason: str

    @property
    def max_conf(self) -> float:
        return max((d.conf for d in self.kept), default=0.0)


def _top(dets: Sequence[Detection], k: int = 3) -> str:
    s = sorted(dets, key=lambda d: -d.conf)
    text = ", ".join(f"{d.name} {d.conf:.2f}" for d in s[:k])
    return text + (f" (+{len(s) - k} more)" if len(s) > k else "")


def decide(
    detections: Sequence[Detection],
    cfg: GateConfig,
    anomaly_score: Optional[float] = None,
    image_hw: Optional[tuple[int, int]] = None,
) -> GateResult:
    """Apply per-class thresholds and the Stage 1 signal; return the decision."""
    kept: list[Detection] = []
    for d in detections:
        if image_hw and cfg.min_box_frac > 0:
            frac = d.area_frac / float(image_hw[0] * image_hw[1])
            if frac < cfg.min_box_frac:
                continue
        if d.conf >= cfg.review_floor(d.name):
            kept.append(d)
    auto = [d for d in kept if d.conf >= cfg.auto_floor(d.name)]
    fired = bool(cfg.anomaly_threshold is not None and anomaly_score is not None and anomaly_score >= cfg.anomaly_threshold)

    if auto:
        names = _top(auto)
        return GateResult(Decision.AUTO_FLAG, kept, auto, anomaly_score, fired, f"confident detection: {names}")
    if kept:
        names = _top(kept)
        suffix = " (anomaly screen also fired)" if fired else ""
        return GateResult(Decision.REVIEW, kept, auto, anomaly_score, fired, f"uncertain detection: {names}{suffix}")
    if fired:
        return GateResult(
            Decision.UNKNOWN, kept, auto, anomaly_score, fired,
            f"surface abnormal (score {anomaly_score:.3f} >= {cfg.anomaly_threshold:.3f}) but no known defect class matched",
        )
    return GateResult(Decision.PASS, kept, auto, anomaly_score, fired, "no defect at or above the review floor")


# Status palette (good / warning / critical / serious). Always shown with a label, never colour alone.
DECISION_COLOURS_HEX = {
    Decision.PASS: "#0ca30c",
    Decision.REVIEW: "#fab219",
    Decision.AUTO_FLAG: "#d03b3b",
    Decision.UNKNOWN: "#ec835a",
}
DECISION_COLOURS_BGR = {
    Decision.PASS: (12, 163, 12),
    Decision.REVIEW: (25, 178, 250),
    Decision.AUTO_FLAG: (59, 59, 208),
    Decision.UNKNOWN: (90, 131, 236),
}
DECISION_ICONS = {
    Decision.PASS: "✅",
    Decision.REVIEW: "🟡",
    Decision.AUTO_FLAG: "🔴",
    Decision.UNKNOWN: "🟠",
}
