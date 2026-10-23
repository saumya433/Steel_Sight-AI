"""Process layer: defect map along the coil and roll diagnosis.

Insight 4 from the deck: "defect patterns are process data". A mark that
repeats along the strip at a fixed pitch is being stamped by one rotating
roll, and the pitch equals that roll's circumference. So:

    roll diameter = pitch / pi

and a roll table (name -> diameter) turns an inspection result into a named
maintenance order.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclasses.dataclass
class CoilEvent:
    position_m: float
    name: str
    conf: float
    decision: str
    frame: int
    xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


class CoilTracker:
    """Converts frame index -> position along the coil and keeps the events.

    Position comes either from an encoder (preferred) or from line speed x time.
    frame_length_m is how much strip one frame covers along the running direction.
    """

    def __init__(self, coil_id: str, line_speed_m_per_min: float = 300.0, fps: float = 20.0,
                 frame_length_m: Optional[float] = None):
        self.coil_id = coil_id
        self.speed = line_speed_m_per_min
        self.fps = fps
        self.frame_length_m = frame_length_m if frame_length_m else (line_speed_m_per_min / 60.0) / fps
        self.events: list[CoilEvent] = []
        self.frames_seen = 0
        self.decisions: dict[str, int] = {}

    def position_for_frame(self, frame: int, encoder_m: Optional[float] = None) -> float:
        return float(encoder_m) if encoder_m is not None else frame * self.frame_length_m

    def add_frame(self, frame: int, decision: str, detections: Iterable, encoder_m: Optional[float] = None,
                  image_h: Optional[int] = None) -> float:
        pos = self.position_for_frame(frame, encoder_m)
        self.frames_seen += 1
        self.decisions[decision] = self.decisions.get(decision, 0) + 1
        for d in detections:
            # refine with the box centre along the running direction (image y axis)
            offset = 0.0
            if image_h:
                cy = (d.xyxy[1] + d.xyxy[3]) / 2.0
                offset = (cy / float(image_h)) * self.frame_length_m
            self.events.append(CoilEvent(pos + offset, d.name, d.conf, decision, frame, tuple(d.xyxy)))
        return pos

    @property
    def length_m(self) -> float:
        return self.frames_seen * self.frame_length_m

    def summary(self) -> dict:
        per_class: dict[str, int] = {}
        for e in self.events:
            per_class[e.name] = per_class.get(e.name, 0) + 1
        n = self.frames_seen or 1
        flagged = self.decisions.get("AUTO_FLAG", 0) + self.decisions.get("REVIEW", 0) + self.decisions.get("UNKNOWN", 0)
        return {
            "coil_id": self.coil_id,
            "length_m": round(self.length_m, 2),
            "frames": self.frames_seen,
            "decisions": self.decisions,
            "flagged_fraction": round(flagged / n, 4),
            "defects_per_100m": round(100.0 * len(self.events) / max(self.length_m, 1e-6), 2),
            "per_class": per_class,
            "grade_hint": grade_hint(flagged / n, per_class),
        }


def grade_hint(flagged_fraction: float, per_class: dict[str, int]) -> str:
    """Indicative coil grade from the flag density. Thresholds are placeholders
    to be calibrated against QA in Phase 2 - the point is that grading becomes
    a consistent, recorded rule instead of a subjective call."""
    if flagged_fraction < 0.02:
        return "PRIME"
    if flagged_fraction < 0.10:
        return "SECOND"
    return "DOWNGRADE"


@dataclasses.dataclass
class RollDiagnosis:
    pitch_m: Optional[float]
    diameter_mm: Optional[float]
    support: int                      # number of event pairs that agree with the pitch
    confidence: float                 # 0..1, how dominant the pitch peak is
    candidate_roll: Optional[str]
    candidate_error_pct: Optional[float]
    message: str


def roll_diagnosis(
    positions_m: Sequence[float],
    roll_table_mm: Optional[dict[str, float]] = None,
    min_pitch_m: float = 0.3,
    max_pitch_m: float = 6.0,
    bin_m: float = 0.02,
    min_support: int = 4,
    tolerance_pct: float = 5.0,
) -> RollDiagnosis:
    """Find a repeating pitch in defect positions and map it to a roll.

    Uses the histogram of pairwise position differences: a periodic defect
    produces a sharp peak at the pitch (and its multiples). Robust to missed
    detections and to unrelated defects mixed in.
    """
    pos = np.sort(np.asarray(positions_m, dtype=np.float64))
    if len(pos) < min_support + 1:
        return RollDiagnosis(None, None, 0, 0.0, None, None, f"not enough events ({len(pos)}) for a periodicity test")

    diffs = pos[None, :] - pos[:, None]
    diffs = diffs[np.triu_indices(len(pos), k=1)]
    diffs = diffs[(diffs >= min_pitch_m) & (diffs <= max_pitch_m)]
    if len(diffs) < min_support:
        return RollDiagnosis(None, None, 0, 0.0, None, None, "no event pairs inside the plausible pitch range")

    edges = np.arange(min_pitch_m, max_pitch_m + bin_m, bin_m)
    hist, _ = np.histogram(diffs, bins=edges)
    # Smooth over 3 bins so a pitch that straddles a bin edge is not split.
    k = np.array([1.0, 1.0, 1.0])
    sm = np.convolve(hist, k, mode="same")
    peak = int(np.argmax(sm))
    support = int(sm[peak])
    # Prominence: a real pitch is a sharp peak above the histogram's typical level.
    # Dense, unrelated defects give a flat histogram (prominence ~ 0) however high the counts.
    baseline = float(np.median(sm))
    prominence = float((support - baseline) / (support + 1.0))
    if support < min_support or prominence < 0.5:
        return RollDiagnosis(None, None, support, round(max(prominence, 0.0), 3), None, None,
                             f"no repeating pitch (support {support}, prominence {max(prominence, 0.0):.2f})")

    # Refine: mean of the diffs inside the winning window.
    lo, hi = edges[max(0, peak - 1)], edges[min(len(edges) - 1, peak + 2)]
    in_win = diffs[(diffs >= lo) & (diffs < hi)]
    pitch = float(in_win.mean())
    # Fundamental check: if pitch/2 also has strong support, the true pitch is pitch/2.
    half = pitch / 2.0
    if half >= min_pitch_m:
        half_support = int(((diffs >= half - 1.5 * bin_m) & (diffs <= half + 1.5 * bin_m)).sum())
        if half_support >= 0.8 * support:
            pitch, support = half, half_support
    confidence = float(min(1.0, prominence))
    diameter_mm = pitch / math.pi * 1000.0

    cand, err = None, None
    if roll_table_mm:
        name, d = min(roll_table_mm.items(), key=lambda kv: abs(kv[1] - diameter_mm))
        e = abs(d - diameter_mm) / d * 100.0
        if e <= tolerance_pct:
            cand, err = name, round(e, 2)
    msg = f"pitch {pitch:.3f} m -> roll diameter {diameter_mm:.0f} mm (support {support}, confidence {confidence:.2f})"
    if cand:
        msg += f"; matches {cand} within {err}%"
    elif roll_table_mm:
        msg += "; no roll in the table within tolerance"
    return RollDiagnosis(round(pitch, 4), round(diameter_mm, 1), support, round(confidence, 3), cand, err, msg)


def diagnose_by_class(
    events: Sequence[CoilEvent],
    roll_table_mm: Optional[dict[str, float]] = None,
    decisions: tuple[str, ...] = ("AUTO_FLAG",),
    extra_positions: Optional[Sequence[float]] = None,
    **kw,
) -> dict[str, RollDiagnosis]:
    """Run the periodicity test separately per defect class, on confident events only.

    A roll stamps the SAME kind of mark each revolution, so mixing classes (or
    low-confidence noise) only hides the pitch. Returns {class: diagnosis};
    ``extra_positions`` (e.g. a synthetic demo pitch) are tested as their own
    "synthetic" entry.
    """
    min_conf = kw.pop("min_conf", 0.5)
    out: dict[str, RollDiagnosis] = {}
    # One event per class per frame (the most confident box): a roll stamps one mark per
    # revolution, while a noisy model can draw dozens of boxes in one frame.
    best_per_frame: dict[tuple[str, int], CoilEvent] = {}
    for e in events:
        if e.decision not in decisions or e.conf < min_conf:
            continue
        key = (e.name, e.frame)
        if key not in best_per_frame or e.conf > best_per_frame[key].conf:
            best_per_frame[key] = e
    by_cls: dict[str, list[float]] = {}
    for (name, _), e in best_per_frame.items():
        by_cls.setdefault(name, []).append(e.position_m)
    n_frames = kw.pop("n_frames", None)
    for name, pos in by_cls.items():
        # A class present in most frames is a continuous / distributed defect: its events
        # are spaced by the camera's frame spacing, which would masquerade as a pitch.
        if n_frames and len(pos) / n_frames > 0.5:
            out[name] = RollDiagnosis(None, None, len(pos), 0.0, None, None,
                                      f"present in {100 * len(pos) / n_frames:.0f}% of frames: continuous defect, not a periodic roll mark")
            continue
        out[name] = roll_diagnosis(pos, roll_table_mm, **kw)
    if extra_positions:
        out["synthetic"] = roll_diagnosis(list(extra_positions), roll_table_mm, **kw)
    return out


def best_diagnosis(diags: dict[str, RollDiagnosis]) -> Optional[tuple[str, RollDiagnosis]]:
    found = [(k, d) for k, d in diags.items() if d.pitch_m is not None]
    if not found:
        return None
    return max(found, key=lambda kd: (kd[1].confidence, kd[1].support))


DEFAULT_ROLL_TABLE_MM = {
    # Placeholder diameters; replace with the line's real roll table.
    "work roll (stand 1)": 450.0,
    "work roll (stand 2)": 480.0,
    "bridle roll": 600.0,
    "deflector roll": 300.0,
    "tension leveller roll": 150.0,
}
