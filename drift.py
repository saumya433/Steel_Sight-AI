"""Drift monitor: knows when the world changed under the model.

A new grade, a lamp ageing, a dirty lens or a camera knock all shift the
input distribution. The model keeps producing confident-looking output, so
you need a separate signal. We watch two cheap statistics:

    - the Stage 1 anomaly score (sensitive to ANY appearance change)
    - the mean detector confidence of kept boxes

and raise an alert when the recent window drifts from the baseline by more
than ``z`` standard deviations. Baseline = the first ``baseline_n`` frames
after a (re)calibration.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class DriftStatus:
    metric: str
    baseline_mean: float
    baseline_std: float
    recent_mean: float
    z: float
    alert: bool

    @property
    def message(self) -> str:
        state = "DRIFT ALERT" if self.alert else "stable"
        return f"{self.metric}: {state} (recent {self.recent_mean:.3f} vs baseline {self.baseline_mean:.3f} +/- {self.baseline_std:.3f}, z={self.z:.1f})"


class DriftMonitor:
    def __init__(self, baseline_n: int = 200, window: int = 100, z: float = 3.0):
        self.baseline_n = baseline_n
        self.window = window
        self.z = z
        self._baseline: dict[str, list[float]] = collections.defaultdict(list)
        self._recent: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=window))

    def update(self, **metrics: Optional[float]) -> None:
        for k, v in metrics.items():
            if v is None or not np.isfinite(v):
                continue
            if len(self._baseline[k]) < self.baseline_n:
                self._baseline[k].append(float(v))
            else:
                self._recent[k].append(float(v))

    def status(self, metric: str) -> Optional[DriftStatus]:
        b = self._baseline.get(metric, [])
        r = self._recent.get(metric)
        if len(b) < max(10, self.baseline_n // 4) or not r or len(r) < max(10, self.window // 4):
            return None
        bm, bs = float(np.mean(b)), float(np.std(b) + 1e-9)
        rm = float(np.mean(r))
        z = (rm - bm) / (bs / np.sqrt(len(r)))
        return DriftStatus(metric, bm, bs, rm, float(z), abs(z) >= self.z)

    def reset_baseline(self) -> None:
        self._baseline.clear()
        self._recent.clear()
