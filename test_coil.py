import math

import numpy as np

from steelsight.coil import CoilTracker, grade_hint, roll_diagnosis
from steelsight.detector import Detection
from steelsight.drift import DriftMonitor


def test_roll_diagnosis_recovers_planted_pitch():
    rng = np.random.default_rng(0)
    pitch = 1.413                                    # m  -> diameter 450 mm
    periodic = [k * pitch + rng.normal(0, 0.01) for k in range(1, 30)]
    noise = list(rng.uniform(0, 45, 25))             # unrelated defects
    diag = roll_diagnosis(periodic + noise, {"work roll": 450.0, "bridle": 600.0})
    assert diag.pitch_m is not None
    assert abs(diag.pitch_m - pitch) < 0.03
    assert abs(diag.diameter_mm - pitch / math.pi * 1000) < 10
    assert diag.candidate_roll == "work roll"


def test_roll_diagnosis_needs_support():
    diag = roll_diagnosis([0.5, 3.1, 7.9], {"a": 300.0})
    assert diag.pitch_m is None


def test_roll_diagnosis_prefers_fundamental_over_harmonic():
    pitch = 0.8
    pos = [k * pitch for k in range(1, 40)]
    diag = roll_diagnosis(pos)
    assert abs(diag.pitch_m - pitch) < 0.03


def test_tracker_positions_and_summary():
    t = CoilTracker("C1", line_speed_m_per_min=300, fps=20)     # 0.25 m per frame
    assert abs(t.frame_length_m - 0.25) < 1e-9
    d = Detection(0, "scratches", 0.7, (0, 100, 50, 100))
    t.add_frame(0, "PASS", [])
    t.add_frame(4, "AUTO_FLAG", [d], image_h=200)
    assert t.events[0].position_m == 4 * 0.25 + 0.5 * 0.25
    s = t.summary()
    assert s["frames"] == 2 and s["decisions"]["AUTO_FLAG"] == 1 and s["per_class"]["scratches"] == 1


def test_grade_hint_bands():
    assert grade_hint(0.0, {}) == "PRIME"
    assert "SECOND" in grade_hint(0.05, {})
    assert "DOWNGRADE" in grade_hint(0.5, {})


def test_drift_monitor_alerts_on_shift():
    m = DriftMonitor(baseline_n=50, window=20, z=3.0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        m.update(anomaly=float(rng.normal(1.0, 0.1)))
    for _ in range(20):
        m.update(anomaly=float(rng.normal(1.6, 0.1)))
    s = m.status("anomaly")
    assert s is not None and s.alert
