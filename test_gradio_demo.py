"""Gradio demo logic with a stub detector (no weights, no network). Skipped if gradio/PIL are absent."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gradio")
PIL_Image = pytest.importorskip("PIL.Image")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gradio_demo  # noqa: E402

from steelsight.gate import Decision  # noqa: E402
from steelsight.scene_gate import SceneGate  # noqa: E402


class StubPipe:
    def __init__(self):
        self.calls = []

    def inspect(self, frame, draw_heatmap=True):
        self.calls.append(frame.shape)
        det = SimpleNamespace(name="scratches", conf=0.81)
        gate = SimpleNamespace(reason="confident detection: scratches 0.81", kept=[det])
        return SimpleNamespace(decision=Decision.AUTO_FLAG, gate=gate, annotated=np.zeros((64, 64, 3), np.uint8))


def textured(h=300, w=400, seed=0):
    rng = np.random.default_rng(seed)
    return PIL_Image.fromarray(rng.normal(110, 14, (h, w, 3)).clip(0, 255).astype(np.uint8))


def test_rear_camera_is_requested():
    cfg = gradio_demo.camera_input().get_config()
    wo = cfg["webcam_options"]
    assert wo["constraints"]["video"]["facingMode"] == {"ideal": "environment"}
    assert wo["mirror"] is False
    assert set(cfg["sources"]) == {"upload", "webcam"}


def test_steel_image_is_inspected_on_a_square_crop():
    pipe = StubPipe()
    fn = gradio_demo.make_inspector(pipe, SceneGate(use_face=False))
    img, report = fn(textured())
    assert img is not None and "DECISION: AUTO_FLAG" in report and "scratches: 81%" in report
    assert pipe.calls == [(300, 300, 3)]                    # 400x300 photo centre-cropped to 300x300


def test_rejected_image_is_never_echoed_or_inspected():
    pipe = StubPipe()
    gate = SceneGate(use_face=False)
    gate.clip = SimpleNamespace(score=lambda bgr: (0.1, "a desk or table"))
    fn = gradio_demo.make_inspector(pipe, gate)
    img, report = fn(textured())
    assert img is None and report.startswith("NOT INSPECTED - NOT STEEL") and "desk or table" in report
    assert pipe.calls == []


def test_no_image_message():
    img, report = gradio_demo.make_inspector(StubPipe(), SceneGate(use_face=False))(None)
    assert img is None and "Take a photo" in report
