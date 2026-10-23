"""Scene gate + frame preparation tests (no network, no model downloads)."""

import cv2
import numpy as np
import pytest

from steelsight.frames import prepare_frame
from steelsight.scene_gate import SceneGate, describe_prompt


def steel_like(kind="scratch", size=300, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.normal(110, 14, (size, size))
    if kind == "scratch":
        for y in rng.integers(10, size - 10, 10):
            t[y:y + 2, 20:size - 20] += 80
    elif kind == "streak":
        for x in rng.integers(10, size - 10, 25):
            t[:, x:x + 1] += 40
    elif kind == "blotch":
        t = cv2.GaussianBlur(t, (0, 0), 9) + rng.normal(0, 6, (size, size))
    g = t.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


class StubClip:
    def __init__(self, p, other="a desk or table"):
        self.p, self.other = p, other

    def score(self, bgr):
        return self.p, self.other


def test_blank_dark_and_washed_out_rejected():
    g = SceneGate(use_face=False)
    for im, word in [(np.full((200, 200, 3), 128, np.uint8), "Blank"),
                     (np.random.default_rng(0).integers(0, 6, (200, 200, 3)).astype(np.uint8), "black"),
                     (np.clip(np.random.default_rng(1).normal(252, 4, (200, 200, 3)), 0, 255).astype(np.uint8), "washed")]:
        v = g.check(im)
        assert not v.ok and v.label == "unusable" and word in v.reason


@pytest.mark.parametrize("kind", ["scratch", "streak", "blotch"])
def test_steel_textures_are_not_faces(kind):
    g = SceneGate(use_face=True)
    if not g.faces.available:
        pytest.skip("face cascades unavailable (offline and not bundled)")
    v = g.check(steel_like(kind))
    assert v.ok and v.label == "steel", v.reason


def test_real_face_is_rejected():
    data = pytest.importorskip("skimage.data")
    g = SceneGate(use_face=True)
    if not g.faces.available:
        pytest.skip("face cascades unavailable (offline and not bundled)")
    face = data.astronaut()[:, :, ::-1].copy()          # public-domain NASA portrait
    v = g.check(face)
    assert not v.ok and v.label == "face" and v.face_boxes
    assert "not inspected or logged" in v.reason


def test_clip_threshold_logic_with_stub():
    g = SceneGate(use_face=False)
    im = steel_like()
    g.clip = StubClip(0.20)
    v = g.check(im)
    assert not v.ok and v.label == "not_steel" and "desk or table" in v.reason and v.clip_steel_prob == 0.20
    g.clip = StubClip(0.93)
    v = g.check(im)
    assert v.ok and v.label == "steel" and v.clip_steel_prob == 0.93


def test_ok_message_lists_only_checks_that_ran():
    im = steel_like()
    g = SceneGate(use_face=False)
    assert g.check(im).reason == "OK - not blank. Other non-steel objects are not checked."
    g.clip = StubClip(0.93)
    assert "looks like metal (93%)" in g.check(im).reason and "not checked" not in g.check(im).reason
    g2 = SceneGate(use_face=True)
    g2.faces.available = False                      # e.g. cascades could not be downloaded offline
    assert "no face" not in g2.check(im).reason


def test_status_is_honest_about_what_is_checked():
    assert "scene (CLIP) check: off" in SceneGate(use_face=False, use_clip=False).status
    g = SceneGate(use_face=False, use_clip=False)
    g.clip_error = "OSError: offline"
    assert "UNAVAILABLE" in g.status


def test_blur_and_tiny_image_are_warnings_not_rejections():
    g = SceneGate(use_face=False)
    blurry = cv2.GaussianBlur(steel_like("scratch"), (0, 0), 6)
    gray = cv2.cvtColor(blurry, cv2.COLOR_BGR2GRAY)
    assert gray.std() > 3.0                                  # texture survives, so only the focus warning can fire
    v = g.check(blurry)
    assert v.ok and any("focus" in w for w in v.warnings)
    tiny = g.check(steel_like(size=64))
    assert tiny.ok and any("small" in w for w in tiny.warnings)


def test_prepare_frame_square_gray_and_resize():
    rect = np.random.default_rng(0).integers(0, 255, (3000, 4000, 3)).astype(np.uint8)
    out = prepare_frame(rect, max_side=640)
    assert out.shape == (640, 640, 3)
    assert np.array_equal(out[:, :, 0], out[:, :, 1]) and np.array_equal(out[:, :, 1], out[:, :, 2])
    already = np.zeros((200, 200, 3), np.uint8)
    assert prepare_frame(already).shape == (200, 200, 3)
    assert prepare_frame(np.zeros((50, 80), np.uint8)).shape == (50, 50, 3)


def test_describe_prompt():
    assert describe_prompt("a photo of a desk or table") == "a desk or table"
    assert describe_prompt("a selfie of a person") == "a selfie of a person"
