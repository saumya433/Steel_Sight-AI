"""Stage 1 mechanism test on synthetic texture: a planted blob must score higher
than plain texture. Skips when ImageNet weights cannot be downloaded."""

import numpy as np
import pytest

from steelsight.anomaly import AnomalyConfig, AnomalyScreen, auroc, boxes_to_mask


def _texture(rng, size=200):
    base = rng.normal(128, 18, (size, size)).astype(np.float32)
    base = np.clip(base + 12 * np.sin(np.linspace(0, 40, size))[None, :], 0, 255)
    return np.repeat(base[:, :, None], 3, axis=2).astype(np.uint8)


def test_auroc_rank_based():
    assert auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert auroc(np.array([0.1, 0.2]), np.array([0.9, 0.8])) == 0.0
    assert abs(auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) - 0.5) < 1e-9


def test_boxes_to_mask():
    m = boxes_to_mask([(10, 10, 20, 30)], 40, 40)
    assert m.sum() == 10 * 20 and m[15, 15] == 1 and m[0, 0] == 0


def test_planted_defect_scores_higher():
    try:
        screen = AnomalyScreen(AnomalyConfig(device="cpu", coreset_fraction=0.2, max_bank=3000))
    except Exception as e:  # pragma: no cover - fully offline CI (no ImageNet and no YOLO weights)
        pytest.skip(f"no pretrained backbone available: {e}")
    rng = np.random.default_rng(0)
    normal = [_texture(rng) for _ in range(12)]
    screen.fit(normal)
    clean = _texture(rng)
    defect = clean.copy()
    defect[80:130, 70:150] = 30                      # dark patch
    s_clean, _ = screen.score(clean)
    s_defect, heat = screen.score(defect)
    assert s_defect > s_clean
    assert heat.shape == defect.shape[:2]
    assert heat[100, 110] > heat[20, 20]              # hottest where the blob is
    screen.calibrate(screen.patch_scores([_texture(rng) for _ in range(4)]).ravel(), percentile=99.0)
    assert screen.is_anomalous(s_defect)
