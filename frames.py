"""Make a phone / webcam photo look like the frames the detector was trained on.

NEU-DET frames are 200x200 grayscale close-ups. A raw phone photo is a large colour
rectangle: letterboxing shrinks the defect and the colour channels carry information
the model never saw. Centre-crop to a square and convert to 3-channel grayscale.
"""

from __future__ import annotations

import cv2
import numpy as np


def prepare_frame(bgr: np.ndarray, square: bool = True, gray: bool = True, max_side: int | None = 1280) -> np.ndarray:
    im = bgr
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if square:
        h, w = im.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        im = im[y0:y0 + s, x0:x0 + s]
    if max_side and max(im.shape[:2]) > max_side:
        f = max_side / max(im.shape[:2])
        im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    if gray:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(im)
