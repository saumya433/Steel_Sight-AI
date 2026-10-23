"""Scene gate: is this image a steel surface at all?

The defect detector only knows six steel defect classes. Shown a face, a desk or a
blank frame it has no way to say "this is not steel" - it just draws its best-guess
boxes. This module answers that question BEFORE the detector runs.

Three layers, cheapest first:

1. Usability   blank / covered lens / far too dark or bright.        (numpy, always on)
2. Face        OpenCV Haar cascades (frontal faces).                 (offline, ~10 ms)
3. Scene       CLIP zero-shot "metal surface" vs "face / room / ..." (optional, ~600 MB
               model downloaded on first use; needs `transformers` + `torch`)

Layer 2 only catches faces. Layer 3 is what rejects arbitrary non-steel objects.
If layer 3 cannot load (no internet, no transformers) the gate degrades to layers 1-2
and says so in `SceneGate.status` - it never silently pretends to check more than it does.

A rejected image is never inspected, never echoed back in a result panel and never written to
the audit store. (Upload caches belong to the web framework - e.g. Gradio's temp folder - and are
outside this module's control.)
"""

from __future__ import annotations

import dataclasses
import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

CASCADES = ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/{}"
CACHE_DIR = Path(os.environ.get("STEELSIGHT_CACHE", Path.home() / ".cache" / "steelsight"))

# CLIP prompts. "positive" mass = steel; the best-scoring negative names what it saw instead.
POSITIVE_PROMPTS = [
    "a close-up photo of a metal surface",
    "a close-up photo of a steel sheet",
    "a close-up photo of brushed or polished stainless steel",
    "a close-up photo of a rough metal texture with scratches or defects",
    "a grayscale microscope-style image of a metal surface",
]
NEGATIVE_PROMPTS = [
    "a photo of a person's face",
    "a selfie of a person",
    "a photo of a person",
    "a photo of a hand or fingers",
    "a photo of a room or an office",
    "a photo of a desk or table",
    "a photo of a wall or a ceiling",
    "a photo of a computer screen or a phone",
    "a photo of a document or printed text",
    "a photo of clothes or fabric",
    "a photo of an animal",
    "a photo of food",
    "a photo of the outdoors, sky or trees",
]


def describe_prompt(prompt: str) -> str:
    """'a photo of a desk or table' -> 'a desk or table' (for the rejection message)."""
    return prompt.removeprefix("a photo of ") if prompt.startswith("a photo of ") else prompt


@dataclasses.dataclass
class SceneVerdict:
    ok: bool
    label: str                      # "steel" | "face" | "not_steel" | "unusable"
    reason: str
    face_boxes: list = dataclasses.field(default_factory=list)
    clip_steel_prob: Optional[float] = None
    warnings: list = dataclasses.field(default_factory=list)


class FaceFinder:
    """Frontal-face detector built on OpenCV's bundled Haar cascades."""

    def __init__(self) -> None:
        self.cascades = []
        for name in CASCADES:
            path = self._locate(name)
            if path is None:
                continue
            c = cv2.CascadeClassifier(str(path))
            if not c.empty():
                self.cascades.append(c)
        self.available = bool(self.cascades)

    @staticmethod
    def _locate(name: str) -> Optional[Path]:
        bundled = getattr(getattr(cv2, "data", None), "haarcascades", None)
        for folder in filter(None, [bundled, str(CACHE_DIR)]):
            p = Path(folder) / name
            if p.exists():
                return p
        try:  # some OpenCV builds ship without the XML files: fetch once and cache
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            dest = CACHE_DIR / name
            urllib.request.urlretrieve(CASCADE_URL.format(name), dest)
            return dest
        except Exception as e:  # noqa: BLE001 - offline is a normal state
            log.warning("face cascade %s unavailable (%s)", name, e)
            return None

    def find(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        scale = 640.0 / max(h, w)
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        gray = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_side = max(32, int(0.08 * min(gray.shape[:2])))
        inv = 1.0 / min(scale, 1.0)
        boxes: list[tuple[int, int, int, int]] = []
        for c in self.cascades:
            for (x, y, bw, bh) in c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side)):
                boxes.append((int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)))
        return boxes


class ClipScreen:
    """Zero-shot 'is this a metal surface?' using CLIP. Loaded lazily; may raise on load."""

    def __init__(self, device: Optional[str] = None, model_name: Optional[str] = None) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        name = model_name or os.environ.get("STEELSIGHT_CLIP", "openai/clip-vit-base-patch32")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
        self.n_pos = len(POSITIVE_PROMPTS)
        self._torch = torch

    def score(self, bgr: np.ndarray) -> tuple[float, str]:
        """Returns (probability the image is a metal surface, what it looks like instead)."""
        from PIL import Image

        rgb = cv2.cvtColor(bgr if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        inputs = self.proc(text=self.prompts, images=Image.fromarray(rgb), return_tensors="pt", padding=True).to(self.device)
        with self._torch.no_grad():
            probs = self.model(**inputs).logits_per_image[0].softmax(-1).cpu().numpy()
        steel = float(probs[: self.n_pos].sum())
        other = self.prompts[self.n_pos + int(np.argmax(probs[self.n_pos:]))]
        return steel, describe_prompt(other)


class SceneGate:
    def __init__(self, use_face: bool = True, use_clip: bool = False, clip_min_steel: float = 0.5,
                 device: Optional[str] = None) -> None:
        self.faces = FaceFinder() if use_face else None
        self.clip: Optional[ClipScreen] = None
        self.clip_error: Optional[str] = None
        self.clip_min_steel = clip_min_steel
        if use_clip:
            try:
                self.clip = ClipScreen(device)
            except Exception as e:  # noqa: BLE001 - no internet / no transformers: degrade, loudly
                self.clip_error = f"{type(e).__name__}: {e}"
                log.warning("CLIP scene check unavailable: %s", self.clip_error)

    @property
    def status(self) -> str:
        parts = ["blank/dark check: on"]
        if self.faces is not None:
            parts.append("face check: on" if self.faces.available else "face check: UNAVAILABLE (cascade files missing)")
        else:
            parts.append("face check: off")
        if self.clip is not None:
            parts.append("scene (CLIP) check: on")
        elif self.clip_error:
            parts.append(f"scene (CLIP) check: UNAVAILABLE ({self.clip_error[:80]})")
        else:
            parts.append("scene (CLIP) check: off - other non-steel objects are not rejected")
        return " | ".join(parts)

    def check(self, bgr: np.ndarray) -> SceneVerdict:
        if bgr is None or bgr.size == 0:
            return SceneVerdict(False, "unusable", "No image.")
        gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mean, std = float(gray.mean()), float(gray.std())
        warnings: list[str] = []

        # brightness first: a black frame is also low-contrast, but "add light" is the useful message
        if mean < 12:
            return SceneVerdict(False, "unusable", f"Frame is almost black (mean brightness {mean:.0f}/255). Add light.")
        if mean > 248:
            return SceneVerdict(False, "unusable", f"Frame is washed out (mean brightness {mean:.0f}/255). Reduce glare.")
        if std < 3.0:
            return SceneVerdict(False, "unusable", "Blank frame - lens covered, or a flat surface with no visible texture.")

        if self.faces is not None and self.faces.available:
            boxes = self.faces.find(bgr)
            if boxes:
                return SceneVerdict(False, "face", "A human face was detected - this is not a steel surface. "
                                                   "The image was not inspected or logged.", face_boxes=boxes)

        steel_prob = None
        if self.clip is not None:
            steel_prob, other = self.clip.score(bgr)
            if steel_prob < self.clip_min_steel:
                return SceneVerdict(False, "not_steel", f"This looks like {other}, not a steel surface "
                                                        f"(steel match {steel_prob:.0%}, needs {self.clip_min_steel:.0%}).",
                                    clip_steel_prob=steel_prob)

        small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        if cv2.Laplacian(small, cv2.CV_64F).var() < 6.0:
            warnings.append("Image looks out of focus - defects may be missed.")
        if min(gray.shape[:2]) < 100:
            warnings.append("Image is very small (under 100 px) - move the camera closer or use a higher resolution.")

        ran = ["not blank"]                      # say only what was actually checked
        if self.faces is not None and self.faces.available:
            ran.append("no face")
        if steel_prob is not None:
            ran.append(f"looks like metal ({steel_prob:.0%})")
        note = "" if steel_prob is not None else " Other non-steel objects are not checked."
        return SceneVerdict(True, "steel", f"OK - {', '.join(ran)}.{note}", clip_steel_prob=steel_prob, warnings=warnings)
