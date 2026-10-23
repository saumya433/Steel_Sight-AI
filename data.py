"""Small dataset helpers shared by the scripts (YOLO label I/O, class list)."""

from __future__ import annotations

from pathlib import Path

import yaml

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def load_classes(data_yaml: str | Path = "config/data.yaml") -> list[str]:
    p = Path(data_yaml)
    if not p.exists():
        return list(CLASSES)
    names = yaml.safe_load(p.read_text()).get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def read_yolo_labels(label_path: str | Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    """YOLO txt (cls cx cy bw bh, normalised) -> list of (cls, x1, y1, x2, y2) in pixels."""
    p = Path(label_path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def label_path_for(image_path: str | Path) -> Path:
    """Ultralytics convention: .../images/x.jpg -> .../labels/x.txt"""
    p = Path(image_path)
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def list_images(folder: str | Path, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> list[Path]:
    f = Path(folder)
    return sorted(p for p in f.rglob("*") if p.suffix.lower() in exts)
