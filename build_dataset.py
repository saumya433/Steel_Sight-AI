#!/usr/bin/env python
"""NEU-DET (Pascal VOC XML, class subfolders) -> YOLO layout.

    python scripts/build_dataset.py --src NEU-DET --dst YOLO

Creates YOLO/{train,val}/{images,labels} and config/data.yaml. Skips any
image that has no annotation in its own split (the crazing_240 quirk).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steelsight.data import CLASSES  # noqa: E402

CLS_ID = {c: i for i, c in enumerate(CLASSES)}


def voc_to_yolo(xml_path: Path) -> str:
    root = ET.parse(xml_path).getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLS_ID:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        lines.append(f"{CLS_ID[name]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return "\n".join(lines)


def build(src: Path, split_in: str, dst: Path, split_out: str) -> int:
    img_dir, lbl_dir = dst / split_out / "images", dst / split_out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    images_root = src / split_in / "images"
    for cls_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            xml = src / split_in / "annotations" / (img.stem + ".xml")
            if not xml.exists():
                print(f"  skip {img.name}: no annotation in {split_in}")
                continue
            shutil.copy(img, img_dir / img.name)
            (lbl_dir / (img.stem + ".txt")).write_text(voc_to_yolo(xml))
            n += 1
    print(f"{split_out}: {n} image/label pairs")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="NEU-DET")
    ap.add_argument("--dst", default="YOLO")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    build(src, "train", dst, "train")
    build(src, "validation", dst, "val")
    cfg = Path("config/data.yaml")
    cfg.parent.mkdir(exist_ok=True)
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    cfg.write_text(f"path: {dst.resolve()}\ntrain: train/images\nval: val/images\nnames:\n{names}\n")
    print(f"wrote {cfg}")


if __name__ == "__main__":
    main()
