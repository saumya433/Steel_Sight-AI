"""Gradio camera demo: scene check -> detector -> confidence gate.

Fixes the usual problems with a bare `gr.Image(sources=["webcam"])` demo:

* asks the phone for its REAR camera (Gradio's default asks for no camera in particular,
  and phones then give you the front one)
* refuses faces / blank frames (and, with --clip, any non-steel scene) instead of letting
  the detector draw boxes on them
* runs the detector at the size it was trained at (--imgsz), on a square grayscale crop

    python scripts/gradio_demo.py --weights models/best_v1.pt --imgsz 224 --clip --share
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steelsight.frames import prepare_frame  # noqa: E402
from steelsight.gate import GateConfig  # noqa: E402
from steelsight.pipeline import InspectionPipeline  # noqa: E402
from steelsight.scene_gate import SceneGate  # noqa: E402

REAR_CAMERA = {"video": {"facingMode": {"ideal": "environment"}, "width": {"ideal": 1920}, "height": {"ideal": 1440}}}


def make_inspector(pipe: InspectionPipeline, gate: SceneGate):
    """Returns fn(PIL image | None) -> (annotated PIL | None, report str). Pure function, easy to test."""
    from PIL import Image

    def inspect_steel(pil):
        if pil is None:
            return None, "Take a photo or upload one of a steel surface."
        bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        v = gate.check(bgr)
        if not v.ok:
            # never echo a rejected image back (it may be a face)
            return None, f"NOT INSPECTED - {v.label.upper().replace('_', ' ')}\n{v.reason}"
        res = pipe.inspect(prepare_frame(bgr), draw_heatmap=True)
        lines = [f"DECISION: {res.decision.value}", res.gate.reason, f"({v.reason})"]
        lines += [f"- {d.name}: {d.conf:.0%}" for d in sorted(res.gate.kept, key=lambda d: -d.conf)[:10]]
        lines += [f"! {w}" for w in v.warnings]
        lines.append("Research prototype - a human inspector makes the final call.")
        return Image.fromarray(res.annotated[:, :, ::-1]), "\n".join(lines)

    return inspect_steel


def camera_input():
    """Image input: upload (phone: opens the native camera app) + webcam tab that asks for the rear camera."""
    import gradio as gr

    return gr.Image(
        sources=["upload", "webcam"], type="pil", label="Photo of a steel surface",
        webcam_options=gr.WebcamOptions(mirror=False, constraints=REAR_CAMERA),
    )


def build_demo(weights: str, imgsz: int = 224, use_clip: bool = True, thresholds: str | None = None,
               device: str | None = None):
    import gradio as gr

    th = Path(thresholds) if thresholds else ROOT / "config" / "thresholds.json"
    cfg = GateConfig.load(th) if th.exists() else GateConfig()
    pipe = InspectionPipeline(weights, cfg, imgsz=imgsz, device=device)
    gate = SceneGate(use_face=True, use_clip=use_clip, device=device)
    print("Input check:", gate.status)

    return gr.Interface(
        fn=make_inspector(pipe, gate),
        inputs=camera_input(),
        outputs=[gr.Image(type="pil", label="Result"), gr.Textbox(label="Report", lines=10)],
        title="SteelSight AI - camera inspection",
        description=("Photograph a close-up of the steel surface under even light. On a phone use the upload button -> "
                     "Camera for the rear camera, or the webcam tab (asks for the rear camera). "
                     "Faces and non-steel images are rejected before inspection."),
        flagging_mode="never",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/best_v1.pt")
    ap.add_argument("--imgsz", type=int, default=224, help="size the model was TRAINED at (trained with 200 -> 224)")
    ap.add_argument("--clip", action="store_true", help="also reject non-steel scenes with CLIP (downloads ~600 MB)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--share", action="store_true", help="public gradio.live link (needed for a phone camera from Colab)")
    a = ap.parse_args()
    build_demo(a.weights, a.imgsz, a.clip, device=a.device).launch(share=a.share)


if __name__ == "__main__":
    main()
