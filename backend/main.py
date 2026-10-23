
import os
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.getenv("STEELSIGHT_WEIGHTS", str(ROOT / "best_v1.pt"))
)

app = FastAPI(
    title="SteelSight AI API",
    description="Steel surface defect inspection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = None


@app.get("/")
def home():
    return {
        "project": "SteelSight AI",
        "status": "API is running",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model weights are not loaded. Check STEELSIGHT_WEIGHTS.",
        )

    if file.content_type not in {
        "image/jpeg", "image/png", "image/webp", "image/bmp"
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, WEBP or BMP image.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
        image_array = np.array(image)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="The uploaded file is not a valid image."
        ) from exc

    results = model.predict(image_array, imgsz=640, verbose=False)
    result = results[0]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ]
            detections.append({
                "class_id": class_id,
                "class_name": model.names.get(class_id, str(class_id)),
                "confidence": round(confidence, 4),
                "box": [x1, y1, x2, y2],
            })

    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(
        annotated_rgb, cv2.COLOR_RGB2BGR
    ))
    if not ok:
        raise HTTPException(
            status_code=500, detail="Could not encode the result image."
        )

    import base64

    return {
        "filename": file.filename,
        "image_width": image.width,
        "image_height": image.height,
        "detection_count": len(detections),
        "detections": detections,
        "annotated_image": base64.b64encode(encoded).decode("utf-8"),
        "note": "Predictions are model outputs and require validation for production use.",
    }
