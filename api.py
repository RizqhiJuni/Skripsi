"""FastAPI service untuk deteksi rokok via upload gambar.

Jalankan:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoint:
    GET  /                -> info
    GET  /health          -> health check
    POST /predict         -> upload image, return JSON
    POST /predict/visual  -> upload image, return PNG dengan bbox
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from ultralytics import YOLO

from utils import (
    CLASS_NAMES,
    draw_detections,
    logger,
    read_image_bytes,
    results_to_json,
)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH", "runs/train/rokok_yolov12/weights/best.pt"
)
_DEVICE_ENV = os.getenv("DEVICE", "auto").lower()
if _DEVICE_ENV in ("auto", ""):
    DEVICE = "0" if torch.cuda.is_available() else "cpu"
elif _DEVICE_ENV != "cpu" and not torch.cuda.is_available():
    DEVICE = "cpu"
else:
    DEVICE = _DEVICE_ENV
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.25"))
DEFAULT_IOU = float(os.getenv("DEFAULT_IOU", "0.45"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"}

_model: Optional[YOLO] = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise RuntimeError(
                f"Model tidak ditemukan: {MODEL_PATH}. "
                f"Train dulu dengan train.py atau set env MODEL_PATH."
            )
        logger.info(f"Loading model: {MODEL_PATH} (device={DEVICE})")
        _model = YOLO(MODEL_PATH)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort preload (tidak fatal jika model belum ada)
    try:
        get_model()
    except Exception as e:
        logger.warning(f"Preload model gagal: {e}")
    yield


app = FastAPI(
    title="Deteksi Aktivitas Merokok - YOLOv12",
    description="API deteksi 4-class: asap_rokok, membakar_rokok, "
                "pegang_rokok, merokok",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
async def _read_upload(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipe file tidak didukung: {file.content_type}. "
                   f"Gunakan: {sorted(ALLOWED_TYPES)}",
        )
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File terlalu besar ({size_mb:.2f} MB). "
                   f"Maks {MAX_UPLOAD_MB} MB.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="File kosong.")
    return data


def _validate_conf(conf: float) -> float:
    if not 0.0 <= conf <= 1.0:
        raise HTTPException(status_code=400,
                            detail="conf harus antara 0.0 - 1.0")
    return conf


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Deteksi Aktivitas Merokok",
        "model": MODEL_PATH,
        "classes": CLASS_NAMES,
        "endpoints": ["/health", "/predict", "/predict/visual"],
    }


@app.get("/health")
def health():
    loaded = _model is not None
    return {"status": "ok", "model_loaded": loaded, "model_path": MODEL_PATH}


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Gambar JPG/PNG/WEBP"),
    conf: float = Form(DEFAULT_CONF, description="Confidence threshold 0-1"),
    iou: float = Form(DEFAULT_IOU, description="IoU threshold 0-1"),
):
    """Deteksi rokok pada gambar. Return JSON."""
    _validate_conf(conf)
    _validate_conf(iou)

    try:
        model = get_model()
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=503, detail=str(e))

    raw = await _read_upload(file)

    try:
        img = read_image_bytes(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        results = model.predict(
            source=img,
            conf=conf,
            iou=iou,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False,
        )
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500,
                            detail=f"Inference gagal: {e}")

    detections = results_to_json(results, conf_threshold=conf)
    h, w = img.shape[:2]

    return JSONResponse({
        "filename": file.filename,
        "image_size": {"width": w, "height": h},
        "params": {"conf": conf, "iou": iou, "imgsz": IMG_SIZE},
        "num_detections": len(detections),
        "detections": detections,
    })


@app.post("/predict/visual")
async def predict_visual(
    file: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    iou: float = Form(DEFAULT_IOU),
):
    """Sama seperti /predict tapi mengembalikan image PNG dengan bbox."""
    _validate_conf(conf)
    _validate_conf(iou)

    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw = await _read_upload(file)
    try:
        img = read_image_bytes(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        results = model.predict(
            source=img, conf=conf, iou=iou,
            imgsz=IMG_SIZE, device=DEVICE, verbose=False,
        )
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference gagal: {e}")

    detections = results_to_json(results, conf_threshold=conf)
    vis = draw_detections(img, detections)

    ok, buf = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="Gagal encode PNG")
    return Response(content=buf.tobytes(), media_type="image/png",
                    headers={"X-Num-Detections": str(len(detections))})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
