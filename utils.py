"""Helper functions untuk project deteksi aktivitas merokok."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(name: str = "rokok-detector", level: int = logging.INFO) -> logging.Logger:
    """Buat / ambil logger dengan format konsisten."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logger()

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
CLASS_NAMES: Dict[int, str] = {
    0: "asap rokok",
    1: "membakar rokok",
    2: "merokok",
    3: "pegang rokok",
}

# Warna BGR per kelas (untuk drawing)
CLASS_COLORS: Dict[int, tuple] = {
    0: (200, 200, 200),  # asap rokok - abu-abu
    1: (0, 0, 255),      # membakar rokok - merah
    2: (0, 255, 255),    # merokok - kuning
    3: (0, 165, 255),    # pegang rokok - oranye
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """Pastikan folder ada, return Path-nya."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_file(path: str | Path) -> Path:
    """Validasi file ada. Raise FileNotFoundError jika tidak."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File tidak ditemukan: {p}")
    return p


# ---------------------------------------------------------------------------
# Result formatter
# ---------------------------------------------------------------------------
def results_to_json(results, conf_threshold: float = 0.0) -> List[Dict[str, Any]]:
    """Konversi ultralytics Results -> list JSON-serializable.

    Format setiap deteksi:
        {
            "class_id": int,
            "class_name": str,
            "confidence": float,
            "bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
            "bbox_xywh": {"x": float, "y": float, "w": float, "h": float}
        }
    """
    detections: List[Dict[str, Any]] = []
    if results is None:
        return detections

    # results bisa list (per-image) atau single Results
    iterable = results if isinstance(results, list) else [results]

    for res in iterable:
        if res.boxes is None or len(res.boxes) == 0:
            continue

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        names = res.names if hasattr(res, "names") and res.names else CLASS_NAMES

        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            if float(conf) < conf_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append({
                "class_id": int(cls_id),
                "class_name": names.get(int(cls_id), str(cls_id)),
                "confidence": round(float(conf), 4),
                "bbox": {"x1": round(x1, 2), "y1": round(y1, 2),
                         "x2": round(x2, 2), "y2": round(y2, 2)},
                "bbox_xywh": {
                    "x": round((x1 + x2) / 2, 2),
                    "y": round((y1 + y2) / 2, 2),
                    "w": round(x2 - x1, 2),
                    "h": round(y2 - y1, 2),
                },
            })
    return detections


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Gambar bounding box + label di atas image (BGR)."""
    img = image.copy()
    for det in detections:
        x1, y1 = int(det["bbox"]["x1"]), int(det["bbox"]["y1"])
        x2, y2 = int(det["bbox"]["x2"]), int(det["bbox"]["y2"])
        cls_id = det["class_id"]
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        label = f'{det["class_name"]} {det["confidence"]:.2f}'

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def read_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode bytes -> BGR numpy array."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal decode image. Pastikan file gambar valid.")
    return img
