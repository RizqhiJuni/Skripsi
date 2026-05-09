"""Inferensi pada gambar (single file / folder).

Contoh:
    python predict.py --source path/to/img.jpg
    python predict.py --source test/images --conf 0.4 --save
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import torch
from ultralytics import YOLO

from utils import (
    draw_detections,
    ensure_dir,
    logger,
    results_to_json,
    validate_file,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict gambar dengan YOLOv12")
    parser.add_argument("--weights", type=str,
                        default="runs/train/rokok_yolov12/weights/best.pt",
                        help="Path ke model .pt")
    parser.add_argument("--source", type=str, required=True,
                        help="File gambar atau folder berisi gambar")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold untuk NMS")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' | 'cpu' | '0' | '0,1' ...")
    parser.add_argument("--save", action="store_true",
                        help="Simpan gambar hasil deteksi")
    parser.add_argument("--output", type=str, default="runs/predict",
                        help="Folder output")
    return parser.parse_args()


def collect_images(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted([p for p in source.rglob("*") if p.suffix.lower() in IMG_EXT])
    raise FileNotFoundError(f"Source tidak valid: {source}")


def _resolve_device(requested: str) -> str:
    if requested.lower() in ("auto", ""):
        return "0" if torch.cuda.is_available() else "cpu"
    if requested.lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        logger.warning(f"CUDA tidak tersedia, fallback device='{requested}' -> 'cpu'")
        return "cpu"
    return requested


def main() -> None:
    args = parse_args()
    args.device = _resolve_device(args.device)

    try:
        weights = validate_file(args.weights)
    except FileNotFoundError as e:
        logger.error(f"{e}. Jalankan train.py dulu atau berikan --weights.")
        raise SystemExit(1)

    source = Path(args.source)
    try:
        images = collect_images(source)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise SystemExit(1)

    if not images:
        logger.warning("Tidak ada gambar pada source.")
        return

    logger.info(f"Loading model: {weights}")
    try:
        model = YOLO(str(weights))
    except Exception as e:
        logger.exception(f"Gagal load model: {e}")
        raise SystemExit(1)

    out_dir = ensure_dir(args.output)
    all_results = []

    for img_path in images:
        logger.info(f"Predict: {img_path}")
        try:
            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Gagal predict {img_path}: {e}")
            continue

        detections = results_to_json(results, conf_threshold=args.conf)
        item = {
            "image": str(img_path),
            "num_detections": len(detections),
            "detections": detections,
        }
        all_results.append(item)
        logger.info(f"  -> {len(detections)} objek terdeteksi")

        if args.save:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError("imread mengembalikan None")
                vis = draw_detections(img, detections)
                out_path = out_dir / img_path.name
                cv2.imwrite(str(out_path), vis)
            except Exception as e:
                logger.warning(f"Gagal menyimpan visualisasi {img_path.name}: {e}")

    json_path = out_dir / "predictions.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    logger.info(f"Hasil JSON disimpan ke: {json_path}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
