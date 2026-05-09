"""Training script untuk model deteksi aktivitas merokok (YOLOv12).

Contoh penggunaan:
    python train.py
    python train.py --epochs 100 --imgsz 640 --batch 16 --model yolov12n.pt
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import torch
from ultralytics import YOLO

from utils import ensure_dir, logger, validate_file


def resolve_device(requested: str) -> str:
    """Validasi device. Auto-fallback ke CPU bila CUDA tidak tersedia."""
    if requested.lower() in ("auto", ""):
        return "0" if torch.cuda.is_available() else "cpu"
    if requested.lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        logger.warning(
            f"CUDA tidak tersedia (torch={torch.__version__}). "
            f"Mengganti device='{requested}' -> 'cpu'."
        )
        return "cpu"
    return requested


# URL kandidat untuk auto-download weights bila ultralytics gagal
WEIGHT_URLS = {
    "yolov12n.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt",
    "yolov12s.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt",
    "yolov12m.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12m.pt",
    "yolov12l.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12l.pt",
    "yolov12x.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12x.pt",
}

# Fallback bila YOLOv12 tidak tersedia
FALLBACK_CHAIN = ["yolo11n.pt", "yolov8n.pt"]


def ensure_weights(model_name: str) -> str:
    """Pastikan file weights tersedia. Return path ke file lokal.

    Urutan:
      1. Sudah ada path/file lokal -> pakai apa adanya
      2. Coba download dari WEIGHT_URLS (untuk yolov12*)
      3. Fallback ke yolo11n.pt / yolov8n.pt (auto-download oleh ultralytics)
    """
    p = Path(model_name)
    if p.exists():
        return str(p)

    # Coba download manual untuk yolov12
    if model_name in WEIGHT_URLS:
        url = WEIGHT_URLS[model_name]
        try:
            logger.info(f"Mengunduh {model_name} dari {url}")
            urllib.request.urlretrieve(url, model_name)
            if Path(model_name).exists():
                logger.info(f"Berhasil mengunduh {model_name}")
                return model_name
        except Exception as e:
            logger.warning(f"Gagal download {model_name}: {e}")

    # Biarkan ultralytics yang handle (akan auto-download utk yolov8/yolo11)
    # Jika nama tidak dikenali ultralytics, coba fallback chain
    try:
        logger.info(f"Mencoba load {model_name} via ultralytics (auto-download)...")
        YOLO(model_name)
        return model_name
    except Exception as e:
        logger.warning(f"Tidak bisa memuat {model_name}: {e}")

    for fb in FALLBACK_CHAIN:
        try:
            logger.warning(f"Fallback ke {fb}")
            YOLO(fb)
            return fb
        except Exception as e:
            logger.warning(f"Fallback {fb} gagal: {e}")

    raise RuntimeError(
        f"Tidak dapat menyiapkan weights '{model_name}'. "
        f"Download manual dan letakkan di folder project."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv12 untuk deteksi rokok")
    parser.add_argument("--data", type=str, default="data.yaml",
                        help="Path ke data.yaml")
    parser.add_argument("--model", type=str, default="yolov12n.pt",
                        help="Pretrained weights (yolov12n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto' (default), 'cpu', '0', '0,1', dst.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="rokok_yolov12")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)

    try:
        validate_file(args.data)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise SystemExit(1)

    ensure_dir(args.project)

    logger.info("Memulai training YOLOv12")
    logger.info(f"  data    : {args.data}")
    logger.info(f"  model   : {args.model}")
    logger.info(f"  epochs  : {args.epochs}")
    logger.info(f"  imgsz   : {args.imgsz}")
    logger.info(f"  batch   : {args.batch}")
    logger.info(f"  device  : {args.device}")
    logger.info(f"  output  : {args.project}/{args.name}")

    try:
        weights_path = ensure_weights(args.model)
        logger.info(f"Menggunakan weights: {weights_path}")
        model = YOLO(weights_path)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            project=args.project,
            name=args.name,
            resume=args.resume,
            exist_ok=True,
            verbose=True,
        )
    except Exception as e:
        logger.exception(f"Training gagal: {e}")
        raise SystemExit(1)

    save_dir = Path(args.project) / args.name
    best = save_dir / "weights" / "best.pt"
    logger.info("Training selesai.")
    if best.exists():
        logger.info(f"Model terbaik: {best.resolve()}")
    else:
        logger.warning(f"best.pt tidak ditemukan di {save_dir/'weights'}")

    # Validasi akhir
    try:
        logger.info("Menjalankan validasi akhir...")
        metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
        logger.info(f"mAP50    : {metrics.box.map50:.4f}")
        logger.info(f"mAP50-95 : {metrics.box.map:.4f}")
    except Exception as e:
        logger.warning(f"Validasi akhir dilewati: {e}")


if __name__ == "__main__":
    main()
