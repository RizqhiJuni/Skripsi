"""Deteksi realtime dari video file atau webcam.

Contoh:
    python video.py --source 0                  # webcam
    python video.py --source video.mp4 --save
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime video detection YOLOv12")
    parser.add_argument("--weights", type=str,
                        default="runs/train/rokok_yolov12/weights/best.pt")
    parser.add_argument("--source", type=str, default="0",
                        help="Path video atau index webcam (mis: 0)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' | 'cpu' | '0' | '0,1' ...")
    parser.add_argument("--save", action="store_true",
                        help="Simpan hasil ke video mp4")
    parser.add_argument("--output", type=str, default="runs/video")
    parser.add_argument("--no-show", action="store_true",
                        help="Jangan tampilkan window (headless)")
    parser.add_argument("--save-json", action="store_true",
                        help="Simpan deteksi per-frame ke JSON")
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested.lower() in ("auto", ""):
        return "0" if torch.cuda.is_available() else "cpu"
    if requested.lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        logger.warning(f"CUDA tidak tersedia, fallback device='{requested}' -> 'cpu'")
        return "cpu"
    return requested


def open_source(src: str) -> cv2.VideoCapture:
    # webcam index?
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_ANY)
    else:
        validate_file(src)
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka source video: {src}")
    return cap


def main() -> None:
    args = parse_args()
    args.device = _resolve_device(args.device)

    try:
        weights = validate_file(args.weights)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise SystemExit(1)

    logger.info(f"Loading model: {weights}")
    try:
        model = YOLO(str(weights))
    except Exception as e:
        logger.exception(f"Gagal load model: {e}")
        raise SystemExit(1)

    try:
        cap = open_source(args.source)
    except Exception as e:
        logger.error(str(e))
        raise SystemExit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Source: {args.source} | {w}x{h} @ {fps_in:.1f}fps")

    writer = None
    out_dir = ensure_dir(args.output)
    if args.save:
        out_path = out_dir / f"detect_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (w, h))
        logger.info(f"Output video: {out_path}")

    json_log = []
    frame_idx = 0
    t0 = time.time()
    fps_disp = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.info("Stream selesai.")
                break

            t_start = time.time()
            try:
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )
            except Exception as e:
                logger.error(f"Predict error frame {frame_idx}: {e}")
                continue

            detections = results_to_json(results, conf_threshold=args.conf)
            vis = draw_detections(frame, detections)

            # FPS overlay
            dt = time.time() - t_start
            fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / dt if dt > 0 else 0.0)
            cv2.putText(vis, f"FPS: {fps_disp:.1f}  Det: {len(detections)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(vis)

            if args.save_json:
                json_log.append({
                    "frame": frame_idx,
                    "timestamp": round(time.time() - t0, 3),
                    "detections": detections,
                })

            if not args.no_show:
                cv2.imshow("Deteksi Rokok - YOLOv12", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Dihentikan oleh user.")
                    break

            frame_idx += 1
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        if args.save_json and json_log:
            json_path = out_dir / f"detections_{int(t0)}.json"
            json_path.write_text(json.dumps(json_log, indent=2), encoding="utf-8")
            logger.info(f"JSON deteksi disimpan: {json_path}")

    elapsed = time.time() - t0
    logger.info(f"Total {frame_idx} frame dalam {elapsed:.1f}s "
                f"(rata-rata {frame_idx/elapsed if elapsed>0 else 0:.1f} fps)")


if __name__ == "__main__":
    main()
