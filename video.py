"""Deteksi realtime dari video file / webcam / IP Webcam HP + alarm WA (WAHA).

Contoh:
    # Webcam laptop
    python video.py --source 0

    # File video
    python video.py --source video.mp4 --save

    # IP Webcam dari HP Android (app: "IP Webcam")
    python video.py --source http://192.168.1.10:8080/video

    # Pakai VIDEO_SOURCE dari .env (tidak perlu --source)
    python video.py

Alarm WhatsApp dikirim otomatis ke WA_RECIPIENT (lihat .env.example).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from alarm import WhatsAppAlarm
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
                        default=os.getenv("MODEL_PATH",
                                          "runs/train/rokok_yolov12/weights/best.pt"))
    parser.add_argument("--source", type=str,
                        default=os.getenv("VIDEO_SOURCE", "0"),
                        help="Path video, index webcam (mis: 0), atau URL "
                             "IP Webcam/RTSP (http://ip:8080/video)")
    parser.add_argument("--conf", type=float,
                        default=float(os.getenv("DEFAULT_CONF", "0.25")))
    parser.add_argument("--iou", type=float,
                        default=float(os.getenv("DEFAULT_IOU", "0.45")))
    parser.add_argument("--imgsz", type=int,
                        default=int(os.getenv("IMG_SIZE", "640")))
    parser.add_argument("--device", type=str,
                        default=os.getenv("DEVICE", "auto"),
                        help="'auto' | 'cpu' | '0' | '0,1' ...")
    parser.add_argument("--save", action="store_true",
                        help="Simpan hasil ke video mp4")
    parser.add_argument("--output", type=str, default="runs/video")
    parser.add_argument("--no-show", action="store_true",
                        help="Jangan tampilkan window (headless)")
    parser.add_argument("--save-json", action="store_true",
                        help="Simpan deteksi per-frame ke JSON")
    parser.add_argument("--no-alarm", action="store_true",
                        help="Nonaktifkan pengiriman alarm WA")
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
    """Buka source video: webcam index, file lokal, atau URL stream (HTTP/RTSP)."""
    # webcam index?
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_ANY)
    elif src.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        # Stream jaringan (IP Webcam HP, RTSP camera, dll.)
        logger.info(f"Membuka stream jaringan: {src}")
        cap = cv2.VideoCapture(src)
    else:
        validate_file(src)
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(
            f"Gagal membuka source video: {src}\n"
            "Periksa: HP & PC satu jaringan? URL benar? App IP Webcam jalan?"
        )
    # Best-effort: kecilkan buffer agar tidak ngumpulin frame lama (lag).
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


class FrameGrabber:
    """Thread terpisah yang terus baca stream supaya buffer tidak menumpuk.

    Main loop selalu dapat frame TERBARU; frame lama otomatis di-drop.
    Ini krusial untuk MJPEG/RTSP: tanpa ini, kalau predict + draw lebih
    lambat dari fps stream, OpenCV menumpuk frame internal -> video lag.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="FrameGrabber", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                # Stream putus / file selesai. Beri jeda kecil agar tidak
                # busy-loop, lalu tetap coba lagi (untuk stream jaringan).
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self, last_seq: int) -> tuple[bool, "cv2.Mat | None", int]:
        """Ambil frame jika ada yang BARU dibanding last_seq."""
        with self._lock:
            if self._frame is None or self._seq == last_seq:
                return False, None, last_seq
            return True, self._frame.copy(), self._seq

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


class InferenceWorker:
    """Thread terpisah yang menjalankan YOLO predict tanpa memblok display.

    Pipeline: Grabber -> Inference -> Display (3 thread).
    Inference selalu mengambil frame TERBARU dari grabber; frame yang
    tidak sempat diinferensi otomatis di-skip -> tidak ada antrian.
    """

    def __init__(
        self,
        model: YOLO,
        grabber: FrameGrabber,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
    ) -> None:
        self.model = model
        self.grabber = grabber
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

        self._vis = None
        self._dets: list = []
        self._infer_ms: float = 0.0
        self._out_seq: int = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="InferenceWorker", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        last_in_seq = -1
        while not self._stop.is_set():
            ok, frame, seq = self.grabber.read(last_in_seq)
            if not ok:
                time.sleep(0.003)
                continue
            last_in_seq = seq
            t0 = time.time()
            try:
                results = self.model.predict(
                    source=frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
            except Exception as e:
                logger.error(f"Predict error seq={seq}: {e}")
                continue
            dets = results_to_json(results, conf_threshold=self.conf)
            vis = draw_detections(frame, dets)
            dt_ms = (time.time() - t0) * 1000.0
            with self._lock:
                self._vis = vis
                self._dets = dets
                self._infer_ms = dt_ms
                self._out_seq += 1

    def get_latest(
        self, last_consumed: int
    ) -> tuple[bool, "cv2.Mat | None", list, float, int]:
        """Ambil hasil inferensi terbaru jika baru dibanding last_consumed."""
        with self._lock:
            if self._vis is None or self._out_seq == last_consumed:
                return False, None, [], 0.0, last_consumed
            return True, self._vis.copy(), list(self._dets), self._infer_ms, self._out_seq

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


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

    # ----- Alarm WhatsApp (WAHA) -----
    alarm = None
    if not args.no_alarm:
        try:
            alarm = WhatsAppAlarm()
            if alarm.enabled:
                alarm.check_session()
                alarm.start_worker()
        except Exception as e:
            logger.warning(f"Init WhatsAppAlarm gagal: {e}")
            alarm = None
    camera_location = os.getenv("CAMERA_LOCATION", "")

    try:
        cap = open_source(args.source)
    except Exception as e:
        logger.error(str(e))
        raise SystemExit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Source: {args.source} | {w}x{h} @ {fps_in:.1f}fps")

    grabber = FrameGrabber(cap)
    inferer = InferenceWorker(
        model, grabber,
        conf=args.conf, iou=args.iou,
        imgsz=args.imgsz, device=args.device,
    )
    last_seq = -1

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
    last_disp_t = time.time()
    no_frame_t = time.time()

    try:
        while True:
            ok, vis, detections, infer_ms, last_seq = inferer.get_latest(last_seq)
            if not ok:
                # Belum ada hasil inferensi baru -> UI tetap responsif.
                if time.time() - no_frame_t > 10.0:
                    logger.warning("10s tidak ada hasil inferensi baru.")
                    no_frame_t = time.time()
                if not args.no_show:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Dihentikan oleh user.")
                        break
                else:
                    time.sleep(0.003)
                continue
            no_frame_t = time.time()

            # ----- Trigger alarm WA (event-based, async non-blocking) -----
            # process_frame() menggabungkan deteksi berturut-turut menjadi
            # satu "event merokok": 1 pesan saat event mulai + 1 pesan
            # ringkasan saat event selesai. Ini menggantikan kombinasi
            # lama (should_trigger + enqueue) yang spam tiap cooldown.
            if alarm is not None and alarm.enabled:
                extra = f"Lokasi : {camera_location}" if camera_location else ""
                alarm.process_frame(
                    detections, frame=vis, extra_text=extra,
                )
                if detections and logger.isEnabledFor(10):  # DEBUG=10
                    top = max(detections, key=lambda d: d.get("confidence", 0.0))
                    logger.debug(
                        f"det top: {top.get('class_name')} "
                        f"{top.get('confidence', 0):.2%}"
                    )

            # FPS display (loop UI) — ukur jarak antar frame yang ditampilkan
            now = time.time()
            dt = now - last_disp_t
            last_disp_t = now
            fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / dt if dt > 0 else 0.0)
            cv2.putText(
                vis,
                f"FPS:{fps_disp:.1f} Infer:{infer_ms:.0f}ms Det:{len(detections)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(vis)

            if args.save_json:
                json_log.append({
                    "frame": frame_idx,
                    "timestamp": round(now - t0, 3),
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
        inferer.stop()
        grabber.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if alarm is not None:
            alarm.stop()

        if args.save_json and json_log:
            json_path = out_dir / f"detections_{int(t0)}.json"
            json_path.write_text(json.dumps(json_log, indent=2), encoding="utf-8")
            logger.info(f"JSON deteksi disimpan: {json_path}")

    elapsed = time.time() - t0
    logger.info(f"Total {frame_idx} frame dalam {elapsed:.1f}s "
                f"(rata-rata {frame_idx/elapsed if elapsed>0 else 0:.1f} fps)")


if __name__ == "__main__":
    main()
