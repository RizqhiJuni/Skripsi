"""Modul alarm WhatsApp via WAHA (WhatsApp HTTP API).

Cara kerja:
    1. Jalankan WAHA (contoh Docker):
           docker run -it --rm -p 3000:3000 devlikeapro/waha
    2. Buka http://localhost:3000 -> Start session "default" -> SCAN QR
       dengan WA di HP (nomor ini akan menjadi PENGIRIM alarm).
    3. Isi `.env`:
           WAHA_URL=http://localhost:3000
           WAHA_SESSION=default
           WA_RECIPIENT=628123456789       # nomor PENERIMA alarm
           ALERT_COOLDOWN_SEC=30
           ALERT_CONF_THRESHOLD=0.5
           ALERT_CLASSES=merokok,pegang rokok
    4. Jalankan video.py -- alarm akan otomatis terkirim saat deteksi positif.

Catatan:
    * Nomor penerima TIDAK perlu diawali '+'. Format: kode negara + nomor.
      Contoh +62 812-3456-789  ->  628123456789
    * `chatId` WAHA: <nomor>@c.us (personal) atau <id>@g.us (grup).
      Modul ini otomatis menambahkan suffix '@c.us' bila belum ada.
"""
from __future__ import annotations

import os
import json
import queue
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import requests

from utils import logger

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv opsional -- tetap jalan kalau env diset manual
    pass


def _normalize_chat_id(number: str) -> str:
    """Normalisasi nomor WA -> chatId WAHA.

    Contoh:
        "628123456789"          -> "628123456789@c.us"
        "+62 812-3456-789"      -> "628123456789@c.us"
        "628123456789@c.us"     -> "628123456789@c.us"
        "12345-67890@g.us"      -> "12345-67890@g.us" (grup, biarkan)
    """
    n = number.strip()
    if "@" in n:
        return n
    digits = "".join(ch for ch in n if ch.isdigit())
    if not digits:
        raise ValueError(f"Nomor WA tidak valid: '{number}'")
    return f"{digits}@c.us"


def _parse_recipients(raw: str | None) -> List[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [_normalize_chat_id(p) for p in parts if p]


class WhatsAppAlarm:
    """Client WAHA sederhana dengan cooldown anti-spam."""

    def __init__(
        self,
        waha_url: Optional[str] = None,
        session: Optional[str] = None,
        recipients: Optional[Iterable[str]] = None,
        cooldown_sec: Optional[float] = None,
        conf_threshold: Optional[float] = None,
        alert_classes: Optional[Iterable[str]] = None,
        timeout: float = 10.0,
        # --- konfigurasi 2-tier + temporal ---
        suspect_conf: Optional[float] = None,
        confirm_conf: Optional[float] = None,
        min_hits: Optional[int] = None,
        window_sec: Optional[float] = None,
        suspect_cooldown_sec: Optional[float] = None,
    ) -> None:
        self.url = (waha_url or os.getenv("WAHA_URL", "http://localhost:3000")).rstrip("/")
        self.session = session or os.getenv("WAHA_SESSION", "default")

        if recipients is None:
            recipients = _parse_recipients(os.getenv("WA_RECIPIENT"))
        else:
            recipients = [_normalize_chat_id(r) for r in recipients]
        self.recipients: List[str] = list(recipients)

        # Cooldown alarm "confirm" (jeda antar alarm penuh)
        self.cooldown = float(
            cooldown_sec if cooldown_sec is not None
            else os.getenv("ALERT_COOLDOWN_SEC", "10")
        )
        # Cooldown alarm "suspect" (biasanya lebih panjang -> tidak spam)
        self.suspect_cooldown = float(
            suspect_cooldown_sec if suspect_cooldown_sec is not None
            else os.getenv("ALERT_SUSPECT_COOLDOWN_SEC", "30")
        )
        # Apakah tier "suspect" boleh mengirim WA? Default: false
        # (hanya tier "confirm" yang akan mengirim pesan).
        self.suspect_enabled = (
            os.getenv("ALERT_SUSPECT_ENABLED", "false").strip().lower()
            in ("1", "true", "yes", "y", "on")
        )

        # Ambang 2 tingkat. Backward-compat: jika user hanya set
        # ALERT_CONF_THRESHOLD lama, pakai sbg confirm threshold.
        legacy_conf = conf_threshold if conf_threshold is not None else \
            os.getenv("ALERT_CONF_THRESHOLD")
        self.confirm_conf = float(
            confirm_conf if confirm_conf is not None
            else os.getenv("ALERT_CONF_CONFIRM", legacy_conf or "0.70")
        )
        self.suspect_conf = float(
            suspect_conf if suspect_conf is not None
            else os.getenv("ALERT_CONF_SUSPECT", "0.45")
        )
        if self.suspect_conf > self.confirm_conf:
            logger.warning(
                f"ALERT_CONF_SUSPECT ({self.suspect_conf}) > "
                f"ALERT_CONF_CONFIRM ({self.confirm_conf}); swap nilai."
            )
            self.suspect_conf, self.confirm_conf = self.confirm_conf, self.suspect_conf

        # Temporal confirmation: butuh >= min_hits dalam window_sec
        self.min_hits = int(
            min_hits if min_hits is not None
            else os.getenv("ALERT_MIN_HITS", "2")
        )
        # Min hits KHUSUS utk class confirm (mis. "merokok"). Default 1:
        # event menghisap rokok itu cepat (sering hanya 1-2 frame),
        # jadi 1 frame dengan conf >= confirm_conf sudah cukup memicu.
        # False positive di-redam oleh confirm_conf yg tinggi (0.70).
        self.confirm_min_hits = int(
            os.getenv("ALERT_CONFIRM_MIN_HITS", "1")
        )
        self.window_sec = float(
            window_sec if window_sec is not None
            else os.getenv("ALERT_WINDOW_SEC", "2.0")
        )
        # Resolusi max & kualitas JPEG snapshot (semakin kecil -> upload
        # WAHA semakin cepat). Default 1280px width @ JPEG 75 ~ sweet spot.
        self.snapshot_max_width = int(os.getenv("ALERT_SNAPSHOT_MAX_W", "1280"))
        self.snapshot_jpeg_quality = int(os.getenv("ALERT_JPEG_QUALITY", "75"))

        if alert_classes is None:
            raw = os.getenv(
                "ALERT_CLASSES",
                "terdeteksi merokok,terdeteksi pegang rokok,"
                "merokok,pegang rokok",
            )
            alert_classes = [c.strip().lower() for c in raw.split(",") if c.strip()]
        else:
            alert_classes = [c.strip().lower() for c in alert_classes]
        self.alert_classes = set(alert_classes)

        # Kelas yang BOLEH naik ke level "confirm". Kelas di alert_classes
        # tapi TIDAK ada di sini akan di-cap maksimal "suspect"
        # (mis. "pegang rokok" -> belum tentu menghisap).
        raw_confirm = os.getenv(
            "ALERT_CONFIRM_CLASSES",
            "terdeteksi merokok,merokok",
        )
        self.confirm_classes = {
            c.strip().lower() for c in raw_confirm.split(",") if c.strip()
        }

        self.timeout = timeout
        # Riwayat (timestamp, detection_dict) utk temporal smoothing
        self._history: Deque[Tuple[float, dict]] = deque()
        # Cooldown per-level
        self._last_sent_confirm: float = 0.0
        self._last_sent_suspect: float = 0.0
        # Kompat property lama
        self.conf_threshold = self.confirm_conf
        # Async worker (lazy start)
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._cooldown_lock = threading.Lock()

        # ---------------- Event-based debouncing ----------------
        # Daripada men-trigger setiap kali cooldown habis (spam!),
        # rangkai deteksi berturut-turut sebagai SATU event:
        #   - Pesan "start" terkirim saat event dimulai.
        #   - Pesan "summary" terkirim saat event berakhir (gap >
        #     EVENT_GAP_SEC tanpa deteksi, atau durasi melewati
        #     EVENT_MAX_DUR_SEC).
        # Hasil: 1 sesi merokok = 2 pesan, bukan belasan.
        self.event_gap_sec = float(os.getenv("ALERT_EVENT_GAP_SEC", "15"))
        self.event_max_dur_sec = float(
            os.getenv("ALERT_EVENT_MAX_DUR_SEC", "120")
        )
        self.event_inter_cooldown = float(
            os.getenv("ALERT_EVENT_INTER_COOLDOWN_SEC", "30")
        )
        self.event_send_summary = (
            os.getenv("ALERT_EVENT_SEND_SUMMARY", "true").strip().lower()
            in ("1", "true", "yes", "y", "on")
        )
        # Persistensi event ke JSONL (untuk dashboard / evaluasi skripsi).
        # Format: 1 baris JSON per event yang SELESAI -- append only.
        self.event_log_path = Path(
            os.getenv("EVENT_LOG_PATH", "runs/events.jsonl")
        )
        self.event_snapshot_dir = Path(
            os.getenv("EVENT_SNAPSHOT_DIR", "runs/alarm")
        )
        self._jsonl_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._event_active: bool = False
        self._event_id: str = ""
        self._event_level: str = "suspect"
        self._event_start_ts: float = 0.0
        self._event_last_det_ts: float = 0.0
        self._event_frame_count: int = 0
        self._event_peak_det: Optional[dict] = None
        self._event_peak_frame: Optional[np.ndarray] = None
        self._event_first_det: Optional[dict] = None
        self._event_class_counts: dict = {}
        self._event_end_ts: float = 0.0
        self._event_camera_location: str = ""

        if not self.recipients:
            logger.warning(
                "WhatsAppAlarm: tidak ada penerima (WA_RECIPIENT kosong). "
                "Alarm WA dinonaktifkan."
            )
        else:
            logger.info(
                f"WhatsAppAlarm aktif | WAHA={self.url} session={self.session} "
                f"recipients={self.recipients} "
                f"suspect>={self.suspect_conf} confirm>={self.confirm_conf} "
                f"min_hits={self.min_hits}/{self.window_sec}s "
                f"confirm_min_hits={self.confirm_min_hits} "
                f"cooldown(confirm/suspect)={self.cooldown}/{self.suspect_cooldown}s "
                f"classes={sorted(self.alert_classes)} "
                f"confirm_classes={sorted(self.confirm_classes)}"
            )

    # ------------------------------------------------------------------ utils
    @property
    def enabled(self) -> bool:
        return bool(self.recipients)

    def _on_cooldown(self, level: str = "confirm") -> bool:
        now = time.time()
        if level == "confirm":
            return (now - self._last_sent_confirm) < self.cooldown
        return (now - self._last_sent_suspect) < self.suspect_cooldown

    def _prune_history(self, now: float) -> None:
        while self._history and (now - self._history[0][0]) > self.window_sec:
            self._history.popleft()

    def _evaluate_level(
        self, detections: list[dict], now: float
    ) -> Tuple[Optional[str], Optional[dict]]:
        """Versi internal should_trigger TANPA pengecekan cooldown.

        Dipakai oleh event-based debouncer (``process_frame``) yang punya
        mekanisme anti-spam sendiri (per-event, bukan per-detection).
        """
        if not self.enabled:
            return None, None
        # Kumpulkan SEMUA bbox sah di frame ini (bukan hanya max).
        # Ini krusial: kalau frame punya "pegang_rokok 0.85" + "merokok 0.72",
        # keduanya HARUS masuk window -> supaya "merokok" tdk hilang.
        for det in detections:
            conf = float(det.get("confidence", 0.0))
            if conf < self.suspect_conf:
                continue
            name = str(det.get("class_name", "")).lower()
            if self.alert_classes and name not in self.alert_classes:
                # Peringatkan SEKALI per nama kelas asing supaya bug
                # mismatch ALERT_CLASSES vs data.yaml gampang ketahuan.
                if not hasattr(self, "_warned_unknown"):
                    self._warned_unknown = set()
                if name and name not in self._warned_unknown:
                    self._warned_unknown.add(name)
                    logger.warning(
                        f"Deteksi class '{name}' (conf={conf:.2%}) diabaikan: "
                        f"tidak ada di ALERT_CLASSES={sorted(self.alert_classes)}. "
                        "Periksa nama kelas di data.yaml vs .env."
                    )
                continue
            self._history.append((now, det))
        self._prune_history(now)

        if not self._history:
            return None, None

        # --- (1) Prioritas: cari kandidat CONFIRM di window ---
        confirm_hits = [
            d for (_, d) in self._history
            if float(d.get("confidence", 0.0)) >= self.confirm_conf
            and (
                not self.confirm_classes
                or str(d.get("class_name", "")).lower() in self.confirm_classes
            )
        ]
        if len(confirm_hits) >= self.confirm_min_hits:
            best = max(confirm_hits, key=lambda d: float(d["confidence"]))
            return "confirm", best

        # --- (2) Fallback: tier SUSPECT (butuh >= min_hits) ---
        if len(self._history) < self.min_hits:
            return None, None
        best_in_window = max(
            self._history, key=lambda x: float(x[1]["confidence"])
        )[1]
        return "suspect", best_in_window

    def should_trigger(
        self, detections: list[dict]
    ) -> Tuple[Optional[str], Optional[dict]]:
        """Versi lama (per-detection + cooldown). Dipertahankan utk kompat.

        Untuk pemakaian baru, gunakan :meth:`process_frame` yang memakai
        event-based debouncing (jauh lebih sedikit notifikasi).
        """
        now = time.time()
        level, det = self._evaluate_level(detections, now)
        if level is None or det is None:
            return None, None
        if self._on_cooldown(level):
            return None, None
        return level, det

    # ---------------------------------------------------------------- WAHA IO
    def check_session(self) -> bool:
        """Verifikasi session WAHA dalam status WORKING."""
        try:
            r = requests.get(
                f"{self.url}/api/sessions/{self.session}",
                timeout=self.timeout,
            )
            r.raise_for_status()
            status = r.json().get("status", "UNKNOWN")
            logger.info(f"WAHA session '{self.session}' status: {status}")
            return status == "WORKING"
        except Exception as e:
            logger.warning(f"Tidak bisa cek session WAHA: {e}")
            return False

    def _post(self, endpoint: str, payload: dict) -> bool:
        url = f"{self.url}{endpoint}"
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            if r.status_code >= 400:
                logger.error(f"WAHA {endpoint} HTTP {r.status_code}: {r.text[:300]}")
                return False
            return True
        except Exception as e:
            logger.error(f"WAHA {endpoint} error: {e}")
            return False

    def send_text(self, text: str) -> bool:
        ok_any = False
        for chat in self.recipients:
            payload = {"chatId": chat, "text": text, "session": self.session}
            if self._post("/api/sendText", payload):
                ok_any = True
        return ok_any

    def send_image_bytes(
        self, jpg_bytes: bytes, filename: str, caption: str = ""
    ) -> bool:
        """Kirim gambar dari bytes JPEG (tanpa baca disk). Lebih cepat."""
        import base64
        data = base64.b64encode(jpg_bytes).decode("ascii")
        ok_any = False
        for chat in self.recipients:
            payload = {
                "chatId": chat,
                "session": self.session,
                "file": {
                    "mimetype": "image/jpeg",
                    "filename": filename,
                    "data": data,
                },
                "caption": caption,
            }
            if self._post("/api/sendImage", payload):
                ok_any = True
        return ok_any

    def send_image(self, image_path: str | Path, caption: str = "") -> bool:
        """Kirim gambar bukti deteksi. WAHA Core menerima file via URL/base64.

        Untuk kesederhanaan kami pakai base64 (tidak perlu hosting).
        """
        p = Path(image_path)
        if not p.exists():
            logger.warning(f"Image bukti tidak ditemukan: {p}")
            return self.send_text(caption) if caption else False
        return self.send_image_bytes(p.read_bytes(), p.name, caption=caption)

    # ============================ EVENT-BASED API ============================
    # Strategi anti-spam: gabungkan deteksi berturut-turut menjadi satu
    # "event merokok" -> 1 pesan saat mulai + 1 pesan ringkasan saat
    # selesai. Drastis mengurangi notifikasi (mis. 1 sesi 5 menit dari
    # belasan pesan -> 2 pesan).

    def process_frame(
        self,
        detections: list[dict],
        frame: Optional[np.ndarray] = None,
        extra_text: str = "",
        snapshot_dir: str | Path = "runs/alarm",
    ) -> None:
        """Update mesin state event dan kirim notifikasi bila perlu.

        Panggil method ini SETIAP frame dari pipeline video (gantikan
        kombinasi ``should_trigger`` + ``enqueue`` yang lama).
        """
        if not self.enabled:
            return
        now = time.time()
        level, det = self._evaluate_level(detections, now)

        with self._event_lock:
            if self._event_active:
                if det is not None:
                    self._event_last_det_ts = now
                    self._event_frame_count += 1
                    cls = str(det.get("class_name", "?"))
                    self._event_class_counts[cls] = (
                        self._event_class_counts.get(cls, 0) + 1
                    )
                    cur_conf = float(det.get("confidence", 0.0))
                    peak_conf = (
                        float(self._event_peak_det.get("confidence", 0.0))
                        if self._event_peak_det else 0.0
                    )
                    if cur_conf > peak_conf:
                        self._event_peak_det = det
                        if frame is not None:
                            self._event_peak_frame = frame.copy()
                    if level == "confirm" and self._event_level != "confirm":
                        self._event_level = "confirm"
                        logger.info(
                            "Event di-upgrade: suspect -> confirm "
                            f"(conf={cur_conf:.2%})"
                        )
                # Cek kondisi penutupan event
                gap = now - self._event_last_det_ts
                dur = now - self._event_start_ts
                if gap > self.event_gap_sec or dur > self.event_max_dur_sec:
                    self._close_event_locked(now, extra_text, snapshot_dir)
            else:
                if level is None or det is None:
                    return
                # Cooldown antar-event supaya tidak langsung buka event
                # baru sesaat setelah yg lama tutup.
                if (now - self._event_end_ts) < self.event_inter_cooldown:
                    return
                self._start_event_locked(
                    now, level, det, frame, extra_text, snapshot_dir
                )

    def _start_event_locked(
        self,
        now: float,
        level: str,
        detection: dict,
        frame: Optional[np.ndarray],
        extra_text: str,
        snapshot_dir: str | Path,
    ) -> None:
        self._event_active = True
        self._event_id = uuid.uuid4().hex[:12]
        self._event_level = level
        self._event_start_ts = now
        self._event_last_det_ts = now
        self._event_frame_count = 1
        self._event_peak_det = detection
        self._event_peak_frame = frame.copy() if frame is not None else None
        self._event_first_det = detection
        cls = str(detection.get("class_name", "?"))
        self._event_class_counts = {cls: 1}
        # Simpan camera_location dari extra_text agar konsisten di summary.
        self._event_camera_location = self._extract_camera_location(extra_text)

        if level == "suspect" and not self.suspect_enabled:
            conf = float(detection.get("confidence", 0.0))
            logger.info(
                f"Event suspect dimulai tapi tidak dikirim "
                f"(ALERT_SUSPECT_ENABLED=false, conf={conf:.2%})"
            )
            return

        msg = self._build_event_start_message(detection, level, extra_text)
        self._enqueue_built_message(
            msg, frame, snapshot_dir,
            prefix=("suspect" if level == "suspect" else "alarm") + "_start",
            tag=f"{level}-start",
        )

    def _close_event_locked(
        self,
        now: float,
        extra_text: str,
        snapshot_dir: str | Path,
    ) -> None:
        # Akhiri pada saat deteksi TERAKHIR (bukan saat kita baru sadar).
        duration = max(0.0, self._event_last_det_ts - self._event_start_ts)
        level = self._event_level
        event_id = self._event_id or uuid.uuid4().hex[:12]
        peak_det = self._event_peak_det or self._event_first_det or {}
        frame_to_send = self._event_peak_frame
        frame_count = self._event_frame_count
        class_counts = dict(self._event_class_counts)
        started_at = self._event_start_ts
        ended_at = self._event_last_det_ts
        camera_location = self._event_camera_location

        # Reset state SEBELUM enqueue agar event baru bisa dimulai segera
        # setelah inter-cooldown.
        self._event_active = False
        self._event_id = ""
        self._event_end_ts = now
        self._event_peak_det = None
        self._event_peak_frame = None
        self._event_first_det = None
        self._event_class_counts = {}
        self._event_frame_count = 0
        self._event_camera_location = ""

        # Simpan snapshot peak ke path deterministik utk dashboard.
        snapshot_path = self._save_event_snapshot(event_id, frame_to_send)

        # Persist record event ke JSONL (untuk dashboard / evaluasi).
        wa_will_send = self.event_send_summary and not (
            level == "suspect" and not self.suspect_enabled
        )
        record = {
            "event_id": event_id,
            "started_at": round(started_at, 3),
            "ended_at": round(ended_at, 3),
            "duration_sec": round(duration, 3),
            "level": level,
            "peak_class": peak_det.get("class_name", "?"),
            "peak_confidence": round(float(peak_det.get("confidence", 0.0)), 4),
            "peak_bbox": peak_det.get("box") or peak_det.get("bbox"),
            "frame_count": frame_count,
            "class_breakdown": class_counts,
            "snapshot_path": snapshot_path,
            "camera_location": camera_location,
            "wa_sent": wa_will_send,
        }
        self._persist_event_record(record)

        if wa_will_send:
            msg = self._build_event_summary_message(
                level=level,
                peak_det=peak_det,
                duration=duration,
                frame_count=frame_count,
                class_counts=class_counts,
                started_at=started_at,
                ended_at=ended_at,
                extra_text=extra_text,
            )
            self._enqueue_built_message(
                msg, frame_to_send, snapshot_dir,
                prefix=("suspect" if level == "suspect" else "alarm") + "_summary",
                tag=f"{level}-summary",
            )
        else:
            logger.info(
                f"Event [{level}] selesai (dur={duration:.1f}s, "
                f"frames={frame_count}) -- summary tidak dikirim."
            )

    # ---------- Helpers persistensi (JSONL + snapshot deterministik) -------
    @staticmethod
    def _extract_camera_location(extra_text: str) -> str:
        """Ambil 'Lokasi : XYZ' dari extra_text (best-effort)."""
        if not extra_text:
            return ""
        for line in extra_text.splitlines():
            s = line.strip()
            low = s.lower()
            if low.startswith("lokasi"):
                _, _, val = s.partition(":")
                return val.strip()
        return ""

    def _save_event_snapshot(
        self, event_id: str, frame: Optional[np.ndarray]
    ) -> Optional[str]:
        """Simpan peak frame ke path deterministik ``event_<id>.jpg``.

        Path ini dirujuk oleh JSONL record sehingga dashboard tahu file
        mana yang harus ditampilkan utk event tsb.
        """
        if frame is None:
            return None
        try:
            self.event_snapshot_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.event_snapshot_dir / f"event_{event_id}.jpg"
            f = frame
            h, w = f.shape[:2]
            if w > self.snapshot_max_width:
                scale = self.snapshot_max_width / float(w)
                f = cv2.resize(
                    f, (self.snapshot_max_width, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buf = cv2.imencode(
                ".jpg", f,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.snapshot_jpeg_quality],
            )
            if not ok:
                return None
            out_path.write_bytes(buf.tobytes())
            return str(out_path).replace("\\", "/")
        except Exception as e:
            logger.warning(f"Gagal simpan event snapshot: {e}")
            return None

    def _persist_event_record(self, record: dict) -> None:
        """Append-only JSONL writer (1 baris JSON per event)."""
        try:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with self._jsonl_lock:
                with self.event_log_path.open("a", encoding="utf-8") as fp:
                    fp.write(line + "\n")
        except Exception as e:
            logger.warning(f"Gagal tulis event ke JSONL: {e}")

    def _enqueue_built_message(
        self,
        message: str,
        frame: Optional[np.ndarray],
        snapshot_dir: str | Path,
        prefix: str,
        tag: str,
    ) -> bool:
        """Enqueue pesan yang sudah jadi (dipakai event lifecycle)."""
        if self._queue is None or self._worker is None or not self._worker.is_alive():
            # Worker belum jalan -> kirim synchronous (mungkin blocking).
            logger.warning(
                f"Worker alarm belum start, fallback sync utk [{tag}]."
            )
            ok = self._send_with_image(message, frame, snapshot_dir, prefix)
            if ok:
                logger.info(f"Alarm WA [{tag}] terkirim ke {self.recipients}")
            return ok
        job = {
            "message": message,
            "frame": frame.copy() if frame is not None else None,
            "snapshot_dir": snapshot_dir,
            "prefix": prefix,
            "tag": tag,
        }
        try:
            self._queue.put_nowait(job)
            logger.info(
                f"Alarm WA [{tag}] queued (qsize={self._queue.qsize()})"
            )
            return True
        except queue.Full:
            logger.warning(f"Antrian alarm penuh, drop job [{tag}].")
            return False

    def flush_event(
        self,
        extra_text: str = "",
        snapshot_dir: str | Path = "runs/alarm",
    ) -> None:
        """Tutup paksa event yang masih aktif (mis. saat shutdown)."""
        with self._event_lock:
            if self._event_active:
                self._close_event_locked(time.time(), extra_text, snapshot_dir)

    def _build_event_start_message(
        self, detection: dict, level: str, extra_text: str
    ) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cls = detection.get("class_name", "?")
        conf = float(detection.get("confidence", 0.0))
        if level == "suspect":
            header = "⚠️ *EVENT DICURIGAI MEROKOK DIMULAI*"
            note = ("_Confidence di bawah ambang konfirmasi — "
                    "mohon verifikasi visual._")
        else:
            header = "🚨 *EVENT DETEKSI MEROKOK DIMULAI*"
            note = ""
        msg = (
            f"{header}\n"
            f"Waktu  : {ts}\n"
            f"Kelas  : {cls}\n"
            f"Confidence: {conf:.2%}\n"
        )
        if extra_text:
            msg += f"{extra_text}\n"
        msg += (
            f"\n_Ringkasan akan dikirim saat event selesai "
            f"(idle > {self.event_gap_sec:.0f}s atau durasi > "
            f"{self.event_max_dur_sec:.0f}s)._\n"
        )
        if note:
            msg += f"\n{note}\n"
        return msg

    def _build_event_summary_message(
        self,
        level: str,
        peak_det: dict,
        duration: float,
        frame_count: int,
        class_counts: dict,
        started_at: float,
        ended_at: float,
        extra_text: str,
    ) -> str:
        t_start = time.strftime("%H:%M:%S", time.localtime(started_at))
        t_end = time.strftime("%H:%M:%S", time.localtime(ended_at))
        peak_cls = peak_det.get("class_name", "?")
        peak_conf = float(peak_det.get("confidence", 0.0))
        if level == "suspect":
            header = "⚠️ *RINGKASAN EVENT DICURIGAI MEROKOK*"
        else:
            header = "🚨 *RINGKASAN EVENT DETEKSI MEROKOK*"
        # Format breakdown kelas: "merokok: 12, pegang rokok: 5"
        if class_counts:
            breakdown = ", ".join(
                f"{k}: {v}" for k, v in sorted(
                    class_counts.items(), key=lambda x: -x[1]
                )
            )
        else:
            breakdown = "-"
        msg = (
            f"{header}\n"
            f"Mulai   : {t_start}\n"
            f"Selesai : {t_end}\n"
            f"Durasi  : {duration:.1f} detik\n"
            f"Frame   : {frame_count}\n"
            f"Peak    : {peak_cls} ({peak_conf:.2%})\n"
            f"Kelas   : {breakdown}\n"
        )
        if extra_text:
            msg += f"{extra_text}\n"
        msg += "\n_Snapshot terlampir adalah frame dgn confidence tertinggi._\n"
        return msg

    # ----------------------------------------------------------------- public
    def _build_message(
        self, detection: dict, level: str, extra_text: str
    ) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cls = detection.get("class_name", "?")
        conf = float(detection.get("confidence", 0.0))
        if level == "suspect":
            header = "⚠️ *DICURIGAI MEROKOK*"
            note = ("_Confidence di bawah ambang konfirmasi — "
                    "mohon verifikasi visual._")
        else:
            header = "🚨 *ALARM DETEKSI MEROKOK*"
            note = ""
        msg = (
            f"{header}\n"
            f"Waktu  : {ts}\n"
            f"Kelas  : {cls}\n"
            f"Confidence: {conf:.2%}\n"
        )
        if extra_text:
            msg += f"{extra_text}\n"
        if note:
            msg += f"\n{note}\n"
        return msg

    def _do_send(
        self,
        detection: dict,
        frame: Optional[np.ndarray],
        snapshot_dir: str | Path,
        extra_text: str,
        level: str,
    ) -> bool:
        """Kirim ke WAHA tanpa menyentuh cooldown. Aman dipanggil dari worker."""
        msg = self._build_message(detection, level, extra_text)
        prefix = "suspect" if level == "suspect" else "alarm"
        return self._send_with_image(msg, frame, snapshot_dir, prefix)

    def _send_with_image(
        self,
        msg: str,
        frame: Optional[np.ndarray],
        snapshot_dir: str | Path,
        prefix: str,
    ) -> bool:
        """Encode frame -> simpan snapshot -> kirim via WAHA. Fallback text."""
        if frame is None:
            return self.send_text(msg)

        # Resize ke max width (utk memperkecil payload upload).
        h, w = frame.shape[:2]
        if w > self.snapshot_max_width:
            scale = self.snapshot_max_width / float(w)
            new_size = (self.snapshot_max_width, int(h * scale))
            try:
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            except Exception:
                pass

        # Encode JPEG di memori (lebih cepat drpd write disk + read disk).
        try:
            ok_enc, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.snapshot_jpeg_quality],
            )
        except Exception as e:
            logger.error(f"Gagal encode JPEG: {e}")
            return self.send_text(msg)
        if not ok_enc:
            return self.send_text(msg)
        jpg_bytes = buf.tobytes()

        # Simpan snapshot utk audit (best effort, tidak fatal kalau gagal).
        filename = f"{prefix}_{int(time.time())}.jpg"
        try:
            snap_dir = Path(snapshot_dir)
            snap_dir.mkdir(parents=True, exist_ok=True)
            (snap_dir / filename).write_bytes(jpg_bytes)
        except Exception as e:
            logger.warning(f"Gagal simpan snapshot: {e}")

        # Kirim ke WAHA langsung dari bytes (tanpa disk roundtrip).
        ok = self.send_image_bytes(jpg_bytes, filename, caption=msg)
        if not ok:
            # Fallback: paling tidak teks terkirim.
            ok = self.send_text(msg)
        return ok

    def trigger(
        self,
        detection: dict,
        frame: Optional[np.ndarray] = None,
        snapshot_dir: str | Path = "runs/alarm",
        extra_text: str = "",
        level: str = "confirm",
    ) -> bool:
        """Kirim alarm WA secara SYNCHRONOUS. Hormati cooldown per-level.

        Catatan: untuk pipeline realtime gunakan ``enqueue`` agar tidak
        memblokir loop video.
        """
        if not self.enabled or self._on_cooldown(level):
            return False

        conf = float(detection.get("confidence", 0.0))
        ok = self._do_send(detection, frame, snapshot_dir, extra_text, level)
        if ok:
            self._mark_sent(level)
            logger.info(
                f"Alarm WA [{level}] terkirim ke {self.recipients} "
                f"(conf={conf:.2%})"
            )
        else:
            logger.error(f"Alarm WA [{level}] GAGAL dikirim.")
        return ok

    # ------------------------------------------------------------ async API
    def _mark_sent(self, level: str) -> None:
        with self._cooldown_lock:
            now = time.time()
            if level == "confirm":
                self._last_sent_confirm = now
                # confirm juga mendiamkan suspect agar tidak dobel
                self._last_sent_suspect = now
            else:
                self._last_sent_suspect = now

    def start_worker(self) -> None:
        """Mulai thread background untuk mengirim alarm tanpa memblokir."""
        if self._worker is not None and self._worker.is_alive():
            return
        if not self.enabled:
            logger.info("Worker alarm tidak dimulai (recipients kosong).")
            return
        self._queue = queue.Queue(maxsize=16)
        self._worker_stop.clear()
        self._worker = threading.Thread(
            target=self._worker_loop, name="WhatsAppAlarmWorker", daemon=True
        )
        self._worker.start()
        logger.info("WhatsAppAlarm worker dimulai (async).")

    def _worker_loop(self) -> None:
        assert self._queue is not None
        while not self._worker_stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                self._queue.task_done()
                break
            try:
                # --- Job event-based (pesan sudah dibuild di luar) ---
                if "message" in job:
                    ok = self._send_with_image(
                        job["message"], job["frame"],
                        job["snapshot_dir"], job.get("prefix", "event"),
                    )
                    tag = job.get("tag", "event")
                    if ok:
                        logger.info(
                            f"Alarm WA [{tag}] terkirim ke {self.recipients}"
                        )
                    else:
                        logger.error(f"Alarm WA [{tag}] GAGAL dikirim.")
                    continue
                # --- Job legacy (per-detection + cooldown) ---
                conf = float(job["detection"].get("confidence", 0.0))
                level = job["level"]
                ok = self._do_send(
                    job["detection"], job["frame"], job["snapshot_dir"],
                    job["extra_text"], level,
                )
                if ok:
                    logger.info(
                        f"Alarm WA [{level}] terkirim ke {self.recipients} "
                        f"(conf={conf:.2%})"
                    )
                else:
                    # Gagal kirim -> rollback cooldown agar bisa dicoba lagi
                    # pada deteksi berikutnya.
                    with self._cooldown_lock:
                        if level == "confirm":
                            self._last_sent_confirm = 0.0
                        # suspect cooldown tetap, biar tidak nge-spam retry
                    logger.error(
                        f"Alarm WA [{level}] GAGAL dikirim "
                        f"(cooldown confirm di-reset)."
                    )
            except Exception as e:
                logger.exception(f"Alarm worker error: {e}")
            finally:
                self._queue.task_done()

    def enqueue(
        self,
        detection: dict,
        frame: Optional[np.ndarray] = None,
        snapshot_dir: str | Path = "runs/alarm",
        extra_text: str = "",
        level: str = "confirm",
    ) -> bool:
        """Antrekan alarm ke worker async. Cooldown di-mark seketika.

        Return True jika berhasil dimasukkan ke antrian.
        """
        if not self.enabled:
            return False
        if level == "suspect" and not self.suspect_enabled:
            conf = float(detection.get("confidence", 0.0))
            logger.info(
                f"Suspect diabaikan (ALERT_SUSPECT_ENABLED=false), "
                f"conf={conf:.2%}"
            )
            return False
        if self._queue is None or self._worker is None or not self._worker.is_alive():
            # Worker belum jalan -> fallback synchronous (mungkin blocking,
            # tapi lebih baik dibanding alarm hilang).
            logger.warning("Worker alarm belum start, fallback synchronous.")
            return self.trigger(
                detection, frame=frame, snapshot_dir=snapshot_dir,
                extra_text=extra_text, level=level,
            )
        if self._on_cooldown(level):
            return False
        # Mark cooldown SEBELUM enqueue supaya main loop tidak terus
        # mengantrekan job duplikat selama worker sedang mengirim.
        self._mark_sent(level)
        job = {
            "detection": detection,
            # Copy frame supaya aman dari mutasi loop video.
            "frame": frame.copy() if frame is not None else None,
            "snapshot_dir": snapshot_dir,
            "extra_text": extra_text,
            "level": level,
        }
        try:
            self._queue.put_nowait(job)
            conf = float(detection.get("confidence", 0.0))
            logger.info(
                f"Alarm WA [{level}] queued (conf={conf:.2%}, "
                f"qsize={self._queue.qsize()})"
            )
            return True
        except queue.Full:
            logger.warning(f"Antrian alarm penuh, drop job [{level}].")
            return False

    def stop(self, timeout: float = 5.0) -> None:
        """Hentikan worker dengan rapi (flush event aktif + flush antrian)."""
        # Jika ada event yang masih aktif, paksa tutup dulu supaya
        # ringkasannya tetap terkirim sebelum proses berakhir.
        try:
            self.flush_event()
        except Exception as e:
            logger.warning(f"flush_event() gagal saat stop: {e}")

        if self._worker is None:
            return
        self._worker_stop.set()
        try:
            if self._queue is not None:
                self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=timeout)
        logger.info("WhatsAppAlarm worker dihentikan.")
