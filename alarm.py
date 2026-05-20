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
import queue
import threading
import time
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
            raw = os.getenv("ALERT_CLASSES", "merokok,pegang rokok")
            alert_classes = [c.strip().lower() for c in raw.split(",") if c.strip()]
        else:
            alert_classes = [c.strip().lower() for c in alert_classes]
        self.alert_classes = set(alert_classes)

        # Kelas yang BOLEH naik ke level "confirm". Kelas di alert_classes
        # tapi TIDAK ada di sini akan di-cap maksimal "suspect"
        # (mis. "pegang rokok" -> belum tentu menghisap).
        raw_confirm = os.getenv(
            "ALERT_CONFIRM_CLASSES",
            "dataset-deteksi-merokok,merokok",
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

    def should_trigger(
        self, detections: list[dict]
    ) -> Tuple[Optional[str], Optional[dict]]:
        """Evaluasi deteksi -> kembalikan (level, detection_terbaik).

        Strategi:
            1. Kumpulkan SEMUA bbox per frame yang lolos kelas + suspect_conf
               (bukan hanya yang tertinggi -> supaya event "merokok" yang
               singkat tidak tertutup oleh "pegang rokok" yang dominan).
            2. PRIORITAS: kalau di window ada >= confirm_min_hits bbox dari
               confirm-class (mis. "merokok") dgn conf >= confirm_conf,
               langsung trigger CONFIRM. Default confirm_min_hits=1 supaya
               event hisap-rokok yg cepat tidak terlewat.
            3. Kalau tidak, fallback ke logika lama: butuh >= min_hits di
               window, pilih conf tertinggi -> tier suspect.
        """
        if not self.enabled:
            return None, None

        now = time.time()
        # Kumpulkan SEMUA bbox sah di frame ini (bukan hanya max).
        # Ini krusial: kalau frame punya "pegang_rokok 0.85" + "merokok 0.72",
        # keduanya HARUS masuk window -> supaya "merokok" tdk hilang.
        for det in detections:
            conf = float(det.get("confidence", 0.0))
            if conf < self.suspect_conf:
                continue
            name = str(det.get("class_name", "")).lower()
            if self.alert_classes and name not in self.alert_classes:
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
            if self._on_cooldown("confirm"):
                return None, None
            best = max(confirm_hits, key=lambda d: float(d["confidence"]))
            return "confirm", best

        # --- (2) Fallback: tier SUSPECT (butuh >= min_hits) ---
        if len(self._history) < self.min_hits:
            return None, None
        best_in_window = max(
            self._history, key=lambda x: float(x[1]["confidence"])
        )[1]
        if self._on_cooldown("suspect"):
            return None, None
        return "suspect", best_in_window

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
        prefix = "suspect" if level == "suspect" else "alarm"
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
        """Hentikan worker dengan rapi (flush antrian)."""
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
