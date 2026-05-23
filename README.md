# Deteksi Aktivitas Merokok — YOLOv12 + Alarm WhatsApp (WAHA)

Project deteksi aktivitas merokok berbasis YOLOv12 yang mengirim **alarm WhatsApp otomatis** ke nomor tujuan saat mendeteksi orang merokok. Mendukung kamera webcam, file video, maupun **kamera HP via IP Webcam** (tanpa USB).

Kelas yang dideteksi: `merokok`, `pegang rokok` (sesuai `data.yaml`).

## Arsitektur ringkas

```
┌─────────────┐  Wi-Fi/HTTP   ┌──────────────┐   event     ┌─────────┐
│ Kamera HP   │ ─────────────▶│ PC + YOLOv12 │────────────▶│  WAHA   │
│ (IP Webcam) │   MJPEG       │  (video.py)  │  HTTP POST  │ /wa-bot │
└─────────────┘               └──────┬───────┘             └────┬────┘
                                     │ JSONL                    │ WhatsApp
                                     ▼                          ▼
                              runs/events.jsonl        ┌──────────────────┐
                                     │                 │ Nomor penerima   │
                                     ▼                 │ (WA_RECIPIENT)   │
                              ┌──────────────┐         └──────────────────┘
                              │ dashboard.py │  ← evaluasi & labeling
                              │  (Streamlit) │
                              └──────────────┘
```

> **Event-based debouncing:** sistem mengelompokkan deteksi berurutan
> menjadi **satu event**. Tiap event hanya menghasilkan **2 pesan WA**
> (start + summary) — bukan 1 pesan per frame. Detail tiap event
> (timestamp, durasi, peak confidence, snapshot) disimpan ke
> `runs/events.jsonl` untuk dianalisis di dashboard.

## Struktur file

| File | Deskripsi |
|------|-----------|
| `train.py`     | Training model YOLOv12 |
| `predict.py`   | Inferensi gambar (file / folder) |
| `video.py`     | Deteksi realtime video / webcam / IP Webcam + alarm WA |
| `api.py`       | FastAPI service untuk upload gambar |
| `alarm.py`     | Client WAHA + **event-based debouncing** + JSONL logger |
| `dashboard.py` | Dashboard Streamlit (analitik event + labeling false-positive) |
| `wa-bot/`      | Bridge Node.js (whatsapp-web.js) — alternatif WAHA tanpa Docker |
| `utils.py`     | Helper (logger, drawing, JSON formatter) |
| `data.yaml`    | Konfigurasi dataset |
| `.env.example` | Template konfigurasi (salin ke `.env`) |

## Instalasi

```bash
pip install -r requirements.txt
```

> Catatan: weights `yolov12n.pt` akan otomatis diunduh oleh ultralytics
> saat pertama kali training. Untuk dashboard, paket `streamlit` & `pandas`
> sudah ada di `requirements.txt`. Bridge WA Node.js (`wa-bot/`) butuh
> Node.js LTS \u2265 18.

---

## \ud83d\ude80 Quick Start \u2014 Menjalankan Sistem End-to-End

Sistem terdiri dari **3 komponen** yang dijalankan paralel di terminal
berbeda. Asumsinya `.env` sudah diisi dan model `best.pt` sudah ada
(hasil training, atau pakai `yolov12n.pt` untuk uji cepat).

### Terminal 1 \u2014 Bot WhatsApp (WAHA / wa-bot)

Pilih **salah satu**:

**A. Bridge Node.js (direkomendasikan, tanpa Docker)**
```bash
cd wa-bot
npm install        # sekali saja
npm start
```
Buka `http://localhost:3000` \u2192 scan QR pakai HP **pengirim** alarm
(WhatsApp \u2192 Setelan \u2192 Perangkat Tertaut \u2192 Tautkan Perangkat).
Tunggu status berubah menjadi **WORKING**.

**B. WAHA via Docker**
```bash
docker run -it --rm -p 3000:3000 devlikeapro/waha
```
Buka `http://localhost:3000` \u2192 session `default` \u2192 Start \u2192 scan QR \u2192
tunggu status `WORKING`.

> Sesi tersimpan di `wa-bot/.wwebjs_auth/` (Opsi A) atau volume Docker
> (Opsi B). Cukup scan QR sekali.

### Terminal 2 \u2014 Deteksi Realtime (HP IP Webcam)

1. Pastikan **HP dan PC satu Wi-Fi** (lihat detail di bagian
   [Setup kamera HP](#setup-kamera-hp-tanpa-kabel-usb) di bawah).
2. Jalankan app **IP Webcam** di HP \u2192 tekan **Start server** \u2192 catat URL
   (mis. `http://192.168.1.10:8080`).
3. Set di `.env`:
   ```env
   VIDEO_SOURCE=http://192.168.1.10:8080/video
   MODEL_PATH=runs/train/rokok_yolov12/weights/best.pt
   WAHA_URL=http://localhost:3000
   WA_RECIPIENT=628xxxxxxxxxx
   CAMERA_LOCATION=Lab AI Lantai 2
   ```
4. Jalankan deteksi:
   ```bash
   python video.py
   ```
   Atau override sementara:
   ```bash
   python video.py --source http://192.168.1.10:8080/video
   ```

Saat YOLO mendeteksi `merokok` / `pegang rokok`:
- Sistem membuka **event**, kirim pesan WA **"EVENT DIMULAI"**.
- Setelah deteksi berhenti (gap > `ALERT_EVENT_GAP_SEC` detik), event
  ditutup \u2192 kirim **"RINGKASAN EVENT"** (durasi, peak confidence,
  jumlah frame, snapshot terbaik).
- Event direkam ke `runs/events.jsonl` + snapshot ke
  `runs/alarm/event_<id>.jpg`.

Tekan `q` di jendela video untuk berhenti.

### Terminal 3 \u2014 Dashboard

```bash
streamlit run dashboard.py
```

Browser otomatis terbuka di `http://localhost:8501`. Dashboard akan
mem-polling `runs/events.jsonl` setiap 5 detik. Lihat detail fitur di
bagian [Dashboard](#dashboard).

---

## Training

```bash
python train.py --epochs 100 --imgsz 640 --batch 16 --model yolov12n.pt
```

Hasil disimpan di `runs/train/rokok_yolov12/weights/best.pt`.

## Predict gambar

```bash
python predict.py --source test/images --conf 0.4 --save
```

Hasil JSON disimpan di `runs/predict/predictions.json`.

## Deteksi video / webcam / IP Webcam HP

```bash
python video.py --source 0                                  # webcam laptop
python video.py --source video.mp4 --save                   # file video
python video.py --source http://192.168.1.10:8080/video     # IP Webcam HP
python video.py                                             # pakai VIDEO_SOURCE dari .env
python video.py --no-alarm                                  # testing tanpa kirim WA
```

Tekan `q` untuk keluar.

### Setup kamera HP (tanpa kabel USB)

#### 1. Persiapan jaringan
- Pastikan **HP dan PC berada di jaringan Wi-Fi yang sama**.
- Disarankan **mematikan "Isolasi AP"** (AP Isolation) di router supaya
  perangkat satu Wi-Fi bisa saling akses.
- Idealnya gunakan Wi-Fi **5 GHz** untuk latensi rendah.

#### 2. Install aplikasi
- Android: **IP Webcam** oleh Pavel Khlebovich (Play Store).
- iPhone: alternatif **Iriun Webcam** atau **EpocCam** (lihat catatan di bawah).

#### 3. Konfigurasi IP Webcam (Android) — penting

Buka app **IP Webcam** lalu atur di menu utama (scroll dari atas):

| Pengaturan | Nilai yang disarankan | Alasan |
|---|---|---|
| **Video preferences → Video resolution** | `640x480` (atau `800x600`) | Resolusi tinggi memperlambat inference YOLO |
| **Video preferences → Quality** | `50`–`70` | Kompresi lebih ringan di jaringan |
| **Video preferences → FPS limit** | `15`–`20` | Lebih dari 20 fps biasanya tidak terkejar GPU/CPU |
| **Video preferences → Orientation** | `Landscape` | Sesuai posisi pengawasan |
| **Video preferences → Disable video** | OFF | Wajib aktif, kita butuh video |
| **Power management → Disable sleeping when streaming** | ON | Supaya HP tidak tidur saat streaming |
| **Power management → Wake-lock** | ON | Cegah CPU HP throttling |
| **Optional permissions → Run in background** | ON | Streaming tetap jalan saat layar mati |
| **Connection settings → Login/Password** | (kosong) | Untuk skripsi/demo. Isi kalau di jaringan publik |
| **Connection settings → Port** | `8080` (default) | Sesuaikan kalau bentrok |
| **Connection settings → Use IPv6** | OFF | OpenCV kadang bermasalah dengan IPv6 |
| **Audio preferences** | Disabled | Tidak diperlukan, hemat bandwidth |
| **Main camera** | Belakang | Lensa belakang biasanya lebih tajam |
| **Focus mode** | `Continuous video` | Auto-fokus terus-menerus |

Setelah semua di-set, scroll ke paling bawah → tekan **Start server**.

#### 4. Catat URL stream

Setelah server jalan akan muncul tulisan seperti:
```
http://192.168.1.10:8080
```
Endpoint yang dipakai aplikasi ini adalah varian MJPEG di bawah ini — **paling kompatibel dengan OpenCV**:

```
http://<ip-hp>:8080/video
```

Contoh URL alternatif yang juga didukung IP Webcam (kalau `/video` lambat):

| Endpoint | Format | Catatan |
|---|---|---|
| `http://<ip>:8080/video`         | MJPEG  | Default, paling stabil |
| `http://<ip>:8080/videofeed`     | MJPEG  | Sama dengan `/video`, alias lama |
| `http://<ip>:8080/shot.jpg`      | JPEG snapshot | Untuk fallback (bukan stream realtime) |

> Catatan: endpoint **`/h264`** atau **`/audio.wav`** tidak akan jalan
> dengan `cv2.VideoCapture` standar. Tetap pakai `/video`.

#### 5. (Opsional) IP statis untuk HP

Supaya URL stream tidak berubah saat HP reconnect Wi-Fi:
- Masuk dashboard router → daftar DHCP → cari MAC HP →
  **reservasi IP statis** (contoh selalu `192.168.1.10`).
- Atau, di HP: *Setelan Wi-Fi → Konfigurasi IP → Statis*.

#### 6. Masukkan URL ke project

Edit `.env`:
```env
VIDEO_SOURCE=http://192.168.1.10:8080/video
```

Atau override sementara via CLI:
```bash
python video.py --source http://192.168.1.10:8080/video
```

#### 7. Test koneksi cepat

Sebelum menjalankan deteksi, pastikan stream bisa dibuka:
- Buka URL `http://192.168.1.10:8080` di **browser PC** — harus muncul
  dashboard IP Webcam dan preview video.
- Atau test di Python:
  ```python
  import cv2
  cap = cv2.VideoCapture("http://192.168.1.10:8080/video")
  print("opened:", cap.isOpened())
  ok, f = cap.read(); print("frame:", ok, None if f is None else f.shape)
  ```

#### Troubleshooting IP Webcam

| Gejala | Solusi |
|---|---|
| Browser PC tidak bisa buka URL HP | HP & PC tidak satu Wi-Fi, atau AP Isolation aktif. Cek router. |
| `Gagal membuka source video` di OpenCV | Pastikan `/video` di akhir URL. Coba `/videofeed`. |
| Stream lag / patah-patah | Turunkan resolusi ke 640x480, FPS 15, Quality 50. |
| Frame kadang membeku lalu lompat | Aktifkan **Wake-lock** & **Disable sleeping when streaming**. |
| HP panas / baterai cepat habis | Colok charger; turunkan FPS & resolusi. |
| IP berubah setiap hari | Atur **IP statis / DHCP reservation** di router. |
| Login prompt di browser saat buka URL | Kosongkan field Username/Password di app, atau isi di URL: `http://user:pass@ip:8080/video`. |

> Alternatif app: **DroidCam** (Android/iOS), **RTSP Camera Server**, atau
> iPhone via **Iriun Webcam** / **EpocCam** — semua menghasilkan URL
> HTTP/RTSP yang bisa langsung dipakai pada `--source` / `VIDEO_SOURCE`.

## Alarm WhatsApp (WAHA)

### 1. Jalankan WAHA

Pilih salah satu cara di bawah ini.

#### Opsi A — Bridge Node.js (whatsapp-web.js, tanpa Docker, direkomendasikan)

Folder `wa-bot/` berisi mini-bridge berbasis [whatsapp-web.js](https://wwebjs.dev/) yang
**meniru endpoint WAHA**, jadi `alarm.py` & `.env` tidak perlu diubah.

Syarat: Node.js LTS (≥ 18) — download di https://nodejs.org/.

```bash
cd wa-bot
npm install        # sekali saja
npm start
```

Lalu buka `http://localhost:3000` → muncul QR → scan dengan WA HP pengirim
(WhatsApp → Setelan → Perangkat Tertaut → Tautkan Perangkat). Status di
halaman akan berubah menjadi **WORKING** saat siap.

> Sesi WA tersimpan di `wa-bot/.wwebjs_auth/`, jadi tidak perlu scan ulang
> setelah restart.

#### Opsi B — WAHA via Docker

```bash
docker run -it --rm -p 3000:3000 devlikeapro/waha
```

Buka `http://localhost:3000`:

1. Pilih session `default` → tekan **Start**.
2. **Scan QR Code** dengan HP yang nomornya akan dijadikan **pengirim** alarm.
3. Tunggu sampai status session berubah menjadi `WORKING`.

> Nomor pengirim tidak perlu diisi di `.env` — identitasnya melekat
> pada session WAHA / bot Node yang sudah scan QR.

### 2. Konfigurasi `.env`

Salin template lalu sesuaikan:

```bash
cp .env.example .env
```

Variabel utama:

| Variabel | Keterangan |
|---|---|
| `WAHA_URL`             | URL service WAHA (default `http://localhost:3000`) |
| `WAHA_SESSION`         | Nama session WAHA (default `default`) |
| `WA_RECIPIENT`         | Nomor penerima alarm. Format `628xxxx`. Pisah koma untuk banyak nomor |
| `ALERT_COOLDOWN_SEC`   | Jeda antar alarm (detik), anti-spam |
| `ALERT_CONF_THRESHOLD` | Confidence minimum agar alarm dipicu (0.0–1.0) |
| `ALERT_CLASSES`        | Kelas pemicu alarm (lowercase, dipisah koma) |
| `CAMERA_LOCATION`      | Teks lokasi yang ikut dikirim di pesan |
| `VIDEO_SOURCE`         | URL IP Webcam HP / `0` webcam laptop / path video |
| `MODEL_PATH`           | Path ke `best.pt` hasil training |

**Variabel event-debouncing (anti-spam pintar):**

| Variabel | Default | Keterangan |
|---|---|---|
| `ALERT_EVENT_GAP_SEC`            | `15`  | Gap tanpa deteksi (detik) sebelum event ditutup |
| `ALERT_EVENT_MAX_DUR_SEC`        | `120` | Durasi maks satu event (force-close agar tidak nempel selamanya) |
| `ALERT_EVENT_INTER_COOLDOWN_SEC` | `30`  | Jeda minimum antar event (cegah event langsung disambung) |
| `ALERT_EVENT_SEND_SUMMARY`       | `true`| Kirim pesan summary saat event ditutup |

**Variabel persistensi (dipakai dashboard):**

| Variabel | Default | Keterangan |
|---|---|---|
| `EVENT_LOG_PATH`     | `runs/events.jsonl` | File JSONL berisi 1 baris per event selesai |
| `EVENT_SNAPSHOT_DIR` | `runs/alarm`        | Folder snapshot peak frame (`event_<id>.jpg`) |
| `LABEL_LOG_PATH`     | `runs/labels.jsonl` | File label false-positive (ditulis dashboard) |

### 3. Jalankan

```bash
python video.py
```

Ketika YOLO mendeteksi kelas yang termasuk `ALERT_CLASSES` dengan
confidence di atas `ALERT_CONF_THRESHOLD`, sistem akan:

1. **Membuka event** dan mengirim pesan **"EVENT DIMULAI"** ke WA.
2. Setiap frame deteksi selanjutnya hanya memperbarui state event
   (peak confidence, jumlah frame, breakdown kelas) — **tidak** kirim WA.
3. Bila tidak ada deteksi selama `ALERT_EVENT_GAP_SEC` detik, event
   ditutup. Sistem mengirim **"RINGKASAN EVENT"** berisi durasi, peak
   confidence, jumlah frame, dan **snapshot terbaik** ke WA.
4. Record event ditulis ke `runs/events.jsonl`; snapshot ke
   `runs/alarm/event_<id>.jpg` — keduanya dipakai dashboard.
5. Event berikutnya baru bisa dimulai setelah
   `ALERT_EVENT_INTER_COOLDOWN_SEC` detik.

Hasilnya: kalau seseorang merokok selama 30 detik, kamu cukup menerima
**2 pesan** (start + summary) — bukan puluhan notifikasi per frame.

Contoh pesan summary yang diterima:

```
🚨 RINGKASAN EVENT MEROKOK
Mulai     : 2026-05-19 14:23:11
Selesai   : 2026-05-19 14:23:41
Durasi    : 30.2 detik
Frame     : 24
Peak      : merokok (87.45%)
Lokasi    : Lab AI Lantai 2
```

### Troubleshooting

| Masalah | Solusi |
|---|---|
| `WAHA session status: SCAN_QR_CODE` | Scan QR ulang di dashboard WAHA |
| `Tidak bisa cek session WAHA` | Pastikan WAHA jalan & `WAHA_URL` benar |
| `Gagal membuka source video` | HP & PC harus satu Wi-Fi, IP/URL benar, IP Webcam jalan |
| Alarm tidak terkirim | Cek `WA_RECIPIENT` & status session = `WORKING` |
| FPS lambat saat pakai IP Webcam | Turunkan resolusi di app IP Webcam ke 640x480 |

## Dashboard

Dashboard Streamlit untuk **monitoring + evaluasi** event deteksi.

### Jalankan

```bash
streamlit run dashboard.py
```

Buka `http://localhost:8501`. Dashboard otomatis membaca:
- `runs/events.jsonl` \u2014 daftar event (ditulis `alarm.py`)
- `runs/alarm/event_<id>.jpg` \u2014 snapshot tiap event
- `runs/labels.jsonl` \u2014 label false-positive (ditulis dashboard)

> Jika file `runs/events.jsonl` belum ada, dashboard akan menampilkan
> pesan info \u2014 jalankan dulu `python video.py` sampai ada event tercatat.

### Fitur

- **Filter** (sidebar): rentang tanggal, level (confirm/suspect),
  kelas peak, status label.
- **Metrik ringkas:** total event, confirm vs suspect, durasi
  rata-rata, jumlah false-positive.
- **\ud83d\udcc9 Reduksi notifikasi WhatsApp:** menghitung **berapa persen**
  notifikasi berhasil ditekan sistem dibanding skenario 1-pesan-per-frame
  (kontribusi utama yang bisa dilaporkan di skripsi).
- **Grafik:** event per jam, distribusi kelas peak, distribusi peak
  confidence, distribusi durasi event.
- **Daftar event** (expandable) dengan **thumbnail snapshot**, detail
  lengkap, dan tombol label:
  - \u274c **False Positive** \u2014 tandai event sebagai salah deteksi
  - \u2705 **True Positive** \u2014 konfirmasi deteksi benar
  - \u21ba **Reset label**
- **Ekspor CSV** event yang difilter (untuk analisis di Excel/notebook
  skripsi).

### Konfigurasi (opsional)

Override path file via env var (sama dengan `alarm.py`):

```env
EVENT_LOG_PATH=runs/events.jsonl
EVENT_SNAPSHOT_DIR=runs/alarm
LABEL_LOG_PATH=runs/labels.jsonl
```

### Catatan

- Format JSONL **append-only** \u2014 aman dijalankan paralel dengan
  `video.py`. Dashboard pakai cache TTL 5 detik (klik **Muat ulang data**
  di sidebar untuk refresh manual).
- Label tersimpan terpisah di `runs/labels.jsonl` (record terakhir per
  `event_id` menang) supaya `events.jsonl` tetap immutable.

## API (FastAPI)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET  /health`
- `POST /predict` — multipart upload `file`, return JSON
- `POST /predict/visual` — return PNG dengan bounding box

Contoh request:

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@test/images/sample.jpg" \
     -F "conf=0.35"
```

Konfigurasi via env var: `MODEL_PATH`, `DEVICE`, `DEFAULT_CONF`,
`DEFAULT_IOU`, `IMG_SIZE`, `MAX_UPLOAD_MB`.

## Format JSON deteksi

```json
{
  "class_id": 3,
  "class_name": "merokok",
  "confidence": 0.87,
  "bbox": {"x1": 120.5, "y1": 80.2, "x2": 240.1, "y2": 310.7},
  "bbox_xywh": {"x": 180.3, "y": 195.4, "w": 119.6, "h": 230.5}
}
```
