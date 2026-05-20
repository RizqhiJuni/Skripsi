# Deteksi Aktivitas Merokok — YOLOv12 + Alarm WhatsApp (WAHA)

Project deteksi aktivitas merokok berbasis YOLOv12 yang mengirim **alarm WhatsApp otomatis** ke nomor tujuan saat mendeteksi orang merokok. Mendukung kamera webcam, file video, maupun **kamera HP via IP Webcam** (tanpa USB).

Kelas yang dideteksi: `merokok`, `pegang rokok` (sesuai `data.yaml`).

## Arsitektur ringkas

```
┌─────────────┐  Wi-Fi/HTTP   ┌──────────────┐   alarm     ┌─────────┐
│ Kamera HP   │ ─────────────▶│ PC + YOLOv12 │────────────▶│  WAHA   │
│ (IP Webcam) │   MJPEG       │  (video.py)  │  HTTP POST  │ (Docker)│
└─────────────┘               └──────────────┘             └────┬────┘
                                                                │ WhatsApp
                                                                ▼
                                                       ┌──────────────────┐
                                                       │ Nomor penerima   │
                                                       │ (WA_RECIPIENT)   │
                                                       └──────────────────┘
```

## Struktur file

| File | Deskripsi |
|------|-----------|
| `train.py`     | Training model YOLOv12 |
| `predict.py`   | Inferensi gambar (file / folder) |
| `video.py`     | Deteksi realtime video / webcam / IP Webcam + alarm WA |
| `api.py`       | FastAPI service untuk upload gambar |
| `alarm.py`     | Client WAHA + cooldown anti-spam + snapshot bukti |
| `utils.py`     | Helper (logger, drawing, JSON formatter) |
| `data.yaml`    | Konfigurasi dataset |
| `.env.example` | Template konfigurasi (salin ke `.env`) |

## Instalasi

```bash
pip install -r requirements.txt
```

> Catatan: weights `yolov12n.pt` akan otomatis diunduh oleh ultralytics
> saat pertama kali training.

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

### 3. Jalankan

```bash
python video.py
```

Ketika YOLO mendeteksi kelas yang termasuk `ALERT_CLASSES` dengan
confidence di atas `ALERT_CONF_THRESHOLD`, sistem akan:

1. Menyimpan snapshot bukti ke `runs/alarm/alarm_<timestamp>.jpg`.
2. Mengirim **gambar + caption** ke setiap nomor `WA_RECIPIENT`
   melalui WAHA (`POST /api/sendImage`).
3. Menerapkan cooldown `ALERT_COOLDOWN_SEC` agar tidak spam.

Contoh pesan yang diterima:

```
🚨 ALARM DETEKSI MEROKOK
Waktu  : 2026-05-19 14:23:11
Kelas  : merokok
Confidence: 87.45%
Lokasi : Kamera HP - Ruang Utama
```

### Troubleshooting

| Masalah | Solusi |
|---|---|
| `WAHA session status: SCAN_QR_CODE` | Scan QR ulang di dashboard WAHA |
| `Tidak bisa cek session WAHA` | Pastikan WAHA jalan & `WAHA_URL` benar |
| `Gagal membuka source video` | HP & PC harus satu Wi-Fi, IP/URL benar, IP Webcam jalan |
| Alarm tidak terkirim | Cek `WA_RECIPIENT` & status session = `WORKING` |
| FPS lambat saat pakai IP Webcam | Turunkan resolusi di app IP Webcam ke 640x480 |

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
