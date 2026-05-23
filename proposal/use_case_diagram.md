# Use Case Diagram — Sistem Deteksi Aktivitas Merokok

Versi Mermaid (bisa langsung di-render di VS Code / GitHub).
Lihat juga versi PlantUML di [use_case_diagram.puml](use_case_diagram.puml).

```mermaid
%%{init: {'theme':'base'}}%%
flowchart LR
    %% ===== Aktor =====
    Admin(["👤 Admin / Peneliti"])
    Operator(["👤 Operator Pengawas"])
    Penerima(["👤 Penerima Alarm"])
    Kamera(["📷 Kamera<br/>(Webcam / IP Webcam HP)"])
    WAHA(["🟢 WAHA<br/>(WhatsApp HTTP API)"])
    Model(["🧠 Model YOLOv12"])

    %% ===== Sistem =====
    subgraph SYS["Sistem Deteksi Aktivitas Merokok"]
        direction TB

        %% Training
        UC_Dataset(("Menyiapkan<br/>Dataset"))
        UC_Train(("Melatih Model<br/>YOLOv12"))
        UC_Val(("Memvalidasi /<br/>Menguji Model"))

        %% Prediksi gambar / API
        UC_PredictImg(("Memprediksi<br/>Gambar"))
        UC_API(("Mengakses API<br/>Deteksi (FastAPI)"))
        UC_Upload(("Upload Gambar<br/>via API"))
        UC_Visual(("Melihat Hasil<br/>Visual (bbox)"))
        UC_SaveJson(("Menyimpan<br/>Hasil JSON"))

        %% Deteksi realtime
        UC_Realtime(("Menjalankan Deteksi<br/>Realtime (video.py)"))
        UC_ConfigCam(("Konfigurasi<br/>Sumber Kamera"))
        UC_GrabFrame(("Mengambil<br/>Frame Video"))
        UC_Detect(("Mendeteksi<br/>Aktivitas Merokok"))
        UC_ShowBox(("Menampilkan<br/>Bounding Box"))

        %% Alarm
        UC_Alarm(("Mengirim Alarm<br/>WhatsApp"))
        UC_Snapshot(("Menyimpan<br/>Snapshot Bukti"))
        UC_Cooldown(("Cooldown<br/>(anti-spam)"))
        UC_Filter(("Filter Confidence<br/>& Kelas Target"))
        UC_Notif(("Menerima Notifikasi<br/>WhatsApp"))
    end

    %% ===== Aktor utama -> Use case =====
    Admin --- UC_Dataset
    Admin --- UC_Train
    Admin --- UC_Val
    Admin --- UC_PredictImg
    Admin --- UC_API

    Operator --- UC_Realtime
    Operator --- UC_ConfigCam
    Operator --- UC_PredictImg
    Operator --- UC_Upload

    Penerima --- UC_Notif

    %% ===== Aktor sekunder =====
    UC_GrabFrame  --- Kamera
    UC_Detect     --- Model
    UC_PredictImg --- Model
    UC_API        --- Model
    UC_Alarm      --- WAHA
    WAHA          --- UC_Notif

    %% ===== Include / Extend =====
    UC_Train      -. include .-> UC_Dataset
    UC_Val        -. include .-> UC_Train

    UC_PredictImg -. include .-> UC_SaveJson
    UC_PredictImg -. extend  .-> UC_Visual
    UC_Upload     -. extend  .-> UC_API
    UC_API        -. extend  .-> UC_Visual
    UC_API        -. extend  .-> UC_SaveJson

    UC_Realtime   -. include .-> UC_ConfigCam
    UC_Realtime   -. include .-> UC_GrabFrame
    UC_Realtime   -. include .-> UC_Detect
    UC_Realtime   -. include .-> UC_ShowBox

    UC_Detect     -. include .-> UC_Filter
    UC_Detect     -. extend  .-> UC_Alarm
    UC_Alarm      -. include .-> UC_Cooldown
    UC_Alarm      -. include .-> UC_Snapshot
```

## Penjelasan Aktor

| Aktor | Peran |
|---|---|
| **Admin / Peneliti** | Menyiapkan dataset, melatih (`train.py`), dan menguji model. |
| **Operator Pengawas** | Menjalankan deteksi realtime (`video.py`), mengatur sumber kamera, melakukan upload gambar ke API. |
| **Penerima Alarm** | Pihak yang menerima pesan WhatsApp ketika sistem mendeteksi aktivitas merokok (mis. pengawas / orang tua). |
| **Kamera** *(sekunder)* | Sumber input video — webcam laptop, file video, atau IP Webcam HP. |
| **Model YOLOv12** *(sekunder)* | Mesin inferensi yang melakukan deteksi objek `merokok` & `pegang rokok`. |
| **WAHA** *(sekunder)* | WhatsApp HTTP API (Docker) yang meneruskan alarm ke nomor penerima. |

## Penjelasan Use Case Utama

- **Melatih Model YOLOv12** — pipeline training pada `train.py` memakai `data.yaml`.
- **Memprediksi Gambar** — inferensi batch via `predict.py`, output JSON & visual.
- **Mengakses API Deteksi** — endpoint FastAPI (`/predict`, `/predict/visual`) di `api.py`.
- **Menjalankan Deteksi Realtime** — `video.py` membaca stream kamera dan menjalankan inferensi per-frame.
- **Mendeteksi Aktivitas Merokok** — memfilter kelas target & ambang confidence (`ALERT_CONF_THRESHOLD`, `ALERT_CLASSES`).
- **Mengirim Alarm WhatsApp** — `alarm.py` mengirim pesan + snapshot bukti ke `WA_RECIPIENT` lewat WAHA, dengan **cooldown** (`ALERT_COOLDOWN_SEC`) untuk mencegah spam.
