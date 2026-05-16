# Deteksi Aktivitas Merokok — YOLOv12

Project deteksi aktivitas merokok dengan 2 kelas:
`merokok`, `pegang_rokok`.

## Struktur

| File | Deskripsi |
|------|-----------|
| `train.py`  | Training model YOLOv12 |
| `predict.py`| Inferensi gambar (file / folder) |
| `video.py`  | Deteksi realtime video / webcam |
| `api.py`    | FastAPI service untuk upload gambar |
| `utils.py`  | Helper (logger, drawing, JSON formatter) |
| `data.yaml` | Konfigurasi dataset |

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

## Deteksi video / webcam

```bash
python video.py --source 0                 # webcam
python video.py --source video.mp4 --save  # file video
```

Tekan `q` untuk keluar.

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
