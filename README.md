# 🚦 AI-Based Road Sign Reflectivity Detection for Proactive Road Safety Maintenance

## 📌 Project Overview

An end-to-end computer vision system that detects Indian road warning signs
from images and assigns each sign a **reflectivity score (0–10)** to enable
proactive maintenance before signs become safety hazards.

| Metric | Value |
|---|---|
| Model | YOLO11m (fine-tuned) |
| mAP50 | **86.9%** |
| Recall | **85.5%** |
| Precision | 81.6% |
| Dataset | 2,746 images |
| Score range | 0–10 per sign |

---

## 🎯 Problem Statement

Road signs lose reflectivity over time due to weathering and UV exposure.
Current manual inspection is infrequent and subjective. This system automates
detection and scoring to flag degraded signs before they cause accidents.

---

## 🏗️ Architecture
Image Upload → FastAPI → YOLO11m Inference → Reflectivity Scorer
↓
JSON Response + Annotated Image
↓
Prometheus Metrics → Grafana Dashboard

---

## 📊 Classes & Scoring

| Class | Score Range | Action |
|---|---|---|
| 🔴 bad_sign | 0.0 – 3.5 | Urgent replacement |
| 🟡 moderate | 3.6 – 6.5 | Schedule maintenance |
| 🟢 good_sign | 6.6 – 10.0 | No action needed |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLO11m (Ultralytics) |
| API | FastAPI + Uvicorn |
| Image Processing | OpenCV + Matplotlib |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + Docker Compose |
| Training | Google Colab (Tesla T4) |
| Inference | NVIDIA RTX 3050 4GB |

---

## 📁 Project Structure
road_sign_monitor/
├── app/
│   ├── init.py
│   ├── main.py          # FastAPI routes
│   ├── model.py         # YOLO inference + annotation
│   ├── scoring.py       # Reflectivity score (0-10)
│   └── metrics.py       # Prometheus metrics
├── models/
│   └── best.pt          # ← Download separately (see below)
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── provisioning/
├── demo.html            # Frontend UI
├── docker-compose.yml
├── requirements.txt
└── run.bat              # Windows one-click start

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.9 (Anaconda recommended)
- NVIDIA GPU with CUDA 11.8 (CPU fallback supported)
- Docker Desktop

### Installation

```bash
# Clone repository
git clone https://github.com/arpitareva/road-sign-reflectivity-detection-and-scoring.git
cd road-sign-reflectivity-detection

# Install dependencies
pip install -r requirements.txt

# Download model weights (see Model Weights section below)
# Place best.pt inside models/ folder
```

### Running

```bash
# Start Prometheus + Grafana
docker-compose up -d

# Start FastAPI
conda activate sperm_gpu
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access

| Service | URL |
|---|---|
| Demo UI | Open `demo.html` in Chrome |
| API Docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/roadsign123) |

---

## 📦 Model Weights

The trained model (`best.pt`) is too large for GitHub.
Download from Google Drive:

🔗 **[Download best.pt]:https://drive.google.com/file/d/1CuT165q7EdfwrwosXcvWyFtGMCK4SQ36/view?usp=sharing**

Place it in the `models/` folder before running.

---

## 📡 API Usage

```python
import requests

with open("road_sign.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
        params={"conf": 0.35}
    )

data = response.json()
print(f"Signs detected: {data['summary']['total_signs']}")
print(f"Avg score: {data['summary']['avg_score']}/10")
print(f"Alert: {data['summary']['alert']}")
```

**Sample Response:**
```json
{
  "status": "success",
  "latency_seconds": 2.3,
  "summary": {
    "total_signs": 1,
    "avg_score": 3.2,
    "bad_signs_count": 1,
    "alert": true
  },
  "detections": [{
    "class": "bad_sign",
    "confidence": 0.881,
    "reflectivity_score": 3.2,
    "reflectivity_label": "Low Reflectivity"
  }]
}
```

---

## 📈 Results

| Model | mAP50 | Precision | Recall | bad_sign mAP50 |
|---|---|---|---|---|
| Roboflow YOLOv8 (baseline) | 0.820 | 0.900 | 0.730 | — |
| YOLO11m v1 (initial) | 0.538 | 0.582 | 0.557 | 0.312 |
| YOLO11m v2 (balanced) | 0.846 | 0.861 | 0.788 | 0.804 |
| **YOLO11m Fine-tuned** | **0.869** | **0.816** | **0.855** | **0.846** |

---

## 🔭 Future Scope

- Real-time dashcam video stream inference
- GPS coordinate tagging per detection
- Mobile inspection app for field officers
- Quantized YOLO11n for edge/embedded deployment
- Expansion to all Indian road sign types

---

## 👩 Author

**Arpita Revankar**
- Project built as part of Data Science portfolio
- Tools: Python, PyTorch, YOLO11m, FastAPI, Docker, Grafana