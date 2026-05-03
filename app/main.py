# app/main.py
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.model import run_detection, generate_annotated_image
from app.metrics import (
    registry,
    REQUEST_COUNT, REQUEST_LATENCY,
    DETECTIONS_BY_CLASS, REFLECTIVITY_SCORE,
    AVG_REFLECTIVITY, BAD_SIGN_ALERTS,
    TOTAL_SIGNS, NO_DETECTION_COUNT,
    score_history
)

app = FastAPI(
    title       = "Road Sign Reflectivity API",
    description = "Detects road signs and scores reflectivity using YOLO11m",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
    <body style="font-family:Arial; padding:40px;
                 background:#1a1a2e; color:white">
      <h1>🚦 Road Sign Reflectivity API</h1>
      <p>Version 1.0.0 — Running on RTX 3050</p>
      <ul>
        <li><a href="/docs"    style="color:#00d4ff">
            📖 /docs    — Swagger UI (upload images here)</a></li>
        <li><a href="/health"  style="color:#00d4ff">
            💚 /health  — Health check</a></li>
        <li><a href="/metrics" style="color:#00d4ff">
            📊 /metrics — Prometheus metrics</a></li>
        <li><a href="/stats"   style="color:#00d4ff">
            📈 /stats   — Running statistics</a></li>
      </ul>
    </body>
    </html>
    """


@app.get("/health")
def health():
    import torch
    return {
        "status"  : "healthy",
        "gpu"     : torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "cpu",
        "classes" : ["bad_sign", "good_sign", "moderate"],
        "version" : "1.0.0"
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    conf: float      = 0.35,
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected image, got {file.content_type}"
        )

    start = time.time()
    REQUEST_COUNT.inc()

    try:
        img_bytes            = await file.read()
        img_rgb, detections  = run_detection(img_bytes, conf)
        latency              = time.time() - start

        # Update Prometheus metrics
        REQUEST_LATENCY.observe(latency)
        TOTAL_SIGNS.inc(len(detections))

        if not detections:
            NO_DETECTION_COUNT.inc()
        else:
            for d in detections:
                cls   = d["class"]
                score = d["reflectivity_score"]

                DETECTIONS_BY_CLASS.labels(class_name=cls).inc()
                REFLECTIVITY_SCORE.labels(class_name=cls).set(score)

                # Rolling average
                score_history[cls].append(score)
                if len(score_history[cls]) > 100:
                    score_history[cls].pop(0)
                avg = sum(score_history[cls]) / len(score_history[cls])
                AVG_REFLECTIVITY.labels(class_name=cls).set(avg)

                if cls == "bad_sign" or score < 35:
                    BAD_SIGN_ALERTS.inc()

        annotated = generate_annotated_image(img_rgb, detections)

        scores  = [d["reflectivity_score"] for d in detections]
        summary = {
            "total_signs"     : len(detections),
            "avg_score"       : round(sum(scores)/len(scores), 1)
                                if scores else None,
            "min_score"       : min(scores) if scores else None,
            "max_score"       : max(scores) if scores else None,
            "bad_signs_count" : sum(1 for d in detections
                                    if d["class"] == "bad_sign"),
            "good_signs_count": sum(1 for d in detections
                                    if d["class"] == "good_sign"),
            "moderate_count"  : sum(1 for d in detections
                                    if d["class"] == "moderate"),
            "alert"           : any(d["class"] == "bad_sign"
                                    for d in detections),
        }

        return JSONResponse({
            "status"          : "success",
            "filename"        : file.filename,
            "latency_seconds" : round(latency, 3),
            "summary"         : summary,
            "detections"      : detections,
            "annotated_image" : annotated,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(
        content    = generate_latest(registry),
        media_type = CONTENT_TYPE_LATEST
    )


@app.get("/stats")
def stats():
    return {
        cls: {
            "count": len(v),
            "avg"  : round(sum(v)/len(v), 1) if v else None,
            "min"  : min(v) if v else None,
            "max"  : max(v) if v else None,
        }
        for cls, v in score_history.items()
    }