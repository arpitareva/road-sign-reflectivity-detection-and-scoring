# app/metrics.py
from prometheus_client import (
    Counter, Histogram, Gauge,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST
)

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "roadsign_requests_total",
    "Total API requests",
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "roadsign_request_latency_seconds",
    "Request latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=registry
)

DETECTIONS_BY_CLASS = Counter(
    "roadsign_detections_total",
    "Total detections per class",
    ["class_name"],
    registry=registry
)

REFLECTIVITY_SCORE = Gauge(
    "roadsign_reflectivity_score",
    "Latest reflectivity score per class",
    ["class_name"],
    registry=registry
)

AVG_REFLECTIVITY = Gauge(
    "roadsign_avg_reflectivity",
    "Rolling average reflectivity per class",
    ["class_name"],
    registry=registry
)

BAD_SIGN_ALERTS = Counter(
    "roadsign_bad_sign_alerts_total",
    "Total bad sign detections",
    registry=registry
)

TOTAL_SIGNS = Counter(
    "roadsign_total_signs_processed",
    "Total signs processed",
    registry=registry
)

NO_DETECTION_COUNT = Counter(
    "roadsign_no_detection_total",
    "Images with no detections",
    registry=registry
)

# Rolling score history (last 100 per class)
score_history = {
    "bad_sign"  : [],
    "good_sign" : [],
    "moderate"  : [],
}