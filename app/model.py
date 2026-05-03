# app/model.py
import os
import cv2
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO
from app.scoring import (
    compute_brightness,
    compute_reflectivity_score,
    score_to_label,
    is_valid_detection,           # ← add this
    is_realistic_aspect_ratio, 
)

# ── Config ────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "models/best.pt")
DEVICE      = os.getenv("DEVICE", "0")   # "0" = RTX 3050
CLASS_NAMES = ["bad_sign", "good_sign", "moderate"]

# ── Load model once at startup ────────────────────────────
assert Path(MODEL_PATH).exists(), (
    f"❌ Model not found: {MODEL_PATH}\n"
    f"Make sure best.pt is inside the models/ folder"
)

model = YOLO(MODEL_PATH)
print(f"✅ Model loaded : {MODEL_PATH}")
print(f"✅ Device       : {DEVICE}")
print(f"✅ Classes      : {CLASS_NAMES}")


# app/model.py — update run_detection() function
# app/model.py — add this helper function

def remove_duplicate_boxes(detections, overlap_threshold=0.5):
    """
    Remove duplicate detections of the same physical sign.
    Keeps only the highest confidence box when two boxes
    overlap significantly.
    """
    if len(detections) <= 1:
        return detections

    def iou(a, b):
        ax1,ay1,ax2,ay2 = a["bounding_box"].values()
        bx1,by1,bx2,by2 = b["bounding_box"].values()
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2-ix1) * (iy2-iy1)
        area_a = (ax2-ax1) * (ay2-ay1)
        area_b = (bx2-bx1) * (by2-by1)
        return inter / (area_a + area_b - inter)

    # Sort by confidence descending
    sorted_dets = sorted(
        detections, key=lambda d: d["confidence"], reverse=True
    )

    kept = []
    for det in sorted_dets:
        overlap = False
        for k in kept:
            if iou(det, k) > overlap_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(det)

    # Re-assign sign IDs
    for i, d in enumerate(kept):
        d["sign_id"] = i + 1

    return kept
def run_detection(img_bytes: bytes, conf: float = 0.35):
    nparr   = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]

    result = model.predict(
        source       = img_rgb,
        conf         = conf,
        iou          = 0.3,        # aggressive duplicate removal
        imgsz        = 640,
        device       = DEVICE,
        verbose      = False,
        agnostic_nms = True,       # merge boxes across classes
    )[0]

    detections = []
    skipped    = []

    if result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_score      = float(box.conf[0].cpu().numpy())
            cls_id          = int(box.cls[0].cpu().numpy())
            cls_name        = CLASS_NAMES[cls_id]

            # Filter 1: box size
            valid, reason = is_valid_detection(
                x1, y1, x2, y2, h, w,
                min_area_ratio = 0.002,
                max_area_ratio = 0.90
            )
            if not valid:
                skipped.append(f"Sign {i+1} ({cls_name}): {reason}")
                continue

            # Filter 2: aspect ratio
            if not is_realistic_aspect_ratio(x1, y1, x2, y2):
                skipped.append(
                    f"Sign {i+1} ({cls_name}): bad aspect ratio"
                )
                continue

            # Filter 3: spans full image
            margin = 5
            if (x1 < margin and y1 < margin and
                x2 > w-margin and y2 > h-margin):
                skipped.append(
                    f"Sign {i+1} ({cls_name}): spans full image"
                )
                continue

            brightness = compute_brightness(img_rgb, x1, y1, x2, y2)
            ref_score  = compute_reflectivity_score(cls_name, brightness)

            detections.append({
                "sign_id"            : len(detections) + 1,
                "class"              : cls_name,
                "confidence"         : round(conf_score, 4),
                "reflectivity_score" : ref_score,
                "reflectivity_label" : score_to_label(ref_score),
                "brightness"         : round(brightness, 2),
                "bounding_box"       : {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                }
            })

    # Remove overlapping duplicate boxes
    detections = remove_duplicate_boxes(detections, overlap_threshold=0.5)

    if skipped:
        print(f"Filtered {len(skipped)} false positive(s):")
        for s in skipped: print(f"  - {s}")

    return img_rgb, detections
def generate_annotated_image(img_rgb, detections):
    """Draw boxes + scores → return base64 PNG string."""
    COLOR_HEX = {
        "bad_sign"  : "#e74c3c",
        "good_sign" : "#2ecc71",
        "moderate"  : "#f39c12",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.axis("off")

    if not detections:
        h, w = img_rgb.shape[:2]
        ax.text(
            w//2, h//2, "No signs detected",
            fontsize=16, color="red", ha="center",
            bbox=dict(facecolor="white", alpha=0.8)
        )
    else:
        for d in detections:
            b               = d["bounding_box"]
            x1,y1,x2,y2    = b["x1"],b["y1"],b["x2"],b["y2"]
            color           = COLOR_HEX.get(d["class"], "#3498db")
            score           = d["reflectivity_score"]

            # Bounding box
            ax.add_patch(patches.Rectangle(
                (x1,y1), x2-x1, y2-y1,
                linewidth=2.5, edgecolor=color, facecolor="none"
            ))

            # Label
            ax.text(
                x1, y1-8,
                f"{d['class']}  |  Score: {score}  |  "
                f"{d['reflectivity_label']}  |  conf:{d['confidence']}",
                fontsize=8, color="white", fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.9,
                          pad=3, edgecolor="none")
            )

            # Score bar background
            bw = x2 - x1
            ax.add_patch(patches.Rectangle(
                (x1,y2), bw, 8,
                linewidth=0, facecolor="#333333", alpha=0.5
            ))
            # Score bar fill
            ax.add_patch(patches.Rectangle(
                (x1,y2), bw*(score/10), 8,
                linewidth=0, facecolor=color, alpha=0.9
            ))
            # Score text
            ax.text(
                x1+bw/2, y2+4, f"{score}/10",
                fontsize=7, color="white",
                ha="center", va="center", fontweight="bold"
            )

    plt.title(f"{len(detections)} sign(s) detected", fontsize=12)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")