# app/scoring.py
import cv2
import numpy as np

# ── Score ranges changed to 0-10 scale ───────────────────
CLASS_SCORE_RANGE = {
    "bad_sign"  : (0.0, 3.5),    # low reflectivity
    "moderate"  : (3.6, 6.5),    # medium reflectivity
    "good_sign" : (6.6, 10.0),   # high reflectivity
}

def compute_brightness(img_rgb, x1, y1, x2, y2, padding=3):
    h, w  = img_rgb.shape[:2]
    x1c   = max(0, int(x1) - padding)
    y1c   = max(0, int(y1) - padding)
    x2c   = min(w, int(x2) + padding)
    y2c   = min(h, int(y2) + padding)
    crop  = img_rgb[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return 128.0
    hsv        = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    brightness = float(np.percentile(hsv[:, :, 2], 75))
    return brightness

def compute_reflectivity_score(cls_name, brightness):
    low, high = CLASS_SCORE_RANGE.get(cls_name, (0.0, 10.0))
    score     = low + (brightness / 255.0) * (high - low)
    score     = max(low, min(high, score))
    return round(score, 1)   # e.g. 3.2, 7.8, 9.4

def score_to_label(score):
    if score <= 3.5: return "Low Reflectivity"
    if score <= 6.5: return "Moderate Reflectivity"
    return "High Reflectivity"

def score_to_hex_color(score):
    if score <= 3.5: return "#e74c3c"
    if score <= 6.5: return "#f39c12"
    return "#2ecc71"

def get_action_required(cls_name, score):
    if score < 1.5:
        return "CRITICAL — Immediate safety hazard"
    actions = {
        "bad_sign"  : "URGENT — Replace sign immediately",
        "moderate"  : "SCHEDULE — Plan replacement within 3 months",
        "good_sign" : "OK — No action required",
    }
    return actions.get(cls_name, "Unknown")

def is_valid_detection(x1, y1, x2, y2, img_h, img_w,
                       min_area_ratio=0.005,
                       max_area_ratio=0.85):
    box_area   = (x2 - x1) * (y2 - y1)
    img_area   = img_h * img_w
    area_ratio = box_area / img_area
    if area_ratio < min_area_ratio:
        return False, f"Too small ({area_ratio:.3f} < {min_area_ratio})"
    if area_ratio > max_area_ratio:
        return False, f"Too large ({area_ratio:.3f} > {max_area_ratio})"
    return True, "OK"

def is_realistic_aspect_ratio(x1, y1, x2, y2,
                               min_ratio=0.3,
                               max_ratio=3.0):
    w = x2 - x1
    h = y2 - y1
    if h == 0:
        return False
    return min_ratio <= (w / h) <= max_ratio