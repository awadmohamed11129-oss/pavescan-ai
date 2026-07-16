"""YOLO crack detection model loading and inference.

Supports YOLO11/YOLOv8, SAHI tiled inference, TTA, and configurable resolution.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Where trained models live
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Custom-trained model names (checked in order — v2 preferred, v1 fallback)
CUSTOM_MODELS = [
    "pavescan_crack_seg_v2.pt",   # YOLO11-seg — 1-class segmentation (crack)
    "pavescan_rdd2022_v2.pt",     # YOLO11 — 4-class detection (road damage)
    "pavescan_crack_seg.pt",      # YOLOv8n-seg fallback
    "pavescan_rdd2022.pt",        # YOLOv8n fallback
]

# Per-version slot map: each ensemble has one segmentation model + one detection model.
# RDD2022 detector hasn't been retrained yet, so V2 mode shares the V1 detector.
MODEL_VERSIONS = {
    "v1": {"seg": "pavescan_crack_seg.pt",    "det": "pavescan_rdd2022.pt"},
    "v2": {"seg": "pavescan_crack_seg_v2.pt", "det": "pavescan_rdd2022.pt"},
}

# Fallback: pretrained COCO model (detects generic objects, not cracks)
FALLBACK_MODEL = "yolov8n-seg.pt"

CONFIDENCE_THRESHOLD = 0.15

# IoU threshold for deduplication in ensemble mode
IOU_THRESHOLD = 0.5

# Weighted Box Fusion settings (alternative ensemble fusion method)
WBF_IOU_THRESHOLD = 0.55              # Standard for object-detection ensembles
WBF_SKIP_BOX_THRESHOLD = 0.0          # Already pre-filtered by confidence
WBF_LABEL_HARMONIZE_IOU = 0.5         # IoU above which generic "crack" is relabelled

# Default inference settings
DEFAULT_IMGSZ = 1280
DEFAULT_SAHI_SLICE_SIZE = 640
DEFAULT_SAHI_OVERLAP_RATIO = 0.2

# ---------------------------------------------------------------------------
# ASTM D6433-informed safety classification constants
# ---------------------------------------------------------------------------

# Per-class risk weight: how dangerous is this defect type?
# Based on ASTM D6433 distress types and real-world safety impact.
# Scale 0.0–1.0 where 1.0 = maximum danger to road users.
DEFECT_RISK_WEIGHTS = {
    # RDD2022 classes
    "D40": 1.0,                  # Pothole — tire blowouts, motorcycle ejection, pedestrian falls
    "D20": 0.90,                 # Alligator cracking — structural failure, imminent pothole
    "D00": 0.65,                 # Longitudinal crack — water infiltration, edge failure
    "D10": 0.50,                 # Transverse crack — ride quality, water entry
    # Segmentation model classes
    "Pothole": 1.0,
    "Alligator_Crack": 0.90,
    "Longitudinal_Crack": 0.65,
    "Transverse_Crack": 0.50,
    "crack": 0.45,               # Generic crack (segmentation model) — moderate risk
}
DEFAULT_RISK_WEIGHT = 0.30       # Unknown class — conservative baseline

# Structural failure types: these indicate the pavement structure itself is failing.
# ANY detection of these at ANY size is safety-relevant.
STRUCTURAL_FAILURE_TYPES = {
    "D40", "Pothole",            # Active structural failure
    "D20", "Alligator_Crack",   # Fatigue cracking → imminent structural failure
}

# What happens if this defect type goes untreated (ASTM D6433 deterioration curves)
RAPID_DETERIORATION_TYPES = {
    "D40": "Pothole expands 2-5x per freeze-thaw cycle; sub-base erosion accelerates",
    "Pothole": "Pothole expands 2-5x per freeze-thaw cycle; sub-base erosion accelerates",
    "D20": "Alligator cracking → pothole formation within 6-18 months",
    "Alligator_Crack": "Alligator cracking → pothole formation within 6-18 months",
    "D00": "Longitudinal crack widens → water infiltration → sub-base failure in 1-3 years",
    "Longitudinal_Crack": "Longitudinal crack widens → water infiltration → sub-base failure in 1-3 years",
    "D10": "Transverse crack widens with thermal cycling → spalling in 2-4 years",
    "Transverse_Crack": "Transverse crack widens with thermal cycling → spalling in 2-4 years",
    "crack": "Unclassified crack → monitor for width progression and branching",
}

# Safety priority levels — answers "what to fix FIRST" (separate from severity)
SAFETY_PRIORITIES = {
    "critical": {
        "label": "CRITICAL",
        "color_hex": "#FF1744",       # Red
        "color_bgr": (68, 23, 255),   # BGR for OpenCV
        "icon": "\u26a0\ufe0f",
        "order": 0,                    # Sort order (0 = first)
        "timeline": "Immediate — within 24-48 hours",
        "action": "Emergency repair or barricade required",
    },
    "urgent": {
        "label": "URGENT",
        "color_hex": "#FF9100",       # Orange
        "color_bgr": (0, 145, 255),
        "icon": "\U0001f7e0",
        "order": 1,
        "timeline": "Short-term — within 1-2 weeks",
        "action": "Schedule priority maintenance",
    },
    "monitor": {
        "label": "MONITOR",
        "color_hex": "#FFC400",       # Amber
        "color_bgr": (0, 196, 255),
        "icon": "\U0001f7e1",
        "order": 2,
        "timeline": "Planned — within 1-3 months",
        "action": "Include in next maintenance cycle",
    },
    "routine": {
        "label": "ROUTINE",
        "color_hex": "#00C853",       # Green
        "color_bgr": (83, 200, 0),
        "icon": "\U0001f7e2",
        "order": 3,
        "timeline": "Next scheduled inspection",
        "action": "Log and monitor at next inspection",
    },
}

# ASTM D6433 crack width thresholds (pixels — approximate at typical survey distance)
# <10mm → low severity, 10-75mm → medium, >75mm → high
# At ~3m survey height with 12MP camera: ~0.5mm/px, so:
CRACK_WIDTH_THRESHOLDS = {
    "low_max_px": 20,       # ~10mm
    "medium_max_px": 150,   # ~75mm
}


def load_model(model_path: str | None = None) -> YOLO:
    """Load a single YOLO model for crack detection.

    Searches for models in this order:
    1. Explicit model_path (if provided)
    2. Custom-trained models in models/ directory
    3. Fallback pretrained model
    """
    if model_path and Path(model_path).exists():
        return YOLO(model_path)

    for name in CUSTOM_MODELS:
        path = MODELS_DIR / name
        if path.exists():
            return YOLO(str(path))

    fallback_path = MODELS_DIR / FALLBACK_MODEL
    if fallback_path.exists():
        return YOLO(str(fallback_path))

    return YOLO(FALLBACK_MODEL)


def list_available_models() -> list[dict]:
    """List all .pt model files in the models/ directory with their info.

    Returns list of dicts: {"name", "path", "task", "classes", "num_classes"}
    """
    models = []
    if not MODELS_DIR.exists():
        return models

    for pt_file in sorted(MODELS_DIR.glob("*.pt")):
        try:
            m = YOLO(str(pt_file))
            models.append({
                "name": pt_file.stem,
                "path": str(pt_file),
                "task": m.task,
                "classes": list(m.names.values()),
                "num_classes": len(m.names),
            })
        except Exception:
            continue

    return models


def load_ensemble_models(version: str = "auto") -> tuple[dict[str, YOLO], dict[str, str]]:
    """Load one segmentation + one detection model for ensemble inference.

    Args:
        version: "v1" forces V1 seg + V1 det. "v2" forces V2 seg + V1 det
            (no V2 detector exists yet). "auto" prefers V2 seg with V1 fallback.

    Returns:
        models: dict mapping model file stem to loaded YOLO model
        model_paths: dict mapping model file stem to file path (needed for SAHI)
    """
    if version == "auto":
        slots = dict(MODEL_VERSIONS["v2"])
        if not (MODELS_DIR / slots["seg"]).exists():
            slots["seg"] = MODEL_VERSIONS["v1"]["seg"]
    elif version in MODEL_VERSIONS:
        slots = MODEL_VERSIONS[version]
    else:
        raise ValueError(f"Unknown model version: {version!r}. Expected 'v1', 'v2', or 'auto'.")

    models = {}
    model_paths = {}
    for filename in (slots["seg"], slots["det"]):
        path = MODELS_DIR / filename
        if not path.exists():
            if version in ("v1", "v2"):
                raise FileNotFoundError(
                    f"Required {version.upper()} model missing: {filename}. "
                    f"Drop the file into {MODELS_DIR} and retry."
                )
            continue
        stem = path.stem
        models[stem] = YOLO(str(path))
        model_paths[stem] = str(path)
    return models, model_paths


def compute_iou(box_a: list, box_b: list) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Geometry helpers — crack width & aspect ratio from segmentation masks
# ---------------------------------------------------------------------------

def estimate_crack_width_pixels(mask: np.ndarray | None) -> float:
    """Estimate crack width in pixels from a binary segmentation mask.

    Uses OpenCV minAreaRect on the largest contour — the shorter dimension
    of the rotated bounding rectangle approximates crack width.
    Returns 0.0 if mask is None or empty (e.g. SAHI / detection-only models).

    ASTM D6433 crack width thresholds:
      <10mm → low severity, 10-75mm → medium, >75mm → high
    """
    if mask is None:
        return 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    # Use largest contour (primary defect region)
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 0.0

    _, (w, h), _ = cv2.minAreaRect(largest)
    # Shorter dimension ≈ crack width
    return min(w, h)


def compute_aspect_ratio(mask: np.ndarray | None) -> float:
    """Compute aspect ratio (length/width) of defect from segmentation mask.

    High ratio → elongated crack. Low ratio (~1.0) → pothole or patch.
    Returns 1.0 if mask is None or empty.
    """
    if mask is None:
        return 1.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 1.0

    _, (w, h), _ = cv2.minAreaRect(largest)
    if min(w, h) == 0:
        return 1.0

    return max(w, h) / min(w, h)


# ---------------------------------------------------------------------------
# Scoring sub-functions — continuous 0-1 curves, not hard bins
# ---------------------------------------------------------------------------

def _compute_area_score(area_ratio: float) -> float:
    """Convert defect area ratio to a continuous 0-1 score.

    Piecewise linear curve calibrated to pavement engineering thresholds:
      0.0% → 0.0  (no defect)
      0.5% → 0.15 (tiny defect)
      1.0% → 0.30 (small defect)
      2.0% → 0.50 (moderate)
      5.0% → 0.80 (large)
      10%+ → 1.0  (massive)
    """
    if area_ratio <= 0:
        return 0.0
    if area_ratio >= 0.10:
        return 1.0

    # Piecewise linear breakpoints: (area_ratio, score)
    breakpoints = [
        (0.0, 0.0),
        (0.005, 0.15),
        (0.01, 0.30),
        (0.02, 0.50),
        (0.05, 0.80),
        (0.10, 1.0),
    ]

    for i in range(1, len(breakpoints)):
        x0, y0 = breakpoints[i - 1]
        x1, y1 = breakpoints[i]
        if area_ratio <= x1:
            t = (area_ratio - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return 1.0


def _compute_width_score(width_px: float) -> float:
    """Convert estimated crack width (pixels) to a continuous 0-1 score.

    Based on ASTM D6433 crack width severity criteria:
      0 px  → 0.0 (unknown or hairline)
      20 px → 0.3 (~10mm — low/medium boundary)
      80 px → 0.6 (~40mm — medium range)
      150px → 0.9 (~75mm — medium/high boundary)
      200+  → 1.0 (wide open crack)

    Returns 0.0 when width is unknown (mask=None → width_px=0).
    This is NOT a penalty — it means width doesn't contribute to the score,
    so area and class risk weight carry more influence.
    """
    if width_px <= 0:
        return 0.0
    if width_px >= 200:
        return 1.0

    breakpoints = [
        (0, 0.0),
        (20, 0.30),
        (80, 0.60),
        (150, 0.90),
        (200, 1.0),
    ]

    for i in range(1, len(breakpoints)):
        x0, y0 = breakpoints[i - 1]
        x1, y1 = breakpoints[i]
        if width_px <= x1:
            t = (width_px - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return 1.0


# ---------------------------------------------------------------------------
# Multi-factor severity classification (replaces naive area-only system)
# ---------------------------------------------------------------------------

def classify_severity(
    area_pixels: int,
    image_area: int,
    class_name: str,
    confidence: float,
    mask: np.ndarray | None = None,
    defect_count: int = 1,
    _precomputed_width_px: float | None = None,
) -> tuple[str, float]:
    """Classify defect severity using ASTM D6433-informed multi-factor scoring.

    Factors (weights adapt based on mask availability):
      With mask:  45% area + 30% width + 25% class risk
      No mask:    65% area + 35% class risk

    Additional modifiers:
      - Confidence penalty for uncertain detections
      - Density bonus when multiple defects present
      - SAFETY FLOOR: structural failures get minimum "medium" severity

    Args:
        area_pixels: Defect area in pixels (from mask or bbox)
        image_area: Total image area in pixels
        class_name: Detected defect class name
        confidence: Detection confidence (0-1)
        mask: Binary segmentation mask (None for detection-only / SAHI)
        defect_count: Total defects in this image (for density bonus)
        _precomputed_width_px: Pre-computed width to avoid double computation

    Returns:
        (severity_level, severity_score) where:
          severity_level: "low", "medium", or "high"
          severity_score: float 0.0-1.0 (continuous, for ranking)
    """
    if image_area == 0:
        return ("low", 0.0)

    area_ratio = area_pixels / image_area

    # --- Factor 1: Area score (continuous curve) ---
    area_score = _compute_area_score(area_ratio)

    # --- Factor 2: Width score (from mask, or 0 if unavailable) ---
    if _precomputed_width_px is not None:
        width_px = _precomputed_width_px
    else:
        width_px = estimate_crack_width_pixels(mask)
    width_score = _compute_width_score(width_px)

    # --- Factor 3: Class risk weight ---
    risk_weight = DEFECT_RISK_WEIGHTS.get(class_name, DEFAULT_RISK_WEIGHT)

    # --- Combine factors (adaptive weights based on mask availability) ---
    if mask is not None and width_score > 0:
        # Full information: area + width + class risk
        raw_score = 0.45 * area_score + 0.30 * width_score + 0.25 * risk_weight
    else:
        # No mask / unknown width: area + class risk carry full weight
        raw_score = 0.65 * area_score + 0.35 * risk_weight

    # --- Confidence modifier ---
    # Very uncertain detections get penalized, but not zeroed out
    if confidence < 0.20:
        raw_score *= 0.50
    elif confidence < 0.35:
        raw_score *= 0.75

    # --- Density bonus (more defects → worse pavement condition) ---
    if defect_count >= 5:
        raw_score *= 1.15
    elif defect_count >= 3:
        raw_score *= 1.08

    # Clamp to [0, 1]
    severity_score = max(0.0, min(1.0, raw_score))

    # --- Map to severity level ---
    if severity_score >= 0.60:
        severity_level = "high"
    elif severity_score >= 0.30:
        severity_level = "medium"
    else:
        severity_level = "low"

    # --- SAFETY FLOOR: structural failures are NEVER "low" ---
    # A pothole at any size is at least medium. This is the core safety fix.
    if class_name in STRUCTURAL_FAILURE_TYPES and confidence >= 0.20:
        if severity_level == "low":
            severity_level = "medium"
            severity_score = max(severity_score, 0.30)

    return (severity_level, severity_score)


# ---------------------------------------------------------------------------
# Defect density, safety priority, deterioration risk, and batch classifier
# ---------------------------------------------------------------------------

def compute_defect_density(detections: list[dict], image_area: int) -> dict:
    """Analyze defect density for systemic failure detection.

    ASTM D6433: high defect density indicates structural pavement failure,
    not just isolated distresses. This changes the maintenance strategy
    from spot repairs to full section rehabilitation.

    Returns:
        {count, structural_count, total_area_ratio, cluster_risk}
        cluster_risk = True when density suggests systemic failure
    """
    count = len(detections)
    structural_count = sum(
        1 for d in detections if d["class_name"] in STRUCTURAL_FAILURE_TYPES
    )
    total_area = sum(d.get("area_pixels", 0) for d in detections)
    total_area_ratio = total_area / image_area if image_area > 0 else 0.0

    # Systemic failure: 6+ defects OR 3+ structural defects OR >15% total area
    cluster_risk = (
        count >= 6
        or structural_count >= 3
        or total_area_ratio > 0.15
    )

    return {
        "count": count,
        "structural_count": structural_count,
        "total_area_ratio": total_area_ratio,
        "cluster_risk": cluster_risk,
    }


def classify_safety_priority(
    class_name: str,
    severity: str,
    severity_score: float,
    area_ratio: float,
    cluster_risk: bool,
) -> tuple[str, str]:
    """Classify safety priority — what to fix FIRST.

    This is SEPARATE from severity. Severity = "how bad is it".
    Priority = "what order do we fix things in" based on safety risk.

    Rules (in order of precedence):
      1. ALL potholes → CRITICAL (any size causes immediate vehicle/pedestrian danger)
      2. High severity structural failure → CRITICAL
      3. Cluster risk + high severity → CRITICAL (systemic failure)
      4. High severity non-structural → URGENT
      5. Medium severity structural → URGENT
      6. Medium severity non-structural → MONITOR
      7. Everything else → ROUTINE

    Returns:
        (priority_key, recommended_action)
    """
    is_structural = class_name in STRUCTURAL_FAILURE_TYPES
    is_pothole = class_name in {"D40", "Pothole"}

    # Rule 1: ALL potholes are CRITICAL — non-negotiable safety rule
    if is_pothole:
        return ("critical", "Emergency repair: pothole poses immediate danger to all road users")

    # Rule 2: High severity structural failure
    if severity == "high" and is_structural:
        return ("critical", "Emergency repair: severe structural failure requires immediate attention")

    # Rule 3: Systemic failure with high severity
    if cluster_risk and severity == "high":
        return ("critical", "Section rehabilitation: systemic pavement failure detected")

    # Rule 4: High severity non-structural
    if severity == "high":
        return ("urgent", "Priority repair: significant defect requiring prompt maintenance")

    # Rule 5: Medium severity structural
    if severity == "medium" and is_structural:
        return ("urgent", "Schedule repair: structural defect will deteriorate rapidly if untreated")

    # Rule 6: Medium severity non-structural
    if severity == "medium":
        return ("monitor", "Include in next maintenance cycle; monitor for progression")

    # Rule 7: Low severity
    return ("routine", "Log and monitor at next scheduled inspection")


def assess_deterioration_risk(
    class_name: str,
    severity: str,
    area_ratio: float,
) -> dict:
    """Assess how quickly this defect will worsen if left untreated.

    Based on ASTM D6433 deterioration curves and field engineering data.

    Returns:
        {risk_level, warning, estimated_escalation}
    """
    warning = RAPID_DETERIORATION_TYPES.get(class_name, "Monitor for changes")

    is_structural = class_name in STRUCTURAL_FAILURE_TYPES

    if is_structural and severity == "high":
        return {
            "risk_level": "rapid",
            "warning": warning,
            "estimated_escalation": "Days to weeks — active safety hazard escalating",
        }
    elif is_structural and severity == "medium":
        return {
            "risk_level": "accelerating",
            "warning": warning,
            "estimated_escalation": "Weeks to months — will reach critical state",
        }
    elif severity == "high":
        return {
            "risk_level": "moderate",
            "warning": warning,
            "estimated_escalation": "Months — seasonal cycles will worsen damage",
        }
    elif severity == "medium" and area_ratio > 0.02:
        return {
            "risk_level": "slow",
            "warning": warning,
            "estimated_escalation": "6-18 months — gradual progression",
        }
    else:
        return {
            "risk_level": "stable",
            "warning": warning,
            "estimated_escalation": "Monitor — may remain stable with proper drainage",
        }


def classify_all_detections(detections: list[dict], image_shape: tuple) -> list[dict]:
    """Batch classifier: enrich all detections with severity, priority, and risk.

    This is the TWO-PASS architecture entry point. Called AFTER extraction
    and deduplication, when we have the complete detection set and can
    compute density-dependent features.

    Mutates detection dicts in-place and returns them.
    """
    if not detections:
        return detections

    img_h, img_w = image_shape[:2]
    image_area = img_h * img_w

    # --- Pass 1: Compute per-detection geometry (width, aspect ratio) ---
    for det in detections:
        det["width_pixels"] = estimate_crack_width_pixels(det.get("mask"))
        det["aspect_ratio"] = compute_aspect_ratio(det.get("mask"))

    # --- Density analysis (needs full detection count) ---
    density = compute_defect_density(detections, image_area)
    defect_count = density["count"]
    cluster_risk = density["cluster_risk"]

    # --- Pass 2: Classify each detection with full context ---
    for det in detections:
        area_ratio = det["area_pixels"] / image_area if image_area > 0 else 0.0

        # Multi-factor severity scoring
        severity_level, severity_score = classify_severity(
            area_pixels=det["area_pixels"],
            image_area=image_area,
            class_name=det["class_name"],
            confidence=det["confidence"],
            mask=det.get("mask"),
            defect_count=defect_count,
            _precomputed_width_px=det["width_pixels"],
        )
        det["severity"] = severity_level
        det["severity_score"] = severity_score

        # Safety priority classification
        priority_key, recommended_action = classify_safety_priority(
            class_name=det["class_name"],
            severity=severity_level,
            severity_score=severity_score,
            area_ratio=area_ratio,
            cluster_risk=cluster_risk,
        )
        det["safety_priority"] = priority_key
        det["recommended_action"] = recommended_action

        # Capture originals so the inspector-override layer can toggle back
        # to AI suggestion without re-running classification.
        det["original_safety_priority"] = priority_key
        det["original_recommended_action"] = recommended_action

        # Deterioration risk assessment
        det["deterioration_risk"] = assess_deterioration_risk(
            class_name=det["class_name"],
            severity=severity_level,
            area_ratio=area_ratio,
        )

    return detections


def _extract_detections(result, image_shape: tuple) -> list[dict]:
    """Extract raw detections from a single YOLO result object.

    Two-pass architecture: extracts geometry ONLY — no severity classification.
    Severity is computed later by classify_all_detections() once we have the
    complete detection set (needed for density analysis).
    """
    detections = []

    if result.boxes is None:
        return detections

    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        bbox = box.xyxy[0].cpu().numpy().tolist()

        # Extract mask if segmentation model
        mask = None
        area_pixels = 0
        if result.masks is not None and i < len(result.masks):
            mask = result.masks[i].data.cpu().numpy().squeeze()
            mask = (mask > 0.5).astype(np.uint8)
            area_pixels = int(mask.sum())
        else:
            # Detection model: use bounding box area
            area_pixels = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        detections.append({
            "class_name": cls_name,
            "confidence": conf,
            "bbox": bbox,
            "mask": mask,
            "area_pixels": area_pixels,
        })

    return detections


def deduplicate_detections(all_detections: list[dict]) -> list[dict]:
    """Remove duplicate detections from multiple models using IoU.

    When two detections overlap > IOU_THRESHOLD:
    - Keep the specific class name over generic "crack"
    - Keep the higher confidence
    - Keep the mask if one exists
    """
    if not all_detections:
        return []

    # Sort by confidence descending so we keep the best ones
    sorted_dets = sorted(all_detections, key=lambda d: d["confidence"], reverse=True)
    keep = []
    suppressed = [False] * len(sorted_dets)

    for i in range(len(sorted_dets)):
        if suppressed[i]:
            continue

        best = sorted_dets[i]

        for j in range(i + 1, len(sorted_dets)):
            if suppressed[j]:
                continue

            iou = compute_iou(best["bbox"], sorted_dets[j]["bbox"])
            if iou > IOU_THRESHOLD:
                # Merge: prefer specific class name over generic "crack"
                other = sorted_dets[j]
                if best["class_name"].lower() == "crack" and other["class_name"].lower() != "crack":
                    best["class_name"] = other["class_name"]
                # Keep mask from segmentation model if available
                if best["mask"] is None and other["mask"] is not None:
                    best["mask"] = other["mask"]
                    best["area_pixels"] = other["area_pixels"]

                suppressed[j] = True

        keep.append(best)

    return keep


def _harmonize_class_labels(
    detections_by_model: dict[str, list[dict]],
    iou_threshold: float = WBF_LABEL_HARMONIZE_IOU,
) -> dict[str, list[dict]]:
    """Relabel generic 'crack' detections to a specific class when they overlap one.

    Without this, WBF can't fuse a seg-model 'crack' detection with the same
    physical defect from the detection model labelled 'Pothole' — they have
    different label IDs in WBF's eyes and stay as two separate detections.
    The IoU-dedup path handles this implicitly by preferring specific names.
    """
    specific_dets = [
        d for dets in detections_by_model.values()
        for d in dets if d["class_name"].lower() != "crack"
    ]
    if not specific_dets:
        return detections_by_model

    harmonized: dict[str, list[dict]] = {}
    for model_name, dets in detections_by_model.items():
        new_dets = []
        for d in dets:
            new_d = dict(d)
            if new_d["class_name"].lower() == "crack":
                best_iou = 0.0
                best_specific = None
                for s in specific_dets:
                    iou = compute_iou(new_d["bbox"], s["bbox"])
                    if iou > best_iou and iou > iou_threshold:
                        best_iou = iou
                        best_specific = s
                if best_specific is not None:
                    new_d["class_name"] = best_specific["class_name"]
            new_dets.append(new_d)
        harmonized[model_name] = new_dets
    return harmonized


def fuse_detections_wbf(
    detections_by_model: dict[str, list[dict]],
    image_shape: tuple,
    iou_threshold: float = WBF_IOU_THRESHOLD,
) -> list[dict]:
    """Merge cross-model detections using Weighted Box Fusion.

    WBF averages overlapping boxes weighted by confidence rather than dropping
    the loser like IoU-dedup. Typically yields +1-3 mAP on multi-model ensembles
    when both models partially see the same defect.

    Masks aren't preserved by WBF itself — we recover the closest-IoU original
    mask so downstream width estimation and area computation still work.
    """
    from ensemble_boxes import weighted_boxes_fusion

    H, W = image_shape[:2]
    if H == 0 or W == 0:
        return []

    detections_by_model = _harmonize_class_labels(detections_by_model)

    all_class_names: set[str] = set()
    for dets in detections_by_model.values():
        for d in dets:
            all_class_names.add(d["class_name"])
    if not all_class_names:
        return []

    class_to_id = {name: i for i, name in enumerate(sorted(all_class_names))}
    id_to_class = {i: name for name, i in class_to_id.items()}

    boxes_list: list[list[list[float]]] = []
    scores_list: list[list[float]] = []
    labels_list: list[list[int]] = []
    flat_originals: list[dict] = []

    for model_name in detections_by_model:
        dets = detections_by_model[model_name]
        boxes: list[list[float]] = []
        scores: list[float] = []
        labels: list[int] = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            nx1 = max(0.0, min(1.0, x1 / W))
            ny1 = max(0.0, min(1.0, y1 / H))
            nx2 = max(0.0, min(1.0, x2 / W))
            ny2 = max(0.0, min(1.0, y2 / H))
            if nx2 <= nx1 or ny2 <= ny1:
                continue  # degenerate box, skip
            boxes.append([nx1, ny1, nx2, ny2])
            scores.append(float(d["confidence"]))
            labels.append(class_to_id[d["class_name"]])
            flat_originals.append(d)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    if not any(boxes_list):
        return []

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=iou_threshold,
        skip_box_thr=WBF_SKIP_BOX_THRESHOLD,
        conf_type="avg",
    )

    merged: list[dict] = []
    for fb, fs, fl in zip(fused_boxes, fused_scores, fused_labels):
        bbox = [float(fb[0]) * W, float(fb[1]) * H, float(fb[2]) * W, float(fb[3]) * H]
        class_name = id_to_class[int(fl)]

        # Recover mask + area from the closest original detection of the same class
        best_match = None
        best_iou = 0.0
        for orig in flat_originals:
            if orig["class_name"] != class_name:
                continue
            iou = compute_iou(bbox, orig["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = orig

        if best_match is not None and best_match.get("mask") is not None:
            mask = best_match["mask"]
            area_pixels = best_match.get("area_pixels", int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        else:
            mask = None
            area_pixels = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        merged.append({
            "class_name": class_name,
            "confidence": float(fs),
            "bbox": bbox,
            "mask": mask,
            "area_pixels": area_pixels,
        })

    return merged


# ---------------------------------------------------------------------------
# SAHI tiled inference
# ---------------------------------------------------------------------------

def _create_sahi_detection_model(model_path: str):
    """Create a SAHI AutoDetectionModel from a YOLO model path.

    SAHI uses model_type="yolov8" for both YOLOv8 and YOLO11
    (same ultralytics predict API under the hood).
    """
    from sahi import AutoDetectionModel

    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device="cuda:0" if _cuda_available() else "cpu",
    )


def _cuda_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_sahi_inference(
    model_path: str,
    image: np.ndarray,
    confidence: float = CONFIDENCE_THRESHOLD,
    slice_size: int = DEFAULT_SAHI_SLICE_SIZE,
    overlap_ratio: float = DEFAULT_SAHI_OVERLAP_RATIO,
) -> list[dict]:
    """Run SAHI tiled inference on a single image.

    Slices the image into overlapping tiles, runs detection on each,
    and merges results with NMS. Returns list of detection dicts.

    Note: SAHI does not support segmentation mask merging across tiles,
    so all results are bounding-box only (mask=None).
    """
    from sahi.predict import get_sliced_prediction

    sahi_model = _create_sahi_detection_model(model_path)
    sahi_model.confidence_threshold = confidence

    result = get_sliced_prediction(
        image=image,
        detection_model=sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        verbose=0,
    )

    return _convert_sahi_results(result, image.shape)


def _convert_sahi_results(sahi_result, image_shape: tuple) -> list[dict]:
    """Convert SAHI prediction results to our raw detection dict format.

    Two-pass architecture: no severity here — classified later by
    classify_all_detections(). SAHI results never have masks.
    """
    detections = []

    for pred in sahi_result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        conf = pred.score.value
        cls_name = pred.category.name
        area_pixels = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        detections.append({
            "class_name": cls_name,
            "confidence": conf,
            "bbox": bbox,
            "mask": None,  # SAHI doesn't merge masks across tiles
            "area_pixels": area_pixels,
        })

    return detections


# ---------------------------------------------------------------------------
# Main inference functions
# ---------------------------------------------------------------------------

def run_inference(
    model: YOLO,
    image: np.ndarray,
    confidence: float = CONFIDENCE_THRESHOLD,
    imgsz: int = DEFAULT_IMGSZ,
    augment: bool = False,
    model_name: str = "single",
) -> dict:
    """Run crack detection on a single image with one model.

    Two-pass architecture: extract raw detections, then classify batch
    with full context (density, width, priority).

    Returns dict with "annotated_image", "detections", and "inference_time_ms".
    """
    t0 = time.perf_counter()
    results = model.predict(image, conf=confidence, imgsz=imgsz, augment=augment, verbose=False)
    inference_ms = (time.perf_counter() - t0) * 1000

    result = results[0]
    detections = _extract_detections(result, image.shape)

    # Stamp single-model provenance so clustering downstream has a consistent shape
    for det in detections:
        det["models_agreeing"] = [model_name]

    # Two-pass: classify all detections with full context
    classify_all_detections(detections, image.shape)

    # Phase 1: spatial clustering for the cluster-card UI. Imported locally to
    # avoid the model<->clustering import cycle.
    from .clustering import cluster_detections
    clusters = cluster_detections(detections)

    # Two annotated images: raw boxes (Compare mode + legacy callers) and
    # cluster outlines (Phase 1 default per-image card view).
    annotated = _draw_detections(image.copy(), detections)
    annotated_clusters = _draw_clusters(image.copy(), detections, clusters)

    return {
        "annotated_image": annotated,
        "annotated_clusters": annotated_clusters,
        "detections": detections,
        "clusters": clusters,
        "inference_time_ms": inference_ms,
    }


def _track_model_agreement(
    merged: list[dict],
    detections_by_model: dict[str, list[dict]],
    iou_threshold: float = 0.4,
) -> None:
    """Stamp each merged detection with the list of models whose original
    detections overlap it at IoU >= threshold.

    Multi-model agreement is the structural trust signal we surface in cluster
    cards — independent agreement across 3 sub-models is far more meaningful
    than any single confidence number.
    """
    for det in merged:
        agreeing = []
        for model_name, originals in detections_by_model.items():
            for orig in originals:
                if compute_iou(det["bbox"], orig["bbox"]) >= iou_threshold:
                    agreeing.append(model_name)
                    break
        det["models_agreeing"] = sorted(set(agreeing))


def run_ensemble_inference(
    models: dict[str, YOLO],
    image: np.ndarray,
    confidence: float = CONFIDENCE_THRESHOLD,
    imgsz: int = DEFAULT_IMGSZ,
    augment: bool = False,
    use_sahi: bool = False,
    sahi_slice_size: int = DEFAULT_SAHI_SLICE_SIZE,
    model_paths: dict[str, str] | None = None,
    fusion_method: str = "iou_dedup",
) -> dict:
    """Run multiple models on the same image and merge results.

    Args:
        models: dict of {model_name: loaded YOLO model}
        image: BGR image array
        confidence: Detection confidence threshold
        imgsz: Inference resolution
        augment: Enable TTA
        use_sahi: Enable SAHI tiled inference
        sahi_slice_size: Tile size for SAHI slicing
        model_paths: dict of {model_name: file path} (required when use_sahi=True)
        fusion_method: "iou_dedup" (default) keeps the highest-confidence overlapping
            detection; "wbf" averages overlapping boxes weighted by confidence.

    Returns dict with:
        - "annotated_image": image with all detection overlays
        - "detections": deduplicated list of detections from all models
        - "per_model_counts": {model_name: count} for UI display
        - "inference_time_ms": total inference time
    """
    t0 = time.perf_counter()
    detections_by_model: dict[str, list[dict]] = {}
    per_model_counts = {}

    for model_name, model in models.items():
        if use_sahi and model_paths and model_name in model_paths:
            # SAHI tiled inference — catches small cracks in high-res images
            dets = run_sahi_inference(
                model_path=model_paths[model_name],
                image=image,
                confidence=confidence,
                slice_size=sahi_slice_size,
            )
        else:
            # Standard inference with resolution and TTA
            results = model.predict(image, conf=confidence, imgsz=imgsz, augment=augment, verbose=False)
            result = results[0]
            dets = _extract_detections(result, image.shape)

        per_model_counts[model_name] = len(dets)
        detections_by_model[model_name] = dets

    # Merge cross-model detections using selected fusion strategy
    if fusion_method == "wbf":
        merged = fuse_detections_wbf(detections_by_model, image.shape)
    else:
        flat = [d for dets in detections_by_model.values() for d in dets]
        merged = deduplicate_detections(flat)

    # Tag each merged detection with the set of models that fired on its region.
    # Multi-model agreement is the headline trust signal for cluster cards.
    _track_model_agreement(merged, detections_by_model)

    # Two-pass: classify all merged detections with full context
    classify_all_detections(merged, image.shape)

    # Phase 1: spatial clustering for the cluster-card UI. Imported locally to
    # avoid the model<->clustering import cycle.
    from .clustering import cluster_detections
    clusters = cluster_detections(merged)

    # Two annotated images: raw boxes (Compare mode side-by-side) and cluster
    # outlines (Phase 1 default per-image card view).
    annotated = _draw_detections(image.copy(), merged)
    annotated_clusters = _draw_clusters(image.copy(), merged, clusters)

    inference_ms = (time.perf_counter() - t0) * 1000

    return {
        "annotated_image": annotated,
        "annotated_clusters": annotated_clusters,
        "detections": merged,
        "clusters": clusters,
        "per_model_counts": per_model_counts,
        "inference_time_ms": inference_ms,
    }


def _draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes, labels, and priority indicators on an image.

    Uses safety priority colors when available, falls back to severity colors.
    Critical/urgent defects get thicker borders for visual emphasis.
    """
    severity_colors = {
        "low": (0, 200, 0),       # green
        "medium": (0, 200, 255),   # yellow
        "high": (0, 0, 255),       # red
    }

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Priority color if available, else severity color
        priority_key = det.get("safety_priority")
        if priority_key and priority_key in SAFETY_PRIORITIES:
            color = SAFETY_PRIORITIES[priority_key]["color_bgr"]
        else:
            color = severity_colors.get(det.get("severity", "low"), (200, 200, 200))

        # Thicker borders for critical/urgent
        box_thickness = 3 if priority_key in ("critical", "urgent") else 2

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

        # Build label: "ClassName 85% [CRITICAL]"
        priority_tag = ""
        if priority_key and priority_key in SAFETY_PRIORITIES:
            priority_tag = f" [{SAFETY_PRIORITIES[priority_key]['label']}]"
        label = f"{det['class_name']} {det['confidence']:.0%}{priority_tag}"

        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Draw mask overlay if available
        if det.get("mask") is not None:
            mask_resized = cv2.resize(
                det["mask"], (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            overlay = image.copy()
            overlay[mask_resized > 0] = color
            image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            contours, _ = cv2.findContours(
                mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            outline_thickness = 3 if priority_key in ("critical", "urgent") else 2
            cv2.drawContours(image, contours, -1, color, outline_thickness)

    return image


# Neutral cluster outline color (BGR). Severity-keyed colors are intentionally
# NOT used here — Phase 1 decouples severity from default display.
CLUSTER_OUTLINE_BGR = (51, 153, 255)  # orange in BGR


def _draw_clusters(
    image: np.ndarray,
    detections: list[dict],
    clusters: list[dict],
) -> np.ndarray:
    """Draw one neutral-orange union outline per cluster with a header label.

    Phase 1 default render mode. Per-detection boxes are intentionally omitted
    so the inspector sees ~5-8 candidate regions instead of 30+ overlapping
    boxes. Severity color-coding is left out by design — the inspector
    classifies severity manually via the cluster card UI.
    """
    if not clusters:
        return image

    color = CLUSTER_OUTLINE_BGR
    for cluster in clusters:
        x1, y1, x2, y2 = (int(v) for v in cluster["bbox_union"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        n_models = len(cluster.get("models_agreeing", []))
        n_signals = cluster.get("detection_count", 0)
        cid = cluster.get("cluster_id", 0)
        label = f"Region {cid + 1} | {n_signals} signals | {n_models} models"

        font_scale = 0.55
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            image, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
        )

    return image


def get_severity_color(severity: str) -> tuple:
    """Return BGR color for a severity level."""
    colors = {
        "low": (0, 200, 0),       # green
        "medium": (0, 200, 255),   # yellow
        "high": (0, 0, 255),       # red
    }
    return colors.get(severity, (200, 200, 200))


def summarize_detections(detections: list[dict]) -> dict:
    """Produce a summary of detection results including safety priority breakdown."""
    if not detections:
        return {
            "total_detections": 0,
            "by_severity": {"low": 0, "medium": 0, "high": 0},
            "by_priority": {"critical": 0, "urgent": 0, "monitor": 0, "routine": 0},
            "by_class": {},
            "avg_confidence": 0.0,
            "cluster_risk": False,
            "has_structural_failure": False,
        }

    by_severity = {"low": 0, "medium": 0, "high": 0}
    by_priority = {"critical": 0, "urgent": 0, "monitor": 0, "routine": 0}
    by_class: dict[str, int] = {}
    has_structural = False

    for d in detections:
        by_severity[d.get("severity", "low")] += 1
        priority = d.get("safety_priority", "routine")
        by_priority[priority] = by_priority.get(priority, 0) + 1
        by_class[d["class_name"]] = by_class.get(d["class_name"], 0) + 1
        if d["class_name"] in STRUCTURAL_FAILURE_TYPES:
            has_structural = True

    # Cluster risk: 6+ defects or 3+ structural
    structural_count = sum(1 for d in detections if d["class_name"] in STRUCTURAL_FAILURE_TYPES)
    cluster_risk = len(detections) >= 6 or structural_count >= 3

    return {
        "total_detections": len(detections),
        "by_severity": by_severity,
        "by_priority": by_priority,
        "by_class": by_class,
        "avg_confidence": sum(d["confidence"] for d in detections) / len(detections),
        "cluster_risk": cluster_risk,
        "has_structural_failure": has_structural,
    }
