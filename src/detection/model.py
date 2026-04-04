"""YOLOv8 crack detection model loading and inference."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Where trained models live
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Custom-trained model names (checked in order for single-model mode)
CUSTOM_MODELS = [
    "pavescan_crack_seg.pt",  # 1-class segmentation (crack) — best on close-up images
    "pavescan_rdd2022.pt",    # 4-class detection (longitudinal, transverse, alligator, pothole)
]

# Fallback: pretrained COCO model (detects generic objects, not cracks)
FALLBACK_MODEL = "yolov8n-seg.pt"

CONFIDENCE_THRESHOLD = 0.15

# Area-based severity thresholds (percentage of image area)
SEVERITY_AREA_THRESHOLDS = {
    "high": 0.05,       # > 5% of image = high severity
    "medium": 0.01,     # > 1% of image = medium severity
}

# Crack types that are inherently more severe (lower area threshold for "high")
SEVERE_CLASSES = {"Pothole", "Alligator_Crack"}
SEVERE_CLASS_HIGH_THRESHOLD = 0.02  # > 2% of image = high for these classes

# IoU threshold for deduplication in ensemble mode
IOU_THRESHOLD = 0.5


def load_model(model_path: str | None = None) -> YOLO:
    """Load a single YOLOv8 model for crack detection.

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


def load_ensemble_models() -> dict[str, YOLO]:
    """Load all custom-trained models for ensemble inference.

    Returns dict mapping model name to loaded YOLO model.
    Only loads models that exist in the models/ directory.
    """
    models = {}
    for name in CUSTOM_MODELS:
        path = MODELS_DIR / name
        if path.exists():
            models[name] = YOLO(str(path))
    return models


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


def classify_severity(area_pixels: int, image_area: int, class_name: str, confidence: float) -> str:
    """Classify defect severity based on area relative to image size.

    Large defects = high severity. Potholes and alligator cracks get
    bumped up because they're structurally more dangerous.
    """
    if image_area == 0:
        return "low"

    area_ratio = area_pixels / image_area

    # Very uncertain detections capped at medium
    if confidence < 0.2:
        if area_ratio > SEVERITY_AREA_THRESHOLDS["medium"]:
            return "medium"
        return "low"

    # Potholes and alligator cracks are inherently more severe
    if class_name in SEVERE_CLASSES:
        if area_ratio > SEVERE_CLASS_HIGH_THRESHOLD:
            return "high"

    if area_ratio > SEVERITY_AREA_THRESHOLDS["high"]:
        return "high"
    elif area_ratio > SEVERITY_AREA_THRESHOLDS["medium"]:
        return "medium"

    return "low"


def _extract_detections(result, image_shape: tuple) -> list[dict]:
    """Extract detections from a single YOLO result object.

    Works for both detection and segmentation models.
    """
    img_h, img_w = image_shape[:2]
    image_area = img_h * img_w
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

        severity = classify_severity(area_pixels, image_area, cls_name, conf)

        detections.append({
            "class_name": cls_name,
            "confidence": conf,
            "bbox": bbox,
            "mask": mask,
            "severity": severity,
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


def run_inference(
    model: YOLO,
    image: np.ndarray,
    confidence: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """Run crack detection on a single image with one model.

    Returns dict with "annotated_image" and "detections" list.
    """
    results = model.predict(image, conf=confidence, verbose=False)
    result = results[0]

    detections = _extract_detections(result, image.shape)
    annotated = result.plot()

    return {
        "annotated_image": annotated,
        "detections": detections,
    }


def run_ensemble_inference(
    models: dict[str, YOLO],
    image: np.ndarray,
    confidence: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """Run multiple models on the same image and merge results.

    Returns dict with:
        - "annotated_image": image with all detection overlays
        - "detections": deduplicated list of detections from all models
        - "per_model_counts": {model_name: count} for UI display
    """
    all_detections = []
    per_model_counts = {}
    annotated = image.copy()

    for model_name, model in models.items():
        results = model.predict(image, conf=confidence, verbose=False)
        result = results[0]

        dets = _extract_detections(result, image.shape)
        per_model_counts[model_name] = len(dets)
        all_detections.extend(dets)

        # Use the last model's annotated image as base (we'll re-draw below)

    # Deduplicate overlapping detections
    merged = deduplicate_detections(all_detections)

    # Draw all merged detections on the image
    annotated = _draw_detections(image.copy(), merged)

    return {
        "annotated_image": annotated,
        "detections": merged,
        "per_model_counts": per_model_counts,
    }


def _draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on an image."""
    severity_colors = {
        "low": (0, 200, 0),       # green
        "medium": (0, 200, 255),   # yellow
        "high": (0, 0, 255),       # red
    }

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = severity_colors.get(det["severity"], (200, 200, 200))

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{det['class_name']} {det['confidence']:.0%}"
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Draw mask overlay if available
        if det["mask"] is not None:
            mask_resized = cv2.resize(det["mask"], (image.shape[1], image.shape[0]))
            overlay = image.copy()
            overlay[mask_resized > 0] = color
            image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

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
    """Produce a summary of detection results."""
    if not detections:
        return {
            "total_detections": 0,
            "by_severity": {"low": 0, "medium": 0, "high": 0},
            "by_class": {},
            "avg_confidence": 0.0,
        }

    by_severity = {"low": 0, "medium": 0, "high": 0}
    by_class: dict[str, int] = {}

    for d in detections:
        by_severity[d["severity"]] += 1
        by_class[d["class_name"]] = by_class.get(d["class_name"], 0) + 1

    return {
        "total_detections": len(detections),
        "by_severity": by_severity,
        "by_class": by_class,
        "avg_confidence": sum(d["confidence"] for d in detections) / len(detections),
    }
