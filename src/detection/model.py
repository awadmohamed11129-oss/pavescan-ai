"""YOLOv8 crack detection model loading and inference."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Where trained models live
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Custom-trained model names (checked in order)
CUSTOM_MODELS = [
    "pavescan_rdd2022.pt",    # 4-class detection (longitudinal, transverse, alligator, pothole)
    "pavescan_crack_seg.pt",  # 1-class segmentation (crack)
]

# Fallback: pretrained COCO model (detects generic objects, not cracks)
FALLBACK_MODEL = "yolov8n-seg.pt"

CONFIDENCE_THRESHOLD = 0.25

# Crack severity classification based on detection confidence and area
SEVERITY_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8,
}


def load_model(model_path: str | None = None) -> YOLO:
    """Load a YOLOv8 model for crack detection.

    Searches for models in this order:
    1. Explicit model_path (if provided)
    2. Custom-trained models in models/ directory
    3. Fallback pretrained model

    Args:
        model_path: Path to a specific .pt file. If None, auto-detects.
    """
    if model_path and Path(model_path).exists():
        return YOLO(model_path)

    # Search for custom-trained models
    for name in CUSTOM_MODELS:
        path = MODELS_DIR / name
        if path.exists():
            return YOLO(str(path))

    # Fallback: check if pretrained model is in models/ dir
    fallback_path = MODELS_DIR / FALLBACK_MODEL
    if fallback_path.exists():
        return YOLO(str(fallback_path))

    # Last resort: download pretrained model
    return YOLO(FALLBACK_MODEL)


def run_inference(
    model: YOLO,
    image: np.ndarray,
    confidence: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """Run crack detection on a single image.

    Args:
        model: Loaded YOLO model.
        image: BGR image as numpy array (from cv2.imread or uploaded file).
        confidence: Minimum confidence threshold for detections.

    Returns:
        dict with keys:
            - "annotated_image": image with detection overlays drawn
            - "detections": list of dicts with keys:
                - "class_name": str
                - "confidence": float
                - "bbox": [x1, y1, x2, y2]
                - "mask": binary mask (numpy array) or None
                - "severity": "low" | "medium" | "high"
                - "area_pixels": int (number of pixels in the mask)
    """
    results = model.predict(image, conf=confidence, verbose=False)
    result = results[0]

    detections = []

    if result.boxes is not None:
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

            # Classify severity based on confidence
            severity = "low"
            if conf >= SEVERITY_THRESHOLDS["high"]:
                severity = "high"
            elif conf >= SEVERITY_THRESHOLDS["medium"]:
                severity = "medium"

            detections.append({
                "class_name": cls_name,
                "confidence": conf,
                "bbox": bbox,
                "mask": mask,
                "severity": severity,
                "area_pixels": area_pixels,
            })

    # Get the annotated image with overlays
    annotated = result.plot()

    return {
        "annotated_image": annotated,
        "detections": detections,
    }


def get_severity_color(severity: str) -> tuple:
    """Return BGR color for a severity level."""
    colors = {
        "low": (0, 200, 0),       # green
        "medium": (0, 200, 255),   # yellow
        "high": (0, 0, 255),       # red
    }
    return colors.get(severity, (200, 200, 200))


def summarize_detections(detections: list[dict]) -> dict:
    """Produce a summary of detection results.

    Returns dict with:
        - total_detections: int
        - by_severity: {"low": int, "medium": int, "high": int}
        - by_class: {class_name: int}
        - avg_confidence: float
    """
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
