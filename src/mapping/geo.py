"""GPS extraction and geo-mapping utilities for PaveScan AI.

Handles:
- Extracting GPS coordinates from drone image EXIF data
- Generating demo coordinates when images lack GPS info
- Building marker and heatmap data for the Map page
"""

import io
import math
import random
from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


# Demo mode: University of Toronto / downtown Toronto area
DEMO_CENTER = (43.6605, -79.3955)

SEVERITY_COLORS = {
    "low": "#00C853",
    "medium": "#FFD600",
    "high": "#FF1744",
}

PRIORITY_COLORS = {
    "critical": "#FF1744",
    "urgent": "#FF9100",
    "monitor": "#FFC400",
    "routine": "#00C853",
}


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """Convert GPS coordinates from degrees/minutes/seconds to decimal degrees.

    Args:
        dms: Tuple of (degrees, minutes, seconds) as IFDRational values.
        ref: Reference direction ('N', 'S', 'E', 'W').
    """
    degrees = float(dms[0])
    minutes = float(dms[1])
    seconds = float(dms[2])

    decimal = degrees + minutes / 60.0 + seconds / 3600.0

    if ref in ("S", "W"):
        decimal = -decimal

    return decimal


def extract_gps_from_bytes(file_bytes: bytes) -> tuple[float, float] | None:
    """Extract GPS latitude/longitude from image EXIF data.

    Args:
        file_bytes: Raw bytes of the image file.

    Returns:
        (latitude, longitude) as decimal degrees, or None if no GPS data found.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        exif_data = img._getexif()

        if not exif_data:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag_name] = gps_value
                break

        if not gps_info:
            return None

        if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
            return None

        lat = _dms_to_decimal(
            gps_info["GPSLatitude"],
            gps_info.get("GPSLatitudeRef", "N"),
        )
        lon = _dms_to_decimal(
            gps_info["GPSLongitude"],
            gps_info.get("GPSLongitudeRef", "W"),
        )

        return (lat, lon)

    except Exception:
        return None


def generate_demo_coordinates(
    num_points: int,
    center: tuple[float, float] = DEMO_CENTER,
    spread: float = 0.003,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Generate realistic road-like coordinates for demo mode.

    Creates a path that simulates a drone flying along a road,
    with small random offsets to look natural.

    Args:
        num_points: Number of coordinate pairs to generate.
        center: (lat, lon) center point for the demo area.
        spread: How far points spread from center (in degrees, ~0.001 = 111m).
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)

    coords = []
    lat, lon = center

    # Start slightly offset from center
    current_lat = lat - spread / 2
    current_lon = lon - spread / 2

    for i in range(num_points):
        # Move mostly in one direction (simulating road travel)
        # with small perpendicular jitter
        progress = i / max(num_points - 1, 1)
        current_lat = lat - spread / 2 + progress * spread + rng.gauss(0, spread * 0.05)
        current_lon = lon - spread / 2 + progress * spread * 0.6 + rng.gauss(0, spread * 0.03)

        coords.append((current_lat, current_lon))

    return coords


def assign_coordinates_to_results(
    detection_results: list[dict],
    uploaded_files: list,
) -> tuple[list[dict], bool]:
    """Assign GPS coordinates to detection results.

    Tries to extract GPS from each uploaded image's EXIF data.
    Falls back to demo coordinates if no images have GPS info.

    Args:
        detection_results: List of {"filename": str, "result": {...}} from detection page.
        uploaded_files: List of uploaded file objects (with .read() and .name).

    Returns:
        Tuple of (geo_results, is_demo_mode):
        - geo_results: List of {"filename", "result", "lat", "lon"} dicts.
        - is_demo_mode: True if demo coordinates were used.
    """
    geo_results = []
    has_real_gps = False

    # Build a lookup of filename -> file bytes
    file_bytes_map = {}
    for f in uploaded_files:
        try:
            f.seek(0)
            file_bytes_map[f.name] = f.read()
            f.seek(0)
        except Exception:
            pass

    # Try GPS extraction for each result
    for item in detection_results:
        filename = item["filename"]
        coords = None

        if filename in file_bytes_map:
            coords = extract_gps_from_bytes(file_bytes_map[filename])

        if coords:
            has_real_gps = True
            geo_results.append({
                "filename": filename,
                "result": item["result"],
                "lat": coords[0],
                "lon": coords[1],
            })
        else:
            # Placeholder — will be replaced with demo coords below if needed
            geo_results.append({
                "filename": filename,
                "result": item["result"],
                "lat": None,
                "lon": None,
            })

    # If no real GPS data, assign demo coordinates
    is_demo = not has_real_gps
    if is_demo:
        demo_coords = generate_demo_coordinates(len(geo_results))
        for i, item in enumerate(geo_results):
            item["lat"] = demo_coords[i][0]
            item["lon"] = demo_coords[i][1]

    return geo_results, is_demo


def build_marker_data(geo_results: list[dict]) -> list[dict]:
    """Flatten geo results into individual marker records for the map.

    Each detection in each image becomes one marker.

    Returns:
        List of dicts with keys: lat, lon, class_name, confidence, severity, filename, area_pixels.
    """
    markers = []

    for item in geo_results:
        lat = item["lat"]
        lon = item["lon"]
        filename = item["filename"]
        detections = item["result"]["detections"]

        if not detections:
            # Still place a marker for images with no detections
            markers.append({
                "lat": lat,
                "lon": lon,
                "class_name": "No defects",
                "confidence": 0.0,
                "severity": "low",
                "filename": filename,
                "area_pixels": 0,
            })
        else:
            for det in detections:
                # Add small random offset so overlapping markers are visible
                offset_lat = random.gauss(0, 0.00005)
                offset_lon = random.gauss(0, 0.00005)

                markers.append({
                    "lat": lat + offset_lat,
                    "lon": lon + offset_lon,
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "severity": det.get("severity", "low"),
                    "filename": filename,
                    "area_pixels": det.get("area_pixels", 0),
                    "safety_priority": det.get("safety_priority", "routine"),
                    "recommended_action": det.get("recommended_action", ""),
                    "severity_score": det.get("severity_score", 0.0),
                    "width_pixels": det.get("width_pixels", 0.0),
                    "deterioration_risk": det.get("deterioration_risk", {}),
                })

    return markers


def compute_heatmap_data(markers: list[dict]) -> list[list[float]]:
    """Convert marker data into [lat, lon, weight] triples for Folium HeatMap.

    Weight is based on confidence score — higher confidence = hotter spot.
    """
    return [
        [m["lat"], m["lon"], m["confidence"]]
        for m in markers
        if m["confidence"] > 0
    ]
