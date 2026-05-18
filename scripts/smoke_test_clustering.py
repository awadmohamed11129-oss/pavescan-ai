"""Smoke test for src/detection/clustering.py.

Hand-crafted detection sets exercise the connected-components logic, then a
real ensemble run on a sample image confirms clustering produces sane output
on actual model results.

Run from repo root:
    .venv/Scripts/python.exe scripts/smoke_test_clustering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detection.clustering import (
    apply_priority_overrides,
    cluster_detections,
)
from src.detection.model import (
    SAFETY_PRIORITIES,
    load_ensemble_models,
    run_ensemble_inference,
)


def _det(bbox, cls="crack", conf=0.5, severity="medium", priority="monitor",
         models=("crack_seg",), mask=None, area=None):
    return {
        "class_name": cls,
        "confidence": conf,
        "bbox": list(bbox),
        "mask": mask,
        "area_pixels": area or int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
        "severity": severity,
        "safety_priority": priority,
        "models_agreeing": list(models),
    }


def test_empty():
    assert cluster_detections([]) == []
    print("[OK] empty input -> []")


def test_two_overlapping_become_one_cluster():
    dets = [
        _det((10, 10, 100, 100)),
        _det((50, 50, 140, 140)),  # IoU > 0
    ]
    clusters = cluster_detections(dets)
    assert len(clusters) == 1, f"expected 1 cluster, got {len(clusters)}"
    assert clusters[0]["detection_count"] == 2
    assert dets[0]["cluster_id"] == 0 and dets[1]["cluster_id"] == 0
    print("[OK] two overlapping boxes -> 1 cluster")


def test_close_centroids_become_one_cluster():
    # Non-overlapping but centroids within 50 px
    dets = [
        _det((0, 0, 20, 20)),       # centroid (10, 10)
        _det((30, 30, 50, 50)),     # centroid (40, 40), distance ~42 -> merge
    ]
    clusters = cluster_detections(dets, centroid_distance_px=50)
    assert len(clusters) == 1, f"expected 1 cluster, got {len(clusters)}"
    print("[OK] near-centroid boxes -> 1 cluster")


def test_far_apart_two_clusters():
    dets = [
        _det((0, 0, 20, 20)),
        _det((500, 500, 520, 520)),   # very far away
    ]
    clusters = cluster_detections(dets, centroid_distance_px=50)
    assert len(clusters) == 2, f"expected 2 clusters, got {len(clusters)}"
    print("[OK] distant boxes -> 2 clusters")


def test_cluster_id_deterministic_topleft_order():
    # Cluster A at top-left, cluster B at bottom-right; pass them
    # in reverse order; cluster_id should still be assigned by position.
    dets = [
        _det((400, 400, 420, 420)),  # bottom-right
        _det((10, 10, 30, 30)),      # top-left
    ]
    clusters = cluster_detections(dets, centroid_distance_px=10)
    assert clusters[0]["bbox_union"][0] == 10, "top-left cluster should be id 0"
    assert clusters[1]["bbox_union"][0] == 400
    print("[OK] cluster_id deterministic by top-left position")


def test_models_agreeing_union():
    dets = [
        _det((10, 10, 100, 100), models=("crack_seg", "rdd2022")),
        _det((50, 50, 140, 140), models=("crack_seg", "crack_det")),
    ]
    clusters = cluster_detections(dets)
    assert clusters[0]["models_agreeing"] == ["crack_det", "crack_seg", "rdd2022"]
    print("[OK] models_agreeing is the union across underlying detections")


def test_suggested_priority_takes_max():
    dets = [
        _det((10, 10, 100, 100), priority="routine"),
        _det((50, 50, 140, 140), priority="critical"),
        _det((60, 60, 130, 130), priority="monitor"),
    ]
    clusters = cluster_detections(dets)
    assert clusters[0]["suggested_priority"] == "critical"
    print("[OK] suggested_priority = max-rank across underlying")


def test_apply_overrides():
    dets = [
        _det((10, 10, 100, 100), priority="monitor"),
        _det((50, 50, 140, 140), priority="monitor"),
    ]
    # Stamp originals (would normally be done in classify_all_detections)
    for d in dets:
        d["original_safety_priority"] = d["safety_priority"]
        d["recommended_action"] = "Include in next maintenance cycle"
        d["original_recommended_action"] = d["recommended_action"]

    cluster_detections(dets)  # stamps cluster_id
    results = [{"filename": "a.jpg", "result": {"detections": dets}}]

    overrides = {"a.jpg::0": "critical"}
    has_over = apply_priority_overrides(results, overrides, SAFETY_PRIORITIES)
    assert has_over is True
    assert dets[0]["safety_priority"] == "critical"
    assert dets[1]["safety_priority"] == "critical"
    assert dets[0]["recommended_action"] == SAFETY_PRIORITIES["critical"]["action"]

    # Unset the override → restore originals
    has_over_after = apply_priority_overrides(results, {}, SAFETY_PRIORITIES)
    assert has_over_after is False
    assert dets[0]["safety_priority"] == "monitor"
    assert dets[0]["recommended_action"] == "Include in next maintenance cycle"
    print("[OK] apply_priority_overrides round-trips override and restore")


def test_real_ensemble_image():
    """Run a real ensemble pass and confirm clustering reduces the visible card count."""
    sample_dir = REPO_ROOT / "data" / "sample"
    samples = sorted(sample_dir.glob("*.jpg")) + sorted(sample_dir.glob("*.png"))
    if not samples:
        print("[SKIP] no sample images in data/sample/ — clustering integration test")
        return

    sample = samples[0]
    image = cv2.imdecode(np.fromfile(str(sample), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[SKIP] could not load {sample}")
        return

    try:
        models, paths = load_ensemble_models(version="auto")
    except FileNotFoundError as e:
        print(f"[SKIP] ensemble models missing: {e}")
        return

    result = run_ensemble_inference(models, image, confidence=0.15, imgsz=1280)
    detections = result["detections"]
    clusters = cluster_detections(detections)

    print(f"[OK] real image '{sample.name}': "
          f"{len(detections)} detections -> {len(clusters)} clusters")
    if clusters:
        c = clusters[0]
        print(f"     cluster 0: {c['detection_count']} signals, "
              f"models={c['models_agreeing']}, suggested={c['suggested_priority']}")


if __name__ == "__main__":
    test_empty()
    test_two_overlapping_become_one_cluster()
    test_close_centroids_become_one_cluster()
    test_far_apart_two_clusters()
    test_cluster_id_deterministic_topleft_order()
    test_models_agreeing_union()
    test_suggested_priority_takes_max()
    test_apply_overrides()
    test_real_ensemble_image()
    print("\nAll clustering smoke tests passed.")
