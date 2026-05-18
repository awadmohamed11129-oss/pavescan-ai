"""Spatial clustering of detections for the cluster-card UI.

Phase 1 frames inspection honestly: AI surfaces candidate regions via spatial
clustering with multi-model agreement as the trust signal; the inspector
classifies severity via a 4-button override per cluster. Per-detection
auto-severity proved structurally unreliable on top of single-class V1.
"""

from __future__ import annotations

from typing import Any

from .model import compute_iou


PRIORITY_RANK = {"critical": 4, "urgent": 3, "monitor": 2, "routine": 1}
SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


def _centroid(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _centroid_distance(b1: list[float], b2: list[float]) -> float:
    cx1, cy1 = _centroid(b1)
    cx2, cy2 = _centroid(b2)
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def _union_bbox(bboxes: list[list[float]]) -> list[float]:
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def cluster_detections(
    detections: list[dict[str, Any]],
    iou_overlap_threshold: float = 0.0,
    centroid_distance_px: float = 50.0,
) -> list[dict[str, Any]]:
    """Group spatially co-located detections into clusters.

    Edge between A and B iff IoU(A, B) > iou_overlap_threshold OR centroid
    distance (px) < centroid_distance_px. Connected components → clusters.

    Side effect: stamps ``det["cluster_id"]`` on each detection so downstream
    consumers (Report page, override apply) can map detection → cluster
    without re-clustering.

    Returns clusters sorted by union bbox top-left (y then x) so cluster_id
    is deterministic across reruns of the same input.
    """
    if not detections:
        return []

    n = len(detections)
    uf = _UnionFind(n)

    for i in range(n):
        bi = detections[i]["bbox"]
        for j in range(i + 1, n):
            bj = detections[j]["bbox"]
            if compute_iou(bi, bj) > iou_overlap_threshold:
                uf.union(i, j)
                continue
            if _centroid_distance(bi, bj) < centroid_distance_px:
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    raw_clusters: list[dict[str, Any]] = []
    for indices in groups.values():
        bboxes = [detections[i]["bbox"] for i in indices]
        union = _union_bbox(bboxes)

        classes = sorted({detections[i].get("class_name", "") for i in indices})

        models_set: set[str] = set()
        for i in indices:
            for m in detections[i].get("models_agreeing", []):
                models_set.add(m)

        priorities = [detections[i].get("safety_priority", "routine") for i in indices]
        suggested_priority = max(priorities, key=lambda p: PRIORITY_RANK.get(p, 0))

        severities = [detections[i].get("severity", "low") for i in indices]
        suggested_severity = max(severities, key=lambda s: SEVERITY_RANK.get(s, 0))

        mask_pixels = 0
        for i in indices:
            mask = detections[i].get("mask")
            if mask is not None:
                mask_pixels += int(mask.sum())
        union_area = max(1, int((union[2] - union[0]) * (union[3] - union[1])))
        mask_coverage_pct = min(100.0, 100.0 * mask_pixels / union_area)

        confs = [float(detections[i].get("confidence", 0.0)) for i in indices]
        mean_conf = sum(confs) / len(confs) if confs else 0.0

        raw_clusters.append({
            "bbox_union": union,
            "detection_indices": sorted(indices),
            "detection_count": len(indices),
            "classes": classes,
            "models_agreeing": sorted(models_set),
            "suggested_priority": suggested_priority,
            "suggested_severity": suggested_severity,
            "mask_coverage_pct": mask_coverage_pct,
            "mean_confidence": mean_conf,
        })

    raw_clusters.sort(key=lambda c: (c["bbox_union"][1], c["bbox_union"][0]))

    for cid, cluster in enumerate(raw_clusters):
        cluster["cluster_id"] = cid
        for idx in cluster["detection_indices"]:
            detections[idx]["cluster_id"] = cid

    return raw_clusters


def apply_priority_overrides(
    detection_results: list[dict[str, Any]],
    overrides: dict[str, str],
    safety_priorities: dict[str, dict[str, Any]],
) -> bool:
    """Apply inspector priority overrides to detection dicts in-place.

    For each detection, if its cluster has an override, swap ``safety_priority``
    and ``recommended_action`` accordingly; otherwise restore from the
    ``original_safety_priority`` / ``original_recommended_action`` fields stamped
    during ``classify_all_detections``.

    Args:
        detection_results: list of {"filename": str, "result": {"detections": [...]}}.
        overrides: ``{f"{filename}::{cluster_id}": priority_key}``.
        safety_priorities: SAFETY_PRIORITIES constant (for action lookup).

    Returns:
        True if any detection's effective priority differs from its original.
    """
    has_overrides = False
    for item in detection_results:
        filename = item.get("filename", "")
        for det in item.get("result", {}).get("detections", []):
            cid = det.get("cluster_id")
            if cid is None:
                continue
            key = f"{filename}::{cid}"
            if key in overrides:
                new_priority = overrides[key]
                det["safety_priority"] = new_priority
                action = safety_priorities.get(new_priority, {}).get("action", "")
                det["recommended_action"] = action
                original = det.get("original_safety_priority")
                if original is not None and new_priority != original:
                    has_overrides = True
            else:
                if "original_safety_priority" in det:
                    det["safety_priority"] = det["original_safety_priority"]
                if "original_recommended_action" in det:
                    det["recommended_action"] = det["original_recommended_action"]
    return has_overrides
