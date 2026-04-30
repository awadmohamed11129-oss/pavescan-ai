"""Programmatic smoke test for the ASTM D6433 safety-priority pipeline.

Runs detection on a few sample images and verifies the new fields are populated
on every detection (safety_priority, severity_score, recommended_action,
deterioration_risk, width_pixels). Catches Python-side errors before browser test.

Run: python scripts/smoke_test_safety.py
"""

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    print("=" * 70)
    print("Safety-Priority Pipeline Smoke Test")
    print("=" * 70)

    try:
        from src.detection.model import (
            DEFECT_RISK_WEIGHTS,
            SAFETY_PRIORITIES,
            classify_safety_priority,
            classify_severity,
            estimate_crack_width_pixels,
            load_ensemble_models,
            run_ensemble_inference,
            summarize_detections,
        )
    except Exception:
        print("[FAIL] Import error in src.detection.model:")
        traceback.print_exc()
        return 1

    print("[OK] Imports succeeded")
    print(f"     DEFECT_RISK_WEIGHTS keys: {sorted(DEFECT_RISK_WEIGHTS.keys())}")
    print(f"     SAFETY_PRIORITIES keys:   {sorted(SAFETY_PRIORITIES.keys())}")

    # 1. Test classify_safety_priority on synthetic inputs.
    print("\n--- Test 1: classify_safety_priority logic ---")
    cases = [
        ("Pothole", "high", 0.85, 0.05, "isolated"),
        ("Alligator_Crack", "high", 0.75, 0.10, "systemic"),
        ("Longitudinal_Crack", "medium", 0.45, 0.02, "isolated"),
        ("crack", "low", 0.20, 0.005, "isolated"),
    ]
    for cls, sev, score, area, cluster in cases:
        try:
            pkey, action = classify_safety_priority(cls, sev, score, area, cluster)
            print(f"     {cls:25s} sev={sev:6s} score={score:.2f} -> {pkey:8s} | {action[:50]}")
        except Exception:
            print(f"[FAIL] classify_safety_priority crashed for {cls}:")
            traceback.print_exc()
            return 1

    # 2. Test estimate_crack_width_pixels on None and a fake mask.
    print("\n--- Test 2: estimate_crack_width_pixels ---")
    import cv2
    import numpy as np
    try:
        w_none = estimate_crack_width_pixels(None)
        print(f"     mask=None -> {w_none}")
        # Use an irregular blob (more like a real YOLO mask) instead of a perfect
        # rectangle — CHAIN_APPROX_SIMPLE compresses rectangles to 4 points which
        # the function rejects as degenerate.
        fake_mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.ellipse(fake_mask, (100, 100), (80, 15), 30, 0, 360, 1, -1)
        w_mask = estimate_crack_width_pixels(fake_mask)
        print(f"     elongated ellipse (160x30 @30deg) -> {w_mask:.2f} px (expect ~30)")
        if w_mask <= 0:
            print("[FAIL] expected non-zero width for valid mask")
            return 1
    except Exception:
        print("[FAIL] estimate_crack_width_pixels crashed:")
        traceback.print_exc()
        return 1

    # 3. Run real ensemble inference on a few sample images.
    print("\n--- Test 3: real inference on sample images ---")
    sample_dir = PROJECT_ROOT / "data" / "sample"
    if not sample_dir.exists():
        print(f"[SKIP] No sample dir at {sample_dir}")
        return 0

    samples = sorted(sample_dir.glob("*.jpg"))[:3]
    if not samples:
        print(f"[SKIP] No .jpg samples in {sample_dir}")
        return 0
    print(f"     Found {len(samples)} sample images")

    try:
        models, model_paths = load_ensemble_models()
    except Exception:
        print("[FAIL] load_ensemble_models crashed:")
        traceback.print_exc()
        return 1
    print(f"     Loaded {len(models)} models: {sorted(models.keys())}")

    required_fields = {
        "safety_priority",
        "severity",
        "severity_score",
        "recommended_action",
        "deterioration_risk",
        "width_pixels",
        "aspect_ratio",
        "class_name",
        "confidence",
        "bbox",
    }

    total_dets = 0
    field_failures = []

    for img_path in samples:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[FAIL] Could not read {img_path.name}")
            return 1
        try:
            result = run_ensemble_inference(
                models,
                image,
                confidence=0.15,
                use_sahi=False,
                augment=False,
                model_paths=model_paths,
            )
        except Exception:
            print(f"[FAIL] run_ensemble_inference crashed on {img_path.name}:")
            traceback.print_exc()
            return 1

        dets = result.get("detections", [])
        total_dets += len(dets)
        print(f"     {img_path.name}: {len(dets)} detections")

        for i, det in enumerate(dets):
            missing = required_fields - set(det.keys())
            if missing:
                field_failures.append((img_path.name, i, missing))
                continue
            # Sanity-check value ranges
            score = det.get("severity_score", -1)
            if not (0.0 <= score <= 1.0):
                field_failures.append((img_path.name, i, f"severity_score out of range: {score}"))

    if field_failures:
        print("\n[FAIL] Detection field issues:")
        for name, i, prob in field_failures[:10]:
            print(f"     {name} det#{i}: {prob}")
        return 1

    if total_dets == 0:
        print("[WARN] No detections across any sample — model may be misconfigured")
    else:
        print(f"\n[OK] All {total_dets} detections have required safety fields")

    # 4. Test summarize_detections returns expected keys.
    print("\n--- Test 4: summarize_detections ---")
    flat_dets = []
    for img_path in samples:
        image = cv2.imread(str(img_path))
        result = run_ensemble_inference(
            models, image, confidence=0.15, use_sahi=False, augment=False,
            model_paths=model_paths,
        )
        flat_dets.extend(result.get("detections", []))

    try:
        summary = summarize_detections(flat_dets)
    except Exception:
        print("[FAIL] summarize_detections crashed:")
        traceback.print_exc()
        return 1

    expected_summary_keys = {"by_priority", "by_class", "by_severity", "total_detections"}
    missing_keys = expected_summary_keys - set(summary.keys())
    if missing_keys:
        print(f"[FAIL] Summary missing keys: {missing_keys}")
        print(f"       Got: {sorted(summary.keys())}")
        return 1
    print(f"[OK] Summary keys: {sorted(summary.keys())}")
    print(f"     by_priority: {summary.get('by_priority')}")
    print(f"     by_severity: {summary.get('by_severity')}")
    print(f"     total_detections: {summary.get('total_detections')}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
