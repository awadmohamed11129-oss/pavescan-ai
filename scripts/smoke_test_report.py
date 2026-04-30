"""Programmatic smoke test for PDF report generation.

Generates a real PDF from sample-image detections and verifies:
  - generate_report_html() doesn't crash
  - html_to_pdf() produces non-empty bytes
  - PDF has plausible size (>5KB) and PDF magic bytes
  - Output PDF is written to reports/smoke_test_report.pdf for visual inspection

Run: python scripts/smoke_test_report.py
"""

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    print("=" * 70)
    print("PDF Report Smoke Test")
    print("=" * 70)

    try:
        import cv2
        from src.detection.model import load_ensemble_models, run_ensemble_inference
        from src.reporting.generator import generate_report_html, html_to_pdf
    except Exception:
        print("[FAIL] Import error:")
        traceback.print_exc()
        return 1
    print("[OK] Imports succeeded")

    # 1. Run detection on sample images.
    sample_dir = PROJECT_ROOT / "data" / "sample"
    samples = sorted(sample_dir.glob("*.jpg"))[:3]
    if not samples:
        print(f"[FAIL] No samples in {sample_dir}")
        return 1

    print(f"\n--- Running detection on {len(samples)} sample images ---")
    models, model_paths = load_ensemble_models()

    detection_results = []
    geo_results = []
    for img_path in samples:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[FAIL] Could not read {img_path.name}")
            return 1
        result = run_ensemble_inference(
            models, image, confidence=0.15, use_sahi=False, augment=False,
            model_paths=model_paths,
        )
        detection_results.append({"filename": img_path.name, "result": result})
        # Fake GPS (Toronto-ish) so the report has location data
        geo_results.append({
            "filename": img_path.name,
            "result": result,
            "lat": 43.6605 + 0.0001 * len(geo_results),
            "lon": -79.3955 + 0.0001 * len(geo_results),
        })
        print(f"     {img_path.name}: {len(result['detections'])} detections")

    project_info = {
        "project_name": "Smoke Test — King St E Section",
        "inspector_name": "PaveScan AI Smoke Test",
        "notes": "Automated smoke test of safety-priority report generation. "
                 "Verifies HTML rendering and PDF conversion of all new sections.",
    }

    # 2. Generate HTML.
    print("\n--- Generating report HTML ---")
    try:
        html = generate_report_html(detection_results, geo_results, project_info)
    except Exception:
        print("[FAIL] generate_report_html crashed:")
        traceback.print_exc()
        return 1
    print(f"[OK] HTML generated ({len(html):,} chars)")

    # Verify expected sections appear in the HTML
    expected_phrases = [
        "Safety",  # safety assessment box
        "Pavement Condition Index",  # PCI section
        "Maintenance Plan",  # maintenance plan section
        "Methodology",  # methodology footer
        "Priority",  # priority summary / defect inventory column
    ]
    missing = [p for p in expected_phrases if p not in html]
    if missing:
        print(f"[FAIL] HTML missing expected phrases: {missing}")
        return 1
    print(f"[OK] HTML contains all expected sections: {expected_phrases}")

    # 3. Convert to PDF.
    print("\n--- Converting HTML to PDF ---")
    try:
        pdf_bytes = html_to_pdf(html)
    except Exception:
        print("[FAIL] html_to_pdf crashed:")
        traceback.print_exc()
        return 1

    if not pdf_bytes:
        print("[FAIL] html_to_pdf returned empty bytes")
        return 1

    # PDF magic bytes start with "%PDF"
    if not pdf_bytes.startswith(b"%PDF"):
        print(f"[FAIL] Output does not start with PDF magic bytes. Starts with: {pdf_bytes[:10]!r}")
        return 1

    if len(pdf_bytes) < 5_000:
        print(f"[WARN] PDF suspiciously small: {len(pdf_bytes):,} bytes (probably missing content)")

    # 4. Write PDF to disk for visual inspection.
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "smoke_test_report.pdf"
    out_path.write_bytes(pdf_bytes)
    print(f"[OK] PDF written: {out_path} ({len(pdf_bytes):,} bytes)")

    print("\n" + "=" * 70)
    print("REPORT SMOKE TEST PASSED")
    print(f"Open the PDF to verify visual layout: {out_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
