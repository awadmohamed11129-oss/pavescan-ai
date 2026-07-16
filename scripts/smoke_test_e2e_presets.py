"""E2E smoke test — presets drive REAL ensemble inference end to end (headless).

Unlike smoke_test_preset_ui.py (which stubs model loading), this runs the actual
Detection page against the real model weights and the real Phase A survey photo:
Balanced preset vs Max Recall (SAHI), then the display toggle on real results.

Run: python scripts/smoke_test_e2e_presets.py   (exit 0 = pass; several minutes on CPU)
"""

import io
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from streamlit.testing.v1 import AppTest

PAGE = str(project_root / "app" / "pages" / "2_Detection.py")
PHOTO = project_root / "data" / "raw" / "user_phase_a" / "img_street_pothole.png"

failures = []


def check(condition: bool, message: str) -> None:
    status = "ok " if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        failures.append(message)


class FakeUpload(io.BytesIO):
    """Minimal stand-in for st.file_uploader's UploadedFile (name + read + seek)."""

    def __init__(self, path: Path):
        super().__init__(path.read_bytes())
        self.name = path.name


def run_detection(at: AppTest) -> tuple[int, str]:
    """Click Run Detection, return (merged detection count, success message)."""
    buttons = [b for b in at.button if "Run Detection" in (b.label or "")]
    assert buttons, "Run Detection button not found"
    buttons[0].click()
    at.run()
    assert not at.exception, f"exception during detection: {[e.value for e in at.exception]}"
    results = at.session_state["detection_results"]
    count = sum(len(r["result"]["detections"]) for r in results)
    success = " | ".join(s.value for s in at.success)
    return count, success


print(f"== setup: photo={PHOTO.name} exists={PHOTO.exists()} ==")
check(PHOTO.exists(), "Phase A real photo present")

at = AppTest.from_file(PAGE, default_timeout=900)
at.session_state["uploaded_files"] = [FakeUpload(PHOTO)]
at.run()
check(not at.exception, "initial run clean")

# Production path: Ensemble mode (auto = V2 seg + V1 det)
at.sidebar.radio(key="detection_mode").set_value("Ensemble (Both Models)")
at.run()
check(not at.exception, "ensemble mode selected")

print("== Balanced (app default) — native 1280px ==")
count_balanced, msg = run_detection(at)
print(f"    -> {count_balanced} detections | {msg}")
check(count_balanced > 0, f"Balanced finds detections ({count_balanced})")
check("1280px" in msg, "success message reports 1280px")

print("== Max Recall (SAHI) preset ==")
at.sidebar.selectbox(key="detection_preset").select("Max Recall (SAHI)")
at.run()
count_sahi, msg = run_detection(at)
print(f"    -> {count_sahi} detections | {msg}")
check(count_sahi > 0, f"Max Recall finds detections ({count_sahi})")
check("SAHI 320px" in msg, "success message reports SAHI 320px")
# The preset's whole point — more recall than Balanced on the same image.
check(
    count_sahi > count_balanced,
    f"Max Recall recall gain on auto ensemble ({count_sahi} vs {count_balanced})",
)

print("== display toggle on REAL results ==")
at.sidebar.radio(key="display_style").set_value("Raw boxes")
at.run()
check(not at.exception, "raw-boxes render clean")
check(len(at.dataframe) == 1, "raw table rendered")
headers = [x.label for x in at.expander]
check(any("detections" in h for h in headers), f"raw header ({headers})")

at.sidebar.radio(key="display_style").set_value("Both")
at.run()
check(not at.exception, "both-mode render clean")
override_buttons = [b for b in at.button if b.key and b.key.startswith("override_")]
check(len(override_buttons) > 0, "cluster cards present in Both mode")

print("== Classic V1-era (640px) preset — sanity ==")
at.sidebar.selectbox(key="detection_preset").select("Classic V1-era (640px)")
at.run()
count_classic, msg = run_detection(at)
print(f"    -> {count_classic} detections | {msg}")
check("640px" in msg, "success message reports 640px")

print()
print(f"COUNTS  balanced={count_balanced}  sahi={count_sahi}  classic640={count_classic}")
if failures:
    print(f"E2E SMOKE TEST FAILED — {len(failures)} failure(s)")
    sys.exit(1)
print("E2E SMOKE TEST PASSED — presets drive real inference correctly")
