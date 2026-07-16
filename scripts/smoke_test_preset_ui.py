"""Smoke test — Detection page preset menu + display toggle (headless AppTest).

Drives app/pages/2_Detection.py with streamlit.testing.v1.AppTest:
1. Defaults: preset starts at Balanced and knob values equal app defaults.
2. Picking a preset copies its values into the knobs (incl. conditional SAHI slider).
3. Manually changing a knob flips the preset to Custom.
4. Display toggle renders cluster cards / raw-box table / both from a fabricated
   detection result (no real model inference; model loading is monkeypatched).

Run: python scripts/smoke_test_preset_ui.py   (exit 0 = pass)
"""

import sys
import types
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from streamlit.testing.v1 import AppTest

import src.detection.model as model_module
from src.detection.presets import CUSTOM_PRESET, DEFAULT_PRESET, DETECTION_PRESETS

PAGE = str(project_root / "app" / "pages" / "2_Detection.py")

# The page loads YOLO models on every run — irrelevant to widget logic and slow,
# so stub the loaders for this test.
model_module.load_ensemble_models = lambda version="auto": (
    {"stub_model": object()},
    {"stub_model": "stub.pt"},
)
model_module.load_model = lambda path: object()

failures = []


def check(condition: bool, message: str) -> None:
    status = "ok " if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        failures.append(message)


def fresh_apptest(with_results: bool = False) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.session_state["uploaded_files"] = [types.SimpleNamespace(name="fake.png")]
    if with_results:
        det = {
            "class_name": "crack",
            "confidence": 0.9,
            "severity": "high",
            "severity_score": 0.7,
            "safety_priority": "urgent",
            "original_safety_priority": "urgent",
            "recommended_action": "repair",
            "original_recommended_action": "repair",
            "width_pixels": 12.0,
            "models_agreeing": ["stub_model"],
            "bbox": [10, 10, 60, 60],
            "mask": None,
            "cluster_id": 0,
        }
        cluster = {
            "cluster_id": 0,
            "bbox_union": [10, 10, 60, 60],
            "detection_indices": [0],
            "detection_count": 1,
            "models_agreeing": ["stub_model"],
            "mask_coverage_pct": 10.0,
            "suggested_priority": "urgent",
        }
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        at.session_state["detection_results"] = [
            {
                "filename": "fake.png",
                "result": {
                    "annotated_image": blank,
                    "annotated_clusters": blank.copy(),
                    "detections": [det],
                    "clusters": [cluster],
                    "inference_time_ms": 5.0,
                },
            }
        ]
    return at


def run_checked(at: AppTest, label: str) -> AppTest:
    at.run()
    check(not at.exception, f"{label}: no exception ({[e.value for e in at.exception]})")
    return at


print("== scenario 1: defaults ==")
at = run_checked(fresh_apptest(), "initial run")
preset_box = at.sidebar.selectbox(key="detection_preset")
check(preset_box.value == DEFAULT_PRESET, f"preset starts at '{DEFAULT_PRESET}'")
check(at.sidebar.slider(key="confidence").value == 0.15, "confidence default 0.15")
check(at.sidebar.select_slider(key="imgsz").value == 1280, "imgsz default 1280")
check(at.sidebar.toggle(key="use_sahi").value is False, "SAHI off by default")
sahi_sliders = [w for w in at.sidebar.select_slider if w.key == "sahi_slice_size"]
check(len(sahi_sliders) == 0, "SAHI tile slider hidden while SAHI off")

print("== scenario 2: preset applies values ==")
at.sidebar.selectbox(key="detection_preset").select("Max Recall (SAHI)")
run_checked(at, "select Max Recall")
values = DETECTION_PRESETS["Max Recall (SAHI)"]["values"]
check(at.sidebar.slider(key="confidence").value == values["confidence"], "confidence -> 0.10")
check(at.sidebar.toggle(key="use_sahi").value is True, "SAHI turned on")
check(
    at.sidebar.select_slider(key="sahi_slice_size").value == values["sahi_slice_size"],
    "SAHI tile slider rendered at 320",
)

print("== scenario 3: every preset applies cleanly ==")
for name, preset in DETECTION_PRESETS.items():
    at.sidebar.selectbox(key="detection_preset").select(name)
    run_checked(at, f"select {name}")
    check(
        at.sidebar.slider(key="confidence").value == preset["values"]["confidence"]
        and at.sidebar.select_slider(key="imgsz").value == preset["values"]["imgsz"]
        and at.sidebar.toggle(key="use_tta").value == preset["values"]["use_tta"]
        and at.sidebar.toggle(key="use_sahi").value == preset["values"]["use_sahi"]
        and at.sidebar.toggle(key="use_wbf").value == preset["values"]["use_wbf"],
        f"{name}: all knobs match preset",
    )

print("== scenario 4: manual knob flips to Custom ==")
at = run_checked(fresh_apptest(), "fresh run")
at.sidebar.select_slider(key="imgsz").set_value(640)
run_checked(at, "manual imgsz change")
check(
    at.sidebar.selectbox(key="detection_preset").value == CUSTOM_PRESET,
    "preset flipped to Custom after manual knob change",
)
check(at.sidebar.slider(key="confidence").value == 0.15, "other knobs untouched")

print("== scenario 5: display toggle ==")
at = run_checked(fresh_apptest(with_results=True), "run with fabricated results")
radio = at.sidebar.radio(key="display_style")
check(radio.value == "Cluster cards", "display defaults to Cluster cards")
headers = [x.label for x in at.expander]
check(any("regions" in h and "detections" not in h for h in headers),
      f"cluster-cards header shows regions ({headers})")

at.sidebar.radio(key="display_style").set_value("Raw boxes")
run_checked(at, "switch to Raw boxes")
headers = [x.label for x in at.expander]
check(any("1 detections" in h for h in headers), f"raw header shows detections ({headers})")
check(len(at.dataframe) == 1, "raw mode renders exactly one table (no cluster cards)")
captions = " | ".join(c.value for c in at.caption)
check("Unfiltered raw model output" in captions, "raw-mode caption present")
override_buttons = [b for b in at.button if b.key and b.key.startswith("override_")]
check(len(override_buttons) == 0, "no inspector override buttons in raw mode")

at.sidebar.radio(key="display_style").set_value("Both")
run_checked(at, "switch to Both")
headers = [x.label for x in at.expander]
check(any("regions" in h and "detections" in h for h in headers),
      f"both-mode header shows regions + detections ({headers})")
override_buttons = [b for b in at.button if b.key and b.key.startswith("override_")]
check(len(override_buttons) == 4, "cluster cards (4 override buttons) present in Both mode")

print()
if failures:
    print(f"SMOKE TEST FAILED — {len(failures)} failure(s)")
    sys.exit(1)
print("SMOKE TEST PASSED — preset menu + display toggle behave correctly")
