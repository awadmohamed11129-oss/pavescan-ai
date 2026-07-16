"""Smoke test — detection preset definitions stay valid for the Detection page.

Validates every preset in src/detection/presets.py against:
1. The exact widget option grids used in app/pages/2_Detection.py (a preset value
   that isn't a legal widget option makes Streamlit throw at render time).
2. The run_ensemble_inference signature (preset knobs must map onto real params).
3. The presets-over-defaults rule: the default preset must equal current app defaults.

Run: python scripts/smoke_test_presets.py   (exit 0 = pass)
"""

import inspect
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.detection.presets import CUSTOM_PRESET, DEFAULT_PRESET, DETECTION_PRESETS

# Mirrors of the widget grids in app/pages/2_Detection.py — update together.
CONFIDENCE_MIN, CONFIDENCE_MAX, CONFIDENCE_STEP = 0.05, 0.9, 0.05
IMGSZ_OPTIONS = [640, 960, 1280, 1600]
SAHI_SLICE_OPTIONS = [320, 480, 640, 800, 1024]

REQUIRED_KEYS = {"confidence", "imgsz", "use_tta", "use_sahi", "sahi_slice_size", "use_wbf"}

# Current app defaults (the values the widgets initialize to without presets).
APP_DEFAULTS = {
    "confidence": 0.15,
    "imgsz": 1280,
    "use_tta": False,
    "use_sahi": False,
    "sahi_slice_size": 640,
    "use_wbf": False,
}

failures = []


def check(condition: bool, message: str) -> None:
    status = "ok " if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        failures.append(message)


print("== preset inventory ==")
check(len(DETECTION_PRESETS) == 5, f"exactly 5 presets defined (got {len(DETECTION_PRESETS)})")
check(CUSTOM_PRESET not in DETECTION_PRESETS, "Custom is not a values preset")
check(DEFAULT_PRESET in DETECTION_PRESETS, f"default preset '{DEFAULT_PRESET}' exists")

print("== per-preset value grids ==")
for name, preset in DETECTION_PRESETS.items():
    values = preset["values"]
    check(set(values) == REQUIRED_KEYS, f"{name}: keys == required knob set")
    check(
        CONFIDENCE_MIN <= values["confidence"] <= CONFIDENCE_MAX
        and round((values["confidence"] - CONFIDENCE_MIN) / CONFIDENCE_STEP, 6) % 1 == 0,
        f"{name}: confidence {values['confidence']} on slider grid",
    )
    check(values["imgsz"] in IMGSZ_OPTIONS, f"{name}: imgsz {values['imgsz']} is a widget option")
    check(
        values["sahi_slice_size"] in SAHI_SLICE_OPTIONS,
        f"{name}: sahi_slice_size {values['sahi_slice_size']} is a widget option",
    )
    check(
        isinstance(values["use_tta"], bool)
        and isinstance(values["use_sahi"], bool)
        and isinstance(values["use_wbf"], bool),
        f"{name}: toggles are booleans",
    )
    check(
        not (values["use_tta"] and values["use_sahi"]),
        f"{name}: TTA+SAHI not combined (SAHI path ignores augment)",
    )
    check(bool(preset.get("description")), f"{name}: has a description")

print("== default preset == app defaults (presets-over-defaults rule) ==")
check(
    DETECTION_PRESETS[DEFAULT_PRESET]["values"] == APP_DEFAULTS,
    f"'{DEFAULT_PRESET}' values equal current app defaults",
)

print("== knobs map onto run_ensemble_inference ==")
from src.detection.model import run_ensemble_inference

sig_params = set(inspect.signature(run_ensemble_inference).parameters)
# use_tta -> augment, use_wbf -> fusion_method; the rest map by name.
mapped = {"confidence", "imgsz", "use_sahi", "sahi_slice_size"}
check(mapped <= sig_params, "direct-name knobs exist in signature")
check({"augment", "fusion_method"} <= sig_params, "augment + fusion_method exist for TTA/WBF")

print()
if failures:
    print(f"SMOKE TEST FAILED — {len(failures)} failure(s)")
    sys.exit(1)
print("SMOKE TEST PASSED — all preset definitions valid")
