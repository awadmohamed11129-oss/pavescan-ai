"""Detection presets — named bundles of the Detection page's inference knobs.

DRR Phase B (2026-07-15). Values are evidence-based, taken from the Phase A
recall diagnostic on a real 1200x1600 survey photo
(reports/diagnosis_v1_20260517_222144.txt):

- Native V1 ensemble @1280px/conf 0.15 found 8 detections; conf 0.10 found 9.
- SAHI slice=320 found 22 detections on the same image (~2.4x recall).
- WBF roughly matched iou_dedup on counts (+1) but averages overlapping boxes.
- 640px/conf 0.25 reproduces the pre-upgrade (V1-era) app behavior.

Presets set widget values; they never replace the app defaults (Mohamad's
presets-over-defaults rule) — "Balanced" IS the current defaults, and any
manual knob change flips the menu to Custom.
"""

# Selectbox entry that means "no preset — manual knob control".
CUSTOM_PRESET = "Custom"

# Preset whose values must always equal the app's built-in widget defaults.
DEFAULT_PRESET = "Balanced (app default)"

DETECTION_PRESETS = {
    "Balanced (app default)": {
        "description": "The standard settings the app has always used. Good starting point.",
        "values": {
            "confidence": 0.15,
            "imgsz": 1280,
            "use_tta": False,
            "use_sahi": False,
            "sahi_slice_size": 640,
            "use_wbf": False,
        },
    },
    "High Precision": {
        "description": "Fewer, high-certainty detections (confidence 0.40). Use when the report must contain only defects you'd defend in the field.",
        "values": {
            "confidence": 0.40,
            "imgsz": 1280,
            "use_tta": False,
            "use_sahi": False,
            "sahi_slice_size": 640,
            "use_wbf": False,
        },
    },
    "Max Recall (SAHI)": {
        "description": "Tiled inference with small 320px tiles + low threshold — found ~2.4x more defects than Balanced on a real survey photo. Slowest per image.",
        "values": {
            "confidence": 0.10,
            "imgsz": 1280,
            "use_tta": False,
            "use_sahi": True,
            "sahi_slice_size": 320,
            "use_wbf": False,
        },
    },
    "Thorough (TTA + WBF)": {
        "description": "Test-time augmentation plus weighted box fusion at a low threshold. Better boxes and a small recall bump, ~3-4x slower.",
        "values": {
            "confidence": 0.10,
            "imgsz": 1280,
            "use_tta": True,
            "use_sahi": False,
            "sahi_slice_size": 640,
            "use_wbf": True,
        },
    },
    "Classic V1-era (640px)": {
        "description": "Reproduces the original app behavior: 640px inference, confidence 0.25. Useful as an A/B baseline against the newer settings.",
        "values": {
            "confidence": 0.25,
            "imgsz": 640,
            "use_tta": False,
            "use_sahi": False,
            "sahi_slice_size": 640,
            "use_wbf": False,
        },
    },
}
