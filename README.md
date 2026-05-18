# PaveScan AI

**Drone-based pavement inspection — computer vision crack detection with ASTM D6433 PCI scoring, safety-priority triage, and professional PDF reports.**

A unified civil engineering platform that ingests pavement imagery (drone or hand-held), runs an ensemble of YOLO11 models for crack and pothole detection, scores severity via a multi-factor system informed by ASTM D6433, and produces interactive maps + engineering-grade PDF reports. Built as an end-to-end portfolio project demonstrating hardware → AI → software → civil engineering.

---

## At a glance

```
┌────────────┐   ┌─────────────────┐   ┌────────────────────────┐   ┌────────────┐   ┌──────────┐
│  Drone /   │──▶│  Upload images  │──▶│  Ensemble detection    │──▶│  Map +     │──▶│  PDF     │
│  phone     │   │  (Streamlit)    │   │  (YOLO11 seg + det)    │   │  filters   │   │  report  │
└────────────┘   └─────────────────┘   │  + SAHI + TTA + WBF    │   └────────────┘   └──────────┘
                                        │                        │          │
                                        │  ASTM D6433 severity   │          │
                                        │  + safety priority     │          ▼
                                        └────────────────────────┘   ┌────────────┐
                                                                     │  Mission   │
                                                                     │  planner   │
                                                                     └────────────┘
```

---

## Module 1 — pavement inspection (built)

**Five Streamlit pages:**

1. **Upload** — drag-and-drop images (JPG/PNG/TIF, 200 MB cap), session-scoped storage.
2. **Detection** — ensemble inference with three orthogonal accuracy levers:
   - **SAHI** tiled inference for high-res drone images
   - **TTA** (test-time augmentation) for harder cases
   - **WBF** (Weighted Box Fusion) for ensemble fusion — averages overlapping cross-model boxes weighted by confidence rather than dropping the loser
   - Inference resolution slider (640 / 960 / 1280 / 1600)
3. **Map** — Folium interactive map with priority-sized markers, severity heatmap, multi-style basemaps. Falls back to Toronto demo coords when EXIF GPS is absent (clearly signposted in UI).
4. **Report** — WeasyPrint PDF with safety assessment, ASTM D6433 PCI score, defect inventory, maintenance plan grouped by urgency, methodology section.
5. **Mission Planner** — drone flight planning: route distance (haversine), photo coverage from camera FOV at altitude, photo intervals from configurable forward/side overlap, total flight time, battery feasibility check, and **ArduPilot-compatible `.waypoints` export** (QGC WPL 110 format).

### Multi-factor severity (ASTM D6433-informed)

The severity classifier is **not** a single area threshold — it composes four signals:

| Factor          | Weight (seg path) | Weight (det path) | What it captures                                  |
| --------------- | ----------------- | ----------------- | ------------------------------------------------- |
| Defect type     | 25%               | 35%               | D40/Pothole = 1.0, D20/Alligator = 0.90, etc.     |
| Area ratio      | 45%               | 65%               | Continuous curve, not bins                        |
| Estimated width | 30%               | —                 | From cv2.minAreaRect on segmentation mask         |
| Confidence      | 15% (penalty)     | 15% (penalty)     | Down-weights uncertain detections                 |

Plus density bonus (3+ defects → 1.08×, 5+ → 1.15×) and a **safety floor**: structural failures (potholes, alligator cracking) at confidence ≥ 0.20 always score ≥ medium, regardless of size — because a small pothole is still a tire blowout.

### Safety priority (separate from severity)

Severity asks "how bad is this defect?" Priority asks "what gets fixed FIRST?" Four levels:

- **CRITICAL** — emergency repair, 24–48 hr (e.g., high-severity pothole)
- **URGENT** — within 1–2 weeks (e.g., severe alligator cracking, systemic failure cluster)
- **ROUTINE** — within 1–3 months (typical preventive maintenance)
- **MONITOR** — track at next scheduled inspection

PDF reports group recommended fixes by these urgency timelines.

### PCI calculation

Implements the ASTM D6433 framework with **type-weighted deduct values** and **corrected deduct values** (`CDV = max(1.0 − 0.08·i, 0.20)` for the i-th highest deduct). Pothole high = 25 deduct, alligator high = 20, longitudinal high = 12, transverse high = 10.

---

## Accuracy

### Model lineup

| Slot                  | v1 (current `models/`)              | v2 (after Colab training)                |
| --------------------- | ----------------------------------- | ---------------------------------------- |
| Crack segmentation    | `pavescan_crack_seg.pt`             | `pavescan_crack_seg_v2.pt`               |
| Architecture          | YOLOv8n-seg (3.26M params, 640px)   | YOLO11m-seg (T4) / YOLO11l-seg (A100), 1280px |
| Training              | 50 epochs, default aug              | 200 epochs, multi-scale, aggressive aug  |
| Road damage detection | `pavescan_rdd2022.pt`               | `pavescan_rdd2022_v2.pt`                 |
| Architecture          | YOLOv8n (640px, 30 epochs trained)  | YOLO11m / YOLO11l, 1280px, 200 epochs, fl_gamma=1.5 |

### Crack-Seg val (200 imgs, 249 instances)

| Metric             | v1 (YOLOv8n-seg, 640px) | v2 (YOLO11, 1280px) |
| ------------------ | ----------------------- | ------------------- |
| Box mAP50          | **0.818**               | _pending Colab run_ |
| Box mAP50-95       | **0.636**               | _pending_           |
| Seg mAP50          | **0.658**               | _pending_           |
| Seg mAP50-95       | **0.226**               | _pending_           |
| Box Precision      | 0.861                   | _pending_           |
| Box Recall         | 0.744                   | _pending_           |

v1 numbers were generated locally on CPU using `ultralytics` 8.4.36 with default confidence threshold; raw output is in `models/metrics_v1_seg.json`. v2 numbers populate `models/metrics_v2_seg.json` automatically when the training notebook runs.

### How v2 metrics are produced

The training notebook (`notebooks/train_crack_detector.ipynb`) includes a **v1-vs-v2 comparison cell** for each option. After training:

1. Loads the trained v2 model, runs `.val()` on the same val split
2. If you upload your v1 weights to `/content/`, also runs `.val()` on v1 with identical settings
3. Prints a delta table and writes `metrics_v2_seg.json` (or `_det.json`)
4. Auto-downloads both the `.pt` and the `.json` to your machine

This is the only way to make accuracy claims that survive scrutiny — the val set is fixed, the only difference is the model weights.

---

## Architecture

```
pavescan-ai/
├── app/
│   ├── dashboard.py              # Streamlit entry point — ALWAYS launch from here
│   └── pages/
│       ├── 1_Upload.py
│       ├── 2_Detection.py        # Ensemble + SAHI + TTA + WBF toggles
│       ├── 3_Map.py              # Folium with priority-sized markers
│       ├── 4_Report.py           # PDF generation
│       └── 5_Mission.py          # Flight planning + ArduPilot waypoint export
├── src/
│   ├── detection/
│   │   └── model.py              # Multi-factor severity, safety priority, ensemble fusion
│   ├── mapping/
│   │   └── geo.py                # EXIF GPS extraction, demo fallback
│   ├── reporting/
│   │   ├── generator.py          # PCI calc, ASTM D6433 deduct values, CDV
│   │   └── templates/            # WeasyPrint Jinja2 templates
│   └── mission/
│       └── planner.py            # Haversine, FOV geometry, battery check, .waypoints export
├── notebooks/
│   └── train_crack_detector.ipynb  # YOLO11 training, v1-vs-v2 comparison, metrics save
├── scripts/
│   ├── smoke_test_safety.py      # End-to-end pipeline regression test
│   ├── smoke_test_report.py      # PDF generation regression test
│   └── download_samples.py
├── models/                       # .pt files (gitignored) + metrics_v*.json (committed)
├── data/sample/                  # Sample pavement images
└── tests/                        # (planned — pytest suite)
```

---

## Install & run

**Requires:** Python 3.14, ~5 GB disk for models + datasets.

```bash
git clone https://github.com/awadmohamed11129-oss/pavescan-ai
cd pavescan-ai

python -m venv .venv
source .venv/Scripts/activate    # Windows Git Bash
# or .venv\Scripts\activate.bat for cmd

pip install -r requirements.txt

# Launch dashboard — ALWAYS from dashboard.py, never individual pages
streamlit run app/dashboard.py
```

Open http://localhost:8501. Upload a pavement image, click through Detection → Map → Report.

### Smoke tests

```bash
python scripts/smoke_test_safety.py    # exits 0 if pipeline is healthy
python scripts/smoke_test_report.py    # exits 0 if PDF generation works
```

### Training v2 models (Google Colab)

1. Open `notebooks/train_crack_detector.ipynb` in Colab
2. **Runtime → Change runtime type → A100 GPU** (Pro recommended) or **T4** (free)
3. **(Optional)** Drag your existing v1 `.pt` files into `/content/` so the notebook can produce a v1-vs-v2 comparison table
4. In the `TRAINING_OPTION` cell, set to `"A"` (crack segmentation) or `"C"` (RDD2022 detection, US+Czech), or `"B"` (RDD2022 all 6 countries — Pro only)
5. **Runtime → Run All**, walk away (~45 min – 4 hr depending on option and GPU)
6. The trained `.pt` and a `metrics_v2_*.json` auto-download — drop the `.pt` in `models/` and the JSON in repo root

The detection code in `src/detection/model.py:18-23` already auto-prefers v2 weights with v1 as fallback.

---

## Methodology

- **Detection:** YOLO11 ensemble — a 1-class crack segmentation model (pixel masks) + a 4-class road damage detection model (Longitudinal / Transverse / Alligator / Pothole). The two models contribute complementary information; SAHI handles small cracks in high-resolution drone imagery; TTA is a final accuracy lever for hard cases.
- **Ensemble fusion:** Two strategies. **IoU-dedup** (default) keeps the highest-confidence box and merges metadata. **WBF** averages overlapping boxes weighted by confidence. WBF pre-harmonizes generic "crack" labels to specific classes (Pothole/Alligator/etc.) so the fusion can merge across-model agreements rather than producing duplicates.
- **Severity:** multi-factor scoring described above, calibrated to the ASTM D6433 distress identification manual.
- **Safety priority:** orthogonal axis to severity — encodes "what must be fixed first" so a small pothole doesn't get triaged as routine despite its low area ratio.
- **PCI:** 100 − sum of deduct values, with corrected deduct values applied to handle multiple defects on the same section.

---

## Known limitations

- **Width estimation is in pixels, not millimeters.** A 25-pixel-wide crack scored at 5 m altitude vs 50 m altitude is the same defect at very different real-world widths. The current severity score is calibrated for a typical drone altitude (~5–10 m, smartphone-equivalent FoV); deploy at radically different altitudes and severity will drift. Roadmap: add a manual reference-length calibration UI (click two endpoints on a known-length object once per mission → mm/px persists in session state).
- **No real-world drone imagery yet.** All sample images are from public RDD2022 / Crack-Seg datasets. The custom-built drone is currently on hold (cost overruns); software stack is fully exercised against free datasets and OpenDroneMap demo data.
- **Module 2 (site mapping & surveying) is planned, not built.** OpenDroneMap orthomosaic viewer, click-to-measure tools, before/after comparison, and survey-PDF generation are tracked as the next major chunk of work.
- **No formal test suite yet.** Smoke tests cover the integration path; unit tests are TODO.

---

## Module 2 roadmap (planned)

A second platform module for **site mapping & surveying** — the natural complement to Module 1's pavement inspection:

1. Upload overlapping drone survey photos
2. Process into orthomosaics via **OpenDroneMap** (we use it the way we use YOLOv8 — train + integrate, don't reinvent)
3. Display georeferenced orthomosaic on the map
4. Measurement tools: distance, area, elevation profiles
5. Before/after comparison of surveys from different dates (construction progress monitoring)
6. Survey PDF reports
7. **The killer integration:** fly grid → orthomosaic → run Module 1 detection ON the orthomosaic → full-road crack map with PCI overlaid on georeferenced coordinates

---

## Tech stack

- **Python** 3.14, **Streamlit** 1.56+, **Ultralytics** 8.3+, **PyTorch**, **OpenCV**
- **SAHI** for tiled inference, **ensemble-boxes** for WBF
- **Folium** + **streamlit-folium** for maps
- **WeasyPrint** + **xhtml2pdf** for PDF reports
- **Plotly** for charts
- **OpenDroneMap** / **WebODM** (Module 2, planned)

---

## License

See [LICENSE](LICENSE).

<!-- live demo: https://pavescan-ai-kctjew6jj8tccs79an5dcd.streamlit.app/ — auto-deploy via Streamlit Community Cloud -->
