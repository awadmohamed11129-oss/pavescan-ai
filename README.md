# PaveScan AI

**Automated pavement-defect inspection from phone or dashcam footage. YOLO11 ensemble plus ASTM D6433 PCI scoring plus printable PDF reports.**

**[Live demo →](https://pavescan-ai-kctjew6jj8tccs79an5dcd.streamlit.app/)** · [Source on GitHub](https://github.com/awadmohamed11129-oss/pavescan-ai)

_Live demo runs on Streamlit Community Cloud's free tier. Recruiters: one-time Google or GitHub sign-in is the platform's anti-abuse gate, not a paywall._

A unified civil engineering platform that ingests pavement imagery (phone, dashcam, or drone), runs an ensemble of YOLO11 models for crack and pothole detection, scores severity via a multi-factor system informed by ASTM D6433, and produces interactive maps and engineering-grade PDF reports.

---

## At a glance

```
┌────────────┐   ┌─────────────────┐   ┌────────────────────────┐   ┌────────────┐   ┌──────────┐
│  Phone or  │──▶│  Upload images  │──▶│  Ensemble detection    │──▶│  Map +     │──▶│  PDF     │
│  dashcam   │   │  (Streamlit)    │   │  (YOLO11 seg + det)    │   │  filters   │   │  report  │
└────────────┘   └─────────────────┘   │  + SAHI + TTA + WBF    │   └────────────┘   └──────────┘
                                        │                        │
                                        │  ASTM D6433 severity   │
                                        │  + safety priority     │
                                        └────────────────────────┘
```

---

## Module 1 — pavement inspection (built)

**Four pages, plus a legacy flight-planning module in the codebase:**

1. **Upload:** drag-and-drop images (JPG/PNG/TIF, 200 MB cap), session-scoped storage.
2. **Detection:** ensemble inference with three orthogonal accuracy levers:
   - **SAHI** tiled inference for high-resolution photos
   - **TTA** (test-time augmentation) for harder cases
   - **WBF** (Weighted Box Fusion) for ensemble fusion: averages overlapping cross-model boxes weighted by confidence rather than dropping the loser
   - Inference resolution slider (640 / 960 / 1280 / 1600)
3. **Map:** Folium interactive map with priority-sized markers, severity heatmap, multi-style basemaps. Falls back to Toronto demo coords when EXIF GPS is absent (clearly signposted in UI).
4. **Report:** WeasyPrint PDF with safety assessment, ASTM D6433 PCI score, defect inventory, maintenance plan grouped by urgency, methodology section.
5. **Flight planning module** _(hidden from sidebar; preserved in repo at `src/mission/` and `app/pages/_5_Mission.py`)_: built during the original drone-hardware iteration of this project. The drone build was shelved in favour of phone and dashcam capture; the module stays in the repo as a record of the engineering pivot.

### Multi-factor severity (ASTM D6433-informed)

The severity classifier is **not** a single area threshold — it composes four signals:

| Factor          | Weight (seg path) | Weight (det path) | What it captures                                  |
| --------------- | ----------------- | ----------------- | ------------------------------------------------- |
| Defect type     | 25%               | 35%               | D40/Pothole = 1.0, D20/Alligator = 0.90, etc.     |
| Area ratio      | 45%               | 65%               | Continuous curve, not bins                        |
| Estimated width | 30%               | —                 | From cv2.minAreaRect on segmentation mask         |
| Confidence      | 15% (penalty)     | 15% (penalty)     | Down-weights uncertain detections                 |

Plus density bonus (3+ defects → 1.08×, 5+ → 1.15×) and a **safety floor**: structural failures (potholes, alligator cracking) at confidence ≥ 0.20 always score ≥ medium, regardless of size, because a small pothole is still a tire blowout.

### Safety priority (separate from severity)

Severity asks "how bad is this defect?" Priority asks "what gets fixed FIRST?" Four levels:

- **CRITICAL:** emergency repair, 24–48 hr (e.g., high-severity pothole)
- **URGENT:** within 1–2 weeks (e.g., severe alligator cracking, systemic failure cluster)
- **ROUTINE:** within 1–3 months (typical preventive maintenance)
- **MONITOR:** track at next scheduled inspection

PDF reports group recommended fixes by these urgency timelines.

### PCI calculation

Implements the ASTM D6433 framework with **type-weighted deduct values** and **corrected deduct values** (`CDV = max(1.0 − 0.08·i, 0.20)` for the i-th highest deduct). Pothole high = 25 deduct, alligator high = 20, longitudinal high = 12, transverse high = 10.

---

## Accuracy

### Model lineup

| Slot                  | v1 (current `models/`)              | v2 (after Colab training)                |
| --------------------- | ----------------------------------- | ---------------------------------------- |
| Crack segmentation    | `pavescan_crack_seg.pt`             | `pavescan_crack_seg_v2.pt`               |
| Architecture          | YOLOv8n-seg (3.26M params, 640px)   | YOLO11l-seg (A100), 1280px               |
| Training              | 50 epochs, default aug              | 200 epochs, multi-scale, aggressive aug  |
| Road damage detection | `pavescan_rdd2022.pt`               | `pavescan_rdd2022_v2.pt`                 |
| Architecture          | YOLOv8n (640px, 30 epochs trained)  | YOLO11l, 1280px, 200 epochs, fl_gamma=1.5 |

### Crack-Seg val (200 imgs, 249 instances)

| Metric             | v1 (YOLOv8n-seg, 640px) | v2 (YOLO11l, 1280px)        |
| ------------------ | ----------------------- | --------------------------- |
| Box mAP50          | **0.818**               | **0.816** _(≈ V1)_          |
| Box mAP50-95       | **0.636**               | _pending_                   |
| Seg mAP50          | **0.658**               | **0.395**                   |
| Seg mAP50-95       | **0.226**               | _pending_                   |
| Box Precision      | 0.861                   | _pending_                   |
| Box Recall         | 0.744                   | _pending_                   |

v1 numbers were generated locally on CPU using `ultralytics` 8.4.36 with default confidence threshold; raw output is in `models/metrics_v1_seg.json`. V2 numbers come from the YOLO11l 1280-pixel run on 2026-05-03 (see `notebooks/train_crack_detector.ipynb` for the full training record). No `metrics_v2_seg.json` is currently committed; values are reported from the training console output. Box detection held steady from V1 to V2. Segmentation mAP50 dropped from V1's 0.658 to V2's 0.395, meaning the larger model on 1280px inputs did not converge on the mask head within 200 epochs. Next training round: AMP-off baseline, longer schedule, lighter augmentation. The full incident write-up (including the AMP/EMA checkpoint corruption that ate the first V2 run) is on the portfolio site at https://mohamadawad.vercel.app/projects/pavescan-ai.

### How v2 metrics are produced

The training notebook (`notebooks/train_crack_detector.ipynb`) includes a **v1-vs-v2 comparison cell** for each option. After training:

1. Loads the trained v2 model, runs `.val()` on the same val split
2. If you upload your v1 weights to `/content/`, also runs `.val()` on v1 with identical settings
3. Prints a delta table and writes `metrics_v2_seg.json` (or `_det.json`)
4. Auto-downloads both the `.pt` and the `.json` to your machine

This is the only way to make accuracy claims that survive scrutiny: the val set is fixed, the only difference is the model weights.

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
│       └── _5_Mission.py         # Flight planning (hidden from sidebar after drone pivot — see Module 1)
├── src/
│   ├── detection/
│   │   ├── model.py              # Multi-factor severity, safety priority, ensemble fusion
│   │   └── clustering.py         # Defect-cluster grouping for the inspector view
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
│   ├── fetch_weights.py          # Downloads model weights from the GitHub release on first start
│   ├── smoke_test_safety.py      # End-to-end pipeline regression test
│   ├── smoke_test_report.py      # PDF generation regression test
│   └── smoke_test_clustering.py  # Cluster-grouping regression test
├── models/                       # .pt files (gitignored) + metrics_v*.json (committed)
├── data/sample/                  # Sample pavement images
├── packages.txt                  # apt packages for Streamlit Cloud build
├── requirements.txt              # Pinned (==) for cloud reproducibility
└── requirements.lock.txt         # Full transitive snapshot (human reference)
```

---

## Install & run

### Try it online

The live dashboard at https://pavescan-ai-kctjew6jj8tccs79an5dcd.streamlit.app/ runs the full pipeline. Upload a pavement photo, walk Detection → Map → Report, download the PDF. No install needed.

### Local install

**Requires:** Python 3.14, ~5 GB disk for models + datasets.

```bash
git clone https://github.com/awadmohamed11129-oss/pavescan-ai
cd pavescan-ai

python -m venv .venv
source .venv/Scripts/activate    # Windows Git Bash
# or .venv\Scripts\activate.bat for cmd

pip install -r requirements.txt

# First launch of dashboard.py downloads model weights from the GitHub release
# (~66 MB, sha256-verified). Or fetch them upfront:
python scripts/fetch_weights.py    # idempotent; safe to skip if running dashboard.py directly

# Launch dashboard — ALWAYS from dashboard.py, never individual pages
streamlit run app/dashboard.py
```

Open http://localhost:8501. Upload a pavement image, click through Detection → Map → Report.

### Smoke tests

```bash
python scripts/smoke_test_safety.py     # exits 0 if pipeline is healthy
python scripts/smoke_test_report.py     # exits 0 if PDF generation works
python scripts/smoke_test_clustering.py # exits 0 if cluster grouping is wired
```

### Training v2 models (Google Colab)

1. Open `notebooks/train_crack_detector.ipynb` in Colab
2. **Runtime → Change runtime type → A100 GPU** (Pro recommended) or **T4** (free)
3. **(Optional)** Drag your existing v1 `.pt` files into `/content/` so the notebook can produce a v1-vs-v2 comparison table
4. In the `TRAINING_OPTION` cell, set to `"A"` (crack segmentation) or `"C"` (RDD2022 detection, US+Czech), or `"B"` (RDD2022 all 6 countries — Pro only)
5. **Runtime → Run All**, walk away (~45 min – 4 hr depending on option and GPU)
6. The trained `.pt` and a `metrics_v2_*.json` auto-download — drop the `.pt` in `models/` and the JSON in repo root

The detection code in `src/detection/model.py` already auto-prefers v2 weights with v1 as fallback.

---

## Deployment

The repo is wired for Streamlit Community Cloud. Every push to `master` triggers a rebuild within ~3 minutes.

- `requirements.txt`: pinned (`==`) for cloud reproducibility. The local `requirements.lock.txt` captures the full transitive snapshot for human reference.
- `packages.txt`: apt packages Streamlit Cloud installs at build time (`libgl1`, `libglib2.0-0t64`, `libcairo2-dev`, `pkg-config`).
- `scripts/fetch_weights.py`: downloads the three model weights from the `weights-v1.0` GitHub release on first start; sha256-verified against pinned hashes.

See the GitHub release at https://github.com/awadmohamed11129-oss/pavescan-ai/releases/tag/weights-v1.0 if you want to verify hashes or grab the weights directly.

---

## Methodology

- **Detection:** YOLO11 ensemble — a 1-class crack segmentation model (pixel masks) + a 4-class road damage detection model (Longitudinal / Transverse / Alligator / Pothole). The two models contribute complementary information; SAHI handles small cracks in high-resolution photos; TTA is a final accuracy lever for hard cases.
- **Ensemble fusion:** Two strategies. **IoU-dedup** (default) keeps the highest-confidence box and merges metadata. **WBF** averages overlapping boxes weighted by confidence. WBF pre-harmonizes generic "crack" labels to specific classes (Pothole/Alligator/etc.) so the fusion can merge across-model agreements rather than producing duplicates.
- **Severity:** multi-factor scoring described above, calibrated to the ASTM D6433 distress identification manual.
- **Safety priority:** orthogonal axis to severity — encodes "what must be fixed first" so a small pothole doesn't get triaged as routine despite its low area ratio.
- **PCI:** 100 − sum of deduct values, with corrected deduct values applied to handle multiple defects on the same section.

---

## Known limitations

- **Width estimation is in pixels, not millimeters.** A 25-pixel-wide crack scored at one mounting height vs another is the same defect at very different real-world widths. The current severity score is calibrated for phone-from-windshield or dashcam mounting height (~1.5 m, wide-angle FoV); deploy at radically different capture geometries and severity will drift. Roadmap: add a manual reference-length calibration UI (click two endpoints on a known-length object once per session, mm/px persists in session state).
- **No real-world capture data yet.** All sample images are from public RDD2022 and Crack-Seg datasets. The next milestone is real dashcam footage from a Toronto street loop, scored against a manual PCI walk-through for ground truth.
- **Custom drone build shelved.** The original roadmap included a custom F450 build for aerial capture; the project pivoted to dashcam and phone capture, which is cheaper, faster to iterate, and matches the cost-of-survey problem better than dedicated drone time per kilometre. The flight-planning module is preserved in `src/mission/` as a record of the pivot.
- **Module 2 (site mapping & surveying) is planned, not built.** OpenDroneMap orthomosaic viewer, click-to-measure tools, before/after comparison, and survey-PDF generation are tracked as the next major chunk of work.
- **No formal test suite yet.** Smoke tests cover the integration path; unit tests are TODO.

---

## Module 2 roadmap (planned)

A second platform module for **site mapping and surveying** is on the roadmap. It is the natural complement to Module 1's pavement inspection:

1. Upload overlapping aerial or drive-by survey photos
2. Process into orthomosaics via **OpenDroneMap** (we use it the way we use YOLOv8: train and integrate, don't reinvent)
3. Display georeferenced orthomosaic on the map
4. Measurement tools: distance, area, elevation profiles
5. Before/after comparison of surveys from different dates (construction progress monitoring)
6. Survey PDF reports
7. **The killer integration:** capture grid (aerial or drive-by) → orthomosaic → run Module 1 detection ON the orthomosaic → full-road crack map with PCI overlaid on georeferenced coordinates

---

## Tech stack

- **Python** 3.14, **Streamlit** 1.56+, **Ultralytics** 8.3+, **PyTorch**, **OpenCV**
- **SAHI** for tiled inference, **ensemble-boxes** for WBF
- **Folium** + **streamlit-folium** for maps
- **WeasyPrint** + **xhtml2pdf** for PDF reports
- **Plotly** for charts
- **Streamlit Community Cloud** for hosting + auto-redeploy
- **OpenDroneMap** / **WebODM** (Module 2, planned)

---

## License

See [LICENSE](LICENSE).
