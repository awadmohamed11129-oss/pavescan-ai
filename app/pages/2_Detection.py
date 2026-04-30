"""Detection page — run YOLO crack detection on uploaded images."""

import sys
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.detection.model import (
    SAFETY_PRIORITIES,
    list_available_models,
    load_ensemble_models,
    load_model,
    run_ensemble_inference,
    run_inference,
    summarize_detections,
)

st.set_page_config(page_title="Detection | PaveScan AI", layout="wide")
st.title("AI Crack Detection")

# Sidebar controls
with st.sidebar:
    st.header("Detection Settings")

    # Model mode selector
    mode = st.radio(
        "Detection Mode",
        options=["Ensemble (Both Models)", "Single Model"],
        index=0,
        help="Ensemble runs both models and merges results for best coverage. Single uses one model.",
    )

    available_models = list_available_models()
    model_names = [m["name"] for m in available_models]

    selected_model_path = None
    if mode == "Single Model" and model_names:
        selected_name = st.selectbox(
            "Select Model",
            options=model_names,
            index=0,
            help="Choose which model to run.",
        )
        selected_model_path = next(
            m["path"] for m in available_models if m["name"] == selected_name
        )
        info = next(m for m in available_models if m["name"] == selected_name)
        st.caption(
            f"**Type:** {info['task']}  \n"
            f"**Classes ({info['num_classes']}):** {', '.join(info['classes'])}"
        )

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.9,
        value=0.15,
        step=0.05,
        help="Lower = more detections (possibly noisy). Higher = fewer but more certain.",
    )

    st.markdown("---")
    st.header("Advanced Settings")

    imgsz = st.select_slider(
        "Inference Resolution",
        options=[640, 960, 1280, 1600],
        value=1280,
        help="Higher resolution = more detail = better detection of small cracks. Uses more memory and is slower.",
    )

    use_tta = st.toggle(
        "Test-Time Augmentation (TTA)",
        value=False,
        help="Runs inference multiple times with flips/scales and averages results. ~3-4x slower but ~1-3% more accurate.",
    )

    use_sahi = st.toggle(
        "SAHI Tiled Inference",
        value=False,
        help="Slices large images into overlapping tiles for detection. Critical for finding small cracks in high-res drone images. Much slower.",
    )

    sahi_slice_size = 640
    if use_sahi:
        sahi_slice_size = st.select_slider(
            "SAHI Tile Size",
            options=[320, 480, 640, 800, 1024],
            value=640,
            help="Size of each tile. Smaller = more tiles = catches smaller cracks but slower.",
        )

    # Priority filter
    st.markdown("---")
    st.header("Display Filters")
    priority_filter = st.multiselect(
        "Show Priority Levels",
        options=["critical", "urgent", "monitor", "routine"],
        default=["critical", "urgent", "monitor", "routine"],
        format_func=lambda x: SAFETY_PRIORITIES[x]["label"],
        help="Filter which priority levels to display in results.",
    )

    # Show ensemble info
    if mode == "Ensemble (Both Models)":
        ensemble_models = [m for m in available_models if m["name"] != "yolov8n-seg"]
        if ensemble_models:
            st.markdown("---")
            st.markdown("**Ensemble models:**")
            for m in ensemble_models:
                st.caption(f"- {m['name']} ({m['task']}, {m['num_classes']} classes)")

# Check for uploaded images
uploaded_files = st.session_state.get("uploaded_files", [])

if not uploaded_files:
    st.warning("No images uploaded. Go to the **Upload** page first.")
    st.page_link("pages/1_Upload.py", label="Go to Upload", icon="\U0001f4e4")
    st.stop()

# Load model(s)
with st.spinner("Loading model(s)..."):
    if mode == "Ensemble (Both Models)":
        ensemble, ensemble_paths = load_ensemble_models()
        if not ensemble:
            st.error("No custom models found in models/ directory.")
            st.stop()
        st.success(f"Ensemble loaded: {', '.join(ensemble.keys())}")
    else:
        model = load_model(selected_model_path)
        st.success("Model loaded")

# Run detection button
if st.button("Run Detection on All Images", type="primary"):
    all_results = []
    total_per_model = {}
    total_inference_ms = 0

    progress = st.progress(0, text="Running detection...")

    for i, file in enumerate(uploaded_files):
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        file.seek(0)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if mode == "Ensemble (Both Models)":
            result = run_ensemble_inference(
                ensemble, image,
                confidence=confidence,
                imgsz=imgsz,
                augment=use_tta,
                use_sahi=use_sahi,
                sahi_slice_size=sahi_slice_size,
                model_paths=ensemble_paths,
            )
            # Accumulate per-model counts
            for name, count in result.get("per_model_counts", {}).items():
                total_per_model[name] = total_per_model.get(name, 0) + count
        else:
            result = run_inference(
                model, image,
                confidence=confidence,
                imgsz=imgsz,
                augment=use_tta,
            )

        total_inference_ms += result.get("inference_time_ms", 0)

        all_results.append({
            "filename": file.name,
            "result": result,
        })

        progress.progress(
            (i + 1) / len(uploaded_files),
            text=f"Processing {file.name}... ({i + 1}/{len(uploaded_files)})",
        )

    st.session_state["detection_results"] = all_results
    progress.empty()

    # Show results summary
    total_merged = sum(len(r["result"]["detections"]) for r in all_results)

    # Timing display
    time_str = f"{total_inference_ms / 1000:.1f}s" if total_inference_ms >= 1000 else f"{total_inference_ms:.0f}ms"
    settings_parts = [f"{imgsz}px"]
    if use_tta:
        settings_parts.append("TTA")
    if use_sahi:
        settings_parts.append(f"SAHI {sahi_slice_size}px")
    settings_str = " | ".join(settings_parts)

    if mode == "Ensemble (Both Models)" and total_per_model:
        raw_total = sum(total_per_model.values())
        model_breakdown = ", ".join(f"{n}: {c}" for n, c in total_per_model.items())
        st.success(
            f"Detection complete in **{time_str}** ({settings_str}) — {model_breakdown} | "
            f"After dedup: **{total_merged}** unique detections"
        )
    else:
        st.success(
            f"Detection complete in **{time_str}** ({settings_str}) — "
            f"{total_merged} detections across {len(uploaded_files)} image(s)"
        )

# Display results
if "detection_results" in st.session_state:
    results = st.session_state["detection_results"]

    # Summary metrics
    all_dets = [d for r in results for d in r["result"]["detections"]]
    summary = summarize_detections(all_dets)

    # ==========================================
    # SAFETY ALERT BANNERS
    # ==========================================
    by_priority = summary.get("by_priority", {})

    if by_priority.get("critical", 0) > 0:
        st.error(
            f"\u26a0\ufe0f **CRITICAL SAFETY ALERT** — "
            f"{by_priority['critical']} critical defect(s) detected requiring immediate attention. "
            f"These pose direct danger to road users."
        )

    if by_priority.get("urgent", 0) > 0:
        st.warning(
            f"\U0001f7e0 **URGENT** — "
            f"{by_priority['urgent']} defect(s) require priority maintenance within 1-2 weeks."
        )

    if summary.get("cluster_risk", False):
        st.error(
            "\u26a0\ufe0f **SYSTEMIC PAVEMENT FAILURE DETECTED** — "
            "High defect density indicates structural failure across this section. "
            "Recommend full section assessment and possible rehabilitation rather than spot repairs."
        )

    # ==========================================
    # SEVERITY METRICS ROW
    # ==========================================
    st.subheader("Severity Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Detections", summary["total_detections"])
    col2.metric("High Severity", summary["by_severity"]["high"])
    col3.metric("Medium Severity", summary["by_severity"]["medium"])
    col4.metric("Low Severity", summary["by_severity"]["low"])

    # ==========================================
    # PRIORITY METRICS ROW
    # ==========================================
    st.subheader("Safety Priority")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)

    critical_count = by_priority.get("critical", 0)
    urgent_count = by_priority.get("urgent", 0)
    monitor_count = by_priority.get("monitor", 0)
    routine_count = by_priority.get("routine", 0)

    pcol1.markdown(
        f"<div style='text-align:center;padding:8px;background:#FF1744;color:white;border-radius:8px;'>"
        f"<b>CRITICAL</b><br><span style='font-size:2em;'>{critical_count}</span></div>",
        unsafe_allow_html=True,
    )
    pcol2.markdown(
        f"<div style='text-align:center;padding:8px;background:#FF9100;color:white;border-radius:8px;'>"
        f"<b>URGENT</b><br><span style='font-size:2em;'>{urgent_count}</span></div>",
        unsafe_allow_html=True,
    )
    pcol3.markdown(
        f"<div style='text-align:center;padding:8px;background:#FFC400;color:#333;border-radius:8px;'>"
        f"<b>MONITOR</b><br><span style='font-size:2em;'>{monitor_count}</span></div>",
        unsafe_allow_html=True,
    )
    pcol4.markdown(
        f"<div style='text-align:center;padding:8px;background:#00C853;color:white;border-radius:8px;'>"
        f"<b>ROUTINE</b><br><span style='font-size:2em;'>{routine_count}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ==========================================
    # CHARTS
    # ==========================================
    if summary["total_detections"] > 0:
        col_chart, col_class = st.columns(2)

        with col_chart:
            fig = go.Figure(data=[go.Pie(
                labels=["Critical", "Urgent", "Monitor", "Routine"],
                values=[critical_count, urgent_count, monitor_count, routine_count],
                marker_colors=["#FF1744", "#FF9100", "#FFC400", "#00C853"],
                hole=0.4,
            )])
            fig.update_layout(title="Safety Priority Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col_class:
            if summary["by_class"]:
                fig2 = go.Figure(data=[go.Bar(
                    x=list(summary["by_class"].keys()),
                    y=list(summary["by_class"].values()),
                    marker_color="#FF4B4B",
                )])
                fig2.update_layout(title="Detections by Class")
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # PER-IMAGE RESULTS (sorted by priority)
    # ==========================================
    priority_order = {"critical": 0, "urgent": 1, "monitor": 2, "routine": 3}

    for item in results:
        dets = item["result"]["detections"]

        # Filter by selected priority levels
        filtered_dets = [
            d for d in dets
            if d.get("safety_priority", "routine") in priority_filter
        ]

        with st.expander(
            f"\U0001f4f7 {item['filename']} — {len(filtered_dets)} detections"
            + (f" ({len(dets) - len(filtered_dets)} filtered out)" if len(filtered_dets) < len(dets) else "")
        ):
            annotated = item["result"]["annotated_image"]
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)

            # Per-image timing
            img_time = item["result"].get("inference_time_ms", 0)
            if img_time > 0:
                st.caption(f"Inference time: {img_time:.0f}ms")

            if filtered_dets:
                # Sort by priority (critical first), then by severity score (highest first)
                sorted_dets = sorted(
                    filtered_dets,
                    key=lambda d: (
                        priority_order.get(d.get("safety_priority", "routine"), 99),
                        -d.get("severity_score", 0),
                    ),
                )

                for det in sorted_dets:
                    priority_key = det.get("safety_priority", "routine")
                    priority_info = SAFETY_PRIORITIES.get(priority_key, SAFETY_PRIORITIES["routine"])
                    severity = det.get("severity", "low")
                    severity_score = det.get("severity_score", 0.0)
                    width_px = det.get("width_pixels", 0.0)
                    deterioration = det.get("deterioration_risk", {})

                    # Priority badge
                    badge_color = priority_info["color_hex"]
                    badge_text = priority_info["label"]
                    text_color = "#333" if priority_key == "monitor" else "white"

                    st.markdown(
                        f"<div style='border-left:4px solid {badge_color};padding:8px 12px;margin:4px 0;background:#f8f8f8;border-radius:4px;'>"
                        f"<span style='background:{badge_color};color:{text_color};padding:2px 8px;border-radius:4px;font-weight:bold;font-size:0.8em;'>"
                        f"{badge_text}</span> "
                        f"<b>{det['class_name']}</b> &mdash; "
                        f"Confidence: {det['confidence']:.1%} &mdash; "
                        f"Severity: {severity.upper()} ({severity_score:.2f}) &mdash; "
                        f"Area: {det['area_pixels']:,} px"
                        + (f" &mdash; Width: {width_px:.0f} px" if width_px > 0 else "")
                        + f"<br><small style='color:#666;'>"
                        f"\U0001f527 {det.get('recommended_action', '')}"
                        + (f" &mdash; \u26a0\ufe0f {deterioration.get('warning', '')}" if deterioration.get("warning") else "")
                        + f"</small></div>",
                        unsafe_allow_html=True,
                    )

    st.markdown("---")
    st.info("Head to the **Map** page to see detections on an interactive map.")
