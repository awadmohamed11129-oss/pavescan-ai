"""Detection page — run YOLOv8 crack detection on uploaded images."""

import sys
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.detection.model import (
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

    # Show ensemble info
    if mode == "Ensemble (Both Models)":
        ensemble_models = [m for m in available_models if m["name"] != "yolov8n-seg"]
        if ensemble_models:
            st.markdown("**Ensemble models:**")
            for m in ensemble_models:
                st.caption(f"- {m['name']} ({m['task']}, {m['num_classes']} classes)")

# Check for uploaded images
uploaded_files = st.session_state.get("uploaded_files", [])

if not uploaded_files:
    st.warning("No images uploaded. Go to the **Upload** page first.")
    st.page_link("pages/1_Upload.py", label="Go to Upload", icon="📤")
    st.stop()

# Load model(s)
with st.spinner("Loading model(s)..."):
    if mode == "Ensemble (Both Models)":
        ensemble = load_ensemble_models()
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

    progress = st.progress(0, text="Running detection...")

    for i, file in enumerate(uploaded_files):
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        file.seek(0)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if mode == "Ensemble (Both Models)":
            result = run_ensemble_inference(ensemble, image, confidence=confidence)
            # Accumulate per-model counts
            for name, count in result.get("per_model_counts", {}).items():
                total_per_model[name] = total_per_model.get(name, 0) + count
        else:
            result = run_inference(model, image, confidence=confidence)

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

    # Show ensemble stats
    total_merged = sum(len(r["result"]["detections"]) for r in all_results)
    if mode == "Ensemble (Both Models)" and total_per_model:
        raw_total = sum(total_per_model.values())
        model_breakdown = ", ".join(f"{n}: {c}" for n, c in total_per_model.items())
        st.success(
            f"Detection complete — {model_breakdown} | "
            f"After dedup: **{total_merged}** unique detections"
        )
    else:
        st.success(f"Detection complete — {total_merged} detections across {len(uploaded_files)} image(s)")

# Display results
if "detection_results" in st.session_state:
    results = st.session_state["detection_results"]

    # Summary metrics
    all_dets = [d for r in results for d in r["result"]["detections"]]
    summary = summarize_detections(all_dets)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Detections", summary["total_detections"])
    col2.metric("High Severity", summary["by_severity"]["high"])
    col3.metric("Medium Severity", summary["by_severity"]["medium"])
    col4.metric("Low Severity", summary["by_severity"]["low"])

    st.markdown("---")

    # Severity distribution chart
    if summary["total_detections"] > 0:
        col_chart, col_class = st.columns(2)

        with col_chart:
            fig = go.Figure(data=[go.Pie(
                labels=["Low", "Medium", "High"],
                values=[
                    summary["by_severity"]["low"],
                    summary["by_severity"]["medium"],
                    summary["by_severity"]["high"],
                ],
                marker_colors=["#00C853", "#FFD600", "#FF1744"],
                hole=0.4,
            )])
            fig.update_layout(title="Severity Distribution")
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

    # Per-image results
    for item in results:
        with st.expander(f"📷 {item['filename']} — {len(item['result']['detections'])} detections"):
            annotated = item["result"]["annotated_image"]
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)

            if item["result"]["detections"]:
                for det in item["result"]["detections"]:
                    severity_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                    st.markdown(
                        f"{severity_emoji[det['severity']]} "
                        f"**{det['class_name']}** — "
                        f"Confidence: {det['confidence']:.1%} — "
                        f"Severity: {det['severity'].upper()} — "
                        f"Area: {det['area_pixels']:,} px"
                    )

    st.markdown("---")
    st.info("Head to the **Map** page to see detections on an interactive map.")
