"""Detection page — run YOLOv8 crack detection on uploaded images."""

import sys
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.detection.model import load_model, run_inference, summarize_detections

st.set_page_config(page_title="Detection | PaveScan AI", layout="wide")
st.title("AI Crack Detection")

# Sidebar controls
with st.sidebar:
    st.header("Detection Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Minimum confidence for a detection to be shown. Lower = more detections (possibly noisy).",
    )

    model_path = st.text_input(
        "Custom Model Path (optional)",
        placeholder="models/best.pt",
        help="Path to a custom-trained YOLOv8 .pt file. Leave blank for default.",
    )

# Check for uploaded images
uploaded_files = st.session_state.get("uploaded_files", [])

if not uploaded_files:
    st.warning("No images uploaded. Go to the **Upload** page first.")
    st.page_link("pages/1_Upload.py", label="Go to Upload", icon="📤")
    st.stop()

# Load model
with st.spinner("Loading YOLOv8 model..."):
    model = load_model(model_path if model_path else None)
st.success("Model loaded")

# Run detection button
if st.button("Run Detection on All Images", type="primary"):
    all_detections = []

    progress = st.progress(0, text="Running detection...")

    for i, file in enumerate(uploaded_files):
        # Read image
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        file.seek(0)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run inference
        result = run_inference(model, image, confidence=confidence)

        # Store results
        all_detections.append({
            "filename": file.name,
            "result": result,
        })

        progress.progress(
            (i + 1) / len(uploaded_files),
            text=f"Processing {file.name}... ({i + 1}/{len(uploaded_files)})",
        )

    # Save to session state
    st.session_state["detection_results"] = all_detections
    progress.empty()
    st.success(f"Detection complete on {len(uploaded_files)} image(s)")

# Display results
if "detection_results" in st.session_state:
    results = st.session_state["detection_results"]

    # Summary metrics
    total_dets = sum(
        len(r["result"]["detections"]) for r in results
    )
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
                for j, det in enumerate(item["result"]["detections"]):
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
