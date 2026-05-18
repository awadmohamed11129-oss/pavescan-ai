"""PaveScan AI — Main Streamlit Dashboard."""

from pathlib import Path

import streamlit as st

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_AVAILABLE_VERSIONS = []
if (_MODELS_DIR / "pavescan_crack_seg.pt").exists():
    _AVAILABLE_VERSIONS.append("V1")
if (_MODELS_DIR / "pavescan_crack_seg_v2.pt").exists():
    _AVAILABLE_VERSIONS.append("V2")
_VERSIONS_LABEL = "+".join(_AVAILABLE_VERSIONS) if _AVAILABLE_VERSIONS else "none"

st.set_page_config(
    page_title="PaveScan AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("PaveScan AI")
st.subheader("Automated pavement inspection from phone or dashcam footage")

st.markdown("---")

# Overview cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Pipeline", "Upload → Detect")
    st.caption("Drop in pavement photos")

with col2:
    st.metric("AI Model", f"YOLO11-seg ({_VERSIONS_LABEL})")
    st.caption("Crack detection & segmentation")

with col3:
    st.metric("Standard", "ASTM D6433")
    st.caption("Pavement Condition Index")

with col4:
    st.metric("Output", "PDF Report")
    st.caption("Professional inspection report")

st.markdown("---")

st.markdown("""
### How It Works

1. **Upload** — Drop in pavement photos from a phone, dashcam, or any camera
2. **Detect** — AI identifies cracks, potholes, and surface defects
3. **Map** — View detections on an interactive map with severity heatmap
4. **Report** — Generate a professional PDF inspection report with PCI scores

### Get Started

Use the **sidebar** to navigate between pages, or click below:
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link("pages/1_Upload.py", label="Upload Images", icon="📤")
with col2:
    st.page_link("pages/2_Detection.py", label="Run Detection", icon="🔍")
with col3:
    st.page_link("pages/3_Map.py", label="View Map", icon="🗺️")
with col4:
    st.page_link("pages/4_Report.py", label="Generate Report", icon="📄")

st.markdown("---")
st.caption("PaveScan AI | Built with YOLO11, Streamlit, and Python | ASTM D6433 Compliant")
