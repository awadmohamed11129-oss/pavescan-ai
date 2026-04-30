"""PaveScan AI — Main Streamlit Dashboard."""

import streamlit as st

st.set_page_config(
    page_title="PaveScan AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("PaveScan AI")
st.subheader("Drone-Based Automated Pavement Inspection System")

st.markdown("---")

# Overview cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Mission", "Route Planner")
    st.caption("Plan autonomous survey flights")

with col2:
    st.metric("Pipeline", "Upload → Detect")
    st.caption("Drag-and-drop drone images")

with col3:
    st.metric("AI Model", "YOLO11-seg")
    st.caption("Crack detection & segmentation")

with col4:
    st.metric("Standard", "ASTM D6433")
    st.caption("Pavement Condition Index")

with col5:
    st.metric("Output", "PDF Report")
    st.caption("Professional inspection report")

st.markdown("---")

st.markdown("""
### How It Works

1. **Plan** — Draw a flight route and configure mission parameters
2. **Upload** — Drag and drop drone images or an orthomosaic
3. **Detect** — AI identifies cracks, potholes, and surface defects
4. **Map** — View detections on an interactive map with severity heatmap
5. **Report** — Generate a professional PDF inspection report with PCI scores

### Get Started

Use the **sidebar** to navigate between pages, or click below:
""")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.page_link("pages/5_Mission.py", label="Plan Mission", icon="✈️")
with col2:
    st.page_link("pages/1_Upload.py", label="Upload Images", icon="📤")
with col3:
    st.page_link("pages/2_Detection.py", label="Run Detection", icon="🔍")
with col4:
    st.page_link("pages/3_Map.py", label="View Map", icon="🗺️")
with col5:
    st.page_link("pages/4_Report.py", label="Generate Report", icon="📄")

st.markdown("---")
st.caption("PaveScan AI | Built with YOLO11, Streamlit, and Python | ASTM D6433 Compliant")
