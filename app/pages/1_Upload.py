"""Upload page — drop in pavement photos for inspection."""

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Add project root to path so we can import src modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Upload | PaveScan AI", layout="wide")
st.title("Upload Images")
st.markdown("Upload pavement photos from a phone, dashcam, or any camera.")

# File uploader
uploaded_files = st.file_uploader(
    "Drag and drop images here",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
    help="Upload pavement photos. Supported formats: JPG, PNG, TIFF.",
)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} image(s)")

    # Store in session state for use in other pages
    prior = st.session_state.get("uploaded_files", [])
    prior_names = {f.name for f in prior}
    new_names = {f.name for f in uploaded_files}
    st.session_state["uploaded_files"] = uploaded_files

    # If the upload set changed, drop any stale detection results + overrides so
    # cluster_id-based override keys don't bleed across different image batches.
    if prior_names != new_names:
        st.session_state["priority_overrides"] = {}
        for k in (
            "detection_results",
            "detection_results_v1",
            "detection_results_v2",
            "compare_mode",
            "compare_active_version",
            "compare_timing",
            "report_html",
            "report_pdf",
        ):
            st.session_state.pop(k, None)

    # Preview images in a grid
    cols = st.columns(min(len(uploaded_files), 3))
    for i, file in enumerate(uploaded_files):
        col = cols[i % 3]
        with col:
            # Read image bytes and decode
            file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
            file.seek(0)  # Reset for later use
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            st.image(image_rgb, caption=file.name, width="stretch", output_format="PNG")
            st.caption(f"Size: {image.shape[1]}x{image.shape[0]} px")

    st.markdown("---")
    st.info("Head to the **Detection** page to run AI crack detection on your images.")
else:
    st.markdown("""
    ### No images yet

    Upload pavement photos to get started. For best results:
    - Shoot **perpendicular to the road surface** when possible (top-down framing reads cracks best)
    - **JPEG or PNG** format recommended
    - Higher resolution = better crack detection
    - Good light beats high resolution in bad light

    **Don't have photos yet?** Sample images are included in the `data/sample/` folder.
    """)
