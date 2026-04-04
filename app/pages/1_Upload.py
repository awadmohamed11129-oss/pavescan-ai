"""Upload page — drag and drop drone images for inspection."""

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
st.markdown("Upload drone-captured pavement images for AI inspection.")

# File uploader
uploaded_files = st.file_uploader(
    "Drag and drop images here",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
    help="Upload drone-captured pavement images. Supported formats: JPG, PNG, TIFF.",
)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} image(s)")

    # Store in session state for use in other pages
    st.session_state["uploaded_files"] = uploaded_files

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

            st.image(image_rgb, caption=file.name, use_container_width=True)
            st.caption(f"Size: {image.shape[1]}x{image.shape[0]} px")

    st.markdown("---")
    st.info("Head to the **Detection** page to run AI crack detection on your images.")
else:
    st.markdown("""
    ### No images yet

    Upload drone-captured pavement images to get started. For best results:
    - Use images captured at **25-30m altitude** (nadir/straight-down angle)
    - **JPEG or PNG** format recommended
    - Higher resolution = better crack detection

    **Don't have drone images?** Sample images are included in the `data/sample/` folder.
    """)
