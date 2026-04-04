"""Map page — interactive map with detection overlays and heatmap."""

import streamlit as st

st.set_page_config(page_title="Map | PaveScan AI", layout="wide")
st.title("Inspection Map")

st.info("""
**Coming in Phase 3 (Weeks 5-6)**

This page will display an interactive map with:
- Color-coded markers for each detected defect (green/yellow/red by severity)
- Heatmap layer showing damage density across the surveyed area
- Orthomosaic overlay on the base map
- Click-to-inspect popups with defect details

This requires georeferenced images (with GPS coordinates). The map will use
**Folium** for interactive Leaflet maps and **PyDeck** for 3D heatmaps.
""")
