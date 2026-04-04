"""Map page — interactive map with severity markers and heatmap overlay."""

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.mapping.geo import (
    DEMO_CENTER,
    SEVERITY_COLORS,
    assign_coordinates_to_results,
    build_marker_data,
    compute_heatmap_data,
)

st.set_page_config(page_title="Map | PaveScan AI", layout="wide")
st.title("Inspection Map")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Map Settings")

    show_heatmap = st.toggle("Show Heatmap Layer", value=False)

    severity_filter = st.multiselect(
        "Filter by Severity",
        options=["low", "medium", "high"],
        default=["low", "medium", "high"],
    )

    tile_choice = st.selectbox(
        "Map Style",
        options=["Street", "Satellite", "Topo"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Legend")
    st.markdown(
        f'<span style="color:{SEVERITY_COLORS["high"]}">&#9679;</span> High Severity',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span style="color:{SEVERITY_COLORS["medium"]}">&#9679;</span> Medium Severity',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span style="color:{SEVERITY_COLORS["low"]}">&#9679;</span> Low Severity',
        unsafe_allow_html=True,
    )

# --- Map tile providers ---
TILE_OPTIONS = {
    "Street": {"tiles": "OpenStreetMap"},
    "Satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri",
    },
    "Topo": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "OpenTopoMap",
    },
}

# --- Guard: need detection results first ---
if "detection_results" not in st.session_state or not st.session_state["detection_results"]:
    st.warning("No detection results found. Run detection first.")
    st.page_link("pages/2_Detection.py", label="Go to Detection", icon="🔍")
    st.stop()

detection_results = st.session_state["detection_results"]
uploaded_files = st.session_state.get("uploaded_files", [])

# --- Assign coordinates ---
geo_results, is_demo = assign_coordinates_to_results(detection_results, uploaded_files)

if is_demo:
    st.info(
        "**Demo Mode:** Your images don't contain GPS data. "
        "Showing simulated locations near the University of Toronto campus. "
        "When you use real drone images with GPS coordinates, markers will appear at actual locations."
    )

# --- Build map data ---
all_markers = build_marker_data(geo_results)

# Apply severity filter
markers = [m for m in all_markers if m["severity"] in severity_filter]

# --- Summary metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Defects", len(all_markers))
col2.metric(
    "High Severity",
    len([m for m in all_markers if m["severity"] == "high"]),
)
col3.metric(
    "Medium Severity",
    len([m for m in all_markers if m["severity"] == "medium"]),
)
col4.metric(
    "Low Severity",
    len([m for m in all_markers if m["severity"] == "low"]),
)

st.markdown("---")

# --- Build Folium map ---
if markers:
    # Center on the mean of all marker positions
    center_lat = sum(m["lat"] for m in markers) / len(markers)
    center_lon = sum(m["lon"] for m in markers) / len(markers)
else:
    center_lat, center_lon = DEMO_CENTER

tile_config = TILE_OPTIONS[tile_choice]
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=15,
    **tile_config,
)

# Add severity-colored circle markers with popups
for marker in markers:
    color = SEVERITY_COLORS.get(marker["severity"], "#888888")

    popup_html = f"""
    <div style="font-family: Arial, sans-serif; min-width: 180px;">
        <b style="font-size: 14px;">{marker["class_name"]}</b><br>
        <b>Confidence:</b> {marker["confidence"]:.1%}<br>
        <b>Severity:</b> <span style="color:{color}; font-weight:bold;">{marker["severity"].upper()}</span><br>
        <b>File:</b> {marker["filename"]}<br>
        <b>Area:</b> {marker["area_pixels"]:,} px
    </div>
    """

    folium.CircleMarker(
        location=[marker["lat"], marker["lon"]],
        radius=8 if marker["severity"] == "high" else 6 if marker["severity"] == "medium" else 4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{marker['class_name']} ({marker['severity']})",
    ).add_to(m)

# Optional heatmap layer
if show_heatmap and markers:
    heat_data = compute_heatmap_data(markers)
    if heat_data:
        HeatMap(
            heat_data,
            radius=25,
            blur=15,
            min_opacity=0.4,
            gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
        ).add_to(m)

# Render the map (returned_objects=[] prevents rerun loops / flickering)
st_folium(m, height=600, use_container_width=True, returned_objects=[])

# --- Defect inventory table ---
st.markdown("---")
st.subheader("Defect Inventory")

if markers:
    df = pd.DataFrame([
        {
            "File": m["filename"],
            "Defect Type": m["class_name"],
            "Confidence": f"{m['confidence']:.1%}",
            "Severity": m["severity"].upper(),
            "Latitude": f"{m['lat']:.6f}",
            "Longitude": f"{m['lon']:.6f}",
            "Area (px)": m["area_pixels"],
        }
        for m in markers
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No defects match the current severity filter.")

# --- Navigation footer ---
st.markdown("---")
st.info("Next step: Generate an inspection **Report** with all findings.")
st.page_link("pages/4_Report.py", label="Go to Report", icon="📄")
