"""Map page — interactive map with priority-colored markers and heatmap overlay."""

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.detection.model import SAFETY_PRIORITIES
from src.mapping.geo import (
    DEMO_CENTER,
    PRIORITY_COLORS,
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

    priority_filter = st.multiselect(
        "Filter by Priority",
        options=["critical", "urgent", "monitor", "routine"],
        default=["critical", "urgent", "monitor", "routine"],
        format_func=lambda x: SAFETY_PRIORITIES[x]["label"],
    )

    tile_choice = st.selectbox(
        "Map Style",
        options=["Street", "Satellite", "Topo"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Priority Legend")
    for key in ["critical", "urgent", "monitor", "routine"]:
        color = PRIORITY_COLORS[key]
        label = SAFETY_PRIORITIES[key]["label"]
        timeline = SAFETY_PRIORITIES[key]["timeline"]
        st.markdown(
            f'<span style="color:{color}; font-size:1.4em;">&#9679;</span> '
            f'**{label}** — {timeline}',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Severity Legend")
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
    st.page_link("pages/2_Detection.py", label="Go to Detection", icon="\U0001f50d")
    st.stop()

detection_results = st.session_state["detection_results"]
uploaded_files = st.session_state.get("uploaded_files", [])

# --- Assign coordinates ---
geo_results, is_demo = assign_coordinates_to_results(detection_results, uploaded_files)

if is_demo:
    st.info(
        "**Demo Mode:** Your images don't contain GPS data. "
        "Showing simulated locations near the University of Toronto campus. "
        "When your photos contain GPS metadata (most phone cameras embed it by default), markers will appear at actual locations."
    )

# --- Build map data ---
all_markers = build_marker_data(geo_results)

# Apply severity + priority filters
markers = [
    m for m in all_markers
    if m["severity"] in severity_filter
    and m.get("safety_priority", "routine") in priority_filter
]

# --- Summary metrics ---
st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Defects", len(all_markers))

# Priority counts
critical_count = len([m for m in all_markers if m.get("safety_priority") == "critical"])
urgent_count = len([m for m in all_markers if m.get("safety_priority") == "urgent"])
monitor_count = len([m for m in all_markers if m.get("safety_priority") == "monitor"])

col2.metric("Critical", critical_count)
col3.metric("Urgent", urgent_count)
col4.metric("Monitor / Routine", monitor_count + len(all_markers) - critical_count - urgent_count - monitor_count)

st.markdown("---")

# --- Build Folium map ---
if markers:
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

# Marker sizing by priority: critical = largest and most visible
PRIORITY_RADIUS = {"critical": 14, "urgent": 10, "monitor": 6, "routine": 4}

# Add priority-colored circle markers with enriched popups
for marker in markers:
    priority_key = marker.get("safety_priority", "routine")
    color = PRIORITY_COLORS.get(priority_key, "#888888")
    radius = PRIORITY_RADIUS.get(priority_key, 6)
    priority_label = SAFETY_PRIORITIES.get(priority_key, {}).get("label", "ROUTINE")
    severity = marker.get("severity", "low")
    action = marker.get("recommended_action", "")
    deterioration = marker.get("deterioration_risk", {})
    deterioration_warning = deterioration.get("warning", "")

    popup_html = f"""
    <div style="font-family: Arial, sans-serif; min-width: 200px;">
        <b style="font-size: 14px;">{marker["class_name"]}</b><br>
        <span style="background:{color};color:white;padding:2px 6px;border-radius:3px;font-weight:bold;font-size:0.85em;">
            {priority_label}
        </span><br><br>
        <b>Confidence:</b> {marker["confidence"]:.1%}<br>
        <b>Severity:</b> {severity.upper()} ({marker.get("severity_score", 0):.2f})<br>
        <b>File:</b> {marker["filename"]}<br>
        <b>Area:</b> {marker["area_pixels"]:,} px<br>
    """
    if marker.get("width_pixels", 0) > 0:
        popup_html += f"<b>Width:</b> {marker['width_pixels']:.0f} px<br>"
    if action:
        popup_html += f"<br><b>Action:</b> {action}<br>"
    if deterioration_warning:
        popup_html += f"<b>Risk:</b> {deterioration_warning}<br>"
    popup_html += "</div>"

    folium.CircleMarker(
        location=[marker["lat"], marker["lon"]],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=f"{marker['class_name']} [{priority_label}]",
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

# Render the map
st_folium(m, height=600, use_container_width=True, returned_objects=[])

# --- Defect inventory table ---
st.markdown("---")
st.subheader("Defect Inventory")

if markers:
    df = pd.DataFrame([
        {
            "File": mk["filename"],
            "Defect Type": mk["class_name"],
            "Priority": SAFETY_PRIORITIES.get(mk.get("safety_priority", "routine"), {}).get("label", "ROUTINE"),
            "Confidence": f"{mk['confidence']:.1%}",
            "Severity": mk["severity"].upper(),
            "Score": f"{mk.get('severity_score', 0):.2f}",
            "Width (px)": f"{mk.get('width_pixels', 0):.0f}" if mk.get("width_pixels", 0) > 0 else "-",
            "Latitude": f"{mk['lat']:.6f}",
            "Longitude": f"{mk['lon']:.6f}",
            "Area (px)": mk["area_pixels"],
        }
        for mk in markers
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No defects match the current filters.")

# --- Navigation footer ---
st.markdown("---")
st.info("Next step: Generate an inspection **Report** with all findings.")
st.page_link("pages/4_Report.py", label="Go to Report", icon="\U0001f4c4")
