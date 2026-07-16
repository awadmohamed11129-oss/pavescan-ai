"""Mission Planner page — draw flight route, configure parameters, export waypoints."""

import sys
from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.mission.planner import (
    format_area,
    format_distance,
    format_time,
    generate_waypoints_file,
    plan_mission,
)

st.set_page_config(page_title="Mission Planner | PaveScan AI", layout="wide")
st.title("Mission Planner")
st.caption("Plan an autonomous drone survey route for pavement inspection")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "mission_waypoints" not in st.session_state:
    st.session_state["mission_waypoints"] = []

if "last_click_processed" not in st.session_state:
    st.session_state["last_click_processed"] = None

# ---------------------------------------------------------------------------
# Sidebar — Mission Parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Mission Parameters")

    altitude = st.slider(
        "Altitude (m)",
        min_value=15, max_value=50, value=25, step=1,
        help="Flight altitude above ground. Lower = more detail, higher = more coverage.",
    )

    speed = st.slider(
        "Speed (m/s)",
        min_value=2.0, max_value=8.0, value=4.0, step=0.5,
        help="Drone ground speed. Slower = sharper photos.",
    )

    overlap = st.slider(
        "Photo Overlap (%)",
        min_value=60, max_value=90, value=75, step=5,
        help="Overlap between consecutive photos. Higher = better stitching, more photos.",
    )

    num_passes = st.radio(
        "Number of Passes",
        options=[1, 2],
        index=0,
        help="2 passes = fly the route twice for better coverage.",
        horizontal=True,
    )

    st.markdown("---")

    st.markdown("### Route Controls")

    if st.button("Undo Last Point", width="stretch"):
        if st.session_state["mission_waypoints"]:
            st.session_state["mission_waypoints"].pop()
            st.rerun()

    if st.button("Clear All Points", type="secondary", width="stretch"):
        st.session_state["mission_waypoints"] = []
        st.session_state["last_click_processed"] = None
        st.rerun()

    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown(
        "1. **Click on the map** to add waypoints\n"
        "2. Adjust altitude, speed, and overlap\n"
        "3. Review the mission summary\n"
        "4. Export the `.waypoints` file"
    )

# ---------------------------------------------------------------------------
# Map — Click to add waypoints
# ---------------------------------------------------------------------------

waypoints = st.session_state["mission_waypoints"]

# Center map on last waypoint or default (University of Toronto area)
if waypoints:
    center = [waypoints[-1][0], waypoints[-1][1]]
    zoom = 16
else:
    center = [43.6605, -79.3955]
    zoom = 15

m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

# Draw existing waypoints and route line
if waypoints:
    # Route line
    folium.PolyLine(
        waypoints,
        color="#1976D2",
        weight=3,
        opacity=0.8,
    ).add_to(m)

    # Numbered markers for each waypoint
    for i, (lat, lon) in enumerate(waypoints):
        is_first = i == 0
        is_last = i == len(waypoints) - 1

        if is_first:
            icon_color = "green"
            prefix = "S"
        elif is_last:
            icon_color = "red"
            prefix = "E"
        else:
            icon_color = "blue"
            prefix = str(i + 1)

        folium.Marker(
            location=[lat, lon],
            popup=f"WP {i + 1}: ({lat:.6f}, {lon:.6f})",
            tooltip=f"Waypoint {i + 1}" + (" (Start)" if is_first else " (End)" if is_last else ""),
            icon=folium.DivIcon(
                html=f'<div style="background:{icon_color};color:white;border-radius:50%;'
                     f'width:24px;height:24px;text-align:center;line-height:24px;'
                     f'font-size:12px;font-weight:bold;border:2px solid white;'
                     f'box-shadow:0 1px 3px rgba(0,0,0,0.3);">{prefix}</div>',
                icon_size=(24, 24),
                icon_anchor=(12, 12),
            ),
        ).add_to(m)

st.markdown("**Click on the map to add waypoints.** Green = start, Red = end.")

# Render map and capture clicks
map_data = st_folium(m, height=500, use_container_width=True)

# Process click — add new waypoint
if map_data and map_data.get("last_clicked"):
    click = map_data["last_clicked"]
    click_key = (round(click["lat"], 8), round(click["lng"], 8))

    if click_key != st.session_state["last_click_processed"]:
        st.session_state["last_click_processed"] = click_key
        st.session_state["mission_waypoints"].append((click["lat"], click["lng"]))
        st.rerun()

# ---------------------------------------------------------------------------
# Mission Summary
# ---------------------------------------------------------------------------

st.markdown("---")

if len(waypoints) >= 2:
    summary = plan_mission(
        waypoints=waypoints,
        altitude_m=float(altitude),
        speed_ms=float(speed),
        overlap_pct=float(overlap),
        num_passes=int(num_passes),
    )

    st.subheader("Mission Summary")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Distance", format_distance(summary.total_distance_m))
    col2.metric("Flight Time", format_time(summary.flight_time_s))
    col3.metric("Photos", str(summary.num_photos))
    col4.metric("Coverage", format_area(summary.coverage_area_m2))

    # Secondary info
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Waypoints", str(len(waypoints)))
    col6.metric("Swath Width", f"{summary.swath_width_m:.1f} m")
    col7.metric("Photo Interval", f"{summary.photo_interval_m:.1f} m")
    col8.metric("Passes", str(num_passes))

    # Battery status
    if summary.battery_ok:
        st.success(
            f"Battery OK — {summary.battery_margin_pct:.0f}% margin remaining "
            f"(~{summary.flight_time_s / 60:.1f} min of ~18 min max)"
        )
    else:
        st.error(
            f"Route too long for battery! Estimated {summary.flight_time_s / 60:.1f} min "
            f"exceeds ~18 min max. Shorten the route or increase speed."
        )

    # ---------------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------------

    st.markdown("---")
    st.subheader("Export Mission")

    wp_content = generate_waypoints_file(
        waypoints=waypoints,
        altitude_m=float(altitude),
        speed_ms=float(speed),
    )

    col_dl, col_preview = st.columns([1, 2])

    with col_dl:
        st.download_button(
            label="Download .waypoints File",
            data=wp_content,
            file_name="pavescan_mission.waypoints",
            mime="text/plain",
            width="stretch",
            type="primary",
        )
        st.caption(
            "Load this file in **QGroundControl** or **Mission Planner** "
            "to fly the route autonomously with ArduPilot."
        )

    with col_preview:
        with st.expander("Preview .waypoints file"):
            st.code(wp_content, language="text")

elif len(waypoints) == 1:
    st.info("Add at least one more waypoint to see the mission summary.")
else:
    st.info("Click on the map to start placing waypoints for your survey route.")

# ---------------------------------------------------------------------------
# Waypoint table
# ---------------------------------------------------------------------------

if waypoints:
    st.markdown("---")
    with st.expander(f"Waypoint Coordinates ({len(waypoints)} points)"):
        for i, (lat, lon) in enumerate(waypoints):
            label = "Start" if i == 0 else "End" if i == len(waypoints) - 1 else f"WP {i + 1}"
            st.text(f"{label}: {lat:.6f}, {lon:.6f}")

# ---------------------------------------------------------------------------
# Navigation footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "PaveScan AI Mission Planner | "
    "Compatible with ArduPilot + QGroundControl | "
    "F450 Quadcopter"
)
