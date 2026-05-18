"""Report page — generate professional PDF inspection reports."""

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.detection.clustering import apply_priority_overrides
from src.detection.model import SAFETY_PRIORITIES, summarize_detections
from src.mapping.geo import assign_coordinates_to_results
from src.reporting.generator import (
    compute_pci_score,
    generate_report_html,
    html_to_pdf,
)

st.set_page_config(page_title="Report | PaveScan AI", layout="wide")
st.title("Inspection Report")

# --- Sidebar: report settings ---
with st.sidebar:
    st.header("Report Settings")

    project_name = st.text_input(
        "Project Name",
        value="Pavement Inspection",
        help="Name for the inspection project (appears on the report header).",
    )

    inspector_name = st.text_input(
        "Inspector Name",
        value="",
        placeholder="Your name",
        help="Name of the person conducting the inspection.",
    )

    notes = st.text_area(
        "Inspector Notes",
        value="",
        placeholder="Any additional notes about the inspection...",
        help="Optional notes to include in the report.",
        height=120,
    )

# --- Guard: need detection results ---
if "detection_results" not in st.session_state or not st.session_state["detection_results"]:
    st.warning("No detection results found. Run detection first.")
    st.page_link("pages/2_Detection.py", label="Go to Detection", icon="🔍")
    st.stop()

detection_results = st.session_state["detection_results"]
uploaded_files = st.session_state.get("uploaded_files", [])

# Apply inspector overrides BEFORE summary/PCI so the report reflects the
# human-classified safety priority everywhere (priority counts, maintenance
# plan, executive summary). PCI itself is computed from raw `severity` and
# stays an automated metric.
inspector_has_overrides = apply_priority_overrides(
    detection_results,
    st.session_state.get("priority_overrides", {}),
    SAFETY_PRIORITIES,
)

# --- Compute summary stats ---
all_detections = [d for r in detection_results for d in r["result"]["detections"]]
summary = summarize_detections(all_detections)
pci_score, pci_rating, pci_color = compute_pci_score(all_detections)

# --- PCI Score preview ---
st.markdown("---")

col_pci, col_stats = st.columns([1, 2])

with col_pci:
    st.markdown(
        f"""
        <div style="text-align:center; padding:20px; border-radius:10px; border:2px solid {pci_color};">
            <div style="font-size:48px; font-weight:bold; color:{pci_color};">{pci_score}</div>
            <div style="font-size:18px; font-weight:bold; color:{pci_color};">{pci_rating}</div>
            <div style="font-size:12px; color:#888; margin-top:5px;">Pavement Condition Index</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_stats:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Defects", summary["total_detections"])
    c2.metric("High Severity", summary["by_severity"]["high"])
    c3.metric("Medium Severity", summary["by_severity"]["medium"])
    c4.metric("Low Severity", summary["by_severity"]["low"])

    # Safety priority metrics
    by_priority = summary.get("by_priority", {})
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(
        f"<div style='text-align:center;padding:6px;background:#FF1744;color:white;border-radius:6px;'>"
        f"<b style='font-size:0.75em;'>CRITICAL</b><br><span style='font-size:1.5em;'>{by_priority.get('critical', 0)}</span></div>",
        unsafe_allow_html=True,
    )
    p2.markdown(
        f"<div style='text-align:center;padding:6px;background:#FF9100;color:white;border-radius:6px;'>"
        f"<b style='font-size:0.75em;'>URGENT</b><br><span style='font-size:1.5em;'>{by_priority.get('urgent', 0)}</span></div>",
        unsafe_allow_html=True,
    )
    p3.markdown(
        f"<div style='text-align:center;padding:6px;background:#FFC400;color:#333;border-radius:6px;'>"
        f"<b style='font-size:0.75em;'>MONITOR</b><br><span style='font-size:1.5em;'>{by_priority.get('monitor', 0)}</span></div>",
        unsafe_allow_html=True,
    )
    p4.markdown(
        f"<div style='text-align:center;padding:6px;background:#00C853;color:white;border-radius:6px;'>"
        f"<b style='font-size:0.75em;'>ROUTINE</b><br><span style='font-size:1.5em;'>{by_priority.get('routine', 0)}</span></div>",
        unsafe_allow_html=True,
    )

    if summary["total_detections"] > 0:
        st.markdown(f"**Average Confidence:** {summary['avg_confidence']:.1%}")

    st.markdown(f"**Images Analyzed:** {len(detection_results)}")

st.markdown("---")

# --- Generate report ---
if st.button("Generate PDF Report", type="primary"):
    with st.spinner("Generating report..."):
        # Get geo data
        geo_results, _ = assign_coordinates_to_results(detection_results, uploaded_files)

        project_info = {
            "project_name": project_name,
            "inspector_name": inspector_name or "Not specified",
            "notes": notes,
            "inspector_has_overrides": inspector_has_overrides,
        }

        # Generate HTML and convert to PDF
        html = generate_report_html(detection_results, geo_results, project_info)
        pdf_bytes = html_to_pdf(html)

    st.success("Report generated!")

    # Store in session state for preview
    st.session_state["report_html"] = html
    st.session_state["report_pdf"] = pdf_bytes

# --- Download and preview ---
if "report_pdf" in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pavescan_report_{timestamp}.pdf"

    st.download_button(
        label="Download PDF Report",
        data=st.session_state["report_pdf"],
        file_name=filename,
        mime="application/pdf",
        type="primary",
    )

    st.markdown("---")
    st.subheader("Report Preview")
    st.components.v1.html(
        st.session_state["report_html"],
        height=800,
        scrolling=True,
    )
