"""Report page — generate professional PDF inspection report."""

import streamlit as st

st.set_page_config(page_title="Report | PaveScan AI", layout="wide")
st.title("Inspection Report")

st.info("""
**Coming in Phase 4 (Week 7)**

This page will generate a professional PDF inspection report including:
- Executive Summary with overall PCI score
- Inspection methodology and equipment details
- Color-coded condition map
- Section-by-section defect inventory with severity classification
- Treatment recommendations (priority-ranked)
- Model accuracy metrics

Reports will be generated using **WeasyPrint** with a professional
HTML/CSS template rendered via **Jinja2**.
""")
