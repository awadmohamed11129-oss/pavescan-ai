"""PDF inspection report generation for PaveScan AI.

Generates professional PDF reports from detection results using
Jinja2 HTML templates rendered to PDF via WeasyPrint.
"""

import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from xhtml2pdf import pisa


SEVERITY_COLORS = {
    "low": "#00C853",
    "medium": "#FFD600",
    "high": "#FF1744",
}

PCI_RATINGS = [
    (85, "Good", "#00C853"),
    (70, "Satisfactory", "#66BB6A"),
    (55, "Fair", "#FFD600"),
    (40, "Poor", "#FF9100"),
    (0, "Very Poor", "#FF1744"),
]


def compute_pci_score(detections: list[dict]) -> tuple[int, str, str]:
    """Compute a simplified Pavement Condition Index score.

    Based on severity counts:
    - Each high severity detection: -15 points
    - Each medium severity detection: -5 points
    - Each low severity detection: -2 points
    Starting from 100 (perfect condition).

    Returns:
        (score, rating_label, color)
    """
    score = 100

    for d in detections:
        severity = d.get("severity", "low")
        if severity == "high":
            score -= 15
        elif severity == "medium":
            score -= 5
        else:
            score -= 2

    score = max(0, min(100, score))

    for threshold, label, color in PCI_RATINGS:
        if score >= threshold:
            return score, label, color

    return score, "Very Poor", "#FF1744"


def generate_chart_base64(summary: dict) -> str:
    """Create a severity distribution pie chart and return as base64 PNG."""
    values = [
        summary["by_severity"]["low"],
        summary["by_severity"]["medium"],
        summary["by_severity"]["high"],
    ]
    labels = ["Low", "Medium", "High"]
    colors = [SEVERITY_COLORS["low"], SEVERITY_COLORS["medium"], SEVERITY_COLORS["high"]]

    # Filter out zero values
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]

    if not filtered:
        return ""

    labels, values, colors = zip(*filtered)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title("Severity Distribution", fontsize=12, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def generate_report_html(
    detection_results: list[dict],
    geo_results: list[dict] | None,
    project_info: dict,
) -> str:
    """Generate the full HTML report.

    Args:
        detection_results: List of {"filename", "result"} from detection page.
        geo_results: List of {"filename", "result", "lat", "lon"} or None.
        project_info: Dict with keys: project_name, inspector_name, notes.
    """
    from src.detection.model import summarize_detections

    # Gather all detections
    all_detections = [d for r in detection_results for d in r["result"]["detections"]]
    summary = summarize_detections(all_detections)
    pci_score, pci_rating, pci_color = compute_pci_score(all_detections)

    # Generate chart
    chart_b64 = generate_chart_base64(summary)

    # Build defect rows
    defect_rows = []
    for item in detection_results:
        filename = item["filename"]
        lat = lon = "N/A"

        # Find matching geo result
        if geo_results:
            for g in geo_results:
                if g["filename"] == filename:
                    lat = f"{g['lat']:.6f}"
                    lon = f"{g['lon']:.6f}"
                    break

        for det in item["result"]["detections"]:
            defect_rows.append({
                "filename": filename,
                "class_name": det["class_name"],
                "confidence": f"{det['confidence']:.1%}",
                "severity": det["severity"].upper(),
                "severity_color": SEVERITY_COLORS.get(det["severity"], "#888"),
                "area_pixels": f"{det.get('area_pixels', 0):,}",
                "lat": lat,
                "lon": lon,
            })

    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    project_name = project_info.get("project_name", "Pavement Inspection")
    inspector_name = project_info.get("inspector_name", "Inspector")
    notes = project_info.get("notes", "")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: letter;
        margin: 1.5cm;
    }}
    body {{
        font-family: Helvetica, Arial, sans-serif;
        color: #333333;
        line-height: 1.5;
        font-size: 11px;
    }}
    .header {{
        background-color: #1a237e;
        color: white;
        padding: 25px 30px;
        margin-bottom: 20px;
    }}
    .header h1 {{
        margin: 0 0 5px 0;
        font-size: 24px;
        color: white;
    }}
    .header .subtitle {{
        font-size: 13px;
        color: #ccccdd;
    }}
    .header .meta {{
        margin-top: 10px;
        font-size: 11px;
        color: #bbbbcc;
    }}
    .section {{
        margin: 20px 0;
    }}
    .section h2 {{
        color: #1a237e;
        font-size: 16px;
        border-bottom: 2px solid #1a237e;
        padding-bottom: 5px;
        margin-bottom: 12px;
    }}
    .pci-box {{
        text-align: center;
        padding: 20px;
        border: 2px solid {pci_color};
        margin: 15px 0;
    }}
    .pci-score {{
        font-size: 48px;
        font-weight: bold;
        color: {pci_color};
    }}
    .pci-label {{
        font-size: 18px;
        color: {pci_color};
        font-weight: bold;
    }}
    .pci-subtitle {{
        font-size: 11px;
        color: #666666;
        margin-top: 5px;
    }}
    .stats-table {{
        width: 100%;
        margin: 15px 0;
    }}
    .stats-table td {{
        width: 25%;
        text-align: center;
        padding: 12px;
        background-color: #f5f5f5;
        border: none;
    }}
    .stat-value {{
        font-size: 24px;
        font-weight: bold;
        color: #1a237e;
    }}
    .stat-label {{
        font-size: 10px;
        color: #666666;
        text-transform: uppercase;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 10px;
        margin: 10px 0;
    }}
    th {{
        background-color: #1a237e;
        color: white;
        padding: 8px 6px;
        text-align: left;
        font-size: 10px;
    }}
    td {{
        padding: 6px;
        border-bottom: 1px solid #e0e0e0;
    }}
    .severity-badge {{
        padding: 2px 8px;
        color: white;
        font-weight: bold;
        font-size: 9px;
    }}
    .chart-container {{
        text-align: center;
        margin: 15px 0;
    }}
    .chart-container img {{
        width: 350px;
    }}
    .notes {{
        background-color: #f5f5f5;
        padding: 12px;
        border-left: 4px solid #1a237e;
        font-style: italic;
    }}
    .footer {{
        margin-top: 30px;
        text-align: center;
        color: #999999;
        font-size: 9px;
        border-top: 1px solid #e0e0e0;
        padding-top: 10px;
    }}
    .pci-scale-table {{
        width: 100%;
        margin: 10px 0;
        font-size: 9px;
    }}
    .pci-scale-table td {{
        width: 20%;
        text-align: center;
        padding: 4px 2px;
        color: white;
        font-weight: bold;
        border: none;
    }}
</style>
</head>
<body>

<div class="header">
    <h1>{project_name}</h1>
    <div class="subtitle">Automated Pavement Inspection Report</div>
    <div class="meta">
        Inspector: {inspector_name} &nbsp;|&nbsp;
        Date: {now} &nbsp;|&nbsp;
        Images Analyzed: {len(detection_results)}
    </div>
</div>

<div class="section">
    <h2>Pavement Condition Index (PCI)</h2>
    <div class="pci-box">
        <div class="pci-score">{pci_score}</div>
        <div class="pci-label">{pci_rating}</div>
        <div class="pci-subtitle">Based on ASTM D6433 simplified scoring</div>
    </div>
    <table class="pci-scale-table">
        <tr>
            <td style="background-color: #FF1744;">Very Poor (0-39)</td>
            <td style="background-color: #FF9100;">Poor (40-54)</td>
            <td style="background-color: #FFD600; color: #333333;">Fair (55-69)</td>
            <td style="background-color: #66BB6A;">Satisfactory (70-84)</td>
            <td style="background-color: #00C853;">Good (85-100)</td>
        </tr>
    </table>
</div>

<div class="section">
    <h2>Executive Summary</h2>
    <table class="stats-table">
        <tr>
            <td>
                <div class="stat-value">{summary['total_detections']}</div>
                <div class="stat-label">Total Defects</div>
            </td>
            <td>
                <div class="stat-value" style="color: #FF1744;">{summary['by_severity']['high']}</div>
                <div class="stat-label">High Severity</div>
            </td>
            <td>
                <div class="stat-value" style="color: #FF9100;">{summary['by_severity']['medium']}</div>
                <div class="stat-label">Medium Severity</div>
            </td>
            <td>
                <div class="stat-value" style="color: #00C853;">{summary['by_severity']['low']}</div>
                <div class="stat-label">Low Severity</div>
            </td>
        </tr>
    </table>
    {"<p><b>Average Confidence:</b> " + f"{summary['avg_confidence']:.1%}</p>" if summary['total_detections'] > 0 else ""}
    <p><b>Images Analyzed:</b> {len(detection_results)}</p>
</div>

{"<div class='section'><h2>Severity Distribution</h2><div class='chart-container'><img src='data:image/png;base64," + chart_b64 + "'/></div></div>" if chart_b64 else ""}

<div class="section">
    <h2>Defect Inventory</h2>
    {"<table><tr><th>#</th><th>File</th><th>Defect Type</th><th>Confidence</th><th>Severity</th><th>Area (px)</th><th>Latitude</th><th>Longitude</th></tr>" + "".join(
        f"<tr><td>{i+1}</td><td>{r['filename']}</td><td>{r['class_name']}</td><td>{r['confidence']}</td><td><span class='severity-badge' style='background-color:{r['severity_color']}'>{r['severity']}</span></td><td>{r['area_pixels']}</td><td>{r['lat']}</td><td>{r['lon']}</td></tr>"
        for i, r in enumerate(defect_rows)
    ) + "</table>" if defect_rows else "<p>No defects detected in the analyzed images.</p>"}
</div>

{f"<div class='section'><h2>Inspector Notes</h2><div class='notes'>{notes}</div></div>" if notes else ""}

<div class="footer">
    Generated by PaveScan AI &mdash; Automated Pavement Inspection System<br>
    {now}
</div>

</body>
</html>"""

    return html


def html_to_pdf(html_string: str) -> bytes:
    """Convert an HTML string to PDF bytes using xhtml2pdf."""
    buf = io.BytesIO()
    pisa.CreatePDF(html_string, dest=buf)
    buf.seek(0)
    return buf.read()
