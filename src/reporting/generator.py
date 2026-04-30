"""PDF inspection report generation for PaveScan AI.

Generates professional, engineering-grade PDF reports from detection results.
Includes ASTM D6433-informed PCI scoring, safety priority assessment,
recommended maintenance plan, and defect deterioration analysis.
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

# ASTM D6433-informed deduct values by (defect_type_category, severity).
# These reflect the actual impact each defect type has on pavement serviceability.
# Values calibrated to ASTM D6433 deduct value curves for asphalt pavements.
PCI_DEDUCT_VALUES = {
    # Potholes — most damaging to vehicles and pedestrians
    ("Pothole", "high"): 25,
    ("Pothole", "medium"): 18,
    ("Pothole", "low"): 12,
    # Alligator/fatigue cracking — structural failure indicator
    ("Alligator_Crack", "high"): 20,
    ("Alligator_Crack", "medium"): 14,
    ("Alligator_Crack", "low"): 8,
    # Longitudinal cracking — water infiltration risk
    ("Longitudinal_Crack", "high"): 12,
    ("Longitudinal_Crack", "medium"): 7,
    ("Longitudinal_Crack", "low"): 3,
    # Transverse cracking — ride quality, water entry
    ("Transverse_Crack", "high"): 10,
    ("Transverse_Crack", "medium"): 5,
    ("Transverse_Crack", "low"): 2,
    # Generic crack — moderate
    ("crack", "high"): 8,
    ("crack", "medium"): 4,
    ("crack", "low"): 2,
}

# Default deducts for unknown defect types
DEFAULT_DEDUCTS = {"high": 10, "medium": 5, "low": 2}

# Mapping from detection model class names to PCI deduct categories
_CLASS_TO_CATEGORY = {
    "D40": "Pothole",
    "Pothole": "Pothole",
    "D20": "Alligator_Crack",
    "Alligator_Crack": "Alligator_Crack",
    "D00": "Longitudinal_Crack",
    "Longitudinal_Crack": "Longitudinal_Crack",
    "D10": "Transverse_Crack",
    "Transverse_Crack": "Transverse_Crack",
    "crack": "crack",
}


def compute_pci_score(detections: list[dict]) -> tuple[int, str, str]:
    """Compute Pavement Condition Index with ASTM D6433-informed deduct values.

    Uses type-weighted deduct values (potholes penalize more than hairline cracks)
    with Corrected Deduct Value (CDV) diminishing returns — the 10th defect hurts
    less than the 1st, matching real pavement engineering practice.

    Returns:
        (score, rating_label, color)
    """
    if not detections:
        for threshold, label, color in PCI_RATINGS:
            if 100 >= threshold:
                return 100, label, color

    # Compute individual deduct values
    deducts = []
    for d in detections:
        class_name = d.get("class_name", "")
        severity = d.get("severity", "low")
        category = _CLASS_TO_CATEGORY.get(class_name, "")

        deduct = PCI_DEDUCT_VALUES.get(
            (category, severity),
            DEFAULT_DEDUCTS.get(severity, 2),
        )
        deducts.append(deduct)

    # Sort descending — largest deducts first
    deducts.sort(reverse=True)

    # CDV correction: diminishing returns for multiple deducts.
    # First defect takes full hit; each subsequent one is discounted.
    # This models the real ASTM D6433 corrected deduct value approach.
    total_deduct = 0.0
    for i, dv in enumerate(deducts):
        # Discount factor: 1.0 for first, decaying for subsequent
        # Factor = max(1.0 - 0.08*i, 0.2) ensures minimum 20% contribution
        factor = max(1.0 - 0.08 * i, 0.20)
        total_deduct += dv * factor

    score = max(0, min(100, round(100 - total_deduct)))

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


def _generate_priority_chart_base64(by_priority: dict) -> str:
    """Create a safety priority distribution pie chart and return as base64 PNG."""
    priority_colors = {
        "critical": "#FF1744",
        "urgent": "#FF9100",
        "monitor": "#FFC400",
        "routine": "#00C853",
    }
    labels_map = {
        "critical": "Critical",
        "urgent": "Urgent",
        "monitor": "Monitor",
        "routine": "Routine",
    }

    filtered = [
        (labels_map[k], by_priority.get(k, 0), priority_colors[k])
        for k in ["critical", "urgent", "monitor", "routine"]
        if by_priority.get(k, 0) > 0
    ]

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
    ax.set_title("Safety Priority Distribution", fontsize=12, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def _build_safety_assessment_html(summary: dict, pci_score: int) -> str:
    """Build the safety assessment alert box at the top of the report."""
    by_priority = summary.get("by_priority", {})
    critical = by_priority.get("critical", 0)
    urgent = by_priority.get("urgent", 0)
    cluster_risk = summary.get("cluster_risk", False)

    if critical > 0 or cluster_risk:
        bg_color = "#FFEBEE"
        border_color = "#FF1744"
        icon = "&#9888;"
        title = "IMMEDIATE ACTION REQUIRED"
        details = []
        if critical > 0:
            details.append(
                f"<b>{critical} critical defect(s)</b> detected that pose direct "
                f"danger to road users. Emergency repair or barricading required "
                f"within 24-48 hours."
            )
        if cluster_risk:
            details.append(
                "<b>Systemic pavement failure detected.</b> High defect density "
                "indicates structural degradation across this section. "
                "Recommend full section assessment and possible rehabilitation "
                "rather than spot repairs."
            )
        body = "<br>".join(details)
    elif urgent > 0:
        bg_color = "#FFF3E0"
        border_color = "#FF9100"
        icon = "&#9888;"
        title = "PRIORITY MAINTENANCE NEEDED"
        body = (
            f"<b>{urgent} urgent defect(s)</b> require priority maintenance "
            f"within 1-2 weeks to prevent further deterioration and safety risks."
        )
    elif pci_score < 55:
        bg_color = "#FFF8E1"
        border_color = "#FFC400"
        icon = "&#9888;"
        title = "PAVEMENT CONDITION BELOW ACCEPTABLE THRESHOLD"
        body = (
            f"PCI score of {pci_score} indicates fair to poor condition. "
            f"Scheduled maintenance recommended within the next maintenance cycle."
        )
    else:
        bg_color = "#E8F5E9"
        border_color = "#00C853"
        icon = "&#10004;"
        title = "PAVEMENT IN ACCEPTABLE CONDITION"
        body = (
            f"PCI score of {pci_score}. No critical or urgent defects detected. "
            f"Continue routine monitoring."
        )

    return f"""
    <div style="background:{bg_color}; border:2px solid {border_color};
                padding:15px 20px; margin:15px 0; page-break-inside:avoid;">
        <div style="font-size:14px; font-weight:bold; color:{border_color};
                    margin-bottom:8px;">
            {icon} {title}
        </div>
        <div style="font-size:11px; color:#333;">{body}</div>
    </div>
    """


def _build_maintenance_plan_html(all_detections: list[dict]) -> str:
    """Build the recommended maintenance plan grouped by urgency timeline."""
    from src.detection.model import SAFETY_PRIORITIES

    # Group detections by priority
    groups: dict[str, list[dict]] = {
        "critical": [],
        "urgent": [],
        "monitor": [],
        "routine": [],
    }
    for d in all_detections:
        priority = d.get("safety_priority", "routine")
        groups.setdefault(priority, [])
        groups[priority].append(d)

    sections = []

    for key in ["critical", "urgent", "monitor", "routine"]:
        dets = groups.get(key, [])
        if not dets:
            continue

        info = SAFETY_PRIORITIES[key]
        color = info["color_hex"]
        label = info["label"]
        timeline = info["timeline"]
        action = info["action"]

        # Summarize defect types in this group
        type_counts: dict[str, int] = {}
        for d in dets:
            cn = d.get("class_name", "Unknown")
            type_counts[cn] = type_counts.get(cn, 0) + 1
        type_summary = ", ".join(f"{v}x {k}" for k, v in sorted(type_counts.items(), key=lambda x: -x[1]))

        sections.append(f"""
        <div style="border-left:4px solid {color}; padding:8px 12px; margin:8px 0;
                    background:#fafafa; page-break-inside:avoid;">
            <div style="font-weight:bold; color:{color}; font-size:12px;">
                {label} &mdash; {timeline}
            </div>
            <div style="font-size:11px; margin-top:4px;">
                <b>Action:</b> {action}<br>
                <b>Defects ({len(dets)}):</b> {type_summary}
            </div>
        </div>
        """)

    if not sections:
        return ""

    # Cost escalation warning
    cost_warning = ""
    critical_count = len(groups.get("critical", []))
    urgent_count = len(groups.get("urgent", []))
    if critical_count + urgent_count > 0:
        cost_warning = """
        <div style="background:#FFF8E1; border:1px solid #FFC400;
                    padding:10px; margin:10px 0; font-size:10px;
                    page-break-inside:avoid;">
            <b>&#9888; Cost Escalation Warning:</b> Deferred maintenance on critical and urgent
            defects typically increases repair costs by 4-8x. A pothole that costs $50-150 to patch
            today may require $2,000-5,000 in structural rehabilitation if left untreated through
            one freeze-thaw season. Early intervention is the most cost-effective strategy.
        </div>
        """

    return (
        '<div class="section"><h2>Recommended Maintenance Plan</h2>'
        + "".join(sections)
        + cost_warning
        + "</div>"
    )


def _build_methodology_html() -> str:
    """Build the methodology section for professional credibility."""
    return """
    <div class="section" style="page-break-inside:avoid;">
        <h2>Methodology</h2>
        <div style="font-size:10px; color:#444; line-height:1.6;">
            <p>
                This report was generated using PaveScan AI, an automated pavement inspection system
                that combines deep learning-based defect detection with engineering assessment standards.
            </p>
            <p>
                <b>Detection:</b> YOLO-based object detection and instance segmentation models trained
                on pavement distress imagery. Ensemble inference with IoU-based deduplication is used
                for improved coverage. Optional SAHI tiled inference detects small defects in high-resolution
                drone imagery.
            </p>
            <p>
                <b>Severity Assessment:</b> Multi-factor scoring informed by ASTM D6433 Standard Practice
                for Roads and Parking Lots Pavement Condition Index Surveys. Factors include defect area ratio,
                estimated crack width (from segmentation masks), defect type risk weight, detection confidence,
                and defect density. Structural failure types (potholes, alligator cracking) receive a minimum
                severity floor regardless of size.
            </p>
            <p>
                <b>Safety Priority:</b> Four-tier classification (Critical, Urgent, Monitor, Routine) based on
                defect type, severity score, and systemic failure indicators. Prioritization follows a
                safety-first approach: all potholes are classified as critical regardless of size due to their
                direct hazard to road users.
            </p>
            <p>
                <b>PCI Scoring:</b> Type-weighted deduct values with Corrected Deduct Value (CDV)
                diminishing returns, following the ASTM D6433 framework. Deducts are calibrated by defect type
                and severity level.
            </p>
            <p style="color:#888; font-style:italic;">
                Note: This automated assessment supplements but does not replace professional engineering judgment.
                Critical findings should be verified by a qualified pavement engineer before major repair decisions.
            </p>
        </div>
    </div>
    """


def generate_report_html(
    detection_results: list[dict],
    geo_results: list[dict] | None,
    project_info: dict,
) -> str:
    """Generate the full HTML report with safety-priority assessment.

    Args:
        detection_results: List of {"filename", "result"} from detection page.
        geo_results: List of {"filename", "result", "lat", "lon"} or None.
        project_info: Dict with keys: project_name, inspector_name, notes.
    """
    from src.detection.model import SAFETY_PRIORITIES, summarize_detections

    # Gather all detections
    all_detections = [d for r in detection_results for d in r["result"]["detections"]]
    summary = summarize_detections(all_detections)
    pci_score, pci_rating, pci_color = compute_pci_score(all_detections)
    by_priority = summary.get("by_priority", {})

    # Generate charts
    chart_b64 = generate_chart_base64(summary)
    priority_chart_b64 = _generate_priority_chart_base64(by_priority)

    # Build safety assessment
    safety_assessment_html = _build_safety_assessment_html(summary, pci_score)

    # Build maintenance plan
    maintenance_plan_html = _build_maintenance_plan_html(all_detections)

    # Build methodology
    methodology_html = _build_methodology_html()

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
            priority_key = det.get("safety_priority", "routine")
            priority_info = SAFETY_PRIORITIES.get(priority_key, SAFETY_PRIORITIES["routine"])
            deterioration = det.get("deterioration_risk", {})
            width_px = det.get("width_pixels", 0.0)

            defect_rows.append({
                "filename": filename,
                "class_name": det["class_name"],
                "confidence": f"{det['confidence']:.1%}",
                "severity": det["severity"].upper(),
                "severity_color": SEVERITY_COLORS.get(det["severity"], "#888"),
                "severity_score": f"{det.get('severity_score', 0):.2f}",
                "priority_label": priority_info["label"],
                "priority_color": priority_info["color_hex"],
                "area_pixels": f"{det.get('area_pixels', 0):,}",
                "width_px": f"{width_px:.0f}" if width_px > 0 else "-",
                "deterioration": deterioration.get("warning", ""),
                "action": det.get("recommended_action", ""),
                "lat": lat,
                "lon": lon,
            })

    # Sort defect rows by priority (critical first), then severity score descending
    priority_order = {"CRITICAL": 0, "URGENT": 1, "MONITOR": 2, "ROUTINE": 3}
    defect_rows.sort(key=lambda r: (priority_order.get(r["priority_label"], 99), -float(r["severity_score"])))

    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    project_name = project_info.get("project_name", "Pavement Inspection")
    inspector_name = project_info.get("inspector_name", "Inspector")
    notes = project_info.get("notes", "")

    # Priority summary counts
    critical_count = by_priority.get("critical", 0)
    urgent_count = by_priority.get("urgent", 0)
    monitor_count = by_priority.get("monitor", 0)
    routine_count = by_priority.get("routine", 0)

    # Build priority summary row
    priority_summary_html = f"""
    <table class="stats-table">
        <tr>
            <td style="border-bottom:3px solid #FF1744;">
                <div class="stat-value" style="color:#FF1744;">{critical_count}</div>
                <div class="stat-label">Critical</div>
            </td>
            <td style="border-bottom:3px solid #FF9100;">
                <div class="stat-value" style="color:#FF9100;">{urgent_count}</div>
                <div class="stat-label">Urgent</div>
            </td>
            <td style="border-bottom:3px solid #FFC400;">
                <div class="stat-value" style="color:#B8860B;">{monitor_count}</div>
                <div class="stat-label">Monitor</div>
            </td>
            <td style="border-bottom:3px solid #00C853;">
                <div class="stat-value" style="color:#00C853;">{routine_count}</div>
                <div class="stat-label">Routine</div>
            </td>
        </tr>
    </table>
    """

    # Build defect inventory table
    if defect_rows:
        inventory_rows = "".join(
            f"<tr>"
            f"<td>{i+1}</td>"
            f"<td>{r['filename']}</td>"
            f"<td>{r['class_name']}</td>"
            f"<td><span class='severity-badge' style='background-color:{r['priority_color']};'>"
            f"{r['priority_label']}</span></td>"
            f"<td>{r['confidence']}</td>"
            f"<td><span class='severity-badge' style='background-color:{r['severity_color']};'>"
            f"{r['severity']}</span> ({r['severity_score']})</td>"
            f"<td>{r['width_px']}</td>"
            f"<td>{r['area_pixels']}</td>"
            f"<td>{r['lat']}</td>"
            f"<td>{r['lon']}</td>"
            f"<td style='font-size:8px;'>{r['deterioration']}</td>"
            f"</tr>"
            for i, r in enumerate(defect_rows)
        )
        inventory_html = (
            "<table>"
            "<tr><th>#</th><th>File</th><th>Defect Type</th><th>Priority</th>"
            "<th>Confidence</th><th>Severity</th><th>Width (px)</th>"
            "<th>Area (px)</th><th>Latitude</th><th>Longitude</th>"
            "<th>Deterioration Risk</th></tr>"
            + inventory_rows
            + "</table>"
        )
    else:
        inventory_html = "<p>No defects detected in the analyzed images.</p>"

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
        width: 300px;
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

{safety_assessment_html}

<div class="section">
    <h2>Pavement Condition Index (PCI)</h2>
    <div class="pci-box">
        <div class="pci-score">{pci_score}</div>
        <div class="pci-label">{pci_rating}</div>
        <div class="pci-subtitle">Based on ASTM D6433 type-weighted deduct values with CDV correction</div>
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
    <h2>Safety Priority Summary</h2>
    {priority_summary_html}
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

<div class="section">
    <h2>Distribution Analysis</h2>
    <table style="border:none;"><tr>
        <td style="width:50%; border:none; vertical-align:top;">
            {"<div class='chart-container'><img src='data:image/png;base64," + chart_b64 + "'/></div>" if chart_b64 else "<p>No severity data.</p>"}
        </td>
        <td style="width:50%; border:none; vertical-align:top;">
            {"<div class='chart-container'><img src='data:image/png;base64," + priority_chart_b64 + "'/></div>" if priority_chart_b64 else "<p>No priority data.</p>"}
        </td>
    </tr></table>
</div>

{maintenance_plan_html}

<div class="section">
    <h2>Defect Inventory</h2>
    {inventory_html}
</div>

{f"<div class='section'><h2>Inspector Notes</h2><div class='notes'>{notes}</div></div>" if notes else ""}

{methodology_html}

<div class="footer">
    Generated by PaveScan AI &mdash; Automated Pavement Inspection System<br>
    Assessment methodology informed by ASTM D6433 Standard Practice for Pavement Condition Index Surveys<br>
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
