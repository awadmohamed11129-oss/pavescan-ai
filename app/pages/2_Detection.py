"""Detection page — run YOLO crack detection on uploaded images."""

import sys
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.detection.model import (
    MODELS_DIR,
    MODEL_VERSIONS,
    SAFETY_PRIORITIES,
    compute_iou,
    list_available_models,
    load_ensemble_models,
    load_model,
    run_ensemble_inference,
    run_inference,
    summarize_detections,
)
from src.detection.clustering import apply_priority_overrides

st.set_page_config(page_title="Detection | PaveScan AI", layout="wide")
st.title("AI Crack Detection")

# Sidebar controls
with st.sidebar:
    st.header("Detection Settings")

    # Model mode selector
    v2_seg_present = (MODELS_DIR / MODEL_VERSIONS["v2"]["seg"]).exists()
    mode_options = ["Ensemble (Both Models)", "Compare V1 vs V2", "Single Model"]
    mode = st.radio(
        "Detection Mode",
        options=mode_options,
        index=1,
        key="detection_mode",
        help="Ensemble merges seg + det for best coverage. Compare runs V1 and V2 ensembles side-by-side. Single uses one model.",
    )

    _prev_detection_mode = st.session_state.get("_prev_detection_mode")
    if (
        _prev_detection_mode is not None
        and _prev_detection_mode != mode
        and mode != "Compare V1 vs V2"
    ):
        for _stale_key in (
            "compare_mode",
            "detection_results_v1",
            "detection_results_v2",
            "compare_timing",
            "compare_active_version",
        ):
            st.session_state.pop(_stale_key, None)
    st.session_state["_prev_detection_mode"] = mode

    if mode == "Compare V1 vs V2":
        if not v2_seg_present:
            st.error(
                f"V2 model not found. Drop `{MODEL_VERSIONS['v2']['seg']}` into `models/` and reload."
            )
            st.stop()
        st.caption(
            "Runs V1 and V2 ensembles on the same image. "
            "Map/Report use V2 results by default — switch below results."
        )
        st.caption("⚠️ Compare loads 4 models (~1.5 GB RAM peak).")

    available_models = list_available_models()
    model_names = [m["name"] for m in available_models]

    selected_model_path = None
    if mode == "Single Model" and model_names:
        selected_name = st.selectbox(
            "Select Model",
            options=model_names,
            index=0,
            help="Choose which model to run.",
        )
        selected_model_path = next(
            m["path"] for m in available_models if m["name"] == selected_name
        )
        info = next(m for m in available_models if m["name"] == selected_name)
        st.caption(
            f"**Type:** {info['task']}  \n"
            f"**Classes ({info['num_classes']}):** {', '.join(info['classes'])}"
        )

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.9,
        value=0.15,
        step=0.05,
        help="Lower = more detections (possibly noisy). Higher = fewer but more certain.",
    )

    st.markdown("---")
    st.header("Advanced Settings")

    imgsz = st.select_slider(
        "Inference Resolution",
        options=[640, 960, 1280, 1600],
        value=1280,
        help="Higher resolution = more detail = better detection of small cracks. Uses more memory and is slower.",
    )

    use_tta = st.toggle(
        "Test-Time Augmentation (TTA)",
        value=False,
        help="Runs inference multiple times with flips/scales and averages results. ~3-4x slower but ~1-3% more accurate.",
    )

    use_sahi = st.toggle(
        "SAHI Tiled Inference",
        value=False,
        help="Slices large images into overlapping tiles for detection. Critical for finding small cracks in high-resolution photos. Much slower.",
    )

    sahi_slice_size = 640
    if use_sahi:
        sahi_slice_size = st.select_slider(
            "SAHI Tile Size",
            options=[320, 480, 640, 800, 1024],
            value=640,
            help="Size of each tile. Smaller = more tiles = catches smaller cracks but slower.",
        )

    use_wbf = st.toggle(
        "Weighted Box Fusion (WBF)",
        value=False,
        help="Ensemble-only. Averages overlapping boxes from both models weighted by confidence instead of dropping the loser. Typically +1-3 mAP at minimal extra cost.",
    )

    # Priority filter
    st.markdown("---")
    st.header("Display Filters")
    priority_filter = st.multiselect(
        "Show Priority Levels",
        options=["critical", "urgent", "monitor", "routine"],
        default=["critical", "urgent", "monitor", "routine"],
        format_func=lambda x: SAFETY_PRIORITIES[x]["label"],
        help="Filter which priority levels to display in results.",
    )

    # Show ensemble info
    if mode in ("Ensemble (Both Models)", "Compare V1 vs V2"):
        ensemble_models = [m for m in available_models if m["name"] != "yolov8n-seg"]
        if ensemble_models:
            st.markdown("---")
            st.markdown("**Ensemble models:**")
            for m in ensemble_models:
                st.caption(f"- {m['name']} ({m['task']}, {m['num_classes']} classes)")

# Check for uploaded images
uploaded_files = st.session_state.get("uploaded_files", [])

# Inspector severity override store: {f"{filename}::{cluster_id}": priority_key}
if "priority_overrides" not in st.session_state:
    st.session_state["priority_overrides"] = {}

if not uploaded_files:
    st.warning("No images uploaded. Go to the **Upload** page first.")
    st.page_link("pages/1_Upload.py", label="Go to Upload", icon="\U0001f4e4")
    st.stop()

# Load model(s)
with st.spinner("Loading model(s)..."):
    if mode == "Ensemble (Both Models)":
        ensemble, ensemble_paths = load_ensemble_models(version="auto")
        if not ensemble:
            st.error("No custom models found in models/ directory.")
            st.stop()
        st.success(f"Ensemble loaded: {', '.join(ensemble.keys())}")
    elif mode == "Compare V1 vs V2":
        ensemble_v1, paths_v1 = load_ensemble_models(version="v1")
        ensemble_v2, paths_v2 = load_ensemble_models(version="v2")
        st.success(
            f"V1: {', '.join(ensemble_v1.keys())}  |  V2: {', '.join(ensemble_v2.keys())}"
        )
    else:
        model = load_model(selected_model_path)
        st.success("Model loaded")

# Run detection button
if st.button("Run Detection on All Images", type="primary"):
    progress = st.progress(0, text="Running detection...")

    if mode == "Compare V1 vs V2":
        all_v1, all_v2 = [], []
        time_v1 = time_v2 = 0.0
        fusion = "wbf" if use_wbf else "iou_dedup"

        for i, file in enumerate(uploaded_files):
            file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
            file.seek(0)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            r1 = run_ensemble_inference(
                ensemble_v1, image,
                confidence=confidence, imgsz=imgsz, augment=use_tta,
                use_sahi=use_sahi, sahi_slice_size=sahi_slice_size,
                model_paths=paths_v1, fusion_method=fusion,
            )
            r2 = run_ensemble_inference(
                ensemble_v2, image,
                confidence=confidence, imgsz=imgsz, augment=use_tta,
                use_sahi=use_sahi, sahi_slice_size=sahi_slice_size,
                model_paths=paths_v2, fusion_method=fusion,
            )
            time_v1 += r1.get("inference_time_ms", 0)
            time_v2 += r2.get("inference_time_ms", 0)
            all_v1.append({"filename": file.name, "result": r1, "original": image.copy()})
            all_v2.append({"filename": file.name, "result": r2, "original": image.copy()})

            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"Comparing {file.name}... ({i + 1}/{len(uploaded_files)})",
            )

        # Canonical results for Map/Report = V2 by default
        st.session_state["detection_results"] = all_v2
        st.session_state["detection_results_v1"] = all_v1
        st.session_state["detection_results_v2"] = all_v2
        st.session_state["compare_mode"] = True
        st.session_state["compare_active_version"] = "v2"
        st.session_state["compare_timing"] = {"v1": time_v1, "v2": time_v2}
        progress.empty()

        total_v1 = sum(len(r["result"]["detections"]) for r in all_v1)
        total_v2 = sum(len(r["result"]["detections"]) for r in all_v2)
        st.success(
            f"Comparison complete — V1: **{total_v1}** dets in "
            f"{time_v1/1000:.1f}s  |  V2: **{total_v2}** dets in {time_v2/1000:.1f}s"
        )

    else:
        all_results = []
        total_per_model = {}
        total_inference_ms = 0

        for i, file in enumerate(uploaded_files):
            file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
            file.seek(0)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if mode == "Ensemble (Both Models)":
                result = run_ensemble_inference(
                    ensemble, image,
                    confidence=confidence,
                    imgsz=imgsz,
                    augment=use_tta,
                    use_sahi=use_sahi,
                    sahi_slice_size=sahi_slice_size,
                    model_paths=ensemble_paths,
                    fusion_method="wbf" if use_wbf else "iou_dedup",
                )
                # Accumulate per-model counts
                for name, count in result.get("per_model_counts", {}).items():
                    total_per_model[name] = total_per_model.get(name, 0) + count
            else:
                result = run_inference(
                    model, image,
                    confidence=confidence,
                    imgsz=imgsz,
                    augment=use_tta,
                    model_name=Path(selected_model_path).stem if selected_model_path else "single",
                )

            total_inference_ms += result.get("inference_time_ms", 0)

            all_results.append({
                "filename": file.name,
                "result": result,
            })

            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"Processing {file.name}... ({i + 1}/{len(uploaded_files)})",
            )

        st.session_state["detection_results"] = all_results
        st.session_state.pop("compare_mode", None)
        progress.empty()

        # Show results summary
        total_merged = sum(len(r["result"]["detections"]) for r in all_results)

        # Timing display
        time_str = f"{total_inference_ms / 1000:.1f}s" if total_inference_ms >= 1000 else f"{total_inference_ms:.0f}ms"
        settings_parts = [f"{imgsz}px"]
        if use_tta:
            settings_parts.append("TTA")
        if use_sahi:
            settings_parts.append(f"SAHI {sahi_slice_size}px")
        if use_wbf and mode == "Ensemble (Both Models)":
            settings_parts.append("WBF")
        settings_str = " | ".join(settings_parts)

        if mode == "Ensemble (Both Models)" and total_per_model:
            raw_total = sum(total_per_model.values())
            model_breakdown = ", ".join(f"{n}: {c}" for n, c in total_per_model.items())
            st.success(
                f"Detection complete in **{time_str}** ({settings_str}) — {model_breakdown} | "
                f"After dedup: **{total_merged}** unique detections"
            )
        else:
            st.success(
                f"Detection complete in **{time_str}** ({settings_str}) — "
                f"{total_merged} detections across {len(uploaded_files)} image(s)"
            )

# ==========================================
# COMPARISON VIEW (V1 vs V2)
# ==========================================
# Belt-and-braces: clear a stale compare_mode flag if the result lists are gone
# (e.g. partial cleanup left the flag set without populated v1/v2 results).
if st.session_state.get("compare_mode") and not (
    st.session_state.get("detection_results_v1")
    and st.session_state.get("detection_results_v2")
):
    st.session_state.pop("compare_mode", None)

if st.session_state.get("compare_mode"):
    results_v1 = st.session_state.get("detection_results_v1", [])
    results_v2 = st.session_state.get("detection_results_v2", [])
    timing = st.session_state.get("compare_timing", {"v1": 0, "v2": 0})

    all_v1 = [d for r in results_v1 for d in r["result"]["detections"]]
    all_v2 = [d for r in results_v2 for d in r["result"]["detections"]]
    sum_v1 = summarize_detections(all_v1)
    sum_v2 = summarize_detections(all_v2)

    # Spatial agreement: V1 detection counts as a match if any V2 detection on
    # the same image overlaps it at IoU >= 0.5.
    agree_count = 0
    for r1, r2 in zip(results_v1, results_v2):
        d1s = r1["result"]["detections"]
        d2s = r2["result"]["detections"]
        for d1 in d1s:
            for d2 in d2s:
                if compute_iou(d1["bbox"], d2["bbox"]) >= 0.5:
                    agree_count += 1
                    break

    st.subheader("V1 vs V2 Comparison")

    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        st.markdown("### V1")
        st.metric("Total detections", sum_v1["total_detections"])
        st.metric("Inference time", f"{timing['v1']/1000:.1f}s")
        st.caption(
            f"Critical: {sum_v1['by_priority'].get('critical', 0)}  |  "
            f"Urgent: {sum_v1['by_priority'].get('urgent', 0)}  |  "
            f"Monitor: {sum_v1['by_priority'].get('monitor', 0)}  |  "
            f"Routine: {sum_v1['by_priority'].get('routine', 0)}"
        )
    with cmp_col2:
        st.markdown("### V2")
        st.metric("Total detections", sum_v2["total_detections"])
        st.metric("Inference time", f"{timing['v2']/1000:.1f}s")
        st.caption(
            f"Critical: {sum_v2['by_priority'].get('critical', 0)}  |  "
            f"Urgent: {sum_v2['by_priority'].get('urgent', 0)}  |  "
            f"Monitor: {sum_v2['by_priority'].get('monitor', 0)}  |  "
            f"Routine: {sum_v2['by_priority'].get('routine', 0)}"
        )

    if sum_v1["total_detections"] > 0:
        st.caption(
            f"Spatial agreement (IoU≥0.5): **{agree_count} of {sum_v1['total_detections']}** "
            f"V1 detections match a V2 detection."
        )

    st.markdown("---")

    # Side-by-side per-image. Compare mode is intentionally raw-boxes —
    # clustering applies only to single-version views (Single + Ensemble).
    for r1, r2 in zip(results_v1, results_v2):
        with st.expander(
            f"\U0001f4f7 {r1['filename']} — V1: {len(r1['result']['detections'])} dets  |  "
            f"V2: {len(r2['result']['detections'])} dets"
        ):
            img_col0, img_col1, img_col2 = st.columns(3)
            dets_v1 = r1["result"]["detections"]
            dets_v2 = r2["result"]["detections"]
            mask_count_v1 = sum(1 for d in dets_v1 if d.get("mask") is not None)
            mask_count_v2 = sum(1 for d in dets_v2 if d.get("mask") is not None)
            with img_col0:
                st.markdown("**Original**")
                st.image(
                    cv2.cvtColor(r1["original"], cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    output_format="PNG",
                )
                st.caption("Photo as uploaded")
            with img_col1:
                st.markdown(f"**V1** — {r1['result'].get('inference_time_ms', 0):.0f}ms")
                st.image(
                    cv2.cvtColor(r1["result"]["annotated_image"], cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    output_format="PNG",
                )
                st.caption(f"Mask outlines: {mask_count_v1}/{len(dets_v1)}")
            with img_col2:
                st.markdown(f"**V2** — {r2['result'].get('inference_time_ms', 0):.0f}ms")
                st.image(
                    cv2.cvtColor(r2["result"]["annotated_image"], cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    output_format="PNG",
                )
                st.caption(f"Mask outlines: {mask_count_v2}/{len(dets_v2)}")

    st.markdown("---")

    # Result-set toggle for downstream Map/Report pages
    active = st.session_state.get("compare_active_version", "v2")
    choice = st.radio(
        "Use which results for Map/Report?",
        options=["V2", "V1"],
        index=0 if active == "v2" else 1,
        horizontal=True,
        key="compare_result_set",
    )
    new_active = "v2" if choice == "V2" else "v1"
    if new_active != active:
        st.session_state["compare_active_version"] = new_active
        st.session_state["detection_results"] = (
            st.session_state["detection_results_v2"]
            if new_active == "v2"
            else st.session_state["detection_results_v1"]
        )
        st.rerun()
    else:
        st.session_state["detection_results"] = (
            st.session_state["detection_results_v2"]
            if new_active == "v2"
            else st.session_state["detection_results_v1"]
        )

    st.caption(f"Map and Report will use **{choice}** results.")
    st.markdown("---")

# Display results
if "detection_results" in st.session_state:
    results = st.session_state["detection_results"]

    # Apply inspector overrides BEFORE summary so banners + metrics + filter
    # all reflect the human-classified severity. Streamlit reruns on every
    # button click, so this happens fresh each interaction.
    apply_priority_overrides(
        results,
        st.session_state.get("priority_overrides", {}),
        SAFETY_PRIORITIES,
    )

    # Summary metrics
    all_dets = [d for r in results for d in r["result"]["detections"]]
    summary = summarize_detections(all_dets)

    # ==========================================
    # SAFETY ALERT BANNERS
    # ==========================================
    by_priority = summary.get("by_priority", {})

    if by_priority.get("critical", 0) > 0:
        st.error(
            f"\u26a0\ufe0f **CRITICAL SAFETY ALERT** — "
            f"{by_priority['critical']} critical defect(s) detected requiring immediate attention. "
            f"These pose direct danger to road users."
        )

    if by_priority.get("urgent", 0) > 0:
        st.warning(
            f"\U0001f7e0 **URGENT** — "
            f"{by_priority['urgent']} defect(s) require priority maintenance within 1-2 weeks."
        )

    if summary.get("cluster_risk", False):
        st.error(
            "\u26a0\ufe0f **SYSTEMIC PAVEMENT FAILURE DETECTED** — "
            "High defect density indicates structural failure across this section. "
            "Recommend full section assessment and possible rehabilitation rather than spot repairs."
        )

    # ==========================================
    # SEVERITY METRICS ROW
    # ==========================================
    st.subheader("Severity Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Detections", summary["total_detections"])
    col2.metric("High Severity", summary["by_severity"]["high"])
    col3.metric("Medium Severity", summary["by_severity"]["medium"])
    col4.metric("Low Severity", summary["by_severity"]["low"])

    # ==========================================
    # PRIORITY METRICS ROW
    # ==========================================
    st.subheader("Safety Priority")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)

    critical_count = by_priority.get("critical", 0)
    urgent_count = by_priority.get("urgent", 0)
    monitor_count = by_priority.get("monitor", 0)
    routine_count = by_priority.get("routine", 0)

    pcol1.markdown(
        f"<div style='text-align:center;padding:8px;background:#FF1744;color:white;border-radius:8px;'>"
        f"<b>CRITICAL</b><br><span style='font-size:2em;'>{critical_count}</span></div>",
        unsafe_allow_html=True,
    )
    pcol2.markdown(
        f"<div style='text-align:center;padding:8px;background:#FF9100;color:white;border-radius:8px;'>"
        f"<b>URGENT</b><br><span style='font-size:2em;'>{urgent_count}</span></div>",
        unsafe_allow_html=True,
    )
    pcol3.markdown(
        f"<div style='text-align:center;padding:8px;background:#FFC400;color:#333;border-radius:8px;'>"
        f"<b>MONITOR</b><br><span style='font-size:2em;'>{monitor_count}</span></div>",
        unsafe_allow_html=True,
    )
    pcol4.markdown(
        f"<div style='text-align:center;padding:8px;background:#00C853;color:white;border-radius:8px;'>"
        f"<b>ROUTINE</b><br><span style='font-size:2em;'>{routine_count}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ==========================================
    # CHARTS
    # ==========================================
    if summary["total_detections"] > 0:
        col_chart, col_class = st.columns(2)

        with col_chart:
            fig = go.Figure(data=[go.Pie(
                labels=["Critical", "Urgent", "Monitor", "Routine"],
                values=[critical_count, urgent_count, monitor_count, routine_count],
                marker_colors=["#FF1744", "#FF9100", "#FFC400", "#00C853"],
                hole=0.4,
            )])
            fig.update_layout(title="Safety Priority Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col_class:
            if summary["by_class"]:
                fig2 = go.Figure(data=[go.Bar(
                    x=list(summary["by_class"].keys()),
                    y=list(summary["by_class"].values()),
                    marker_color="#FF4B4B",
                )])
                fig2.update_layout(title="Detections by Class")
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # PER-IMAGE RESULTS — cluster cards (Phase 1)
    # ==========================================
    # AI surfaces candidate regions via spatial clustering with multi-model
    # agreement as the trust signal; the inspector classifies severity via
    # 4-button override per cluster. Per-detection auto-severity is shown
    # only inside the "Underlying detections" expander as raw model output.

    overrides = st.session_state["priority_overrides"]
    priority_levels = ["routine", "monitor", "urgent", "critical"]
    priority_rank = {"routine": 0, "monitor": 1, "urgent": 2, "critical": 3}

    for item in results:
        clusters = item["result"].get("clusters", [])
        dets = item["result"].get("detections", [])

        def _eff_priority(cluster, _filename=item["filename"]):
            key = f"{_filename}::{cluster['cluster_id']}"
            return overrides.get(key, cluster.get("suggested_priority", "routine"))

        filtered_clusters = [c for c in clusters if _eff_priority(c) in priority_filter]
        hidden_count = len(clusters) - len(filtered_clusters)

        header = f"\U0001f4f7 {item['filename']} — {len(filtered_clusters)} regions"
        if hidden_count > 0:
            header += f" ({hidden_count} filtered out)"

        with st.expander(header):
            # Cluster outline image (raw boxes are kept for Compare mode).
            # `is not None` check, not `or` — numpy arrays raise on bool().
            clusters_img = item["result"].get("annotated_clusters")
            annotated = clusters_img if clusters_img is not None else item["result"]["annotated_image"]
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True, output_format="PNG")

            img_time = item["result"].get("inference_time_ms", 0)
            if img_time > 0:
                st.caption(f"Inference time: {img_time:.0f}ms")

            if not filtered_clusters:
                if clusters:
                    st.caption("All regions filtered out by sidebar priority filter.")
                else:
                    st.caption("No detections on this image.")
                continue

            sorted_clusters = sorted(
                filtered_clusters,
                key=lambda c: (
                    -priority_rank.get(_eff_priority(c), 0),
                    -c.get("detection_count", 0),
                ),
            )

            for cluster in sorted_clusters:
                cid = cluster["cluster_id"]
                key = f"{item['filename']}::{cid}"
                suggested = cluster.get("suggested_priority", "routine")
                effective = overrides.get(key, suggested)
                eff_info = SAFETY_PRIORITIES.get(effective, SAFETY_PRIORITIES["routine"])
                is_overridden = key in overrides and overrides[key] != suggested

                n_signals = cluster.get("detection_count", 0)
                n_models = len(cluster.get("models_agreeing", []))
                models_str = ", ".join(cluster.get("models_agreeing", [])) or "—"
                mask_pct = cluster.get("mask_coverage_pct", 0.0)

                chip_color = eff_info["color_hex"]
                chip_text_color = "#333" if effective == "monitor" else "white"
                chip_label = eff_info["label"]
                if is_overridden:
                    chip_label += " (inspector)"

                st.markdown(
                    f"<div style='border-left:4px solid {chip_color};padding:10px 14px;"
                    f"margin:8px 0;background:#fafafa;border-radius:6px;'>"
                    f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"
                    f"<span style='font-size:1.05em;font-weight:600;'>"
                    f"Region {cid + 1} &middot; {n_signals} signals &middot; {n_models}-model agreement</span>"
                    f"<span style='background:{chip_color};color:{chip_text_color};"
                    f"padding:3px 10px;border-radius:12px;font-weight:700;font-size:0.78em;'>"
                    f"{chip_label}</span>"
                    f"</div>"
                    f"<div style='color:#888;font-size:0.85em;margin-top:4px;'>"
                    f"AI suggests: <b>{suggested.upper()}</b> &middot; "
                    f"Models: {models_str} &middot; Mask coverage: {mask_pct:.0f}%"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                bcols = st.columns(4)
                for i, level in enumerate(priority_levels):
                    info = SAFETY_PRIORITIES[level]
                    btn_label = info["label"].title()
                    if effective == level:
                        btn_label = f"✓ {btn_label}"
                    if bcols[i].button(
                        btn_label,
                        key=f"override_{item['filename']}_{cid}_{level}",
                        use_container_width=True,
                    ):
                        if level == suggested:
                            st.session_state["priority_overrides"].pop(key, None)
                        else:
                            st.session_state["priority_overrides"][key] = level
                        st.rerun()

                indices = cluster.get("detection_indices", [])
                with st.expander(f"Underlying detections ({len(indices)})", expanded=False):
                    rows = []
                    for idx in indices:
                        if idx >= len(dets):
                            continue
                        d = dets[idx]
                        rows.append({
                            "class": d.get("class_name", ""),
                            "confidence": f"{d.get('confidence', 0):.0%}",
                            "severity": d.get("severity", ""),
                            "raw_priority": d.get("original_safety_priority", d.get("safety_priority", "")),
                            "width_px": f"{d.get('width_pixels', 0):.0f}",
                            "models": ", ".join(d.get("models_agreeing", [])),
                        })
                    if rows:
                        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info("Head to the **Map** page to see detections on an interactive map.")
