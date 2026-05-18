"""DRR Phase A — V1 recall regression diagnostic.

Sweeps imgsz x confidence x fusion across V1/V2 single + ensemble configurations
on a sample image, plus an independent SAHI sweep and a pipeline bypass test.
Writes a numbers-only report; no annotated outputs.

Run from pavescan-ai/:
    .venv/Scripts/python.exe scripts/diagnose_v1_recall.py
    .venv/Scripts/python.exe scripts/diagnose_v1_recall.py path/to/image.jpg
"""

from __future__ import annotations

import hashlib
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detection.model import (  # noqa: E402
    MODELS_DIR,
    _extract_detections,
    deduplicate_detections,
    load_model,
    run_ensemble_inference,
    run_inference,
    run_sahi_inference,
)

IMGSZ_SWEEP = [640, 960, 1280, 1600]
CONF_SWEEP = [0.05, 0.10, 0.15, 0.25]
SAHI_SLICE_SWEEP = [320, 480, 640]
SAHI_OVERLAP_SWEEP = [0.2, 0.4]

V1_SEG_NAME = "pavescan_crack_seg.pt"
V1_DET_NAME = "pavescan_rdd2022.pt"
V2_SEG_NAME = "pavescan_crack_seg_v2.pt"

SAMPLE_DIR = REPO_ROOT / "data" / "sample"
REPORTS_DIR = REPO_ROOT / "reports"


class Tee:
    """Mirror writes to a file and stdout so the user can read live or copy from disk."""

    def __init__(self, path: Path):
        self.fp = path.open("w", encoding="utf-8")

    def __call__(self, line: str = "") -> None:
        print(line)
        self.fp.write(line + "\n")
        self.fp.flush()

    def close(self) -> None:
        self.fp.close()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_cell(n: int, mean_conf: float | None) -> str:
    if n == 0:
        return f"{n:>3} /  ----"
    return f"{n:>3} / {mean_conf:.3f}"


def mean_conf(dets: list[dict]) -> float | None:
    if not dets:
        return None
    return statistics.fmean(float(d["confidence"]) for d in dets)


def resolve_images() -> list[Path]:
    if len(sys.argv) >= 2:
        p = Path(sys.argv[1])
        if not p.exists():
            sys.exit(f"image path does not exist: {p}")
        return [p]
    images = sorted(SAMPLE_DIR.glob("*.jpg"))[:3]
    if not images:
        sys.exit(f"no .jpg images found in {SAMPLE_DIR}")
    return images


def integrity_check(out: Tee, weight_names: list[str]) -> dict[str, "YOLO"]:
    """SHA + sanity-load each weight. Returns loaded models keyed by filename."""
    out("=" * 78)
    out("WEIGHT INTEGRITY CHECK")
    out("=" * 78)
    loaded: dict[str, object] = {}
    for name in weight_names:
        path = MODELS_DIR / name
        out(f"\n{name}")
        if not path.exists():
            out(f"  MISSING — expected at {path}")
            continue
        stat = path.stat()
        out(f"  size      : {stat.st_size:,} bytes")
        out(f"  mtime     : {datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds')}")
        out(f"  sha256    : {sha256_file(path)}")
        t0 = time.perf_counter()
        try:
            m = load_model(str(path))
            ms = (time.perf_counter() - t0) * 1000
            try:
                num_params = sum(p.numel() for p in m.model.parameters())
            except Exception as e:
                num_params = f"<err: {e}>"
            out(f"  task      : {m.task}")
            out(f"  classes   : {list(m.names.values())}")
            out(f"  n_params  : {num_params:,}" if isinstance(num_params, int) else f"  n_params  : {num_params}")
            out(f"  load time : {ms:.0f} ms")
            loaded[name] = m
        except Exception as e:
            out(f"  LOAD FAILED: {e}")
    return loaded


def matrix_header(out: Tee, title: str) -> None:
    out("")
    out(f"--- {title} ---")
    header = f"{'imgsz':>6} | " + " | ".join(f"conf={c:.2f}".rjust(14) for c in CONF_SWEEP)
    out(header)
    out("-" * len(header))


def matrix_row(out: Tee, imgsz: int, cells: list[tuple[int, float | None]]) -> None:
    row = f"{imgsz:>6} | " + " | ".join(fmt_cell(n, mc).rjust(14) for n, mc in cells)
    out(row)


def sweep_single(out: Tee, label: str, model, image) -> None:
    matrix_header(out, label)
    for imgsz in IMGSZ_SWEEP:
        cells: list[tuple[int, float | None]] = []
        for conf in CONF_SWEEP:
            try:
                r = run_inference(model, image, confidence=conf, imgsz=imgsz)
                dets = r["detections"]
                cells.append((len(dets), mean_conf(dets)))
            except Exception as e:
                out(f"    [error imgsz={imgsz} conf={conf}: {e}]")
                cells.append((-1, None))
        matrix_row(out, imgsz, cells)


def sweep_ensemble(out: Tee, label: str, models: dict, image, fusion: str) -> None:
    matrix_header(out, f"{label} (fusion={fusion})")
    for imgsz in IMGSZ_SWEEP:
        cells: list[tuple[int, float | None]] = []
        for conf in CONF_SWEEP:
            try:
                r = run_ensemble_inference(
                    models=models,
                    image=image,
                    confidence=conf,
                    imgsz=imgsz,
                    fusion_method=fusion,
                )
                dets = r["detections"]
                cells.append((len(dets), mean_conf(dets)))
            except Exception as e:
                out(f"    [error imgsz={imgsz} conf={conf}: {e}]")
                cells.append((-1, None))
        matrix_row(out, imgsz, cells)


def sweep_sahi(out: Tee, image) -> None:
    out("")
    out("--- V1 seg SAHI sweep (conf=0.10) ---")
    header = f"{'slice':>6} | " + " | ".join(f"overlap={o:.1f}".rjust(16) for o in SAHI_OVERLAP_SWEEP)
    out(header)
    out("-" * len(header))
    for slice_size in SAHI_SLICE_SWEEP:
        row_cells: list[str] = []
        for overlap in SAHI_OVERLAP_SWEEP:
            try:
                dets = run_sahi_inference(
                    model_path=str(MODELS_DIR / V1_SEG_NAME),
                    image=image,
                    confidence=0.10,
                    slice_size=slice_size,
                    overlap_ratio=overlap,
                )
                row_cells.append(fmt_cell(len(dets), mean_conf(dets)).rjust(16))
            except Exception as e:
                row_cells.append(f"[err: {type(e).__name__}]".rjust(16))
        out(f"{slice_size:>6} | " + " | ".join(row_cells))


def bypass_test(out: Tee, model, image) -> None:
    out("")
    out("--- Bypass test (V1 seg, conf=0.10, imgsz=640) ---")
    out("Compares: raw YOLO boxes -> _extract_detections -> deduplicate_detections")
    try:
        results = model.predict(image, conf=0.10, imgsz=640, verbose=False)
        result = results[0]
        raw_n = 0 if result.boxes is None else len(result.boxes)
        extracted = _extract_detections(result, image.shape)
        deduped = deduplicate_detections(list(extracted))
        out(f"  raw model.predict boxes      : {raw_n}")
        out(f"  after _extract_detections    : {len(extracted)}")
        out(f"  after deduplicate_detections : {len(deduped)}")
        delta_extract = raw_n - len(extracted)
        delta_dedup = len(extracted) - len(deduped)
        out(f"  drop in extract step         : {delta_extract}")
        out(f"  drop in dedup step           : {delta_dedup}")
    except Exception as e:
        out(f"  BYPASS FAILED: {e}")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"diagnosis_v1_{timestamp}.txt"
    out = Tee(out_path)

    try:
        out(f"DRR Phase A — V1 recall diagnostic")
        out(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
        out(f"report   : {out_path}")

        images = resolve_images()
        out(f"images   : {len(images)} picked")
        for p in images:
            out(f"  - {p}")

        loaded = integrity_check(out, [V1_SEG_NAME, V1_DET_NAME, V2_SEG_NAME])

        v1_seg = loaded.get(V1_SEG_NAME)
        v1_det = loaded.get(V1_DET_NAME)
        v2_seg = loaded.get(V2_SEG_NAME)

        if v1_seg is None:
            out("\nABORT: V1 seg weight missing or failed to load. Cannot continue.")
            return

        for idx, img_path in enumerate(images, 1):
            image = cv2.imread(str(img_path))
            if image is None:
                out(f"\nSKIP image {idx}: cv2.imread returned None for {img_path}")
                continue

            out("")
            out("=" * 78)
            out(f"IMAGE {idx}/{len(images)}: {img_path.name}  shape={image.shape}")
            out("=" * 78)

            if v1_seg is not None:
                sweep_single(out, "V1 seg alone (pavescan_crack_seg.pt)", v1_seg, image)
            if v1_det is not None:
                sweep_single(out, "V1 det alone (pavescan_rdd2022.pt)", v1_det, image)
            if v2_seg is not None:
                sweep_single(out, "V2 seg alone (pavescan_crack_seg_v2.pt)", v2_seg, image)

            if v1_seg is not None and v1_det is not None:
                ens_v1 = {"crack_seg": v1_seg, "rdd2022": v1_det}
                sweep_ensemble(out, "V1 ensemble", ens_v1, image, fusion="iou_dedup")
                sweep_ensemble(out, "V1 ensemble", ens_v1, image, fusion="wbf")

            if v2_seg is not None and v1_det is not None:
                ens_v2 = {"crack_seg_v2": v2_seg, "rdd2022": v1_det}
                sweep_ensemble(out, "V2 ensemble (rdd2022 fallback to V1 det)", ens_v2, image, fusion="iou_dedup")

            sweep_sahi(out, image)

            if idx == 1 and v1_seg is not None:
                bypass_test(out, v1_seg, image)

        out("")
        out("=" * 78)
        out("DONE")
        out(f"report written to: {out_path}")
        out("=" * 78)
    finally:
        out.close()


if __name__ == "__main__":
    main()
