"""Download sample crack images for testing the PaveScan AI dashboard.

Downloads a few images from the Ultralytics crack-seg dataset.
Run this once to populate data/sample/ with test images.

Usage:
    python scripts/download_samples.py
"""

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

CRACK_SEG_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip"
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
NUM_SAMPLES = 10  # how many images to extract


def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading crack-seg dataset (~92 MB)...")
    response = urlopen(CRACK_SEG_URL)
    zip_bytes = io.BytesIO(response.read())
    print("Download complete. Extracting sample images...")

    count = 0
    with zipfile.ZipFile(zip_bytes) as zf:
        # Get test images first (most representative), then train
        image_files = [
            f for f in zf.namelist()
            if f.endswith(".jpg") and "images/" in f
        ]

        # Prefer test images
        test_images = [f for f in image_files if "test/" in f]
        train_images = [f for f in image_files if "train/" in f]
        selected = (test_images + train_images)[:NUM_SAMPLES]

        for file_path in selected:
            filename = Path(file_path).name
            out_path = SAMPLE_DIR / filename
            with zf.open(file_path) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            count += 1
            print(f"  [{count}/{NUM_SAMPLES}] {filename}")

    print(f"\nDone! {count} sample images saved to {SAMPLE_DIR}")
    print("You can now test the Upload and Detection pages in the dashboard.")


if __name__ == "__main__":
    main()
