"""Download model weights from the GitHub Release on first start.

Idempotent: re-runs are cheap (sha256 check only) when weights are already
present and uncorrupted. Safe to call from app startup, from a CLI shell, or
from a CI/CD step.
"""

import hashlib
import urllib.request
from pathlib import Path

WEIGHTS_RELEASE_TAG = "weights-v1.0"
WEIGHTS_BASE_URL = (
    "https://github.com/awadmohamed11129-oss/pavescan-ai/"
    f"releases/download/{WEIGHTS_RELEASE_TAG}"
)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# {filename: (sha256, size_bytes)} — sizes are advisory (used by progress UI)
EXPECTED_WEIGHTS = {
    "pavescan_crack_seg.pt": (
        "a53e9af5ad685796cb3b5269eda4c5a111a483d07b5d9a33029de5424aded323",
        6_782_260,
    ),
    "pavescan_crack_seg_v2.pt": (
        "8740dbaf489e679d177bb88d09237348b583f6ea942bab8369c09e7c8813d99d",
        55_993_569,
    ),
    "pavescan_rdd2022.pt": (
        "fac7777db23e1f82fbd5de2e0f02836006f597f5a39ac5a18bf2d92a8225c572",
        6_247_914,
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_missing_or_invalid() -> list[str]:
    """Return filenames whose local copy is missing or whose sha256 doesn't match."""
    needed = []
    for name, (expected_hash, _size) in EXPECTED_WEIGHTS.items():
        path = MODELS_DIR / name
        if not path.exists() or _sha256(path) != expected_hash:
            needed.append(name)
    return needed


def ensure_weights_present(progress_callback=None) -> None:
    """Download any missing or corrupted weights from the GitHub Release.

    progress_callback, if provided, is called as `cb(filename, downloaded_bytes, total_bytes)`
    during each download. With no callback, status is logged via print().
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in list_missing_or_invalid():
        expected_hash, total = EXPECTED_WEIGHTS[name]
        url = f"{WEIGHTS_BASE_URL}/{name}"
        dest = MODELS_DIR / name

        if progress_callback:
            progress_callback(name, 0, total)
        else:
            print(f"[fetch_weights] downloading {name} ({total / 1e6:.1f} MB) from {url}")

        def _hook(blocks, blocksize, _totalsize, _name=name, _total=total):
            done = min(blocks * blocksize, _total)
            if progress_callback:
                progress_callback(_name, done, _total)

        try:
            urllib.request.urlretrieve(url, dest, reporthook=_hook)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {name} from {url}. "
                f"Confirm the GitHub Release '{WEIGHTS_RELEASE_TAG}' exists and has "
                f"'{name}' attached as an asset. Underlying error: {exc}"
            ) from exc

        got = _sha256(dest)
        if got != expected_hash:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch for {name}: expected {expected_hash}, got {got}. "
                f"Downloaded file has been deleted. The asset on the GitHub Release "
                f"may have been replaced or corrupted in transit."
            )


if __name__ == "__main__":
    ensure_weights_present()
    print("[fetch_weights] all weights present and verified")
