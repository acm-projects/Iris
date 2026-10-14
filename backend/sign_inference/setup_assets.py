"""Download and verify the assets required by standalone camera inference."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    path: Path
    sha256: str | None = None


ASSETS = (
    Asset(
        name="WLASL-100 TGCN checkpoint",
        url=(
            "https://huggingface.co/sharonn18/tgcn-wlasl/resolve/"
            "dacb4568719caa03c44764034f599a9f8a0f63f4/"
            "checkpoints/asl100/pytorch_model.bin?download=true"
        ),
        path=ROOT / "checkpoints" / "asl100" / "pytorch_model.bin",
        sha256="0a66a799f12dd5d8a48a02b441e7499b2ed89ada3e130e059c7e9afc63312e19",
    ),
    Asset(
        name="MediaPipe pose landmarker",
        url=(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        ),
        path=ROOT / "mediapipe_models" / "pose_landmarker.task",
    ),
    Asset(
        name="MediaPipe hand landmarker",
        url=(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        ),
        path=ROOT / "mediapipe_models" / "hand_landmarker.task",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(asset: Asset) -> tuple[bool, str]:
    if not asset.path.is_file():
        return False, "missing"
    if asset.path.stat().st_size == 0:
        return False, "empty"
    if asset.sha256 is not None:
        actual = file_sha256(asset.path)
        if actual != asset.sha256:
            return False, f"checksum mismatch ({actual})"
    return True, f"ok ({asset.path.stat().st_size:,} bytes)"


def download(asset: Asset) -> None:
    asset.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = asset.path.with_suffix(asset.path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "Iris-WLASL-asset-setup/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            with temporary.open("wb") as destination:
                shutil.copyfileobj(response, destination, length=1024 * 1024)
        temporary.replace(asset.path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="download every asset again, even when the existing file verifies",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check local assets without downloading missing files",
    )
    args = parser.parse_args()

    failed = False
    for asset in ASSETS:
        valid, detail = verify(asset)
        if valid and not args.force:
            print(f"[ok] {asset.name}: {detail}")
            continue
        if args.verify_only:
            print(f"[missing] {asset.name}: {detail}", file=sys.stderr)
            failed = True
            continue
        print(f"[download] {asset.name}")
        try:
            download(asset)
        except (OSError, urllib.error.URLError) as error:
            print(f"[error] {asset.name}: {error}", file=sys.stderr)
            failed = True
            continue
        valid, detail = verify(asset)
        stream = sys.stdout if valid else sys.stderr
        print(f"[{'ok' if valid else 'error'}] {asset.name}: {detail}", file=stream)
        failed = failed or not valid

    if failed:
        print("One or more inference assets are unavailable.", file=sys.stderr)
        return 1
    print("All inference assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
