#!/usr/bin/env python3
"""RatCatcher AI -- Hailo Calibration Set Builder

Builds the calibration array that the Hailo Dataflow Compiler needs to
quantize ratcatcher_best.onnx from float32 to the INT8 the NPU runs.

WHY A CALIBRATION SET

Quantization picks a scale and zero-point per tensor by observing the
range of activations on real data.  Feed it noise and the ranges are
wrong, so the quantized model loses accuracy in a way that no amount of
threshold tuning afterwards recovers.  Feed it images from the same
distribution as deployment and the INT8 model tracks the float model
closely.

PREPROCESSING MUST MATCH INFERENCE

This is the part that is easy to get wrong and hard to debug, because a
mismatch degrades accuracy without ever raising an error.  The array
written here is preprocessed exactly as HailoDetector.detect() does at
runtime:

  * plain cv2.resize to 640x640 -- a STRETCH, not a letterbox.  Both
    shipped backends (hailo_detector.py and opencv_detector.py) stretch,
    and rescale boxes back with independent scale_x/scale_y, so
    calibrating on letterboxed images would introduce a distribution
    mismatch.  (Ultralytics trains with letterbox, so this stretch is a
    pre-existing train/deploy discrepancy in its own right -- but it is
    consistent across both backends, and calibration must match what
    actually runs, not what would be ideal.)
  * BGR to RGB.
  * uint8 in 0-255, NOT scaled to 0-1.  Normalization is compiled into
    the HEF as an on-chip layer (see build_hef.py), which is what lets
    the runtime feed uint8 straight over PCIe at a quarter the bandwidth
    of float32.

Output is an .npy array of shape (N, 640, 640, 3), dtype uint8 (NHWC).

This script has no Hailo dependencies -- it runs anywhere numpy and
OpenCV do, including the Pi.  Only build_hef.py needs the DFC.

Usage:
    python training/build_calibration_set.py
    python training/build_calibration_set.py --count 1024 --seed 7
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# Extensions Open Images ships and that cv2.imread handles.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Hailo's guidance is 64 images minimum; the Model Zoo uses 1024 for
# detection models.  256 sits where the accuracy curve has flattened for
# a 5-class model but the optimize step still finishes in a few minutes.
DEFAULT_COUNT = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Hailo INT8 calibration set from training images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images",
        type=str,
        default="datasets/ratcatcher/train/images",
        help="Directory of source images to sample from.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/calibration_set.npy",
        help="Destination .npy file.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of images to sample.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square network input size. Must match the compiled HEF.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed, so the same calibration set can be rebuilt exactly.",
    )
    return parser.parse_args()


def collect_images(images_dir: Path) -> list[Path]:
    """List usable image files under ``images_dir``, sorted for determinism."""
    if not images_dir.is_dir():
        print(f"[ERROR] Not a directory: {images_dir.resolve()}")
        print("[ERROR] Fetch the training data first with:")
        print("[ERROR]   python training/download_data.py")
        sys.exit(1)

    paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        print(f"[ERROR] No images found in {images_dir.resolve()}")
        print(f"[ERROR] Looked for: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        sys.exit(1)
    return paths


def preprocess(image_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """Resize and convert one BGR image the way HailoDetector.detect() does.

    Returns an (imgsz, imgsz, 3) uint8 RGB array.
    """
    resized = cv2.resize(
        image_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR
    )
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def build(args: argparse.Namespace) -> int:
    images_dir = Path(args.images)
    output_path = Path(args.output)

    paths = collect_images(images_dir)
    print(f"[INFO] Found {len(paths)} images in {images_dir}")

    if args.count > len(paths):
        print(
            f"[WARN] Requested {args.count} images but only {len(paths)} "
            f"exist. Using all {len(paths)}."
        )
        selected = paths
    else:
        # Seeded sample over a sorted list: reproducible across machines.
        selected = random.Random(args.seed).sample(paths, args.count)

    print(f"[INFO] Sampling {len(selected)} images (seed={args.seed})")
    print(f"[INFO] Preprocessing to {args.imgsz}x{args.imgsz} RGB uint8 "
          f"(stretch, matching HailoDetector.detect)")

    batch = np.empty(
        (len(selected), args.imgsz, args.imgsz, 3), dtype=np.uint8
    )

    written = 0
    skipped: list[str] = []
    for path in selected:
        # cv2.imread returns None rather than raising on a truncated or
        # unreadable file, and Open Images downloads do contain a few.
        # Filesystem/decode boundary: handle it, do not assume.
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            skipped.append(f"{path.name}: {exc}")
            continue

        if image is None or image.size == 0:
            skipped.append(f"{path.name}: unreadable or empty")
            continue

        batch[written] = preprocess(image, args.imgsz)
        written += 1

        if written % 64 == 0:
            print(f"[INFO]   {written}/{len(selected)}")

    if written == 0:
        print("[ERROR] Every sampled image failed to decode. Nothing written.")
        return 1

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} unreadable images:")
        for line in skipped[:10]:
            print(f"[WARN]   {line}")
        if len(skipped) > 10:
            print(f"[WARN]   ... and {len(skipped) - 10} more")

    batch = batch[:written]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, batch)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("")
    print(f"[OK] Wrote {output_path}")
    print(f"[OK]   shape {batch.shape}, dtype {batch.dtype}, {size_mb:.1f} MB")
    print(f"[OK]   value range {int(batch.min())}-{int(batch.max())}")
    print("")
    print("Next, on the x86 machine with the Hailo DFC installed:")
    print("  ./scripts/build_hef.sh")
    return 0


def main() -> int:
    return build(parse_args())


if __name__ == "__main__":
    sys.exit(main())
