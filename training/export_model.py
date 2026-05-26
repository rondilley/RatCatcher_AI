#!/usr/bin/env python3
"""RatCatcher AI -- Model Export Script

Converts trained YOLOv8n weights to deployment formats for the Raspberry Pi 5:
  - ONNX  (primary, for OpenCV DNN backend)
  - NCNN  (CPU fallback, ARM-optimized)

Optionally validates exports with synthetic inference and prints Hailo HEF
conversion instructions (the Hailo Dataflow Compiler is a separate install
that cannot be done via pip).

Usage:
    python training/export_model.py --weights runs/train/ratcatcher/weights/best.pt
    python training/export_model.py --weights runs/train/ratcatcher/weights/best.pt --output models/ --formats onnx,ncnn
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# Supported export formats and their expected output file extensions
FORMAT_EXTENSIONS = {
    "onnx": ".onnx",
    "ncnn": "_ncnn_model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trained YOLOv8n weights to deployment formats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained YOLOv8n .pt weights file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/",
        help="Output directory for exported model files.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="onnx,ncnn",
        help="Comma-separated list of export formats (onnx, ncnn).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for export.",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a quick inference test on each exported model.",
    )
    return parser.parse_args()


def validate_weights(weights_path: str) -> Path:
    """Ensure the .pt weights file exists."""
    path = Path(weights_path)
    if not path.is_file():
        print(f"[ERROR] Weights file not found: {path.resolve()}")
        print("[ERROR] Train a model first with training/train_detector.py")
        sys.exit(1)
    if path.suffix != ".pt":
        print(f"[ERROR] Expected a .pt file, got: {path.name}")
        sys.exit(1)
    return path


def format_file_size(path: Path) -> str:
    """Return a human-readable file size string."""
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    elif path.is_file():
        total = path.stat().st_size
    else:
        return "unknown"

    if total < 1024:
        return f"{total} B"
    if total < 1024 * 1024:
        return f"{total / 1024:.1f} KB"
    return f"{total / (1024 * 1024):.1f} MB"


def copy_exported_file(source: Path, output_dir: Path, fmt: str) -> Path:
    """Copy the exported model to the output directory with ratcatcher_ prefix.

    For single-file formats (ONNX), copies the file directly.
    For directory formats (NCNN), copies the entire directory.

    Returns the destination path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        dest = output_dir / f"ratcatcher_{fmt}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
    else:
        dest = output_dir / f"ratcatcher_{source.stem}{source.suffix}"
        shutil.copy2(source, dest)

    return dest


def find_exported_file(weights_path: Path, fmt: str) -> Path:
    """Locate the exported file produced by Ultralytics model.export().

    Ultralytics places exports next to the source .pt file.  For single-file
    formats the export path is returned directly by model.export().  For
    directory formats like NCNN, the directory is created next to the .pt.
    """
    parent = weights_path.parent
    stem = weights_path.stem

    ext = FORMAT_EXTENSIONS.get(fmt, f".{fmt}")

    candidate = parent / f"{stem}{ext}"
    if candidate.exists():
        return candidate

    # Ultralytics sometimes returns the path as a string from export();
    # we also check for a common variant without the stem prefix
    candidate_alt = parent / f"{stem}_{fmt}_model"
    if candidate_alt.exists():
        return candidate_alt

    # Last resort: glob for anything matching
    matches = list(parent.glob(f"*{fmt}*"))
    if matches:
        return matches[0]

    return candidate  # return expected path even if missing, caller will report


def export_format(model: YOLO, fmt: str, imgsz: int, weights_path: Path) -> Path:
    """Export the model to one format and return the path to the exported file."""
    print(f"[INFO] Exporting to {fmt.upper()}...")
    start = time.time()

    export_kwargs = {
        "format": fmt,
        "imgsz": imgsz,
    }

    # Format-specific options
    if fmt == "onnx":
        export_kwargs["simplify"] = True
        export_kwargs["opset"] = 17
    elif fmt == "ncnn":
        export_kwargs["half"] = False  # NCNN on RPi CPU uses FP32

    export_result = model.export(**export_kwargs)

    elapsed = time.time() - start
    print(f"[INFO] Export to {fmt.upper()} finished in {elapsed:.1f}s")

    # model.export() returns the path to the exported file as a string
    if export_result and Path(str(export_result)).exists():
        return Path(str(export_result))

    # Fallback: find it ourselves
    return find_exported_file(weights_path, fmt)


def run_validation(model_path: Path, fmt: str, imgsz: int) -> None:
    """Run inference on a synthetic image to verify the exported model works."""
    print(f"[INFO] Validating {fmt.upper()} export with synthetic inference...")

    # Create a synthetic test image (random noise, 3-channel uint8)
    test_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    try:
        # Load the exported model for inference
        if model_path.is_dir():
            # For NCNN, the directory contains the model files
            val_model = YOLO(str(model_path))
        else:
            val_model = YOLO(str(model_path))

        results = val_model.predict(test_image, imgsz=imgsz, verbose=False)

        det_count = 0
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                det_count = len(boxes)

        print(f"[INFO] Validation OK: {fmt.upper()} model loaded and ran inference "
              f"({det_count} detections on random noise, expected ~0)")

    except Exception as exc:
        print(f"[WARNING] Validation failed for {fmt.upper()}: {exc}")
        print(f"[WARNING] The exported file may still be valid -- test manually.")


def print_deployment_instructions(output_dir: Path, exported_files: dict) -> None:
    """Print instructions for deploying models to the Raspberry Pi."""
    print("")
    print("=" * 64)
    print("  RatCatcher AI -- Deployment Instructions")
    print("=" * 64)
    print("")
    print("  Exported models:")
    for fmt, path in exported_files.items():
        size = format_file_size(path)
        print(f"    {fmt.upper():<8} {path.resolve()}  ({size})")
    print("")

    print("  1. Copy models to the Raspberry Pi:")
    print(f"     scp -r {output_dir.resolve()}/* pi@ratcatcher:~/ratcatcher/models/")
    print("")

    print("  2. Update config/default.yaml on the RPi:")
    if "onnx" in exported_files:
        onnx_name = exported_files["onnx"].name
        print(f"     For OpenCV DNN backend:")
        print(f'       model_path: "{onnx_name}"')
    if "ncnn" in exported_files:
        ncnn_name = exported_files["ncnn"].name
        print(f"     For NCNN backend:")
        print(f'       model_path: "{ncnn_name}"')
    print("")

    print("  3. Hailo HEF conversion (for Hailo-8L NPU):")
    print("     The Hailo Dataflow Compiler (DFC) cannot be installed via pip.")
    print("     It requires the full Hailo AI Software Suite.")
    print("")
    print("     On a machine with the Hailo SDK installed:")
    print("")
    if "onnx" in exported_files:
        onnx_path = exported_files["onnx"].resolve()
        print(f"     # Parse the ONNX model")
        print(f"     hailo parser onnx {onnx_path} \\")
        print(f"       --hw-arch hailo8l \\")
        print(f"       --end-node-names /model.22/cv2.0/cv2.0.2/Conv \\")
        print(f"                        /model.22/cv3.0/cv3.0.2/Conv \\")
        print(f"                        /model.22/cv2.1/cv2.1.2/Conv \\")
        print(f"                        /model.22/cv3.1/cv3.1.2/Conv \\")
        print(f"                        /model.22/cv2.2/cv2.2.2/Conv \\")
        print(f"                        /model.22/cv3.2/cv3.2.2/Conv")
        print("")
        print(f"     # Optimize (quantize to INT8 -- needs calibration images)")
        print(f"     hailo optimize ratcatcher_best.har \\")
        print(f"       --hw-arch hailo8l \\")
        print(f"       --calib-set-path /path/to/calibration/images/")
        print("")
        print(f"     # Compile to HEF")
        print(f"     hailo compiler ratcatcher_best_quantized.har \\")
        print(f"       --hw-arch hailo8l \\")
        print(f"       -o models/ratcatcher_best.hef")
        print("")
        print(f"     Then copy ratcatcher_best.hef to the RPi and set:")
        print(f'       model_path: "ratcatcher_best.hef"')
        print(f"       backend: \"hailo\"")
    else:
        print("     Export to ONNX first (needed as input for Hailo DFC).")
    print("")
    print("=" * 64)


def main() -> None:
    args = parse_args()

    print("=" * 64)
    print("  RatCatcher AI -- Model Export")
    print("=" * 64)
    print("")

    # Validate inputs
    weights_path = validate_weights(args.weights)
    print(f"[INFO] Weights      : {weights_path.resolve()}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output dir   : {output_dir.resolve()}")

    # Parse requested formats
    requested_formats = [f.strip().lower() for f in args.formats.split(",")]
    unsupported = [f for f in requested_formats if f not in FORMAT_EXTENSIONS]
    if unsupported:
        print(f"[ERROR] Unsupported formats: {', '.join(unsupported)}")
        print(f"[ERROR] Supported formats: {', '.join(FORMAT_EXTENSIONS.keys())}")
        sys.exit(1)

    print(f"[INFO] Formats      : {', '.join(f.upper() for f in requested_formats)}")
    print(f"[INFO] Image size   : {args.imgsz}")
    print(f"[INFO] Validate     : {args.validate}")
    print("")

    # Load the trained model
    print(f"[INFO] Loading model from {weights_path.name}...")
    model = YOLO(str(weights_path))
    print("[INFO] Model loaded successfully.")
    print("")

    # Export each format
    exported_files = {}
    for fmt in requested_formats:
        print("-" * 40)
        exported_path = export_format(model, fmt, args.imgsz, weights_path)

        if not exported_path.exists():
            print(f"[ERROR] Export file not found: {exported_path}")
            print(f"[ERROR] {fmt.upper()} export may have failed. Check output above.")
            continue

        # Copy to output directory with ratcatcher_ prefix
        dest = copy_exported_file(exported_path, output_dir, fmt)
        exported_files[fmt] = dest

        size = format_file_size(dest)
        print(f"[INFO] Copied to: {dest.resolve()} ({size})")
        print("")

    # Validate exports
    if args.validate and exported_files:
        print("-" * 40)
        print("[INFO] Running validation inference...")
        print("")
        for fmt, path in exported_files.items():
            run_validation(path, fmt, args.imgsz)
        print("")

    # Print deployment instructions
    if exported_files:
        print_deployment_instructions(output_dir, exported_files)
    else:
        print("[ERROR] No models were exported successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
