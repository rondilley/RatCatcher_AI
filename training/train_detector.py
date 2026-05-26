#!/usr/bin/env python3
"""RatCatcher AI -- YOLOv8n Training Script

Fine-tunes a COCO-pretrained YOLOv8n model on the RatCatcher custom dataset
(bird, squirrel, rat, cat, unknown_animal) using the Ultralytics training API.

The dataset must already be prepared in YOLO format with a dataset.yaml file
(see training/download_data.py or prepare it manually).

Usage:
    python training/train_detector.py --data datasets/ratcatcher/dataset.yaml --epochs 100
    python training/train_detector.py --data datasets/ratcatcher/dataset.yaml --resume
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n on the RatCatcher feeder-wildlife dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset.yaml describing the YOLO-format dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (reduce if running out of GPU memory).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Compute device: "auto" to detect CUDA/CPU, or specify "0", "cpu", etc.',
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Starting weights. Ultralytics downloads COCO-pretrained yolov8n.pt automatically.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Parent directory for training output.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ratcatcher",
        help="Run name (subdirectory under --project).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint in --project/--name.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs with no mAP improvement).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers.",
    )
    return parser.parse_args()


def detect_device(requested: str) -> str:
    """Resolve the compute device.

    Returns the device string that Ultralytics model.train() expects:
    a CUDA device index like "0", or "cpu".
    """
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"[INFO] CUDA GPU detected: {gpu_name} ({gpu_mem_gb:.1f} GB)")
        return "0"

    print("[WARNING] No CUDA GPU detected -- training will run on CPU.")
    print("[WARNING] CPU training is extremely slow. Consider using a machine with a GPU.")
    return "cpu"


def validate_data_path(data_path: str) -> Path:
    """Ensure the dataset.yaml file exists and is readable."""
    path = Path(data_path)
    if not path.is_file():
        print(f"[ERROR] Dataset config not found: {path.resolve()}")
        print("[ERROR] Prepare the dataset first (see training/download_data.py).")
        sys.exit(1)
    if path.suffix not in (".yaml", ".yml"):
        print(f"[ERROR] Expected a .yaml file, got: {path.name}")
        sys.exit(1)
    return path


def print_results_summary(results, best_weights: Path) -> None:
    """Print final training metrics in a readable table."""
    print("")
    print("=" * 60)
    print("  RatCatcher AI -- Training Complete")
    print("=" * 60)

    # The results object from model.train() stores metrics in results.results_dict
    metrics = getattr(results, "results_dict", {})

    # Build a table of the most important metrics
    metric_labels = [
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP@0.5"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Cls Loss"),
        ("train/dfl_loss", "Train DFL Loss"),
        ("val/box_loss", "Val Box Loss"),
        ("val/cls_loss", "Val Cls Loss"),
        ("val/dfl_loss", "Val DFL Loss"),
    ]

    print("")
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-' * 25} {'-' * 10}")
    for key, label in metric_labels:
        value = metrics.get(key)
        if value is not None:
            print(f"  {label:<25} {value:>10.4f}")

    print("")
    print(f"  Best weights: {best_weights.resolve()}")
    print("")
    print("  Next steps:")
    print("    1. Evaluate:  yolo val model={} data=<dataset.yaml>".format(best_weights))
    print("    2. Export:    python training/export_model.py --weights {}".format(best_weights))
    print("=" * 60)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  RatCatcher AI -- YOLOv8n Training")
    print("=" * 60)
    print("")

    # Validate dataset path
    data_path = validate_data_path(args.data)
    print(f"[INFO] Dataset config : {data_path.resolve()}")
    print(f"[INFO] Starting weights: {args.weights}")
    print(f"[INFO] Epochs          : {args.epochs}")
    print(f"[INFO] Batch size      : {args.batch}")
    print(f"[INFO] Image size      : {args.imgsz}")
    print(f"[INFO] Patience        : {args.patience}")
    print(f"[INFO] Workers         : {args.workers}")
    print(f"[INFO] Output          : {args.project}/{args.name}")
    print("")

    # Detect device
    device = detect_device(args.device)
    print(f"[INFO] Device          : {device}")
    print("")

    # Load model -- either fresh or resume from checkpoint
    if args.resume:
        last_weights = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_weights.is_file():
            print(f"[ERROR] Cannot resume: {last_weights.resolve()} not found.")
            print("[ERROR] Run a fresh training first, then use --resume.")
            sys.exit(1)
        print(f"[INFO] Resuming from checkpoint: {last_weights.resolve()}")
        model = YOLO(str(last_weights))
    else:
        print(f"[INFO] Loading pretrained weights: {args.weights}")
        print("[INFO] (Ultralytics will download COCO weights automatically if needed)")
        model = YOLO(args.weights)

    print("")
    print("[INFO] Starting training...")
    start_time = time.time()

    # Run training
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        save=True,
        save_period=-1,  # save best and last only
        exist_ok=True,
        pretrained=True,
        resume=args.resume,
        # Augmentation defaults from Ultralytics are good for detection
        verbose=True,
    )

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60.0
    print(f"[INFO] Training finished in {elapsed_min:.1f} minutes.")

    # Locate best weights
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_weights.is_file():
        # Fallback: Ultralytics may use a slightly different path
        save_dir = getattr(results, "save_dir", None)
        if save_dir is not None:
            candidate = Path(save_dir) / "weights" / "best.pt"
            if candidate.is_file():
                best_weights = candidate

    if not best_weights.is_file():
        print("[WARNING] best.pt not found at expected location.")
        print("[WARNING] Check the training output directory for weights.")
    else:
        print_results_summary(results, best_weights)


if __name__ == "__main__":
    main()
