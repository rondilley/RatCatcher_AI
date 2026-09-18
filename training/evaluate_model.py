"""
evaluate_model.py
=================
Compares detector checkpoints on the two questions that matter in the
field, and reports them side by side.

**Why this exists.** `docs/MODEL_TRAINING.md` section 5 is blunt about
it: "Do not report a model as improved on Open Images val numbers alone;
that metric is what allowed the current failure." The deployed model
scores mAP@0.5 = 0.751 on its own validation split and is approximately
100 percent false positives on the non-bird classes in the field. mAP
computed over a set in which every image contains an animal cannot
express the failure, because the failure is boxes drawn on empty sky.

So this script reports two numbers, not one:

1. **mAP**, on the labelled portion, for continuity with past runs. It
   is reported and explicitly not used as the gate.
2. **False positives per empty image**, on the background portion --
   images whose correct answer is "nothing here". This is the operative
   metric. The field equivalent is false positives per camera-hour;
   per-image is what a held-out set can measure, and the two move
   together.

Both are computed at `detection.confidence_threshold` from
`config/default.yaml` rather than at the 0.001 that mAP conventionally
uses, because the deployment discards everything below that threshold
and a false positive it never sees is not a false positive.

**What this cannot tell you.** The background images here are generic
garden scenes, not frames from the deployment. They cannot measure
whether the model still fires on that specific cypress or that specific
plant pot. Section 4.3 of the report describes the capture mode needed
to collect those, and until it exists this measurement is a proxy: a
strong improvement here is necessary for a field improvement, and not
sufficient for one.

Usage:
    python training/evaluate_model.py \\
        --models models/ratcatcher_best.onnx runs/.../best.pt \\
        --data datasets/ratcatcher_v2/dataset.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_split(data_yaml: Path) -> tuple:
    """Return (image paths, label paths) for the validation split."""
    with open(data_yaml) as fh:
        config = yaml.safe_load(fh)
    root = Path(config.get("path", data_yaml.parent))
    images_dir = root / config.get("val", "val/images")
    labels_dir = Path(str(images_dir).replace("/images", "/labels"))
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No validation images at {images_dir}")

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    return images, labels_dir


def partition(images: list, labels_dir: Path) -> tuple:
    """Split validation images into labelled and empty (background)."""
    labelled, empty = [], []
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        try:
            content = label.read_text().strip() if label.is_file() else ""
        except OSError:
            content = ""
        (labelled if content else empty).append(image)
    return labelled, empty


def count_false_positives(model, images: list, conf: float, imgsz: int, device: str,
                          batch: int, names: dict) -> tuple:
    """Count detections on images that should produce none.

    Every detection on a background image is a false positive by
    definition: the image contains no animal. Returns the total and a
    per-class breakdown, because "which class does it hallucinate" is
    the question the field audit actually asked.
    """
    from collections import Counter

    total = 0
    per_class = Counter()
    images_with_fp = 0

    for start in range(0, len(images), batch):
        chunk = [str(p) for p in images[start : start + batch]]
        results = model.predict(chunk, conf=conf, imgsz=imgsz, device=device, verbose=False)
        for result in results:
            found = len(result.boxes)
            if found:
                images_with_fp += 1
            total += found
            for box in result.boxes:
                per_class[names.get(int(box.cls[0]), str(int(box.cls[0])))] += 1
    return total, images_with_fp, per_class


def evaluate(model_path: Path, data_yaml: Path, conf: float, imgsz: int,
             device: str, batch: int) -> dict:
    """Run both measurements for one checkpoint."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    # The deployed ONNX was exported with a static batch of 1, so any
    # larger batch is rejected outright by onnxruntime rather than
    # handled. Detect it from the suffix instead of catching the error,
    # because the error arrives inside Ultralytics' warmup.
    if model_path.suffix.lower() == ".onnx":
        batch = 1

    images, labels_dir = load_split(data_yaml)
    labelled, empty = partition(images, labels_dir)

    # mAP over the labelled portion only. Ultralytics computes it over
    # the whole split, and including the background images would not
    # change mAP -- they contribute no ground truth boxes -- but it
    # would make the number harder to compare against past runs.
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device,
                        batch=batch, verbose=False, plots=False, split="val")

    fp_total, fp_images, fp_by_class = (0, 0, {})
    if empty:
        fp_total, fp_images, fp_by_class = count_false_positives(
            model, empty, conf, imgsz, device, batch, names
        )

    return {
        "model": model_path,
        "labelled": len(labelled),
        "empty": len(empty),
        "map50": float(metrics.box.map50),
        "map": float(metrics.box.map),
        "per_class_map50": {
            names.get(c, str(c)): float(metrics.box.maps[i]) if i < len(metrics.box.maps) else 0.0
            for i, c in enumerate(sorted(names))
        },
        "fp_total": fp_total,
        "fp_images": fp_images,
        "fp_by_class": dict(fp_by_class),
    }


def report(results: list, conf: float) -> None:
    print("")
    print("=" * 70)
    print("  Detector comparison")
    print("=" * 70)
    print(f"  Confidence threshold: {conf} (from config/default.yaml)")
    empty = results[0]["empty"]
    print(f"  Empty (background) validation images: {empty}")
    print(f"  Labelled validation images:           {results[0]['labelled']}")

    print("")
    print("  THE GATE -- false positives on images containing no animal")
    print(f"    {'model':<34} {'FP':>7} {'FP/img':>9} {'images hit':>12}")
    print(f"    {'-' * 34} {'-' * 7} {'-' * 9} {'-' * 12}")
    for r in results:
        rate = r["fp_total"] / r["empty"] if r["empty"] else 0.0
        hit = f"{r['fp_images']}/{r['empty']}"
        print(f"    {r['model'].name:<34} {r['fp_total']:>7} {rate:>9.3f} {hit:>12}")

    if len(results) == 2 and results[0]["fp_total"]:
        before, after = results[0]["fp_total"], results[1]["fp_total"]
        if after == 0:
            print(f"\n    Reduction: {before} -> 0 false positives")
        else:
            print(f"\n    Reduction: {before / after:.1f}x ({before} -> {after})")
        print("    The report's bar is a factor of 10 or better.")

    print("")
    print("  FOR CONTINUITY ONLY -- Open Images mAP, explicitly not the gate")
    print(f"    {'model':<34} {'mAP@0.5':>9} {'mAP@0.5:0.95':>14}")
    print(f"    {'-' * 34} {'-' * 9} {'-' * 14}")
    for r in results:
        print(f"    {r['model'].name:<34} {r['map50']:>9.3f} {r['map']:>14.3f}")

    for r in results:
        if r["fp_by_class"]:
            print("")
            print(f"  {r['model'].name}: false positives by class")
            for name, count in sorted(r["fp_by_class"].items(), key=lambda x: -x[1]):
                print(f"    {name:<20} {count:>6}")
    print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare detectors on empty-scene false positives and mAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path,
                        default=Path("datasets/ratcatcher_v2/dataset.yaml"))
    parser.add_argument("--conf", type=float, default=None,
                        help="Defaults to detection.confidence_threshold "
                             "from config/default.yaml.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def deployment_threshold(default: float = 0.45) -> float:
    """Read the confidence the deployment actually uses."""
    config_path = Path("config/default.yaml")
    try:
        with open(config_path) as fh:
            config = yaml.safe_load(fh)
        return float(config["detection"]["confidence_threshold"])
    except (OSError, KeyError, TypeError, ValueError):
        # Falling back is correct here: the number is a reporting
        # parameter, and a missing config should not stop an evaluation.
        print(f"[WARNING] Could not read {config_path}, using conf={default}")
        return default


def main() -> None:
    args = parse_args()
    conf = args.conf if args.conf is not None else deployment_threshold()

    sys.path.insert(0, str(Path(__file__).parent))
    import rocm_compat
    for note in rocm_compat.apply():
        print(note)

    results = []
    for model_path in args.models:
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            sys.exit(1)
        print(f"[INFO] Evaluating {model_path}")
        results.append(evaluate(model_path, args.data, conf, args.imgsz,
                                args.device, args.batch))

    report(results, conf)


if __name__ == "__main__":
    main()
