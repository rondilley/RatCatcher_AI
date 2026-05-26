"""
download_data.py
================
Downloads animal-detection training data from Open Images V7 and prepares it
in YOLO format for the RatCatcher AI project.

Usage:
    python training/download_data.py --output datasets/ratcatcher --max-per-class 3000

Dependencies (pip install):
    requests, pandas, tqdm
"""

import argparse
import csv
import logging
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Target classes
# ---------------------------------------------------------------------------

TARGET_CLASSES = {
    "Bird":     {"mid": "/m/015p6",  "class_id": 0, "name": "bird"},
    "Squirrel": {"mid": "/m/071qp",  "class_id": 1, "name": "squirrel"},
    "Mouse":    {"mid": "/m/04rmv",  "class_id": 2, "name": "rat"},
    "Cat":      {"mid": "/m/01yrx",  "class_id": 3, "name": "cat"},
    "Raccoon":  {"mid": "/m/0dq75",  "class_id": 4, "name": "unknown_animal"},
    "Rabbit":   {"mid": "/m/06mf6",  "class_id": 4, "name": "unknown_animal"},
    "Skunk":    {"mid": "/m/0jbk",   "class_id": 4, "name": "unknown_animal"},
}

CLASS_NAMES = {
    0: "bird",
    1: "squirrel",
    2: "rat",
    3: "cat",
    4: "unknown_animal",
}

# Reverse lookup: MID -> (class_id, name)
MID_TO_CLASS = {}
for _label, _info in TARGET_CLASSES.items():
    MID_TO_CLASS[_info["mid"]] = (_info["class_id"], _info["name"])

TARGET_MIDS = set(MID_TO_CLASS.keys())

# ---------------------------------------------------------------------------
# Annotation CSV URLs
# ---------------------------------------------------------------------------

ANNOTATION_URLS = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
}

# ---------------------------------------------------------------------------
# Image base URL
# ---------------------------------------------------------------------------

IMAGE_URL_TEMPLATE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_data")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def download_file(url: str, dest: Path, description: str = "") -> None:
    """Download a file with streaming progress bar.  Skips if *dest* exists."""
    if dest.exists():
        log.info("Cached: %s", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    log.info("Downloading %s -> %s", url, dest)
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Failed to start download for %s: %s", url, exc)
        raise

    total = int(resp.headers.get("Content-Length", 0))
    bar_desc = description or dest.name
    try:
        with open(tmp, "wb") as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=bar_desc,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                fh.write(chunk)
                pbar.update(len(chunk))
    except (IOError, requests.RequestException) as exc:
        log.error("Download interrupted for %s: %s", url, exc)
        if tmp.exists():
            tmp.unlink()
        raise

    tmp.rename(dest)
    log.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


def download_image(
    image_id: str,
    split: str,
    dest_dir: Path,
    max_retries: int = 1,
) -> bool:
    """Download a single image.  Returns True on success."""
    dest = dest_dir / f"{image_id}.jpg"
    if dest.exists():
        return True

    url = IMAGE_URL_TEMPLATE.format(split=split, image_id=image_id)
    for attempt in range(1 + max_retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                fh.write(resp.content)
            return True
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(1)
            else:
                log.warning("Skipping %s after %d attempts: %s", image_id, attempt + 1, exc)
                return False
    return False


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def download_annotations(cache_dir: Path, train_only: bool) -> dict:
    """Download annotation CSVs and return {split: Path} mapping."""
    paths = {}
    for split, url in ANNOTATION_URLS.items():
        if train_only and split != "train":
            continue
        filename = url.rsplit("/", 1)[-1]
        dest = cache_dir / filename
        download_file(url, dest, description=f"{split} annotations")
        paths[split] = dest
    return paths


def filter_annotations(
    csv_path: Path,
    split: str,
    max_per_class: int,
    class_counts: dict,
) -> pd.DataFrame:
    """Read an annotation CSV, filter to target classes, respect per-class caps.

    *class_counts* is a dict of {class_id: set(image_ids)} that is updated
    in-place so that caps are shared across splits and across multiple OI
    labels that map to the same class_id.

    Returns a DataFrame of kept annotation rows.
    """
    log.info("Filtering %s (%s) ...", csv_path.name, split)

    kept_chunks = []

    # The train CSV is very large (~2.2 GB).  Read in chunks.
    reader = pd.read_csv(csv_path, chunksize=500_000)
    total_rows = 0
    kept_rows = 0

    for chunk in reader:
        total_rows += len(chunk)

        # Filter to target labels
        filtered = chunk[chunk["LabelName"].isin(TARGET_MIDS)].copy()
        if filtered.empty:
            continue

        # Quality filters
        for col in ("IsGroupOf", "IsDepiction", "IsInside"):
            if col in filtered.columns:
                filtered = filtered[filtered[col] == 0]
        if filtered.empty:
            continue

        # Add our class_id column
        filtered["class_id"] = filtered["LabelName"].map(
            lambda mid: MID_TO_CLASS[mid][0]
        )
        filtered["split"] = split

        # Enforce per-class cap at the image level
        rows_to_keep = []
        for _, row in filtered.iterrows():
            cid = row["class_id"]
            img_id = row["ImageID"]
            img_set = class_counts.setdefault(cid, set())
            if img_id in img_set:
                # Image already accepted for this class -- keep all its boxes
                rows_to_keep.append(row)
            elif len(img_set) < max_per_class:
                img_set.add(img_id)
                rows_to_keep.append(row)
            # else: cap reached, skip

        if rows_to_keep:
            kept = pd.DataFrame(rows_to_keep)
            kept_chunks.append(kept)
            kept_rows += len(kept)

    log.info(
        "  %s: scanned %d rows, kept %d annotations",
        split,
        total_rows,
        kept_rows,
    )

    if not kept_chunks:
        return pd.DataFrame()

    return pd.concat(kept_chunks, ignore_index=True)


def print_summary_table(annotations: pd.DataFrame) -> None:
    """Print a table showing image counts per class."""
    print("\n--- Annotation Summary ---")
    print(f"{'Class ID':<10} {'Name':<18} {'Images':<10} {'Boxes':<10}")
    print("-" * 50)
    for cid in sorted(CLASS_NAMES.keys()):
        subset = annotations[annotations["class_id"] == cid]
        n_images = subset["ImageID"].nunique() if not subset.empty else 0
        n_boxes = len(subset)
        print(f"{cid:<10} {CLASS_NAMES[cid]:<18} {n_images:<10} {n_boxes:<10}")
    total_images = annotations["ImageID"].nunique()
    total_boxes = len(annotations)
    print("-" * 50)
    print(f"{'TOTAL':<10} {'':<18} {total_images:<10} {total_boxes:<10}")
    print()


def download_images(
    annotations: pd.DataFrame,
    output_dir: Path,
    workers: int,
) -> int:
    """Download all images referenced in *annotations*.  Returns count of
    successfully downloaded images."""
    # Build unique (image_id, split) pairs
    pairs = (
        annotations[["ImageID", "split"]]
        .drop_duplicates()
        .values.tolist()
    )

    images_dir = output_dir / "all_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %d unique images with %d workers ...", len(pairs), workers)

    success = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_image, img_id, split, images_dir): img_id
            for img_id, split in pairs
        }
        with tqdm(total=len(futures), desc="Images", unit="img") as pbar:
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    ok = future.result()
                except Exception as exc:
                    log.warning("Unexpected error downloading %s: %s", img_id, exc)
                    ok = False
                if ok:
                    success += 1
                else:
                    fail += 1
                pbar.update(1)

    log.info("Images downloaded: %d success, %d failed", success, fail)
    return success


def write_yolo_labels(annotations: pd.DataFrame, labels_dir: Path) -> int:
    """Write YOLO-format .txt label files.  Returns count of files written."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    grouped = annotations.groupby("ImageID")
    count = 0
    for image_id, group in grouped:
        lines = []
        for _, row in group.iterrows():
            cid = int(row["class_id"])
            xmin = float(row["XMin"])
            xmax = float(row["XMax"])
            ymin = float(row["YMin"])
            ymax = float(row["YMax"])
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
            lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        label_path = labels_dir / f"{image_id}.txt"
        try:
            with open(label_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            count += 1
        except IOError as exc:
            log.warning("Failed to write label %s: %s", label_path, exc)
    return count


def split_dataset(
    annotations: pd.DataFrame,
    output_dir: Path,
    val_split: float,
) -> None:
    """Split images into train/ and val/ subdirectories and write labels."""
    all_images_dir = output_dir / "all_images"
    all_labels_dir = output_dir / "all_labels"

    # Write all labels first
    log.info("Writing YOLO label files ...")
    n_labels = write_yolo_labels(annotations, all_labels_dir)
    log.info("Wrote %d label files", n_labels)

    # Collect image IDs that actually have a downloaded image
    available_ids = []
    for image_id in annotations["ImageID"].unique():
        img_path = all_images_dir / f"{image_id}.jpg"
        if img_path.exists():
            available_ids.append(image_id)

    log.info(
        "Splitting %d available images (%.0f%% val) ...",
        len(available_ids),
        val_split * 100,
    )

    random.seed(42)
    random.shuffle(available_ids)

    val_count = max(1, int(len(available_ids) * val_split))
    val_ids = set(available_ids[:val_count])
    train_ids = set(available_ids[val_count:])

    # Create directory structure
    for subset in ("train", "val"):
        (output_dir / subset / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / subset / "labels").mkdir(parents=True, exist_ok=True)

    # Move files
    moved_train = 0
    moved_val = 0

    for image_id in tqdm(available_ids, desc="Organizing files", unit="img"):
        subset = "val" if image_id in val_ids else "train"
        src_img = all_images_dir / f"{image_id}.jpg"
        src_lbl = all_labels_dir / f"{image_id}.txt"
        dst_img = output_dir / subset / "images" / f"{image_id}.jpg"
        dst_lbl = output_dir / subset / "labels" / f"{image_id}.txt"

        try:
            if src_img.exists() and not dst_img.exists():
                shutil.move(str(src_img), str(dst_img))
            elif src_img.exists():
                src_img.unlink()
        except IOError as exc:
            log.warning("Failed to move image %s: %s", image_id, exc)
            continue

        try:
            if src_lbl.exists() and not dst_lbl.exists():
                shutil.move(str(src_lbl), str(dst_lbl))
            elif src_lbl.exists():
                src_lbl.unlink()
        except IOError as exc:
            log.warning("Failed to move label %s: %s", image_id, exc)

        if subset == "train":
            moved_train += 1
        else:
            moved_val += 1

    # Clean up temporary directories
    for tmp_dir in (all_images_dir, all_labels_dir):
        try:
            if tmp_dir.exists():
                shutil.rmtree(str(tmp_dir))
        except IOError:
            pass

    log.info("Train: %d images, Val: %d images", moved_train, moved_val)


def write_dataset_yaml(output_dir: Path) -> None:
    """Write dataset.yaml for YOLO training."""
    yaml_path = output_dir / "dataset.yaml"
    abs_path = str(output_dir.resolve()).replace("\\", "/")

    lines = [
        f"path: {abs_path}",
        "train: train/images",
        "val: val/images",
        "",
        "names:",
    ]
    for cid in sorted(CLASS_NAMES.keys()):
        lines.append(f"  {cid}: {CLASS_NAMES[cid]}")
    lines.append("")

    try:
        with open(yaml_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        log.info("Wrote %s", yaml_path)
    except IOError as exc:
        log.error("Failed to write dataset.yaml: %s", exc)
        raise


def print_final_summary(output_dir: Path) -> None:
    """Print a final count of files in the dataset."""
    print("\n--- Final Dataset Summary ---")
    for subset in ("train", "val"):
        img_dir = output_dir / subset / "images"
        lbl_dir = output_dir / subset / "labels"
        n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {subset}: {n_img} images, {n_lbl} labels")
    yaml_path = output_dir / "dataset.yaml"
    if yaml_path.exists():
        print(f"  dataset.yaml: {yaml_path.resolve()}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Open Images V7 animal data and prepare YOLO dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/ratcatcher",
        help="Output directory for the YOLO dataset (default: datasets/ratcatcher)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=3000,
        help="Maximum number of images per class_id (default: 3000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download threads (default: 8)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of images to use for validation (default: 0.15)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="training/cache",
        help="Directory to cache downloaded annotation CSVs (default: training/cache)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip image downloading; only regenerate labels from cached annotations",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only download train split (skip Open Images validation split)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("Output directory : %s", output_dir)
    log.info("Cache directory  : %s", cache_dir)
    log.info("Max per class    : %d", args.max_per_class)
    log.info("Val split        : %.0f%%", args.val_split * 100)
    log.info("Workers          : %d", args.workers)
    if args.skip_download:
        log.info("Mode             : skip-download (labels only)")
    if args.train_only:
        log.info("Splits           : train only")

    # ------------------------------------------------------------------
    # Step 1: Download annotation CSVs
    # ------------------------------------------------------------------
    log.info("=== Step 1: Download annotation CSVs ===")
    csv_paths = download_annotations(cache_dir, args.train_only)

    if not csv_paths:
        log.error("No annotation CSVs available. Nothing to do.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Filter annotations
    # ------------------------------------------------------------------
    log.info("=== Step 2: Filter annotations ===")

    # class_counts tracks {class_id: set(image_ids)} across all splits
    # so that the per-class cap is global.
    class_counts = {}
    all_annotations = []

    for split, csv_path in csv_paths.items():
        df = filter_annotations(csv_path, split, args.max_per_class, class_counts)
        if not df.empty:
            all_annotations.append(df)

    if not all_annotations:
        log.error("No annotations matched the target classes. Nothing to do.")
        sys.exit(1)

    annotations = pd.concat(all_annotations, ignore_index=True)
    print_summary_table(annotations)

    # Save filtered annotations for potential re-use
    filtered_cache = cache_dir / "filtered_annotations.csv"
    try:
        annotations.to_csv(filtered_cache, index=False)
        log.info("Cached filtered annotations: %s", filtered_cache)
    except IOError as exc:
        log.warning("Could not cache filtered annotations: %s", exc)

    # ------------------------------------------------------------------
    # Step 3: Download images
    # ------------------------------------------------------------------
    if not args.skip_download:
        log.info("=== Step 3: Download images ===")
        download_images(annotations, output_dir, args.workers)
    else:
        log.info("=== Step 3: Skipped (--skip-download) ===")

    # ------------------------------------------------------------------
    # Step 4: Split train/val, write labels
    # ------------------------------------------------------------------
    log.info("=== Step 4: Split dataset and write YOLO labels ===")
    split_dataset(annotations, output_dir, args.val_split)

    # ------------------------------------------------------------------
    # Step 5: Write dataset.yaml
    # ------------------------------------------------------------------
    log.info("=== Step 5: Write dataset.yaml ===")
    write_dataset_yaml(output_dir)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print_final_summary(output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
