"""
download_backgrounds.py
=======================
Downloads background (negative) images from Open Images V7 for the
RatCatcher AI detector.

A background image is one that contains no animal and therefore carries
an empty YOLO label file. `docs/MODEL_TRAINING.md` section 2.1 measured
zero of them in the committed dataset:

    train: 9730 label files, 0 EMPTY
    val:   1717 label files, 0 EMPTY

Every training image contained an animal, so the model was never shown a
scene whose correct answer is "nothing here". In the field it duly put a
box on whatever most resembled an animal, which over 4073 stored
detections meant clouds, a hanging plant pot, a shepherd's hook and a
cypress tree. That report calls this "the single largest contributor to
the cloud detections".

**What this script can and cannot do.** The negatives that would help
most are frames from the deployment itself -- that exact cypress, that
exact plant pot -- and those are not collectable yet, because the Pi
stores no suitable image (report section 4.3). This script covers
section 4.4 instead: generic scene imagery that teaches "nothing here"
in general rather than for one yard. Expect it to reduce false positives
substantially without eliminating the deployment-specific ones.

**Why images are chosen by class rather than at random.** A random
animal-free photograph is usually an indoor object shot and teaches
little about a garden. Selection prefers images annotated with outdoor
scene furniture -- trees, plants, flowerpots, buildings, fences, garden
seating -- because those are what the deployment actually mistakes for
animals. Note that Open Images has no boxable Sky or Cloud class, so
open sky cannot be selected for directly; it arrives incidentally in
images of trees and buildings.

**Why exclusion uses the class hierarchy.** Excluding only the five
target classes would leave dogs, horses and insects in the negatives,
teaching the detector that those are background. Every one of the 111
classes under Open Images' `Animal` node (/m/0jbk) is excluded instead,
so a background image contains no animal of any kind.

Usage:
    python training/download_backgrounds.py --output datasets/backgrounds \\
        --train 1500 --val 260

Dependencies: requests, tqdm
"""

import argparse
import csv
import json
import logging
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Open Images metadata
# ---------------------------------------------------------------------------

HIERARCHY_URL = (
    "https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json"
)
CLASS_DESC_URL = (
    "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
)
ANNOTATION_URLS = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
}
IMAGE_URL_TEMPLATE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

# Root of the animal subtree. Everything under it is excluded from the
# negatives; see the module docstring for why this is not just the five
# target classes.
ANIMAL_ROOT_MID = "/m/0jbk"

# Vegetation and garden furniture. An image must carry at least one of
# these to be considered: they are what the deployment actually mistakes
# for animals -- the cypress, the hanging plant pot, the shepherd's hook.
#
# The first version of this list also held Building, House, Window, Door,
# Chair and Table on the theory that they meant "outdoor scene". A
# contact sheet of the result showed what they really select for:
# conference halls, living rooms, museum interiors, street portraits.
# Those are animal-free, so they were valid negatives, but they teach
# "nothing here" in a domain a garden camera never sees. Requiring
# vegetation is what keeps the negatives in the domain that matters.
VEGETATION_MIDS = {
    "/m/07j7r": "Tree",
    "/m/0cdl1": "Palm tree",
    "/m/05s2s": "Plant",
    "/m/03fp41": "Houseplant",
    "/m/0fm3zh": "Flowerpot",
    "/m/0c9ph5": "Flower",
    "/m/0cvnqh": "Bench",
    "/m/0220r2": "Fountain",
}

# Indoor markers. Vegetation alone is not enough -- a houseplant beside a
# sofa is still a living room, and Open Images labels a great many of
# those. An image carrying any of these is rejected however much
# vegetation it also holds.
INDOOR_MIDS = {
    "/m/02crq1",  # Couch
    "/m/03ssj5",  # Bed
    "/m/07c52",  # Television
    "/m/01y9k5",  # Desk
    "/m/04bcr3",  # Table
    "/m/01mzpv",  # Chair
    "/m/078n6m",  # Coffee table
    "/m/0h8n5zk",  # Kitchen & dining room table
    "/m/0642b4",  # Cupboard
    "/m/03__z0",  # Bookcase
    "/m/040b_t",  # Refrigerator
    "/m/029bxz",  # Oven
    "/m/0130jx",  # Sink
    "/m/09g1w",  # Toilet
    "/m/03dnzn",  # Bathtub
    "/m/03rszm",  # Curtain
    "/m/034c16",  # Pillow
    "/m/0gjbg72",  # Shelf
    "/m/0b3fp9",  # Countertop
    "/m/02522",  # Computer monitor
    "/m/01c648",  # Laptop
    "/m/09tvcd",  # Wine glass
    "/m/0fx9l",  # Microwave oven
    "/m/02vkqh8",  # Wardrobe
    "/m/02z51p",  # Nightstand
    "/m/03m3pdh",  # Sofa bed
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_backgrounds")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_to_cache(url: str, dest: Path, description: str = "") -> Path:
    """Download *url* to *dest* unless it is already cached."""
    if dest.exists():
        log.info("Cached: %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    log.info("Downloading %s", description or dest.name)
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=description or dest.name
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                fh.write(chunk)
                pbar.update(len(chunk))
    except (IOError, requests.RequestException) as exc:
        log.error("Download failed for %s: %s", url, exc)
        if tmp.exists():
            tmp.unlink()
        raise
    tmp.rename(dest)
    return dest


def load_animal_mids(cache_dir: Path) -> set:
    """Return every MID under the Open Images `Animal` node."""
    dest = cache_dir / "bbox_labels_600_hierarchy.json"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(HIERARCHY_URL, dest)
        except (urllib.error.URLError, OSError) as exc:
            log.error("Could not fetch the class hierarchy: %s", exc)
            raise

    with open(dest) as fh:
        hierarchy = json.load(fh)

    def find(node, target):
        if node.get("LabelName") == target:
            return node
        for child in node.get("Subcategory", []):
            found = find(child, target)
            if found:
                return found
        return None

    def collect(node, out):
        out.add(node["LabelName"])
        for child in node.get("Subcategory", []):
            collect(child, out)

    animal = find(hierarchy, ANIMAL_ROOT_MID)
    if animal is None:
        raise RuntimeError(
            f"{ANIMAL_ROOT_MID} not found in the Open Images hierarchy. "
            "The published hierarchy may have changed shape."
        )
    mids = set()
    collect(animal, mids)
    return mids


def select_background_ids(
    annotations_csv: Path, animal_mids: set, wanted: int
) -> list:
    """Return image ids holding vegetation, no animal and no indoor marker.

    One streaming pass over the annotation CSV, which is 2.2 GB for the
    train split. Per image it keeps three bits -- saw an animal, saw
    vegetation, saw an indoor marker -- rather than the annotation rows,
    so peak memory stays proportional to the image count and not to the
    file size.
    """
    has_animal = set()
    has_scene = set()
    has_indoor = set()

    log.info("Scanning %s (this takes a minute)", annotations_csv.name)
    with open(annotations_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        try:
            id_col = header.index("ImageID")
            label_col = header.index("LabelName")
        except ValueError as exc:
            raise RuntimeError(
                f"Unexpected annotation CSV header in {annotations_csv}: {exc}"
            ) from exc

        for row in tqdm(reader, unit=" rows", unit_scale=True, desc="annotations"):
            label = row[label_col]
            if label in animal_mids:
                has_animal.add(row[id_col])
            elif label in VEGETATION_MIDS:
                has_scene.add(row[id_col])
            elif label in INDOOR_MIDS:
                has_indoor.add(row[id_col])

    candidates = sorted(has_scene - has_animal - has_indoor)
    log.info(
        "%d images with vegetation; %d after removing animals and interiors",
        len(has_scene),
        len(candidates),
    )
    if len(candidates) < wanted:
        log.warning(
            "Only %d candidates available, %d requested", len(candidates), wanted
        )
    return candidates[:wanted]


def download_image(image_id: str, split: str, dest_dir: Path) -> bool:
    """Download one image. Returns True on success."""
    dest = dest_dir / f"{image_id}.jpg"
    if dest.exists():
        return True
    url = IMAGE_URL_TEMPLATE.format(split=split, image_id=image_id)
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except requests.RequestException as exc:
            if attempt == 0:
                time.sleep(1)
            else:
                log.debug("Skipping %s: %s", image_id, exc)
    return False


def fetch_split(
    image_ids: list, split: str, out_dir: Path, workers: int
) -> int:
    """Download a split's images and write an empty label beside each one."""
    images = out_dir / split / "images"
    labels = out_dir / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    # Open Images serves the validation split under "validation"; the
    # YOLO tree calls the same split "val".
    remote_split = "validation" if split == "val" else split

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_image, i, remote_split, images): i for i in image_ids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"{split} images"
        ):
            image_id = futures[future]
            if future.result():
                # The empty label file is the whole point: it is what
                # tells Ultralytics this image is a negative rather than
                # an unlabelled one.
                (labels / f"{image_id}.txt").write_text("")
                ok += 1
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download animal-free background images from Open Images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/backgrounds"))
    parser.add_argument("--cache", type=Path, default=Path("training/cache"))
    parser.add_argument("--train", type=int, default=1500, help="Train backgrounds.")
    parser.add_argument("--val", type=int, default=260, help="Val backgrounds.")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    animal_mids = load_animal_mids(args.cache)
    log.info("Excluding %d animal classes", len(animal_mids))

    total = 0
    for split, wanted, url in (
        ("train", args.train, ANNOTATION_URLS["train"]),
        ("val", args.val, ANNOTATION_URLS["validation"]),
    ):
        if wanted <= 0:
            continue
        csv_path = fetch_to_cache(
            url, args.cache / url.rsplit("/", 1)[-1], f"{split} annotations"
        )
        ids = select_background_ids(csv_path, animal_mids, wanted)
        got = fetch_split(ids, split, args.output, args.workers)
        log.info("%s: %d background images written", split, got)
        total += got

    print("")
    print(f"[INFO] {total} background images in {args.output.resolve()}")
    print("[INFO] Each has an empty .txt label, which is what marks it a negative.")
    print("[INFO] Compose them into a training set with training/build_dataset.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted; already-downloaded files are kept.")
        sys.exit(130)
