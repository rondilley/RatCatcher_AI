"""
download_camera_traps.py
========================
Fetches boxed camera-trap imagery from LILA BC, which is the closest
public match to what this system's cameras actually see.

**Why this source.** Every other source of rats used here needs its boxes
proposed by a machine and reviewed by eye. This one carries boxes drawn
by the people who ran the study: 16,338 rat boxes over 6,886 images, plus
4,694 images with a cat and 77,670 verified empty frames, all from fixed
cameras that switch to infrared at night. That is the deployment
geometry -- a camera bolted in one place, an animal at a few metres, and
a monochrome frame after dark -- and it is what the iNaturalist
photographs are not: measured there, 3.2 percent are monochrome and dark,
51 percent are closer than a third of the frame, and of the observations
that carry an "Alive or Dead" annotation, 55 percent are dead.

The Island Conservation set is the default. `--dataset` selects another
of the LILA sets that use the same COCO Camera Traps layout.

**Why the empty frames matter as much as the rats.** `MODEL_TRAINING.md`
section 4.3 asks for hard negatives from the deployment, and the Pi
cannot yet store them. These are the nearest available thing: frames from
the same cameras, at the same sites, in the same light, whose correct
answer is "nothing here". The generic garden backgrounds now in the
training set cannot say whether a night-time IR frame of an empty path
produces a false rat.

**Why an image is taken whole or not at all.** In this format every
animal in a frame is boxed. Writing only the boxes for classes this
project knows would leave the other animals unlabelled, which asserts
that a photograph of a petrel contains no animal -- the same defect that
`label_rats.py` avoids by dropping an image rather than writing an empty
label. So every category is mapped onto one of the five RatCatcher
classes, and an image holding a category that cannot be mapped is
skipped entirely.

The one deliberate exception is `human`. This detector has no human
class and must not acquire one, so a person is left unlabelled on
purpose: the frame then teaches that a person is background, which is
the wanted behaviour.

**Licence.** The Island Conservation, Channel Islands, SWG and Wellington
sets are released under CDLA-Permissive 1.0, which permits use and
redistribution of the data. Nothing is committed regardless --
`datasets/` is gitignored in full -- and `PROVENANCE.csv` records the
dataset, its licence and the original file name for every image written.

Usage:
    python training/download_camera_traps.py
    python training/download_camera_traps.py --max-empty 6000 --workers 16
"""

import argparse
import csv
import io
import json
import logging
import random
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_camera_traps")

BASE = "https://storage.googleapis.com/public-datasets-lila"

# Each entry: the bucket prefix, the metadata archive, and the licence.
DATASETS = {
    "island_conservation": {
        "prefix": "islandconservationcameratraps",
        "metadata": "island_conservation_camera_traps_1.02.zip",
        "licence": "CDLA-Permissive-1.0",
    },
    "channel_islands": {
        "prefix": "channelislandcameratraps",
        "metadata": "channel_islands_camera_traps.json.zip",
        "licence": "CDLA-Permissive-1.0",
    },
}

# RatCatcher class ids, from config/species.yaml. Named here rather than
# imported because a label file written with the wrong id is silently
# wrong: it trains a confident cat detector.
BIRD, SQUIRREL, RAT, CAT, UNKNOWN = 0, 1, 2, 3, 4

# Every animal category in the Island Conservation set, mapped onto the
# five classes this detector knows. Seabirds are birds even though this
# project's birds are feeder birds: the shape of the argument a detector
# makes is the silhouette, and the species classifier is what tells one
# bird from another downstream.
CATEGORY_MAP = {
    "rat": RAT,
    "cat": CAT,
    "petrel": BIRD,
    "petrel_chick": BIRD,
    "shearwater": BIRD,
    "raven": BIRD,
    "rooster": BIRD,
    "chicken": BIRD,
    "megapode": BIRD,
    "short-eared_owl": BIRD,
    "owl": BIRD,
    "bird": BIRD,
    "white-winged_dove": BIRD,
    "yellow-crowned_night_heron": BIRD,
    "great_blue_heron": BIRD,
    "green_heron": BIRD,
    "brown_noddy": BIRD,
    "storm_petrel": BIRD,
    "nicobar_pigeon": BIRD,
    "pigeon": BIRD,
    "dove": BIRD,
    "rail": BIRD,
    "passerine": BIRD,
    "zorzal": BIRD,
    "mockingbird": BIRD,
    "american_kestrel": BIRD,
    "kestrel": BIRD,
    "burrowing_owl": BIRD,
    "barred_owl": BIRD,
    "rabbit": UNKNOWN,
    "iguana": UNKNOWN,
    "goat": UNKNOWN,
    "pig": UNKNOWN,
    "donkey": UNKNOWN,
    "monitor_lizard": UNKNOWN,
    "coati": UNKNOWN,
    "dog": UNKNOWN,
    "cow": UNKNOWN,
    "seal": UNKNOWN,
    "sea_turtle": UNKNOWN,
    "mouse": RAT,
    "squirrel": SQUIRREL,
}

# Categories that carry no box and mean the frame holds nothing.
EMPTY_CATEGORIES = {"empty"}

# Present, unmapped, and deliberately left unlabelled. The frame is kept
# and the object is not boxed, which teaches the detector that the object
# is background -- and for every entry here that is the wanted answer.
#
# A person is not one of the five classes and this detector must not
# acquire a human class. The invertebrates are the more useful half: a
# moth crossing an 850 nm illuminator at night is the classic false
# trigger for a feeder camera, and these frames are the only training
# signal available that says a moth is nothing. Mapping them to
# `unknown_animal` instead would have trained exactly the alert they
# should suppress.
IGNORED_CATEGORIES = {"human", "insect", "moth", "spider", "crab",
                      "hermit_crab"}

# An animal nobody could name, or a pair of eyes in the dark. Neither can
# be given a class, and leaving either unlabelled would teach the model
# that an animal is background, so the whole image is skipped.
UNSAFE_CATEGORIES = {"unknown", "eye_shine"}


def fetch_metadata(dataset: dict, cache: Path) -> dict:
    """Download and cache the COCO Camera Traps metadata."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        log.info("Using cached metadata at %s", cache)
        return json.loads(cache.read_text())

    url = f"{BASE}/{dataset['prefix']}/{dataset['metadata']}"
    log.info("Fetching metadata %s", url)
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not fetch metadata: {exc}") from exc

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    name = next(n for n in archive.namelist() if n.endswith(".json"))
    data = json.loads(archive.read(name))
    cache.write_text(json.dumps(data))
    log.info("Cached metadata: %d images, %d annotations",
             len(data["images"]), len(data["annotations"]))
    return data


def plan(data: dict, max_rat: int, max_cat: int, max_empty: int,
         seed: int) -> tuple:
    """Choose which images to fetch, and the boxes each one carries.

    Returns (selected, skipped) where selected maps an image record to
    the list of (class id, bbox) it should be written with. An empty list
    is a background image and is correct here, unlike in `label_rats.py`,
    because the annotation says the frame is empty rather than a proposer
    failing to find something.
    """
    categories = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}

    per_image = {}
    for annotation in data["annotations"]:
        per_image.setdefault(annotation["image_id"], []).append(annotation)

    rats, cats, empties, skipped = [], [], [], Counter()

    for image_id, annotations in per_image.items():
        names = {categories.get(a["category_id"], "") for a in annotations}

        if names & UNSAFE_CATEGORIES:
            skipped["unnameable animal in frame"] += 1
            continue

        if names <= (EMPTY_CATEGORIES | IGNORED_CATEGORIES):
            empties.append((image_id, []))
            continue

        boxes = []
        unmapped = False
        for annotation in annotations:
            name = categories.get(annotation["category_id"], "")
            if name in EMPTY_CATEGORIES or name in IGNORED_CATEGORIES:
                continue
            if name not in CATEGORY_MAP:
                unmapped = True
                skipped[f"unmapped category: {name}"] += 1
                break
            if "bbox" not in annotation:
                unmapped = True
                skipped[f"no box on {name}"] += 1
                break
            boxes.append((CATEGORY_MAP[name], annotation["bbox"]))
        if unmapped or not boxes:
            continue

        classes = {cls for cls, _ in boxes}
        if RAT in classes:
            rats.append((image_id, boxes))
        elif CAT in classes:
            cats.append((image_id, boxes))
        # Anything else -- a frame of only seabirds or only goats -- is
        # left out. This fetch exists to repair the rat class, and a
        # thousand petrels would shift the class balance for no gain.

    rng = random.Random(seed)
    rng.shuffle(rats)
    rng.shuffle(cats)
    rng.shuffle(empties)

    selected = rats[:max_rat] + cats[:max_cat] + empties[:max_empty]
    log.info("Selected %d rat, %d cat, %d empty images",
             min(len(rats), max_rat), min(len(cats), max_cat),
             min(len(empties), max_empty))
    return [(images[i], boxes) for i, boxes in selected], skipped


def to_yolo(boxes: list, width: int, height: int) -> list:
    """Convert COCO absolute xywh boxes to YOLO normalised lines."""
    lines = []
    for class_id, (x, y, w, h) in boxes:
        cx, cy = (x + w / 2) / width, (y + h / 2) / height
        bw, bh = w / width, h / height
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def download_one(job: tuple) -> dict:
    """Fetch one image and write its label. Returns a provenance row."""
    record, boxes, prefix, images_dir, labels_dir = job
    name = record["file_name"]
    stem = name.replace("/", "_").rsplit(".", 1)[0]
    target = images_dir / f"{stem}.jpg"

    if not target.is_file():
        url = f"{BASE}/{prefix}/public/{name}"
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            log.debug("Failed %s (%s)", name, exc)
            return {}
        target.write_bytes(response.content)

    lines = to_yolo(boxes, record["width"], record["height"])
    (labels_dir / f"{stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else "")
    )

    return {
        "file": target.name,
        "source_file": name,
        "location": record.get("location", ""),
        "datetime": record.get("datetime", ""),
        "boxes": len(lines),
    }


def write_provenance(rows: list, out_dir: Path, dataset: str,
                     licence: str) -> Path:
    """Record where each image came from and under what licence."""
    path = out_dir / "PROVENANCE.csv"
    fields = ["file", "dataset", "licence", "source_file", "location",
              "datetime", "boxes"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row, dataset=dataset, licence=licence)
            writer.writerow({key: row.get(key, "") for key in fields})
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch boxed camera-trap images from LILA BC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS),
                        default="island_conservation")
    parser.add_argument("--output", type=Path,
                        default=Path("datasets/camera_traps"))
    parser.add_argument("--max-rat", type=int, default=7000)
    parser.add_argument("--max-cat", type=int, default=2500)
    parser.add_argument("--max-empty", type=int, default=3000,
                        help="Verified empty frames from the same cameras. "
                             "The nearest available stand-in for the "
                             "deployment hard negatives.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DATASETS[args.dataset]

    cache = args.output / f"{args.dataset}_metadata.json"
    try:
        data = fetch_metadata(dataset, cache)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    selected, skipped = plan(data, args.max_rat, args.max_cat,
                             args.max_empty, args.seed)
    if not selected:
        print("[ERROR] Nothing selected")
        sys.exit(1)

    images_dir = args.output / "images"
    labels_dir = args.output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(record, boxes, dataset["prefix"], images_dir, labels_dir)
            for record, boxes in selected]

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in tqdm(pool.map(download_one, jobs), total=len(jobs),
                        desc="fetching"):
            if row:
                rows.append(row)

    provenance = write_provenance(rows, args.output, args.dataset,
                                  dataset["licence"])

    boxed = sum(1 for r in rows if r["boxes"])
    print("")
    print(f"[INFO] Images    : {len(rows)} written, {len(jobs) - len(rows)} failed")
    print(f"[INFO] With boxes: {boxed}")
    print(f"[INFO] Empty     : {len(rows) - boxed}")
    print(f"[INFO] Boxes     : {sum(r['boxes'] for r in rows)}")
    print(f"[INFO] Provenance: {provenance}")
    if skipped:
        print("")
        print("  Images skipped rather than mislabelled:")
        for reason, count in skipped.most_common(10):
            print(f"    {reason:<34} {count:>7}")


if __name__ == "__main__":
    main()
