"""
label_rats.py
=============
Proposes bounding boxes for the iNaturalist rat photographs fetched by
`training/download_rats.py`, and writes them as YOLO labels for class 2.

**Why this step exists.** iNaturalist supplies photographs and a species
identification. It does not supply bounding boxes, and a detector cannot
be trained without them.

**Why the boxes are machine-proposed.** Hand-drawing boxes on 3000
photographs is days of work. An open-vocabulary detector (YOLO-World)
takes a text prompt and returns boxes, which turns the job into review
rather than drawing. The output of this script is a *proposal*: run it
with `--sheets` and look at the contact sheets before training on the
result. `docs/MODEL_TRAINING.md` exists because the last model was
trained on data nobody had looked at.

**Why an image with no detection is dropped rather than left empty.**
This is the one decision here that can actively make the detector worse.
Every one of these photographs contains a rat -- that is what
research-grade identification means. If the proposer misses it and the
image is kept with an empty label file, it becomes a background image
that asserts there is no rat in a picture of a rat, training the model
to suppress exactly what it is meant to find. Backgrounds come from
`training/download_backgrounds.py`, where the claim is true. Here a miss
means the image is skipped.

**Why the confidence floor is not tight.** A false box on a photograph
that does contain a rat is usually a loose box around the rat rather
than a box on something else, which costs localisation accuracy. A
missed rat costs a training example entirely. The floor is therefore
permissive by default, and review is what catches the rest.

Usage:
    python training/label_rats.py --input datasets/rats --sheets
    python training/label_rats.py --input datasets/rats --conf 0.05
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Class id 2 in config/species.yaml and datasets/ratcatcher/dataset.yaml.
# Named here rather than passed in, because a label file written with the
# wrong id is silently wrong -- it trains a confident cat detector.
RAT_CLASS_ID = 2

# Text prompts for the open-vocabulary proposer. Several phrasings are
# given because the prompt wording measurably moves recall, and the union
# of the boxes is filtered afterwards.
PROMPTS = ["rat", "brown rat", "rodent", "mouse"]

# Prompts that are added to the vocabulary and whose boxes are then
# thrown away. They exist to give the proposer somewhere else to put an
# animal that is not a rat.
#
# **Why this is needed for video and not for iNaturalist.** An
# open-vocabulary detector assigns every object it finds to the closest
# entry in the vocabulary it was given. With only rat words in that
# vocabulary, a terrier has nowhere else to go. Measured on 350 frames
# cut from urban ratting footage: 22 frames were accepted and most of
# the boxes were on the dogs, not the rats. The photographs from
# iNaturalist never showed this, because a research-grade Rattus
# observation really does contain a rat -- so the flag is off by default
# and the photograph path is unchanged.
DISTRACTOR_PROMPTS = ["dog", "puppy", "person", "human hand", "cat", "bird"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("label_rats")


def load_proposer(weights: str, device: str):
    """Load the open-vocabulary detector."""
    try:
        from ultralytics import YOLOWorld
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from exc

    model = YOLOWorld(weights)
    log.info("Proposer %s on device %s", weights, device)
    return model


def iou(a, b) -> float:
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def merge_boxes(boxes: list, threshold: float = 0.6) -> list:
    """Collapse boxes that overlap heavily, keeping the most confident.

    The four prompts frequently fire on the same animal, so the raw
    output holds near-duplicates. Keeping all of them would train the
    model on several slightly different truths for one object.
    """
    kept = []
    for box in sorted(boxes, key=lambda b: -b[4]):
        if all(iou(box, other) < threshold for other in kept):
            kept.append(box)
    return kept


def reject_overlapping(boxes: list, guards: list, threshold: float,
                       guard_floor: float) -> list:
    """Drop a proposal that the distractor vocabulary explains confidently.

    **Why the guard's own confidence decides.** Asked for dogs and people
    and nothing else, the proposer still puts a box on a rat, because it
    must assign whatever it finds to the vocabulary it was given. So
    "a guard box overlaps this" is true of every proposal and rejects
    everything -- measured on 350 frames of urban ratting footage, it
    took 23 boxes to 0. Comparing the two confidences does not separate
    them either: the guard outscores the rat prompt on the true rats as
    well, because the rat pass runs at a deliberately permissive floor.

    What does separate them is how confident the guard is. Measured over
    those 23 boxes: the terriers score 0.55 to 0.89 as "dog", while the
    three unambiguous rats score 0.09, 0.21 and 0.37. The default sits in
    that gap. It is a threshold on one measured distribution from one
    source of footage, so `--reject-guard-conf` exists, and the review
    sheet remains the thing that decides whether it held.
    """
    kept = []
    for box in boxes:
        explained = any(guard[4] >= guard_floor and iou(box, guard) >= threshold
                        for guard in guards)
        if not explained:
            kept.append(box)
    return kept


def write_label(path: Path, boxes: list, width: int, height: int) -> None:
    """Write boxes as YOLO normalised cx cy w h lines."""
    lines = []
    for x1, y1, x2, y2, _conf in boxes:
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        # Clamp: the proposer can return boxes a pixel outside the frame,
        # and Ultralytics rejects a label file with out-of-range values.
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{RAT_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def build_sheet(records: list, out_path: Path, cell: int = 240, cols: int = 6) -> None:
    """Tile sample images with their proposed boxes drawn, for review."""
    rows = max(1, (len(records) + cols - 1) // cols)
    sheet = np.full((rows * cell, cols * cell, 3), 40, np.uint8)
    for i, (image_path, boxes) in enumerate(records):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for x1, y1, x2, y2, conf in boxes:
            cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3
            )
            cv2.putText(
                image, f"{conf:.2f}", (int(x1), max(20, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
            )
        scale = cell / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
        r, c = divmod(i, cols)
        sheet[r * cell : r * cell + image.shape[0],
              c * cell : c * cell + image.shape[1]] = image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propose YOLO boxes for iNaturalist rat photographs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=Path("datasets/rats"))
    parser.add_argument("--weights", type=str, default="yolov8x-worldv2.pt")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument(
        "--min-box-frac",
        type=float,
        default=0.0015,
        help="Reject boxes smaller than this fraction of the image area.",
    )
    parser.add_argument(
        "--max-box-frac",
        type=float,
        default=0.95,
        help="Reject boxes covering nearly the whole frame; those are "
        "the proposer failing to localise rather than a large animal.",
    )
    parser.add_argument(
        "--reject-distractors",
        action="store_true",
        help="Add dog, person and similar to the proposer vocabulary and "
             "discard their boxes. Needed for video frames, where the "
             "footage contains animals that are not rats.",
    )
    parser.add_argument(
        "--reject-iou",
        type=float,
        default=0.5,
        help="Drop a rat proposal overlapping a rejected box by this much.",
    )
    parser.add_argument(
        "--reject-guard-conf",
        type=float,
        default=0.45,
        help="Only reject when the distractor box scores at least this. "
             "Below it the proposer is guessing, and the guess deletes "
             "true rats.",
    )
    parser.add_argument("--sheets", action="store_true", help="Write review sheets.")
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def propose(weights: str, files: list, vocabulary: list, args, desc: str) -> tuple:
    """Run the proposer over every file with one vocabulary.

    Returns {path: (width, height, boxes)} and a count of files lost to
    a failed batch.

    **Why each pass loads its own proposer.** Setting a vocabulary
    encodes the prompt text against the model's weights, and the first
    `predict` has by then moved those weights to the GPU while the new
    token indices are still built on the CPU. Re-using one instance
    across two vocabularies therefore fails with "Expected all tensors to
    be on the same device". A second load costs a few seconds against a
    run of thousands of images.
    """
    model = load_proposer(weights, args.device)
    model.set_classes(vocabulary)
    log.info("%s with %s", desc, vocabulary)
    found, failed = {}, 0

    for start in tqdm(range(0, len(files), args.batch), desc=desc):
        chunk = files[start : start + args.batch]
        try:
            results = model.predict(
                [str(p) for p in chunk], conf=args.conf,
                device=args.device, verbose=False,
            )
        except (RuntimeError, OSError) as exc:
            # A corrupt JPEG in the batch takes the whole batch down.
            # Losing a few images matters far less than losing the run.
            log.warning("Batch at %d failed (%s), skipping", start, exc)
            failed += len(chunk)
            continue

        for image_path, result in zip(chunk, results):
            height, width = result.orig_shape
            boxes = [
                (*(float(v) for v in box.xyxy[0]), float(box.conf[0]))
                for box in result.boxes
            ]
            found[image_path] = (width, height, boxes)

    return found, failed


def main() -> None:
    args = parse_args()
    images_dir = args.input / "images"
    labels_dir = args.input / "labels"
    if not images_dir.is_dir():
        print(f"[ERROR] No images at {images_dir.resolve()}")
        print("[ERROR] Run training/download_rats.py first.")
        sys.exit(1)
    labels_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(images_dir.glob("*.jpg"))
    log.info("Labelling %d photographs", len(files))

    proposals, dropped = propose(args.weights, files, PROMPTS, args, "proposing")
    guards = {}
    if args.reject_distractors:
        guards, _ = propose(args.weights, files, DISTRACTOR_PROMPTS,
                            args, "guarding")

    labelled = 0
    rejected = 0
    total_boxes = 0
    reviewed = []

    for image_path in files:
        if image_path not in proposals:
            continue
        width, height, raw = proposals[image_path]
        area = float(width * height)

        boxes = []
        for x1, y1, x2, y2, conf in raw:
            frac = ((x2 - x1) * (y2 - y1)) / area
            if frac < args.min_box_frac or frac > args.max_box_frac:
                continue
            boxes.append((x1, y1, x2, y2, conf))

        boxes = merge_boxes(boxes)
        if image_path in guards:
            before = len(boxes)
            boxes = reject_overlapping(boxes, guards[image_path][2],
                                       args.reject_iou, args.reject_guard_conf)
            rejected += before - len(boxes)

        if not boxes:
            # See the module docstring: an empty label here would
            # assert there is no rat in a photograph of a rat.
            dropped += 1
            continue

        write_label(labels_dir / f"{image_path.stem}.txt", boxes, width, height)
        labelled += 1
        total_boxes += len(boxes)
        if args.sheets and len(reviewed) < 24:
            reviewed.append((image_path, boxes))

    print("")
    print(f"[INFO] Labelled  : {labelled} images, {total_boxes} boxes")
    print(f"[INFO] Dropped   : {dropped} images with no confident proposal")
    if args.reject_distractors:
        print(f"[INFO] Rejected  : {rejected} boxes sitting on a dog, "
              f"a person or another named distractor")
    if labelled:
        print(f"[INFO] Mean boxes: {total_boxes / labelled:.2f} per image")

    if args.sheets and reviewed:
        random.shuffle(reviewed)
        sheet = args.input / "review" / "proposals.jpg"
        build_sheet(reviewed[:24], sheet)
        print(f"[INFO] Review sheet: {sheet}")
        print("[INFO] Look at it before training. These boxes are proposals.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted; labels already written are kept.")
        sys.exit(130)
