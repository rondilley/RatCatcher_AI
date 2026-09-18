"""
build_dataset.py
================
Composes the RatCatcher training set from its three sources.

`docs/MODEL_TRAINING.md` names four root causes for a detector that
scores mAP@0.5 = 0.751 on its own validation split while being
approximately 100 percent false positives on the non-bird classes in the
field. Two of them are fixable here, in the data:

- section 2.1, the training set contains zero background images
- section 2.2, the "rat" class is trained on pet hamsters

This script fixes both by composing a new dataset rather than editing
the old one in place. `datasets/ratcatcher` stays exactly as it was, so
the deployed model remains reproducible and the two can be compared.

Four sources:

1. `datasets/ratcatcher` -- Open Images animals. Kept for bird,
   squirrel, cat and unknown_animal.
2. `datasets/rats` -- real wild *Rattus* from iNaturalist, boxed by
   `training/label_rats.py`. Replaces class 2 entirely.
3. `datasets/backgrounds` -- animal-free scenes with empty labels.
4. `datasets/camera_traps` -- boxed night and day camera-trap frames
   from LILA BC, fetched by `training/download_camera_traps.py`.

**Why a fourth source, when 2 already replaced the hamsters.** Real wild
rats were the right repair and are still the wrong domain. Measured over
the iNaturalist photographs: median box height is 35.7 percent of the
frame, 3.2 percent are monochrome and dark together, and of the
observations carrying an "Alive or Dead" annotation 55 percent are dead.
The camera-trap frames are the deployment geometry -- a fixed camera, an
animal at a few metres, infrared after dark -- and their median rat box
is 13.1 percent of frame height against about 17 percent for this
feeder. They also carry boxes drawn by people rather than proposed by a
model.

**Why the old class 2 is dropped rather than added to.** The report is
explicit: the Open Images `Mouse` imagery is "actively teaching the
wrong concept". Keeping it alongside real rats would leave the model
splitting class 2 between a close-up indoor hamster face and a distant
outdoor rat silhouette, which is the mixture that produced a meaningless
0.762.

**Why an image that held only a hamster is skipped, not made a
background.** Dropping the class-2 annotation leaves a rodent in the
frame with nothing marking it. Filing that as a background would assert
there is no rodent in a photograph of one -- the same error
`label_rats.py` avoids, in the other direction. Those images leave the
set.

**Why images are symlinked.** The three sources total several
gigabytes and the composed set would double that for no benefit.
Labels are written as real files because they are transformed:
class-2 lines are filtered out and the rat and background labels come
from elsewhere.

Usage:
    python training/build_dataset.py --output datasets/ratcatcher_v2
"""

import argparse
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

# Class 2 in the existing dataset. Every annotation with this id comes
# from Open Images `Mouse` and is discarded; see the module docstring.
HAMSTER_CLASS_ID = 2

# The same id, and deliberately a second name. Class 2 means a pet
# hamster in the Open Images source, where every annotation carrying it
# is discarded, and a real rat in the iNaturalist and camera-trap
# sources, where it is what this rebuild exists to supply. One constant
# used for both would read as though the two were the same thing.
RAT_CLASS_ID = 2

CLASS_NAMES = {
    0: "bird",
    1: "squirrel",
    2: "rat",
    3: "cat",
    4: "unknown_animal",
}


def read_label(path: Path) -> list:
    """Return a label file's lines, ignoring blank ones."""
    try:
        return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    except OSError:
        return []


def link(src: Path, dest: Path) -> None:
    """Symlink *src* to *dest*, replacing any existing link."""
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(src.resolve())


def add_animals(source: Path, out: Path, split: str) -> tuple:
    """Copy the Open Images animals across, minus every class-2 label."""
    images_in = source / split / "images"
    labels_in = source / split / "labels"
    images_out = out / split / "images"
    labels_out = out / split / "labels"

    kept = skipped = 0
    counts = Counter()
    for image in sorted(images_in.glob("*.jpg")):
        label = labels_in / f"{image.stem}.txt"
        lines = read_label(label)
        remaining = [ln for ln in lines if int(ln.split()[0]) != HAMSTER_CLASS_ID]

        if lines and not remaining:
            # The image held nothing but hamsters. See the docstring.
            skipped += 1
            continue

        link(image, images_out / image.name)
        (labels_out / f"{image.stem}.txt").write_text("\n".join(remaining) + "\n")
        for line in remaining:
            counts[int(line.split()[0])] += 1
        kept += 1
    return kept, skipped, counts


def add_rats(source: Path, out: Path, val_fraction: float, seed: int) -> tuple:
    """Split the boxed rat photographs into train and val."""
    images_in = source / "images"
    labels_in = source / "labels"
    if not labels_in.is_dir():
        return {}, Counter()

    # Only photographs that actually got a box. label_rats.py drops the
    # rest rather than writing an empty label for them.
    boxed = sorted(p for p in images_in.glob("*.jpg") if (labels_in / f"{p.stem}.txt").is_file())
    random.Random(seed).shuffle(boxed)
    cut = int(len(boxed) * val_fraction)
    split_map = {"val": boxed[:cut], "train": boxed[cut:]}

    counts = Counter()
    added = {}
    for split, files in split_map.items():
        for image in files:
            link(image, out / split / "images" / image.name)
            lines = read_label(labels_in / f"{image.stem}.txt")
            (out / split / "labels" / f"{image.stem}.txt").write_text(
                "\n".join(lines) + "\n"
            )
            for line in lines:
                counts[int(line.split()[0])] += 1
        added[split] = len(files)
    return added, counts


def add_camera_traps(source: Path, out: Path, val_fraction: float,
                     seed: int, cap: int, max_rats: int) -> tuple:
    """Add the boxed camera-trap frames, split by camera and not at random.

    **Why the split is by location.** Every other source here is a set of
    unrelated photographs, so a random split is sound. These are not:
    one camera fires a burst of frames seconds apart, at one background,
    on one animal, and there are only 50 rat locations in the set. A
    random split puts near-identical frames on both sides of it, and the
    validation score then measures memorisation. Whole cameras are held
    out instead, which also asks the harder and more useful question --
    does this transfer to a camera it has never seen.

    **Why one camera is capped.** Splitting by location is necessary and
    not sufficient. Measured on the fetched set, `micronesia_cam06` is a
    bait station holding 1372 frames at 7.5 rats each -- 10,348 boxes,
    which is 64.8 percent of every rat instance in the corpus. Held out,
    it gave a val split with more rat instances than the train split and
    turned both the val score and the early-stopping signal into a
    measurement of one camera. Kept in, it would set the model's prior to
    a swarm, where a feeder sees one animal. `cap` bounds any single
    location.

    **Why a crowded frame is left out entirely.** The maintainer's
    observation of this feeder is one rat at a time, occasionally two,
    never more. The corpus does not look like that: 236 train frames --
    4 percent of the frames that hold a rat -- carry four or more animals
    each, and those 236 hold 27.9 percent of every rat instance. Nearly
    all come from the one bait station. Training on them sets the model's
    prior to a crowd and teaches it to divide an ambiguous shape into
    several boxes, which at a feeder is a duplicate alert on one animal.
    `max_rats` drops them. The held-out split needs no such filter and
    does not get one: it is already 97.7 percent one- and two-rat frames,
    which is what makes it the correct thing to measure against.

    **Why val locations are chosen by instance share, not image count.**
    The same camera defeats a count of images: 300 of its frames carry as
    many boxes as three thousand ordinary ones. Locations are added to
    the held-out set until it reaches the wanted share of *instances*,
    and a location that would badly overshoot what is left is passed
    over rather than taken.
    """
    images_in = source / "images"
    labels_in = source / "labels"
    provenance = source / "PROVENANCE.csv"
    if not labels_in.is_dir() or not provenance.is_file():
        return {}, Counter()

    location_of = {}
    with open(provenance, newline="") as fh:
        for row in csv.DictReader(fh):
            location_of[Path(row["file"]).stem] = row.get("location") or "unknown"

    by_location = {}
    crowded = 0
    for image in sorted(images_in.glob("*.jpg")):
        label = labels_in / f"{image.stem}.txt"
        if not label.is_file():
            continue
        if max_rats:
            rats = sum(1 for line in read_label(label)
                       if int(line.split()[0]) == RAT_CLASS_ID)
            if rats > max_rats:
                crowded += 1
                continue
        by_location.setdefault(location_of.get(image.stem, "unknown"), []).append(image)

    rng = random.Random(seed)
    for location, images in by_location.items():
        if cap and len(images) > cap:
            rng.shuffle(images)
            by_location[location] = sorted(images[:cap])

    instances = {
        location: sum(len(read_label(labels_in / f"{image.stem}.txt"))
                      for image in images)
        for location, images in by_location.items()
    }

    locations = sorted(by_location)
    rng.shuffle(locations)
    budget = sum(instances.values()) * val_fraction

    val_locations, taken = set(), 0
    for location in locations:
        if taken >= budget:
            break
        # A location that would overshoot the remaining budget by more
        # than half again is passed over. Without this one busy camera
        # takes the whole held-out set on its own.
        if instances[location] > 1.5 * (budget - taken) and val_locations:
            continue
        val_locations.add(location)
        taken += instances[location]

    counts = Counter()
    added = {"train": 0, "val": 0}
    for location, images in by_location.items():
        split = "val" if location in val_locations else "train"
        for image in images:
            # Prefixed for the same reason the backgrounds are: these
            # names come from a different corpus and must not collide.
            name = f"ct_{image.stem}"
            link(image, out / split / "images" / f"{name}.jpg")
            lines = read_label(labels_in / f"{image.stem}.txt")
            (out / split / "labels" / f"{name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            for line in lines:
                counts[int(line.split()[0])] += 1
            added[split] += 1

    added["val_locations"] = len(val_locations)
    added["locations"] = len(locations)
    added["crowded"] = crowded
    return added, counts


def add_backgrounds(source: Path, out: Path) -> dict:
    """Copy the negatives across, keeping their empty label files.

    Names are prefixed because both the backgrounds and the animal
    images come from Open Images, so an id could in principle appear in
    both and silently overwrite a label.
    """
    added = {}
    for split in ("train", "val"):
        images_in = source / split / "images"
        if not images_in.is_dir():
            added[split] = 0
            continue
        count = 0
        for image in sorted(images_in.glob("*.jpg")):
            name = f"bg_{image.stem}"
            link(image, out / split / "images" / f"{name}.jpg")
            # An empty file is what marks a negative to Ultralytics.
            (out / split / "labels" / f"{name}.txt").write_text("")
            count += 1
        added[split] = count
    return added


def write_yaml(out: Path) -> Path:
    path = out / "dataset.yaml"
    lines = [f"path: {out.resolve()}", "train: train/images", "val: val/images", "", "names:"]
    lines += [f"  {i}: {name}" for i, name in sorted(CLASS_NAMES.items())]
    path.write_text("\n".join(lines) + "\n")
    return path


def summarise(out: Path) -> None:
    """Report what was actually written, by reading it back off disk."""
    print("")
    print("=" * 62)
    print("  Composed dataset")
    print("=" * 62)
    for split in ("train", "val"):
        labels = sorted((out / split / "labels").glob("*.txt"))
        counts = Counter()
        empty = 0
        for label in labels:
            lines = read_label(label)
            if not lines:
                empty += 1
                continue
            for line in lines:
                counts[int(line.split()[0])] += 1
        total = len(labels)
        print("")
        print(f"  {split}: {total} images, {empty} background "
              f"({100 * empty / total:.1f}%)" if total else f"  {split}: empty")
        print(f"    {'class':<16} {'instances':>10}")
        print(f"    {'-' * 16} {'-' * 10}")
        for class_id, name in sorted(CLASS_NAMES.items()):
            print(f"    {name:<16} {counts.get(class_id, 0):>10}")
    print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose the RatCatcher training set from its sources.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--animals", type=Path, default=Path("datasets/ratcatcher"))
    parser.add_argument("--rats", type=Path, default=Path("datasets/rats"))
    parser.add_argument("--backgrounds", type=Path, default=Path("datasets/backgrounds"))
    parser.add_argument("--camera-traps", type=Path,
                        default=Path("datasets/camera_traps"))
    parser.add_argument("--max-rats-per-frame", type=int, default=3,
                        help="Drop a camera-trap frame holding more rats "
                             "than this. This feeder sees one, rarely two.")
    parser.add_argument("--camera-trap-cap", type=int, default=300,
                        help="Most frames to take from any one camera. "
                             "One bait station holds 65 percent of the "
                             "rat boxes in the corpus.")
    parser.add_argument("--output", type=Path, default=Path("datasets/ratcatcher_v2"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (args.animals / "train" / "images").is_dir():
        print(f"[ERROR] No animal dataset at {args.animals.resolve()}")
        sys.exit(1)

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "val"):
        (args.output / split / "images").mkdir(parents=True)
        (args.output / split / "labels").mkdir(parents=True)

    total_skipped = 0
    for split in ("train", "val"):
        kept, skipped, _ = add_animals(args.animals, args.output, split)
        total_skipped += skipped
        print(f"[INFO] {split}: {kept} animal images, {skipped} hamster-only skipped")

    rats_added, _ = add_rats(args.rats, args.output, args.val_fraction, args.seed)
    if rats_added:
        print(f"[INFO] rats: {rats_added.get('train', 0)} train, "
              f"{rats_added.get('val', 0)} val")
    else:
        print(f"[WARNING] No boxed rats at {args.rats.resolve()}. "
              "Run download_rats.py then label_rats.py.")

    ct_added, _ = add_camera_traps(args.camera_traps, args.output,
                                   args.val_fraction, args.seed,
                                   args.camera_trap_cap,
                                   args.max_rats_per_frame)
    if ct_added:
        print(f"[INFO] camera traps: {ct_added.get('train', 0)} train, "
              f"{ct_added.get('val', 0)} val, held out "
              f"{ct_added.get('val_locations', 0)} of "
              f"{ct_added.get('locations', 0)} cameras, "
              f"dropped {ct_added.get('crowded', 0)} crowded frames")
    else:
        print(f"[WARNING] No camera-trap frames at "
              f"{args.camera_traps.resolve()}. "
              "Run download_camera_traps.py.")

    bg_added = add_backgrounds(args.backgrounds, args.output)
    if any(bg_added.values()):
        print(f"[INFO] backgrounds: {bg_added.get('train', 0)} train, "
              f"{bg_added.get('val', 0)} val")
    else:
        print(f"[WARNING] No backgrounds at {args.backgrounds.resolve()}. "
              "Section 2.1 of the training report is not addressed without them.")

    yaml_path = write_yaml(args.output)
    summarise(args.output)
    print(f"  Dataset config: {yaml_path.resolve()}")
    print(f"  Discarded {total_skipped} images whose only label was a "
          "hamster from Open Images `Mouse`.")
    print("")


if __name__ == "__main__":
    main()
