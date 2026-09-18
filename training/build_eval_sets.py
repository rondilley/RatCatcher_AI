"""
build_eval_sets.py
==================
Builds the three camera-trap evaluation sets out of a composed dataset:

- ``eval_ct``        every camera-trap frame in the held-out split
- ``eval_ct_night``  the monochrome frames of that set (infrared flash)
- ``eval_ct_day``    the colour frames of that set

`build_dataset.py` holds out whole camera locations, so the camera-trap
frames in its ``val`` split are the frames from cameras the model never
trained on. That is the set that can answer "does this transfer to a
camera it has never seen", and the night half of it is the only set
that measures what the camera-trap source was added for. The composed
``val`` split cannot do this on its own: it also holds Open Images
photographs and garden backgrounds, and a score over the mixture hides
the camera-trap number inside it.

**Why night is decided by colour, not by clock or by brightness.** The
Reconyx cameras in the LILA sets switch to an infrared flash after
dark, and an infrared frame is monochrome whatever its exposure -- a
flash-lit rat at two metres is bright and grey. The provenance
timestamps cannot be used either: the hours in the two hand-built sets
overlap almost completely, because camera clocks are unset, in another
time zone, or in a season where 06:00 is dark at one site and light at
another. Mean chroma separates the two cleanly except at the boundary,
where a handful of frames carry a faint colour cast. The threshold is
`MONO_CHROMA` from `extract_rat_frames.py`, so the composition tables in
`training/README.md` and this division agree by construction rather
than by coincidence.

**Why the sets are rebuilt from scratch.** The first three sets were
made by hand on 2026-09-06 with a rule nobody wrote down, and no
measurement tried here reproduces them exactly: mean chroma at the
threshold above differs on 14 of 1643 frames, all within 0.4 of the
line. A set that only exists on one disk is not a measurement anyone
can repeat. This script replaces them, and `FRAMES.csv` records the
chroma and luma of every frame so the boundary is inspectable.

Usage:
    python training/build_eval_sets.py --dataset datasets/ratcatcher_v3
    python training/build_eval_sets.py --dataset datasets/ratcatcher_v3 --force
"""

import argparse
import csv
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import CLASS_NAMES, RAT_CLASS_ID, link, read_label  # noqa: E402
from extract_rat_frames import MONO_CHROMA, measure  # noqa: E402

# Importing extract_rat_frames configures the root logger. A second
# basicConfig here would be a no-op, so this logger uses that one.
log = logging.getLogger("build_eval_sets")

# build_dataset.py prefixes every camera-trap frame it links so the four
# sources can be told apart in the composed split. This is that prefix.
CAMERA_TRAP_PREFIX = "ct_"

SET_NAMES = ("eval_ct", "eval_ct_night", "eval_ct_day")

MANIFEST_FIELDS = ["file", "set", "luma", "chroma", "boxes", "rat_boxes",
                   "location"]


def classify(chroma: float) -> str:
    """Name the half of the set a frame belongs to."""
    return "eval_ct_night" if chroma < MONO_CHROMA else "eval_ct_day"


def load_locations(provenance: Path) -> dict:
    """Map each camera-trap file stem to its camera location.

    The provenance file is optional: the sets are still built without
    it, with an empty location column.
    """
    if not provenance.is_file():
        log.warning("No provenance at %s; location column will be empty",
                    provenance)
        return {}
    try:
        with provenance.open(newline="") as handle:
            return {Path(row["file"]).stem: row.get("location", "")
                    for row in csv.DictReader(handle)}
    except (OSError, csv.Error, KeyError) as exc:
        log.warning("Could not read %s (%s); location column will be empty",
                    provenance, exc)
        return {}


def write_yaml(out: Path) -> Path:
    """Write the Ultralytics dataset file for one evaluation set.

    ``train`` points at the same images as ``val``. There is nothing to
    train on here, but Ultralytics refuses a dataset file without the
    key, and pointing it at a different directory would be a lie.
    """
    path = out / "dataset.yaml"
    lines = [f"path: {out.resolve()}", "train: val/images", "val: val/images",
             "", "names:"]
    lines += [f"  {i}: {name}" for i, name in sorted(CLASS_NAMES.items())]
    path.write_text("\n".join(lines) + "\n")
    return path


def prepare_output(root: Path, force: bool) -> dict:
    """Make the three empty output trees, replacing old ones only if asked.

    An existing set is recognised by its ``dataset.yaml``. A directory at
    one of these paths that has no such file is not something this
    script wrote, and it is left alone whatever ``force`` says.
    """
    outs = {name: root / name for name in SET_NAMES}
    for name, out in outs.items():
        if not out.exists():
            continue
        if not (out / "dataset.yaml").is_file():
            log.error("%s exists and is not an evaluation set; refusing to "
                      "touch it", out)
            sys.exit(1)
        if not force:
            log.error("%s already exists. Pass --force to rebuild it.", out)
            sys.exit(1)
        shutil.rmtree(out)
    for out in outs.values():
        (out / "val" / "images").mkdir(parents=True)
        (out / "val" / "labels").mkdir(parents=True)
    return outs


def build(dataset: Path, root: Path, provenance: Path, force: bool) -> list:
    """Build the three sets and return the manifest rows."""
    images_in = dataset / "val" / "images"
    labels_in = dataset / "val" / "labels"
    frames = sorted(p for p in images_in.glob(f"{CAMERA_TRAP_PREFIX}*.jpg"))
    if not frames:
        log.error("No %s*.jpg in %s. Run build_dataset.py with the "
                  "camera-trap source first.", CAMERA_TRAP_PREFIX, images_in)
        sys.exit(1)

    locations = load_locations(provenance)
    outs = prepare_output(root, force)

    rows = []
    unreadable = 0
    for image in frames:
        # The composed split holds a symlink into datasets/camera_traps.
        # Resolve it so the evaluation set does not depend on the
        # composed split still being there.
        target = image.resolve()
        frame = cv2.imread(str(target))
        if frame is None:
            log.warning("Could not decode %s; skipped", target)
            unreadable += 1
            continue
        luma, chroma, _ = measure(frame)
        half = classify(chroma)

        lines = read_label(labels_in / f"{image.stem}.txt")
        for name in ("eval_ct", half):
            link(target, outs[name] / "val" / "images" / image.name)
            (outs[name] / "val" / "labels" / f"{image.stem}.txt").write_text(
                "\n".join(lines) + "\n"
            )

        # The provenance file names the frame without the prefix.
        stem = image.stem[len(CAMERA_TRAP_PREFIX):]
        rows.append({
            "file": image.name,
            "set": half,
            "luma": round(luma, 1),
            "chroma": round(chroma, 2),
            "boxes": len(lines),
            "rat_boxes": sum(1 for ln in lines
                             if int(ln.split()[0]) == RAT_CLASS_ID),
            "location": locations.get(stem, ""),
        })

    for out in outs.values():
        write_yaml(out)
    with (outs["eval_ct"] / "FRAMES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if unreadable:
        log.warning("%d frame(s) could not be decoded and are in no set",
                    unreadable)
    return rows


def summarise(rows: list, root: Path) -> None:
    """Report what was written, per set."""
    print("")
    print("=" * 62)
    print("  Camera-trap evaluation sets")
    print("=" * 62)
    print(f"  night is mean chroma below {MONO_CHROMA}")
    for name in SET_NAMES:
        subset = rows if name == "eval_ct" else [r for r in rows if r["set"] == name]
        empty = sum(1 for r in subset if r["boxes"] == 0)
        rats = sum(r["rat_boxes"] for r in subset)
        locations = Counter(r["location"] for r in subset)
        print(f"  {name:<14} frames {len(subset):>5}  empty {empty:>5}  "
              f"rat boxes {rats:>5}  locations {len(locations):>3}")
    print(f"  manifest: {(root / 'eval_ct' / 'FRAMES.csv').resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build eval_ct, eval_ct_night and eval_ct_day from the "
                    "camera-trap frames in a composed dataset's val split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path,
                        default=Path("datasets/ratcatcher_v3"),
                        help="Composed dataset whose val split holds the "
                             "held-out camera-trap frames.")
    parser.add_argument("--provenance", type=Path,
                        default=Path("datasets/camera_traps/PROVENANCE.csv"),
                        help="Camera-trap provenance, for the location column.")
    parser.add_argument("--output-root", type=Path, default=Path("datasets"),
                        help="Directory the three sets are written under.")
    parser.add_argument("--force", action="store_true",
                        help="Replace existing evaluation sets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build(args.dataset, args.output_root, args.provenance, args.force)
    summarise(rows, args.output_root)


if __name__ == "__main__":
    main()
