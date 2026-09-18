"""
extract_rat_frames.py
=====================
Cuts training frames out of the video fetched by
`training/download_rat_videos.py`, and reports what domain they cover.

The output directory has the same shape `label_rats.py` expects, so the
next step is unchanged:

    python training/extract_rat_frames.py --videos datasets/rat_videos
    python training/label_rats.py --input datasets/rat_frames --sheets

**Why frames are sampled and not taken whole.** Video at 30 fps holds
almost no new information between one frame and the next. Keeping all of
them would fill the training set with thousands of copies of one animal
in one pose, which teaches the detector that pose and nothing else, and
which also breaks the measurement: near-duplicates split across train and
val make a validation score that reports memorisation as accuracy.

**Why sampling alone is not enough.** These cameras are mostly static.
A one-second interval on a fixed shot of an empty yard still yields
hundreds of identical frames. A perceptual hash rejects the repeats that
survive the interval, comparing each candidate against every frame kept
so far from the same video. The hash is a difference hash: 8x8 of
adjacent-column comparisons, so it responds to structure and ignores the
overall brightness drift that a night-vision gain control produces.

**Why blur is rejected against the video's own median.** Laplacian
variance has no fixed scale -- `camera/focus.py` documents the same
problem, where it spans four decades across real frames. An absolute
floor would empty a soft night clip and pass every frame of a sharp
daylight one. The floor here is a fraction of the median of the frames
sampled from that same video, which is a comparison within one exposure,
one lens and one scene, and so is meaningful where an absolute number is
not.

**Why every frame carries a measurement of its own domain.** Night
coverage is the entire reason this path exists. A run that yields 4000
frames is worth nothing if they are all daylight, and the only way to
know is to measure each frame and print the composition. `kind` is
derived the same way the iNaturalist set was measured, so the two are
directly comparable.

Nothing here is committed: `datasets/` is gitignored in full.
"""

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_rat_frames")

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")

# A frame is "monochrome" below this mean chroma and "dark" below this
# mean luma. Both thresholds are the ones the iNaturalist set was
# characterised with, so the two compositions can be compared directly.
MONO_CHROMA = 8.0
DARK_LUMA = 90.0


def dhash(image: np.ndarray) -> np.ndarray:
    """64-bit difference hash as 8 bytes.

    Structure, not brightness: each bit says whether one pixel is
    brighter than the pixel to its right, which is invariant to the gain
    changes a night-vision camera makes between frames.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return np.packbits(bits.reshape(-1))


def hamming_min(candidate: np.ndarray, kept: np.ndarray) -> int:
    """Smallest Hamming distance from one hash to a stack of hashes."""
    if len(kept) == 0:
        return 64
    diff = np.bitwise_xor(kept, candidate)
    return int(np.unpackbits(diff, axis=1).sum(axis=1).min())


def measure(image: np.ndarray) -> tuple:
    """Return (mean luma, mean chroma, Laplacian variance)."""
    f = image.astype(np.float32)
    luma = float((0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]).mean())
    chroma = float((f.max(axis=2) - f.min(axis=2)).mean())
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lapvar = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return luma, chroma, lapvar


def classify(luma: float, chroma: float) -> str:
    """Name the domain a frame belongs to."""
    mono = chroma < MONO_CHROMA
    dark = luma < DARK_LUMA
    if mono and dark:
        return "night_mono"
    if mono:
        return "day_mono"
    if dark:
        return "night_colour"
    return "day_colour"


def sample_video(path: Path, images_dir: Path, interval: float,
                 hash_distance: int, max_frames: int, min_side: int) -> list:
    """Sample, deduplicate and write frames from one video."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        log.warning("Cannot open %s", path.name)
        return []

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        # A video-only stream sometimes reports no usable rate. 30 is the
        # common case and only sets the sampling step, so a wrong guess
        # changes the interval, not the correctness of the frames.
        log.info("  %s reports fps %.1f, assuming 30", path.name, fps)
        fps = 30.0
    step = max(1, int(round(fps * interval)))

    records, hashes = [], []
    index = 0
    kept = 0
    while kept < max_frames:
        ok = capture.grab()
        if not ok:
            break
        if index % step:
            index += 1
            continue
        ok, frame = capture.retrieve()
        index += 1
        if not ok or frame is None:
            continue
        if min(frame.shape[:2]) < min_side:
            log.info("  %s is %dx%d, below --min-side", path.name,
                     frame.shape[1], frame.shape[0])
            break

        digest = dhash(frame)
        if hamming_min(digest, np.array(hashes, dtype=np.uint8)) <= hash_distance:
            continue
        hashes.append(digest)

        luma, chroma, lapvar = measure(frame)
        name = f"{path.stem}_f{index:07d}.jpg"
        if not cv2.imwrite(str(images_dir / name), frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            log.warning("  Could not write %s", name)
            continue
        kept += 1
        records.append({
            "file": name,
            "video": path.name,
            "frame_index": index,
            "time_s": round(index / fps, 2),
            "luma": round(luma, 1),
            "chroma": round(chroma, 1),
            "lapvar": round(lapvar, 1),
            "kind": classify(luma, chroma),
        })

    capture.release()
    return records


def drop_blurred(records: list, images_dir: Path, floor_fraction: float) -> list:
    """Remove frames far softer than the median frame of their video."""
    kept = []
    by_video = {}
    for record in records:
        by_video.setdefault(record["video"], []).append(record)

    for video, group in by_video.items():
        median = float(np.median([r["lapvar"] for r in group]))
        floor = median * floor_fraction
        dropped = 0
        for record in group:
            if record["lapvar"] < floor:
                try:
                    (images_dir / record["file"]).unlink()
                except OSError as exc:
                    log.warning("Could not remove %s (%s)", record["file"], exc)
                dropped += 1
                continue
            kept.append(record)
        if dropped:
            log.info("  %s: dropped %d of %d frames below %.1f "
                     "(median %.1f)", video, dropped, len(group), floor, median)
    return kept


def write_index(records: list, out_dir: Path) -> Path:
    """Write the per-frame index, so a frame can be traced to its video."""
    path = out_dir / "FRAMES.csv"
    fields = ["file", "video", "frame_index", "time_s", "luma", "chroma",
              "lapvar", "kind"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda r: (r["video"], r["frame_index"])):
            writer.writerow(record)
    return path


def report(records: list) -> None:
    """Print the domain composition, which is the point of the exercise."""
    if not records:
        return
    counts = {}
    for record in records:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
    print("")
    print("  Domain composition")
    for kind in ("night_mono", "day_mono", "night_colour", "day_colour"):
        n = counts.get(kind, 0)
        print(f"    {kind:<14} {n:>6}  {100 * n / len(records):>5.1f}%")
    print("")
    print("  For comparison, the iNaturalist rat photographs are "
          "3.2 percent night_mono.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut deduplicated training frames out of rat video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--videos", type=Path, default=Path("datasets/rat_videos"))
    parser.add_argument("--output", type=Path, default=Path("datasets/rat_frames"))
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between sampled frames.")
    parser.add_argument("--hash-distance", type=int, default=6,
                        help="Reject a frame within this Hamming distance "
                             "of one already kept from the same video.")
    parser.add_argument("--max-frames", type=int, default=600,
                        help="Cap per video, so one long clip cannot "
                             "dominate the set.")
    parser.add_argument("--min-side", type=int, default=240,
                        help="Skip video whose shorter side is below this.")
    parser.add_argument("--blur-floor", type=float, default=0.35,
                        help="Drop frames below this fraction of the "
                             "median sharpness of their own video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    videos = sorted(p for p in args.videos.glob("*")
                    if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        print(f"[ERROR] No video files in {args.videos.resolve()}")
        print("[ERROR] Run training/download_rat_videos.py first.")
        raise SystemExit(1)

    images_dir = args.output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for index, video in enumerate(videos, start=1):
        log.info("[%d/%d] %s", index, len(videos), video.name)
        records = sample_video(video, images_dir, args.interval,
                               args.hash_distance, args.max_frames,
                               args.min_side)
        if not records:
            # Almost always a codec the reader cannot handle -- AV1 is
            # the common one. Silence here reads as "this clip held no
            # rats", which is a different and much more expensive
            # conclusion, so it is said out loud.
            log.warning("  No frames read from %s. If the log above shows "
                        "a decoder error, re-fetch it: the downloader "
                        "asks for H.264 first for this reason.", video.name)
        else:
            log.info("  kept %d frames", len(records))
        all_records.extend(records)

    all_records = drop_blurred(all_records, images_dir, args.blur_floor)
    index_path = write_index(all_records, args.output)

    print("")
    print(f"[INFO] Videos read : {len(videos)}")
    print(f"[INFO] Frames kept : {len(all_records)}")
    print(f"[INFO] Index       : {index_path}")
    report(all_records)
    print("")
    print(f"[INFO] Next        : python training/label_rats.py "
          f"--input {args.output} --sheets")


if __name__ == "__main__":
    main()
