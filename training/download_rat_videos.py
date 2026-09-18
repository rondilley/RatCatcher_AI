"""
download_rat_videos.py
======================
Fetches video of live rats outdoors, so the detector can be trained on
the domain the camera actually sees.

**Why video and not more photographs.** The iNaturalist set built by
`download_rats.py` is measurably the wrong domain. Measured over the
3264 photographs that carry a box: 3.2 percent are monochrome and dark
together, 51 percent put the animal closer than 35 percent of the frame
height, and of the 515 observations that carry an iNaturalist
"Alive or Dead" annotation, 55 percent are marked dead. A fixed feeder
camera sees none of that. It sees a live animal at a few metres, often
at night, in monochrome under an 850 nm illuminator.

Ratting and pest-control footage is the opposite: a camera at a fixed
distance, animals moving in the open, and a large part of it shot on
night vision. One ten-minute night clip yields more usable night frames
than the whole iNaturalist set holds.

**Why video-only streams.** Nothing here reads the audio track, and
fetching one means downloading a second stream and then muxing it. A
video-only MP4 skips both, and OpenCV decodes it directly. ffmpeg is
therefore not needed on the path this script takes, whether or not it is
installed.

**Why a provenance manifest.** The same reason `datasets/rats/
ATTRIBUTION.csv` exists: a JPEG cut out of a video carries no record of
where it came from, and that record cannot be reconstructed afterwards.
`datasets/` is gitignored in full, so nothing fetched here is committed
or redistributed.

Usage:
    python training/download_rat_videos.py --manifest training/rat_videos.txt
    python training/download_rat_videos.py --urls URL1 URL2 --output datasets/rat_videos
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_rat_videos")

# H.264 (avc1) first, and this is not a preference -- YouTube serves AV1
# for much of its catalogue, and the OpenCV build that reads these frames
# cannot decode it: the file downloads, and every frame read from it
# fails with "Failed to get pixel format". Asking for avc1 at the point
# of download is what keeps the extractor from receiving a file it cannot
# open. The fallbacks step down to any MP4, then any video-only stream,
# then a progressive one, so a host offering nothing else still yields a
# file -- which extract_rat_frames.py then reports as unreadable rather
# than silently skipping.
FORMAT = (
    "bv*[height<=?720][vcodec^=avc1]"
    "/bv*[height<=?720][ext=mp4]"
    "/bv*[height<=?720]"
    "/b[height<=?720]/b"
)

MANIFEST = "VIDEO_SOURCES.csv"


def read_manifest(path: Path) -> list:
    """Read URLs from a manifest, one per line.

    Blank lines and lines beginning with '#' are ignored, so the search
    results can be pasted in with their notes still attached. Anything
    after the first whitespace on a line is treated as a comment, which
    lets a licence or a day/night note sit beside its URL.
    """
    urls = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line.split()[0].rstrip(","))
    return urls


def build_options(output_dir: Path, playlist_max: int) -> dict:
    """yt-dlp options for a quiet, bounded, video-only fetch."""
    return {
        "format": FORMAT,
        "outtmpl": str(output_dir / "%(extractor)s_%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ignoreerrors": True,
        "noplaylist": playlist_max <= 1,
        "playlistend": max(1, playlist_max),
        "allow_unplayable_formats": False,
    }


def record_of(info: dict, path: Path) -> dict:
    """Flatten the fields worth keeping out of yt-dlp's info dict."""
    return {
        "file": path.name,
        "id": info.get("id", ""),
        "url": info.get("webpage_url", ""),
        "title": (info.get("title") or "").replace("\n", " "),
        "uploader": info.get("uploader", ""),
        "uploader_url": info.get("uploader_url", ""),
        "license": info.get("license", ""),
        "upload_date": info.get("upload_date", ""),
        "duration_s": info.get("duration", ""),
        "width": info.get("width", ""),
        "height": info.get("height", ""),
        "fps": info.get("fps", ""),
    }


def fetch(urls: list, output_dir: Path, playlist_max: int) -> list:
    """Download each URL, returning one provenance record per file."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required. Install with: pip install yt-dlp"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for index, url in enumerate(urls, start=1):
        log.info("[%d/%d] %s", index, len(urls), url)
        try:
            with yt_dlp.YoutubeDL(build_options(output_dir, playlist_max)) as ydl:
                info = ydl.extract_info(url, download=True)
        except (yt_dlp.utils.DownloadError, OSError, ValueError) as exc:
            # One dead link must not end a fetch of forty. Network
            # failures, removed videos and region blocks all land here.
            log.warning("Failed: %s (%s)", url, exc)
            continue

        if info is None:
            log.warning("No information returned for %s", url)
            continue

        entries = info.get("entries") if "entries" in info else [info]
        for entry in entries or []:
            if entry is None:
                continue
            downloaded = entry.get("requested_downloads") or []
            for item in downloaded:
                path = Path(item.get("filepath", ""))
                if path.is_file():
                    records.append(record_of(entry, path))
                    log.info("  saved %s (%.1f MB)", path.name,
                             path.stat().st_size / 1e6)

    return records


def write_manifest(records: list, output_dir: Path) -> Path:
    """Write or extend the provenance manifest."""
    path = output_dir / MANIFEST
    existing = []
    if path.is_file():
        with open(path, newline="") as fh:
            existing = [row for row in csv.DictReader(fh)]

    known = {row["file"] for row in existing}
    merged = existing + [r for r in records if r["file"] not in known]

    fields = ["file", "id", "url", "title", "uploader", "uploader_url",
              "license", "upload_date", "duration_s", "width", "height", "fps"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in merged:
            writer.writerow({key: row.get(key, "") for key in fields})
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download rat video for detector training frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path,
                        help="File of URLs, one per line, '#' for comments.")
    parser.add_argument("--urls", nargs="+", default=[],
                        help="URLs given directly on the command line.")
    parser.add_argument("--output", type=Path, default=Path("datasets/rat_videos"))
    parser.add_argument("--playlist-max", type=int, default=1,
                        help="Videos to take from a playlist or channel URL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    urls = list(args.urls)
    if args.manifest:
        if not args.manifest.is_file():
            print(f"[ERROR] No manifest at {args.manifest}")
            sys.exit(1)
        urls.extend(read_manifest(args.manifest))

    if not urls:
        print("[ERROR] Give --manifest or --urls")
        sys.exit(1)

    seen, unique = set(), []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)

    log.info("Fetching %d videos to %s", len(unique), args.output.resolve())
    records = fetch(unique, args.output, args.playlist_max)

    if not records:
        print("[ERROR] Nothing downloaded")
        sys.exit(1)

    manifest = write_manifest(records, args.output)
    total = sum(float(r["duration_s"] or 0) for r in records)
    print("")
    print(f"[INFO] Downloaded : {len(records)} videos, {total / 60:.1f} minutes")
    print(f"[INFO] Manifest   : {manifest}")
    print(f"[INFO] Next       : python training/extract_rat_frames.py "
          f"--videos {args.output}")


if __name__ == "__main__":
    main()
