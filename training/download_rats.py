"""
download_rats.py
================
Downloads real wild rat photographs from iNaturalist for the RatCatcher
AI detector.

**Why this script exists.** `training/download_data.py` maps Open Images
`Mouse` (/m/04rmv) to class id 2 and calls it "rat", because Open Images
has no rat label. `docs/MODEL_TRAINING.md` section 2.2 inspected what
that actually put in the training set:

- Syrian and dwarf hamsters, in cages, on blankets, held in hands
- gerbils, voles, dormice, one chinchilla
- a small number of white laboratory mice
- roughly two images in twenty-four resembling a wild rat

and nearly all extreme close-ups, indoors, under domestic lighting. The
learned concept is "a large close-up furry rodent face indoors". A wild
rat at a feeder is a small, distant, side-on silhouette outdoors, often
at night. The two distributions barely intersect, which is why a
reported `rat mAP@0.5 = 0.762` coexists with a system that has never
detected a rat.

**Licensing.** This repository is GPL-3.0. iNaturalist photographs carry
per-observation licences and many are CC BY-NC, whose NonCommercial term
would follow anyone deploying a commercial derivative. So the images are
fetched at build time and never committed, the same treatment the
BirdNET weights already get, and `--licenses` restricts what is
downloaded. Every photograph is recorded in `ATTRIBUTION.csv` with its
observation id, photographer and licence, because attribution is a
condition of every CC licence here and cannot be reconstructed later
from the files.

**What this script does not do.** iNaturalist supplies photographs, not
bounding boxes. Run `training/label_rats.py` afterwards to propose boxes
and review them. Nothing here writes a YOLO label.

Usage:
    python training/download_rats.py --output datasets/rats --max 3000
    python training/download_rats.py --output datasets/rats --licenses cc0,cc-by

Dependencies: requests, tqdm
"""

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

API = "https://api.inaturalist.org/v1"

# iNaturalist asks that API clients identify themselves.
USER_AGENT = "RatCatcher-AI/0.4 (wildlife detector training; github.com/ratcatcher)"

# The two commensal rats that turn up at feeders. Resolved to taxon ids
# through the API rather than hardcoded, so a taxonomic revision on
# iNaturalist's side is picked up rather than silently returning nothing.
TARGET_SPECIES = ["Rattus norvegicus", "Rattus rattus"]

# Licences that impose no NonCommercial term. The default set is wider
# than this; see --licenses.
PERMISSIVE = {"cc0", "cc-by", "cc-by-sa"}

DEFAULT_LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_rats")


def api_get(path: str, params: dict, retries: int = 3) -> dict:
    """GET one API page, retrying on transient failures.

    iNaturalist rate-limits and occasionally 502s under load. Both are
    expected rather than exceptional, so they are retried with a back-off
    instead of aborting a download that may be thousands of images in.
    """
    for attempt in range(retries):
        try:
            resp = requests.get(
                f"{API}/{path}",
                params=params,
                timeout=60,
                headers={"User-Agent": USER_AGENT},
            )
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                log.warning("Rate limited, waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                log.error("API request failed for %s: %s", path, exc)
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Unreachable: exhausted retries for {path}")


def resolve_taxa(names: list) -> dict:
    """Map species names to iNaturalist taxon ids."""
    taxa = {}
    for name in names:
        data = api_get("taxa", {"q": name, "rank": "species", "per_page": 5})
        for result in data.get("results", []):
            if result.get("name", "").lower() == name.lower():
                taxa[name] = result["id"]
                log.info(
                    "%s -> taxon %d (%d observations)",
                    name,
                    result["id"],
                    result.get("observations_count", 0),
                )
                break
        else:
            log.warning("No exact taxon match for %s", name)
    if not taxa:
        raise RuntimeError("No target taxa resolved; cannot continue.")
    return taxa


def collect_photos(taxon_ids: list, licenses: set, wanted: int) -> list:
    """Collect photo records for research-grade observations.

    Paginates with `id_below` rather than `page`, because the API caps
    ordinary paging at 10000 results and the target counts can exceed
    that once several taxa are combined.
    """
    photos = []
    seen_photo_ids = set()
    id_below = None

    with tqdm(total=wanted, desc="observations") as pbar:
        while len(photos) < wanted:
            params = {
                "taxon_id": ",".join(str(t) for t in taxon_ids),
                "quality_grade": "research",
                "photos": "true",
                "per_page": 200,
                "order_by": "id",
                "order": "desc",
            }
            if id_below is not None:
                params["id_below"] = id_below

            data = api_get("observations", params)
            results = data.get("results", [])
            if not results:
                break

            for obs in results:
                id_below = obs["id"]
                for photo in obs.get("photos", []):
                    code = (photo.get("license_code") or "").lower()
                    if code not in licenses:
                        continue
                    if photo["id"] in seen_photo_ids:
                        continue
                    url = photo.get("url", "")
                    if not url:
                        continue
                    # The API hands back the square thumbnail URL. The
                    # size is a path segment, so asking for "large"
                    # is a substitution rather than a separate request.
                    large = url.replace("/square.", "/large.")
                    seen_photo_ids.add(photo["id"])
                    photos.append(
                        {
                            "photo_id": photo["id"],
                            "observation_id": obs["id"],
                            "url": large,
                            "license": code,
                            "attribution": photo.get("attribution", ""),
                            "observer": (obs.get("user") or {}).get("login", ""),
                            "taxon": (obs.get("taxon") or {}).get("name", ""),
                            "observed_on": obs.get("observed_on") or "",
                        }
                    )
                    pbar.update(1)
                    if len(photos) >= wanted:
                        break
                if len(photos) >= wanted:
                    break
    return photos


def download_photo(record: dict, images_dir: Path) -> bool:
    """Download one photograph. Returns True on success."""
    dest = images_dir / f"rat_{record['photo_id']}.jpg"
    if dest.exists():
        return True
    for attempt in range(2):
        try:
            resp = requests.get(
                record["url"], timeout=45, headers={"User-Agent": USER_AGENT}
            )
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except requests.RequestException as exc:
            if attempt == 0:
                time.sleep(1)
            else:
                log.debug("Skipping photo %s: %s", record["photo_id"], exc)
    return False


def write_attribution(records: list, out_dir: Path) -> Path:
    """Write the attribution manifest.

    Every CC licence in use here requires attribution, and the licence
    of an individual photograph cannot be recovered from the JPEG. This
    file is the only record, so it is written before the images are
    used rather than as a reporting afterthought.
    """
    path = out_dir / "ATTRIBUTION.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "photo_id",
                "observation_id",
                "taxon",
                "observed_on",
                "observer",
                "license",
                "attribution",
            ],
        )
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["file"] = f"rat_{record['photo_id']}.jpg"
            row.pop("url", None)
            writer.writerow(row)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download real wild rat photographs from iNaturalist.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/rats"))
    parser.add_argument("--max", type=int, default=3000, help="Photographs to fetch.")
    parser.add_argument(
        "--licenses",
        type=str,
        default=DEFAULT_LICENSES,
        help="Comma-separated iNaturalist licence codes to accept.",
    )
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    licenses = {code.strip().lower() for code in args.licenses.split(",") if code.strip()}

    images_dir = args.output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    taxa = resolve_taxa(TARGET_SPECIES)
    records = collect_photos(list(taxa.values()), licenses, args.max)
    log.info("Collected %d photograph records", len(records))
    if not records:
        print("[ERROR] No photographs matched. Widen --licenses and retry.")
        sys.exit(1)

    saved = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_photo, r, images_dir): r for r in records}
        for future in tqdm(as_completed(futures), total=len(futures), desc="photos"):
            record = futures[future]
            if future.result():
                saved.append(record)

    manifest = write_attribution(saved, args.output)

    restrictive = sorted({r["license"] for r in saved} - PERMISSIVE)
    print("")
    print(f"[INFO] {len(saved)} rat photographs in {images_dir.resolve()}")
    print(f"[INFO] Attribution manifest: {manifest}")
    if restrictive:
        print(
            f"[WARNING] Non-commercial licences present: {', '.join(restrictive)}. "
            "The images are gitignored, but a model trained on them is a "
            "derived work. Review before any commercial deployment."
        )
    print("[INFO] These have no bounding boxes yet. Run training/label_rats.py next.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted; already-downloaded files are kept.")
        sys.exit(130)
