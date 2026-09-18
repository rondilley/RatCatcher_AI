"""Tests for training/build_eval_sets.py.

A small composed split is written to disk the way build_dataset.py
writes one -- symlinked images with a ``ct_`` prefix on the camera-trap
frames, real label files, a provenance CSV -- and the script is run on
it. The grey frame stands for an infrared flash frame and the coloured
one for daylight; both are real JPEGs measured by the real ``measure``.
"""

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import build_eval_sets  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "training" / "build_eval_sets.py"


def _write_jpeg(path: Path, colour: tuple) -> None:
    frame = np.empty((48, 64, 3), dtype=np.uint8)
    frame[:] = colour
    # Some texture, so a flat frame is not mistaken for a decode fault.
    frame[10:20, 10:30] = tuple(max(0, c - 40) for c in colour)
    assert cv2.imwrite(str(path), frame)


@pytest.fixture
def composed(tmp_path: Path) -> dict:
    """A composed split with two camera-trap frames and one background."""
    source = tmp_path / "camera_traps" / "images"
    source.mkdir(parents=True)
    _write_jpeg(source / "grey.jpg", (120, 120, 120))
    _write_jpeg(source / "colour.jpg", (60, 140, 200))

    dataset = tmp_path / "ratcatcher_v3"
    images = dataset / "val" / "images"
    labels = dataset / "val" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "ct_grey.jpg").symlink_to(source / "grey.jpg")
    (images / "ct_colour.jpg").symlink_to(source / "colour.jpg")
    (labels / "ct_grey.txt").write_text("2 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n")
    (labels / "ct_colour.txt").write_text("")
    # An Open Images photograph in the same split, which must not be taken.
    _write_jpeg(images / "bg_other.jpg", (200, 200, 200))
    (labels / "bg_other.txt").write_text("0 0.5 0.5 0.3 0.3\n")

    provenance = tmp_path / "camera_traps" / "PROVENANCE.csv"
    with provenance.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "dataset", "licence", "source_file",
                         "location", "datetime", "boxes"])
        writer.writerow(["grey.jpg", "island_conservation", "CDLA", "x",
                         "palau_cam01", "2018-09-15 02:06:38", "2"])
        writer.writerow(["colour.jpg", "island_conservation", "CDLA", "x",
                         "chile_cam02", "2013-04-06 13:25:08", "0"])
    return {"dataset": dataset, "provenance": provenance, "root": tmp_path}


def _run(composed: dict, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--dataset", str(composed["dataset"]),
         "--provenance", str(composed["provenance"]),
         "--output-root", str(composed["root"]), *extra],
        capture_output=True, text=True, timeout=120,
    )


def test_grey_frame_goes_to_night_and_colour_frame_to_day(composed):
    result = _run(composed)
    assert result.returncode == 0, result.stdout + result.stderr

    root = composed["root"]
    night = sorted(p.name for p in (root / "eval_ct_night/val/images").iterdir())
    day = sorted(p.name for p in (root / "eval_ct_day/val/images").iterdir())
    both = sorted(p.name for p in (root / "eval_ct/val/images").iterdir())
    assert night == ["ct_grey.jpg"]
    assert day == ["ct_colour.jpg"]
    assert both == ["ct_colour.jpg", "ct_grey.jpg"]


def test_labels_are_copied_verbatim_and_links_resolve_to_the_source(composed):
    assert _run(composed).returncode == 0
    root = composed["root"]
    for name in ("eval_ct", "eval_ct_night"):
        label = root / name / "val" / "labels" / "ct_grey.txt"
        assert label.read_text() == "2 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n"
        image = root / name / "val" / "images" / "ct_grey.jpg"
        assert image.is_symlink()
        # Points at the camera-trap file, not at the composed split's link.
        assert image.readlink() == (root / "camera_traps/images/grey.jpg").resolve()
    empty = root / "eval_ct_day" / "val" / "labels" / "ct_colour.txt"
    assert empty.read_text().strip() == ""


def test_yaml_names_the_five_classes_and_points_train_at_val(composed):
    assert _run(composed).returncode == 0
    for name in build_eval_sets.SET_NAMES:
        text = (composed["root"] / name / "dataset.yaml").read_text()
        assert "train: val/images" in text
        assert "val: val/images" in text
        assert "  2: rat" in text
        assert "  4: unknown_animal" in text


def test_manifest_records_chroma_location_and_rat_count(composed):
    assert _run(composed).returncode == 0
    with (composed["root"] / "eval_ct" / "FRAMES.csv").open(newline="") as handle:
        rows = {row["file"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"ct_grey.jpg", "ct_colour.jpg"}
    grey, colour = rows["ct_grey.jpg"], rows["ct_colour.jpg"]
    assert grey["set"] == "eval_ct_night"
    assert float(grey["chroma"]) < build_eval_sets.MONO_CHROMA
    assert grey["rat_boxes"] == "2" and grey["boxes"] == "2"
    assert grey["location"] == "palau_cam01"
    assert colour["set"] == "eval_ct_day"
    assert float(colour["chroma"]) >= build_eval_sets.MONO_CHROMA
    assert colour["boxes"] == "0"
    assert colour["location"] == "chile_cam02"


def test_existing_set_is_kept_without_force_and_replaced_with_it(composed):
    assert _run(composed).returncode == 0
    stale = composed["root"] / "eval_ct" / "val" / "images" / "ct_stale.jpg"
    stale.write_bytes(b"not a real frame")

    second = _run(composed)
    assert second.returncode == 1
    assert "--force" in second.stdout + second.stderr
    assert stale.exists()

    third = _run(composed, "--force")
    assert third.returncode == 0, third.stdout + third.stderr
    assert not stale.exists()
    assert (composed["root"] / "eval_ct" / "val" / "images" / "ct_grey.jpg").exists()


def test_a_directory_that_is_not_an_evaluation_set_is_never_removed(composed):
    foreign = composed["root"] / "eval_ct_night"
    foreign.mkdir()
    keep = foreign / "keep.txt"
    keep.write_text("someone else's file\n")

    result = _run(composed, "--force")
    assert result.returncode == 1
    assert keep.read_text() == "someone else's file\n"


def test_no_camera_trap_frames_is_an_error_not_an_empty_set(composed):
    for link in (composed["dataset"] / "val" / "images").glob("ct_*.jpg"):
        link.unlink()
    result = _run(composed)
    assert result.returncode == 1
    assert not (composed["root"] / "eval_ct").exists()
