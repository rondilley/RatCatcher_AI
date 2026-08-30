"""Tests for what a thumbnail actually shows.

The complaint these exist for: stored thumbnails were unreadable.  Not
because the camera is poor -- three reductions stack between the sensor
and the JPEG.  A 4056x3040 capture is downscaled to the pipeline frame,
and that whole frame is then reduced to a 320 px longest edge, so a
House Finch measured at 62 px in the capture arrives at under 4 px:
smaller than one 8x8 JPEG block, with nothing left to recognise.

So these measure the animal in the written file rather than asserting
that a file was written.  Real fixture birds, composited into a real
full-resolution capture, through the real cut-and-write path.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ratcatcher.pipeline.engine import _cut_detail
from ratcatcher.storage.thumbnail import create_thumbnail

_FIXTURES = Path(__file__).parent / "fixtures"

CAP_W, CAP_H = 4056, 3040
FRAME_W, FRAME_H = 1440, 1080          # the shipped 4:3 pipeline frame
OBJ_H = 62                             # a House Finch at this feeder
AT = (2600, 2000)                      # (y, x): the bottom quarter


def _bird() -> np.ndarray:
    img = cv2.imread(str(_FIXTURES / "bird_cedar_waxwing.jpg"))
    if img is None:
        pytest.skip("bird fixture not readable")
    return img


def _capture(bird: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """One bird at feeder scale in a full-resolution capture."""
    capture = np.full((CAP_H, CAP_W, 3), 70, dtype=np.uint8)
    capture[::11, ::11] = 150                    # texture, not flat grey
    obj_w = max(2, int(bird.shape[1] * OBJ_H / bird.shape[0]))
    oy, ox = AT
    capture[oy : oy + OBJ_H, ox : ox + obj_w] = cv2.resize(
        bird, (obj_w, OBJ_H), interpolation=cv2.INTER_AREA
    )
    return capture, ox, oy, obj_w


def _detail_for(capture, ox, oy, obj_w):
    """Cut the window and the patch exactly as the pipeline does."""
    from ratcatcher.detection.roi_crop import plan_windows
    from ratcatcher.motion.detector import MotionRegion

    region = MotionRegion(
        x=ox, y=oy, width=obj_w, height=OBJ_H,
        area=(obj_w * OBJ_H) / (CAP_W * CAP_H),
    )
    win = plan_windows([region], CAP_W, CAP_H, window=640)[0]
    wx, wy, ww, wh = win.bounds
    window = capture[wy : wy + wh, wx : wx + ww].copy()
    return _cut_detail(window, (ox - wx, oy - wy, obj_w, OBJ_H))


def test_the_bird_is_bigger_than_a_jpeg_block(tmp_path: Path):
    """The measurement the complaint was about.

    Whole-frame at 320 puts the bird under 4 px.  Both paths are written
    here so the comparison is measured rather than remembered.
    """
    bird = _bird()
    capture, ox, oy, obj_w = _capture(bird)
    frame = cv2.resize(capture, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

    # The old path: the whole downscaled frame, reduced to 320.
    scale_y = CAP_H / FRAME_H
    frame_box = (
        int(ox / (CAP_W / FRAME_W)), int(oy / scale_y),
        int(obj_w / (CAP_W / FRAME_W)), int(OBJ_H / scale_y),
    )
    old_path = tmp_path / "whole_frame.jpg"
    create_thumbnail(frame, frame_box, old_path)
    old = cv2.imread(str(old_path))
    old_bird_h = frame_box[3] * (old.shape[0] / FRAME_H)

    # The new path: native pixels around the box.
    patch, patch_box = _detail_for(capture, ox, oy, obj_w)
    new_path = tmp_path / "detail.jpg"
    create_thumbnail(patch, patch_box, new_path)
    new = cv2.imread(str(new_path))
    new_bird_h = patch_box[3] * (new.shape[0] / patch.shape[0])

    assert old_bird_h < 5, (
        f"the whole-frame thumbnail is supposed to be the bad case, but the "
        f"bird measures {old_bird_h:.1f} px -- recheck the premise"
    )
    assert new_bird_h >= 40, (
        f"bird is {new_bird_h:.1f} px in the thumbnail; under 40 px there is "
        "not enough of it to identify by eye"
    )


def test_the_thumbnail_keeps_the_detail_the_frame_threw_away(tmp_path: Path):
    """Size alone is not the claim -- the pixels have to survive.

    Compares the standard deviation over the bird against the same
    measure over an equal patch of background.  A bird smeared into the
    surrounding grey scores like the background does.
    """
    bird = _bird()
    capture, ox, oy, obj_w = _capture(bird)

    patch, patch_box = _detail_for(capture, ox, oy, obj_w)
    out = tmp_path / "detail.jpg"
    create_thumbnail(patch, None, out)         # no overlay: measure pixels
    img = cv2.imread(str(out))

    bx, by, bw, bh = patch_box
    on_bird = img[by : by + bh, bx : bx + bw]
    background = img[0:bh, 0:bw]

    assert on_bird.std() > background.std() * 2, (
        f"bird std {on_bird.std():.1f} against background "
        f"{background.std():.1f} -- the detail did not survive the write"
    )


def test_the_box_is_drawn_at_full_thickness(tmp_path: Path):
    """Drawn before the resize, a 2 px line became a third of a pixel."""
    bird = _bird()
    capture, ox, oy, obj_w = _capture(bird)
    patch, patch_box = _detail_for(capture, ox, oy, obj_w)

    plain = tmp_path / "plain.jpg"
    boxed = tmp_path / "boxed.jpg"
    create_thumbnail(patch, None, plain)
    create_thumbnail(patch, patch_box, boxed)

    a = cv2.imread(str(plain)).astype(np.int16)
    b = cv2.imread(str(boxed)).astype(np.int16)

    # Pure green, at full value, on the rectangle and nowhere else.
    greener = (b[:, :, 1] - a[:, :, 1]) > 60
    assert greener.sum() > 0, "no box was drawn"

    bx, by, bw, bh = patch_box
    ys, xs = np.nonzero(greener)
    assert xs.min() >= bx - 3 and xs.max() <= bx + bw + 3
    assert ys.min() >= by - 3 and ys.max() <= by + bh + 3


def test_the_box_survives_a_reduction(tmp_path: Path):
    """The draw-order fix, on a source that is actually resized.

    The patch above is written at its own size, so draw order cannot
    show there.  A 1440-wide frame reduced to 320 is where a rectangle
    drawn first becomes 0.44 of a pixel of coverage: still visible, but
    a wash rather than a line.
    """
    frame = np.full((FRAME_H, FRAME_W, 3), 70, dtype=np.uint8)

    plain = tmp_path / "plain_big.jpg"
    boxed = tmp_path / "boxed_big.jpg"
    create_thumbnail(frame, None, plain)
    create_thumbnail(frame, (600, 400, 200, 200), boxed)

    a = cv2.imread(str(plain)).astype(np.int16)
    b = cv2.imread(str(boxed)).astype(np.int16)
    peak = int((b[:, :, 1] - a[:, :, 1]).max())

    # Full-value green on grey 70 is a delta of 185.  Drawn before a
    # 4.5x reduction it averages down to about 80.
    assert peak > 150, (
        f"peak green delta {peak}; the box was diluted by the resize, so it "
        "was drawn before it"
    )


def test_a_frame_smaller_than_max_size_is_not_enlarged(tmp_path: Path):
    """Upscaling would claim detail the sensor did not record."""
    patch = np.full((256, 256, 3), 90, dtype=np.uint8)
    patch[::7, ::7] = 200

    out = tmp_path / "small.jpg"
    create_thumbnail(patch, None, out, max_size=320)
    img = cv2.imread(str(out))

    assert img.shape[:2] == (256, 256)


def test_the_whole_frame_path_still_works_when_there_is_no_patch(tmp_path: Path):
    """Motion-only rows and roi_crop off keep the old behaviour."""
    frame = np.full((FRAME_H, FRAME_W, 3), 70, dtype=np.uint8)
    frame[::11, ::11] = 150

    out = tmp_path / "motion.jpg"
    create_thumbnail(frame, None, out)
    img = cv2.imread(str(out))

    assert max(img.shape[:2]) == 320
    assert img.shape[1] / img.shape[0] == pytest.approx(FRAME_W / FRAME_H, rel=0.02)


# -- Through the real storage loop ---------------------------------------------


def _run_storage(engine):
    import threading

    engine._stop_event.clear()
    t = threading.Thread(target=engine._storage_loop, daemon=True, name="store")
    t.start()
    return t


def _drain(engine, thread, event):
    import time

    deadline = time.monotonic() + 5.0
    while event.thumbnail_path is None and time.monotonic() < deadline:
        time.sleep(0.02)
    engine._stop_event.set()
    thread.join(timeout=5.0)


def _engine_writing_to(tmp_path: Path):
    from ratcatcher.config import Config, SystemConfig
    from ratcatcher.pipeline.engine import PipelineEngine

    return PipelineEngine(Config(system=SystemConfig(data_dir=str(tmp_path))))


def test_storage_writes_the_thumbnail_from_the_patch_when_there_is_one(
    tmp_path: Path,
):
    """The wiring: a windowed detection must not fall back to the frame."""
    from datetime import datetime

    from ratcatcher.pipeline.event import DetectionEvent

    bird = _bird()
    capture, ox, oy, obj_w = _capture(bird)
    frame = cv2.resize(capture, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    patch, patch_box = _detail_for(capture, ox, oy, obj_w)

    engine = _engine_writing_to(tmp_path)
    event = DetectionEvent(
        timestamp=datetime.now(), camera_id=0, frame=frame,
        frame_width=FRAME_W, frame_height=FRAME_H,
        stage="detection", class_name="bird", confidence=0.8,
        bbox=(100, 100, 20, 20),
        detail=patch, detail_bbox=patch_box,
    )
    engine._storage_queue.put(event)

    thread = _run_storage(engine)
    _drain(engine, thread, event)

    assert event.thumbnail_path is not None, "no thumbnail was written"
    img = cv2.imread(event.thumbnail_path)
    assert img.shape[:2] == patch.shape[:2], (
        f"thumbnail is {img.shape[:2]}, not the patch {patch.shape[:2]} -- "
        "the storage loop used the downscaled frame"
    )


def test_storage_falls_back_to_the_frame_without_a_patch(tmp_path: Path):
    """Motion-only rows and roi_crop off keep the old behaviour."""
    from datetime import datetime

    from ratcatcher.pipeline.event import DetectionEvent

    frame = np.full((FRAME_H, FRAME_W, 3), 70, dtype=np.uint8)
    frame[::11, ::11] = 150

    engine = _engine_writing_to(tmp_path)
    event = DetectionEvent(
        timestamp=datetime.now(), camera_id=0, frame=frame,
        frame_width=FRAME_W, frame_height=FRAME_H,
        stage="detection", class_name="squirrel", confidence=0.8,
        bbox=(400, 300, 120, 90),
    )
    engine._storage_queue.put(event)

    thread = _run_storage(engine)
    _drain(engine, thread, event)

    assert event.thumbnail_path is not None, "no thumbnail was written"
    img = cv2.imread(event.thumbnail_path)
    assert max(img.shape[:2]) == 320
