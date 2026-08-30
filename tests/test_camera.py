"""Tests for ratcatcher.camera -- FileSource and FrameBuffer."""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from ratcatcher.camera.capture import FileSource
from ratcatcher.camera.frame_buffer import FrameBuffer


def _create_test_image(path: Path, color: tuple = (0, 128, 255)) -> None:
    """Write a small JPEG image with a colored rectangle."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = color
    cv2.imwrite(str(path), img)


def _make_image_dir(tmp_path: Path, count: int = 3) -> Path:
    """Create a temp directory with *count* JPEG test images and return its path."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    colors = [
        (0, 128, 255),
        (255, 0, 0),
        (0, 255, 0),
    ]
    for i in range(count):
        color = colors[i % len(colors)]
        _create_test_image(img_dir / f"frame_{i:04d}.jpg", color)
    return img_dir


# ===================================================================
# FileSource tests
# ===================================================================


class TestFileSourceFromImages:
    def test_file_source_from_images(self, tmp_path: Path) -> None:
        img_dir = _make_image_dir(tmp_path, count=3)

        src = FileSource(path=img_dir, fps=10)
        src.start()
        assert src.is_running is True

        frames_read = []
        for _ in range(4):  # one extra to trigger end-of-sequence
            ok, frame = src.read()
            if ok and frame is not None:
                frames_read.append(frame)

        assert len(frames_read) == 3
        assert src.is_running is False

        # Verify frame shapes
        for frame in frames_read:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (100, 100, 3)

        src.stop()


class TestFileSourceResolution:
    def test_file_source_resolution(self, tmp_path: Path) -> None:
        img_dir = _make_image_dir(tmp_path, count=1)

        # Without explicit resolution -- should discover from first image
        src = FileSource(path=img_dir)
        src.start()
        assert src.resolution == (100, 100)  # width, height
        src.stop()

    def test_file_source_explicit_resolution(self, tmp_path: Path) -> None:
        img_dir = _make_image_dir(tmp_path, count=1)

        src = FileSource(path=img_dir, resolution=(50, 50))
        src.start()
        assert src.resolution == (50, 50)

        ok, frame = src.read()
        assert ok is True
        assert frame is not None
        # Frame should be resized to the requested resolution
        assert frame.shape[1] == 50  # width
        assert frame.shape[0] == 50  # height
        src.stop()


class TestFileSourceNonexistentPathRaises:
    def test_file_source_nonexistent_path_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "does_not_exist_at_all"
        src = FileSource(path=bogus)
        with pytest.raises(FileNotFoundError):
            src.start()


class TestFileSourceEmptyDirectoryRaises:
    def test_file_source_empty_directory_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        src = FileSource(path=empty_dir)
        with pytest.raises(FileNotFoundError, match="No supported image files"):
            src.start()


class TestFileSourceStopReleases:
    def test_file_source_stop_releases(self, tmp_path: Path) -> None:
        img_dir = _make_image_dir(tmp_path, count=2)

        src = FileSource(path=img_dir)
        src.start()
        assert src.is_running is True

        src.stop()
        assert src.is_running is False

        # After stop, read should return failure
        ok, frame = src.read()
        assert ok is False
        assert frame is None


# ===================================================================
# FrameBuffer tests
# ===================================================================


def _dummy_frame(value: int = 0) -> np.ndarray:
    """Return a small numpy frame filled with *value*."""
    return np.full((10, 10, 3), value, dtype=np.uint8)


class TestFrameBufferPushAndGet:
    def test_frame_buffer_push_and_get(self) -> None:
        buf = FrameBuffer(max_frames=10)

        now = time.time()
        buf.push(_dummy_frame(1), timestamp=now)
        buf.push(_dummy_frame(2), timestamp=now + 0.1)
        buf.push(_dummy_frame(3), timestamp=now + 0.2)

        assert buf.count == 3

        frames = buf.get_frames(seconds=5.0)
        assert len(frames) == 3

        # Frames are ordered oldest-first
        assert frames[0][0][0, 0, 0] == 1
        assert frames[1][0][0, 0, 0] == 2
        assert frames[2][0][0, 0, 0] == 3


class TestFrameBufferMaxCapacity:
    def test_frame_buffer_max_capacity(self) -> None:
        buf = FrameBuffer(max_frames=3)
        assert buf.max_frames == 3

        now = time.time()
        for i in range(5):
            buf.push(_dummy_frame(i), timestamp=now + i)

        # Only 3 frames should remain (the newest three: 2, 3, 4)
        assert buf.count == 3

        frames = buf.get_frames(seconds=100.0)
        assert len(frames) == 3

        values = [f[0][0, 0, 0] for f in frames]
        assert values == [2, 3, 4]


class TestFrameBufferGetBySeconds:
    def test_frame_buffer_get_by_seconds(self) -> None:
        buf = FrameBuffer(max_frames=100)

        base_ts = 1000.0
        # Push 10 frames, 1 second apart
        for i in range(10):
            buf.push(_dummy_frame(i), timestamp=base_ts + i)

        # Ask for last 3 seconds -- should get frames at t=7, 8, 9
        # (cutoff = 1009 - 3 = 1006, so ts >= 1006)
        frames = buf.get_frames(seconds=3.0)
        assert len(frames) == 4  # t=1006, 1007, 1008, 1009

        timestamps = [f[1] for f in frames]
        assert timestamps == [
            pytest.approx(1006.0),
            pytest.approx(1007.0),
            pytest.approx(1008.0),
            pytest.approx(1009.0),
        ]

    def test_frame_buffer_get_by_seconds_zero(self) -> None:
        buf = FrameBuffer(max_frames=10)
        buf.push(_dummy_frame(1), timestamp=100.0)
        assert buf.get_frames(seconds=0.0) == []

    def test_frame_buffer_get_by_seconds_negative(self) -> None:
        buf = FrameBuffer(max_frames=10)
        buf.push(_dummy_frame(1), timestamp=100.0)
        assert buf.get_frames(seconds=-1.0) == []

    def test_frame_buffer_get_by_seconds_empty(self) -> None:
        buf = FrameBuffer(max_frames=10)
        assert buf.get_frames(seconds=5.0) == []


class TestFrameBufferClear:
    def test_frame_buffer_clear(self) -> None:
        buf = FrameBuffer(max_frames=10)

        now = time.time()
        for i in range(5):
            buf.push(_dummy_frame(i), timestamp=now + i)

        assert buf.count == 5
        buf.clear()
        assert buf.count == 0
        assert buf.get_frames(seconds=100.0) == []


class TestFrameBufferThreadSafety:
    def test_frame_buffer_thread_safety(self) -> None:
        buf = FrameBuffer(max_frames=500)
        errors: list = []
        frames_per_thread = 100
        num_threads = 5

        def push_many(thread_id: int) -> None:
            try:
                for i in range(frames_per_thread):
                    value = thread_id * frames_per_thread + i
                    buf.push(
                        _dummy_frame(value % 256),
                        timestamp=time.time(),
                    )
            except Exception as exc:
                errors.append(exc)

        threads = []
        for tid in range(num_threads):
            t = threading.Thread(target=push_many, args=(tid,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0

        # Total pushed: 5 * 100 = 500 -- exactly at capacity
        assert buf.count == 500

        # get_frames should not raise under concurrent access either
        results = buf.get_frames(seconds=60.0)
        assert len(results) == 500


# -- PicameraSource shutdown ---------------------------------------------------
#
# Skipped unless picamera2 imports and a camera is actually attached.  These
# cover the shutdown path that hung the service: stop() discarded the result
# of its capture-thread join and closed the camera regardless, which
# deadlocks picamera2's close() against an outstanding capture_array().

from ratcatcher.camera.capture import _PICAMERA2_AVAILABLE  # noqa: E402

if _PICAMERA2_AVAILABLE:
    from ratcatcher.camera.capture import PicameraSource


def _camera_is_attached() -> bool:
    """Is there a camera this process can actually open?"""
    if not _PICAMERA2_AVAILABLE:
        return False
    try:
        from picamera2 import Picamera2

        return len(Picamera2.global_camera_info()) > 0
    except Exception:
        return False


requires_camera = pytest.mark.skipif(
    not _camera_is_attached(), reason="no attached camera"
)


@requires_camera
def test_stop_returns_promptly_and_releases_the_camera():
    """stop() must complete, not wedge the caller."""
    source = PicameraSource(camera_num=0, resolution=(640, 480), fps=30)
    source.start()
    assert source.is_running

    # Let the capture thread produce at least one frame, so stop() runs
    # against a live pipeline rather than an idle one.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        ok, _ = source.read()
        if ok:
            break
        time.sleep(0.05)

    started = time.monotonic()
    source.stop()
    elapsed = time.monotonic() - started

    # The join alone is bounded at 3s; a successful close adds well under
    # a second.  Anything past 10s means the deadlock is back.
    assert elapsed < 10.0, f"stop() took {elapsed:.1f}s"
    assert not source.is_running


@requires_camera
def test_camera_can_be_reopened_after_stop():
    """A stop that really released the device lets the next start succeed.

    This is what distinguishes a clean close from the deadlock guard, which
    deliberately leaks the device: if stop() had bailed out early, opening
    the same camera again would fail with the device busy.
    """
    for _ in range(3):
        source = PicameraSource(camera_num=0, resolution=(640, 480), fps=30)
        source.start()
        assert source.is_running
        source.stop()
        assert not source.is_running


# -- Channel order -------------------------------------------------------------
#
# PicameraSource must hand out BGR, because every consumer treats it as BGR:
# HailoDetector and the species classifier both cvtColor BGR2RGB before
# inference, the OpenCV backend sets swapRB=True, thumbnails go through
# cv2.imwrite, and the clip writer converts BGR to RGB for FFmpeg.
#
# It asked picamera2 for "BGR888" and got RGB, because libcamera names a
# format by memory byte order and numpy reads that back reversed. Red and
# blue were swapped through the entire video path.
#
# Ground truth has to come from outside the library, so these compare against
# rpicam-still, which writes a correctly-coloured JPEG.

import shutil  # noqa: E402
import subprocess  # noqa: E402


def _rpicam_reference(tmp_path, camera=0):
    """A correctly-coloured BGR image of whatever the camera sees."""
    out = tmp_path / "truth.jpg"
    try:
        subprocess.run(
            ["rpicam-still", "--camera", str(camera), "--width", "1332",
             "--height", "990", "--timeout", "2000", "--nopreview",
             "-o", str(out)],
            check=True, capture_output=True, timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        pytest.skip("rpicam-still did not produce a reference frame")
    img = cv2.imread(str(out))
    if img is None:
        pytest.skip("reference frame unreadable")
    return img


requires_reference = pytest.mark.skipif(
    not (_camera_is_attached() and shutil.which("rpicam-still")),
    reason="needs an attached camera and rpicam-still",
)


def _channel_grid(image, cells=24):
    """A coarse spatial signature per channel.

    Global channel means cannot decide channel order here: rpicam-still
    and the video pipeline apply different white balance, so the two
    captures of one scene differ by a per-channel gain. Comparing the
    spatial *pattern* instead is immune to that -- a gain cancels in the
    normalisation, while sky-versus-wall structure does not.
    """
    small = cv2.resize(image, (cells * 4, cells * 3), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32).reshape(-1, 3)


def _normalised(values):
    return (values - values.mean()) / (values.std() + 1e-6)


def _capture_one(camera=0, size=(1332, 990)):
    source = PicameraSource(camera_num=camera, resolution=size, fps=30)
    source.start()
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            ok, frame = source.read()
            if ok:
                return frame
            time.sleep(0.1)
    finally:
        source.stop()
    return None


@requires_reference
def test_picamera_source_delivers_bgr_not_rgb(tmp_path):
    """The regression that swapped red and blue through the whole pipeline."""
    truth = _rpicam_reference(tmp_path)
    frame = _capture_one()
    assert frame is not None, "camera produced no frame"

    t = _channel_grid(truth)
    a = _channel_grid(frame)

    identity = float(
        np.mean([np.mean(_normalised(a[:, k]) * _normalised(t[:, k])) for k in range(3)])
    )
    reversed_ = float(
        np.mean(
            [np.mean(_normalised(a[:, k]) * _normalised(t[:, 2 - k])) for k in range(3)]
        )
    )

    # Identity against reversal is the whole question, and it is the only
    # one this scene can answer. Asserting that each channel's *best*
    # match is itself would be testing scene statistics rather than
    # channel order: in daylight R and G track luminance together and are
    # nearly collinear, so red's closest match is green whichever way
    # round the channels are.
    assert identity > reversed_ + 0.05, (
        f"channel order is reversed: identity pairing r={identity:.3f} vs "
        f"reversed r={reversed_:.3f}. PicameraSource is handing out RGB while "
        f"every consumer treats it as BGR."
    )
    # Blue is the channel that carries independent information outdoors,
    # so it is the one that pins the ordering.
    blue_to_blue = float(np.mean(_normalised(a[:, 0]) * _normalised(t[:, 0])))
    blue_to_red = float(np.mean(_normalised(a[:, 0]) * _normalised(t[:, 2])))
    assert blue_to_blue > blue_to_red, (
        f"source channel 0 matches truth red ({blue_to_red:.3f}) better than "
        f"truth blue ({blue_to_blue:.3f})"
    )


@requires_reference
def test_blue_sky_lands_in_the_blue_channel(tmp_path):
    """A direct, human-checkable statement of the same thing.

    Skipped rather than failed indoors or after dark, where the top of the
    frame is not sky and the premise does not hold.
    """
    truth = _rpicam_reference(tmp_path)
    top = truth[: truth.shape[0] // 5].reshape(-1, 3).mean(axis=0)
    if not (top[0] > top[2] + 15):
        pytest.skip("top of frame is not blue sky; nothing to compare against")

    source = PicameraSource(camera_num=0, resolution=(1332, 990), fps=30)
    source.start()
    try:
        frame = None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            ok, f = source.read()
            if ok:
                frame = f
                break
            time.sleep(0.1)
        assert frame is not None, "camera produced no frame"
    finally:
        source.stop()

    band = frame[: frame.shape[0] // 5].reshape(-1, 3).mean(axis=0)
    assert band[0] > band[2], (
        f"sky is brightest in channel {int(np.argmax(band))}; in BGR it must "
        f"be channel 0 (blue). Channel means: {np.round(band, 1)}"
    )
