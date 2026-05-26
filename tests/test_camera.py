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
