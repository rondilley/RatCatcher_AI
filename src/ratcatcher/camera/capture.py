"""Camera source abstraction layer for RatCatcher AI.

Provides a Protocol for camera sources and concrete implementations for
video files, image directories, USB webcams, and Raspberry Pi cameras.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# -- Picamera2 import guard --------------------------------------------------
# Picamera2 is only available on Raspberry Pi.  On other platforms the import
# will fail, which is expected.  PicameraSource simply will not be defined.
_PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2  # type: ignore[import-untyped]

    _PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None  # type: ignore[assignment,misc]


# -- Supported image extensions -----------------------------------------------
_IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
})


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CameraSource(Protocol):
    """Minimal interface every camera back-end must satisfy."""

    def start(self) -> None:
        """Begin producing frames."""
        ...

    def stop(self) -> None:
        """Release resources and stop producing frames."""
        ...

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return (success, frame).  Frame is BGR uint8 or None on failure."""
        ...

    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height) of produced frames."""
        ...

    @property
    def fps(self) -> int:
        """Configured frames per second."""
        ...

    @property
    def is_running(self) -> bool:
        """True while the source is actively producing frames."""
        ...


# ---------------------------------------------------------------------------
# FileSource
# ---------------------------------------------------------------------------


class FileSource:
    """Reads frames from a video file *or* a directory of images.

    For a video file the underlying cv2.VideoCapture is used directly.
    For an image directory the images are sorted lexicographically and
    served one at a time at the configured FPS (the caller is expected to
    call ``read()`` in a loop that respects real-time pacing if desired).
    When all frames / images have been consumed ``is_running`` becomes
    ``False``.
    """

    def __init__(
        self,
        path: str | Path,
        fps: int = 30,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        self._path = Path(path)
        self._fps = fps
        self._resolution = resolution
        self._running = False

        # Determined at start()
        self._mode: str = ""  # "video" or "images"
        self._cap: cv2.VideoCapture | None = None
        self._image_paths: list[Path] = []
        self._image_index: int = 0

        # Actual resolution discovered after first frame (or set by caller)
        self._actual_resolution: tuple[int, int] = resolution or (0, 0)

    # -- Protocol properties ------------------------------------------------

    @property
    def resolution(self) -> tuple[int, int]:
        return self._actual_resolution

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_running(self) -> bool:
        return self._running

    # -- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return

        if self._path.is_dir():
            self._mode = "images"
            self._image_paths = sorted(
                p for p in self._path.iterdir()
                if p.suffix.lower() in _IMAGE_EXTENSIONS
            )
            if not self._image_paths:
                raise FileNotFoundError(
                    f"No supported image files found in {self._path}"
                )
            self._image_index = 0
            # Peek at the first image to learn the resolution.
            first = cv2.imread(str(self._image_paths[0]))
            if first is None:
                raise IOError(
                    f"Failed to decode first image: {self._image_paths[0]}"
                )
            h, w = first.shape[:2]
            if self._resolution is not None:
                self._actual_resolution = self._resolution
            else:
                self._actual_resolution = (w, h)
            logger.info(
                "FileSource: directory mode, %d images at %dx%d",
                len(self._image_paths),
                self._actual_resolution[0],
                self._actual_resolution[1],
            )
        elif self._path.is_file():
            self._mode = "video"
            self._cap = cv2.VideoCapture(str(self._path))
            if not self._cap.isOpened():
                raise IOError(f"Failed to open video file: {self._path}")
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self._resolution is not None:
                self._actual_resolution = self._resolution
            else:
                self._actual_resolution = (w, h)
            # If caller did not specify FPS, inherit from the file.
            file_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if file_fps > 0 and self._fps == 30:
                self._fps = int(round(file_fps))
            logger.info(
                "FileSource: video mode, %s at %dx%d @ %d fps",
                self._path.name,
                self._actual_resolution[0],
                self._actual_resolution[1],
                self._fps,
            )
        else:
            raise FileNotFoundError(f"Path does not exist: {self._path}")

        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._image_paths = []
        self._image_index = 0

    # -- Frame reading ------------------------------------------------------

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._running:
            return False, None

        if self._mode == "video":
            return self._read_video()
        return self._read_image()

    def _read_video(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            self._running = False
            return False, None
        frame = self._maybe_resize(frame)
        return True, frame

    def _read_image(self) -> tuple[bool, np.ndarray | None]:
        if self._image_index >= len(self._image_paths):
            self._running = False
            return False, None
        img_path = self._image_paths[self._image_index]
        self._image_index += 1
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Failed to decode image: %s -- skipping", img_path)
            # Recurse to try the next image rather than returning a bad frame.
            return self._read_image()
        frame = self._maybe_resize(frame)
        return True, frame

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        tw, th = self._actual_resolution
        if tw > 0 and th > 0 and (w != tw or h != th):
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
        return frame


# ---------------------------------------------------------------------------
# WebcamSource
# ---------------------------------------------------------------------------


class WebcamSource:
    """USB / built-in webcam via OpenCV VideoCapture."""

    def __init__(
        self,
        device_index: int = 0,
        resolution: tuple[int, int] = (1920, 1080),
        fps: int = 30,
    ) -> None:
        self._device_index = device_index
        self._requested_resolution = resolution
        self._requested_fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._actual_resolution: tuple[int, int] = resolution

    # -- Protocol properties ------------------------------------------------

    @property
    def resolution(self) -> tuple[int, int]:
        return self._actual_resolution

    @property
    def fps(self) -> int:
        return self._requested_fps

    @property
    def is_running(self) -> bool:
        return self._running

    # -- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise IOError(
                f"Cannot open webcam at device index {self._device_index}"
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self._requested_fps)

        # Read back the actual values the driver accepted.
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_resolution = (actual_w, actual_h)
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            self._requested_fps = int(round(actual_fps))

        logger.info(
            "WebcamSource: device %d opened at %dx%d @ %d fps",
            self._device_index,
            actual_w,
            actual_h,
            self._requested_fps,
        )
        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("WebcamSource: device %d released", self._device_index)

    # -- Frame reading ------------------------------------------------------

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._running or self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            logger.warning("WebcamSource: failed to read frame from device %d",
                           self._device_index)
            return False, None
        return True, frame


# ---------------------------------------------------------------------------
# PicameraSource  (only available on Raspberry Pi)
# ---------------------------------------------------------------------------

if _PICAMERA2_AVAILABLE:

    class PicameraSource:
        """Raspberry Pi camera via Picamera2 with threaded capture.

        Supports dual-camera setups through the ``camera_num`` parameter
        (0 for the default camera, 1 for the secondary).  Frame reads are
        thread-safe -- a background thread continuously captures into an
        internal buffer protected by a lock.
        """

        def __init__(
            self,
            camera_num: int = 0,
            resolution: tuple[int, int] = (1920, 1080),
            fps: int = 30,
        ) -> None:
            self._camera_num = camera_num
            self._resolution = resolution
            self._fps = fps
            self._picam: Picamera2 | None = None
            self._running = False

            self._lock = threading.Lock()
            self._latest_frame: np.ndarray | None = None
            self._capture_thread: threading.Thread | None = None

        # -- Protocol properties --------------------------------------------

        @property
        def resolution(self) -> tuple[int, int]:
            return self._resolution

        @property
        def fps(self) -> int:
            return self._fps

        @property
        def is_running(self) -> bool:
            return self._running

        # -- Lifecycle ------------------------------------------------------

        def start(self) -> None:
            if self._running:
                return

            self._picam = Picamera2(camera_num=self._camera_num)

            # "RGB888" is not a typo, and it is why this comment exists.
            # libcamera names a format by its byte order in memory, which
            # numpy then reads back reversed, so picamera2's "RGB888"
            # hands out BGR arrays and its "BGR888" hands out RGB ones.
            # This asked for "BGR888" and therefore delivered RGB, while
            # every consumer -- HailoDetector, the species classifier,
            # the OpenCV backend's swapRB, cv2.imwrite for thumbnails,
            # and the clip writer's BGR-to-RGB step for FFmpeg -- treats
            # a frame as BGR. Red and blue were swapped through the whole
            # video path, on colour-critical work: the species classifier
            # separates finches from siskins largely on plumage colour.
            # Verified against rpicam-still on the same scene, which
            # writes a correct JPEG: reversed, the correlation is 0.9999;
            # direct, it is -0.9999.
            config = self._picam.create_video_configuration(
                main={"size": self._resolution, "format": "RGB888"},
                controls={"FrameRate": self._fps},
            )
            self._picam.configure(config)
            self._picam.start()
            self._running = True

            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="picam-capture"
            )
            self._capture_thread.start()
            logger.info(
                "PicameraSource: camera %d started at %dx%d @ %d fps",
                self._camera_num,
                self._resolution[0],
                self._resolution[1],
                self._fps,
            )

        def stop(self) -> None:
            self._running = False
            if self._capture_thread is not None:
                self._capture_thread.join(timeout=3.0)
                if self._capture_thread.is_alive():
                    # Do not close the camera while the capture thread is
                    # still inside capture_array().  picamera2's close()
                    # waits on the event loop that the outstanding request
                    # holds, and the two block each other forever: the
                    # process logs every loop as ended, never reaches
                    # "Pipeline stopped", and survives SIGTERM, so systemd
                    # waits out TimeoutStopSec and reports the unit failed.
                    # Leaking the device is the cheaper failure -- the
                    # capture thread is a daemon and stop() is only reached
                    # on the way out.
                    logger.warning(
                        "PicameraSource: camera %d capture thread still "
                        "running after 3s; leaving the device open rather "
                        "than deadlocking in close()",
                        self._camera_num,
                    )
                    return
                self._capture_thread = None
            if self._picam is not None:
                self._picam.stop()
                self._picam.close()
                self._picam = None
            with self._lock:
                self._latest_frame = None
            logger.info("PicameraSource: camera %d stopped", self._camera_num)

        # -- Frame reading --------------------------------------------------

        def read(self) -> tuple[bool, np.ndarray | None]:
            with self._lock:
                if self._latest_frame is None:
                    return False, None
                # Return a copy so consumers can mutate freely.
                frame = self._latest_frame.copy()
            return True, frame

        def capture_metadata(self) -> dict[str, Any] | None:
            """The camera's current control values, or None.

            Not part of the CameraSource protocol: only a real sensor has
            an exposure time or a lux estimate to report. Callers ask for
            it by duck-typing and carry on without it, which is what lets
            the focus tool run against FileSource with no hardware.

            These readings are what separate a lens that is merely out of
            focus from one that is covered: a blocked camera measured 7.3
            lux with the gain pinned at 7.9x while the other, in the same
            daylight, sat at 19242 lux and unity gain.
            """
            if self._picam is None:
                return None
            try:
                return dict(self._picam.capture_metadata())
            except Exception as exc:  # noqa: BLE001 -- diagnostics are optional
                logger.debug(
                    "PicameraSource: metadata unavailable on camera %d: %s",
                    self._camera_num,
                    exc,
                )
                return None

        # -- Background capture ---------------------------------------------

        def _capture_loop(self) -> None:
            interval = 1.0 / max(self._fps, 1)
            while self._running and self._picam is not None:
                try:
                    frame = self._picam.capture_array("main")
                    with self._lock:
                        self._latest_frame = frame
                except Exception:
                    logger.exception(
                        "PicameraSource: error capturing frame from camera %d",
                        self._camera_num,
                    )
                time.sleep(interval)
