"""Pipeline orchestrator for RatCatcher AI.

Coordinates the full detection pipeline across multiple cameras:
  Camera capture (thread per camera)
    -> Motion detection
    -> Object detection (shared, Hailo NPU or CPU)
    -> Species classification (CPU, birds only)
    -> Storage (SQLite + clips + thumbnails)
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ratcatcher.camera.capture import CameraSource
from ratcatcher.camera.frame_buffer import FrameBuffer
from ratcatcher.camera.platform_camera import create_camera
from ratcatcher.config import Config
from ratcatcher.detection.roi_crop import plan_windows
from ratcatcher.monitoring.events import log_video_detection
from ratcatcher.motion.detector import MotionDetector
from ratcatcher.pipeline.event import DetectionEvent
from ratcatcher.storage.clip_writer import ClipWriter
from ratcatcher.storage.database import DetectionDatabase
from ratcatcher.storage.thumbnail import create_thumbnail

logger = logging.getLogger(__name__)

_SENTINEL = object()

# Detail-patch geometry, in native window pixels.  The patch is what the
# thumbnail is written from, so it has to hold the animal at a size a
# person can judge: at this feeder a House Finch is about 62 px tall in
# the capture, which a 320-wide thumbnail of the whole 1920x1080 frame
# reduces to 3.7 px -- under one 8x8 JPEG block.  Twice the box gives
# enough surroundings to read what the animal is standing on, and the
# floor keeps a small bird from yielding a patch too small to see.
_DETAIL_PAD = 2.0
_DETAIL_MIN = 256


def _cut_detail(
    window: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Cut a square of native pixels around *bbox* out of *window*.

    Both are in window coordinates.  Returns the patch and the box
    expressed relative to it.  The patch is copied rather than sliced:
    a numpy view would keep the whole 1.2 MB window alive for as long as
    the event sits in a queue, which is the cost this exists to avoid.
    """
    win_h, win_w = window.shape[:2]
    bx, by, bw, bh = bbox

    side = int(min(win_w, win_h, max(_DETAIL_MIN, max(bw, bh) * _DETAIL_PAD)))

    x0 = int(round(bx + bw / 2 - side / 2))
    y0 = int(round(by + bh / 2 - side / 2))
    x0 = max(0, min(x0, win_w - side))
    y0 = max(0, min(y0, win_h - side))

    patch = window[y0 : y0 + side, x0 : x0 + side].copy()
    return patch, (bx - x0, by - y0, bw, bh)


class PipelineEngine:
    """Multi-camera wildlife detection pipeline.

    Threading architecture::

        Camera-0 thread -> motion detect -+
                                          +-> detection_queue
        Camera-1 thread -> motion detect -+
                                          |
                              detection_thread (YOLO)
                                          |
                                  classification_queue (birds)
                                          |
                              classification_thread (species ID)
                                          |
                                  storage_queue
                                          |
                              storage_thread (SQLite + clips)
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        self._detection_queue: queue.Queue = queue.Queue(maxsize=64)
        self._classification_queue: queue.Queue = queue.Queue(maxsize=64)
        self._storage_queue: queue.Queue = queue.Queue(maxsize=256)

        self._cameras: list[CameraSource] = []
        self._camera_configs: list = []
        self._frame_buffers: list[FrameBuffer] = []
        self._motion_detectors: list[MotionDetector] = []

        self._detector = None
        self._classifier = None
        self._db: DetectionDatabase | None = None
        self._clip_writer: ClipWriter | None = None

        self._stats_lock = threading.Lock()
        self._stats = {
            "frames_captured": 0,
            "motion_events": 0,
            # Native-resolution windows cut for the detector. Zero while
            # motion fires means the ROI path is configured off, or every
            # region was rejected as an illumination change.
            "crop_windows": 0,
            "detections": 0,
            "classifications": 0,
            "stored": 0,
            "dropped_frames": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def start(
        self,
        camera_ids: list[int] | None = None,
        skip_classification: bool = False,
    ) -> None:
        """Start the pipeline.

        Parameters
        ----------
        camera_ids:
            Which cameras to activate. None means all enabled cameras.
        skip_classification:
            If True, skip the species classification stage.
        """
        logger.info("Starting RatCatcher AI pipeline")
        self._stop_event.clear()

        data_dir = Path(self._config.system.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        self._db = DetectionDatabase(self._config.db_full_path)

        clip_dir = self._config.clip_full_path
        clip_dir.mkdir(parents=True, exist_ok=True)
        self._clip_writer = ClipWriter(
            clip_dir=clip_dir,
            ffmpeg_preset=self._config.storage.ffmpeg_preset,
        )

        thumb_dir = self._config.thumbnail_full_path
        thumb_dir.mkdir(parents=True, exist_ok=True)

        active_cameras = []
        for cam_cfg in self._config.cameras:
            if not cam_cfg.enabled:
                continue
            if camera_ids is not None and cam_cfg.id not in camera_ids:
                continue
            active_cameras.append(cam_cfg)

        if not active_cameras:
            raise RuntimeError("No cameras configured or selected")

        for cam_cfg in active_cameras:
            try:
                camera = create_camera(cam_cfg)
                camera.start()
            except Exception:
                logger.exception(
                    "Failed to start camera %d -- skipping", cam_cfg.id
                )
                continue

            self._cameras.append(camera)
            self._camera_configs.append(cam_cfg)

            buf_frames = cam_cfg.fps * self._config.storage.clip_pre_seconds
            self._frame_buffers.append(FrameBuffer(max_frames=max(buf_frames, 30)))

            self._motion_detectors.append(MotionDetector(self._config.motion))

            t = threading.Thread(
                target=self._camera_loop,
                args=(len(self._cameras) - 1, cam_cfg.id),
                daemon=True,
                name=f"camera-{cam_cfg.id}",
            )
            self._threads.append(t)

        if not self._cameras:
            raise RuntimeError("No cameras could be started")

        if self._config.detection.enabled:
            self._init_detector()
            t = threading.Thread(
                target=self._detection_loop,
                daemon=True,
                name="detection",
            )
            self._threads.append(t)

        if self._config.classification.enabled and not skip_classification:
            self._init_classifier()
            t = threading.Thread(
                target=self._classification_loop,
                daemon=True,
                name="classification",
            )
            self._threads.append(t)

        t = threading.Thread(
            target=self._storage_loop,
            daemon=True,
            name="storage",
        )
        self._threads.append(t)

        t = threading.Thread(
            target=self._stats_loop,
            daemon=True,
            name="stats",
        )
        self._threads.append(t)

        for t in self._threads:
            t.start()

        logger.info(
            "Pipeline started: %d cameras, detection=%s, classification=%s",
            len(self._cameras),
            "on" if self._detector else "off",
            "on" if self._classifier else "off",
        )

    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        logger.info("Stopping pipeline...")
        self._stop_event.set()

        for cam in self._cameras:
            try:
                cam.stop()
            except Exception:
                logger.exception("Error stopping camera")

        for q in (self._detection_queue, self._classification_queue, self._storage_queue):
            try:
                q.put_nowait(_SENTINEL)
            except queue.Full:
                logger.warning("Queue full during shutdown, forcing sentinel")
                try:
                    q.get_nowait()
                    q.put_nowait(_SENTINEL)
                except queue.Empty:
                    pass

        for t in self._threads:
            t.join(timeout=5.0)
            if t.is_alive():
                logger.warning("Thread %s did not stop within timeout", t.name)

        # Release the NPU before the interpreter exits.
        #
        # HailoDetector holds a VDevice, a configured network group and
        # its vstreams. Left to the garbage collector, that teardown
        # runs during interpreter finalisation and segfaults: the
        # process logged a clean "Pipeline stopped" and then died with
        # SIGSEGV on every single stop, which made systemd report the
        # unit as failed after an ordinary systemctl stop.
        #
        # Duck-typed rather than declared on the ObjectDetector
        # protocol: only the Hailo backend owns a device, and the test
        # that would let this be typed -- importing HailoDetector here
        # to isinstance against it -- would pull hailo_platform into
        # every platform that runs the pipeline.
        if self._detector is not None:
            close = getattr(self._detector, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    logger.exception("Error closing object detector")
            self._detector = None

        if self._db is not None:
            self._db.close()
            self._db = None

        self._cameras.clear()
        self._camera_configs.clear()
        self._frame_buffers.clear()
        self._motion_detectors.clear()
        self._threads.clear()

        logger.info("Pipeline stopped. Stats: %s", self.stats)

    def run_until_stopped(self) -> None:
        """Block until SIGINT/SIGTERM or stop() is called."""
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_signal(signum: int, frame: object) -> None:
            logger.info("Received signal %d, shutting down", signum)
            self._stop_event.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=1.0)
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            self.stop()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_detector(self) -> None:
        try:
            from ratcatcher.detection.factory import create_detector
            models_dir = Path("models")
            self._detector = create_detector(self._config.detection, models_dir)
            logger.info("Object detector initialized: %s", self._detector.backend_name)
        except Exception:
            logger.exception("Failed to initialize object detector -- running without detection")
            self._detector = None

    def _init_classifier(self) -> None:
        try:
            from ratcatcher.classification.taxonomy import Taxonomy
            from ratcatcher.classification.classifier import SpeciesClassifier

            config_dir = Path(
                __import__("os").environ.get(
                    "RATCATCHER_CONFIG_DIR",
                    str(Path(__file__).parent.parent.parent.parent / "config"),
                )
            )
            taxonomy = Taxonomy(config_dir / self._config.classification.species_config)

            model_path = Path("models") / self._config.classification.model_path
            self._classifier = SpeciesClassifier(
                model_path=model_path,
                taxonomy=taxonomy,
                input_size=self._config.classification.input_size,
                top_k=self._config.classification.top_k,
                min_confidence=self._config.classification.min_confidence,
                use_xnnpack=self._config.classification.use_xnnpack,
            )
            logger.info("Species classifier initialized")
        except Exception:
            logger.exception("Failed to initialize species classifier -- running without classification")
            self._classifier = None

    # ------------------------------------------------------------------
    # Thread loops
    # ------------------------------------------------------------------

    def _camera_loop(self, cam_index: int, camera_id: int) -> None:
        """Capture frames from one camera, run motion detection."""
        camera = self._cameras[cam_index]
        frame_buf = self._frame_buffers[cam_index]
        motion_det = self._motion_detectors[cam_index]
        interval = 1.0 / max(camera.fps, 1)

        logger.info("Camera %d loop started", camera_id)
        logged_scale = False

        cam_cfg = self._camera_configs[cam_index]
        crop_cfg = self._config.detection
        # None unless the camera reads out more than the pipeline carries.
        out_size = (
            cam_cfg.resolution
            if cam_cfg.capture_resolution is not None
            and tuple(cam_cfg.capture_resolution) != tuple(cam_cfg.resolution)
            else None
        )

        while not self._stop_event.is_set():
            ok, capture = camera.read()
            if not ok:
                if not camera.is_running:
                    logger.info("Camera %d source exhausted", camera_id)
                    break
                time.sleep(interval)
                continue

            now = time.time()

            # Everything downstream -- clips, the species crop, the
            # stored row -- works from this one.  The capture frame
            # exists only long enough to cut detection windows out of
            # it, and the thumbnail is written from a patch of one of
            # those windows rather than from here.
            if out_size is None:
                frame = capture
                scale = (1.0, 1.0)
            else:
                frame = cv2.resize(
                    capture, out_size, interpolation=cv2.INTER_AREA
                )
                # Per axis.  The two resolutions need not share an aspect
                # ratio, and a single width-derived scale silently
                # stretched every box vertically -- see the note on
                # DetectionEvent.capture_scale.
                scale = (
                    capture.shape[1] / frame.shape[1],
                    capture.shape[0] / frame.shape[0],
                )
                if not logged_scale:
                    logger.info(
                        "Camera %d downscale %dx%d -> %dx%d, scale %.3f/%.3f",
                        camera_id,
                        capture.shape[1], capture.shape[0],
                        frame.shape[1], frame.shape[0],
                        scale[0], scale[1],
                    )
                    logged_scale = True

            frame_buf.push(frame, now)

            with self._stats_lock:
                self._stats["frames_captured"] += 1

            if not self._config.motion.enabled:
                self._enqueue_for_detection(frame, camera_id, now)
                continue

            regions = motion_det.detect(capture)
            if not regions:
                continue

            with self._stats_lock:
                self._stats["motion_events"] += 1

            crops: list[tuple[np.ndarray, int, int]] = []
            if crop_cfg.roi_crop:
                for win in plan_windows(
                    regions,
                    frame_width=capture.shape[1],
                    frame_height=capture.shape[0],
                    window=crop_cfg.roi_crop_window,
                    max_windows=crop_cfg.roi_crop_max_windows,
                ):
                    x, y, w, h = win.bounds
                    crops.append((capture[y : y + h, x : x + w].copy(), x, y))
                if crops:
                    with self._stats_lock:
                        self._stats["crop_windows"] += len(crops)

            self._enqueue_for_detection(
                frame, camera_id, now, crops=crops, capture_scale=scale
            )

        logger.info("Camera %d loop ended", camera_id)

    def _enqueue_for_detection(
        self,
        frame: np.ndarray,
        camera_id: int,
        timestamp: float,
        crops: list[tuple[np.ndarray, int, int]] | None = None,
        capture_scale: float = 1.0,
    ) -> None:
        """Push a frame into the detection queue, dropping if full."""
        event = DetectionEvent(
            timestamp=datetime.fromtimestamp(timestamp),
            camera_id=camera_id,
            frame=frame,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            stage="motion",
            crops=crops or [],
            capture_scale=capture_scale,
        )

        if self._detector is not None:
            try:
                self._detection_queue.put_nowait(event)
            except queue.Full:
                with self._stats_lock:
                    self._stats["dropped_frames"] += 1
        else:
            try:
                self._storage_queue.put_nowait(event)
            except queue.Full:
                with self._stats_lock:
                    self._stats["dropped_frames"] += 1

    def _detection_loop(self) -> None:
        """Run object detection on frames from the detection queue."""
        logger.info("Detection loop started")

        while not self._stop_event.is_set():
            try:
                item = self._detection_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                break

            event: DetectionEvent = item

            try:
                detections = self._detect_for(event)
            except Exception:
                logger.exception("Detection error on camera %d", event.camera_id)
                continue

            if not detections:
                continue

            for det, detail, detail_bbox in detections:
                det_event = DetectionEvent(
                    timestamp=event.timestamp,
                    camera_id=event.camera_id,
                    frame=event.frame,
                    frame_width=event.frame_width,
                    frame_height=event.frame_height,
                    stage="detection",
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    detail=detail,
                    detail_bbox=detail_bbox,
                )

                with self._stats_lock:
                    self._stats["detections"] += 1

                if det_event.is_bird and self._classifier is not None:
                    try:
                        self._classification_queue.put_nowait(det_event)
                    except queue.Full:
                        try:
                            self._storage_queue.put_nowait(det_event)
                        except queue.Full:
                            with self._stats_lock:
                                self._stats["dropped_frames"] += 1
                else:
                    try:
                        self._storage_queue.put_nowait(det_event)
                    except queue.Full:
                        with self._stats_lock:
                            self._stats["dropped_frames"] += 1

        logger.info("Detection loop ended")

    def _detect_for(self, event: DetectionEvent) -> list:
        """Detect on an event, using native windows when they are present.

        Without windows this is the original whole-frame call.  With them
        the detector runs once per window and every box is translated out
        of window coordinates, through the capture frame, and onto
        ``event.frame`` -- which is what the classifier crop and the
        stored row use.

        Returns ``(detection, detail, detail_bbox)`` per hit.  The detail
        patch is cut here because this is the only place that holds both
        the native window and the box in window coordinates; carrying the
        windows onward instead is not an option, as ``storage_queue``
        holds 256 events and each window is 1.2 MB.  It is None for the
        whole-frame path, which has no native pixels to offer.
        """
        if not event.crops:
            return [(det, None, None) for det in self._detector.detect(event.frame)]

        scale_x, scale_y = event.capture_scale
        inv_x = 1.0 / scale_x if scale_x else 1.0
        inv_y = 1.0 / scale_y if scale_y else 1.0

        results = []
        for crop, off_x, off_y in event.crops:
            for det in self._detector.detect(crop):
                bx, by, bw, bh = det.bbox
                detail, detail_bbox = _cut_detail(crop, det.bbox)
                det.bbox = (
                    int(round((bx + off_x) * inv_x)),
                    int(round((by + off_y) * inv_y)),
                    int(round(bw * inv_x)),
                    int(round(bh * inv_y)),
                )
                results.append((det, detail, detail_bbox))
        return results

    def _classification_loop(self) -> None:
        """Run species classification on bird crops."""
        logger.info("Classification loop started")

        while not self._stop_event.is_set():
            try:
                item = self._classification_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                break

            event: DetectionEvent = item

            if event.bbox is None:
                self._storage_queue.put(event)
                continue

            x, y, w, h = event.bbox
            crop = event.frame[y:y + h, x:x + w]

            if crop.size == 0:
                self._storage_queue.put(event)
                continue

            try:
                result = self._classifier.classify(crop)
            except Exception:
                logger.exception(
                    "Classification error on camera %d", event.camera_id
                )
                self._storage_queue.put(event)
                continue

            if result is not None:
                event.stage = "classification"
                event.species = result.species
                event.common_name = result.common_name
                event.species_confidence = result.confidence
                event.top_k_species = result.top_k

                with self._stats_lock:
                    self._stats["classifications"] += 1

            try:
                self._storage_queue.put_nowait(event)
            except queue.Full:
                with self._stats_lock:
                    self._stats["dropped_frames"] += 1

        logger.info("Classification loop ended")

    def _storage_loop(self) -> None:
        """Persist detection events to SQLite and save clips/thumbnails."""
        logger.info("Storage loop started")

        thumb_dir = self._config.thumbnail_full_path

        while not self._stop_event.is_set():
            try:
                item = self._storage_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                break

            event: DetectionEvent = item

            if event.bbox is not None:
                thumb_name = (
                    f"{event.timestamp_iso.replace(':', '-')}"
                    f"_cam{event.camera_id}"
                    f"_{event.class_name or 'unknown'}.jpg"
                )
                thumb_path = thumb_dir / thumb_name
                # Native pixels around the animal when the detection came
                # from a window, the downscaled frame when it did not
                # (motion-only rows, or the ROI-crop path turned off).
                if event.detail is not None:
                    source, box = event.detail, event.detail_bbox
                else:
                    source, box = event.frame, event.bbox
                try:
                    create_thumbnail(source, box, thumb_path)
                    event.thumbnail_path = str(thumb_path)
                except Exception:
                    logger.exception("Failed to create thumbnail")

            if self._db is not None:
                try:
                    self._db.insert_detection(**event.to_db_kwargs())
                    with self._stats_lock:
                        self._stats["stored"] += 1
                except Exception:
                    logger.exception("Failed to store detection")
                else:
                    # Only events that identified something. When
                    # detection is disabled this loop also stores bare
                    # motion events, which name no animal and would bury
                    # the real sightings in a forwarded log.
                    if event.class_name is not None:
                        log_video_detection(
                            camera=event.camera_id,
                            class_name=event.class_name,
                            species=event.species,
                            common_name=event.common_name,
                            confidence=event.confidence,
                            species_confidence=event.species_confidence,
                            pest=event.is_pest,
                            clip_path=event.clip_path,
                            thumbnail_path=event.thumbnail_path,
                        )

        logger.info("Storage loop ended")

    def _stats_loop(self) -> None:
        """Log the pipeline counters every interval.

        The counters name the stage a frame died at, which no other
        record does: a motion event that yields no detection is dropped
        silently in ``_detection_loop``, so a pipeline seeing nothing and
        a pipeline whose gate rejects everything look identical in the
        log and in the database.  They used to print only from
        ``stop()``, so reading them meant stopping the cameras.

        On the ordinary pipeline logger rather than ``ratcatcher.events``
        deliberately.  That stream is kept to sightings and the status
        line so a loghost can carry it; this is diagnostic, and journald
        has it either way.
        """
        interval = self._config.monitoring.pipeline_stats_interval_seconds
        if interval <= 0:
            return

        logger.info("Pipeline stats every %.0fs", interval)

        previous = self.stats
        previous_at = time.monotonic()

        # wait() returns True once stop() sets the event, so shutdown
        # does not have to outlast a full interval.
        while not self._stop_event.wait(interval):
            now = time.monotonic()
            current = self.stats
            elapsed = now - previous_at

            # Aggregate across cameras, and measured rather than
            # configured: the gap between the two is the ISP downscale
            # in the camera loop.
            captured = current["frames_captured"] - previous["frames_captured"]
            fps = captured / elapsed if elapsed > 0 else 0.0

            logger.info(
                "pipeline fps=%.1f frames=%d motion=%d crops=%d "
                "detections=%d classifications=%d stored=%d dropped=%d",
                fps,
                current["frames_captured"],
                current["motion_events"],
                current["crop_windows"],
                current["detections"],
                current["classifications"],
                current["stored"],
                current["dropped_frames"],
            )

            previous = current
            previous_at = now
