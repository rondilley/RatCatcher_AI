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

import numpy as np

from ratcatcher.camera.capture import CameraSource
from ratcatcher.camera.frame_buffer import FrameBuffer
from ratcatcher.camera.platform_camera import create_camera
from ratcatcher.config import Config
from ratcatcher.monitoring.events import log_video_detection
from ratcatcher.motion.detector import MotionDetector
from ratcatcher.pipeline.event import DetectionEvent
from ratcatcher.storage.clip_writer import ClipWriter
from ratcatcher.storage.database import DetectionDatabase
from ratcatcher.storage.thumbnail import create_thumbnail

logger = logging.getLogger(__name__)

_SENTINEL = object()


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

        while not self._stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                if not camera.is_running:
                    logger.info("Camera %d source exhausted", camera_id)
                    break
                time.sleep(interval)
                continue

            now = time.time()
            frame_buf.push(frame, now)

            with self._stats_lock:
                self._stats["frames_captured"] += 1

            if not self._config.motion.enabled:
                self._enqueue_for_detection(frame, camera_id, now)
                continue

            regions = motion_det.detect(frame)
            if not regions:
                continue

            with self._stats_lock:
                self._stats["motion_events"] += 1

            self._enqueue_for_detection(frame, camera_id, now)

        logger.info("Camera %d loop ended", camera_id)

    def _enqueue_for_detection(
        self, frame: np.ndarray, camera_id: int, timestamp: float
    ) -> None:
        """Push a frame into the detection queue, dropping if full."""
        event = DetectionEvent(
            timestamp=datetime.fromtimestamp(timestamp),
            camera_id=camera_id,
            frame=frame,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            stage="motion",
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
                detections = self._detector.detect(event.frame)
            except Exception:
                logger.exception("Detection error on camera %d", event.camera_id)
                continue

            if not detections:
                continue

            for det in detections:
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
                try:
                    create_thumbnail(event.frame, event.bbox, thumb_path)
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
