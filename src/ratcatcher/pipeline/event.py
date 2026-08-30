"""Detection event data structures for the RatCatcher AI pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class DetectionEvent:
    """A single detection event produced by the pipeline.

    Carries the frame, all detection results, and optional species
    classification results through the pipeline into storage.
    """

    timestamp: datetime
    camera_id: int
    frame: np.ndarray
    frame_width: int
    frame_height: int

    # Stage that produced this event ("motion", "detection", "classification")
    stage: str = "motion"

    # Native-resolution windows cut around motion, each paired with the
    # offset it was cut from, in capture coordinates.  Present only when
    # the ROI-crop path is enabled.  ``frame`` stays the downscaled frame
    # the rest of the pipeline works from -- a full-resolution frame is
    # 37 MB and must not enter a queue -- so ``capture_scale`` is what maps
    # a box found in a crop back onto it.
    #
    # Two scales, not one: the capture and the frame need not share an
    # aspect ratio, and with the shipped config they do not (4056x3040
    # into 1920x1080 divides x by 2.113 and y by 2.815).  A single
    # width-derived scale made every box 1.33x too tall and put anything
    # below capture y=2282 off the bottom of the frame entirely.
    crops: list[tuple[np.ndarray, int, int]] = field(default_factory=list)
    capture_scale: tuple[float, float] = (1.0, 1.0)

    # Native-resolution pixels around ``bbox``, cut from the window the
    # detection was found in, at detection time.  The thumbnail is
    # written from this rather than from ``frame``: a House Finch is
    # about 62 px tall in the capture and 3.7 px in a 320-wide thumbnail
    # of the whole frame, which is smaller than one JPEG block.  None
    # for whole-frame and motion-only events, which fall back to
    # ``frame``.  Bounded by construction -- at most one window, and
    # normally 256x256 -- because ``crops`` itself must not travel this
    # far: ``storage_queue`` holds 256 events.
    detail: np.ndarray | None = None
    detail_bbox: tuple[int, int, int, int] | None = None

    # From Stage 1 (YOLO detection)
    class_name: str | None = None
    confidence: float | None = None
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h

    # From Stage 2 (species classification, birds only)
    species: str | None = None
    common_name: str | None = None
    species_confidence: float | None = None
    top_k_species: list[tuple[str, str, float]] = field(default_factory=list)

    # Storage paths (populated after clip/thumbnail writing)
    clip_path: str | None = None
    thumbnail_path: str | None = None

    @property
    def is_bird(self) -> bool:
        return self.class_name == "bird"

    @property
    def is_pest(self) -> bool:
        return self.class_name in ("squirrel", "rat", "cat", "unknown_animal")

    @property
    def timestamp_iso(self) -> str:
        return self.timestamp.isoformat()

    def to_db_kwargs(self) -> dict:
        """Convert to keyword arguments for DetectionDatabase.insert_detection."""
        return {
            "timestamp": self.timestamp_iso,
            "camera_id": self.camera_id,
            "stage": self.stage,
            "class_name": self.class_name,
            "species": self.species,
            "common_name": self.common_name,
            "confidence": self.species_confidence if self.species else self.confidence,
            "bbox_x": self.bbox[0] if self.bbox else None,
            "bbox_y": self.bbox[1] if self.bbox else None,
            "bbox_w": self.bbox[2] if self.bbox else None,
            "bbox_h": self.bbox[3] if self.bbox else None,
            "clip_path": self.clip_path,
            "thumbnail_path": self.thumbnail_path,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "metadata": {
                "top_k": [
                    {"species": s, "common": c, "conf": round(p, 4)}
                    for s, c, p in self.top_k_species
                ],
            } if self.top_k_species else None,
        }
