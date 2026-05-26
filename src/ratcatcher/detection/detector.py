"""Object detection data structures, protocol, and COCO class mapping.

Defines the Detection dataclass returned by all detector backends, the
ObjectDetector protocol that each backend must implement, and the mapping
from COCO-80 class IDs to the reduced RatCatcher class set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# -- Data structures --------------------------------------------------------


@dataclass
class Detection:
    """A single object detection result.

    Attributes
    ----------
    class_id:
        Index into ``RATCATCHER_CLASSES``.
    class_name:
        Human-readable label from ``RATCATCHER_CLASSES``.
    confidence:
        Model confidence score in the range [0, 1].
    bbox:
        Bounding box as (x, y, width, height) in the original frame
        coordinate system.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h in original frame coords


# -- Detector protocol ------------------------------------------------------


class ObjectDetector(Protocol):
    """Interface every detection backend must satisfy."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame and return results."""
        ...

    def get_classes(self) -> list[str]:
        """Return the ordered list of class names this detector can emit."""
        ...

    @property
    def backend_name(self) -> str:
        """Short identifier for the active backend (e.g. 'opencv_dnn')."""
        ...


# -- COCO-to-RatCatcher class mapping ---------------------------------------

RATCATCHER_CLASSES = ["bird", "squirrel", "rat", "cat", "unknown_animal"]

# COCO class IDs that map to our reduced class set.
# Note: COCO has no dedicated squirrel or rat classes; those will be
# populated via the downstream species classifier.  The detection layer
# maps all COCO animal classes into one of these five buckets.
COCO_CLASS_MAP: dict[int, int] = {
    14: 0,   # bird -> bird
    15: 3,   # cat -> cat
    # Remaining COCO animal classes -> unknown_animal
    16: 4,   # dog
    17: 4,   # horse
    18: 4,   # sheep
    19: 4,   # cow
    20: 4,   # elephant
    21: 4,   # bear
    22: 4,   # zebra
    23: 4,   # giraffe
}


def map_coco_class(coco_class_id: int) -> tuple[int, str] | None:
    """Map a COCO class ID to a RatCatcher class.

    Parameters
    ----------
    coco_class_id:
        Zero-based COCO-80 class index.

    Returns
    -------
    A (class_id, class_name) tuple for recognised animal classes, or
    ``None`` if the COCO class is not an animal we track.
    """
    if coco_class_id in COCO_CLASS_MAP:
        rc_id = COCO_CLASS_MAP[coco_class_id]
        return rc_id, RATCATCHER_CLASSES[rc_id]
    return None
