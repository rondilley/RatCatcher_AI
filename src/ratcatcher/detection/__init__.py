"""Object detection subsystem for RatCatcher AI."""

from ratcatcher.detection.detector import (
    Detection,
    ObjectDetector,
    RATCATCHER_CLASSES,
    map_coco_class,
)
from ratcatcher.detection.factory import create_detector

__all__ = [
    "Detection",
    "ObjectDetector",
    "RATCATCHER_CLASSES",
    "create_detector",
    "map_coco_class",
]
