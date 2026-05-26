"""Storage layer -- database, clips, thumbnails, and retention."""

from ratcatcher.storage.database import DetectionDatabase
from ratcatcher.storage.clip_writer import ClipWriter
from ratcatcher.storage.thumbnail import create_thumbnail
from ratcatcher.storage.retention import RetentionManager

__all__ = [
    "DetectionDatabase",
    "ClipWriter",
    "create_thumbnail",
    "RetentionManager",
]
