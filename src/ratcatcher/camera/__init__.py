"""Camera abstraction layer for RatCatcher AI."""

from ratcatcher.camera.capture import CameraSource, FileSource, WebcamSource
from ratcatcher.camera.frame_buffer import FrameBuffer
from ratcatcher.camera.platform_camera import create_camera

__all__ = [
    "CameraSource",
    "FileSource",
    "WebcamSource",
    "FrameBuffer",
    "create_camera",
]
