"""Thumbnail generator with optional bounding-box overlay."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Bounding-box drawing parameters.
_BBOX_COLOR = (0, 255, 0)  # green in BGR
_BBOX_THICKNESS = 2


def create_thumbnail(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    output_path: Path,
    max_size: int = 320,
) -> Path:
    """Create a JPEG thumbnail from a video frame.

    Parameters
    ----------
    frame : np.ndarray
        Source image in BGR (OpenCV) format.
    bbox : tuple of (x, y, w, h) or None
        If provided, a green rectangle is drawn on the thumbnail.
    output_path : Path
        Destination file path for the JPEG.
    max_size : int
        The longest edge of the thumbnail in pixels (default 320).

    Returns
    -------
    Path
        The output path (same as the *output_path* argument).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Work on a copy so the caller's frame is unmodified.
    img = frame.copy()

    # Draw bounding box if supplied.
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), _BBOX_COLOR, _BBOX_THICKNESS)

    # Resize keeping aspect ratio so the longest edge equals max_size.
    src_h, src_w = img.shape[:2]
    if src_w >= src_h:
        scale = max_size / src_w
    else:
        scale = max_size / src_h

    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Encode and write.
    try:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, buf = cv2.imencode(".jpg", img, encode_params)
        if not success:
            logger.error("cv2.imencode failed for %s", output_path)
            raise IOError(f"Failed to encode JPEG for {output_path}")

        with open(output_path, "wb") as fh:
            fh.write(buf.tobytes())
    except OSError as exc:
        logger.error("Could not write thumbnail to %s: %s", output_path, exc)
        raise

    logger.debug("Thumbnail saved: %s (%dx%d)", output_path, new_w, new_h)
    return output_path
