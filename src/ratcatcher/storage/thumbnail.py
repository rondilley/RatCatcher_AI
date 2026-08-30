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
        If provided, a green rectangle is drawn on the thumbnail.  In
        *frame* coordinates; it is scaled with the image.
    output_path : Path
        Destination file path for the JPEG.
    max_size : int
        The longest edge of the thumbnail in pixels (default 320).  A
        frame already at or below this is written at its own size --
        upscaling would claim detail the camera did not record.

    Returns
    -------
    Path
        The output path (same as the *output_path* argument).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resize keeping aspect ratio so the longest edge equals max_size,
    # and never enlarge.
    src_h, src_w = frame.shape[:2]
    scale = min(1.0, max_size / max(src_w, src_h))

    if scale < 1.0:
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # Copy so the caller's frame is unmodified by the overlay.
        new_w, new_h = src_w, src_h
        img = frame.copy()

    # Draw the box after the resize, not before.  Drawn first, a 2 px
    # line on a 1920-wide frame survives a 6x reduction as a third of a
    # pixel of coverage -- a green smear rather than a rectangle.
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(
            img,
            (int(round(x * scale)), int(round(y * scale))),
            (int(round((x + w) * scale)), int(round((y + h) * scale))),
            _BBOX_COLOR,
            _BBOX_THICKNESS,
        )

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
