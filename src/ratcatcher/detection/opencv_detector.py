"""OpenCV DNN detection backend for RatCatcher AI.

Uses ``cv2.dnn.readNetFromONNX`` to load a YOLOv8-format ONNX model and
run inference entirely through OpenCV, so no additional dependencies
beyond opencv-python are required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ratcatcher.detection.detector import (
    Detection,
    RATCATCHER_CLASSES,
    map_coco_class,
)

logger = logging.getLogger(__name__)


class OpenCVDetector:
    """YOLOv8 object detector using the OpenCV DNN module.

    Parameters
    ----------
    model_path:
        Path to a YOLOv8 ``.onnx`` model file.
    confidence_threshold:
        Minimum class probability to keep a detection.
    nms_threshold:
        IoU threshold for non-maximum suppression.
    input_size:
        Network input resolution as ``(width, height)``.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._model_path = Path(model_path)
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._input_size = input_size

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self._model_path}"
            )

        try:
            self._net = cv2.dnn.readNetFromONNX(str(self._model_path))
        except cv2.error as exc:
            raise RuntimeError(
                f"OpenCV failed to load ONNX model {self._model_path}: {exc}"
            ) from exc

        # Prefer OpenCL if available; falls back to CPU transparently.
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Detect whether this is a COCO model (80 classes) or a custom
        # RatCatcher model (5 classes) by probing the output shape.
        # YOLOv8 output shape is (1, 4+num_classes, N).
        self._custom_model = False
        try:
            probe = np.zeros((1, 3, *self._input_size), dtype=np.float32)
            self._net.setInput(probe)
            probe_out = self._net.forward(
                self._net.getUnconnectedOutLayersNames()
            )
            num_outputs = probe_out[0].shape[1]  # 4 + num_classes
            num_classes = num_outputs - 4
            if num_classes == len(RATCATCHER_CLASSES):
                self._custom_model = True
                logger.info(
                    "OpenCVDetector: detected custom %d-class model",
                    num_classes,
                )
        except Exception:
            logger.debug("OpenCVDetector: output probe failed, assuming COCO")

        logger.info(
            "OpenCVDetector: loaded %s (input %dx%d, conf>=%.2f, nms>=%.2f, custom=%s)",
            self._model_path.name,
            input_size[0],
            input_size[1],
            confidence_threshold,
            nms_threshold,
            self._custom_model,
        )

    # -- ObjectDetector protocol ------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLOv8 detection on a BGR frame.

        Parameters
        ----------
        frame:
            BGR image as a NumPy array of any resolution.

        Returns
        -------
        A list of ``Detection`` objects for animal classes that survived
        confidence thresholding and NMS.
        """
        orig_h, orig_w = frame.shape[:2]
        inp_w, inp_h = self._input_size

        # --- 1. Preprocess: create a blob ---
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(inp_w, inp_h),
            swapRB=True,
            crop=False,
        )

        # --- 2. Forward pass ---
        self._net.setInput(blob)
        outputs = self._net.forward(self._net.getUnconnectedOutLayersNames())
        output = outputs[0]  # shape: (1, 84, N) for COCO YOLOv8

        # --- 3. Parse YOLOv8 output ---
        # Transpose from (1, 84, N) to (N, 84).  The first 4 columns are
        # cx, cy, w, h in input-image space; columns 4..83 are class scores.
        predictions = output[0].T  # (N, 84)

        boxes_xywh = predictions[:, :4]        # cx, cy, w, h
        class_scores = predictions[:, 4:]       # (N, 80) for COCO

        # Best class per proposal
        max_scores = np.max(class_scores, axis=1)
        max_class_ids = np.argmax(class_scores, axis=1)

        # --- 4. Confidence filter ---
        mask = max_scores >= self._confidence_threshold
        filtered_boxes = boxes_xywh[mask]
        filtered_scores = max_scores[mask]
        filtered_class_ids = max_class_ids[mask]

        if len(filtered_boxes) == 0:
            return []

        # Convert centre-format to top-left-format (x, y, w, h) for NMS,
        # still in network input coordinates.
        cx = filtered_boxes[:, 0]
        cy = filtered_boxes[:, 1]
        w = filtered_boxes[:, 2]
        h = filtered_boxes[:, 3]

        nms_boxes = np.column_stack([
            cx - w / 2.0,
            cy - h / 2.0,
            w,
            h,
        ])

        # --- 5. Non-Maximum Suppression ---
        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes.tolist(),
            scores=filtered_scores.tolist(),
            score_threshold=self._confidence_threshold,
            nms_threshold=self._nms_threshold,
        )

        if len(indices) == 0:
            return []

        # NMSBoxes returns a flat array or column vector depending on
        # OpenCV version; normalise to a flat list of ints.
        indices = np.array(indices).flatten()

        # --- 6. Build detections ---
        scale_x = orig_w / inp_w
        scale_y = orig_h / inp_h

        results: list[Detection] = []
        for idx in indices:
            raw_class_id = int(filtered_class_ids[idx])

            if self._custom_model:
                if raw_class_id >= len(RATCATCHER_CLASSES):
                    continue
                rc_id = raw_class_id
                rc_name = RATCATCHER_CLASSES[rc_id]
            else:
                mapped = map_coco_class(raw_class_id)
                if mapped is None:
                    continue
                rc_id, rc_name = mapped

            bx = nms_boxes[idx, 0]
            by = nms_boxes[idx, 1]
            bw = nms_boxes[idx, 2]
            bh = nms_boxes[idx, 3]

            # Scale to original frame coordinates and clamp.
            ox = int(round(bx * scale_x))
            oy = int(round(by * scale_y))
            ow = int(round(bw * scale_x))
            oh = int(round(bh * scale_y))

            ox = max(0, min(ox, orig_w - 1))
            oy = max(0, min(oy, orig_h - 1))
            ow = min(max(ow, 1), orig_w - ox)
            oh = min(max(oh, 1), orig_h - oy)

            results.append(Detection(
                class_id=rc_id,
                class_name=rc_name,
                confidence=float(filtered_scores[idx]),
                bbox=(ox, oy, ow, oh),
            ))

        return results

    def get_classes(self) -> list[str]:
        """Return the RatCatcher class list."""
        return list(RATCATCHER_CLASSES)

    @property
    def backend_name(self) -> str:
        return "opencv_dnn"
