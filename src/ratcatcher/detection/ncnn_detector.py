"""NCNN detection backend for RatCatcher AI.

Uses the ``ncnn`` Python package to run YOLOv8 inference on CPU.  NCNN is
a high-performance neural-network inference framework optimised for mobile
and embedded devices, making it a good fit for Raspberry Pi when no
hardware accelerator is available.

If the ``ncnn`` package is not installed the module is still importable,
but ``NCNNDetector.__init__`` will raise ``ImportError`` with an
actionable message.
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

# -- Optional dependency guard ------------------------------------------------

_NCNN_AVAILABLE = False
try:
    import ncnn  # type: ignore[import-untyped]

    _NCNN_AVAILABLE = True
except ImportError:
    ncnn = None  # type: ignore[assignment]


class NCNNDetector:
    """YOLOv8 object detector using the NCNN inference engine.

    The constructor accepts the path to an ``.onnx`` model but expects
    converted ``.param`` and ``.bin`` files with the same stem to exist
    alongside it.  Use ``ncnn``'s ``onnx2ncnn`` tool to produce them.

    Parameters
    ----------
    model_path:
        Path to the original ``.onnx`` file.  The loader will resolve
        ``.param`` and ``.bin`` siblings automatically.
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
        if not _NCNN_AVAILABLE:
            raise ImportError(
                "The 'ncnn' Python package is required for the NCNN backend "
                "but was not found.  Install it with:  pip install ncnn"
            )

        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._input_size = input_size

        base = Path(model_path)
        self._param_path = base.with_suffix(".param")
        self._bin_path = base.with_suffix(".bin")

        if not self._param_path.exists():
            raise FileNotFoundError(
                f"NCNN param file not found: {self._param_path}  "
                f"(convert the ONNX model with onnx2ncnn first)"
            )
        if not self._bin_path.exists():
            raise FileNotFoundError(
                f"NCNN bin file not found: {self._bin_path}  "
                f"(convert the ONNX model with onnx2ncnn first)"
            )

        try:
            self._net = ncnn.Net()
            # Use Vulkan compute if available; harmless if not.
            if hasattr(self._net, "opt"):
                self._net.opt.use_vulkan_compute = False
                self._net.opt.num_threads = 4
            self._net.load_param(str(self._param_path))
            self._net.load_model(str(self._bin_path))
        except Exception as exc:
            raise RuntimeError(
                f"NCNN failed to load model "
                f"({self._param_path.name} / {self._bin_path.name}): {exc}"
            ) from exc

        logger.info(
            "NCNNDetector: loaded %s (input %dx%d, conf>=%.2f, nms>=%.2f)",
            self._param_path.stem,
            input_size[0],
            input_size[1],
            confidence_threshold,
            nms_threshold,
        )

    # -- ObjectDetector protocol ------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLOv8 detection on a BGR frame via NCNN.

        Parameters
        ----------
        frame:
            BGR image as a NumPy array of any resolution.

        Returns
        -------
        A list of ``Detection`` objects for recognised animal classes.
        """
        orig_h, orig_w = frame.shape[:2]
        inp_w, inp_h = self._input_size

        # --- 1. Preprocess: resize and build ncnn.Mat ---
        resized = cv2.resize(frame, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)

        # ncnn.Mat.from_pixels expects HWC uint8 with a pixel type flag.
        mat_in = ncnn.Mat.from_pixels(
            resized,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            inp_w,
            inp_h,
        )

        # Normalise to [0, 1].
        norm_values = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        mean_values = [0.0, 0.0, 0.0]
        mat_in.substract_mean_normalize(mean_values, norm_values)

        # --- 2. Run inference ---
        extractor = self._net.create_extractor()
        extractor.input("in0", mat_in)
        _ret, mat_out = extractor.extract("out0")

        # mat_out is a ncnn.Mat; convert to numpy.
        # Output shape for YOLOv8: (4+num_classes, N).  Transpose to (N, 4+C).
        output = np.array(mat_out)
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T

        predictions = output
        num_classes = predictions.shape[1] - 4
        custom_model = (num_classes == len(RATCATCHER_CLASSES))

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        max_class_ids = np.argmax(class_scores, axis=1)

        # --- 3. Confidence filter ---
        mask = max_scores >= self._confidence_threshold
        filtered_boxes = boxes_xywh[mask]
        filtered_scores = max_scores[mask]
        filtered_class_ids = max_class_ids[mask]

        if len(filtered_boxes) == 0:
            return []

        # Convert centre-format to top-left x, y, w, h for NMS.
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

        # --- 4. NMS (using OpenCV -- always available) ---
        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes.tolist(),
            scores=filtered_scores.tolist(),
            score_threshold=self._confidence_threshold,
            nms_threshold=self._nms_threshold,
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()

        # --- 5. Build detections ---
        scale_x = orig_w / inp_w
        scale_y = orig_h / inp_h

        results: list[Detection] = []
        for idx in indices:
            raw_class_id = int(filtered_class_ids[idx])

            if custom_model:
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
        return "ncnn"
