"""Hailo NPU detection backend for RatCatcher AI.

Uses the HailoRT Python API (``hailo_platform``) to run YOLOv8 inference
on a Hailo-8 or Hailo-8L accelerator attached to a Raspberry Pi.  This
backend provides the lowest-latency, lowest-power detection path.

If ``hailo_platform`` is not installed (i.e. on non-RPi development
machines) the module is still importable, but ``HailoDetector.__init__``
will raise ``ImportError`` with an actionable message.
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

_HAILO_AVAILABLE = False
try:
    from hailo_platform import (  # type: ignore[import-untyped]
        HEF,
        ConfigureParams,
        FormatType,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )

    _HAILO_AVAILABLE = True
except ImportError:
    HEF = None  # type: ignore[assignment,misc]
    VDevice = None  # type: ignore[assignment,misc]


class HailoDetector:
    """YOLOv8 object detector running on a Hailo NPU.

    Parameters
    ----------
    model_path:
        Path to a compiled ``.hef`` model file.
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
        if not _HAILO_AVAILABLE:
            raise ImportError(
                "The 'hailo_platform' package is required for the Hailo NPU "
                "backend but was not found.  This package is only available "
                "on Raspberry Pi with the HailoRT SDK installed.  See: "
                "https://hailo.ai/developer-zone/"
            )

        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._input_size = input_size

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Hailo HEF model not found: {self._model_path}"
            )

        # --- Load HEF and configure the device ---
        try:
            self._hef = HEF(str(self._model_path))
            self._vdevice = VDevice()

            configure_params = ConfigureParams.create_from_hef(
                hef=self._hef,
                interface=HailoStreamInterface.PCIe,
            )
            self._network_group = self._vdevice.configure(
                self._hef, configure_params
            )[0]

            self._network_group_params = (
                self._network_group.create_params()
            )

            # Build stream parameters for input and output virtual streams.
            self._input_vstream_info = self._hef.get_input_vstream_infos()
            self._output_vstream_info = self._hef.get_output_vstream_infos()

            self._input_vstream_params = InputVStreamParams.make(
                self._network_group,
                format_type=FormatType.FLOAT32,
            )
            self._output_vstream_params = OutputVStreamParams.make(
                self._network_group,
                format_type=FormatType.FLOAT32,
            )

        except ImportError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise Hailo device with model "
                f"{self._model_path.name}: {exc}"
            ) from exc

        logger.info(
            "HailoDetector: loaded %s on Hailo NPU (input %dx%d, "
            "conf>=%.2f, nms>=%.2f)",
            self._model_path.name,
            input_size[0],
            input_size[1],
            confidence_threshold,
            nms_threshold,
        )

    # -- ObjectDetector protocol ------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLOv8 detection on a BGR frame via the Hailo NPU.

        If the Hailo device encounters a hardware error (overtemperature,
        disconnection, etc.) the error is logged and an empty list is
        returned so the pipeline can continue gracefully.

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

        # --- 1. Preprocess: resize, BGR->RGB, normalise, NHWC float32 ---
        resized = cv2.resize(
            frame, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # (1, H, W, 3)

        # --- 2. Inference ---
        try:
            input_name = self._input_vstream_info[0].name
            input_dict = {input_name: input_data}

            with InferVStreams(
                self._network_group,
                self._input_vstream_params,
                self._output_vstream_params,
            ) as pipeline:
                raw_results = pipeline.infer(input_dict)

        except Exception as exc:
            # Hardware errors (thermal throttle, device disconnect, etc.)
            # are logged and suppressed so the pipeline stays alive.
            logger.warning(
                "HailoDetector: inference failed (%s) -- returning no "
                "detections for this frame",
                exc,
            )
            return []

        # --- 3. Parse output ---
        # Collect the first (or only) output tensor.
        output_name = self._output_vstream_info[0].name
        output = raw_results[output_name]  # (1, 84, N) or (1, N, 84)

        if output.ndim == 3:
            output = output[0]
        # Ensure shape is (N, 4+num_classes).
        if output.shape[0] < output.shape[1]:
            output = output.T

        predictions = output
        num_classes = predictions.shape[1] - 4
        custom_model = (num_classes == len(RATCATCHER_CLASSES))

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        max_class_ids = np.argmax(class_scores, axis=1)

        # --- 4. Confidence filter ---
        mask = max_scores >= self._confidence_threshold
        filtered_boxes = boxes_xywh[mask]
        filtered_scores = max_scores[mask]
        filtered_class_ids = max_class_ids[mask]

        if len(filtered_boxes) == 0:
            return []

        # Centre-format to top-left x, y, w, h.
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

        # --- 5. NMS ---
        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes.tolist(),
            scores=filtered_scores.tolist(),
            score_threshold=self._confidence_threshold,
            nms_threshold=self._nms_threshold,
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()

        # --- 6. Build detections ---
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
        return "hailo"
