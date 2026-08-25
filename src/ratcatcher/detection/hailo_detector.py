"""Hailo NPU detection backend for RatCatcher AI.

Uses the HailoRT Python API (``hailo_platform``) to run YOLOv8 inference
on a Hailo-8 or Hailo-8L accelerator attached to a Raspberry Pi.  This
backend provides the lowest-latency, lowest-power detection path.

Two flavours of compiled model are supported, distinguished automatically
from the HEF's output vstream format:

* **NMS baked in** (``HAILO_NMS_BY_CLASS``) -- what the Hailo Model Zoo
  ships for stock YOLOv8.  Box decoding, score thresholding and NMS all
  run on-chip; the host receives one array of surviving boxes per class.
* **Raw tensor** (``NHWC`` / ``FCR`` etc.) -- what ``hailo compiler``
  produces when the YOLO head is left unfused, e.g. a custom-trained
  RatCatcher model exported via ``training/export_model.py``.  The host
  decodes and runs NMS itself.

If ``hailo_platform`` is not installed (i.e. on non-RPi development
machines) the module is still importable, but ``HailoDetector.__init__``
will raise ``ImportError`` with an actionable message.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
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
        FormatOrder,
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
    FormatOrder = None  # type: ignore[assignment,misc]


# Output vstream formats where the NMS post-process runs on the NPU.
_NMS_ORDER_NAMES = frozenset({
    "HAILO_NMS_BY_CLASS",
    "HAILO_NMS_BY_SCORE",
    "HAILO_NMS_ON_CHIP",
    "HAILO_NMS_WITH_BYTE_MASK",
})


class HailoDetector:
    """YOLOv8 object detector running on a Hailo NPU.

    The network group is activated once and the inference vstreams are
    held open for the lifetime of the detector -- re-creating them per
    frame costs roughly 1.4 ms/frame.  Call :meth:`close` (or use the
    detector as a context manager) to release the device.

    The accelerator exposes a single ``VDevice``, so only one detector
    may hold it at a time; constructing a second one before the first is
    closed fails with ``HAILO_OUT_OF_PHYSICAL_DEVICES``.  Share one
    instance across camera pipelines rather than building one per stream.

    Parameters
    ----------
    model_path:
        Path to a compiled ``.hef`` model file.
    confidence_threshold:
        Minimum class probability to keep a detection.  Applied on the
        host, on top of any score threshold already compiled into the
        HEF's on-chip NMS.
    nms_threshold:
        IoU threshold for non-maximum suppression.  Only used for HEFs
        that emit a raw tensor; ignored when NMS runs on-chip.
    input_size:
        Network input resolution as ``(width, height)``.  The HEF is
        authoritative -- this is only a fallback if the HEF's input
        shape cannot be read, and a mismatch is logged.
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
                "backend but was not found.  It ships as the Debian package "
                "'python3-hailort' (sudo apt install hailo-all) and is not "
                "available on PyPI, so a virtualenv only sees it when created "
                "with --system-site-packages.  See: "
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

        self._stack = ExitStack()
        self._closed = False

        # --- Load HEF and configure the device ---
        try:
            self._hef = HEF(str(self._model_path))
            self._vdevice = VDevice()
            self._stack.callback(self._vdevice.release)

            configure_params = ConfigureParams.create_from_hef(
                hef=self._hef,
                interface=HailoStreamInterface.PCIe,
            )
            self._network_group = self._vdevice.configure(
                self._hef, configure_params
            )[0]

            self._input_vstream_info = self._hef.get_input_vstream_infos()
            self._output_vstream_info = self._hef.get_output_vstream_infos()
            self._input_name = self._input_vstream_info[0].name
            self._output_name = self._output_vstream_info[0].name

            # The HEF dictates the real input resolution; trust it over config.
            hef_shape = self._input_vstream_info[0].shape  # (H, W, C)
            if len(hef_shape) == 3:
                hef_size = (int(hef_shape[1]), int(hef_shape[0]))  # (W, H)
                if hef_size != self._input_size:
                    logger.warning(
                        "HailoDetector: configured input_size %dx%d does not "
                        "match HEF input %dx%d -- using the HEF's size",
                        self._input_size[0], self._input_size[1],
                        hef_size[0], hef_size[1],
                    )
                self._input_size = hef_size

            # Does this HEF run NMS on-chip?
            out_order = self._output_vstream_info[0].format.order
            self._nms_on_chip = (
                getattr(out_order, "name", str(out_order)) in _NMS_ORDER_NAMES
            )

            # For an on-chip-NMS HEF the class count is baked into the
            # output vstream, so read it rather than inferring it from a
            # tensor shape at inference time.  It decides whether the
            # emitted class IDs are COCO-80 indices or direct indices into
            # RATCATCHER_CLASSES -- the same custom-vs-COCO distinction the
            # raw-tensor path makes from its channel count.  Without this,
            # a custom 5-class HEF emits IDs 0-4, none of which appear in
            # COCO_CLASS_MAP, and every detection is silently discarded.
            self._nms_num_classes: int | None = None
            if self._nms_on_chip:
                nms_shape = getattr(
                    self._output_vstream_info[0], "nms_shape", None
                )
                if nms_shape is not None:
                    self._nms_num_classes = int(nms_shape.number_of_classes)

            # The HEF's input layer is quantised UINT8.  Feeding UINT8
            # directly skips a host-side float conversion and quarters the
            # PCIe traffic.  (Requesting FLOAT32 here would also require
            # feeding values in 0-255, NOT 0-1 -- HailoRT applies the
            # quantisation scale itself.)
            self._input_vstream_params = InputVStreamParams.make(
                self._network_group,
                format_type=FormatType.UINT8,
            )
            self._output_vstream_params = OutputVStreamParams.make(
                self._network_group,
                format_type=FormatType.FLOAT32,
            )

            # Activate the network group and open the vstreams once.  Writing
            # to a vstream before activation raises
            # HailoRTNetworkGroupNotActivatedException on every frame.
            self._stack.enter_context(
                self._network_group.activate(self._network_group.create_params())
            )
            self._pipeline = self._stack.enter_context(
                InferVStreams(
                    self._network_group,
                    self._input_vstream_params,
                    self._output_vstream_params,
                )
            )

        except ImportError:
            self._stack.close()
            raise
        except Exception as exc:
            self._stack.close()
            raise RuntimeError(
                f"Failed to initialise Hailo device with model "
                f"{self._model_path.name}: {exc}"
            ) from exc

        logger.info(
            "HailoDetector: loaded %s on Hailo NPU (input %dx%d, "
            "conf>=%.2f, nms=%s, classes=%s)",
            self._model_path.name,
            self._input_size[0],
            self._input_size[1],
            confidence_threshold,
            "on-chip" if self._nms_on_chip else f"host (IoU>={nms_threshold:.2f})",
            # A stock Model Zoo HEF reports 80 (COCO, no squirrel or rat
            # class); the custom RatCatcher HEF reports 5.  Worth stating
            # outright, because the two load identically and differ only
            # in what they can possibly detect.
            f"{self._nms_num_classes} (custom)"
            if self._nms_num_classes == len(RATCATCHER_CLASSES)
            else f"{self._nms_num_classes} (COCO)"
            if self._nms_num_classes is not None
            else "raw tensor, determined per frame",
        )

    # -- Lifecycle --------------------------------------------------------------

    def close(self) -> None:
        """Release the vstreams, network group activation, and device."""
        if not self._closed:
            self._closed = True
            self._stack.close()

    def __enter__(self) -> HailoDetector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

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
        if self._closed:
            raise RuntimeError("HailoDetector has been closed")

        orig_h, orig_w = frame.shape[:2]
        inp_w, inp_h = self._input_size

        # --- 1. Preprocess: resize, BGR->RGB, NHWC uint8 ---
        resized = cv2.resize(
            frame, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.ascontiguousarray(
            np.expand_dims(rgb, axis=0), dtype=np.uint8
        )  # (1, H, W, 3)

        # --- 2. Inference ---
        try:
            raw_results = self._pipeline.infer({self._input_name: input_data})
        except Exception as exc:
            # Hardware errors (thermal throttle, device disconnect, etc.)
            # are logged and suppressed so the pipeline stays alive.
            logger.warning(
                "HailoDetector: inference failed (%s) -- returning no "
                "detections for this frame",
                exc,
            )
            return []

        output = raw_results[self._output_name]

        # --- 3. Parse output ---
        if self._nms_on_chip:
            return self._parse_nms_output(output, orig_w, orig_h)
        return self._parse_raw_output(output, orig_w, orig_h)

    # -- Output parsers ---------------------------------------------------------

    def _parse_nms_output(
        self, output: object, orig_w: int, orig_h: int
    ) -> list[Detection]:
        """Parse a HAILO_NMS_BY_CLASS output.

        The result is indexed ``[batch][class_id]`` and each entry is an
        ``(n, 5)`` float32 array whose rows are
        ``[y_min, x_min, y_max, x_max, score]`` in normalised (0-1) frame
        coordinates.  Boxes are already de-duplicated on-chip, so no host
        NMS is needed -- only the configured confidence filter, since the
        HEF's compiled-in score threshold is typically looser.

        ``class_id`` is a COCO-80 index for a stock Model Zoo HEF, or a
        direct index into ``RATCATCHER_CLASSES`` for a custom-compiled
        5-class one; the class count read from the HEF at load time tells
        the two apart.
        """
        custom_model = self._nms_num_classes == len(RATCATCHER_CLASSES)

        # Unwrap the batch dimension.
        per_class = output[0] if len(output) > 0 else []  # type: ignore[index]

        results: list[Detection] = []
        for class_id, boxes in enumerate(per_class):
            arr = np.asarray(boxes, dtype=np.float32)
            if arr.size == 0:
                continue

            if custom_model:
                if class_id >= len(RATCATCHER_CLASSES):
                    continue
                rc_id = class_id
                rc_name = RATCATCHER_CLASSES[rc_id]
            else:
                mapped = map_coco_class(class_id)
                if mapped is None:
                    continue
                rc_id, rc_name = mapped

            for row in arr.reshape(-1, arr.shape[-1]):
                score = float(row[4])
                if score < self._confidence_threshold:
                    continue

                # Normalised coords can sit marginally outside [0, 1].
                y0 = min(max(float(row[0]), 0.0), 1.0)
                x0 = min(max(float(row[1]), 0.0), 1.0)
                y1 = min(max(float(row[2]), 0.0), 1.0)
                x1 = min(max(float(row[3]), 0.0), 1.0)

                box = self._to_pixel_box(
                    x0 * orig_w, y0 * orig_h,
                    (x1 - x0) * orig_w, (y1 - y0) * orig_h,
                    orig_w, orig_h,
                )
                if box is None:
                    continue

                results.append(Detection(
                    class_id=rc_id,
                    class_name=rc_name,
                    confidence=score,
                    bbox=box,
                ))

        return results

    def _parse_raw_output(
        self, output: np.ndarray, orig_w: int, orig_h: int
    ) -> list[Detection]:
        """Parse a raw YOLOv8 head tensor, running NMS on the host.

        Used for custom-compiled HEFs whose detection head was left
        unfused -- shape ``(1, 84, N)`` or ``(1, N, 84)``.
        """
        output = np.asarray(output)
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

        # --- Confidence filter ---
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

        # --- NMS ---
        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes.tolist(),
            scores=filtered_scores.tolist(),
            score_threshold=self._confidence_threshold,
            nms_threshold=self._nms_threshold,
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()

        # --- Build detections ---
        inp_w, inp_h = self._input_size
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

            box = self._to_pixel_box(
                nms_boxes[idx, 0] * scale_x,
                nms_boxes[idx, 1] * scale_y,
                nms_boxes[idx, 2] * scale_x,
                nms_boxes[idx, 3] * scale_y,
                orig_w, orig_h,
            )
            if box is None:
                continue

            results.append(Detection(
                class_id=rc_id,
                class_name=rc_name,
                confidence=float(filtered_scores[idx]),
                bbox=box,
            ))

        return results

    @staticmethod
    def _to_pixel_box(
        x: float, y: float, w: float, h: float, orig_w: int, orig_h: int
    ) -> tuple[int, int, int, int] | None:
        """Round and clamp a box to integer pixel coords inside the frame."""
        ox = int(round(x))
        oy = int(round(y))
        ow = int(round(w))
        oh = int(round(h))

        ox = max(0, min(ox, orig_w - 1))
        oy = max(0, min(oy, orig_h - 1))
        ow = min(max(ow, 1), orig_w - ox)
        oh = min(max(oh, 1), orig_h - oy)

        if ow < 1 or oh < 1:
            return None
        return (ox, oy, ow, oh)

    def get_classes(self) -> list[str]:
        """Return the RatCatcher class list."""
        return list(RATCATCHER_CLASSES)

    @property
    def backend_name(self) -> str:
        return "hailo"
