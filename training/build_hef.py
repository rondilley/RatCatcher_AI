#!/usr/bin/env python3
"""RatCatcher AI -- Hailo HEF Compiler

Compiles models/ratcatcher_best.onnx into a .hef that runs the custom
5-class RatCatcher detector on the Hailo NPU.

RUN THIS ON AN x86-64 UBUNTU MACHINE, NOT THE PI.  The Hailo Dataflow
Compiler is distributed as an x86-64 Linux wheel for Python 3.8-3.11 and
has no aarch64 build.  Nothing in this file works on the Raspberry Pi.
See scripts/build_hef.sh for the environment checks and docs/DEPLOYMENT.md
for how to get the DFC.

WHY THIS IS NEEDED AT ALL

The Hailo Model Zoo publishes a prebuilt yolov8n.hef, but it is stock
COCO: 80 classes, none of which are squirrel or rat.  On that HEF the
NPU can only ever report bird and cat, which defeats the point of the
project.  The custom-trained 5-class model has to be compiled, and only
the DFC can do that.

THE GRAPH CUT

The ONNX exported by Ultralytics ends with the YOLOv8 decode tail --
DFL softmax, a strided conv, slices, and concats (nodes /model.22/dfl/*
through /model.22/Concat_3).  Those ops are cheap on a CPU and awkward
on a dataflow accelerator, and compiling them wastes NPU resources on
arithmetic that HailoRT's own post-process does better.

So the graph is cut at the six convolutions that feed the decode -- the
per-scale box-regression and classification heads -- and the decode plus
NMS is reattached as a HailoRT post-process via nms_postprocess().  The
resulting HEF emits HAILO_NMS_BY_CLASS, which HailoDetector already
parses (see _parse_nms_output).

Cut points, verified against models/ratcatcher_best.onnx:

    stride 8   /model.22/cv2.0/cv2.0.2/Conv   (box, 64ch)
               /model.22/cv3.0/cv3.0.2/Conv   (cls,  5ch)
    stride 16  /model.22/cv2.1/cv2.1.2/Conv   (box, 64ch)
               /model.22/cv3.1/cv3.1.2/Conv   (cls,  5ch)
    stride 32  /model.22/cv2.2/cv2.2.2/Conv   (box, 64ch)
               /model.22/cv3.2/cv3.2.2/Conv   (cls,  5ch)

The nms_postprocess config needs the layer names as the DFC assigns them
AFTER translation (conv41, conv52, ...), not the ONNX names above, and
those depend on the translator's numbering.  Rather than hardcode names
that would silently be wrong if anything about the model changed, this
script reads the translated network and derives the pairing from the
output shapes: 640/height gives the stride, a 4*regression_length
channel count marks the box head, and a num_classes channel count marks
the classification head.

NORMALIZATION ON-CHIP

Ultralytics models expect input in 0-1, but HailoDetector feeds uint8
0-255 to keep PCIe traffic down.  A normalization layer with mean 0 and
std 255 is compiled into the HEF so the division happens on the NPU.
Change one without the other and every detection score collapses.

Usage:
    python training/build_hef.py --hw-arch hailo8
    python training/build_hef.py --hw-arch hailo8l --calib models/calibration_set.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The five RatCatcher classes, in the order the detector expects.  Kept
# as a literal rather than imported from ratcatcher.detection.detector so
# this script runs on a bare DFC machine with no project install.  If you
# change RATCATCHER_CLASSES, change this too -- build_hef.sh cross-checks
# the count against the ONNX output channels and will fail loudly.
RATCATCHER_CLASSES = ["bird", "squirrel", "rat", "cat", "unknown_animal"]

# YOLOv8 DFL bins per box side.  4 sides * 16 = 64 channels in each
# box-regression head, which is how those heads are identified below.
REGRESSION_LENGTH = 16

ONNX_END_NODES = [
    "/model.22/cv2.0/cv2.0.2/Conv",
    "/model.22/cv3.0/cv3.0.2/Conv",
    "/model.22/cv2.1/cv2.1.2/Conv",
    "/model.22/cv3.1/cv3.1.2/Conv",
    "/model.22/cv2.2/cv2.2.2/Conv",
    "/model.22/cv3.2/cv3.2.2/Conv",
]

ONNX_INPUT_NODE = "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile ratcatcher_best.onnx to a Hailo HEF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="models/ratcatcher_best.onnx",
        help="Input ONNX model.",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default="models/calibration_set.npy",
        help="Calibration array from training/build_calibration_set.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ratcatcher_best.hef",
        help="Destination .hef file.",
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        choices=["hailo8", "hailo8l"],
        help=(
            "Target NPU. Read it off 'hailortcli fw-control identify' on "
            "the Pi. A hailo8l HEF runs on a Hailo-8; a hailo8 HEF will "
            "NOT load on a Hailo-8L."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square network input size. Must match the calibration set.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.2,
        help=(
            "On-chip NMS score threshold. Keep this LOOSER than the "
            "runtime confidence_threshold in config/default.yaml (0.45); "
            "the host filters again and cannot recover boxes dropped here."
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="On-chip NMS IoU threshold.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=100,
        help="Maximum surviving boxes per class.",
    )
    return parser.parse_args()


def require_dfc():
    """Import the DFC, failing with an actionable message if it is absent."""
    try:
        from hailo_sdk_client import ClientRunner
    except ImportError as exc:
        print("[ERROR] The Hailo Dataflow Compiler is not installed.")
        print(f"[ERROR]   {exc}")
        print("[ERROR]")
        print("[ERROR] The DFC is x86-64 Linux only (Python 3.8-3.11) and is")
        print("[ERROR] NOT on PyPI. Download the wheel from the Hailo")
        print("[ERROR] Developer Zone (free account) and install it into a")
        print("[ERROR] venv on an x86 Ubuntu machine:")
        print("[ERROR]   https://hailo.ai/developer-zone/software-downloads/")
        print("[ERROR]")
        print("[ERROR] This cannot be made to work on the Raspberry Pi.")
        sys.exit(1)
    return ClientRunner


def strip_net_prefix(layer_name: str) -> str:
    """Drop a leading '<network>/' from a HN layer name.

    The DFC accepts either form in an NMS config, but the Model Zoo
    configs use the bare name and matching them keeps diffs readable.
    """
    return layer_name.split("/", 1)[1] if "/" in layer_name else layer_name


def load_hn(runner) -> dict:
    """Return the translated network as a plain dict.

    The DFC has moved this accessor between releases -- ``get_hn_dict()``
    in some, ``get_hn()`` returning either a dict or a JSON string in
    others.  Trying each in turn keeps the script working across the DFC
    versions a user is likely to have, and fails with a clear message
    rather than an AttributeError if none of them exist.
    """
    for accessor in ("get_hn_dict", "get_hn"):
        method = getattr(runner, accessor, None)
        if method is None:
            continue
        hn = method()
        if isinstance(hn, str):
            hn = json.loads(hn)
        if isinstance(hn, dict):
            return hn

    raise RuntimeError(
        "This DFC exposes neither get_hn_dict() nor get_hn() returning a "
        "dict. Inspect the translated network manually and fill in the "
        "bbox_decoders by hand."
    )


def discover_head_layers(
    hn: dict, imgsz: int, num_classes: int
) -> list[dict]:
    """Derive the nms_postprocess bbox_decoders from the translated network.

    Walks the HN's output layers back to the convolutions that feed them,
    then classifies each by channel count and spatial size:

      * channels == 4 * REGRESSION_LENGTH  -> box regression head
      * channels == num_classes            -> classification head
      * stride                             -> imgsz / feature-map height

    Returns one decoder dict per stride, ascending.

    Raising here rather than guessing is deliberate: a wrong pairing
    compiles cleanly and produces a HEF that returns garbage boxes, which
    is far more expensive to diagnose than a failed build.
    """
    layers = hn.get("layers", {})
    if not layers:
        raise RuntimeError("Translated network has no 'layers' section.")

    reg_channels = 4 * REGRESSION_LENGTH

    # Output layers are pass-through markers; the real conv is their input.
    by_stride: dict[int, dict[str, str]] = {}
    for name, spec in layers.items():
        if spec.get("type") != "output_layer":
            continue

        inputs = spec.get("input", [])
        if not inputs:
            raise RuntimeError(f"Output layer '{name}' has no input layer.")
        conv_name = inputs[0]

        conv = layers.get(conv_name)
        if conv is None:
            raise RuntimeError(
                f"Output layer '{name}' names input '{conv_name}', which is "
                f"not in the translated network."
            )

        shapes = conv.get("output_shapes") or []
        if not shapes or len(shapes[0]) != 4:
            raise RuntimeError(
                f"Layer '{conv_name}' has unusable output_shapes {shapes!r}; "
                f"expected one [batch, height, width, channels] entry."
            )

        _, height, _width, channels = shapes[0]
        if not height:
            raise RuntimeError(f"Layer '{conv_name}' has zero height.")

        stride = imgsz // int(height)
        slot = by_stride.setdefault(stride, {})

        if int(channels) == reg_channels:
            slot["reg_layer"] = strip_net_prefix(conv_name)
        elif int(channels) == num_classes:
            slot["cls_layer"] = strip_net_prefix(conv_name)
        else:
            raise RuntimeError(
                f"Layer '{conv_name}' has {channels} output channels, which "
                f"is neither {reg_channels} (box regression, "
                f"4*{REGRESSION_LENGTH}) nor {num_classes} (classes). Either "
                f"the model is not a {num_classes}-class YOLOv8, or the end "
                f"node list in this script no longer matches its graph."
            )

    if not by_stride:
        raise RuntimeError(
            "Found no output layers in the translated network. Check that "
            "the end node names still exist in the ONNX graph."
        )

    decoders = []
    for stride in sorted(by_stride):
        slot = by_stride[stride]
        missing = {"reg_layer", "cls_layer"} - set(slot)
        if missing:
            raise RuntimeError(
                f"Stride {stride} is missing {', '.join(sorted(missing))}. "
                f"Each scale needs one box head and one classification head; "
                f"got {slot!r}."
            )
        decoders.append({
            "name": f"bbox_decoder_{stride}",
            "stride": stride,
            "reg_layer": slot["reg_layer"],
            "cls_layer": slot["cls_layer"],
        })

    if len(decoders) != 3:
        raise RuntimeError(
            f"Expected 3 detection scales (strides 8, 16, 32); found "
            f"{len(decoders)}: {[d['stride'] for d in decoders]}."
        )
    return decoders


def build_nms_config(args: argparse.Namespace, decoders: list[dict]) -> dict:
    """Assemble the YOLOv8 nms_postprocess configuration."""
    return {
        "nms_scores_th": args.score_threshold,
        "nms_iou_th": args.iou_threshold,
        "image_dims": [args.imgsz, args.imgsz],
        "max_proposals_per_class": args.max_per_class,
        "classes": len(RATCATCHER_CLASSES),
        "regression_length": REGRESSION_LENGTH,
        "background_removal": False,
        "background_removal_index": 0,
        "bbox_decoders": decoders,
    }


def build(args: argparse.Namespace) -> int:
    ClientRunner = require_dfc()

    onnx_path = Path(args.onnx)
    calib_path = Path(args.calib)
    output_path = Path(args.output)
    num_classes = len(RATCATCHER_CLASSES)

    if not onnx_path.is_file():
        print(f"[ERROR] ONNX model not found: {onnx_path.resolve()}")
        print("[ERROR] models/*.onnx is gitignored, so it does not arrive")
        print("[ERROR] with a git clone. Copy it from the Pi:")
        print("[ERROR]   scp pi:RatCatcher_AI/models/ratcatcher_best.onnx models/")
        return 1

    if not calib_path.is_file():
        print(f"[ERROR] Calibration set not found: {calib_path.resolve()}")
        print("[ERROR] Build it from the training images:")
        print("[ERROR]   python training/build_calibration_set.py")
        return 1

    # --- Load and sanity-check the calibration set ---------------------
    # np.load on a foreign file is a trust boundary; allow_pickle stays
    # off so a malformed .npy cannot execute code.
    try:
        calib_data = np.load(calib_path, allow_pickle=False)
    except (ValueError, OSError) as exc:
        print(f"[ERROR] Could not read {calib_path}: {exc}")
        return 1

    expected_shape = (args.imgsz, args.imgsz, 3)
    if calib_data.ndim != 4 or calib_data.shape[1:] != expected_shape:
        print(
            f"[ERROR] Calibration set has shape {calib_data.shape}; expected "
            f"(N, {args.imgsz}, {args.imgsz}, 3)."
        )
        print("[ERROR] Rebuild it with a matching --imgsz.")
        return 1

    if calib_data.max() <= 1:
        print("[ERROR] Calibration values are all <= 1, so the set looks")
        print("[ERROR] pre-scaled to 0-1. This HEF normalizes on-chip and")
        print("[ERROR] needs raw 0-255 uint8. Rebuild with")
        print("[ERROR] training/build_calibration_set.py.")
        return 1

    print(f"[INFO] Calibration set: {calib_data.shape} {calib_data.dtype}")
    print(f"[INFO] Target architecture: {args.hw_arch}")
    print("")

    model_name = onnx_path.stem
    runner = ClientRunner(hw_arch=args.hw_arch)

    # --- 1. Translate ONNX, cutting off the decode tail ----------------
    print(f"[1/4] Translating {onnx_path.name} (cutting at the detect head)...")
    for node in ONNX_END_NODES:
        print(f"[1/4]   end node: {node}")
    try:
        runner.translate_onnx_model(
            str(onnx_path),
            model_name,
            start_node_names=[ONNX_INPUT_NODE],
            end_node_names=ONNX_END_NODES,
            net_input_shapes={
                ONNX_INPUT_NODE: [1, 3, args.imgsz, args.imgsz]
            },
        )
    except Exception as exc:
        print(f"[ERROR] Translation failed: {exc}")
        print("[ERROR]")
        print("[ERROR] If this complains about a missing node, the ONNX was")
        print("[ERROR] exported from a different YOLOv8 version and the head")
        print("[ERROR] is not at model.22. List the real node names with:")
        print("[ERROR]   python -c \"import onnx; m=onnx.load('%s');"
              " print([n.name for n in m.graph.node][-45:])\"" % onnx_path)
        return 1
    print("[1/4] OK")
    print("")

    # --- 2. Derive the NMS config from the translated network ----------
    print("[2/4] Discovering detect-head layers...")
    try:
        hn = load_hn(runner)
        decoders = discover_head_layers(hn, args.imgsz, num_classes)
    except (RuntimeError, ValueError, KeyError, AttributeError) as exc:
        print(f"[ERROR] Could not derive the NMS configuration: {exc}")
        return 1

    for dec in decoders:
        print(
            f"[2/4]   stride {dec['stride']:>2}: "
            f"box={dec['reg_layer']}  cls={dec['cls_layer']}"
        )

    nms_config = build_nms_config(args, decoders)
    nms_config_path = output_path.with_name(f"{model_name}_nms_config.json")
    nms_config_path.parent.mkdir(parents=True, exist_ok=True)
    nms_config_path.write_text(json.dumps(nms_config, indent=4))
    print(f"[2/4] Wrote {nms_config_path}")
    print("")

    # --- 3. Quantize -----------------------------------------------------
    # normalization() bakes the 0-255 -> 0-1 division into the HEF, which
    # is what lets HailoDetector send uint8 over PCIe.  nms_postprocess()
    # reattaches the decode and NMS that the graph cut removed.
    alls = "\n".join([
        "normalization_layer = normalization([0.0, 0.0, 0.0], "
        "[255.0, 255.0, 255.0])",
        f'nms_postprocess("{nms_config_path}", meta_arch=yolov8, engine=cpu)',
        "",
    ])
    print("[3/4] Model script:")
    for line in alls.strip().splitlines():
        print(f"[3/4]   {line}")

    print(f"[3/4] Quantizing on {len(calib_data)} images (this takes a while)...")
    try:
        runner.load_model_script(alls)
        runner.optimize(calib_data)
    except Exception as exc:
        print(f"[ERROR] Quantization failed: {exc}")
        return 1
    print("[3/4] OK")
    print("")

    # --- 4. Compile ------------------------------------------------------
    print(f"[4/4] Compiling for {args.hw_arch} (the slow step, minutes)...")
    try:
        hef = runner.compile()
    except Exception as exc:
        print(f"[ERROR] Compilation failed: {exc}")
        return 1

    output_path.write_bytes(hef)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[4/4] OK")
    print("")
    print(f"[OK] Wrote {output_path} ({size_mb:.1f} MB)")
    print("")
    print("Copy it to the Pi and point the config at it:")
    print(f"  scp {output_path} pi:RatCatcher_AI/models/")
    print("  # config/default.yaml -> detection.model_path: "
          "\"ratcatcher_best.onnx\"")
    print("  #   (the factory swaps the suffix for .hef)")
    print("")
    print("Then verify on the Pi that it reports 5 classes, not 80:")
    print(f"  hailortcli parse-hef models/{output_path.name}")
    return 0


def main() -> int:
    return build(parse_args())


if __name__ == "__main__":
    sys.exit(main())
