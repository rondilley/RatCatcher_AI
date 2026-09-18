#!/usr/bin/env bash
# RatCatcher AI -- Build the custom detector HEF (x86 machine only)
#
# Compiles models/ratcatcher_best.onnx into a Hailo HEF so the custom
# 5-class detector runs on the AI HAT+ instead of stock COCO on the CPU.
#
# RUN THIS ON AN x86-64 UBUNTU MACHINE, NOT THE PI.
#
# The Hailo Dataflow Compiler ships as an x86-64 Linux wheel for Python
# 3.8-3.11 with no aarch64 build, so HEF compilation cannot happen on the
# Raspberry Pi.  This script checks the environment, builds the INT8
# calibration set, runs the compile, and tells you what to copy back.
#
# WHAT YOU NEED ON THE x86 BOX
#
#   1. x86-64 Linux. The builds here run on Ubuntu 26.04.
#   2. Python 3.8-3.11 for the DFC venv (NOT 3.12+ -- the DFC has no
#      wheel for it). This box uses python3.10 from ~/.local/bin.
#   3. The Hailo Dataflow Compiler wheel, from a free account at
#        https://hailo.ai/developer-zone/software-downloads/
#      Match the DFC major version to the HailoRT on the Pi (4.23):
#        python3.10 -m venv dfc-venv
#        ./dfc-venv/bin/pip install hailo_dataflow_compiler-*.whl
#      The DFC venv has no OpenCV. Step 1 below needs cv2, so either
#      build the calibration set first with the training venv, or
#        ./dfc-venv/bin/pip install opencv-python-headless
#   4. The ONNX to compile. On this box it comes from the training run:
#        runs/detect/runs/train/<run>/weights/best.onnx
#      models/*.onnx is gitignored, so a fresh clone must copy one in.
#   5. Calibration images: the train split of the composed dataset,
#      e.g. datasets/ratcatcher_v3/train/images, so the quantizer sees
#      the same domain the detector was trained on (infrared night
#      frames included). Or a prebuilt array, which skips the build step.
#
# Usage:
#   ./scripts/build_hef.sh                 # defaults to hailo8
#   HAILO_ARCH=hailo8l ./scripts/build_hef.sh
#   CALIB_COUNT=1024 ./scripts/build_hef.sh
#   HAR=models/ratcatcher_best_quantized.har ./scripts/build_hef.sh
#
# The build that made the deployed v3 HEF on 2026-09-17:
#   venv/bin/python3 training/build_calibration_set.py \
#       --images datasets/ratcatcher_v3/train/images \
#       --output models/v3/calibration_set.npy --count 256
#   PYTHON=./dfc-venv/bin/python HAILO_ARCH=hailo8 \
#       ONNX=models/v3/ratcatcher_best.onnx \
#       CALIB=models/v3/calibration_set.npy \
#       OUTPUT=models/v3/ratcatcher_best.hef \
#       HAR=models/v3/ratcatcher_best_quantized.har \
#       ./scripts/build_hef.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# The board here is a Hailo-8, confirmed with
# 'hailortcli fw-control identify' (Device Architecture: HAILO8), so
# hailo8 is the default.  An 8L HEF loads on a Hailo-8 and merely runs
# slower -- HailoRT says so on every load -- but the reverse does not
# hold: a hailo8 HEF will NOT load on a Hailo-8L.  Build with
# HAILO_ARCH=hailo8l for a board of the smaller kind.
HAILO_ARCH="${HAILO_ARCH:-hailo8}"
CALIB_COUNT="${CALIB_COUNT:-256}"
ONNX="${ONNX:-models/ratcatcher_best.onnx}"
CALIB="${CALIB:-models/calibration_set.npy}"
OUTPUT="${OUTPUT:-models/ratcatcher_best.hef}"
# Optional. When set, the quantized archive is kept beside the HEF so the
# INT8 accuracy cost can be measured on this machine with the DFC's CPU
# emulator, without a device.
HAR="${HAR:-}"
PYTHON="${PYTHON:-python3}"

echo "=== RatCatcher AI -- Hailo HEF Build ==="
echo "Architecture: ${HAILO_ARCH}"
echo "Python:       $(${PYTHON} -V 2>&1)"
echo ""

# ---- [0/3] Environment checks ----
echo "[0/3] Checking the environment..."

ARCH="$(uname -m)"
if [ "${ARCH}" != "x86_64" ]; then
    echo "ERROR: This machine is ${ARCH}, not x86_64."
    echo ""
    echo "The Hailo Dataflow Compiler is x86-64 Linux only. There is no"
    echo "aarch64 build, so a HEF cannot be compiled on the Raspberry Pi."
    echo "Run this on an x86 Ubuntu machine and copy the .hef back."
    exit 1
fi

if ! ${PYTHON} -c "import hailo_sdk_client" 2>/dev/null; then
    echo "ERROR: hailo_sdk_client is not importable by ${PYTHON}."
    echo ""
    echo "Install the Dataflow Compiler wheel (free Hailo Developer Zone"
    echo "account) and either activate that venv or point PYTHON at it:"
    echo "  PYTHON=./dfc-venv/bin/python ./scripts/build_hef.sh"
    exit 1
fi
echo "  OK  hailo_sdk_client importable"

# Step 1 decodes JPEGs with OpenCV, which the DFC venv does not ship.
# Say so here rather than after the environment checks have passed.
if [ ! -f "${CALIB}" ] && ! ${PYTHON} -c "import cv2" 2>/dev/null; then
    echo "ERROR: ${CALIB} does not exist and ${PYTHON} cannot import cv2,"
    echo "so the calibration set cannot be built with it."
    echo ""
    echo "Either build the calibration set first with the training venv:"
    echo "  venv/bin/python3 training/build_calibration_set.py \\"
    echo "      --images datasets/ratcatcher_v3/train/images --output ${CALIB}"
    echo "or install OpenCV into the DFC venv:"
    echo "  ${PYTHON} -m pip install opencv-python-headless"
    exit 1
fi

if [ ! -f "${ONNX}" ]; then
    echo "ERROR: ${ONNX} not found."
    echo ""
    echo "models/*.onnx is gitignored, so it does not arrive with a clone."
    echo "Copy it from the Pi:"
    echo "  scp pi:RatCatcher_AI/${ONNX} models/"
    exit 1
fi
echo "  OK  ${ONNX} ($(du -h "${ONNX}" | cut -f1))"
echo ""

# ---- [1/3] Calibration set ----
echo "[1/3] Calibration set..."
if [ -f "${CALIB}" ]; then
    echo "  Reusing existing ${CALIB} ($(du -h "${CALIB}" | cut -f1))"
    echo "  Delete it to force a rebuild."
else
    echo "  Building from training images (${CALIB_COUNT} samples)..."
    ${PYTHON} training/build_calibration_set.py \
        --count "${CALIB_COUNT}" \
        --output "${CALIB}"
fi
echo ""

# ---- [2/3] Compile ----
echo "[2/3] Compiling (quantize + compile, expect several minutes)..."
${PYTHON} training/build_hef.py \
    --onnx "${ONNX}" \
    --calib "${CALIB}" \
    --output "${OUTPUT}" \
    --hw-arch "${HAILO_ARCH}" \
    ${HAR:+--save-har "${HAR}"}
echo ""

# ---- [3/3] Report ----
echo "[3/3] Result..."
if [ ! -f "${OUTPUT}" ]; then
    echo "ERROR: ${OUTPUT} was not produced."
    exit 1
fi

# HEF files start with the magic bytes \x01HEF.
if ! head -c 4 "${OUTPUT}" | grep -q "HEF"; then
    echo "ERROR: ${OUTPUT} does not carry the HEF magic bytes."
    rm -f "${OUTPUT}"
    exit 1
fi

echo "  OK  ${OUTPUT} ($(du -h "${OUTPUT}" | cut -f1))"
echo ""
echo "=== Build complete ==="
echo ""
echo "Copy the HEF and its NMS config to the Pi:"
echo "  scp ${OUTPUT} pi:RatCatcher_AI/models/"
echo ""
echo "On the Pi, point the detector at it in config/default.yaml:"
echo "  detection:"
echo "    model_path: \"ratcatcher_best.onnx\""
echo ""
echo "The factory swaps .onnx for .hef on the Hailo path and keeps the"
echo ".onnx for the OpenCV DNN fallback, so one setting covers both."
echo ""
echo "Verify it reports 5 classes rather than COCO's 80:"
echo "  hailortcli parse-hef models/$(basename "${OUTPUT}")"
