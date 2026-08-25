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
#   1. Ubuntu 20.04 or 22.04, x86-64
#   2. Python 3.8-3.11 (NOT 3.12+ -- the DFC has no wheel for it)
#   3. The Hailo Dataflow Compiler wheel, from a free account at
#        https://hailo.ai/developer-zone/software-downloads/
#      Match the DFC major version to the HailoRT on the Pi (4.23):
#        python3 -m venv dfc-venv
#        ./dfc-venv/bin/pip install hailo_dataflow_compiler-*.whl
#   4. models/ratcatcher_best.onnx -- gitignored, so copy it from the Pi:
#        scp pi:RatCatcher_AI/models/ratcatcher_best.onnx models/
#   5. Calibration images. Either copy the dataset from the Pi:
#        rsync -a pi:RatCatcher_AI/datasets/ datasets/
#      or re-fetch it here:
#        python training/download_data.py
#      or copy a prebuilt calibration array and skip the build step:
#        scp pi:RatCatcher_AI/models/calibration_set.npy models/
#
# Usage:
#   ./scripts/build_hef.sh                 # defaults to hailo8l
#   HAILO_ARCH=hailo8 ./scripts/build_hef.sh
#   CALIB_COUNT=1024 ./scripts/build_hef.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# A hailo8l HEF runs on a Hailo-8, but a hailo8 HEF will NOT load on a
# Hailo-8L, so hailo8l is the safe default.  Read the real value off the
# Pi with: hailortcli fw-control identify
HAILO_ARCH="${HAILO_ARCH:-hailo8l}"
CALIB_COUNT="${CALIB_COUNT:-256}"
ONNX="${ONNX:-models/ratcatcher_best.onnx}"
CALIB="${CALIB:-models/calibration_set.npy}"
OUTPUT="${OUTPUT:-models/ratcatcher_best.hef}"
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
    --hw-arch "${HAILO_ARCH}"
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
