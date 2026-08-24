#!/usr/bin/env bash
# RatCatcher AI -- Model Download and Conversion Script
#
# Downloads YOLOv8n for object detection and prepares it for the
# configured inference backend.
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh

set -euo pipefail

MODELS_DIR="${1:-models}"
mkdir -p "${MODELS_DIR}"

echo "=== RatCatcher AI -- Model Download ==="
echo "Output directory: ${MODELS_DIR}"
echo ""

# ---- YOLOv8n (Object Detection) ----

YOLO_ONNX="${MODELS_DIR}/yolov8n.onnx"

if [ -f "${YOLO_ONNX}" ]; then
    echo "[SKIP] YOLOv8n ONNX already exists: ${YOLO_ONNX}"
else
    echo "[1/3] Downloading and exporting YOLOv8n to ONNX..."
    if command -v yolo &>/dev/null; then
        yolo export model=yolov8n.pt format=onnx imgsz=640
        mv yolov8n.onnx "${YOLO_ONNX}"
        echo "Exported YOLOv8n to ${YOLO_ONNX}"
    else
        echo "WARNING: ultralytics CLI not found."
        echo "Install with: pip install ultralytics"
        echo "Then run: yolo export model=yolov8n.pt format=onnx imgsz=640"
        echo "And move the result to ${YOLO_ONNX}"
    fi
fi

# ---- NCNN conversion (CPU fallback) ----

YOLO_PARAM="${MODELS_DIR}/yolov8n.param"
YOLO_BIN="${MODELS_DIR}/yolov8n.bin"

if [ -f "${YOLO_PARAM}" ] && [ -f "${YOLO_BIN}" ]; then
    echo "[SKIP] YOLOv8n NCNN already exists"
else
    echo "[2/3] Converting YOLOv8n to NCNN format..."
    if [ -f "${YOLO_ONNX}" ] && command -v onnx2ncnn &>/dev/null; then
        onnx2ncnn "${YOLO_ONNX}" "${YOLO_PARAM}" "${YOLO_BIN}"
        echo "Converted to NCNN: ${YOLO_PARAM}, ${YOLO_BIN}"
    else
        echo "SKIP: onnx2ncnn not found or ONNX model missing."
        echo "For NCNN backend, install ncnn tools and convert manually."
    fi
fi

# ---- Hailo HEF conversion (RPi production) ----

YOLO_HEF="${MODELS_DIR}/yolov8n.hef"

if [ -f "${YOLO_HEF}" ]; then
    echo "[SKIP] YOLOv8n HEF already exists"
else
    echo "[3/3] Fetching Hailo HEF..."
    # Stock YOLOv8n does NOT need the Hailo Dataflow Compiler -- Hailo
    # publishes prebuilt HEFs in the Model Zoo.  The DFC is only required
    # for a custom-trained model (see training/export_model.py).
    #
    # HEFs are compiled per architecture.  A hailo8l HEF runs on Hailo-8
    # hardware but only uses half the compute, so match the real device.
    HAILO_PCI="$(lspci 2>/dev/null | grep -i 'hailo' || true)"
    if echo "${HAILO_PCI}" | grep -qi 'hailo-8l'; then
        HAILO_ARCH="hailo8l"
    else
        HAILO_ARCH="hailo8"
    fi

    # Model Zoo v2.16.0 pairs with HailoRT 4.23 (the version in the
    # Raspberry Pi apt archive).  Bump both together.
    MODELZOO_VER="v2.16.0"
    HEF_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/${MODELZOO_VER}/${HAILO_ARCH}/yolov8n.hef"

    echo "Architecture: ${HAILO_ARCH} (Model Zoo ${MODELZOO_VER})"
    if curl -fsSL --max-time 300 -o "${YOLO_HEF}.tmp" "${HEF_URL}"; then
        # HEF files start with the magic bytes \x01HEF.
        if head -c 4 "${YOLO_HEF}.tmp" | grep -q "HEF"; then
            mv "${YOLO_HEF}.tmp" "${YOLO_HEF}"
            echo "Downloaded ${YOLO_HEF} ($(du -h "${YOLO_HEF}" | cut -f1))"
        else
            rm -f "${YOLO_HEF}.tmp"
            echo "ERROR: downloaded file is not a valid HEF. Skipping."
        fi
    else
        rm -f "${YOLO_HEF}.tmp"
        echo "SKIP: could not download ${HEF_URL}"
        echo "For a custom-trained model you need the Hailo Dataflow"
        echo "Compiler -- see: https://hailo.ai/developer-zone/"
    fi
fi

echo ""
echo "=== Model Download Complete ==="
echo ""
echo "Models in ${MODELS_DIR}:"
ls -la "${MODELS_DIR}/" 2>/dev/null || echo "(empty)"
