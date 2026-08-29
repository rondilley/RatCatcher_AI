#!/usr/bin/env bash
# RatCatcher AI -- Model Download and Conversion Script
#
# Downloads YOLOv8n for object detection and prepares it for the
# configured inference backend.
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh [--birdnet-only] [MODELS_DIR]
#
# --birdnet-only fetches just the BirdNET song model. The Debian package
# ships the detector and species classifier already, and needs only the
# one model it may not redistribute, so it calls this script that way
# rather than carrying a second copy of the download-and-verify logic.

set -euo pipefail

BIRDNET_ONLY=0
MODELS_DIR=""
for arg in "$@"; do
    case "${arg}" in
        --birdnet-only) BIRDNET_ONLY=1 ;;
        -*)
            echo "ERROR: unknown option: ${arg}" >&2
            echo "Usage: $0 [--birdnet-only] [MODELS_DIR]" >&2
            exit 2
            ;;
        *) MODELS_DIR="${arg}" ;;
    esac
done
MODELS_DIR="${MODELS_DIR:-models}"
mkdir -p "${MODELS_DIR}"

echo "=== RatCatcher AI -- Model Download ==="
echo "Output directory: ${MODELS_DIR}"
echo ""

if [ "${BIRDNET_ONLY}" -eq 0 ]; then

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
    #
    # Ask the firmware, not the PCIe ID.  lspci reports BOTH parts as
    # "Hailo-8 AI Processor" -- there is no "hailo-8l" string to grep for,
    # so the earlier lspci test never matched and this always fetched the
    # hailo8 build.  On a real Hailo-8L that HEF fails to load outright.
    # Requires the driver to be up; if it is not, fall back to hailo8l,
    # which is the safe direction (a hailo8l HEF runs on a Hailo-8, but a
    # hailo8 HEF will not run on a Hailo-8L).
    HAILO_ARCH=""
    if command -v hailortcli &>/dev/null; then
        HAILO_ID="$(hailortcli fw-control identify 2>/dev/null || true)"
        case "$(echo "${HAILO_ID}" | grep -i 'Device Architecture')" in
            *HAILO8L*) HAILO_ARCH="hailo8l" ;;
            *HAILO8*)  HAILO_ARCH="hailo8" ;;
        esac
    fi
    if [ -z "${HAILO_ARCH}" ]; then
        HAILO_ARCH="hailo8l"
        echo "WARNING: could not query the Hailo device architecture."
        echo "         Is the driver loaded? Check: hailortcli fw-control identify"
        echo "         Defaulting to ${HAILO_ARCH}, which runs on either part."
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

fi  # BIRDNET_ONLY

# ---------------------------------------------------------------------------
# BirdNET -- bird song identification from the I2S microphones
# ---------------------------------------------------------------------------
#
# LICENSING: the BirdNET model weights are released under CC BY-NC-SA 4.0
# (Attribution, NonCommercial, ShareAlike). They are downloaded at runtime
# rather than committed, so this repository carries no non-commercially
# licensed binaries. Review the terms before any commercial deployment:
#   https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# The archive is hosted on Zenodo, which serves stable direct URLs. The
# project documentation also links Google Drive copies, but those need an
# interactive confirmation token for large files and cannot be scripted.
#
# The zip contains audio-model.tflite (the acoustic classifier, 3-second
# windows of 48 kHz audio to 6522 species), meta-model.tflite (a
# location and date prior, unused here), and labels for 26 languages.

BIRDNET_MODEL="${MODELS_DIR}/BirdNET_v2.4_audio-model.tflite"
BIRDNET_LABELS="${MODELS_DIR}/BirdNET_v2.4_labels_en_us.txt"
BIRDNET_ZIP_URL="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip"

echo ""
echo "--- BirdNET (bird song identification) ---"

if [ -f "${BIRDNET_MODEL}" ] && [ -f "${BIRDNET_LABELS}" ]; then
    echo "SKIP: BirdNET model and labels already present."
elif ! command -v unzip >/dev/null 2>&1; then
    echo "SKIP: unzip is required. Install it with: sudo apt-get install unzip"
else
    BIRDNET_TMP="$(mktemp -d)"
    trap 'rm -rf "${BIRDNET_TMP}"' EXIT

    echo "Downloading BirdNET v2.4 TFLite archive (about 73 MB)..."
    if curl -fsSL --max-time 900 -o "${BIRDNET_TMP}/birdnet.zip" "${BIRDNET_ZIP_URL}"; then
        if unzip -q -o "${BIRDNET_TMP}/birdnet.zip" \
                "audio-model.tflite" "labels/en_us.txt" -d "${BIRDNET_TMP}"; then
            # TFLite files carry the "TFL3" identifier at byte offset 4.
            if dd if="${BIRDNET_TMP}/audio-model.tflite" bs=1 skip=4 count=4 \
                    2>/dev/null | grep -q "TFL3"; then
                mv "${BIRDNET_TMP}/audio-model.tflite" "${BIRDNET_MODEL}"
                mv "${BIRDNET_TMP}/labels/en_us.txt" "${BIRDNET_LABELS}"
                echo "Installed ${BIRDNET_MODEL} ($(du -h "${BIRDNET_MODEL}" | cut -f1))"
                echo "Installed ${BIRDNET_LABELS} ($(wc -l < "${BIRDNET_LABELS}") species)"
            else
                echo "ERROR: extracted file is not a valid TFLite model. Skipping."
            fi
        else
            echo "ERROR: could not extract the BirdNET archive. Skipping."
        fi
    else
        echo "SKIP: could not download ${BIRDNET_ZIP_URL}"
        echo "Model listings: https://birdnet-team.github.io/BirdNET-Analyzer"
    fi

    rm -rf "${BIRDNET_TMP}"
    trap - EXIT
fi

echo ""
echo "=== Model Download Complete ==="
echo ""
echo "Models in ${MODELS_DIR}:"
ls -la "${MODELS_DIR}/" 2>/dev/null || echo "(empty)"
