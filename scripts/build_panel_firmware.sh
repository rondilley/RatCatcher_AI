#!/usr/bin/env bash
#
# Build and flash the RatCatcher status panel firmware.
#
# Target: Elecrow CrowPanel ESP32 2.13-inch e-paper HMI display.
#
# The board ships with an Elecrow demo that ignores the serial port. It
# must be replaced before the panel can show anything from RatCatcher.
# This script installs arduino-cli, the ESP32 core and ArduinoJson into
# a build tree of its own, fetches Elecrow's e-paper driver files, and
# compiles.
#
# The driver files are Elecrow's, not ours. They are downloaded here
# rather than kept in this repository, in the same way that the BirdNET
# weights are, so their terms stay with their author.
#
# It runs from a repository checkout and from an installed package
# alike. See "Build tree" below for where each one puts its files.
#
# Usage:
#   build_panel_firmware.sh                    # compile only
#   build_panel_firmware.sh --upload           # compile and flash
#   build_panel_firmware.sh --upload --port /dev/ttyUSB0
#   build_panel_firmware.sh --clean            # start again
#
set -euo pipefail

# --------------------------------------------------------------------
# Build tree
# --------------------------------------------------------------------
#
# Two layouts. A repository checkout keeps everything under firmware/,
# which .gitignore already covers. An installed package cannot: this
# script sits in /opt/ratcatcher/lib/ and the sketch in
# /opt/ratcatcher/firmware/, both root-owned and both dpkg's, while a
# build writes about 7.3 GB of ESP32 toolchain plus the downloaded
# Elecrow driver files. That would need root and would leave files
# behind that dpkg knows nothing about. So an installed run builds in
# the invoking user's cache directory, with the sketch copied there.
#
# pyproject.toml is the marker for a checkout. Testing whether
# firmware/ is writable would not do: under sudo /opt/ratcatcher is
# writable too, and the 7.3 GB would land in the package directory.

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="$(dirname "${SELF_DIR}")"

if [[ -f "${PREFIX}/pyproject.toml" ]]; then
    FIRMWARE_DIR="${PREFIX}/firmware"
    SKETCH_SRC="${FIRMWARE_DIR}/ratcatcher_panel"
    SKETCH_DIR="${SKETCH_SRC}"
else
    SKETCH_SRC="${PREFIX}/firmware/ratcatcher_panel"
    FIRMWARE_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}/ratcatcher-panel"
    SKETCH_DIR="${FIRMWARE_DIR}/ratcatcher_panel"
fi

TOOLS_DIR="${FIRMWARE_DIR}/.arduino"
BUILD_DIR="${FIRMWARE_DIR}/build"

ARDUINO_CLI="${TOOLS_DIR}/bin/arduino-cli"
FQBN="esp32:esp32:esp32s3"

# Elecrow's driver for this exact panel. GxEPD2 is not used: the board
# ships with one of two driver ICs (SSD1680Z or JD79661) depending on
# revision, GxEPD2 supports only the first, and a board with the second
# would compile cleanly and then show nothing.
EPD_BASE_URL="https://raw.githubusercontent.com/Elecrow-RD/CrowPanel-ESP32-2.13-E-paper-HMI-Display-with-122-250/master/example/arduino-v1.2/main"
EPD_FILES=(EPD.h EPD.cpp EPD_Init.h EPD_Init.cpp EPDfont.h spi.h spi.cpp)

PORT=""
DO_UPLOAD=0
DO_CLEAN=0

usage() {
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload) DO_UPLOAD=1; shift ;;
        --port) PORT="$2"; shift 2 ;;
        --clean) DO_CLEAN=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

need() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is required but not installed."
}

# --------------------------------------------------------------------
# Clean
# --------------------------------------------------------------------

if [[ ${DO_CLEAN} -eq 1 ]]; then
    log "Removing ${TOOLS_DIR} and ${BUILD_DIR}"
    rm -rf "${TOOLS_DIR}" "${BUILD_DIR}"
    for name in "${EPD_FILES[@]}"; do
        rm -f "${SKETCH_DIR}/${name}"
    done
    echo "Clean. Run again without --clean to rebuild."
    exit 0
fi

need uname
need tar

# --------------------------------------------------------------------
# Sketch
# --------------------------------------------------------------------

[[ -f "${SKETCH_SRC}/ratcatcher_panel.ino" ]] \
    || die "No sketch at ${SKETCH_SRC}/ratcatcher_panel.ino"

if [[ "${SKETCH_DIR}" != "${SKETCH_SRC}" ]]; then
    log "Building in ${FIRMWARE_DIR}"
    mkdir -p "${SKETCH_DIR}"
    cp -f "${SKETCH_SRC}/ratcatcher_panel.ino" "${SKETCH_DIR}/"
fi

# --------------------------------------------------------------------
# arduino-cli
# --------------------------------------------------------------------

install_arduino_cli() {
    local arch download_arch url
    arch="$(uname -m)"
    case "${arch}" in
        x86_64)  download_arch="Linux_64bit" ;;
        aarch64) download_arch="Linux_ARM64" ;;
        armv7l)  download_arch="Linux_ARMv7" ;;
        *) die "No arduino-cli build for ${arch}." ;;
    esac

    url="https://downloads.arduino.cc/arduino-cli/arduino-cli_latest_${download_arch}.tar.gz"
    log "Installing arduino-cli for ${arch}"
    mkdir -p "${TOOLS_DIR}/bin"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${url}" | tar -xz -C "${TOOLS_DIR}/bin" arduino-cli
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "${url}" | tar -xz -C "${TOOLS_DIR}/bin" arduino-cli
    else
        die "Neither curl nor wget is installed."
    fi

    chmod +x "${ARDUINO_CLI}"
}

if [[ ! -x "${ARDUINO_CLI}" ]]; then
    install_arduino_cli
fi

# Keep every downloaded core, library and tool inside the repository.
# A build must not depend on, or disturb, an Arduino installation the
# user keeps elsewhere.
export ARDUINO_DIRECTORIES_DATA="${TOOLS_DIR}/data"
export ARDUINO_DIRECTORIES_DOWNLOADS="${TOOLS_DIR}/downloads"
export ARDUINO_DIRECTORIES_USER="${TOOLS_DIR}/user"

log "arduino-cli $("${ARDUINO_CLI}" version)"

# --------------------------------------------------------------------
# Working python3
# --------------------------------------------------------------------
#
# The ESP32 build recipes shell out to python3 for esptool and for the
# partition generator. Whatever python3 is first on PATH is what they
# get, and in this repository that is often the wrong one: an activated
# venv/ built for the Raspberry Pi puts an aarch64 interpreter ahead of
# the system one, and the build dies at the last step with
# "exec format error" after a full compile.
#
# Put a shim directory first on PATH holding a python3 that runs here.

ensure_python3() {
    local shim_dir="${TOOLS_DIR}/shim"
    local candidate

    if python3 -c 'import sys' >/dev/null 2>&1; then
        return 0
    fi

    echo "python3 on PATH does not execute on this machine. Looking for another."
    for candidate in /usr/bin/python3 /usr/local/bin/python3 /bin/python3; do
        if [[ -x "${candidate}" ]] && "${candidate}" -c 'import sys' >/dev/null 2>&1; then
            mkdir -p "${shim_dir}"
            ln -sf "${candidate}" "${shim_dir}/python3"
            export PATH="${shim_dir}:${PATH}"
            echo "Using ${candidate}"
            return 0
        fi
    done

    die "No working python3 found. The ESP32 build needs one.
The python3 first on PATH is $(command -v python3 || echo none), which
cannot run here. Deactivate any virtualenv built for another
architecture, or install a system python3."
}

ensure_python3

# --------------------------------------------------------------------
# Core and libraries
# --------------------------------------------------------------------
#
# Budget 7.3 GB and check you have it. Measured on 2026-08-29 with
# arduino-esp32 3.3.11; the figure here used to say 2.3 GB, which was
# true of an older core and cost a filled root filesystem when it was
# believed against 6.3 GB free.
#
# The bulk is not the compiler. The core installs a separate lib
# package for every chip it supports -- esp32, c3, c5, c6, h2, p4,
# p4_es, s2, s3 -- at roughly 250 MB each, and this board needs
# exactly one of them, esp32s3-libs. arduino-cli offers no way to ask
# for a single target, so all of them arrive or none do.
#
# A tree that is already complete can be borrowed rather than fetched
# again, which matters on a Pi that has no room for a second copy:
#
#   ln -sfn ~/.cache/ratcatcher-panel/.arduino firmware/.arduino
#
# Remove the link afterwards: .gitignore lists firmware/.arduino/ with
# a trailing slash, which matches a directory and not a symlink.

if [[ ! -d "${ARDUINO_DIRECTORIES_DATA}/packages/esp32" ]]; then
    log "Installing the ESP32 core (about 7.3 GB, and slow, the first time)"
    "${ARDUINO_CLI}" config init --overwrite >/dev/null
    "${ARDUINO_CLI}" config add board_manager.additional_urls \
        https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
    "${ARDUINO_CLI}" core update-index
    "${ARDUINO_CLI}" core install esp32:esp32
else
    log "ESP32 core already installed"
fi

if [[ ! -d "${ARDUINO_DIRECTORIES_USER}/libraries/ArduinoJson" ]]; then
    log "Installing ArduinoJson"
    "${ARDUINO_CLI}" lib install "ArduinoJson"
else
    log "ArduinoJson already installed"
fi

# --------------------------------------------------------------------
# Elecrow e-paper driver
# --------------------------------------------------------------------

fetch_epd_sources() {
    log "Fetching the Elecrow e-paper driver"
    for name in "${EPD_FILES[@]}"; do
        local target="${SKETCH_DIR}/${name}"
        [[ -f "${target}" ]] && continue
        echo "  ${name}"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL "${EPD_BASE_URL}/${name}" -o "${target}"
        else
            wget -qO "${target}" "${EPD_BASE_URL}/${name}"
        fi
        [[ -s "${target}" ]] || die "Downloaded ${name} is empty."
    done
}

fetch_epd_sources

# The driver defaults to portrait. The RatCatcher layout is landscape,
# 250 wide by 128 high, which is what USE_HORIZONTIAL 2 selects. The
# vendor file already ships with 2; check rather than assume, because a
# silent change here rotates the whole screen.
if ! grep -q '^#define USE_HORIZONTIAL 2' "${SKETCH_DIR}/EPD_Init.h"; then
    die "EPD_Init.h is not set to USE_HORIZONTIAL 2 (landscape). \
Edit ${SKETCH_DIR}/EPD_Init.h and set it, or run --clean and try again."
fi

# --------------------------------------------------------------------
# Compile
# --------------------------------------------------------------------

log "Compiling ${SKETCH_DIR}"
mkdir -p "${BUILD_DIR}"
"${ARDUINO_CLI}" compile \
    --fqbn "${FQBN}" \
    --build-path "${BUILD_DIR}" \
    "${SKETCH_DIR}"

echo
echo "Built: ${BUILD_DIR}/ratcatcher_panel.ino.bin"

# --------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------

if [[ ${DO_UPLOAD} -eq 0 ]]; then
    echo
    echo "Not flashed. Add --upload to write it to the panel."
    exit 0
fi

if [[ -z "${PORT}" ]]; then
    log "Looking for a board"
    PORT="$("${ARDUINO_CLI}" board list --format json 2>/dev/null \
        | grep -o '"address"[^,]*' | head -1 | cut -d'"' -f4 || true)"
    [[ -n "${PORT}" ]] || die "No serial port found. Pass --port /dev/ttyUSB0."
    echo "Using ${PORT}"
fi

if [[ ! -w "${PORT}" ]]; then
    die "Cannot write to ${PORT}. Add this account to the dialout group:
  sudo usermod -aG dialout \$USER
then log out and log in again."
fi

log "Flashing ${PORT}"
"${ARDUINO_CLI}" upload \
    --fqbn "${FQBN}" \
    --build-path "${BUILD_DIR}" \
    --port "${PORT}" \
    "${SKETCH_DIR}"

echo
echo "Done. Check the panel with:"
echo "  ratcatcher display --port ${PORT} --once"
