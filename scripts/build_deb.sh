#!/usr/bin/env bash
# RatCatcher AI -- Debian package builder
#
# Produces dist/ratcatcher_<version>_arm64.deb, installable with:
#   sudo apt-get install ./dist/ratcatcher_<version>_arm64.deb
#
# Run this ON THE PI. The package vendors binary wheels, and pip resolves
# those for the interpreter and architecture it is running on: a package
# built on x86-64 or against a different python minor version would carry
# wheels the target cannot import. This is the mirror image of
# build_hef.sh, which must NOT run on the Pi.
#
# Usage:
#   ./scripts/build_deb.sh [--output-dir DIR] [--skip-wheel-download]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="${REPO_ROOT}/dist"
SKIP_WHEEL_DOWNLOAD=0
BUILD_ROOT="${REPO_ROOT}/build/deb"

while [ $# -gt 0 ]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --skip-wheel-download) SKIP_WHEEL_DOWNLOAD=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    esac
done

echo "=== RatCatcher AI -- Debian package build ==="

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
echo "[1/7] Preflight..."

ARCH="$(dpkg --print-architecture)"
if [ "${ARCH}" != "arm64" ]; then
    echo "ERROR: this builds an arm64 package, but dpkg reports ${ARCH}."
    echo "       Build on the Raspberry Pi. The vendored wheels are"
    echo "       architecture-specific and cannot be cross-selected here."
    exit 1
fi

for tool in dpkg-deb python3 pip3; do
    command -v "${tool}" >/dev/null 2>&1 || {
        echo "ERROR: ${tool} not found."
        exit 1
    }
done

PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "      arch=${ARCH}  python=${PYVER}"

VERSION="$(python3 - <<'PY'
import re, pathlib
text = pathlib.Path("pyproject.toml").read_text()
m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
if not m:
    raise SystemExit("ERROR: no version in pyproject.toml")
print(m.group(1))
PY
)"
echo "      version=${VERSION}"

# The detector weights are gitignored (too large for the repo) and cannot
# be regenerated here -- ratcatcher_best is custom-trained, and the .hef
# needs the x86-only Hailo Dataflow Compiler. A package without them
# installs a service that cannot detect anything, so refuse to build one.
REQUIRED_MODELS="
models/ratcatcher_best.onnx
models/mobilenet_v2_inat_bird_quant.tflite
models/inat_bird_labels.txt
"
MISSING=""
for m in ${REQUIRED_MODELS}; do
    [ -f "${m}" ] || MISSING="${MISSING} ${m}"
done
if [ -n "${MISSING}" ]; then
    echo "ERROR: required model files are missing:${MISSING}"
    echo "       Run ./scripts/download_models.sh, and see"
    echo "       docs/DEPLOYMENT.md for the custom detector."
    exit 1
fi
if [ ! -f models/ratcatcher_best.hef ]; then
    echo "      WARNING: models/ratcatcher_best.hef is absent. The package"
    echo "               will install without an NPU model and the pipeline"
    echo "               will fall back to the CPU detector."
    echo "               Compile it with scripts/build_hef.sh on x86-64."
fi

# ---------------------------------------------------------------------------
# Staging tree
# ---------------------------------------------------------------------------
echo "[2/7] Staging..."
rm -rf "${BUILD_ROOT}"
STAGE="${BUILD_ROOT}/ratcatcher_${VERSION}_${ARCH}"
mkdir -p "${STAGE}/DEBIAN" \
         "${STAGE}/opt/ratcatcher/lib" \
         "${STAGE}/opt/ratcatcher/firmware/ratcatcher_panel" \
         "${STAGE}/opt/ratcatcher/models" \
         "${STAGE}/opt/ratcatcher/wheels" \
         "${STAGE}/etc/ratcatcher" \
         "${STAGE}/lib/systemd/system" \
         "${STAGE}/usr/bin" \
         "${STAGE}/usr/share/doc/ratcatcher"

# ---------------------------------------------------------------------------
# Wheels
# ---------------------------------------------------------------------------
echo "[3/7] Building the ratcatcher wheel..."
WHEELHOUSE="${STAGE}/opt/ratcatcher/wheels"
CACHE="${REPO_ROOT}/build/wheelcache"
mkdir -p "${CACHE}"

# The cache persists between builds so the third-party closure is not
# re-downloaded every time. Our own wheel must not persist with it: a
# version bump leaves the previous one behind, the whole cache is
# copied into the wheelhouse below, and the package then ships every
# ratcatcher it has ever built. pip picks the highest and installs
# correctly, so this shows up as dead weight rather than a failure --
# which is why it survived until the first bump.
for w in "${CACHE}"/ratcatcher-*.whl; do
    [ -e "${w}" ] || continue
    rm -f "${w}"
done

pip3 wheel --no-deps --wheel-dir "${CACHE}" "${REPO_ROOT}" >/dev/null
echo "      $(ls "${CACHE}" | grep -c '^ratcatcher-') ratcatcher wheel built"

if [ "${SKIP_WHEEL_DOWNLOAD}" -eq 0 ]; then
    echo "      downloading the ai-edge-litert dependency closure..."
    # ai-edge-litert is the one runtime dependency with no Debian
    # package: it is the maintained successor to tflite-runtime, which
    # publishes no wheels for python 3.13 at all. Everything else the
    # pipeline imports (numpy, cv2, yaml, PIL, serial, picamera2,
    # hailo_platform) comes from apt, declared in packaging/control.in.
    pip3 download ai-edge-litert --dest "${CACHE}" >/dev/null
else
    echo "      SKIPPED (--skip-wheel-download); using ${CACHE} as-is"
fi

# numpy is dropped from the wheelhouse on purpose. python3-numpy is an
# apt dependency of this package, and picamera2 and python3-opencv are
# both compiled against that copy. Shipping a second numpy would shadow
# it inside the venv and put two incompatible C ABIs in one interpreter.
PRUNED=0
for w in "${CACHE}"/numpy-*.whl; do
    [ -e "${w}" ] || continue
    rm -f "${w}"
    PRUNED=$((PRUNED + 1))
done
[ "${PRUNED}" -gt 0 ] && echo "      pruned ${PRUNED} numpy wheel(s) -- apt provides python3-numpy"

cp "${CACHE}"/*.whl "${WHEELHOUSE}/"
echo "      wheelhouse: $(ls -1 "${WHEELHOUSE}" | wc -l) wheels, $(du -sh "${WHEELHOUSE}" | cut -f1)"

# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------
echo "[4/7] Copying payload..."

# Models. BirdNET is excluded deliberately: CC BY-NC-SA 4.0 cannot be
# redistributed inside a GPL-3.0 package, and its NonCommercial term
# would follow anyone who deployed this commercially. The postinst
# fetches it from Zenodo so the licence stays between the user and the
# BirdNET team. yolov8n.hef is excluded too -- it is stock COCO-80 with
# no squirrel or rat class, so it cannot report a pest and nothing in
# config/default.yaml refers to it.
for m in ratcatcher_best.onnx ratcatcher_best.hef \
         mobilenet_v2_inat_bird_quant.tflite inat_bird_labels.txt; do
    if [ -f "models/${m}" ]; then
        cp "models/${m}" "${STAGE}/opt/ratcatcher/models/"
        echo "      model: ${m} ($(du -h "models/${m}" | cut -f1))"
    fi
done

cp config/default.yaml config/species.yaml "${STAGE}/etc/ratcatcher/"

install -m 755 packaging/lib/hardware-setup.sh "${STAGE}/opt/ratcatcher/lib/"
install -m 755 packaging/lib/firstboot.sh      "${STAGE}/opt/ratcatcher/lib/"
# Shipped rather than reimplemented: firstboot.sh calls both.
install -m 755 scripts/fix_hailo_driver.sh     "${STAGE}/opt/ratcatcher/lib/"
install -m 755 scripts/download_models.sh      "${STAGE}/opt/ratcatcher/lib/"
# The panel firmware is not flashed at install time -- the board may not
# be attached, and the toolchain is a 7.3 GB download a maintainer
# script cannot fetch. But DEPLOYMENT.md Step 6b has to be followable
# from a package-only install, so the tool and the sketch ship here and
# the operator runs them when the panel is in front of them.
#
# Only our own .ino goes in. Elecrow's e-paper driver sources are theirs
# and are not redistributed; the script downloads them at build time,
# the same arrangement as the BirdNET weights above.
install -m 755 scripts/build_panel_firmware.sh "${STAGE}/opt/ratcatcher/lib/"
install -m 644 firmware/ratcatcher_panel/ratcatcher_panel.ino \
               "${STAGE}/opt/ratcatcher/firmware/ratcatcher_panel/"

install -m 644 systemd/ratcatcher.service \
               packaging/systemd/ratcatcher-firstboot.service \
               "${STAGE}/lib/systemd/system/"

# /usr/bin/ratcatcher puts the CLI on PATH without exposing the venv
# layout. RATCATCHER_CONFIG_DIR matches ratcatcher.service so that a
# command typed by hand reads the same configuration the daemon does.
cat > "${STAGE}/usr/bin/ratcatcher" <<'WRAPPER'
#!/bin/sh
# RatCatcher AI CLI -- wrapper around the packaged virtualenv.
export RATCATCHER_CONFIG_DIR="${RATCATCHER_CONFIG_DIR:-/etc/ratcatcher}"
# HailoRT drops hailort.log into the working directory. That directory is
# /opt/ratcatcher below, which is root-owned, so an ordinary user running
# this gets a warning on every invocation. Point it somewhere writable.
export HAILORT_LOGGER_PATH="${HAILORT_LOGGER_PATH:-${TMPDIR:-/tmp}}"
if [ ! -x /opt/ratcatcher/.venv/bin/ratcatcher ]; then
    echo "ratcatcher: the virtualenv is missing." >&2
    echo "ratcatcher: reinstall with: sudo apt-get install --reinstall ratcatcher" >&2
    exit 1
fi
# Weights are resolved as a relative "models/..." path.
cd /opt/ratcatcher
exec /opt/ratcatcher/.venv/bin/ratcatcher "$@"
WRAPPER
chmod 755 "${STAGE}/usr/bin/ratcatcher"

install -m 644 packaging/copyright "${STAGE}/usr/share/doc/ratcatcher/"
install -m 644 README.md docs/DEPLOYMENT.md docs/ARCHITECTURE.md \
               "${STAGE}/usr/share/doc/ratcatcher/"

printf 'ratcatcher (%s) unstable; urgency=medium\n\n  * Packaged release.\n\n -- Ron Dilley <ron.dilley@gmail.com>  %s\n' \
    "${VERSION}" "$(date -R)" \
    | gzip -9n > "${STAGE}/usr/share/doc/ratcatcher/changelog.Debian.gz"

# Normalise permissions. The working tree here is group-writable, and
# cp/install carry that through; Debian policy wants 0644 for data and
# 0755 for programs and directories, and a group-writable file in /etc or
# /opt is a real (if small) privilege-escalation surface once installed.
chmod -R go-w "${STAGE}"
find "${STAGE}" -type d ! -path "${STAGE}/DEBIAN*" -exec chmod 755 {} +
find "${STAGE}" -type f ! -path "${STAGE}/DEBIAN/*" -exec chmod 644 {} +
chmod 755 "${STAGE}/usr/bin/ratcatcher" "${STAGE}"/opt/ratcatcher/lib/*.sh

# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------
echo "[5/7] Control metadata..."
INSTALLED_SIZE="$(du -ks "${STAGE}" | cut -f1)"
sed -e "s/@VERSION@/${VERSION}/" \
    -e "s/@INSTALLED_SIZE@/${INSTALLED_SIZE}/" \
    packaging/control.in > "${STAGE}/DEBIAN/control"

install -m 644 packaging/conffiles "${STAGE}/DEBIAN/conffiles"
install -m 755 packaging/postinst "${STAGE}/DEBIAN/postinst"
install -m 755 packaging/prerm    "${STAGE}/DEBIAN/prerm"
install -m 755 packaging/postrm   "${STAGE}/DEBIAN/postrm"

# dpkg verifies these on upgrade; a stale digest is a hard error there,
# so they are generated rather than maintained.
( cd "${STAGE}" && find . -type f ! -path './DEBIAN/*' -exec md5sum {} + \
    | sed 's| \./| |' > DEBIAN/md5sums )
chmod 644 "${STAGE}/DEBIAN/md5sums"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "[6/7] Building the package..."
mkdir -p "${OUTPUT_DIR}"
DEB="${OUTPUT_DIR}/ratcatcher_${VERSION}_${ARCH}.deb"
# --root-owner-group: the staging tree belongs to whoever ran this, and
# without it every packaged file would install owned by that uid.
dpkg-deb --root-owner-group --build "${STAGE}" "${DEB}" >/dev/null

echo "[7/7] Verifying..."
dpkg-deb --info "${DEB}" | sed 's/^/      /'
echo "      ---"
echo "      $(dpkg-deb --contents "${DEB}" | wc -l) files, $(du -h "${DEB}" | cut -f1)"

echo
echo "=== Built ${DEB} ==="
echo
echo "Install with:"
echo "  sudo apt-get install ${DEB}"
echo
echo "apt is what resolves the dependencies; 'sudo dpkg -i' does not and"
echo "will leave the package unconfigured if any are missing."
