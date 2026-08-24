#!/usr/bin/env bash
# RatCatcher AI -- Hailo AI HAT+ Installation Script
# Requires: Raspberry Pi 5 with Raspberry Pi OS Bookworm or Trixie (64-bit)
#
# This installs the Hailo RT driver and runtime needed for NPU inference.
# Works with both the Hailo-8L (13 TOPS) and Hailo-8 (26 TOPS) AI HAT+.
#
# Usage:
#   chmod +x scripts/install_hailo.sh
#   sudo ./scripts/install_hailo.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== RatCatcher AI -- Hailo AI HAT+ Setup ==="
echo ""

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (use sudo)."
    exit 1
fi

echo "[0/4] Detecting Hailo hardware..."
HAILO_PCI="$(lspci 2>/dev/null | grep -i 'hailo' || true)"
if [ -z "${HAILO_PCI}" ]; then
    echo "ERROR: No Hailo device found on the PCIe bus."
    echo "Check that the AI HAT+ is seated and the PCIe ribbon is connected."
    exit 1
fi
echo "Found: ${HAILO_PCI}"
if echo "${HAILO_PCI}" | grep -qi 'hailo-8l'; then
    HAILO_ARCH="hailo8l"
else
    HAILO_ARCH="hailo8"
fi
echo "Architecture: ${HAILO_ARCH} (compile/download .hef files for this target)"
echo ""

echo "[1/4] Installing Hailo packages..."
apt-get update
apt-get install -y hailo-all

echo "[2/4] Verifying Hailo device..."
if command -v hailortcli &>/dev/null; then
    echo "hailortcli found at: $(which hailortcli)"
    if hailortcli fw-control identify 2>/dev/null; then
        echo "Hailo device detected and responding."
    else
        echo "WARNING: hailortcli installed but no Hailo device detected."
        echo "Make sure the AI HAT+ is properly seated on the GPIO header."
    fi
else
    echo "WARNING: hailortcli not found after installation."
    echo "A reboot may be required."
fi

echo "[3/4] Enabling PCIe Gen 3 (optional performance boost)..."
CONFIG_TXT="/boot/firmware/config.txt"
if [ -f "${CONFIG_TXT}" ]; then
    if ! grep -q "dtparam=pciex1_gen=3" "${CONFIG_TXT}"; then
        echo "" >> "${CONFIG_TXT}"
        echo "# RatCatcher AI -- PCIe Gen 3 for Hailo (not officially certified)" >> "${CONFIG_TXT}"
        echo "# Uncomment the line below for ~2x Hailo throughput:" >> "${CONFIG_TXT}"
        echo "# dtparam=pciex1_gen=3" >> "${CONFIG_TXT}"
        echo "PCIe Gen 3 option added to ${CONFIG_TXT} (commented out by default)."
        echo "Uncomment dtparam=pciex1_gen=3 if you want higher throughput."
    fi
fi

echo "[4/4] Verifying Python bindings..."
# hailo_platform ships as the apt package python3-hailort and installs into
# /usr/lib/python3/dist-packages -- it is NOT on PyPI and cannot be pip
# installed.  A venv therefore only sees it when it was created with
# --system-site-packages (or has that flag set in pyvenv.cfg).
VENV=""
for CANDIDATE in "${REPO_ROOT}/venv" "${REPO_ROOT}/.venv" "/opt/ratcatcher/.venv"; do
    if [ -d "${CANDIDATE}" ]; then
        VENV="${CANDIDATE}"
        break
    fi
done

if [ -z "${VENV}" ]; then
    echo "WARNING: No RatCatcher venv found. Run setup_rpi.sh first."
elif ! grep -q "include-system-site-packages = true" "${VENV}/pyvenv.cfg"; then
    echo "WARNING: ${VENV} cannot see system packages, so 'hailo_platform'"
    echo "         will never import there. Fix with either:"
    echo "           sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' ${VENV}/pyvenv.cfg"
    echo "         or recreate it:"
    echo "           python3 -m venv --system-site-packages ${VENV}"
elif "${VENV}/bin/python" -c "from hailo_platform import HEF" 2>/dev/null; then
    echo "Hailo Python bindings: OK (${VENV})"
else
    echo "WARNING: 'hailo_platform' still not importable from ${VENV}."
    echo "         Check that python3-hailort installed and matches this"
    echo "         venv's Python version ($("${VENV}/bin/python" -V 2>&1))."
fi

echo ""
echo "=== Hailo Setup Complete ==="
echo ""
echo "A reboot is recommended after installing Hailo drivers."
echo "After reboot, verify with: hailortcli fw-control identify"
