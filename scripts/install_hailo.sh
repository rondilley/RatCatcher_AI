#!/usr/bin/env bash
# RatCatcher AI -- Hailo-8L AI HAT+ Installation Script
# Requires: Raspberry Pi 5 with Raspberry Pi OS Bookworm (64-bit)
#
# This installs the Hailo RT driver and runtime needed for NPU inference.
#
# Usage:
#   chmod +x scripts/install_hailo.sh
#   sudo ./scripts/install_hailo.sh

set -euo pipefail

echo "=== RatCatcher AI -- Hailo-8L AI HAT+ Setup ==="
echo ""

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (use sudo)."
    exit 1
fi

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
VENV="/opt/ratcatcher/.venv"
if [ -d "${VENV}" ]; then
    source "${VENV}/bin/activate"
    if python3 -c "from hailo_platform import HEF; print('Hailo Python bindings: OK')" 2>/dev/null; then
        echo "Python bindings verified."
    else
        echo "WARNING: Hailo Python bindings not available in venv."
        echo "You may need to install hailo_platform in the venv."
    fi
else
    echo "WARNING: RatCatcher venv not found at ${VENV}."
    echo "Run setup_rpi.sh first."
fi

echo ""
echo "=== Hailo Setup Complete ==="
echo ""
echo "A reboot is recommended after installing Hailo drivers."
echo "After reboot, verify with: hailortcli fw-control identify"
