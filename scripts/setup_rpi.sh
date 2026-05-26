#!/usr/bin/env bash
# RatCatcher AI -- Raspberry Pi 5 Setup Script
# Requires: Raspberry Pi OS Bookworm (64-bit, Lite recommended)
#
# Usage:
#   chmod +x scripts/setup_rpi.sh
#   sudo ./scripts/setup_rpi.sh

set -euo pipefail

INSTALL_DIR="/opt/ratcatcher"
DATA_DIR="${INSTALL_DIR}/data"
LOG_DIR="/var/log/ratcatcher"
SERVICE_USER="ratcatcher"

echo "=== RatCatcher AI -- Raspberry Pi 5 Setup ==="
echo ""

# Check we are on an RPi
if [ ! -f /proc/device-tree/model ]; then
    echo "WARNING: /proc/device-tree/model not found. This may not be a Raspberry Pi."
fi

# Check for root
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (use sudo)."
    exit 1
fi

echo "[1/8] Updating system packages..."
apt-get update
apt-get upgrade -y

echo "[2/8] Installing system dependencies..."
apt-get install -y \
    python3-pip python3-venv python3-dev \
    libopencv-dev python3-opencv \
    ffmpeg \
    libcamera-dev python3-libcamera python3-picamera2 \
    sqlite3 \
    git

echo "[3/8] Creating service user..."
if ! id "${SERVICE_USER}" &>/dev/null; then
    useradd --system --create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
    usermod -aG video,i2c,gpio "${SERVICE_USER}"
    echo "Created user: ${SERVICE_USER}"
else
    echo "User ${SERVICE_USER} already exists"
fi

echo "[4/8] Creating directories..."
mkdir -p "${INSTALL_DIR}"
mkdir -p "${DATA_DIR}/clips"
mkdir -p "${DATA_DIR}/thumbnails"
mkdir -p "${DATA_DIR}/db"
mkdir -p "${INSTALL_DIR}/models"
mkdir -p "${INSTALL_DIR}/config"
mkdir -p "${LOG_DIR}"

chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${LOG_DIR}"

echo "[5/8] Setting up Python virtual environment..."
python3 -m venv "${INSTALL_DIR}/.venv"
source "${INSTALL_DIR}/.venv/bin/activate"

echo "[6/8] Installing RatCatcher AI..."
if [ -d "/home/${SUDO_USER:-pi}/RatCatcher_AI" ]; then
    pip install -e "/home/${SUDO_USER:-pi}/RatCatcher_AI[rpi]"
elif [ -d "$(pwd)" ] && [ -f "$(pwd)/pyproject.toml" ]; then
    pip install -e "$(pwd)[rpi]"
else
    echo "WARNING: Could not find RatCatcher_AI source. Install manually:"
    echo "  source ${INSTALL_DIR}/.venv/bin/activate"
    echo "  pip install -e /path/to/RatCatcher_AI[rpi]"
fi

echo "[7/8] Configuring cameras..."
CONFIG_TXT="/boot/firmware/config.txt"
if [ -f "${CONFIG_TXT}" ]; then
    if grep -q "camera_auto_detect=1" "${CONFIG_TXT}"; then
        sed -i 's/camera_auto_detect=1/camera_auto_detect=0/' "${CONFIG_TXT}"
        echo "Disabled camera_auto_detect in ${CONFIG_TXT}"
    fi
    if ! grep -q "dtoverlay=imx477" "${CONFIG_TXT}"; then
        echo "" >> "${CONFIG_TXT}"
        echo "# RatCatcher AI -- Arducam IMX477 cameras" >> "${CONFIG_TXT}"
        echo "dtoverlay=imx477" >> "${CONFIG_TXT}"
        echo "Added dtoverlay=imx477 to ${CONFIG_TXT}"
    fi
else
    echo "WARNING: ${CONFIG_TXT} not found. Camera config must be set manually."
fi

echo "[8/8] Installing systemd service..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "${SCRIPT_DIR}/../systemd/ratcatcher.service" ]; then
    cp "${SCRIPT_DIR}/../systemd/ratcatcher.service" /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable ratcatcher.service
    echo "Systemd service installed and enabled"
else
    echo "WARNING: systemd/ratcatcher.service not found. Install manually."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Install Hailo SDK:  sudo ./scripts/install_hailo.sh"
echo "  2. Download models:    ./scripts/download_models.sh"
echo "  3. Copy config:        cp config/default.yaml ${INSTALL_DIR}/config/"
echo "  4. Start service:      sudo systemctl start ratcatcher"
echo "  5. Check status:       sudo systemctl status ratcatcher"
echo "  6. View logs:          sudo journalctl -u ratcatcher -f"
echo ""
echo "A reboot is required for camera configuration changes to take effect."
