#!/usr/bin/env bash
#
# Enable dual Adafruit SPH0645 I2S MEMS microphones on a Raspberry Pi 5.
#
# Usage:
#   sudo ./scripts/enable_i2s_mics.sh
#
# Wiring (both mics share one I2S bus; SEL selects the channel):
#
#   SPH0645 pin   Pi 5 header        Notes
#   -----------   ----------------   --------------------------------------
#   3V            pin 1  (3V3)       Do not use 5V.
#   GND           pin 6  (GND)       Common ground for both mics.
#   BCLK          pin 12 (GPIO18)    Shared by both mics.
#   LRCL / WS     pin 35 (GPIO19)    Shared by both mics.
#   DOUT          pin 38 (GPIO20)    Shared; the mics take turns by slot.
#   SEL           GND or 3V3         Mic A: GND (left).  Mic B: 3V3 (right).
#
# SEL is the only wiring difference between the two mics. Tying both SEL
# pins the same way puts both mics in the same time slot, where they will
# contend on the shared DOUT line and yield one unusable signal.
#
# After running this script a reboot is required: device tree overlays are
# read by the firmware at boot and cannot be applied to a running kernel.

set -euo pipefail

CONFIG_TXT="/boot/firmware/config.txt"
OVERLAY="googlevoicehat-soundcard"
BACKUP="${CONFIG_TXT}.bak-preaudio"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (use sudo)."
    exit 1
fi

if [ ! -f "${CONFIG_TXT}" ]; then
    echo "ERROR: ${CONFIG_TXT} not found."
    echo "       This script targets Raspberry Pi OS Bookworm or newer."
    exit 1
fi

if [ ! -f "/boot/firmware/overlays/${OVERLAY}.dtbo" ]; then
    echo "ERROR: Overlay ${OVERLAY}.dtbo is not installed."
    echo "       Install it with: apt-get install --reinstall raspberrypi-kernel"
    exit 1
fi

echo "[1/4] Backing up ${CONFIG_TXT}..."
if [ -f "${BACKUP}" ]; then
    echo "      Backup already exists at ${BACKUP}, leaving it alone."
else
    cp "${CONFIG_TXT}" "${BACKUP}"
    echo "      Saved to ${BACKUP}"
fi

echo "[2/4] Enabling the I2S interface..."
if grep -qE '^dtparam=i2s=on$' "${CONFIG_TXT}"; then
    echo "      Already enabled."
else
    sed -i 's|^#dtparam=i2s=on$|dtparam=i2s=on|' "${CONFIG_TXT}"
    if grep -qE '^dtparam=i2s=on$' "${CONFIG_TXT}"; then
        echo "      Uncommented dtparam=i2s=on"
    else
        printf '\ndtparam=i2s=on\n' >> "${CONFIG_TXT}"
        echo "      Appended dtparam=i2s=on"
    fi
fi

echo "[3/4] Adding the ${OVERLAY} overlay..."
if grep -qE "^dtoverlay=${OVERLAY}" "${CONFIG_TXT}"; then
    echo "      Already present."
else
    cat >> "${CONFIG_TXT}" <<'OVERLAY_BLOCK'

# Dual Adafruit SPH0645 I2S MEMS microphones for bird song detection.
# Both mics share BCLK (GPIO18), LRCL/WS (GPIO19) and DOUT (GPIO20). SEL
# is what separates them: SEL to GND puts a mic on the left channel, SEL
# to 3.3V puts it on the right. The result is one stereo capture device
# where channel 0 is the left mic and channel 1 is the right.
#
# googlevoicehat-soundcard is the standard overlay for the SPH0645. It
# loads snd-soc-googlevoicehat-codec and binds to the RP1 I2S controller
# on the Pi 5. The card runs 48 kHz with 32-bit frames, which is what the
# SPH0645 emits (18 data bits left-justified in a 32-bit slot) and also
# the sample rate BirdNET expects, so no resampling is needed.
dtoverlay=googlevoicehat-soundcard
OVERLAY_BLOCK
    echo "      Added dtoverlay=${OVERLAY}"
fi

echo "[4/4] Verifying configuration..."
grep -nE '^dtparam=i2s=on$|^dtoverlay=googlevoicehat' "${CONFIG_TXT}" | sed 's/^/      /'

echo
echo "Configuration written. A REBOOT is required for the firmware to load"
echo "the overlay:"
echo
echo "  sudo reboot"
echo
echo "After rebooting, confirm the microphones enumerated:"
echo
echo "  arecord -l                    # expect a 'snd_rpi_googlevoicehat' card"
echo "  ratcatcher test-mic           # records and reports per-channel level"
echo
echo "If no capture device appears, restore the backup and check the wiring:"
echo "  sudo cp ${BACKUP} ${CONFIG_TXT}"
