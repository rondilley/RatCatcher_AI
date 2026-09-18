#!/usr/bin/env bash
# RatCatcher AI -- Hailo PCIe Driver Repair
#
# Rebuilds and reinstalls the hailo_pci kernel module after a kernel
# upgrade has orphaned it, and registers it with DKMS so this does not
# happen again.
#
# WHY THIS IS NEEDED
#
# The Raspberry Pi archive's hailort-pcie-driver package Depends on
# build-essential, NOT on dkms.  Its postinst tries "make install_dkms"
# first and falls back to a plain "make install" when dkms is missing.
# That fallback compiles the module for the ONE kernel that happens to be
# running at install time and drops it in that kernel's module tree.
#
# The next kernel upgrade therefore silently removes NPU support: the
# PCIe device is still enumerated by lspci, but there is no driver bound
# to it, no /dev/hailo0, and "hailortcli scan" reports no devices.
#
# Observed on this Pi 2026-08-24 -- built against 6.18.34, running 6.18.39:
#   /lib/modules/6.18.34+rpt-rpi-2712/kernel/drivers/misc/hailo_pci.ko.xz
#   /lib/modules/6.18.39+rpt-rpi-2712/kernel/drivers/misc/   <-- no hailo_pci
#
# Installing dkms first makes the reinstall take the DKMS branch, which
# registers the source in /var/lib/dkms and rebuilds automatically on
# every future kernel update.
#
# Usage:
#   sudo ./scripts/fix_hailo_driver.sh

set -euo pipefail

KERNEL="$(uname -r)"

echo "=== RatCatcher AI -- Hailo PCIe Driver Repair ==="
echo "Running kernel: ${KERNEL}"
echo ""

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (use sudo)."
    exit 1
fi

# ---- [0/6] Confirm the hardware is actually present ----
echo "[0/6] Checking the PCIe bus..."
HAILO_PCI="$(lspci 2>/dev/null | grep -i 'hailo' || true)"
if [ -z "${HAILO_PCI}" ]; then
    echo "ERROR: No Hailo device on the PCIe bus."
    echo "This script repairs a missing DRIVER, not missing hardware."
    echo "Check that the AI HAT+ is seated and the PCIe ribbon is connected"
    echo "(and the right way round -- the ribbon is easy to reverse)."
    exit 1
fi
echo "Found: ${HAILO_PCI}"
echo ""

# ---- [1/6] Report the current state before changing anything ----
echo "[1/6] Current state..."
if lsmod | grep -q '^hailo_pci'; then
    echo "hailo_pci is already loaded. Nothing may be wrong -- continuing"
    echo "anyway will still register it with DKMS for future upgrades."
else
    echo "hailo_pci is NOT loaded."
fi
echo "Module files on disk:"
find /lib/modules -iname 'hailo_pci.ko*' 2>/dev/null | sed 's/^/  /' \
    || echo "  (none)"
echo ""

# ---- [2/6] Kernel headers for the RUNNING kernel ----
# Without these the module cannot be compiled, and the failure mode is a
# confusing build error deep in the package postinst rather than a clear
# "no headers" message.
echo "[2/6] Verifying kernel headers for ${KERNEL}..."
if [ ! -d "/lib/modules/${KERNEL}/build" ]; then
    echo "Headers missing. Installing linux-headers-${KERNEL}..."
    apt-get update
    apt-get install -y "linux-headers-${KERNEL}"
else
    echo "Headers present: /lib/modules/${KERNEL}/build"
fi
echo ""

# ---- [3/6] Install dkms so the rebuild becomes automatic ----
echo "[3/6] Installing dkms..."
if command -v dkms &>/dev/null; then
    echo "dkms already installed: $(dkms --version 2>&1 | head -1)"
else
    apt-get update
    apt-get install -y dkms
    echo "dkms installed. The driver reinstall below will now register"
    echo "itself in the DKMS tree instead of building a one-off module."
fi
echo ""

# ---- [4/6] Reinstall the driver package ----
# --reinstall re-runs the postinst, which is what actually compiles and
# installs the module.  With dkms now present it takes the DKMS branch.
echo "[4/6] Reinstalling hailort-pcie-driver..."
apt-get install -y --reinstall hailort-pcie-driver
depmod -a "${KERNEL}"
echo ""

# ---- [5/6] Load the module ----
echo "[5/6] Loading hailo_pci..."
if lsmod | grep -q '^hailo_pci'; then
    echo "Already loaded."
elif modprobe hailo_pci; then
    echo "Loaded."
else
    echo "ERROR: modprobe hailo_pci failed."
    echo "Build log: /var/log/hailort-pcie-driver.deb.log"
    echo "Kernel messages:"
    dmesg | tail -20
    exit 1
fi
echo ""

# ---- [6/6] Verify end to end ----
echo "[6/6] Verifying..."
FAILED=0

if lsmod | grep -q '^hailo_pci'; then
    echo "  OK      module loaded"
else
    echo "  FAIL    module not in lsmod"
    FAILED=1
fi

if [ -e /dev/hailo0 ]; then
    echo "  OK      /dev/hailo0 present"
else
    echo "  FAIL    /dev/hailo0 missing (udev rule did not fire?)"
    FAILED=1
fi

if command -v dkms &>/dev/null && dkms status 2>/dev/null | grep -qi hailo; then
    echo "  OK      registered with DKMS:"
    dkms status | grep -i hailo | sed 's/^/            /'
else
    echo "  WARN    not registered with DKMS -- the next kernel upgrade"
    echo "          will orphan the driver again and you will need to"
    echo "          re-run this script."
fi

echo ""
echo "  Device identity:"
if hailortcli fw-control identify 2>&1 | sed 's/^/    /'; then
    :
else
    echo "    FAIL    hailortcli could not talk to the device"
    FAILED=1
fi

echo ""
if [ "${FAILED}" -ne 0 ]; then
    echo "=== Repair INCOMPLETE -- see the failures above ==="
    echo "Build log: /var/log/hailort-pcie-driver.deb.log"
    exit 1
fi

echo "=== Repair complete ==="
echo ""
echo "Note the 'Device Architecture' line above (HAILO8 or HAILO8L)."
echo "HEFs are architecture-specific: a hailo8l HEF runs on either part,"
echo "but a hailo8 HEF will not load on a Hailo-8L."
echo ""
echo "Next, confirm the detector picks up the NPU:"
echo "  venv/bin/python -m ratcatcher.cli health"
