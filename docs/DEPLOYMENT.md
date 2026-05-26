# Deployment Guide -- RatCatcher AI

## Hardware Assembly

### Components

| Component | Part Number | Notes |
|---|---|---|
| Raspberry Pi 5 (4GB) | | 8GB also works |
| Arducam UC-517 B0270 (x2) | B0270 | IR-Cut variant for day/night |
| Hailo-8L AI HAT+ | SC1166 | 13 TOPS, sits on GPIO header |
| UPS HAT | | Model-specific, see below |
| MicroSD 64GB+ | | A2 rated preferred |
| 22-pin FFC ribbon cables (x2) | | RPi5 uses 22-pin, not 15-pin |
| IP65 enclosure | | With cable glands |
| Active cooler for RPi5 | | Required for sustained load |
| 5V/5A USB-C power supply | | Official RPi5 PSU recommended |

### Assembly Order

1. Flash Raspberry Pi OS Bookworm Lite (64-bit) on the MicroSD card
2. Mount RPi5 on standoffs inside the enclosure
3. Attach the active cooler to the RPi5
4. Seat the Hailo-8L AI HAT+ on the GPIO header
5. Connect both cameras via 22-pin FFC cables to CAM0 and CAM1
6. Mount the UPS HAT
7. Mount cameras behind glass or polycarbonate windows
8. Route power cable through a bottom-mounted IP68 cable gland

### Camera Mounting

- Position cameras to cover different angles of the feeder(s)
- Mount behind optical glass or UV-resistant polycarbonate
- Apply hydrophobic coating to exterior window surface
- Install under a rain hood to minimize water on the lens window
- Angle slightly downward (15-30 degrees) for optimal feeder coverage

## Software Setup

### Step 1: Initial OS Setup

After flashing, enable SSH and configure WiFi:

```bash
# On the SD card (boot partition)
touch ssh
# Create wpa_supplicant.conf with your WiFi credentials

# Or use Raspberry Pi Imager to pre-configure SSH + WiFi
```

SSH into the Pi and run:

```bash
sudo raspi-config
# Set hostname, locale, timezone
# Enable I2C (for UPS HAT monitoring)
# Expand filesystem
```

### Step 2: Install RatCatcher AI

```bash
# Clone the repository
git clone <repo-url> ~/RatCatcher_AI
cd ~/RatCatcher_AI

# Run the setup script (installs packages, creates venv, configures cameras)
sudo ./scripts/setup_rpi.sh
```

This script:
- Installs system packages (Python, OpenCV, FFmpeg, libcamera, Picamera2)
- Creates the `ratcatcher` system user with video/i2c/gpio groups
- Creates `/opt/ratcatcher/` with data directories
- Creates a Python venv and installs the project
- Configures `/boot/firmware/config.txt` for IMX477 cameras
- Installs the systemd service

### Step 3: Install Hailo Drivers

```bash
sudo ./scripts/install_hailo.sh
sudo reboot
```

After reboot, verify:

```bash
hailortcli fw-control identify
# Should show Hailo-8L device info
```

### Step 4: Download Models

```bash
source /opt/ratcatcher/.venv/bin/activate
./scripts/download_models.sh /opt/ratcatcher/models
```

Copy the pre-trained models to the RPi:

```bash
# Custom-trained YOLO detector (11.7 MB, 5 classes)
cp models/ratcatcher_best.onnx /opt/ratcatcher/models/

# Species classifier (3.6 MB, 965 bird species)
cp models/mobilenet_v2_inat_bird_quant.tflite /opt/ratcatcher/models/
cp models/inat_bird_labels.txt /opt/ratcatcher/models/
```

The YOLO model was trained with `training/train_detector.py` on Open
Images V7 data. To retrain with your own feeder images, see
`training/README.md`.

### Step 5: Configure

```bash
cp config/default.yaml /opt/ratcatcher/config/
cp config/species.yaml /opt/ratcatcher/config/
```

Edit `/opt/ratcatcher/config/default.yaml` for your setup:
- Adjust camera resolution and FPS
- Set data_dir to `/opt/ratcatcher/data`
- Tune motion detection sensitivity for your feeder location
- Adjust retention days and max disk usage

### Step 6: Test

```bash
source /opt/ratcatcher/.venv/bin/activate

# Verify cameras
ratcatcher test-camera --camera 0
ratcatcher test-camera --camera 1

# Verify system
ratcatcher health

# Test pipeline briefly
ratcatcher run --cameras 0
# Ctrl+C to stop
```

### Step 7: Start Service

```bash
sudo systemctl start ratcatcher
sudo systemctl status ratcatcher
sudo journalctl -u ratcatcher -f
```

The service auto-starts on boot and restarts on failure.

## Monitoring

### View Logs

```bash
sudo journalctl -u ratcatcher -f        # Live tail
sudo journalctl -u ratcatcher --since today  # Today's logs
```

### Detection Statistics

```bash
source /opt/ratcatcher/.venv/bin/activate
ratcatcher stats --last 24h
ratcatcher stats --last 7d
```

### System Health

```bash
ratcatcher health
# Shows: CPU temp, memory, disk, Hailo status, FFmpeg availability
```

### Query the Database

```bash
sqlite3 /opt/ratcatcher/data/detections.db

-- Recent detections
SELECT timestamp, camera_id, class_name, species, common_name, confidence
FROM detections ORDER BY timestamp DESC LIMIT 20;

-- Species counts today
SELECT species, common_name, COUNT(*) as count
FROM detections
WHERE timestamp >= date('now')
GROUP BY species
ORDER BY count DESC;

-- Pest detections
SELECT timestamp, class_name, confidence
FROM detections
WHERE class_name IN ('squirrel', 'rat', 'cat')
ORDER BY timestamp DESC;
```

## Outdoor Deployment Checklist

- [ ] Enclosure is IP65 rated with sealed cable glands
- [ ] Gore-Tex breathing vents installed to prevent condensation
- [ ] Desiccant packs placed inside enclosure
- [ ] Camera windows are optical glass (not acrylic, which yellows)
- [ ] Hydrophobic coating applied to camera window exterior
- [ ] All cables enter from the bottom (water drainage)
- [ ] Active cooler installed on RPi5
- [ ] UPS HAT charged and tested
- [ ] Solar panel sized for ~12W continuous (50W panel minimum)
- [ ] Battery sized for 1+ day autonomy (40Ah+ LiFePO4)
- [ ] Watchdog timer enabled in systemd service
- [ ] Nightly reboot scheduled via cron
- [ ] Remote SSH access configured
- [ ] Enclosure mounted at 15-30 degree downward angle

## Troubleshooting

### Camera not detected

```bash
# Check camera connections
libcamera-hello --camera 0 --timeout 5000
libcamera-hello --camera 1 --timeout 5000

# Verify config.txt
grep -i "imx477\|camera" /boot/firmware/config.txt
# Should show: camera_auto_detect=0 and dtoverlay=imx477
```

### Hailo not responding

```bash
# Check PCIe device
lspci | grep Hailo

# Check driver
dmesg | grep hailo

# Re-identify
hailortcli fw-control identify
```

### High CPU temperature

- Verify active cooler is running
- Check enclosure ventilation
- Consider adding shade/hood over enclosure
- Reduce inference FPS in config

### Database growing too large

```bash
# Check size
du -sh /opt/ratcatcher/data/

# Adjust retention in config
# retention_days: 14  (reduce from 30)
# max_disk_gb: 5      (reduce from 10)

# Force cleanup
python -c "
from ratcatcher.storage.retention import RetentionManager
from pathlib import Path
rm = RetentionManager(Path('/opt/ratcatcher/data/clips'), Path('/opt/ratcatcher/data/thumbnails'), 14, 5.0)
deleted = rm.cleanup()
print(f'Deleted {deleted} files')
"
```
