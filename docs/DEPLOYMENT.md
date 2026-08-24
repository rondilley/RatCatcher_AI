# Deployment Guide -- RatCatcher AI

## Hardware Assembly

### Components

| Component | Part Number | Notes |
|---|---|---|
| Raspberry Pi 5 (4GB) | | 8GB also works |
| Arducam UC-517 B0270 (x2) | B0270 | IR-Cut variant for day/night |
| Hailo-8L AI HAT+ | SC1166 | 13 TOPS, sits on GPIO header |
| Adafruit SPH0645 I2S mic (x2) | | MEMS breakout, shared I2S bus |
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
7. Wire both I2S microphones (see below)
8. Mount cameras behind glass or polycarbonate windows
9. Route power cable through a bottom-mounted IP68 cable gland

### Camera Mounting

- Position cameras to cover different angles of the feeder(s)
- Mount behind optical glass or UV-resistant polycarbonate
- Apply hydrophobic coating to exterior window surface
- Install under a rain hood to minimize water on the lens window
- Angle slightly downward (15-30 degrees) for optimal feeder coverage

### Microphone Wiring

Both mics share one I2S bus. SEL is the only wiring difference between
them; it selects which half of the stereo frame each mic drives.

| SPH0645 pin | Pi 5 header | Notes |
|---|---|---|
| 3V | pin 1 (3V3) | Do not use 5V |
| GND | pin 6 (GND) | Common ground for both mics |
| BCLK | pin 12 (GPIO18) | Shared by both mics |
| LRCL / WS | pin 35 (GPIO19) | Shared by both mics |
| DOUT | pin 38 (GPIO20) | Shared; the mics take turns by slot |
| SEL | GND or 3V3 | Mic A: GND (left). Mic B: 3V3 (right) |

Tying both SEL pins the same way puts both mics in the same time slot,
where they contend on the shared DOUT line and yield one unusable
signal. Because DOUT and the 3V3 rail are *shared*, a single fault there
silences both channels at once -- see Troubleshooting.

Mount the mics facing the feeder, behind a windscreen, and out of direct
rain. They are MEMS parts with an exposed port; water on the port kills
them.

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

### Step 3: Enable the I2S Microphones

```bash
sudo ./scripts/enable_i2s_mics.sh
sudo reboot
```

The script backs up `/boot/firmware/config.txt`, enables
`dtparam=i2s=on` and `dtoverlay=googlevoicehat-soundcard`, and verifies
the overlay is installed. A reboot is required: device tree overlays are
read by firmware at boot and cannot be applied to a running kernel.

After reboot, verify:

```bash
arecord -l
# Should list card 0: snd_rpi_googlevoicehat_soundcard
```

### Step 4: Install Hailo Drivers

```bash
sudo ./scripts/install_hailo.sh
sudo reboot
```

After reboot, verify:

```bash
hailortcli fw-control identify
# Should show Hailo-8L device info
```

### Step 5: Download Models

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

`download_models.sh` also fetches BirdNET v2.4 (52 MB) from Zenodo and
verifies the TFL3 magic bytes. Those weights are **CC BY-NC-SA 4.0, not
GPL-3**, which is why they are downloaded rather than committed. Review
the terms before any commercial use.

The YOLO model was trained with `training/train_detector.py` on Open
Images V7 data. To retrain with your own feeder images, see
`training/README.md`.

### Step 6: Configure

```bash
cp config/default.yaml /opt/ratcatcher/config/
cp config/species.yaml /opt/ratcatcher/config/
```

Edit `/opt/ratcatcher/config/default.yaml` for your setup:
- Adjust camera resolution and FPS
- Set data_dir to `/opt/ratcatcher/data`
- Tune motion detection sensitivity for your feeder location
- Adjust retention days and max disk usage
- Set `audio.enabled: true` once test-mic reports OK on both channels

### Step 7: Test

```bash
source /opt/ratcatcher/.venv/bin/activate

# Verify cameras
ratcatcher test-camera --camera 0
ratcatcher test-camera --camera 1

# Verify microphones -- both channels must report OK
ratcatcher test-mic

# Optionally confirm BirdNET loads and runs on live audio
ratcatcher test-mic --seconds 30 --identify

# Verify system
ratcatcher health

# Test pipeline briefly
ratcatcher run --cameras 0
# Ctrl+C to stop
```

A healthy `test-mic` run reports a non-zero DC offset (the SPH0645
always has one) and an RMS in the -40s dBFS for ordinary ambient:

```
  channel             RMS       peak   status
  0 (left )      -46.3      -19.1   OK
  1 (right)      -47.7      -13.1   OK
```

`--identify` skips the activity gate by design and reports raw BirdNET
output, so it will name species on room noise. That is expected and is
not what the live pipeline does.

### Step 8: Start Service

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

-- Bird song identifications, with which mic heard them
SELECT timestamp, channel, species, common_name, confidence
FROM audio_detections ORDER BY timestamp DESC LIMIT 20;

-- Both modalities at once, via the unified view
SELECT modality, timestamp, source_id, species, common_name, confidence
FROM detections_all
WHERE timestamp >= date('now')
ORDER BY timestamp DESC;

-- Species seen AND heard today (they are logged independently,
-- so this join is the only thing correlating them)
SELECT species, common_name,
       SUM(modality = 'video') AS seen,
       SUM(modality = 'audio') AS heard
FROM detections_all
WHERE timestamp >= date('now') AND species IS NOT NULL
GROUP BY species ORDER BY seen + heard DESC;
```

## Outdoor Deployment Checklist

- [ ] Enclosure is IP65 rated with sealed cable glands
- [ ] Gore-Tex breathing vents installed to prevent condensation
- [ ] Desiccant packs placed inside enclosure
- [ ] Camera windows are optical glass (not acrylic, which yellows)
- [ ] Microphone ports shielded from direct rain, with windscreens fitted
- [ ] `ratcatcher test-mic` reports OK on both channels after final assembly
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

### Microphones silent or dead

First distinguish "no sound" from "no microphone". Record and inspect
the raw samples:

```bash
arecord -D hw:0,0 -c 2 -r 48000 -f S32_LE -d 3 /tmp/mic.wav
ratcatcher test-mic
```

A *constant* sample value across the whole file means no mic data at
all. In particular `0x00000001` on the left and `0xfffffffe` on the
right is LRCLK bleeding onto a floating GPIO20, not audio: laid out
across the 64-bit frame that pattern is the word-select line delayed by
one bit clock.

Confirm the data line is undriven by forcing an internal pull on GPIO20
during a capture:

```bash
pinctrl set 20 pd   # then record; all-zero samples means nothing drives the line
pinctrl set 20 pu   # then record; all-ones means the same
pinctrl set 20 pn   # restore
```

A live SPH0645 push-pull output easily overpowers the ~50k internal
pull, so if the pull wins, the mic is not driving DOUT. Check, in order:

1. **3V3 on both mics** -- header pin 1, not 5V
2. **DOUT landed on header pin 38 (GPIO20)**
3. **Common ground** -- pin 6; a missing ground floats DOUT even with power

If instead both channels carry *identical* samples, both SEL pins are
tied the same way and the mics are contending for one time slot. Two
healthy mics correlate strongly but are not identical -- expect only a
few percent of samples to match exactly, with a small cross-correlation
lag from their physical spacing.

Also verify the clocking side:

```bash
arecord -l                  # card 0 should be the googlevoicehat soundcard
pinctrl get 18-21           # expect a2: I2S0_SCLK / WS / SDI0 / SDO0
grep -E "i2s|voicehat" /boot/firmware/config.txt
```

### BirdNET reports species that are obviously not present

`test-mic --identify` runs BirdNET with no activity gate, so it reports
raw model output including low-confidence noise matches. The live
pipeline gates first. If false positives persist in the *database*,
raise `audio.min_confidence` (default 0.25) or `gate_snr_margin_db`
(default 2.0) -- but note the gate is tuned for recall: 4 dB already
loses 14% of windows containing a real bird, and 6 dB loses 48%.

BirdNET also covers 6522 classes globally, including non-bird labels
(Engine, Dog, Human), and is not restricted to local species.

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
