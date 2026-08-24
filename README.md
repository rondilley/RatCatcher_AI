# RatCatcher AI

Wildlife detection system for bird feeders. Identifies birds to Genus
and Species by sight *and* by song, and detects pest animals (squirrels,
rats, cats) in real-time on a Raspberry Pi 5.

## How It Works

Two cameras watch bird feeders 24/7. When motion is detected, a YOLO
object detector (running on a Hailo-8L NPU at ~35 FPS) classifies the
visitor as a bird or pest. Birds are then passed to a MobileNet V2
species classifier that identifies the species from ~965 known birds,
with 50 Western US feeder species mapped by name. All detections are
logged to SQLite with thumbnails and optional video clips.

```
Camera -> Motion Filter -> YOLO Detection -> Species Classification -> SQLite + Clips
              (MOG2)       (Hailo NPU)        (TFLite, CPU)
```

Two I2S microphones listen in parallel, identifying bird song with
BirdNET. Audio is an **independent detector**, not a stage of the video
pipeline: it has its own capture and inference threads and meets the
video path only at the database. Song identification keeps working when
the cameras or the NPU are unavailable, and vice versa.

```
I2S Mics -> DC Block -> Activity Gate -> BirdNET -> SQLite + WAV Clips
  (x2)     (SPH0645)   (SNR + tonality)  (TFLite, CPU)
```

The `detections_all` SQL view unions both modalities behind a `modality`
column, so "what species were here today" stays a single query.

## Hardware

| Component | Model | Purpose |
|---|---|---|
| Computer | Raspberry Pi 5 (4GB) | Main controller |
| Cameras | 2x Arducam UC-517 B0270 | IMX477 12MP, IR-Cut for day/night |
| AI Accelerator | Hailo-8L AI HAT+ (13 TOPS) | Real-time object detection |
| Microphones | 2x Adafruit SPH0645 (I2S MEMS) | Bird song capture, shared bus |
| Power | UPS HAT + solar panel | Outdoor uninterrupted power |
| Enclosure | IP65 weatherproof | Outdoor deployment |

## Quick Start (Development)

Development and production both run on the Raspberry Pi 5. The camera
and audio layers fall back to file sources, so most work needs no
attached hardware.

```bash
# Clone and install
git clone <repo-url> && cd RatCatcher_AI
python3 -m venv venv
source venv/bin/activate

pip install -e ".[dev,audio]"

# Run tests (182 tests, no test doubles -- real WAVs, real SQLite)
venv/bin/pytest tests/ -v

# Check system info
ratcatcher health

# Run pipeline on a video file (uses OpenCV DNN backend on CPU)
ratcatcher run --source file --input path/to/video.mp4

# View detection statistics
ratcatcher stats --last 24h
```

## Quick Start (Raspberry Pi 5)

```bash
# Initial setup (installs system packages, creates venv, configures cameras)
sudo ./scripts/setup_rpi.sh

# Install Hailo AI HAT+ drivers
sudo ./scripts/install_hailo.sh

# Enable the I2S microphones (edits config.txt; reboot required)
sudo ./scripts/enable_i2s_mics.sh
sudo reboot

# Verify both microphones carry signal
ratcatcher test-mic

# Download ML models (includes BirdNET from Zenodo)
./scripts/download_models.sh

# Copy config
cp config/default.yaml /opt/ratcatcher/config/

# Start the service
sudo systemctl start ratcatcher
sudo journalctl -u ratcatcher -f
```

## CLI Commands

```
ratcatcher run                     # Start full detection pipeline
ratcatcher run --mode motion       # Motion detection only (no AI)
ratcatcher run --mode detect       # Detection without species ID
ratcatcher run --cameras 0         # Use only camera 0
ratcatcher run --source file --input video.mp4  # Process a video file
ratcatcher run --backend opencv_dnn # Force CPU detection backend

ratcatcher stats                   # Show all-time detection counts
ratcatcher stats --last 24h        # Last 24 hours
ratcatcher stats --last 7d         # Last 7 days

ratcatcher health                  # System health check

ratcatcher run --audio             # Force bird song detection on
ratcatcher run --no-audio          # Force it off, whatever the config says

ratcatcher test-camera             # Capture and save a test frame
ratcatcher test-camera --camera 1  # Test camera 1

ratcatcher test-mic                # Record 5s, report per-channel levels
ratcatcher test-mic --seconds 30   # Longer recording
ratcatcher test-mic --output a.wav # Save the recording
ratcatcher test-mic --identify     # Also run BirdNET on what was heard
```

Note that `test-mic --identify` deliberately skips the activity gate and
reports raw BirdNET output, so it will name species on room noise. The
live pipeline gates first.

## ML Models

### Object Detection (Stage 1)
- **Model:** YOLOv8n custom-trained on Open Images V7 (11.7 MB ONNX)
- **File:** `models/ratcatcher_best.onnx`
- **Backends:** Hailo-8L NPU (production), NCNN (CPU fallback), OpenCV DNN (universal fallback)
- **Classes:** bird, squirrel, rat, cat, unknown_animal
- **Trained metrics:** mAP@0.5 = 0.751 (squirrel 0.957, cat 0.896, rat 0.762, bird 0.608)
- **Performance:** ~35 FPS per camera on Hailo-8L; ~9-15 FPS CPU-only

### Species Classification (Stage 2)
- **Model:** MobileNet V2 iNaturalist Bird Classifier (INT8 quantized)
- **File:** `models/mobilenet_v2_inat_bird_quant.tflite` (3.6 MB)
- **Species:** 965 bird species (50 Western US feeder species mapped by name)
- **Input:** 224x224 RGB
- **Performance:** ~25-50ms per crop on RPi5 CPU

### Bird Song Identification (Audio, independent)
- **Model:** BirdNET v2.4 (TFLite FP32)
- **File:** `models/BirdNET_v2.4_audio-model.tflite` (52 MB)
- **Classes:** 6522, global coverage; includes non-bird labels (Dog, Engine, Human)
- **Input:** 3-second windows of 48 kHz audio
- **Performance:** 62 ms per window on RPi5 CPU -- about 48x realtime,
  roughly 4% of one core for two channels running continuously

**License note:** BirdNET's weights are **CC BY-NC-SA 4.0, not GPL-3.**
They are downloaded from Zenodo by `scripts/download_models.sh` and are
never committed to this repository. Review the terms before any
commercial use.

## Project Structure

```
config/
  default.yaml              System-wide configuration
  species.yaml              50 Western US bird species + 9 pest species

src/ratcatcher/
  cli.py                    CLI entry point (run/stats/health/test-mic/test-camera)
  config.py                 YAML config loading with frozen dataclasses
  camera/                   Camera abstraction (Picamera2, file, webcam)
  motion/                   MOG2 motion detection + ROI masking
  detection/                YOLO object detection (3 backends)
  classification/           MobileNet V2 species classification (TFLite)
  audio/                    I2S capture, conditioning, activity gate, BirdNET
  pipeline/                 Threaded engines (video and audio)
  storage/                  SQLite logging, FFmpeg clips, thumbnails
  monitoring/               System health + detection statistics

models/                     ML model files (.tflite, .onnx, .hef)
training/                   CUDA training pipeline (download, train, export)
scripts/                    RPi5 setup and model download scripts
systemd/                    Systemd service for auto-start
tests/                      182 tests (pytest, no test doubles)
docs/                       Architecture and deployment docs
```

## Training Your Own Detector

Train a custom YOLOv8n on a desktop GPU to improve pest detection:

```bash
# Install training deps (separate from RPi deployment)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r training/requirements.txt

# Download training data from Open Images V7 (~11K images)
python training/download_data.py --output datasets/ratcatcher --max-per-class 3000

# Train (~2.5 hours on RTX 5090)
python training/train_detector.py --data datasets/ratcatcher/dataset.yaml --epochs 100

# Export to ONNX for RPi deployment
python training/export_model.py --weights runs/detect/runs/train/ratcatcher/weights/best.pt
```

See `training/README.md` for full details.

## Configuration

All settings live in YAML files under `config/`. Override the config
directory with the `RATCATCHER_CONFIG_DIR` environment variable.

Key settings in `config/default.yaml`:
- Camera resolution, FPS, and source type
- Motion detection sensitivity and cooldown
- Detection backend and confidence thresholds
- Species classification model path and minimum confidence
- Clip recording pre/post seconds and disk retention
- Alert classes and notification methods
- Audio: ALSA device, activity gate thresholds, BirdNET confidence

Bird song detection is **off by default** (`audio.enabled: false`). Turn
it on once `ratcatcher test-mic` reports OK on both channels.

## 50 Target Species (Western US)

Jays, chickadees, nuthatches, woodpeckers, hummingbirds, finches,
sparrows, towhees, doves, thrushes, warblers, hawks, and more.
See `config/species.yaml` for the full list with Genus, Species,
Family, and model label indices.

Pest detection: Western Gray Squirrel, Fox Squirrel, California Ground
Squirrel, Norway Rat, Roof Rat, House Mouse, Raccoon, Opossum, Cat.

## Platform Support

| Platform | Camera | Detection | Classification | Audio |
|---|---|---|---|---|
| RPi5 (production) | Picamera2 | Hailo-8L NPU | ai-edge-litert | ALSA / I2S mics |
| RPi5 (no Hailo) | Picamera2 | NCNN (CPU) | ai-edge-litert | ALSA / I2S mics |
| Linux (dev) | FileSource / Webcam | NCNN or OpenCV DNN | ai-edge-litert | WAV file source |

Factory patterns in the camera, detection, and audio layers auto-select
a backend at startup, so an unavailable device degrades to a file source
rather than failing.

`tflite-runtime` publishes no wheels for Python 3.13, which current
Raspberry Pi OS ships. `ai-edge-litert`, its maintained successor, is
what the `audio` extra installs; both classifiers try `tflite_runtime`,
`ai_edge_litert`, then `tensorflow` in order.

## License

GPL-3.0-or-later. See `LICENSE`.

BirdNET model weights are **not** covered by that license -- they are
CC BY-NC-SA 4.0 and are fetched at runtime rather than vendored.
