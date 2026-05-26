# RatCatcher AI

Wildlife detection system for bird feeders. Identifies birds to Genus
and Species, and detects pest animals (squirrels, rats, cats) in
real-time using a two-stage neural network pipeline on a Raspberry Pi 5.

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

## Hardware

| Component | Model | Purpose |
|---|---|---|
| Computer | Raspberry Pi 5 (4GB) | Main controller |
| Cameras | 2x Arducam UC-517 B0270 | IMX477 12MP, IR-Cut for day/night |
| AI Accelerator | Hailo-8L AI HAT+ (13 TOPS) | Real-time object detection |
| Power | UPS HAT + solar panel | Outdoor uninterrupted power |
| Enclosure | IP65 weatherproof | Outdoor deployment |

## Quick Start (Development)

Develop on Windows or Linux. No RPi hardware needed.

```bash
# Clone and install
git clone <repo-url> && cd RatCatcher_AI
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -e ".[dev]"

# Run tests (51 tests)
pytest tests/ -v

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

# Download ML models
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

ratcatcher test-camera             # Capture and save a test frame
ratcatcher test-camera --camera 1  # Test camera 1
```

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

## Project Structure

```
config/
  default.yaml              System-wide configuration
  species.yaml              50 Western US bird species + 9 pest species

src/ratcatcher/
  cli.py                    CLI entry point (run/stats/health/test-camera)
  config.py                 YAML config loading with frozen dataclasses
  camera/                   Camera abstraction (Picamera2, file, webcam)
  motion/                   MOG2 motion detection + ROI masking
  detection/                YOLO object detection (3 backends)
  classification/           MobileNet V2 species classification (TFLite)
  pipeline/                 Threaded pipeline engine
  storage/                  SQLite logging, FFmpeg clips, thumbnails
  monitoring/               System health + detection statistics

models/                     ML model files (.tflite, .onnx, .hef)
training/                   CUDA training pipeline (download, train, export)
scripts/                    RPi5 setup and model download scripts
systemd/                    Systemd service for auto-start
tests/                      51 tests (pytest)
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

## 50 Target Species (Western US)

Jays, chickadees, nuthatches, woodpeckers, hummingbirds, finches,
sparrows, towhees, doves, thrushes, warblers, hawks, and more.
See `config/species.yaml` for the full list with Genus, Species,
Family, and model label indices.

Pest detection: Western Gray Squirrel, Fox Squirrel, California Ground
Squirrel, Norway Rat, Roof Rat, House Mouse, Raccoon, Opossum, Cat.

## Platform Support

| Platform | Camera | Detection | Classification |
|---|---|---|---|
| RPi5 (production) | Picamera2 | Hailo-8L NPU | TFLite Runtime |
| RPi5 (no Hailo) | Picamera2 | NCNN (CPU) | TFLite Runtime |
| Windows (dev) | FileSource / Webcam | OpenCV DNN | TensorFlow Lite |
| Linux (dev) | FileSource / Webcam | NCNN or OpenCV DNN | TFLite Runtime |

## License

MIT
