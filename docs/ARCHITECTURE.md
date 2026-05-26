# Architecture -- RatCatcher AI

## System Overview

RatCatcher AI is a real-time wildlife detection system designed for
outdoor bird feeder monitoring. It runs on a Raspberry Pi 5 with two
cameras and a Hailo-8L AI accelerator, detecting pest animals and
identifying bird species by Genus and Species.

## Pipeline Architecture

The system uses a three-stage pipeline with motion pre-filtering:

```
+------------------+     +------------------+     +-------------------+
| Camera Capture   |     | Motion Detection |     | Object Detection  |
| (per camera)     | --> | MOG2 @ 320x240   | --> | YOLOv8n on Hailo  |
| Picamera2 thread |     | ~1ms per frame   |     | ~28ms per frame   |
+------------------+     +------------------+     +-------------------+
                                                          |
                              +---------------------------+
                              |
                    +---------+-----------+
                    |                     |
              [bird detected]       [pest detected]
                    |                     |
          +---------v---------+   +-------v--------+
          | Species Classify  |   | Log + Alert    |
          | MobileNet V2 INT8 |   | SQLite + thumb |
          | ~25-50ms per crop |   +----------------+
          +-------------------+
                    |
          +---------v---------+
          | Log + Clip + Thumb|
          | SQLite + FFmpeg   |
          +-------------------+
```

### Stage 0: Motion Pre-filter

Runs on every frame at low resolution (320x240) using OpenCV MOG2
background subtraction. Frames without motion skip all neural network
inference entirely. This saves ~90% of compute since most frames
contain no activity.

Key parameters:
- Processing resolution: 320x240 (independent of camera resolution)
- Morphological cleanup: erode (3x3) then dilate (7x7)
- Minimum contour area: 0.5% of frame area
- Grid-based cooldown (8x6 cells, 2s default) prevents duplicate detections
- Optional ROI polygon masking to restrict detection zones

### Stage 1: Object Detection (YOLO)

Runs only on frames where motion was detected. YOLOv8n classifies
detected objects into five categories:

| Class | ID | Description |
|---|---|---|
| bird | 0 | Any bird (passed to Stage 2) |
| squirrel | 1 | Squirrel species |
| rat | 2 | Rats, mice |
| cat | 3 | Domestic cats |
| unknown_animal | 4 | Other animals |

Three interchangeable backends:

| Backend | Hardware | Performance | Use Case |
|---|---|---|---|
| HailoDetector | Hailo-8L NPU | ~35 FPS/camera | Production (RPi5) |
| NCNNDetector | ARM CPU | ~12 FPS | RPi5 without Hailo |
| OpenCVDetector | Any CPU | ~4-8 FPS | Development (Windows/Linux) |

The factory in `detection/factory.py` auto-selects the best available
backend at startup.

**Custom-trained model:** A YOLOv8n trained on Open Images V7 data
(`models/ratcatcher_best.onnx`, 11.7 MB) outputs our 5 classes directly.
Backends auto-detect this by checking output tensor shape (9 values per
detection = 5 classes + 4 box coords). No COCO remapping needed.

**COCO fallback:** If using COCO-pretrained weights instead, backends
apply `map_coco_class()` to remap COCO IDs (bird=14, cat=15, other
animals=unknown_animal). Squirrels and rats are not in COCO.

### Stage 2: Species Classification

Runs only when Stage 1 detects a bird. Crops the bird region from the
full-resolution frame and classifies it.

- **Model:** MobileNet V2 iNaturalist Bird Classifier
- **Format:** TFLite INT8 quantized (3.6 MB)
- **Species:** 965 bird species; 50 Western US feeder species in taxonomy
- **Input:** 224x224 RGB
- **Output:** Softmax over 965 classes
- **Threshold:** Predictions below 70% confidence reported as unknown

The taxonomy maps model output indices to species info (Genus, Species,
Common Name, Family). Species not in the taxonomy appear as "unknown_NNN".

## Threading Model

```
Thread              Queue                Thread              Queue
camera-0 --------+                  +-----> classification_queue
                  +--> detection_queue                        |
camera-1 --------+        |         |   classification ------+
                           |         |                        |
                    detection -------+                 storage_queue
                                                              |
                                                       storage -----> SQLite + clips
```

Each camera runs in its own thread, producing frames into a shared
detection queue. The detection thread processes one frame at a time
(on Hailo NPU or CPU). Bird detections go to the classification queue;
pest detections go directly to storage. The storage thread writes to
SQLite and creates thumbnails.

All queues are bounded (64-256 items). When a queue is full, frames are
dropped and a counter incremented. This prevents memory exhaustion
under load while maintaining real-time responsiveness.

Graceful shutdown uses a sentinel object pattern: the main thread puts
a sentinel on each queue, and worker threads exit when they dequeue it.

## Data Storage

### SQLite Database

WAL journal mode for concurrent read/write. Schema:

```sql
detections (
    id, timestamp, camera_id, stage,
    class_name, species, common_name, confidence,
    bbox_x, bbox_y, bbox_w, bbox_h,
    clip_path, thumbnail_path,
    frame_width, frame_height, metadata
)
```

Indexed on timestamp, species, and camera_id. The metadata column
stores JSON (e.g., top-K classification results).

### Video Clips

Optional H.264 MP4 clips recorded around detection events:
- Pre-event buffer: 5 seconds (ring buffer of recent frames)
- Post-event recording: 10 seconds
- Encoding: FFmpeg via subprocess pipe (raw frames -> H.264)

### Thumbnails

JPEG images with bounding box overlay, resized to 320px on the longest
edge. One thumbnail per detection event.

### Retention

Automatic cleanup policy:
- Delete clips/thumbnails older than 30 days (configurable)
- Enforce maximum disk usage (10 GB default)
- Oldest files deleted first when over limit

## Configuration Architecture

All configuration in YAML, loaded into frozen dataclasses at startup.

```
config/
  default.yaml     All settings with defaults
  species.yaml     Species taxonomy + model label index mapping
```

Configuration hierarchy:
1. Hardcoded defaults in dataclass definitions
2. Overridden by YAML file values
3. Config directory overridden by RATCATCHER_CONFIG_DIR env var
4. CLI arguments override specific settings at runtime

## Platform Abstraction

Two factory patterns provide cross-platform support:

### Camera Factory (camera/platform_camera.py)
```
source_type="auto" -> try Picamera2 -> fall back to WebcamSource
source_type="file" -> FileSource (video file or image directory)
source_type="webcam" -> WebcamSource (OpenCV VideoCapture)
source_type="picamera" -> PicameraSource (Picamera2, RPi only)
```

### Detection Factory (detection/factory.py)
```
backend="auto" -> try Hailo -> try NCNN -> fall back to OpenCV DNN
backend="hailo" -> HailoDetector (RPi + Hailo-8L only)
backend="ncnn" -> NCNNDetector (requires ncnn Python package)
backend="opencv_dnn" -> OpenCVDetector (always available)
```

Both factories use lazy imports so unavailable backends don't cause
import errors.

## Training Pipeline (Desktop CUDA)

The `training/` directory contains a self-contained pipeline for
training custom detection models on a desktop GPU.

```
Open Images V7 (Google Cloud Storage)
    |
    v
download_data.py  -- filter by class MID, download images + boxes
    |                 Convert to YOLO format, split train/val
    v
datasets/ratcatcher/
  train/images/ + labels/
  val/images/ + labels/
  dataset.yaml
    |
    v
train_detector.py  -- Ultralytics YOLO fine-tuning on CUDA
    |                  Start from yolov8n.pt (COCO pretrained)
    v
runs/train/ratcatcher/weights/best.pt
    |
    v
export_model.py  -- Export to ONNX, NCNN, or Hailo HEF
    |
    v
models/ratcatcher_best.onnx  -- Deploy to RPi
```

**Data sources:** Open Images V7 with direct HTTP downloads from S3.
No API keys, no heavy dependencies. Images filtered by class MID codes
and quality flags (exclude groups, depictions, occluded).

**Training:** Ultralytics YOLOv8 with PyTorch CUDA. 100 epochs,
early stopping, mosaic augmentation. Results: mAP@0.5 = 0.751 on
~10K images across 5 classes.

**Custom model detection:** Backends auto-detect 5-class models by
probing output tensor shape at load time (OpenCV DNN) or at inference
time (NCNN, Hailo). If output has 9 values per detection (4 box + 5
class scores), COCO remapping is skipped.

## Memory Budget (4GB RPi5)

| Component | Estimated RAM |
|---|---|
| OS (headless) | ~300 MB |
| Python + OpenCV + Picamera2 | ~300 MB |
| YOLO model (Hailo/NCNN) | ~200 MB |
| Species classifier (TFLite INT8) | ~50 MB |
| Frame buffers (2 cameras) | ~200 MB |
| SQLite + Python overhead | ~150 MB |
| **Total** | **~1.2 GB** |

Headroom: ~2.8 GB free on a 4 GB system running headless.

## Deployment

### Systemd Service
- Auto-start on boot with `ratcatcher.service`
- Restart on failure (10s delay)
- Watchdog timer (60s)
- Memory limit: 3 GB (prevents OOM-killing other services)
- Security hardening: NoNewPrivileges, ProtectSystem=strict

### Outdoor Considerations
- IR-Cut cameras for day/night operation
- UPS HAT for power resilience
- IP65 enclosure with Gore-Tex vents
- Active cooling (RPi5 throttles at 80C)
- Nightly scheduled reboot for long-term stability
