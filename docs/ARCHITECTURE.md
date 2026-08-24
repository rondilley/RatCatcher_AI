# Architecture -- RatCatcher AI

## System Overview

RatCatcher AI is a real-time wildlife detection system designed for
outdoor bird feeder monitoring. It runs on a Raspberry Pi 5 with two
cameras and a Hailo-8L AI accelerator, detecting pest animals and
identifying bird species by Genus and Species.

It has two independent detectors: a **video pipeline** (motion -> object
detection -> species classification) and an **audio pipeline** (activity
gate -> song identification). They share the SQLite database and nothing
else -- no queues, no locks, no shared state -- so either keeps working
when the other's hardware is unavailable.

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

## Audio Pipeline (Independent Detector)

```
+------------------+     +------------------+     +-------------------+
| I2S Capture      |     | Conditioning     |     | Activity Gate     |
| 2x SPH0645       | --> | DC block, split  | --> | SNR vs noise floor|
| arecord, 48 kHz  |     | 150 Hz highpass  |     | + spectral flatness|
+------------------+     +------------------+     +-------------------+
                                                          |
                                                   [sound present]
                                                          |
                                                +---------v---------+
                                                | BirdNET v2.4      |
                                                | TFLite FP32, CPU  |
                                                | 62 ms per window  |
                                                +-------------------+
                                                          |
                                                +---------v---------+
                                                | Log + WAV clip    |
                                                | audio_detections  |
                                                +-------------------+
```

### Capture

Two Adafruit SPH0645 I2S MEMS microphones share one I2S bus, separated
by their SEL pin, so ALSA presents them as a single stereo device:
channel 0 is the left mic (SEL to GND), channel 1 the right (SEL to
3V3). `ArecordSource` runs `arecord` as a subprocess and parses S32_LE
frames; `WavFileSource` substitutes a recording for development.

The `AudioSource` protocol exposes `is_realtime`, and the consumer picks
its backpressure policy from it. A live device must drop windows when
the consumer falls behind, because blocking the reader stalls the sound
card into ALSA overruns. A file must block instead, because it delivers
far faster than realtime and dropping silently discards the recording.
The property belongs on the producer because only the producer knows
whether falling behind is recoverable.

### Conditioning

The SPH0645 has no output coupling capacitor, so every sample carries a
large constant bias -- measured at roughly -0.044 full-scale on this
build -- that would otherwise dominate every energy measurement
downstream. DC removal is vectorised over the block rather than looped
per sample. Data arrives as 18 bits left-justified in a 32-bit slot.

### Activity Gate

The audio analogue of the motion pre-filter, but tuned far more
permissively, because the economics are different. Motion detection
guards a ~28 ms NPU inference and skips most frames. The gate guards a
62 ms CPU inference that costs about 4% of one core for two channels
running continuously, so there is little to save by rejecting a window
and everything to lose by rejecting a real bird.

Gating is per channel (the two mics have different ambients) and
frame-based rather than whole-window: whole-window flatness rejected
real warbles. Measured against a field soundscape with BirdNET output as
ground truth, a 2 dB SNR margin keeps 100% of windows containing a real
bird, 4 dB loses 14%, and 6 dB loses 48%. The gate earns its keep on
quiet nights and in steady rain, not in a dawn chorus -- where there is
no quiet baseline to measure against, because the birds *are* the
ambient sound.

### Identification

BirdNET v2.4, TFLite FP32, 52 MB, on CPU. The classifier reads its
window length and class count from the model file at load time rather
than hardcoding them, matching how the detection backends auto-detect
custom versus COCO models.

Coverage is global (6522 classes) and includes non-bird labels (Engine,
Dog, Human). It is not restricted to the ~50 Western US species in
`config/species.yaml`, and BirdNET's location/date meta-model, which
would narrow candidates by geography and season, is not used.

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

The audio engine runs its own two threads alongside, sharing nothing
with the above except the database:

```
I2S stereo --> audio-capture thread (DC block, accumulate 3 s windows)
                            |
                      window_queue
                            |
               audio-analysis thread (gate per channel -> BirdNET)
                            |
                SQLite audio_detections + WAV clip
```

Shutting down `arecord` needs one ordering detail: the read end of its
stdout pipe must be closed *before* signalling. A stopped consumer
leaves arecord blocked writing into a full pipe, where it never reaches
its SIGTERM handler -- `terminate()` alone waits the full timeout and
then needs SIGKILL. Closing the pipe first gives it EPIPE and it exits
immediately.

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

Audio identifications live in a separate table:

```sql
audio_detections (
    id, timestamp, channel,
    species, common_name, confidence,
    duration_seconds,
    band_rms_dbfs, spectral_flatness, peak_frequency_hz,
    clip_path, metadata
)
```

Indexed on timestamp, species, and channel. They are kept apart from
`detections` rather than merged: an audio event has no frame, no
bounding box and no camera, while it does have a channel, a duration and
acoustic measurements. One combined table would be a wide row that is
mostly NULL whichever modality wrote it, and would force relaxing the
NOT NULL constraint on `camera_id`.

A view restores the unified query surface without that cost:

```sql
CREATE VIEW detections_all AS
    SELECT 'video' AS modality, id, timestamp, camera_id AS source_id,
           class_name, species, common_name, confidence, clip_path
    FROM detections
  UNION ALL
    SELECT 'audio' AS modality, id, timestamp, channel AS source_id,
           'bird' AS class_name, species, common_name, confidence, clip_path
    FROM audio_detections;
```

`source_id` is the camera for video rows and the microphone channel for
audio rows. Note that the two modalities are logged independently and
are **not** correlated: a bird seen and heard at the same moment
produces two unlinked rows.

### Audio Clips

WAV, written with the Python standard library rather than FFmpeg, so
the audio path carries no external encoder dependency.

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

### Audio Factory (audio/factory.py)
```
source_type="auto" -> try ArecordSource (ALSA) -> fall back to WavFileSource
source_type="alsa" -> ArecordSource (requires arecord and an I2S device)
source_type="file" -> WavFileSource (16/24/32-bit WAV)
```

All three factories use lazy imports so unavailable backends don't cause
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
| BirdNET (TFLite FP32, 52 MB weights) | ~156 MB (measured) |
| Frame buffers (2 cameras) | ~200 MB |
| Audio window buffers (2 channels) | ~10 MB |
| SQLite + Python overhead | ~150 MB |
| **Total** | **~1.4 GB** |

Headroom: ~2.6 GB free on a 4 GB system running headless. BirdNET is the
single largest model in the system by file size -- it is FP32 where the
two vision models are quantized.

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
