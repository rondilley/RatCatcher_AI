# RatCatcher AI -- Pest Detector Training

Train a custom YOLOv8n model to detect birds, squirrels, rats, cats,
and other animals at bird feeders. Runs on a desktop GPU (NVIDIA CUDA),
deploys to the Raspberry Pi 5.

## Prerequisites

- NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- CUDA toolkit 11.8 or 12.x installed
- Python 3.11+
- ~20GB free disk space (for dataset + training artifacts)

## Setup

```bash
# Create a training venv (separate from the RPi deployment venv)
python -m venv .venv-training
source .venv-training/bin/activate  # Linux
# .\.venv-training\Scripts\Activate.ps1  # Windows

# Install PyTorch with CUDA (adjust cu121 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
pip install -r training/requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

## Step 1: Download Training Data

Downloads labeled animal images from Open Images V7 (Google's dataset).
No API keys needed.

```bash
# Download ~3000 images per class (recommended for good results)
python training/download_data.py --output datasets/ratcatcher --max-per-class 3000

# For a quick test run (smaller dataset, faster)
python training/download_data.py --output datasets/ratcatcher --max-per-class 200
```

This downloads bounding-box-annotated images for:
- **Bird** -- from Open Images "Bird" label
- **Squirrel** -- from Open Images "Squirrel" label
- **Rat** -- from Open Images "Mouse" label (morphologically close)
- **Cat** -- from Open Images "Cat" label
- **Unknown animal** -- from Open Images "Raccoon", "Rabbit", "Skunk"

The download is resumable -- re-running skips already-downloaded images.

The first run downloads a 2.2GB annotation CSV (cached for future runs).

## Step 2: Train

```bash
# Full training (~2-4 hours on RTX 3080)
python training/train_detector.py --data datasets/ratcatcher/dataset.yaml --epochs 100

# Quick test (verify setup works)
python training/train_detector.py --data datasets/ratcatcher/dataset.yaml --epochs 5

# Resume interrupted training
python training/train_detector.py --data datasets/ratcatcher/dataset.yaml --resume
```

Training options:
- `--batch 32` -- increase batch size if you have VRAM (default 16)
- `--imgsz 640` -- image size (must match RPi inference size)
- `--device 0` -- GPU device ID (default: auto-detect)
- `--patience 20` -- early stopping epochs without improvement

Output: `runs/train/ratcatcher/weights/best.pt`

The training script prints mAP, precision, and recall for each class.

## Step 3: Export for Raspberry Pi

```bash
# Export to ONNX (for OpenCV DNN backend)
python training/export_model.py --weights runs/train/ratcatcher/weights/best.pt

# Export to both ONNX and NCNN
python training/export_model.py --weights runs/train/ratcatcher/weights/best.pt --formats onnx,ncnn
```

Output files in `models/`:
- `ratcatcher_yolov8n.onnx` -- for OpenCV DNN backend
- `ratcatcher_yolov8n_ncnn_model/` -- for NCNN backend

## Step 4: Deploy to Raspberry Pi

```bash
# Copy the model to the RPi
scp models/ratcatcher_yolov8n.onnx pi@raspberrypi:/opt/ratcatcher/models/

# On the RPi, update the config
# Edit /opt/ratcatcher/config/default.yaml:
#   detection:
#     model_path: "ratcatcher_yolov8n.onnx"

# Restart the service
sudo systemctl restart ratcatcher
```

The detection backends auto-detect whether the model has 80 COCO classes
or 5 RatCatcher classes and adjust the class mapping accordingly.

### Hailo HEF Export

Hailo HEF conversion requires the Hailo Dataflow Compiler (DFC), which
is not pip-installable. On a machine with the DFC SDK:

```bash
hailo parser onnx models/ratcatcher_yolov8n.onnx
hailo optimize ratcatcher_yolov8n.har --hw-arch hailo8l
hailo compile ratcatcher_yolov8n_optimized.har --hw-arch hailo8l -o models/ratcatcher_yolov8n.hef
```

## Training Data Classes

| YOLO ID | Class | Open Images Source | Notes |
|---|---|---|---|
| 0 | bird | Bird (/m/015p6) | Very common in OI |
| 1 | squirrel | Squirrel (/m/071qp) | Hundreds of images |
| 2 | rat | Mouse (/m/04rmv) | Proxy for rat detection |
| 3 | cat | Cat (/m/01yrx) | Very common in OI |
| 4 | unknown_animal | Raccoon, Rabbit, Skunk | Catch-all pest class |

## Expected Results

With 3000 images per class and 100 epochs, expect:
- Bird mAP@0.5: 0.70-0.85
- Cat mAP@0.5: 0.75-0.90
- Squirrel mAP@0.5: 0.50-0.70 (fewer training images)
- Overall mAP@0.5: 0.60-0.80

These numbers improve with more training data, especially from your
own bird feeder cameras.

## Adding Your Own Training Data

To add custom images from your feeder cameras:

1. Place images in `datasets/ratcatcher/train/images/`
2. Create matching YOLO label files in `datasets/ratcatcher/train/labels/`
3. Label format: `class_id x_center y_center width height` (normalized 0-1)
4. Re-run training

Tools for labeling: [Label Studio](https://labelstud.io/),
[CVAT](https://www.cvat.ai/), [Roboflow](https://roboflow.com/)

## Troubleshooting

**CUDA out of memory:** Reduce `--batch` (try 8 or 4).

**Download hangs:** The train annotations CSV is 2.2 GB. Be patient on
the first download. Re-run to resume.

**Low mAP for squirrel/rat:** These classes have fewer training images.
Add more from your own cameras or Roboflow datasets.

**Model not detecting on RPi:** Verify `config/default.yaml` has the
correct `model_path`. Run `ratcatcher health` to check backends.
