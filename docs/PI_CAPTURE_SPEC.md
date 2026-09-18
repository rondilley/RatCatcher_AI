# Specification: Training Capture Mode on the Raspberry Pi

This document specifies a change to the Raspberry Pi code. The change
lets the deployment collect its own training data.

`docs/MODEL_TRAINING.md` section 4.3 gives deployment hard negatives
the highest value for each hour of work. The database has 1232 non-bird
detections that are incorrect. They come from this camera, this lens and
this scene: clouds, the plant pot on its hook, the shepherd's hook, the
cypress tree and the feeder roof. These images teach the model that
these objects are not animals.

The GPU work does not include this change. The GPU work uses generic
garden images from Open Images. Those images decrease the incorrect
detections, but they cannot remove the ones that these objects cause.
Only data from this deployment does that.

**Status: this document is a specification. Nobody wrote the code.**

---

## 1. Why the deployment cannot supply training data today

The system writes three record types. No record type is sufficient.

**Thumbnails show a crop, not the detector input.** `thumbnail_path`
points to a patch of approximately two times the detection box, with a
floor of 256 px. `storage/thumbnail.py` makes this patch for a person to
examine. A model cannot train on it, because the patch is not the image
that the detector received.

**The system discards the detector input.** The camera loop cuts up to
two 640x640 windows from the 4056x3040 sensor image. `_detect_for` in
`pipeline/engine.py` reads these windows. The loop then makes a new
`DetectionEvent` for each detection, and that event has no `crops`
field. This design is deliberate. `storage_queue` holds 256 entries, and
each window is 1.2 MB, thus the queue could hold more than one gigabyte
on a 4 GB board.

**The system writes no video clips.** `clip_path` is NULL on all 4073 rows, and
`data/clips/` is empty. `pipeline/engine.py` makes a `ClipWriter` at
line 167 and reads `clip_path` at line 725. No line of code calls the
writer. This specification does not correct that defect. It is a
different fault, and this document reports it only.

---

## 2. The design

### 2.1 Write the window where the code has it

`_detect_for` in `pipeline/engine.py` is the correct location. At that
point the code has the 640x640 window and the box in window coordinates
together. `_cut_detail` cuts a patch at this same location for the same cause.

This location removes the memory problem in section 1. The code makes a
JPEG from the window and writes it to disk immediately. The window does
not go on the event, thus `storage_queue` does not become larger. A
640x640 JPEG at quality 90 is approximately 90 KB. The raw array is
1.2 MB.

### 2.2 What to write

Write two files for each captured window.

An image file with the window as JPEG:

```
data/capture/20260830_142317_412_cam0_w0.jpg
```

A label file with the boxes in YOLO format, in window coordinates:

```
data/capture/20260830_142317_412_cam0_w0.txt
```

The label file uses the same format as the training set: one line for
each box, with `class_id cx cy w h`, normalised to the window. A window
with no detection gets an empty label file. That empty file is correct
here, because the detector found no animal in that window.

### 2.3 What to capture

Capture windows in two conditions.

Capture a window when the detector reports a detection. These become the
hard negatives after a person examines them, because most of these
detections are incorrect.

Capture a window when motion fired and the detector reported no animal.
These images show this scene with no animal in it. They cost one JPEG each, and
they are the images that show empty sky, cloud and the plant pot.

Use a sample rate for the second condition. Motion fires more frequently
than detection.

---

## 3. Configuration

Add a `capture:` section to `config/default.yaml`. The file is a dpkg
conffile. Thus an upgrade that keeps the previous file has no `capture:`
section. All defaults must give the current behaviour.

```yaml
capture:
  # Off by default. This mode writes images continuously and the root
  # filesystem has little free space. See section 4.
  enabled: false
  directory: "data/capture"
  # Capture each window that gave a detection.
  on_detection: true
  # Capture 1 window in N that gave motion but no detection.
  motion_sample_rate: 20
  # Stop when the directory reaches this size. Do not delete.
  max_gb: 4.0
  jpeg_quality: 90
```

`max_gb` must stop the capture and must not delete files. Deletion removes data in a
manner that a person cannot see. The logs show a capture that stops.

---

## 4. Constraints on the Raspberry Pi

**Disk space is the primary constraint.** The root filesystem was 89
percent full with 3.3 GB free on 2026-08-30. Audio clips use 3.5 GB.
Attach external storage, or move the data off the machine on a schedule,
before you start a collection.

**Retention deletes the collection.** `storage.retention_days` is 30 and
`storage.max_disk_gb` is 10. `storage/retention.py` must not delete the
capture directory. Retention applies to detection clips and thumbnails.
Training data has a different life cycle.

**The capture must not stop the cameras.** A full disk, a permission
fault or a slow write must not stop `_detect_for`. Catch `OSError` at
the write, count the failure, and continue.

---

## 5. How to know that it operates

1. Set `capture.enabled: true` and restart the service.
2. Wait for one hour of daylight.
3. `ls data/capture/*.jpg | wc -l` gives a count above zero.
4. Each `.jpg` file has a `.txt` file with the same name.
5. Open one image. Its dimensions are 640x640.
6. Draw the boxes from the `.txt` file on the image. The boxes agree
   with the objects.

Step 6 is the important one. It tests the coordinate transform. The
previous `capture_scale` defect made each stored box 1.333 times too
tall for the life of the deployment. A capture with incorrect boxes is
worse than no capture, because it trains the model on incorrect data.

---

## 6. How to use the result

1. Copy `data/capture/` from the Raspberry Pi to the training machine.
2. Examine the images with a contact sheet. Group them into two sets.
3. Images with no animal become background images. Make the label file
   empty.
4. Images with an animal in them keep their boxes. Correct the boxes by
   hand if they are not accurate.
5. Give the two sets to `training/build_dataset.py` as a fourth source.
6. Hold back a part of the images. These become the field validation set
   in `docs/MODEL_TRAINING.md` section 5.

Step 6 gives the measurement that this project does not have. The
current test procedure uses generic garden images. It cannot report
incorrect detections for each camera-hour on the deployment scene.
