# Training the RatCatcher Detector

A brief for retraining the object detector on a GPU workstation.

This document is written to be read by a coding agent with no prior
context on the deployment. It states what the currently deployed model
does in the field, why it does it, what data has to change, and what
"correct" looks like when the work is done. Every number in it was
measured on the production Raspberry Pi 5 on 2026-08-30 unless marked
otherwise.

`training/README.md` holds the mechanical steps -- venv setup, script
invocations, export commands. This document holds the reasoning and the
requirements. Read this one first; do not follow `training/README.md`
blindly, because parts of the workflow it describes are what produced
the failures below.

---

## 1. The situation

The deployed detector is a YOLOv8n trained on Open Images V7, running on
a Hailo-8 NPU. On its own validation split it reports:

```
mAP@0.5 = 0.751    squirrel 0.957   cat 0.896   rat 0.762   bird 0.608
```

In the field, over 4073 stored detections, the non-bird classes are
approximately 100 percent false positives.

| class | rows stored | what the stored thumbnails actually show |
|---|---:|---|
| bird | 2432 | mostly real birds |
| squirrel | 607 | a cypress tree; a bird on the feeder roof |
| unknown_animal | 518 | clouds, a hanging plant pot, a shepherd's hook, birds |
| cat | 101 | clouds and a hanging plant pot. Zero cats |
| rat | 6 | the shepherd's hook, twice |

The audit was visual: contact sheets built from every surviving
thumbnail, sampled across the full confidence range. Not one real
mammal appeared in the sample. Retention had already pruned most
thumbnails, so the direct visual evidence is strong for `cat` (18
thumbnails) and `unknown_animal` (82), and thin for `squirrel` (2 of
607). Treat the squirrel figure as consistent with the pattern rather
than as independently proven.

Confidence does not separate the classes. Per-class confidence over all
stored rows:

| class | min | mean | max |
|---|---:|---:|---:|
| bird | 0.451 | 0.638 | 0.969 |
| squirrel | 0.451 | 0.603 | 0.935 |
| unknown_animal | 0.451 | 0.569 | 0.917 |
| cat | 0.451 | 0.577 | 0.906 |
| rat | 0.473 | 0.570 | 0.719 |

The highest-confidence surviving `cat` thumbnail is a cloud at 0.81.
The class maximum is 0.906. Real birds span 0.45 to 0.97. There is no
threshold that separates a cloud from an animal, so raising
`detection.confidence_threshold` is not a fix and must not be proposed
as one.

The project is named for rat detection. It has never detected a rat.

---

## 2. Root causes

Four causes, all in the data or the deployment envelope rather than in
the inference code. They are independent; fixing any three still leaves
a broken detector.

### 2.1 The training set contains zero background images

Measured over the committed dataset:

```
train: 9730 label files, 0 EMPTY (background/negative images)
val:   1717 label files, 0 EMPTY (background/negative images)
```

Every training image contains at least one animal. The model has never
been shown an image whose correct answer is "nothing here". A detector
trained this way has no representation of empty sky, and will place a
box on whatever most resembles an animal in whatever it is given.
Ultralytics recommends roughly 10 percent background images
specifically to suppress this failure.

This is the single largest contributor to the cloud detections.

### 2.2 The "rat" class is trained on pet hamsters

`training/download_data.py:37` maps Open Images `Mouse` (`/m/04rmv`) to
class id 2, named "rat". Open Images has no rat label, and the comment
in `training/README.md` describes the substitution as "a close shape".

Inspection of the actual class-2 training crops shows they are
overwhelmingly:

- Syrian and dwarf hamsters, in cages, on blankets, held in hands
- gerbils, voles, dormice, one chinchilla
- a small number of white laboratory mice
- approximately two images in twenty-four resembling a wild rat

and almost all are extreme close-ups, indoors, under domestic lighting.

The learned concept is "a large close-up furry rodent face indoors". A
wild rat at a feeder is a small, distant, side-on silhouette outdoors,
frequently at night. The two distributions barely intersect.

Class 2 is also the smallest class in the set, by a factor of 2.4:

| class id | name | train instances | val instances |
|---:|---|---:|---:|
| 0 | bird | 6002 | 1165 |
| 1 | squirrel | 1657 | 264 |
| 2 | rat | 704 | 117 |
| 3 | cat | 2992 | 546 |
| 4 | unknown_animal | 4417 | 764 |

The reported `rat mAP@0.5 = 0.762` is real. It measures the model's
ability to find hamsters in pet photography. It carries no information
about finding rats at a bird feeder, and it is the reason the defect
survived to deployment.

### 2.3 The validation set cannot predict field behaviour

Both splits are drawn from Open Images. The val split therefore shares
every bias of the train split: studio and hobbyist photographs, animals
centred and large in frame, daylight, no empty scenes. A model can
score 0.751 on it while placing boxes on clouds all day.

No amount of retraining fixes this while the val set stays as it is.
A held-out set drawn from the deployment itself is a hard requirement,
not an improvement. See section 5.

### 2.4 The video pipeline is blind at night

Detections by hour, all time, video against audio. Audio is the control:
it shares the database and the machine, so a non-zero audio count proves
the box was powered and processing during that hour.

```
hour   video   audio
00       0     1189
01       0     1086
02       0      810
03       0      521
04       0      756
05      14     1106
...
19      59     1734
20       1     2248
21       0     2663
22       0     1534
23       0     1501
```

Zero video detections from 21:00 to 04:00, against 10,060 audio
detections over the same eight hours. Video stops at dusk and resumes
at 05:00.

There was no infrared illuminator in the bill of materials. The
Arducam IMX477 modules have an IR-Cut auto-switch, so they can see
infrared, but nothing was emitting any. An 850 nm illuminator has since
been ordered and is not yet installed.

Rats are nocturnal. The system as built has never had the opportunity
to see one, so no change to the model alone can produce a rat
detection.

---

## 3. Deployment envelope the model must satisfy

The retrained model runs on fixed hardware with fixed preprocessing.
These are constraints, not preferences.

**Target and format.** Hailo-8 NPU, 26 TOPS. `hailortcli fw-control
identify` reports `Device Architecture: HAILO8`. Compile for `hailo8`,
not `hailo8l`: an 8L HEF loads on a Hailo-8 and runs slower, while a
hailo8 HEF will not load on an 8L at all. The currently deployed HEF is
already `models/ratcatcher_best_hailo8.hef`, symlinked as
`models/ratcatcher_best.hef`.

**Input.** 640x640, uint8 0-255. Normalization (mean 0, std 255) is
compiled into the HEF so the runtime can send uint8 over PCIe at a
quarter the bandwidth of float32. `HailoDetector` feeds uint8 on that
assumption. Change one without the other and every score collapses.

**Classes.** Five, in this order, emitted directly with no COCO
remapping: `bird(0), squirrel(1), rat(2), cat(3), unknown_animal(4)`.
Both Hailo output paths check the class count -- the raw-tensor path
infers it from the tensor channels, the on-chip-NMS path reads it from
the HEF at load. Do not change the count or the order without updating
`config/species.yaml` and the detection backends together.

**Aspect handling: stretch, not letterbox.** Both shipped backends
resize the 640x640 window by a plain stretch and rescale boxes with
independent scale_x and scale_y. Ultralytics trains with letterbox.
This is a known, pre-existing train/deploy mismatch. It is consistent
across backends, so it is an accuracy opportunity rather than a
correctness bug -- but if you change it, you must change the calibration
preprocessing in `training/build_calibration_set.py` and both backends
in the same commit, because the calibration set has to match what
actually runs.

**What the detector is actually handed.** Not the whole frame. The
camera loop reads the full 4056x3040 sensor and cuts up to two 640x640
windows out of it around motion (`detection/roi_crop.py`,
`detection.roi_crop_window: 640`, `roi_crop_max_windows: 2`). Nothing is
rescaled on the way in. This matters for training data: the model sees
native-resolution 640x640 crops of a yard, not downscaled whole frames.

**Object sizes in those windows.** Derived independently from a masonry
course and from the feeder body measured at 100x71 px, the scene runs
0.18 to 0.24 px/mm. In a native-resolution window a House Finch is about
62 px tall and a squirrel about 110 px. Detector recall against object
size collapses below roughly 100 px and is zero at 50 px. Measured on
100 labelled crops composited into a live frame from this Pi:

| species | full frame | ROI at 1080p | ROI at sensor res |
|---|---:|---:|---:|
| House Finch | 0/100 | 1/100 | 38/100 |
| Squirrel | 1/100 | 25/100 | 55/100 |
| Cat | 2/100 | 50/100 | 78/100 |

Training crops should reflect that size range. A dataset of animals
filling the frame trains a model that cannot find a 62 px bird.

---

## 4. What data to collect

Four separate collections. They are listed in order of impact on the
stated goal (detecting rats).

### 4.1 Night-time infrared footage from the deployment

Blocked until the illuminator is mounted. Nothing else recovers rat
detection without it.

Once mounted, note that night frames are a different visual domain, not
merely darker ones. With the IR-Cut filter swung out, all three channels
see infrared: the image is effectively monochrome, foliage reflects
brightly, and fur reflectance bears no relation to daylight colour. A
model trained only on daylight RGB is out of distribution at night in
exactly the way it is currently out of distribution on clouds.

Collect several full nights before training. Include:

- clear nights and rain (rain close to the lens is brilliantly lit by a
  near-field illuminator and is the classic night false positive)
- spider webs across the lens, which appear within days outdoors
- moths and insects near the illuminator
- the empty scene, at length. These become background images.

### 4.2 Real wild rats

Open Images cannot supply these. Candidate sources, in rough order of
usefulness:

- iNaturalist research-grade observations of *Rattus norvegicus* and
  *Rattus rattus*. Real animals, outdoors, varied distance and pose.
  Licences vary per observation and must be checked individually; many
  are CC BY-NC, which matters because this repository is GPL-3.0.
- Camera-trap datasets containing commensal rodents. These have the
  right framing, distance and lighting, which is worth more than image
  count.
- The deployment itself, once the illuminator is running. This is the
  best data and the slowest to obtain.

Target at least parity with the other classes, so on the order of 2500
to 3000 instances rather than 704. Prioritise: outdoors, at distance,
side-on, at night, in infrared. Discard the pet-hamster imagery
entirely rather than adding to it -- it is actively teaching the wrong
concept.

Also consider whether `rat` and `squirrel` should remain separate
classes. Both are the same alarm to the user ("a rodent is at the
feeder"), they are frequently confused at low pixel counts, and merging
them would put roughly 4300 instances behind one well-populated class.
This is a design decision for the maintainer, not one to take
unilaterally: it changes `config/species.yaml`, the class count checks
in both Hailo output paths, and `_VIDEO_CATEGORY` in
`monitoring/stats.py`.

### 4.3 Hard negatives from this deployment

The highest value per unit of effort, and available today without new
hardware. The database holds 1232 non-bird detections that are known
false positives from this exact scene, framing, and lens: clouds, the
hanging plant pot, the shepherd's hook, the cypress tree, the feeder
roof.

Added as background images -- present in `train/images/` with a
corresponding empty `.txt` in `train/labels/` -- these directly teach
the model that these specific objects are not animals.

**There is a problem to solve first: the system does not currently
store anything suitable.** Verified on the running deployment:

- `thumbnail_path` rows point at a *crop*, roughly twice the detection
  box with a 256 px floor, not the frame and not the detector's input
  window. Useful for auditing, not for training.
- The 640x640 window the detector actually saw is discarded. It is not
  copied onto the outgoing `DetectionEvent`, deliberately:
  `storage_queue` holds 256 entries at 1.2 MB per window, which is over
  a gigabyte on a 4 GB board.
- `clip_path` is NULL on all 4073 rows and `data/clips/` is empty.
  `ClipWriter` is constructed at `src/ratcatcher/pipeline/engine.py:167`
  and `clip_path` is read at line 725, but no write call exists in the
  video path. Video clip writing appears never to have been wired up.
  Flagging only -- do not fix as part of the training work.

So collecting a training set requires a deliberate capture mode that
writes the full frame or the 640x640 window, with the box coordinates,
to disk. That is a change to the Pi codebase and should be scoped
separately from the GPU work.

Two further deployment constraints on collection:

- The root filesystem is 89 percent full with 3.3 GB free. Audio clips
  already occupy 3.5 GB. A meaningful image collection needs external
  storage or a scheduled offload.
- `storage.retention_days` is 30 and `storage.max_disk_gb` is 10.
  Retention has already pruned most thumbnails from the analysis window.
  Raise both, or offload continuously, before a collection run.

### 4.4 Generic background imagery

Cheaper than deployment data and complementary to it: sky, cloud, foliage,
fences, garden furniture, empty feeders, from any source, with empty
label files. This generalises the "nothing here" concept beyond the
specific objects in one yard, which matters if the camera is ever moved
or a second unit is deployed.

Aim for background images at roughly 10 to 15 percent of the final set,
weighted toward deployment-specific negatives.

---

## 5. Validation that predicts field behaviour

This section is the acceptance criteria. Do not report a model as
improved on Open Images val numbers alone; that metric is what allowed
the current failure.

**Build a field validation set.** Hold out frames from the deployment,
labelled by hand, never seen in training. It must contain:

- empty scenes: clear sky, moving cloud, the plant pot, the shepherd's
  hook, the cypress, at various times of day. These carry no boxes.
- real animals at realistic size, ideally 40 to 150 px.
- night infrared frames, both empty and with animals.

**Report false positives per hour, not only mAP.** The operative
failure is 1232 false detections in about 15 hours of daylight. The
headline metric should be false positives per camera-hour on empty
scenes, which mAP on a set with no empty scenes cannot express at all.

**Minimum bar before deploying a replacement:**

1. False positives per camera-hour on the empty-scene portion of the
   field set reduced by at least an order of magnitude against the
   current model, measured on the same footage.
2. No regression in bird recall on the field set. Birds currently work
   and are the bulk of true detections.
3. Non-zero recall on real rats at realistic size and lighting. The
   current model's field recall is unmeasured but the field precision
   is 0 of 6.
4. Open Images val mAP reported alongside, for continuity, but
   explicitly not used as the gate.

**Measure the INT8 cost separately.** Quantization for the NPU costs
some accuracy and the size of that drop for this model has never been
measured. Compare the ONNX and the HEF on the same field set before
trusting NPU numbers. Re-quantizing happens on every HEF build, so this
comparison has to be redone per build rather than assumed to carry
across.

---

## 6. Training

Mechanics are in `training/README.md`. What follows is what should
change from the defaults it documents.

The current run used `training/train_detector.py` with:

```
epochs 100   batch 16   imgsz 640   patience 20   weights yolov8n.pt
optimizer and augmentation: Ultralytics defaults
```

Changes to consider, in order:

- **Background images.** The single most important change. Ultralytics
  accepts them as images with empty label files; no code change is
  needed in `train_detector.py`.
- **Class balance.** With rat brought to parity the set is roughly even.
  If it is not, weight or resample rather than letting a 6:1 imbalance
  stand.
- **Image size.** Keep `imgsz 640`. It must agree with the deployed
  input, and the ROI-crop path was designed around it.
- **Model size.** YOLOv8n is the current choice. The Hailo-8 has
  headroom -- the pipeline measures about 9.3 fps per camera and is
  bound by a CPU resize, not by the NPU -- so YOLOv8s is worth
  evaluating if accuracy is short. Confirm it compiles and fits before
  committing to it, and re-measure end to end rather than assuming the
  NPU absorbs it.
- **Augmentation.** Defaults are reasonable for detection. If night IR
  data is included, verify that HSV augmentation is not destroying the
  monochrome character of those frames; consider training a single
  model on both domains only after checking that it does not degrade
  either.

Starting from `yolov8n.pt` COCO weights remains correct.

---

## 7. Export and deploy

Full procedure in `training/README.md` and `docs/DEPLOYMENT.md`. The
three ways to get this wrong:

1. **Do not use bare `hailo parser` / `hailo optimize` / `hailo
   compile`.** They compile the Ultralytics DFL decode tail onto the
   NPU. The result builds without error, wastes NPU resources, and does
   not match what `HailoDetector` reads back. Use
   `scripts/build_hef.sh`, which cuts the graph at the six `model.22`
   head convolutions and reattaches decode and NMS as a HailoRT
   post-process.
2. **The HEF must be built on x86-64.** The Hailo Dataflow Compiler is
   an x86-64 Linux wheel for Python 3.8 to 3.11 with no aarch64 build.
   `scripts/build_hef.sh` refuses to run on aarch64 rather than failing
   halfway. This is the mirror image of the Debian package, which must
   be built on the Pi because it vendors architecture-specific wheels.
3. **Calibration must match inference preprocessing.** Calibration
   images are preprocessed exactly as `HailoDetector.detect()` does: a
   plain stretch to 640x640, not a letterbox. If section 3's letterbox
   question is ever revisited, this changes with it.

Build for `hailo8`. Deploy the ONNX as well: it is the CPU fallback and
the reference for the quantization comparison in section 5.

---

## 8. Related deployment issues, for context

These are not model problems and should not be fixed as part of the
training work, but they shape what the model is asked to do and are
listed so the picture is complete.

- **No region-of-interest mask.** `roi: []` for both cameras at
  `/etc/ratcatcher/default.yaml:42` and `:53`. Drifting cloud in open
  sky therefore drives motion, window planning and inference all day.
  A polygon excluding sky would remove most cloud false positives with
  no retraining, at the cost of birds in flight -- which are the
  majority of the good detections, so this is a genuine trade rather
  than a free win.
- **Motion cooldown too short for drift.** `motion.cooldown_seconds:
  2.0`. One cloud produced 26 detections in a single minute at 10:30.
  Tuned for animals, not for weather.
- **No day/night handling anywhere in the pipeline.** `Lux` and
  `ExposureTime` are read only by the focus bring-up tools
  (`camera/focus.py`, `pipeline/focus_engine.py`, `web/focus_server.py`).
  `PicameraSource` sets only `FrameRate`
  (`src/ratcatcher/camera/capture.py:392`) and lets auto-exposure run
  uncapped. Under an illuminator that means long exposures and motion
  blur on a moving rat, which is the one artefact that destroys a small
  fast target. An `ExposureTime` or `FrameDurationLimits` cap will
  probably be needed; the value cannot be chosen without measuring under
  the actual illuminator.
- **Illuminator power draw.** The UPS pack is about 78 Wh against a
  measured 10 to 14 W load, giving 5 to 7 hours. A typical 850 nm array
  draws 3 to 10 W, taking runtime to roughly 4.3 hours at plus 6 W, and
  it draws during exactly the hours the solar panel contributes nothing.
- **Illuminator mounting.** The enclosure has a clear door with the
  cameras behind it. An illuminator mounted inside will reflect off the
  polycarbonate straight back into the lens and wash the frame to white.
  It must be mounted outside, or optically isolated from the lens
  compartment and sealed against the window.
- **The species classifier reads the downscaled frame.** Top-1 agreement
  with the native-resolution answer falls from 86 percent at 160 px to
  9 percent at 66 px while mean confidence stays high. Separate from
  detection, but it means a distant bird can produce a high-confidence
  wrong species. `DetectionEvent.detail` already carries the correct
  native crop.

---

## 9. Known documentation discrepancies

Found while preparing this brief. Listed, not corrected.

- `training/README.md` states the current HEF is compiled for 8L.
  `models/ratcatcher_best.hef` is now a symlink to
  `ratcatcher_best_hailo8.hef`, and no 8L warning appears in the logs.
  `CLAUDE.md` carries the same stale note.
- `training/README.md` describes expected results as "Bird mAP@0.5:
  0.70-0.85". The shipped model measures bird at 0.608, the weakest of
  the five classes, which is not reflected there.

---

## 10. Task list

Ordered. Items 1 and 2 are independent of the illuminator and can start
immediately.

1. Build the deployment hard-negative set. Requires a capture mode on
   the Pi that stores full frames or 640x640 windows (section 4.3).
   Scope separately from the GPU work.
2. Re-source the rat class from real wild rats and discard the Open
   Images `Mouse` imagery (section 4.2). Decide whether rat and squirrel
   remain separate classes.
3. Build the field validation set, including empty scenes (section 5).
   Without this there is no way to tell whether anything improved.
4. After the illuminator is mounted: collect night infrared footage,
   several full nights, empty and populated (section 4.1).
5. Retrain once, on day hard-negatives and night data together, rather
   than retraining twice.
6. Validate against section 5's bar. Compare ONNX against HEF on the
   same footage to measure the INT8 cost.
7. Export, compile for `hailo8`, deploy, and re-measure false positives
   per camera-hour on the real deployment before declaring it fixed.

---

## Appendix: reproducing the evidence

Run on the Pi. The database is world-readable at
`/opt/ratcatcher/data/detections.db`.

Class counts and confidence spread:

```sql
SELECT class_name, COUNT(*) n,
       ROUND(MIN(confidence),3) min_c,
       ROUND(AVG(confidence),3) avg_c,
       ROUND(MAX(confidence),3) max_c
FROM detections GROUP BY class_name ORDER BY n DESC;
```

Video against audio by hour, which is what demonstrates night blindness:

```sql
SELECT substr(timestamp,12,2) hr, COUNT(*) FROM detections GROUP BY hr;
SELECT substr(timestamp,12,2) hr, COUNT(*) FROM audio_detections GROUP BY hr;
```

Background-image count in the dataset, run on the training machine:

```python
import glob
for split in ("train", "val"):
    files = glob.glob(f"datasets/ratcatcher/{split}/labels/*.txt")
    empty = sum(1 for f in files if not open(f).read().strip())
    print(f"{split}: {len(files)} label files, {empty} empty")
```

The contact sheets that produced the visual audit were built by reading
`thumbnail_path` from the database, sampling across the confidence
range, and tiling with `cv2`. Rebuild them against a fresh collection
rather than the current one, which retention has largely pruned.
