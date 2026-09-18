"""
compare_hef_accuracy.py
=======================
Measures what INT8 quantization cost the detector, without a Hailo device.

`docs/MODEL_TRAINING.md` section 5 and `tasks/todo.md` both carry the
same open item: the HEF has never been compared with the ONNX it was
compiled from. The two run on different machines -- the ONNX anywhere,
the HEF only on the NPU -- so the comparison never happened. The Dataflow
Compiler can run the *quantized* graph on the CPU, which is the same
arithmetic the HEF performs, only slower. `build_hef.py --save-har` keeps
that graph, and this script runs it and the ONNX over the same frames
with the same decode, the same thresholds and the same scoring code.

**Why the same code scores both.** Ultralytics can validate an ONNX, but
it cannot validate a HAR, and two evaluators disagree on details -- how
a box on the image edge is clipped, whether a tie in confidence is a hit
or a miss. Any such difference would be reported as a quantization cost.
So both models go through one decode (the ONNX's DFL tail is already
folded into its output; the HAR's on-chip NMS is emulated) and one
AP calculation.

**Why the score floor is 0.2.** The HEF's NMS is compiled with a score
threshold of 0.2 (`build_hef.py --score-threshold`), and nothing below
it survives to the host. Scoring the ONNX at 0.001, as mAP conventionally
does, would credit it with recall the deployed HEF cannot have, so both
are cut at the same floor.

Usage (from the DFC venv, which has onnxruntime and PIL):
    dfc-venv/bin/python training/compare_hef_accuracy.py \\
        --onnx models/v3/ratcatcher_best.onnx \\
        --har models/v3/ratcatcher_best_quantized.har \\
        --data datasets/eval_ct_night --limit 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["bird", "squirrel", "rat", "cat", "unknown_animal"]
RAT = 2


# --- Data -----------------------------------------------------------------

def load_split(data: Path, limit: int, seed: int) -> list:
    """Return (image_path, boxes) pairs; boxes are (cls, x0, y0, x1, y1) in 0-1.

    Frames that hold a rat come first so a small ``--limit`` still
    measures the class this detector exists for; the remainder is a
    random sample of the rest, seeded so the run repeats.
    """
    images = sorted((data / "val" / "images").glob("*.jpg"))
    rows = []
    for image in images:
        label = data / "val" / "labels" / f"{image.stem}.txt"
        boxes = []
        try:
            text = label.read_text()
        except OSError:
            text = ""
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            boxes.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        rows.append((image, boxes))
    if limit and limit < len(rows):
        rng = np.random.default_rng(seed)
        with_rat = [r for r in rows if any(b[0] == RAT for b in r[1])]
        rest = [r for r in rows if not any(b[0] == RAT for b in r[1])]
        rng.shuffle(rest)
        rows = (with_rat + rest)[:limit]
    return rows


def load_frame(path: Path, imgsz: int) -> np.ndarray:
    """RGB uint8 stretched to imgsz x imgsz, as HailoDetector feeds the NPU."""
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    return np.asarray(rgb, dtype=np.uint8)


# --- Shared box utilities -------------------------------------------------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` (n,4) and ``b`` (m,4), x0 y0 x1 y1."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list:
    """Greedy NMS, returns kept indices. Same rule the HEF applies per class."""
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return keep


# --- ONNX -----------------------------------------------------------------

class OnnxModel:
    """Runs the exported Ultralytics ONNX and decodes it like OpenCVDetector."""

    def __init__(self, path: Path, score_thr: float, iou_thr: float,
                 max_per_class: int):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path),
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.score_thr, self.iou_thr, self.max_per_class = score_thr, iou_thr, max_per_class

    def detect(self, rgb: np.ndarray) -> list:
        imgsz = rgb.shape[0]
        x = rgb.astype(np.float32)[None].transpose(0, 3, 1, 2) / 255.0
        out = self.session.run(None, {self.input_name: x})[0][0]  # (4+C, 8400)
        boxes_cxcywh, scores = out[:4].T, out[4:].T  # (8400,4), (8400,C)
        cx, cy, w, h = boxes_cxcywh.T
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / imgsz
        results = []
        for c in range(scores.shape[1]):
            s = scores[:, c]
            m = s >= self.score_thr
            if not m.any():
                continue
            b, s = xyxy[m], s[m]
            for i in nms(b, s, self.iou_thr)[:self.max_per_class]:
                results.append((c, float(s[i]), *np.clip(b[i], 0, 1).tolist()))
        return results


# --- Quantized HAR through the DFC emulator -------------------------------

class HarModel:
    """Runs the quantized graph, on-chip NMS included, on the CPU."""

    def __init__(self, path: Path):
        try:
            from hailo_sdk_client import ClientRunner, InferenceContext
        except ImportError:
            print("[ERROR] hailo_sdk_client is not importable. Run this from "
                  "the DFC venv: dfc-venv/bin/python ...")
            sys.exit(1)
        self.runner = ClientRunner(har=str(path))
        self.context = InferenceContext.SDK_QUANTIZED
        self._ctx = None

    def __enter__(self):
        self._cm = self.runner.infer_context(self.context)
        self._ctx = self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def detect_batch(self, frames: list) -> list:
        """Detections for each frame in ``frames``, one list per frame.

        Batched because the emulator's cost is per call, not per frame:
        measured on this machine, one frame takes 8.9 s and sixteen take
        7.6 s. The normalization layer is compiled into the graph, so the
        emulator takes the same 0-255 input the NPU does.
        """
        batch = np.stack(frames).astype(np.float32)
        out = self.runner.infer(self._ctx, batch)
        return [parse_nms(out, i) for i in range(len(frames))]


def parse_nms(out, index: int = 0) -> list:
    """Flatten the emulator's NMS output into (cls, score, x0, y0, x1, y1).

    HailoRT hands HailoDetector a ``[batch][class]`` list of ``(n, 5)``
    arrays holding ``y_min, x_min, y_max, x_max, score``. The emulator
    returns the same five fields in one padded array, but with the field
    axis *before* the proposal axis: ``(batch, classes, 5, max_per_class)``,
    measured on DFC 3.34. Reading that as ``(..., max_per_class, 5)``
    puts a coordinate where the score belongs and every detection reads
    as padding -- which is what the first run of this script reported as
    a quantization that found nothing. A zero score is padding.
    """
    per_class = out[index] if isinstance(out, (list, tuple)) else np.asarray(out)[index]
    results = []
    for c, arr in enumerate(per_class):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 5 and arr.shape[1] != 5:
            arr = arr.T  # (5, max_per_class) -> (max_per_class, 5)
        for y0, x0, y1, x1, score in arr.reshape(-1, 5):
            if score <= 0:
                continue
            results.append((c, float(score),
                            float(np.clip(x0, 0, 1)), float(np.clip(y0, 0, 1)),
                            float(np.clip(x1, 0, 1)), float(np.clip(y1, 0, 1))))
    return results


# --- Scoring --------------------------------------------------------------

def average_precision(preds: list, truths: dict, cls: int, iou_thr: float) -> tuple:
    """AP at one IoU for one class over the whole set, VOC all-point.

    ``preds`` is a list of (image_id, score, x0, y0, x1, y1) for ``cls``;
    ``truths`` maps image_id to an (n,4) array of that class's boxes.
    Returns (ap, n_truth).
    """
    n_truth = sum(len(v) for v in truths.values())
    if n_truth == 0:
        return float("nan"), 0
    preds = sorted(preds, key=lambda p: -p[1])
    matched = {k: np.zeros(len(v), dtype=bool) for k, v in truths.items()}
    tp = np.zeros(len(preds))
    for i, (img, _score, *box) in enumerate(preds):
        gt = truths.get(img)
        if gt is None or len(gt) == 0:
            continue
        ious = iou_matrix(np.array([box]), gt)[0]
        j = int(ious.argmax())
        if ious[j] >= iou_thr and not matched[img][j]:
            matched[img][j] = True
            tp[i] = 1
    fp = 1 - tp
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / n_truth
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    # All-point interpolation: make precision monotone from the right.
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), n_truth


def precision_recall_at(preds: list, truths: dict, iou_thr: float, conf: float) -> tuple:
    """TP, FP, FN for one class at the deployment confidence."""
    tp = fp = 0
    matched = {k: np.zeros(len(v), dtype=bool) for k, v in truths.items()}
    for img, score, *box in sorted(preds, key=lambda p: -p[1]):
        if score < conf:
            continue
        gt = truths.get(img)
        if gt is None or len(gt) == 0:
            fp += 1
            continue
        ious = iou_matrix(np.array([box]), gt)[0]
        j = int(ious.argmax())
        if ious[j] >= iou_thr and not matched[img][j]:
            matched[img][j] = True
            tp += 1
        else:
            fp += 1
    fn = sum(int((~m).sum()) for m in matched.values())
    return tp, fp, fn


def score_model(name: str, detections: dict, rows: list, conf: float) -> dict:
    """Per-class AP50 plus rat precision/recall and empty-frame FP at ``conf``."""
    report = {"model": name, "ap50": {}, "rat": {}, "empty_fp": 0}
    empty = {img.name for img, boxes in rows if not boxes}
    for c, cname in enumerate(CLASS_NAMES):
        truths = {img.name: np.array([b[1:] for b in boxes if b[0] == c]).reshape(-1, 4)
                  for img, boxes in rows}
        preds = [(img, s, *box) for img, dets in detections.items()
                 for (cls, s, *box) in dets if cls == c]
        ap, n = average_precision(preds, truths, c, 0.5)
        report["ap50"][cname] = {"ap": ap, "instances": n}
        if c == RAT:
            tp, fp, fn = precision_recall_at(preds, truths, 0.5, conf)
            report["rat"] = {"tp": tp, "fp": fp, "fn": fn,
                             "precision": tp / max(tp + fp, 1),
                             "recall": tp / max(tp + fn, 1)}
    report["empty_fp"] = sum(1 for img, dets in detections.items()
                             if img in empty and any(s >= conf for _c, s, *_ in dets))
    report["empty_images"] = len(empty)
    return report


def print_report(reports: list, rows: list, conf: float, seconds: dict) -> None:
    print("")
    print("=" * 66)
    print("  ONNX (float) against quantized HAR (INT8, DFC emulator)")
    print("=" * 66)
    print(f"  frames {len(rows)}, deployment confidence {conf}, IoU 0.5")
    print("")
    print(f"  {'class':<16}{'instances':>10}" + "".join(f"{r['model']:>14}" for r in reports))
    for cname in CLASS_NAMES:
        n = reports[0]["ap50"][cname]["instances"]
        cells = "".join(f"{r['ap50'][cname]['ap']:>14.3f}" if n else f"{'--':>14}"
                        for r in reports)
        print(f"  {cname:<16}{n:>10}{cells}")
    valid = [c for c in CLASS_NAMES if reports[0]["ap50"][c]["instances"]]
    means = [np.mean([r["ap50"][c]["ap"] for c in valid]) for r in reports]
    print(f"  {'mAP@0.5':<16}{'':>10}" + "".join(f"{m:>14.3f}" for m in means))
    print("")
    print(f"  rat at conf {conf}:")
    for r in reports:
        rat = r["rat"]
        print(f"    {r['model']:<10} precision {rat['precision']:.3f}  "
              f"recall {rat['recall']:.3f}  (tp {rat['tp']} fp {rat['fp']} fn {rat['fn']})")
    print(f"  empty frames with a detection at conf {conf}:")
    for r in reports:
        print(f"    {r['model']:<10} {r['empty_fp']} of {r['empty_images']}")
    print("")
    for r in reports:
        print(f"  {r['model']:<10} {seconds[r['model']] / len(rows) * 1000:.0f} ms per frame")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an ONNX detector with its quantized HAR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx", type=Path, default=Path("models/ratcatcher_best.onnx"))
    parser.add_argument("--har", type=Path,
                        default=Path("models/ratcatcher_best_quantized.har"),
                        help="From build_hef.py --save-har.")
    parser.add_argument("--data", type=Path, default=Path("datasets/eval_ct_night"),
                        help="A set with val/images and val/labels.")
    parser.add_argument("--limit", type=int, default=200,
                        help="Frames to run; 0 for all. Rat frames first.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--score-threshold", type=float, default=0.2,
                        help="Must equal the HEF's compiled NMS score threshold.")
    parser.add_argument("--iou-threshold", type=float, default=0.7,
                        help="Must equal the HEF's compiled NMS IoU threshold.")
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.45,
                        help="Deployment confidence, detection.confidence_threshold.")
    parser.add_argument("--batch", type=int, default=16,
                        help="Frames per emulator call.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write the report here.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.onnx, args.har):
        if not path.is_file():
            print(f"[ERROR] Not found: {path}")
            return 1
    rows = load_split(args.data, args.limit, args.seed)
    if not rows:
        print(f"[ERROR] No frames under {args.data}/val/images")
        return 1
    print(f"[INFO] {len(rows)} frames from {args.data}")

    frames = {}
    for image, _ in rows:
        try:
            frames[image.name] = load_frame(image, args.imgsz)
        except (OSError, ValueError) as exc:
            print(f"[WARNING] Could not read {image.name}: {exc}")
    rows = [r for r in rows if r[0].name in frames]

    seconds = {}
    onnx = OnnxModel(args.onnx, args.score_threshold, args.iou_threshold,
                     args.max_per_class)
    t0 = time.perf_counter()
    onnx_dets = {name: onnx.detect(f) for name, f in frames.items()}
    seconds["onnx"] = time.perf_counter() - t0
    print(f"[INFO] ONNX done in {seconds['onnx']:.1f} s")

    t0 = time.perf_counter()
    names = list(frames)
    with HarModel(args.har) as har:
        har_dets = {}
        for start in range(0, len(names), args.batch):
            chunk = names[start:start + args.batch]
            for name, dets in zip(chunk, har.detect_batch([frames[n] for n in chunk])):
                har_dets[name] = dets
            done = start + len(chunk)
            print(f"[INFO] HAR {done}/{len(names)} "
                  f"({(time.perf_counter() - t0) / done:.2f} s per frame)")
    seconds["har"] = time.perf_counter() - t0

    reports = [score_model("onnx", onnx_dets, rows, args.conf),
               score_model("har", har_dets, rows, args.conf)]
    print_report(reports, rows, args.conf, seconds)
    if args.json:
        args.json.write_text(json.dumps(
            {"frames": len(rows), "data": str(args.data), "conf": args.conf,
             "reports": reports}, indent=2, default=float))
        print(f"[INFO] Wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
