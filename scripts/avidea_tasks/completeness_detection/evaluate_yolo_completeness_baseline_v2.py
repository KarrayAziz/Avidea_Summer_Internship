#!/usr/bin/env python3
"""
Evaluate a no-training YOLO boundary heuristic for vehicle-view completeness.

Expected labels can be arranged in either common structure:

    dataset/
      front/
        complete/
        incomplete/
      back/
        complete/
        incomplete/
      left/
        complete/
        incomplete/
      right/
        complete/
        incomplete/

or:

    dataset/
      complete/
        front/
        back/
        left/
        right/
      incomplete/
        front/
        back/
        left/
        right/

The script searches recursively and infers:
- view: front, back/rear, left, right
- label: complete or incomplete

It runs YOLO once per image, selects the largest detected vehicle, records its
normalized distance from each image boundary, applies a view-specific boundary
threshold, and writes:
- predictions.csv
- metrics.json
- threshold_sweep.csv (optional)
- annotated error images (optional)

Heuristic:
    incomplete if the largest vehicle box is too close to a relevant frame edge.
    complete otherwise.

Detection failures and detections below --min-area-ratio are conservatively
classified as incomplete.

For a fast test, use --samples-per-view N and --device auto (or --device 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VIEW_ALIASES = {
    "front": "front",
    "back": "back",
    "rear": "back",
    "left": "left",
    "right": "right",
}

LABEL_ALIASES = {
    "complete": "complete",
    "completed": "complete",
    "full": "complete",
    "uncropped": "complete",
    "incomplete": "incomplete",
    "cropped": "incomplete",
    "partial": "incomplete",
}


@dataclass
class Sample:
    path: str
    view: str
    true_label: str


@dataclass
class DetectionRecord:
    path: str
    view: str
    true_label: str
    predicted_label: str
    correct: bool
    detected: bool
    yolo_class_id: Optional[int]
    yolo_confidence: Optional[float]
    box_area_ratio: Optional[float]
    left_margin: Optional[float]
    right_margin: Optional[float]
    top_margin: Optional[float]
    bottom_margin: Optional[float]
    min_relevant_margin: Optional[float]
    threshold: float
    reason: str
    inference_seconds: float


def normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def infer_metadata(image_path: Path, dataset_root: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Infer view and completeness label from path components relative to root.
    """
    relative_parts = image_path.relative_to(dataset_root).parts[:-1]
    tokens = [normalize_token(part) for part in relative_parts]

    found_views = {
        VIEW_ALIASES[token]
        for token in tokens
        if token in VIEW_ALIASES
    }
    found_labels = {
        LABEL_ALIASES[token]
        for token in tokens
        if token in LABEL_ALIASES
    }

    view = next(iter(found_views)) if len(found_views) == 1 else None
    label = next(iter(found_labels)) if len(found_labels) == 1 else None
    return view, label


def discover_samples(dataset_root: Path) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    skipped: list[str] = []

    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        view, label = infer_metadata(path, dataset_root)
        if view is None or label is None:
            skipped.append(str(path))
            continue

        samples.append(Sample(path=str(path), view=view, true_label=label))

    return samples, skipped



def sample_per_view(
    samples: list[Sample],
    samples_per_view: Optional[int],
    seed: int,
) -> list[Sample]:
    """
    Limit evaluation to at most N images per view.

    Sampling is stratified by completeness label, so each view contains as even
    a complete/incomplete split as the available data allows. This makes quick
    tests more informative than sampling mostly from the majority class.
    """
    if samples_per_view is None:
        return samples

    if samples_per_view <= 0:
        raise ValueError("--samples-per-view must be greater than zero.")

    rng = random.Random(seed)
    selected: list[Sample] = []

    for view in ("front", "back", "left", "right"):
        view_samples = [sample for sample in samples if sample.view == view]
        complete = [sample for sample in view_samples if sample.true_label == "complete"]
        incomplete = [sample for sample in view_samples if sample.true_label == "incomplete"]

        rng.shuffle(complete)
        rng.shuffle(incomplete)

        # Aim for an even split. If one class lacks enough examples, fill the
        # remaining quota from the other class.
        target_incomplete = min(len(incomplete), samples_per_view // 2)
        target_complete = min(len(complete), samples_per_view - target_incomplete)

        remaining = samples_per_view - target_complete - target_incomplete
        if remaining > 0:
            extra_complete = min(len(complete) - target_complete, remaining)
            target_complete += extra_complete
            remaining -= extra_complete

        if remaining > 0:
            extra_incomplete = min(len(incomplete) - target_incomplete, remaining)
            target_incomplete += extra_incomplete

        selected.extend(complete[:target_complete])
        selected.extend(incomplete[:target_incomplete])

    rng.shuffle(selected)
    return selected


def resolve_device(requested_device: str) -> str:
    """
    Resolve 'auto' to GPU 0 when CUDA is available, otherwise CPU.
    """
    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "0"

    return "cpu"


def relevant_margins(
    view: str,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> dict[str, float]:
    """
    All four edges matter because the prompt requires the vehicle silhouette
    to remain fully inside the frame.

    Keeping this function view-aware makes it easy to change the rule later.
    """
    if view in {"front", "back", "left", "right"}:
        return {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
        }

    raise ValueError(f"Unsupported view: {view}")


def threshold_for_view(args: argparse.Namespace, view: str) -> float:
    return {
        "front": args.front_threshold,
        "back": args.back_threshold,
        "left": args.left_threshold,
        "right": args.right_threshold,
    }[view]


def predict_from_margins(
    *,
    detected: bool,
    area_ratio: Optional[float],
    min_relevant_margin: Optional[float],
    threshold: float,
    min_area_ratio: float,
) -> tuple[str, str]:
    if not detected:
        return "incomplete", "NO_VEHICLE_DETECTED"

    if area_ratio is None or area_ratio < min_area_ratio:
        return "incomplete", "DETECTION_TOO_SMALL"

    if min_relevant_margin is None:
        return "incomplete", "MARGIN_UNAVAILABLE"

    if min_relevant_margin <= threshold:
        return "incomplete", "BOX_NEAR_FRAME_EDGE"

    return "complete", "BOX_CLEAR_OF_FRAME_EDGES"


def select_largest_vehicle_box(result) -> Optional[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return None

    xyxy = result.boxes.xyxy.cpu()
    confs = result.boxes.conf.cpu()
    classes = result.boxes.cls.cpu()

    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    areas = widths * heights

    index = int(areas.argmax().item())
    return {
        "xyxy": [float(v) for v in xyxy[index].tolist()],
        "confidence": float(confs[index].item()),
        "class_id": int(classes[index].item()),
        "area": float(areas[index].item()),
    }


def annotate_image(
    image_path: Path,
    output_path: Path,
    box: Optional[list[float]],
    record: DetectionRecord,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return

    if box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    lines = [
        f"view={record.view}",
        f"true={record.true_label}",
        f"pred={record.predicted_label}",
        f"reason={record.reason}",
        f"threshold={record.threshold:.4f}",
    ]

    if record.min_relevant_margin is not None:
        lines.append(f"min_margin={record.min_relevant_margin:.4f}")

    if record.box_area_ratio is not None:
        lines.append(f"area_ratio={record.box_area_ratio:.4f}")

    y = 28
    for line in lines:
        cv2.putText(
            image,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        y += 27

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(records: Iterable[DetectionRecord]) -> dict:
    records = list(records)

    tp_complete = sum(
        r.true_label == "complete" and r.predicted_label == "complete"
        for r in records
    )
    fn_complete = sum(
        r.true_label == "complete" and r.predicted_label == "incomplete"
        for r in records
    )
    fp_complete = sum(
        r.true_label == "incomplete" and r.predicted_label == "complete"
        for r in records
    )
    tn_complete = sum(
        r.true_label == "incomplete" and r.predicted_label == "incomplete"
        for r in records
    )

    total = len(records)
    accuracy = safe_div(tp_complete + tn_complete, total)

    complete_precision = safe_div(tp_complete, tp_complete + fp_complete)
    complete_recall = safe_div(tp_complete, tp_complete + fn_complete)
    complete_f1 = safe_div(
        2 * complete_precision * complete_recall,
        complete_precision + complete_recall,
    )

    incomplete_precision = safe_div(tn_complete, tn_complete + fn_complete)
    incomplete_recall = safe_div(tn_complete, tn_complete + fp_complete)
    incomplete_f1 = safe_div(
        2 * incomplete_precision * incomplete_recall,
        incomplete_precision + incomplete_recall,
    )

    balanced_accuracy = (complete_recall + incomplete_recall) / 2

    return {
        "count": total,
        "correct": tp_complete + tn_complete,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "complete": {
            "precision": complete_precision,
            "recall": complete_recall,
            "f1": complete_f1,
            "support": tp_complete + fn_complete,
        },
        "incomplete": {
            "precision": incomplete_precision,
            "recall": incomplete_recall,
            "f1": incomplete_f1,
            "support": tn_complete + fp_complete,
        },
        "confusion_matrix": {
            "true_complete_pred_complete": tp_complete,
            "true_complete_pred_incomplete": fn_complete,
            "true_incomplete_pred_complete": fp_complete,
            "true_incomplete_pred_incomplete": tn_complete,
        },
        "detection_failures": sum(not r.detected for r in records),
    }


def grouped_metrics(records: list[DetectionRecord]) -> dict:
    result = {"overall": calculate_metrics(records), "by_view": {}}
    for view in ("front", "back", "left", "right"):
        view_records = [record for record in records if record.view == view]
        result["by_view"][view] = calculate_metrics(view_records)
    return result


def write_predictions_csv(records: list[DetectionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(records[0]).keys()) if records else [
        field.name for field in DetectionRecord.__dataclass_fields__.values()
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def parse_sweep_thresholds(value: str) -> list[float]:
    thresholds = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        threshold = float(token)
        if threshold < 0 or threshold > 0.25:
            raise argparse.ArgumentTypeError(
                "Sweep thresholds must be between 0 and 0.25."
            )
        thresholds.append(threshold)

    if not thresholds:
        raise argparse.ArgumentTypeError("At least one sweep threshold is required.")

    return sorted(set(thresholds))


def build_sweep_rows(
    records: list[DetectionRecord],
    thresholds: list[float],
    min_area_ratio: float,
) -> list[dict]:
    rows: list[dict] = []

    for threshold in thresholds:
        modified: list[DetectionRecord] = []

        for record in records:
            predicted_label, reason = predict_from_margins(
                detected=record.detected,
                area_ratio=record.box_area_ratio,
                min_relevant_margin=record.min_relevant_margin,
                threshold=threshold,
                min_area_ratio=min_area_ratio,
            )

            modified.append(
                DetectionRecord(
                    **{
                        **asdict(record),
                        "predicted_label": predicted_label,
                        "correct": predicted_label == record.true_label,
                        "threshold": threshold,
                        "reason": reason,
                    }
                )
            )

        for scope, subset in [
            ("overall", modified),
            *[
                (view, [record for record in modified if record.view == view])
                for view in ("front", "back", "left", "right")
            ],
        ]:
            metrics = calculate_metrics(subset)
            rows.append(
                {
                    "threshold": threshold,
                    "scope": scope,
                    "count": metrics["count"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "complete_precision": metrics["complete"]["precision"],
                    "complete_recall": metrics["complete"]["recall"],
                    "complete_f1": metrics["complete"]["f1"],
                    "incomplete_precision": metrics["incomplete"]["precision"],
                    "incomplete_recall": metrics["incomplete"]["recall"],
                    "incomplete_f1": metrics["incomplete"]["f1"],
                }
            )

    return rows


def write_dict_rows_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_metric_summary(metrics: dict) -> None:
    print("\n" + "=" * 78)
    print("YOLO BOUNDARY HEURISTIC EVALUATION")
    print("=" * 78)

    scopes = [("overall", metrics["overall"])]
    scopes.extend((view, metrics["by_view"][view]) for view in ("front", "back", "left", "right"))

    for name, values in scopes:
        print(
            f"{name.upper():>8} | "
            f"N={values['count']:4d} | "
            f"Acc={values['accuracy']:.3f} | "
            f"BalAcc={values['balanced_accuracy']:.3f} | "
            f"Incomplete Recall={values['incomplete']['recall']:.3f} | "
            f"Incomplete F1={values['incomplete']['f1']:.3f}"
        )

    print("=" * 78)


def evaluate(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    all_samples, skipped = discover_samples(dataset_root)
    if not all_samples:
        raise RuntimeError(
            "No labeled images were discovered. Ensure folder names contain one "
            "view token (front/back/rear/left/right) and one label token "
            "(complete/incomplete)."
        )

    samples = sample_per_view(
        all_samples,
        samples_per_view=args.samples_per_view,
        seed=args.seed,
    )
    device = resolve_device(args.device)

    print(f"Dataset: {dataset_root}")
    print(f"Discovered labeled images: {len(all_samples)}")
    if args.samples_per_view is not None:
        print(
            f"Quick-test sampling: up to {args.samples_per_view} images per view "
            f"({len(samples)} selected total, stratified by label)"
        )
    print(f"Skipped images with ambiguous/missing path labels: {len(skipped)}")
    print(f"Loading YOLO model: {args.yolo}")
    print(f"Inference device: {device}")

    if device != "cpu" and not torch.cuda.is_available():
        print(
            "WARNING: A GPU device was requested, but PyTorch does not report "
            "CUDA availability. Ultralytics may fail unless CUDA is installed."
        )

    model = YOLO(args.yolo)
    records: list[DetectionRecord] = []

    class_ids = args.classes
    total_start = time.perf_counter()

    for sample in tqdm(samples, desc="Evaluating images", unit="image"):
        image_path = Path(sample.path)
        image = cv2.imread(str(image_path))

        if image is None:
            threshold = threshold_for_view(args, sample.view)
            predicted_label = "incomplete"
            record = DetectionRecord(
                path=str(image_path),
                view=sample.view,
                true_label=sample.true_label,
                predicted_label=predicted_label,
                correct=predicted_label == sample.true_label,
                detected=False,
                yolo_class_id=None,
                yolo_confidence=None,
                box_area_ratio=None,
                left_margin=None,
                right_margin=None,
                top_margin=None,
                bottom_margin=None,
                min_relevant_margin=None,
                threshold=threshold,
                reason="IMAGE_READ_FAILED",
                inference_seconds=0.0,
            )
            records.append(record)
            continue

        image_height, image_width = image.shape[:2]
        image_area = float(image_width * image_height)

        start = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            classes=class_ids,
            verbose=False,
            device=device,
        )
        inference_seconds = time.perf_counter() - start

        selected = select_largest_vehicle_box(results[0]) if results else None
        threshold = threshold_for_view(args, sample.view)

        box: Optional[list[float]] = None
        detected = selected is not None

        if selected is None:
            area_ratio = None
            left_margin = right_margin = top_margin = bottom_margin = None
            min_margin = None
            class_id = None
            confidence = None
        else:
            box = selected["xyxy"]
            x1, y1, x2, y2 = box

            # Clamp small numerical overshoots produced by detection.
            x1 = max(0.0, min(float(image_width), x1))
            x2 = max(0.0, min(float(image_width), x2))
            y1 = max(0.0, min(float(image_height), y1))
            y2 = max(0.0, min(float(image_height), y2))

            area_ratio = selected["area"] / image_area
            left_margin = x1 / image_width
            right_margin = (image_width - x2) / image_width
            top_margin = y1 / image_height
            bottom_margin = (image_height - y2) / image_height

            margins = relevant_margins(
                sample.view,
                left_margin,
                right_margin,
                top_margin,
                bottom_margin,
            )
            min_margin = min(margins.values())
            class_id = selected["class_id"]
            confidence = selected["confidence"]

        predicted_label, reason = predict_from_margins(
            detected=detected,
            area_ratio=area_ratio,
            min_relevant_margin=min_margin,
            threshold=threshold,
            min_area_ratio=args.min_area_ratio,
        )

        record = DetectionRecord(
            path=str(image_path),
            view=sample.view,
            true_label=sample.true_label,
            predicted_label=predicted_label,
            correct=predicted_label == sample.true_label,
            detected=detected,
            yolo_class_id=class_id,
            yolo_confidence=confidence,
            box_area_ratio=area_ratio,
            left_margin=left_margin,
            right_margin=right_margin,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
            min_relevant_margin=min_margin,
            threshold=threshold,
            reason=reason,
            inference_seconds=inference_seconds,
        )
        records.append(record)

        should_save = (
            args.save_annotations == "all"
            or (args.save_annotations == "errors" and not record.correct)
        )
        if should_save:
            relative = image_path.relative_to(dataset_root)
            status_folder = "correct" if record.correct else "errors"
            destination = output_root / "annotations" / status_folder / relative
            annotate_image(image_path, destination, box, record)

    elapsed = time.perf_counter() - total_start

    predictions_path = output_root / "predictions.csv"
    metrics_path = output_root / "metrics.json"
    skipped_path = output_root / "skipped_images.txt"

    write_predictions_csv(records, predictions_path)

    metrics = grouped_metrics(records)
    metrics["configuration"] = {
        "dataset": str(dataset_root),
        "yolo": args.yolo,
        "classes": class_ids,
        "confidence_threshold": args.conf,
        "minimum_area_ratio": args.min_area_ratio,
        "front_threshold": args.front_threshold,
        "back_threshold": args.back_threshold,
        "left_threshold": args.left_threshold,
        "right_threshold": args.right_threshold,
        "device": device,
        "save_annotations": args.save_annotations,
        "samples_per_view": args.samples_per_view,
        "sampling_seed": args.seed,
    }
    metrics["timing"] = {
        "total_seconds": elapsed,
        "average_seconds_per_image": elapsed / len(records) if records else 0.0,
        "yolo_inference_seconds_sum": sum(r.inference_seconds for r in records),
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with skipped_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(skipped))

    if args.sweep_thresholds:
        sweep_rows = build_sweep_rows(
            records,
            args.sweep_thresholds,
            args.min_area_ratio,
        )
        write_dict_rows_csv(sweep_rows, output_root / "threshold_sweep.csv")

    print_metric_summary(metrics)
    print(f"Predictions: {predictions_path}")
    print(f"Metrics:     {metrics_path}")
    if args.sweep_thresholds:
        print(f"Sweep:       {output_root / 'threshold_sweep.csv'}")
    if skipped:
        print(f"Skipped:     {skipped_path}")
    print(f"Output root: {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO frame-boundary heuristic for vehicle completeness."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Root of the labeled completeness dataset.",
    )
    parser.add_argument(
        "--yolo",
        default="/home/aziz/Aziz/DigiCover/usingGeminiApi/models/yolov8m.pt",
        help="YOLO weights path or Ultralytics model name.",
    )
    parser.add_argument(
        "--output",
        default="./completeness_baseline_output",
        help="Directory for CSV, JSON, and optional annotations.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[2, 5, 7],
        help="YOLO class IDs accepted as vehicles. COCO: car=2, bus=5, truck=7.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.05,
        help=(
            "Minimum largest-box area divided by image area. Smaller detections "
            "are classified as incomplete."
        ),
    )
    parser.add_argument(
        "--front-threshold",
        type=float,
        default=0.01,
        help="Normalized boundary threshold for front images.",
    )
    parser.add_argument(
        "--back-threshold",
        type=float,
        default=0.01,
        help="Normalized boundary threshold for back/rear images.",
    )
    parser.add_argument(
        "--left-threshold",
        type=float,
        default=0.01,
        help="Normalized boundary threshold for left-side images.",
    )
    parser.add_argument(
        "--right-threshold",
        type=float,
        default=0.01,
        help="Normalized boundary threshold for right-side images.",
    )
    parser.add_argument(
        "--samples-per-view",
        type=int,
        default=None,
        help=(
            "Quick-test limit per view. For example, 50 evaluates up to 50 "
            "front, 50 back, 50 left, and 50 right images. Sampling is "
            "stratified between complete and incomplete labels."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by quick-test sampling.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            'Ultralytics device: "auto" uses GPU 0 when CUDA is available, '
            'otherwise CPU. You may also pass "cpu", "0", or "0,1".'
        ),
    )
    parser.add_argument(
        "--save-annotations",
        choices=("none", "errors", "all"),
        default="errors",
        help="Save annotated images for no samples, mistakes only, or all samples.",
    )
    parser.add_argument(
        "--sweep-thresholds",
        type=parse_sweep_thresholds,
        default=parse_sweep_thresholds(
            "0,0.0025,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05"
        ),
        help=(
            "Comma-separated thresholds evaluated from cached YOLO detections. "
            'Use "" to disable.'
        ),
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    evaluate(arguments)
