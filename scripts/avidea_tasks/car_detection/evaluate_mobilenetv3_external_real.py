#!/usr/bin/env python3
"""
Evaluate the existing MobileNetV3-Small car-authenticity classifier on an
external REAL-car dataset organized by view:

Clean_Inference_Set/
├── back/
├── front/
├── left/
└── right/

Pipeline:
1. Read each external real image.
2. Run YOLOv8m on car/truck classes.
3. Select the largest valid car/truck box.
4. Require box area >= 8% of image area.
5. Expand the box by 25% on each side.
6. Run the frozen MobileNetV3-Small authenticity classifier.
7. Measure real recall and false rejection rate.
8. Export all real -> toy_scale mistakes for visual inspection.

No retraining or threshold tuning is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
from ultralytics import YOLO


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

VIEWS = ("back", "front", "left", "right")

# COCO class IDs
CAR_CLASS_ID = 2
TRUCK_CLASS_ID = 7
TARGET_CLASS_IDS = [CAR_CLASS_ID, TRUCK_CLASS_ID]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MobileNetV3-Small on external real-car images."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/"
            "data/Clean_Inference_Set"
        ),
        help="External real-car dataset root containing back/front/left/right.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/"
            "models/car_authenticity_mobilenetv3/"
            "best_mobilenet_v3_small.pth"
        ),
        help="Saved MobileNetV3-Small checkpoint.",
    )

    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8m.pt",
        help="YOLO model path/name.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/"
            "models/car_authenticity_mobilenetv3/"
            "external_real_eval"
        ),
        help="Evaluation output directory.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )

    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.08,
        help="Minimum selected bbox area relative to full image.",
    )

    parser.add_argument(
        "--context-expansion",
        type=float,
        default=0.25,
        help="Fraction of bbox width/height added on EACH side.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Classifier device: "auto", "cuda", "cpu", etc.',
    )

    parser.add_argument(
        "--yolo-device",
        type=str,
        default="cpu",
        help='YOLO device, e.g. "cpu", "0", "cuda:0".',
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision for classifier on CUDA.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous output first.",
    )

    return parser.parse_args()


def resolve_classifier_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    return device


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(dropout: float, num_classes: int = 2):
    model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def get_images(folder: Path):
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_largest_valid_vehicle(
    result,
    image_width: int,
    image_height: int,
    min_area_ratio: float,
):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    image_area = image_width * image_height
    candidates = []

    for box in result.boxes:
        class_id = int(box.cls.item())

        if class_id not in TARGET_CLASS_IDS:
            continue

        confidence = float(box.conf.item())

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        area = box_w * box_h
        area_ratio = area / image_area if image_area > 0 else 0.0

        if area_ratio < min_area_ratio:
            continue

        candidates.append(
            {
                "class_id": class_id,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "area_ratio": area_ratio,
            }
        )

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["area"])


def expand_box(
    bbox,
    image_width: int,
    image_height: int,
    expansion: float,
):
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    expand_x = box_w * expansion
    expand_y = box_h * expansion

    ex1 = max(0, int(round(x1 - expand_x)))
    ey1 = max(0, int(round(y1 - expand_y)))
    ex2 = min(image_width, int(round(x2 + expand_x)))
    ey2 = min(image_height, int(round(y2 + expand_y)))

    return ex1, ey1, ex2, ey2


def main():
    args = parse_args()

    dataset_root = args.dataset.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found:\n{dataset_root}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output already exists:\n{output_dir}\n\n"
                "Use --overwrite to recreate it."
            )

    false_rejects_dir = output_dir / "false_rejects"
    skipped_dir = output_dir / "skipped_no_valid_yolo"

    false_rejects_dir.mkdir(parents=True, exist_ok=True)
    skipped_dir.mkdir(parents=True, exist_ok=True)

    classifier_device = resolve_classifier_device(args.device)
    amp_enabled = args.amp and classifier_device.type == "cuda"

    print("=" * 90)
    print("EXTERNAL REAL-CAR AUTHENTICITY EVALUATION")
    print("=" * 90)
    print(f"Dataset          : {dataset_root}")
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"YOLO model       : {args.yolo_model}")
    print(f"Output           : {output_dir}")
    print(f"Classifier device: {classifier_device}")
    print(f"YOLO device      : {args.yolo_device}")
    print(f"AMP              : {amp_enabled}")
    print(f"YOLO confidence  : {args.conf}")
    print(f"Min area ratio   : {args.min_area_ratio}")
    print(
        f"Context expansion: "
        f"{args.context_expansion * 100:.0f}% per side"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=classifier_device,
        weights_only=False,
    )

    class_to_idx = checkpoint.get(
        "class_to_idx",
        {"real": 0, "toy_scale": 1},
    )

    if "real" not in class_to_idx or "toy_scale" not in class_to_idx:
        raise RuntimeError(
            f"Checkpoint class mapping invalid:\n{class_to_idx}"
        )

    idx_to_class = {
        idx: name
        for name, idx in class_to_idx.items()
    }

    real_idx = class_to_idx["real"]
    toy_idx = class_to_idx["toy_scale"]

    image_size = int(checkpoint.get("image_size", 224))
    dropout = float(checkpoint.get("dropout", 0.3))

    classifier = build_model(
        dropout=dropout,
        num_classes=len(class_to_idx),
    )

    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier = classifier.to(classifier_device)
    classifier.eval()

    transform = build_transform(image_size)

    yolo = YOLO(args.yolo_model)

    print(f"Image size       : {image_size}")
    print(f"Class mapping    : {class_to_idx}")
    print(f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}")
    print(
        f"Checkpoint val balanced accuracy: "
        f"{checkpoint.get('val_balanced_accuracy', 'unknown')}"
    )

    predictions = []
    skipped = []

    view_stats = defaultdict(
        lambda: {
            "images": 0,
            "evaluated": 0,
            "predicted_real": 0,
            "predicted_toy": 0,
            "skipped": 0,
        }
    )

    total_discovered = 0

    start_total = time.perf_counter()

    for view in VIEWS:
        view_dir = dataset_root / view

        if not view_dir.exists():
            print(f"\nWARNING: missing view folder: {view_dir}")
            continue

        images = get_images(view_dir)
        total_discovered += len(images)
        view_stats[view]["images"] = len(images)

        print("\n" + "=" * 90)
        print(f"VIEW: {view.upper()} ({len(images)} images)")
        print("=" * 90)

        for image_path in tqdm(images, desc=view):
            image_bgr = cv2.imread(str(image_path))

            if image_bgr is None:
                skipped.append(
                    {
                        "path": str(image_path),
                        "filename": image_path.name,
                        "view": view,
                        "reason": "IMAGE_READ_FAILED",
                    }
                )
                view_stats[view]["skipped"] += 1
                continue

            image_h, image_w = image_bgr.shape[:2]

            yolo_result = yolo.predict(
                source=str(image_path),
                conf=args.conf,
                classes=TARGET_CLASS_IDS,
                verbose=False,
                device=args.yolo_device,
            )[0]

            detection = select_largest_valid_vehicle(
                yolo_result,
                image_w,
                image_h,
                args.min_area_ratio,
            )

            if detection is None:
                skipped.append(
                    {
                        "path": str(image_path),
                        "filename": image_path.name,
                        "view": view,
                        "reason": "NO_VALID_CAR_TRUCK_DETECTION",
                    }
                )
                view_stats[view]["skipped"] += 1

                # copy for easy inspection
                dst = skipped_dir / f"{view}__{image_path.name}"
                shutil.copy2(image_path, dst)
                continue

            ex1, ey1, ex2, ey2 = expand_box(
                detection["bbox"],
                image_w,
                image_h,
                args.context_expansion,
            )

            crop_bgr = image_bgr[ey1:ey2, ex1:ex2]

            if crop_bgr.size == 0:
                skipped.append(
                    {
                        "path": str(image_path),
                        "filename": image_path.name,
                        "view": view,
                        "reason": "EMPTY_CONTEXT_CROP",
                    }
                )
                view_stats[view]["skipped"] += 1
                continue

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            tensor = transform(crop_pil).unsqueeze(0).to(classifier_device)

            with torch.no_grad():
                with torch.autocast(
                    device_type=classifier_device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = classifier(tensor)

                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())

            predicted_class = idx_to_class[pred_idx]
            probability_real = float(probs[real_idx].item())
            probability_toy = float(probs[toy_idx].item())

            row = {
                "path": str(image_path),
                "filename": image_path.name,
                "view": view,
                "true_class": "real",
                "predicted_class": predicted_class,
                "probability_real": probability_real,
                "probability_toy_scale": probability_toy,
                "correct": predicted_class == "real",
                "yolo_class_id": detection["class_id"],
                "yolo_class_name": yolo.names[detection["class_id"]],
                "yolo_confidence": detection["confidence"],
                "bbox_area_ratio": detection["area_ratio"],
                "crop_x1": ex1,
                "crop_y1": ey1,
                "crop_x2": ex2,
                "crop_y2": ey2,
            }

            predictions.append(row)

            view_stats[view]["evaluated"] += 1

            if predicted_class == "real":
                view_stats[view]["predicted_real"] += 1
            else:
                view_stats[view]["predicted_toy"] += 1

                # Save original image for visual inspection
                dst_original = (
                    false_rejects_dir
                    / f"{view}__{image_path.name}"
                )
                shutil.copy2(image_path, dst_original)

                # Also save the exact contextual crop seen by the classifier
                crop_suffix = image_path.suffix.lower()
                crop_name = (
                    f"{view}__{image_path.stem}"
                    f"__context25{crop_suffix}"
                )
                crop_dst = false_rejects_dir / crop_name
                cv2.imwrite(str(crop_dst), crop_bgr)

    elapsed_total = time.perf_counter() - start_total

    if not predictions:
        raise RuntimeError(
            "No images were evaluated. Check YOLO detections and dataset paths."
        )

    total_evaluated = len(predictions)
    predicted_real = sum(
        row["predicted_class"] == "real"
        for row in predictions
    )
    predicted_toy = sum(
        row["predicted_class"] == "toy_scale"
        for row in predictions
    )

    real_recall = predicted_real / total_evaluated
    false_rejection_rate = predicted_toy / total_evaluated

    real_probs = np.array(
        [row["probability_real"] for row in predictions],
        dtype=np.float64,
    )

    toy_probs = np.array(
        [row["probability_toy_scale"] for row in predictions],
        dtype=np.float64,
    )

    false_reject_rows = [
        row
        for row in predictions
        if row["predicted_class"] == "toy_scale"
    ]

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_path = output_dir / "predictions.csv"

    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(predictions[0].keys()),
        )
        writer.writeheader()
        writer.writerows(predictions)

    # --------------------------------------------------------
    # Save false rejects
    # --------------------------------------------------------

    false_rejects_csv = output_dir / "false_rejects.csv"

    with false_rejects_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(predictions[0].keys()),
        )
        writer.writeheader()

        if false_reject_rows:
            writer.writerows(false_reject_rows)

    # --------------------------------------------------------
    # Save skipped
    # --------------------------------------------------------

    skipped_csv = output_dir / "skipped.csv"

    with skipped_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = ["path", "filename", "view", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if skipped:
            writer.writerows(skipped)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    per_view_metrics = {}

    for view in VIEWS:
        stats = view_stats[view]

        evaluated = stats["evaluated"]

        view_real_recall = (
            stats["predicted_real"] / evaluated
            if evaluated > 0
            else None
        )

        view_false_reject = (
            stats["predicted_toy"] / evaluated
            if evaluated > 0
            else None
        )

        per_view_metrics[view] = {
            **stats,
            "real_recall": view_real_recall,
            "false_rejection_rate": view_false_reject,
        }

    metrics = {
        "dataset": str(dataset_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_balanced_accuracy": checkpoint.get(
            "val_balanced_accuracy"
        ),
        "class_to_idx": class_to_idx,
        "yolo_model": args.yolo_model,
        "yolo_confidence": args.conf,
        "minimum_bbox_area_ratio": args.min_area_ratio,
        "context_expansion": args.context_expansion,
        "total_images_discovered": total_discovered,
        "total_images_evaluated": total_evaluated,
        "total_skipped": len(skipped),
        "predicted_real": predicted_real,
        "predicted_toy_scale": predicted_toy,
        "real_recall": real_recall,
        "false_rejection_rate": false_rejection_rate,
        "real_probability_mean": float(real_probs.mean()),
        "real_probability_median": float(np.median(real_probs)),
        "real_probability_min": float(real_probs.min()),
        "real_probability_max": float(real_probs.max()),
        "toy_probability_mean": float(toy_probs.mean()),
        "toy_probability_median": float(np.median(toy_probs)),
        "toy_probability_min": float(toy_probs.min()),
        "toy_probability_max": float(toy_probs.max()),
        "elapsed_seconds": elapsed_total,
        "mean_end_to_end_ms_per_evaluated_image": (
            elapsed_total / total_evaluated * 1000.0
        ),
        "throughput_evaluated_images_per_second": (
            total_evaluated / elapsed_total
            if elapsed_total > 0
            else None
        ),
        "per_view": per_view_metrics,
    }

    metrics_path = output_dir / "metrics.json"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("EXTERNAL REAL-CAR RESULTS")
    print("=" * 90)

    print(
        f"{'View':<10}"
        f"{'Found':>8}"
        f"{'Eval':>8}"
        f"{'Real':>8}"
        f"{'Toy':>8}"
        f"{'Skipped':>10}"
        f"{'Real recall':>14}"
    )

    print("-" * 72)

    for view in VIEWS:
        stats = per_view_metrics[view]

        recall_text = (
            f"{stats['real_recall'] * 100:.2f}%"
            if stats["real_recall"] is not None
            else "N/A"
        )

        print(
            f"{view:<10}"
            f"{stats['images']:>8}"
            f"{stats['evaluated']:>8}"
            f"{stats['predicted_real']:>8}"
            f"{stats['predicted_toy']:>8}"
            f"{stats['skipped']:>10}"
            f"{recall_text:>14}"
        )

    print("-" * 72)

    print(f"\nTotal images discovered       : {total_discovered}")
    print(f"Total images evaluated        : {total_evaluated}")
    print(f"Skipped before classifier     : {len(skipped)}")
    print(f"Predicted real                : {predicted_real}")
    print(f"Predicted toy_scale           : {predicted_toy}")
    print(
        f"Real recall                   : "
        f"{real_recall:.4f} ({real_recall * 100:.2f}%)"
    )
    print(
        f"False rejection rate          : "
        f"{false_rejection_rate:.4f} "
        f"({false_rejection_rate * 100:.2f}%)"
    )

    print("\nConfidence summary:")
    print(
        f"Real probability mean         : "
        f"{real_probs.mean():.4f}"
    )
    print(
        f"Real probability median       : "
        f"{np.median(real_probs):.4f}"
    )
    print(
        f"Real probability minimum      : "
        f"{real_probs.min():.4f}"
    )

    if false_reject_rows:
        false_toy_probs = np.array(
            [
                row["probability_toy_scale"]
                for row in false_reject_rows
            ],
            dtype=np.float64,
        )

        print(
            f"\nFalse-reject toy prob mean    : "
            f"{false_toy_probs.mean():.4f}"
        )
        print(
            f"False-reject toy prob max     : "
            f"{false_toy_probs.max():.4f}"
        )

    print("\nRuntime:")
    print(f"Total end-to-end time         : {elapsed_total:.3f} s")
    print(
        f"Mean end-to-end / eval image  : "
        f"{elapsed_total / total_evaluated * 1000.0:.2f} ms"
    )
    print(
        f"Throughput                    : "
        f"{total_evaluated / elapsed_total:.2f} images/s"
    )

    print("\nSaved:")
    print(f"Predictions                   : {predictions_path}")
    print(f"Metrics                       : {metrics_path}")
    print(f"False rejects CSV             : {false_rejects_csv}")
    print(f"False rejects folder          : {false_rejects_dir}")
    print(f"Skipped CSV                   : {skipped_csv}")
    print(f"Skipped images folder         : {skipped_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
