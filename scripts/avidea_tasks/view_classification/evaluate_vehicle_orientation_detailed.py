#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torchvision import models, transforms
from tqdm import tqdm
from ultralytics import YOLO

CLASS_NAMES = ["back", "front", "left", "right"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_VEHICLE_CLASSES = [2, 7]  # car, truck

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the ResNet-18 vehicle-view classifier with YOLOv8 preprocessing and detailed latency metrics."
    )
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yolo-conf", type=float, default=0.30)
    parser.add_argument("--min-area-percent", type=float, default=8.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--save-all-predictions", action="store_true")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_latency(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {k: 0.0 for k in ["count", "total_ms", "mean_ms", "min_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms"]}
    a = np.asarray(values_ms, dtype=np.float64)
    return {
        "count": int(a.size),
        "total_ms": float(a.sum()),
        "mean_ms": float(a.mean()),
        "min_ms": float(a.min()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
    }


def discover_samples(root: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for class_name in CLASS_NAMES:
        folder = root / class_name
        if not folder.exists():
            print(f"Warning: missing class folder: {folder}")
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                samples.append((path, CLASS_TO_IDX[class_name]))
    return samples


def load_resnet(weights: Path, device: torch.device) -> tuple[nn.Module, float]:
    sync(device)
    start = time.perf_counter()
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    sync(device)
    return model, time.perf_counter() - start


def load_yolo(weights: Path) -> tuple[YOLO, float]:
    start = time.perf_counter()
    model = YOLO(str(weights))
    return model, time.perf_counter() - start


def preprocess_one_image(
    image_path: Path,
    yolo_model: YOLO,
    yolo_conf: float,
    min_area_percent: float,
    yolo_device: str,
) -> tuple[Image.Image, dict[str, Any], Any]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_area = float(width * height)

    results = yolo_model.predict(
        source=str(image_path),
        conf=yolo_conf,
        classes=YOLO_VEHICLE_CLASSES,
        device=yolo_device,
        verbose=False,
    )

    metadata: dict[str, Any] = {
        "used_crop": False,
        "fallback_reason": "no_detection",
        "vehicle_area_percent": 0.0,
        "crop_box": None,
        "detections": 0,
    }

    if not results:
        return image, metadata, None

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return image, metadata, result

    boxes = result.boxes.xyxy.detach().cpu()
    metadata["detections"] = int(len(boxes))

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = int(torch.argmax(areas).item())
    largest_area = float(areas[largest_idx].item())
    area_percent = (largest_area / image_area) * 100.0
    metadata["vehicle_area_percent"] = area_percent

    if area_percent < min_area_percent:
        metadata["fallback_reason"] = "below_area_threshold"
        return image, metadata, result

    x1, y1, x2, y2 = map(int, boxes[largest_idx].tolist())
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    metadata.update({
        "used_crop": True,
        "fallback_reason": "",
        "crop_box": [x1, y1, x2, y2],
    })

    # Exactly one crop per image: the largest valid car/truck detection.
    return image.crop((x1, y1, x2, y2)), metadata, result


def save_debug(
    output: Path,
    source_path: Path,
    true_view: str,
    pred_view: str,
    yolo_result: Any,
    model_input: Image.Image,
    used_crop: bool,
) -> None:
    folder = output / f"misclassified_{true_view}"
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"{true_view}_classified_as_{pred_view}"
    base = source_path.name

    shutil.copy2(source_path, folder / f"{prefix}_RAW_{base}")

    if yolo_result is not None:
        cv2.imwrite(str(folder / f"{prefix}_YOLOBOX_{base}"), yolo_result.plot())

    kind = "CROP" if used_crop else "RAW_FALLBACK"
    model_input.save(folder / f"{prefix}_{kind}_{base}")


def warm_up(
    yolo_model: YOLO,
    resnet_model: nn.Module,
    samples: list[tuple[Path, int]],
    device: torch.device,
    yolo_device: str,
    runs: int,
) -> dict[str, float]:
    if runs <= 0 or not samples:
        return {"runs": 0, "total_seconds": 0.0, "average_seconds_per_run": 0.0}

    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(runs):
            image_path, _ = samples[i % len(samples)]
            yolo_model.predict(
                source=str(image_path),
                conf=0.30,
                classes=YOLO_VEHICLE_CLASSES,
                device=yolo_device,
                verbose=False,
            )
            image = Image.open(image_path).convert("RGB")
            tensor = TEST_TRANSFORM(image).unsqueeze(0).to(device)
            sync(device)
            _ = resnet_model(tensor)
            sync(device)
    elapsed = time.perf_counter() - start
    return {"runs": runs, "total_seconds": elapsed, "average_seconds_per_run": elapsed / runs}


def main() -> None:
    args = parse_args()

    for path, label in [
        (args.test_dir, "test directory"),
        (args.weights, "ResNet weights"),
        (args.yolo_weights, "YOLO weights"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    if args.clear_output and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    requested = args.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        requested = "cpu"

    device = torch.device(requested)
    yolo_device = requested
    samples = discover_samples(args.test_dir)
    if not samples:
        raise RuntimeError(f"No test images found in {args.test_dir}")

    print("=" * 72)
    print("VEHICLE VIEW CLASSIFICATION EVALUATION")
    print("=" * 72)
    print(f"Images:              {len(samples)}")
    print(f"Device:              {device}")
    print(f"YOLO classes:        {YOLO_VEHICLE_CLASSES} (car, truck)")
    print(f"Minimum crop area:   {args.min_area_percent:.2f}%")
    print("Crop policy:         largest valid detection only; raw fallback")

    yolo_model, yolo_load_s = load_yolo(args.yolo_weights)
    resnet_model, resnet_load_s = load_resnet(args.weights, device)
    total_load_s = yolo_load_s + resnet_load_s

    warmup = warm_up(yolo_model, resnet_model, samples, device, yolo_device, args.warmup_runs)

    labels: list[int] = []
    preds: list[int] = []
    rows: list[dict[str, Any]] = []
    yolo_times: list[float] = []
    prep_times: list[float] = []
    resnet_times: list[float] = []
    e2e_times: list[float] = []

    cropped = 0
    raw_no_detection = 0
    raw_below_threshold = 0
    misclassified = 0

    eval_start = time.perf_counter()

    with torch.inference_mode():
        for image_path, true_idx in tqdm(samples, desc="Evaluating", unit="image"):
            image_start = time.perf_counter()

            yolo_start = time.perf_counter()
            model_input, crop_meta, yolo_result = preprocess_one_image(
                image_path,
                yolo_model,
                args.yolo_conf,
                args.min_area_percent,
                yolo_device,
            )
            yolo_ms = (time.perf_counter() - yolo_start) * 1000.0

            prep_start = time.perf_counter()
            tensor = TEST_TRANSFORM(model_input).unsqueeze(0).to(device)
            sync(device)
            prep_ms = (time.perf_counter() - prep_start) * 1000.0

            inf_start = time.perf_counter()
            logits = resnet_model(tensor)
            sync(device)
            inf_ms = (time.perf_counter() - inf_start) * 1000.0

            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
            e2e_ms = (time.perf_counter() - image_start) * 1000.0

            labels.append(true_idx)
            preds.append(pred_idx)
            yolo_times.append(yolo_ms)
            prep_times.append(prep_ms)
            resnet_times.append(inf_ms)
            e2e_times.append(e2e_ms)

            if crop_meta["used_crop"]:
                cropped += 1
            elif crop_meta["fallback_reason"] == "no_detection":
                raw_no_detection += 1
            else:
                raw_below_threshold += 1

            true_view = CLASS_NAMES[true_idx]
            pred_view = CLASS_NAMES[pred_idx]

            if pred_idx != true_idx:
                misclassified += 1
                save_debug(args.output, image_path, true_view, pred_view, yolo_result, model_input, bool(crop_meta["used_crop"]))

            rows.append({
                "path": str(image_path),
                "true_view": true_view,
                "predicted_view": pred_view,
                "correct": pred_idx == true_idx,
                "prediction_confidence": confidence,
                "prob_back": float(probs[0].item()),
                "prob_front": float(probs[1].item()),
                "prob_left": float(probs[2].item()),
                "prob_right": float(probs[3].item()),
                "used_crop": bool(crop_meta["used_crop"]),
                "fallback_reason": crop_meta["fallback_reason"],
                "vehicle_area_percent": crop_meta["vehicle_area_percent"],
                "detections": crop_meta["detections"],
                "crop_box": json.dumps(crop_meta["crop_box"]) if crop_meta["crop_box"] else "",
                "yolo_ms": yolo_ms,
                "preprocessing_ms": prep_ms,
                "resnet_inference_ms": inf_ms,
                "end_to_end_ms": e2e_ms,
            })

    total_eval_s = time.perf_counter() - eval_start

    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(4)),
        zero_division=0,
    )
    recall_per_view = {name: float(recall[i]) for i, name in enumerate(CLASS_NAMES)}

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(4)),
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(4)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))

    timing = {
        "yolo_detection": summarize_latency(yolo_times),
        "preprocessing": summarize_latency(prep_times),
        "resnet_inference": summarize_latency(resnet_times),
        "end_to_end": summarize_latency(e2e_times),
    }
    throughput = len(samples) / total_eval_s if total_eval_s > 0 else 0.0

    metrics = {
        "configuration": {
            "test_dir": str(args.test_dir),
            "resnet_weights": str(args.weights),
            "yolo_weights": str(args.yolo_weights),
            "device": str(device),
            "class_names": CLASS_NAMES,
            "yolo_classes": YOLO_VEHICLE_CLASSES,
            "yolo_confidence_threshold": args.yolo_conf,
            "minimum_vehicle_area_percent": args.min_area_percent,
            "crop_policy": "Exactly one crop: largest valid car/truck; raw image fallback otherwise.",
        },
        "model_loading": {
            "yolo_seconds": yolo_load_s,
            "resnet_seconds": resnet_load_s,
            "total_seconds": total_load_s,
        },
        "warmup": warmup,
        "evaluation": {
            "image_count": len(samples),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "recall_per_view": recall_per_view,
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "misclassified_count": misclassified,
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
        },
        "crop_statistics": {
            "cropped_images": cropped,
            "raw_fallback_no_detection": raw_no_detection,
            "raw_fallback_below_area_threshold": raw_below_threshold,
            "crop_rate": cropped / len(samples),
        },
        "timing": {
            "total_evaluation_seconds": total_eval_s,
            "throughput_images_per_second": throughput,
            "latency": timing,
        },
    }

    metrics_path = args.output / "view_classification_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_path = args.output / "view_classification_predictions.csv"
    if args.save_all_predictions:
        with predictions_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "=" * 72)
    print("UNBIASED TEST REPORT")
    print("=" * 72)
    print(report_text)
    print(f"Accuracy:          {accuracy:.4%}")
    print(f"Balanced accuracy: {balanced_accuracy:.4%}")
    print("\nRecall per view:")
    for name in CLASS_NAMES:
        print(f"  {name:<6}: {recall_per_view[name]:.4%}")

    print("\nConfusion matrix — rows=true, columns=predicted")
    header = f"{'True / Pred':<13}" + "".join(f"{name:<10}" for name in CLASS_NAMES)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<13}" + "".join(f"{int(cm[i, j]):<10}" for j in range(4)))

    print("\nCrop statistics:")
    print(f"  Cropped images:                     {cropped}")
    print(f"  Raw fallback — no detection:        {raw_no_detection}")
    print(f"  Raw fallback — below 8% threshold:  {raw_below_threshold}")

    print("\nModel loading:")
    print(f"  YOLO:      {yolo_load_s:.4f} s")
    print(f"  ResNet-18: {resnet_load_s:.4f} s")
    print(f"  Total:     {total_load_s:.4f} s")

    print("\nLatency, warm-up excluded:")
    for name, values in timing.items():
        print(
            f"  {name:<20} mean={values['mean_ms']:.2f} ms | "
            f"p50={values['p50_ms']:.2f} ms | p95={values['p95_ms']:.2f} ms | "
            f"p99={values['p99_ms']:.2f} ms"
        )

    print(f"\nThroughput:         {throughput:.2f} images/s")
    print(f"Total evaluation:   {total_eval_s:.4f} s")
    print(f"Misclassifications: {misclassified}")
    print(f"Metrics JSON:       {metrics_path}")
    if args.save_all_predictions:
        print(f"Predictions CSV:    {predictions_path}")
    print(f"Debug output:       {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
