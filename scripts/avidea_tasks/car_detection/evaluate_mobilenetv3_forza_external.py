#!/usr/bin/env python3

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


PARQUET_PATH = Path(
    "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/data/"
    "forza_horizon_external/data/train-00000-of-00001-e37f623aec48dc8e.parquet"
)

CHECKPOINT_PATH = Path(
    "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/models/"
    "car_authenticity_mobilenetv3/best_mobilenet_v3_small.pth"
)

OUTPUT_DIR = Path(
    "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/models/"
    "car_authenticity_mobilenetv3/forza_external_eval"
)

YOLO_WEIGHTS = "yolov8m.pt"
YOLO_DEVICE = "cpu"
YOLO_CONF = 0.25
MIN_AREA_RATIO = 0.08
CONTEXT_EXPANSION = 0.25
TARGET_CLASS_IDS = [2, 7]  # car, truck

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def decode_image(obj):
    if not isinstance(obj, dict) or obj.get("bytes") is None:
        raise ValueError("Unexpected Hugging Face image field")
    return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")


def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    class_to_idx = checkpoint["class_to_idx"]
    image_size = int(checkpoint.get("image_size", 224))
    dropout = float(checkpoint.get("dropout", 0.3))

    model = models.mobilenet_v3_small(weights=None)

    # Match the training script exactly
    in_features = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[3] = nn.Linear(in_features, len(class_to_idx))

    # IMPORTANT: the .pth is a checkpoint dictionary, not a raw state_dict
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return checkpoint, model, transform, class_to_idx


def select_largest_valid_vehicle(result, image_w, image_h):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    image_area = image_w * image_h
    best = None
    best_area = -1.0

    for box in result.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in TARGET_CLASS_IDS:
            continue

        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h
        area_ratio = area / image_area if image_area else 0.0

        if area_ratio < MIN_AREA_RATIO:
            continue

        if area > best_area:
            best_area = area
            best = {
                "bbox": (x1, y1, x2, y2),
                "class_id": cls_id,
                "confidence": conf,
                "area_ratio": area_ratio,
            }

    return best


def expand_crop(image, bbox):
    x1, y1, x2, y2 = bbox
    img_w, img_h = image.size

    bw = x2 - x1
    bh = y2 - y1

    dx = bw * CONTEXT_EXPANSION
    dy = bh * CONTEXT_EXPANSION

    cx1 = max(0, int(round(x1 - dx)))
    cy1 = max(0, int(round(y1 - dy)))
    cx2 = min(img_w, int(round(x2 + dx)))
    cy2 = min(img_h, int(round(y2 + dy)))

    if cx2 <= cx1 or cy2 <= cy1:
        return None, None

    return image.crop((cx1, cy1, cx2, cy2)), (cx1, cy1, cx2, cy2)


@torch.no_grad()
def classify(model, transform, crop, class_to_idx):
    x = transform(crop).unsqueeze(0).to(DEVICE)

    with torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.float16,
        enabled=USE_AMP,
    ):
        logits = model(x)

    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return {
        "predicted_label": idx_to_class[pred_idx],
        "prob_real": float(probs[class_to_idx["real"]].item()),
        "prob_toy_scale": float(probs[class_to_idx["toy_scale"]].item()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = OUTPUT_DIR / "raw_images"
    crop_dir = OUTPUT_DIR / "contextual_crops"
    false_dir = OUTPUT_DIR / "false_accepts"

    raw_dir.mkdir(exist_ok=True)
    crop_dir.mkdir(exist_ok=True)
    false_dir.mkdir(exist_ok=True)

    print("=" * 88)
    print("EXTERNAL FORZA VIDEOGAME-CAR EVALUATION")
    print("=" * 88)

    df = pd.read_parquet(PARQUET_PATH)
    checkpoint, model, transform, class_to_idx = load_model()
    yolo = YOLO(YOLO_WEIGHTS)

    print(f"Rows in parquet   : {len(df)}")
    print(f"Device            : {DEVICE}")
    print(f"Class mapping     : {class_to_idx}")
    print(f"Checkpoint epoch  : {checkpoint.get('epoch')}")
    print(
        f"Checkpoint val balanced accuracy: "
        f"{checkpoint.get('val_balanced_accuracy')}"
    )

    rows = []
    skipped = []
    latencies_ms = []

    start_total = time.perf_counter()

    for i, row in df.iterrows():
        caption = str(row["text"])

        try:
            image = decode_image(row["image"])
        except Exception as exc:
            skipped.append({
                "index": i,
                "caption": caption,
                "reason": f"decode_failed: {exc}",
            })
            continue

        raw_path = raw_dir / f"{i:04d}.png"
        image.save(raw_path)

        w, h = image.size

        t0 = time.perf_counter()

        result = yolo.predict(
            source=np.array(image),
            conf=YOLO_CONF,
            classes=TARGET_CLASS_IDS,
            device=YOLO_DEVICE,
            verbose=False,
        )[0]

        det = select_largest_valid_vehicle(result, w, h)

        if det is None:
            skipped.append({
                "index": i,
                "caption": caption,
                "reason": "no_valid_car_or_truck_detection",
            })
            continue

        crop, crop_bbox = expand_crop(image, det["bbox"])

        if crop is None:
            skipped.append({
                "index": i,
                "caption": caption,
                "reason": "invalid_crop",
            })
            continue

        crop_path = crop_dir / f"{i:04d}.png"
        crop.save(crop_path)

        pred = classify(model, transform, crop, class_to_idx)

        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        item = {
            "index": i,
            "caption": caption,
            "expected_label": "toy_scale",
            "predicted_label": pred["predicted_label"],
            "prob_real": pred["prob_real"],
            "prob_toy_scale": pred["prob_toy_scale"],
            "yolo_class_id": det["class_id"],
            "yolo_confidence": det["confidence"],
            "bbox_area_ratio": det["area_ratio"],
            "crop_bbox": crop_bbox,
            "raw_image_path": str(raw_path),
            "crop_path": str(crop_path),
        }

        rows.append(item)

        if pred["predicted_label"] == "real":
            image.save(false_dir / f"{i:04d}_raw.png")
            crop.save(false_dir / f"{i:04d}_crop.png")

    elapsed = time.perf_counter() - start_total

    pred_df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)

    pred_path = OUTPUT_DIR / "predictions.csv"
    skipped_path = OUTPUT_DIR / "skipped.csv"
    metrics_path = OUTPUT_DIR / "metrics.json"

    pred_df.to_csv(pred_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    evaluated = len(pred_df)
    predicted_toy = int(
        (pred_df["predicted_label"] == "toy_scale").sum()
    ) if evaluated else 0
    predicted_real = int(
        (pred_df["predicted_label"] == "real").sum()
    ) if evaluated else 0

    rejection_rate = predicted_toy / evaluated if evaluated else 0.0
    false_acceptance_rate = predicted_real / evaluated if evaluated else 0.0

    toy_probs = (
        pred_df["prob_toy_scale"].to_numpy(dtype=float)
        if evaluated else np.array([])
    )

    metrics = {
        "total_forza_images": len(df),
        "evaluated_images": evaluated,
        "skipped_images": len(skipped_df),
        "predicted_toy_scale": predicted_toy,
        "predicted_real": predicted_real,
        "videogame_rejection_rate": rejection_rate,
        "false_acceptance_rate": false_acceptance_rate,
        "toy_probability_mean": float(toy_probs.mean()) if evaluated else None,
        "toy_probability_median": float(np.median(toy_probs)) if evaluated else None,
        "toy_probability_min": float(toy_probs.min()) if evaluated else None,
        "total_runtime_seconds": elapsed,
        "mean_pipeline_ms_per_evaluated_image": (
            float(np.mean(latencies_ms)) if latencies_ms else None
        ),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 88)
    print("FORZA EXTERNAL RESULTS")
    print("=" * 88)
    print(f"Total Forza images          : {len(df)}")
    print(f"Images evaluated            : {evaluated}")
    print(f"Skipped before classifier   : {len(skipped_df)}")
    print(f"Predicted toy_scale         : {predicted_toy}")
    print(f"Predicted real              : {predicted_real}")

    if evaluated:
        print(
            f"Videogame rejection rate    : "
            f"{rejection_rate:.4f} ({rejection_rate * 100:.2f}%)"
        )
        print(
            f"False acceptance rate       : "
            f"{false_acceptance_rate:.4f} ({false_acceptance_rate * 100:.2f}%)"
        )
        print(f"Toy probability mean        : {toy_probs.mean():.4f}")
        print(f"Toy probability median      : {np.median(toy_probs):.4f}")
        print(f"Toy probability minimum     : {toy_probs.min():.4f}")

    print(f"\nPredictions                 : {pred_path}")
    print(f"Metrics                     : {metrics_path}")
    print(f"Skipped                     : {skipped_path}")
    print(f"False accepts               : {false_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
