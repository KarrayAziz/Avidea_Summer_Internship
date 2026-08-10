#!/usr/bin/env python3
"""
Evaluate the existing MobileNetV3-Small car-authenticity classifier on the
external Roboflow toy-car challenge set.

All images in the challenge folder are treated as ground-truth `toy_scale`.

The script:
- Loads the saved best MobileNetV3-Small checkpoint
- Uses the same ImageNet normalization and 224x224 preprocessing
- Runs inference on every contextual crop
- Reports toy recall / false acceptance rate
- Saves:
    predictions.csv
    metrics.json
    false_accepts/
    false_accepts.csv

A false accept is:
    true toy_scale -> predicted real
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MobileNetV3-Small on external Roboflow toy cars."
    )

    parser.add_argument(
        "--challenge-dir",
        type=Path,
        default=Path(
            "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/"
            "data/toy_cars_roboflow/external_authenticity_challenge/"
            "contextual_crops_capped"
        ),
        help="Folder containing the external toy contextual crops.",
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
        "--output",
        type=Path,
        default=Path(
            "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/"
            "models/car_authenticity_mobilenetv3/"
            "roboflow_external_eval"
        ),
        help="Evaluation output directory.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device: "auto", "cuda", "cpu", "cuda:0", etc.',
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision on CUDA.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous evaluation output first.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    return device


class ExternalToyDataset(Dataset):
    def __init__(self, folder: Path, transform):
        self.folder = folder
        self.transform = transform

        self.images = sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.images:
            raise RuntimeError(f"No images found in: {folder}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        return image, str(image_path)


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


def main():
    args = parse_args()

    challenge_dir = args.challenge_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not challenge_dir.exists():
        raise FileNotFoundError(f"Challenge folder not found:\n{challenge_dir}")

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

    false_accepts_dir = output_dir / "false_accepts"
    output_dir.mkdir(parents=True, exist_ok=True)
    false_accepts_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    print("=" * 88)
    print("EXTERNAL ROBOFLOW TOY-CAR EVALUATION")
    print("=" * 88)
    print(f"Challenge folder : {challenge_dir}")
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"Output           : {output_dir}")
    print(f"Device           : {device}")
    print(f"AMP              : {amp_enabled}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    class_to_idx = checkpoint.get(
        "class_to_idx",
        {"real": 0, "toy_scale": 1},
    )

    if "real" not in class_to_idx or "toy_scale" not in class_to_idx:
        raise RuntimeError(
            f"Checkpoint class mapping does not contain real/toy_scale:\n"
            f"{class_to_idx}"
        )

    idx_to_class = {
        idx: name
        for name, idx in class_to_idx.items()
    }

    real_idx = class_to_idx["real"]
    toy_idx = class_to_idx["toy_scale"]

    image_size = int(checkpoint.get("image_size", 224))
    dropout = float(checkpoint.get("dropout", 0.3))

    print(f"Image size       : {image_size}")
    print(f"Class mapping    : {class_to_idx}")
    print(f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}")
    print(
        f"Checkpoint val balanced accuracy: "
        f"{checkpoint.get('val_balanced_accuracy', 'unknown')}"
    )

    model = build_model(
        dropout=dropout,
        num_classes=len(class_to_idx),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = build_transform(image_size)

    dataset = ExternalToyDataset(
        challenge_dir,
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    print(f"\nExternal toy crops: {len(dataset)}")

    predictions = []

    start = time.perf_counter()

    with torch.no_grad():
        progress = tqdm(loader, desc="Inference")

        for images, paths in progress:
            images = images.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)

            probs = torch.softmax(logits, dim=1)
            pred_indices = logits.argmax(dim=1)

            probs_cpu = probs.cpu().numpy()
            pred_indices_cpu = pred_indices.cpu().numpy()

            for path, pred_idx, prob_vector in zip(
                paths,
                pred_indices_cpu,
                probs_cpu,
            ):
                pred_idx = int(pred_idx)
                predicted_class = idx_to_class[pred_idx]

                predictions.append(
                    {
                        "path": path,
                        "filename": Path(path).name,
                        "true_class": "toy_scale",
                        "predicted_class": predicted_class,
                        "probability_real": float(prob_vector[real_idx]),
                        "probability_toy_scale": float(prob_vector[toy_idx]),
                        "correct": predicted_class == "toy_scale",
                    }
                )

    elapsed = time.perf_counter() - start

    total = len(predictions)
    predicted_toy = sum(
        row["predicted_class"] == "toy_scale"
        for row in predictions
    )
    predicted_real = sum(
        row["predicted_class"] == "real"
        for row in predictions
    )

    toy_recall = predicted_toy / total if total else 0.0
    false_accept_rate = predicted_real / total if total else 0.0

    toy_probs = np.array(
        [row["probability_toy_scale"] for row in predictions],
        dtype=np.float64,
    )

    real_probs = np.array(
        [row["probability_real"] for row in predictions],
        dtype=np.float64,
    )

    false_accept_rows = [
        row
        for row in predictions
        if row["predicted_class"] == "real"
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
    # Export false accepts
    # --------------------------------------------------------

    false_accepts_csv = output_dir / "false_accepts.csv"

    if false_accept_rows:
        with false_accepts_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(false_accept_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(false_accept_rows)

        for row in false_accept_rows:
            src = Path(row["path"])
            dst = false_accepts_dir / src.name

            if dst.exists():
                stem = src.stem
                suffix = src.suffix
                counter = 1

                while dst.exists():
                    dst = (
                        false_accepts_dir
                        / f"{stem}__{counter}{suffix}"
                    )
                    counter += 1

            shutil.copy2(src, dst)

    else:
        # still create an empty CSV with headers
        with false_accepts_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(predictions[0].keys()),
            )
            writer.writeheader()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {
        "challenge_dataset": "roboflow_toy_cars_hqi4o_v2",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_balanced_accuracy": checkpoint.get(
            "val_balanced_accuracy"
        ),
        "class_to_idx": class_to_idx,
        "total_toy_crops": total,
        "predicted_toy_scale": predicted_toy,
        "predicted_real": predicted_real,
        "toy_recall": toy_recall,
        "false_acceptance_rate": false_accept_rate,
        "toy_probability_mean": float(toy_probs.mean()),
        "toy_probability_median": float(np.median(toy_probs)),
        "toy_probability_min": float(toy_probs.min()),
        "toy_probability_max": float(toy_probs.max()),
        "real_probability_mean": float(real_probs.mean()),
        "real_probability_median": float(np.median(real_probs)),
        "real_probability_min": float(real_probs.min()),
        "real_probability_max": float(real_probs.max()),
        "inference_seconds": elapsed,
        "mean_ms_per_image": (elapsed / total * 1000.0) if total else None,
        "throughput_images_per_second": (total / elapsed) if elapsed > 0 else None,
    }

    metrics_path = output_dir / "metrics.json"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 88)
    print("EXTERNAL CHALLENGE RESULTS")
    print("=" * 88)
    print(f"Total toy crops              : {total}")
    print(f"Predicted toy_scale          : {predicted_toy}")
    print(f"Predicted real               : {predicted_real}")
    print(f"Toy recall                   : {toy_recall:.4f} ({toy_recall * 100:.2f}%)")
    print(
        f"False acceptance rate        : "
        f"{false_accept_rate:.4f} ({false_accept_rate * 100:.2f}%)"
    )

    print("\nConfidence summary:")
    print(
        f"Toy probability mean         : "
        f"{toy_probs.mean():.4f}"
    )
    print(
        f"Toy probability median       : "
        f"{np.median(toy_probs):.4f}"
    )
    print(
        f"Toy probability minimum      : "
        f"{toy_probs.min():.4f}"
    )

    if false_accept_rows:
        false_real_probs = np.array(
            [
                row["probability_real"]
                for row in false_accept_rows
            ],
            dtype=np.float64,
        )

        print(
            f"\nFalse-accept real prob mean   : "
            f"{false_real_probs.mean():.4f}"
        )
        print(
            f"False-accept real prob max    : "
            f"{false_real_probs.max():.4f}"
        )

    print("\nRuntime:")
    print(f"Total inference time          : {elapsed:.3f} s")
    print(
        f"Mean end-to-end / image       : "
        f"{elapsed / total * 1000.0:.2f} ms"
    )
    print(
        f"Throughput                    : "
        f"{total / elapsed:.2f} images/s"
    )

    print("\nSaved:")
    print(f"Predictions                   : {predictions_path}")
    print(f"Metrics                       : {metrics_path}")
    print(f"False accepts CSV             : {false_accepts_csv}")
    print(f"False accepts folder          : {false_accepts_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
