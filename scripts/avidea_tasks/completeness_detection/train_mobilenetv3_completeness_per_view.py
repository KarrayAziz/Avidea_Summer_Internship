#!/usr/bin/env python3
"""
Train one MobileNetV3-Small completeness classifier per vehicle view.

Supported dataset layouts:

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

Classes:
    0 = complete
    1 = incomplete
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from tqdm import tqdm



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

CLASS_TO_INDEX = {"complete": 0, "incomplete": 1}
INDEX_TO_CLASS = {0: "complete", 1: "incomplete"}


@dataclass
class Sample:
    path: str
    view: str
    label_name: str
    label: int


class CompletenessDataset(Dataset):
    def __init__(self, samples: list[Sample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        try:
            image = Image.open(sample.path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Could not read image: {sample.path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.label, sample.path


def normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def infer_metadata(image_path: Path, dataset_root: Path) -> tuple[Optional[str], Optional[str]]:
    parts = image_path.relative_to(dataset_root).parts[:-1]
    tokens = [normalize_token(part) for part in parts]

    views = {VIEW_ALIASES[token] for token in tokens if token in VIEW_ALIASES}
    labels = {LABEL_ALIASES[token] for token in tokens if token in LABEL_ALIASES}

    view = next(iter(views)) if len(views) == 1 else None
    label = next(iter(labels)) if len(labels) == 1 else None
    return view, label


def discover_samples(dataset_root: Path) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    skipped: list[str] = []

    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        view, label_name = infer_metadata(path, dataset_root)
        if view is None or label_name is None:
            skipped.append(str(path))
            continue

        samples.append(
            Sample(
                path=str(path),
                view=view,
                label_name=label_name,
                label=CLASS_TO_INDEX[label_name],
            )
        )

    return samples, skipped


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def stratified_split(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must be positive and sum to less than 1.")

    labels = [sample.label for sample in samples]
    train_samples, temp_samples = train_test_split(
        samples,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=labels,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    temp_labels = [sample.label for sample in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_labels,
    )

    return list(train_samples), list(val_samples), list(test_samples)


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(4),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.RandomPerspective(distortion_scale=0.08, p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, eval_transform

def create_weighted_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    counts = Counter(sample.label for sample in samples)

    sample_weights = [
        1.0 / counts[sample.label]
        for sample in samples
    ]

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(samples),
        replacement=True,
    )


def build_model(pretrained: bool, dropout: float) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[3] = nn.Linear(in_features, 2)
    return model


def calculate_class_weights(samples: list[Sample], device: torch.device) -> torch.Tensor:
    counts = Counter(sample.label for sample in samples)
    total = len(samples)

    weights = []
    for class_index in (0, 1):
        count = counts.get(class_index, 0)
        if count == 0:
            raise RuntimeError(f"Training split contains no samples for class {class_index}.")
        weights.append(total / (2 * count))

    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_loader(
    samples: list[Sample],
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        CompletenessDataset(samples, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler,
    device: torch.device,
    use_amp: bool,
) -> dict:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        progress = tqdm(
            loader,
            desc="train" if training else "validation",
            leave=False,
            unit="batch",
        )

        for images, targets, _ in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())

            progress.set_postfix(loss=f"{loss.item():.4f}")

    precision, recall, f1, support = precision_recall_fscore_support(
        targets_all,
        predictions_all,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets_all, predictions_all),
        "balanced_accuracy": balanced_accuracy_score(targets_all, predictions_all),
        "complete_precision": float(precision[0]),
        "complete_recall": float(recall[0]),
        "complete_f1": float(f1[0]),
        "complete_support": int(support[0]),
        "incomplete_precision": float(precision[1]),
        "incomplete_recall": float(recall[1]),
        "incomplete_f1": float(f1[1]),
        "incomplete_support": int(support[1]),
    }


def save_split_manifest(
    path: Path,
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "path", "view", "label_name", "label"],
        )
        writer.writeheader()

        for split_name, split_samples in (
            ("train", train_samples),
            ("val", val_samples),
            ("test", test_samples),
        ):
            for sample in split_samples:
                writer.writerow({"split": split_name, **asdict(sample)})


def save_history(history: list[dict], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict, list[dict]]:
    model.eval()

    targets_all: list[int] = []
    predictions_all: list[int] = []
    incomplete_probabilities: list[float] = []
    paths_all: list[str] = []

    start = time.perf_counter()

    with torch.no_grad():
        for images, targets, paths in tqdm(loader, desc="test", leave=False, unit="batch"):
            images = images.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)

            predictions = probabilities.argmax(dim=1)

            targets_all.extend(targets.tolist())
            predictions_all.extend(predictions.cpu().tolist())
            incomplete_probabilities.extend(probabilities[:, 1].cpu().tolist())
            paths_all.extend(paths)

    elapsed = time.perf_counter() - start
    matrix = confusion_matrix(targets_all, predictions_all, labels=[0, 1])

    metrics = {
        "count": len(targets_all),
        "accuracy": accuracy_score(targets_all, predictions_all),
        "balanced_accuracy": balanced_accuracy_score(targets_all, predictions_all),
        "classification_report": classification_report(
            targets_all,
            predictions_all,
            labels=[0, 1],
            target_names=["complete", "incomplete"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "true_complete_pred_complete": int(matrix[0, 0]),
            "true_complete_pred_incomplete": int(matrix[0, 1]),
            "true_incomplete_pred_complete": int(matrix[1, 0]),
            "true_incomplete_pred_incomplete": int(matrix[1, 1]),
        },
        "timing": {
            "total_seconds": elapsed,
            "average_seconds_per_image": elapsed / len(targets_all) if targets_all else 0.0,
        },
    }

    rows = []
    for path, target, prediction, probability in zip(
        paths_all,
        targets_all,
        predictions_all,
        incomplete_probabilities,
    ):
        rows.append(
            {
                "path": path,
                "true_label": INDEX_TO_CLASS[target],
                "predicted_label": INDEX_TO_CLASS[prediction],
                "probability_incomplete": probability,
                "correct": target == prediction,
            }
        )

    return metrics, rows


def train_one_view(
    view: str,
    samples: list[Sample],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    output_dir = Path(args.output) / view
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples, test_samples = stratified_split(
        samples,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    save_split_manifest(
        output_dir / "split_manifest.csv",
        train_samples,
        val_samples,
        test_samples,
    )

    train_transform, eval_transform = build_transforms(args.image_size)
    pin_memory = device.type == "cuda"

    train_dataset = CompletenessDataset(
        train_samples,
        transform=train_transform,
    )

    train_sampler = create_weighted_sampler(train_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = create_loader(
        val_samples,
        eval_transform,
        args.batch_size,
        args.num_workers,
        False,
        pin_memory,
    )
    test_loader = create_loader(
        test_samples,
        eval_transform,
        args.batch_size,
        args.num_workers,
        False,
        pin_memory,
    )

    model = build_model(
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    class_weights = calculate_class_weights(train_samples, device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.min_learning_rate,
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_metric = -math.inf
    best_epoch = 0
    no_improvement = 0
    history: list[dict] = []
    checkpoint_path = output_dir / "best_mobilenet_v3_small.pth"

    print("\n" + "=" * 76)
    print(f"TRAINING {view.upper()}")
    print("=" * 76)
    print(
        f"train={len(train_samples)} | val={len(val_samples)} | "
        f"test={len(test_samples)}"
    )
    print(
        f"training labels: complete={sum(s.label == 0 for s in train_samples)}, "
        f"incomplete={sum(s.label == 1 for s in train_samples)}"
    )
    print(f"class weights: {class_weights.detach().cpu().tolist()}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            scaler,
            device,
            use_amp,
        )

        scheduler.step(val_metrics["balanced_accuracy"])

        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        save_history(history, output_dir / "training_history.csv")

        print(
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_bal_acc={train_metrics['balanced_accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_incomplete_recall={val_metrics['incomplete_recall']:.4f}"
        )

        current_metric = val_metrics["balanced_accuracy"]
        if current_metric > best_metric + args.min_delta:
            best_metric = current_metric
            best_epoch = epoch
            no_improvement = 0

            torch.save(
                {
                    "view": view,
                    "epoch": epoch,
                    "architecture": "mobilenet_v3_small",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_validation_balanced_accuracy": best_metric,
                    "class_to_index": CLASS_TO_INDEX,
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "pretrained": not args.no_pretrained,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "configuration": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            no_improvement += 1

        if no_improvement >= args.early_stopping_patience:
            print("Early stopping.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, prediction_rows = evaluate_test_set(
        model,
        test_loader,
        device,
        use_amp,
    )

    test_metrics.update(
        {
            "view": view,
            "best_epoch": best_epoch,
            "best_validation_balanced_accuracy": best_metric,
            "dataset_counts": {
                "all": len(samples),
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
        }
    )

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)

    with (output_dir / "test_predictions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(
        f"{view.upper()} TEST | accuracy={test_metrics['accuracy']:.4f} | "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}"
    )
    print(json.dumps(test_metrics["confusion_matrix"], indent=2))

    return test_metrics


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    dataset_root = Path(args.dataset).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_root}")

    samples, skipped = discover_samples(dataset_root)
    if not samples:
        raise RuntimeError("No labeled images were discovered.")

    with (output_root / "skipped_images.txt").open("w", encoding="utf-8") as file:
        file.write("\n".join(skipped))

    device = resolve_device(args.device)

    print(f"Dataset: {dataset_root}")
    print(f"Output:  {output_root}")
    print(f"Device:  {device}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    print(f"Images:  {len(samples)}")
    print(f"Skipped: {len(skipped)}")

    views = args.views or ["front", "back", "left", "right"]
    summary = {}

    for view in views:
        view_samples = [sample for sample in samples if sample.view == view]
        label_counts = Counter(sample.label for sample in view_samples)

        if len(view_samples) < 20 or len(label_counts) < 2:
            print(f"Skipping {view}: insufficient samples or missing one class.")
            continue

        summary[view] = train_one_view(
            view,
            view_samples,
            args,
            device,
        )

    with (output_root / "all_views_test_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"\nDone. Summary: {output_root / 'all_views_test_summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one MobileNetV3-Small completeness classifier per view."
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="./mobilenet_completeness_models")
    parser.add_argument(
        "--views",
        nargs="+",
        choices=["front", "back", "left", "right"],
        default=None,
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-pretrained", action="store_true")

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
