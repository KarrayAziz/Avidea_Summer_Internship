#!/usr/bin/env python3
"""
Train one MobileNetV3-Small completeness classifier per vehicle view.

Improvements over the first version:
- one independent model per view;
- stratified train/validation/test split;
- weighted oversampling of the minority class;
- class-weighted cross-entropy;
- safer augmentations for completeness classification;
- mixed-precision GPU training;
- early stopping and learning-rate reduction;
- checkpoint selection prioritizing incomplete-image recall;
- automatic probability-threshold tuning on validation data;
- final evaluation on untouched test data using the tuned threshold.

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


def infer_metadata(
    image_path: Path,
    dataset_root: Path,
) -> tuple[Optional[str], Optional[str]]:
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
    label_name = next(iter(found_labels)) if len(found_labels) == 1 else None

    return view, label_name


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
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() returned False."
        )

    return torch.device(requested)


def stratified_split(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("Validation and test ratios must be positive.")

    if val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must sum to less than 1.")

    labels = [sample.label for sample in samples]

    train_samples, temporary_samples = train_test_split(
        samples,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=labels,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    temporary_labels = [sample.label for sample in temporary_samples]

    val_samples, test_samples = train_test_split(
        temporary_samples,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temporary_labels,
    )

    return list(train_samples), list(val_samples), list(test_samples)


def build_transforms(image_size: int):
    """
    Avoid rotation and perspective transforms because completeness depends on
    which parts remain visible inside the original frame.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
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


def build_model(pretrained: bool, dropout: float) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[3] = nn.Linear(in_features, 2)

    return model


def calculate_class_weights(
    samples: list[Sample],
    device: torch.device,
) -> torch.Tensor:
    counts = Counter(sample.label for sample in samples)
    total = len(samples)

    weights = []

    for class_index in (0, 1):
        class_count = counts.get(class_index, 0)

        if class_count == 0:
            raise RuntimeError(
                f"Training split contains no samples for class {class_index}."
            )

        weights.append(total / (2 * class_count))

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def create_weighted_sampler(
    samples: list[Sample],
) -> WeightedRandomSampler:
    """
    Makes complete and incomplete examples contribute approximately equally
    during each epoch.
    """
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


def create_standard_loader(
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


def create_weighted_training_loader(
    samples: list[Sample],
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = CompletenessDataset(samples, transform=transform)
    sampler = create_weighted_sampler(samples)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def metrics_from_predictions(
    targets: list[int],
    predictions: list[int],
) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    return {
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": balanced_accuracy_score(targets, predictions),
        "complete_precision": float(precision[0]),
        "complete_recall": float(recall[0]),
        "complete_f1": float(f1[0]),
        "complete_support": int(support[0]),
        "incomplete_precision": float(precision[1]),
        "incomplete_recall": float(recall[1]),
        "incomplete_f1": float(f1[1]),
        "incomplete_support": int(support[1]),
        "confusion_matrix": {
            "true_complete_pred_complete": int(matrix[0, 0]),
            "true_complete_pred_incomplete": int(matrix[0, 1]),
            "true_incomplete_pred_complete": int(matrix[1, 0]),
            "true_incomplete_pred_incomplete": int(matrix[1, 1]),
        },
    }


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.train()

    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(
        loader,
        desc="train",
        leave=False,
        unit="batch",
    )

    for images, targets, _ in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = metrics_from_predictions(
        all_targets,
        all_predictions,
    )

    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def collect_probabilities(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    description: str,
) -> tuple[float, list[int], list[float], list[str]]:
    model.eval()

    total_loss = 0.0
    all_targets: list[int] = []
    all_incomplete_probabilities: list[float] = []
    all_paths: list[str] = []

    with torch.no_grad():
        progress = tqdm(
            loader,
            desc=description,
            leave=False,
            unit="batch",
        )

        for images, targets, paths in progress:
            images = images.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, targets_device)
                probabilities = torch.softmax(logits, dim=1)

            total_loss += loss.item() * images.size(0)

            all_targets.extend(targets.tolist())
            all_incomplete_probabilities.extend(
                probabilities[:, 1].cpu().tolist()
            )
            all_paths.extend(paths)

    average_loss = total_loss / len(loader.dataset)

    return (
        average_loss,
        all_targets,
        all_incomplete_probabilities,
        all_paths,
    )


def tune_probability_threshold(
    targets: list[int],
    incomplete_probabilities: list[float],
    thresholds: list[float],
    minimum_incomplete_recall: float,
) -> tuple[float, dict, list[dict]]:
    """
    Selection policy:
    1. Prefer thresholds meeting minimum incomplete recall.
    2. Among them, maximize complete recall.
    3. Break ties using balanced accuracy.
    4. If none meet the requirement, maximize incomplete recall first.
    """
    sweep_rows: list[dict] = []

    for threshold in thresholds:
        predictions = [
            1 if probability >= threshold else 0
            for probability in incomplete_probabilities
        ]

        metrics = metrics_from_predictions(targets, predictions)

        sweep_rows.append(
            {
                "threshold": threshold,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                },
                **metrics["confusion_matrix"],
            }
        )

    valid_rows = [
        row
        for row in sweep_rows
        if row["incomplete_recall"] >= minimum_incomplete_recall
    ]

    if valid_rows:
        best_row = max(
            valid_rows,
            key=lambda row: (
                row["complete_recall"],
                row["balanced_accuracy"],
                row["incomplete_f1"],
                row["threshold"],
            ),
        )
    else:
        best_row = max(
            sweep_rows,
            key=lambda row: (
                row["incomplete_recall"],
                row["balanced_accuracy"],
                row["complete_recall"],
                row["threshold"],
            ),
        )

    best_threshold = float(best_row["threshold"])

    best_predictions = [
        1 if probability >= best_threshold else 0
        for probability in incomplete_probabilities
    ]

    best_metrics = metrics_from_predictions(
        targets,
        best_predictions,
    )

    return best_threshold, best_metrics, sweep_rows


def checkpoint_selection_score(
    metrics: dict,
    minimum_incomplete_recall: float,
) -> float:
    """
    Prioritize incomplete-image recall. Models below the required recall receive
    a penalty, while models meeting it are ranked by balanced accuracy.
    """
    incomplete_recall = metrics["incomplete_recall"]
    complete_recall = metrics["complete_recall"]

    if incomplete_recall < minimum_incomplete_recall:
        return incomplete_recall - 1.0

    return (
        0.70 * incomplete_recall
        + 0.20 * complete_recall
        + 0.10 * metrics["balanced_accuracy"]
    )


def save_split_manifest(
    output_path: Path,
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "path",
                "view",
                "label_name",
                "label",
            ],
        )
        writer.writeheader()

        for split_name, split_samples in (
            ("train", train_samples),
            ("val", val_samples),
            ("test", test_samples),
        ):
            for sample in split_samples:
                writer.writerow(
                    {
                        "split": split_name,
                        **asdict(sample),
                    }
                )


def save_csv_rows(
    rows: list[dict],
    output_path: Path,
) -> None:
    if not rows:
        return

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    probability_threshold: float,
) -> tuple[dict, list[dict]]:
    start = time.perf_counter()

    (
        loss,
        targets,
        incomplete_probabilities,
        paths,
    ) = collect_probabilities(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        description="test",
    )

    elapsed = time.perf_counter() - start

    predictions = [
        1 if probability >= probability_threshold else 0
        for probability in incomplete_probabilities
    ]

    metrics = metrics_from_predictions(
        targets,
        predictions,
    )

    metrics.update(
        {
            "count": len(targets),
            "loss": loss,
            "probability_threshold": probability_threshold,
            "classification_report": classification_report(
                targets,
                predictions,
                labels=[0, 1],
                target_names=["complete", "incomplete"],
                output_dict=True,
                zero_division=0,
            ),
            "timing": {
                "total_seconds": elapsed,
                "average_seconds_per_image": (
                    elapsed / len(targets)
                    if targets
                    else 0.0
                ),
            },
        }
    )

    prediction_rows = []

    for path, target, prediction, probability in zip(
        paths,
        targets,
        predictions,
        incomplete_probabilities,
    ):
        prediction_rows.append(
            {
                "path": path,
                "true_label": INDEX_TO_CLASS[target],
                "predicted_label": INDEX_TO_CLASS[prediction],
                "probability_incomplete": probability,
                "probability_threshold": probability_threshold,
                "correct": target == prediction,
            }
        )

    return metrics, prediction_rows


def train_one_view(
    view: str,
    samples: list[Sample],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    output_dir = Path(args.output) / view
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples, test_samples = stratified_split(
        samples=samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    save_split_manifest(
        output_path=output_dir / "split_manifest.csv",
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
    )

    train_transform, eval_transform = build_transforms(
        args.image_size
    )

    pin_memory = device.type == "cuda"

    train_loader = create_weighted_training_loader(
        samples=train_samples,
        transform=train_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = create_standard_loader(
        samples=val_samples,
        transform=eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )

    test_loader = create_standard_loader(
        samples=test_samples,
        transform=eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = build_model(
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    class_weights = calculate_class_weights(
        train_samples,
        device,
    )

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
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []

    checkpoint_path = (
        output_dir / "best_mobilenet_v3_small.pth"
    )

    print("\n" + "=" * 82)
    print(f"TRAINING VIEW: {view.upper()}")
    print("=" * 82)
    print(
        f"train={len(train_samples)} | "
        f"val={len(val_samples)} | "
        f"test={len(test_samples)}"
    )
    print(
        "Training labels: "
        f"complete={sum(sample.label == 0 for sample in train_samples)}, "
        f"incomplete={sum(sample.label == 1 for sample in train_samples)}"
    )
    print(
        f"Class weights: "
        f"{class_weights.detach().cpu().tolist()}"
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_training_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        (
            val_loss,
            val_targets,
            val_probabilities,
            _,
        ) = collect_probabilities(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            description="validation",
        )

        val_predictions_default = [
            1 if probability >= 0.5 else 0
            for probability in val_probabilities
        ]

        val_metrics = metrics_from_predictions(
            val_targets,
            val_predictions_default,
        )
        val_metrics["loss"] = val_loss

        current_score = checkpoint_selection_score(
            metrics=val_metrics,
            minimum_incomplete_recall=(
                args.minimum_validation_incomplete_recall
            ),
        )

        scheduler.step(current_score)

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "checkpoint_score": current_score,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
                if key != "confusion_matrix"
            },
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
                if key != "confusion_matrix"
            },
        }

        history.append(row)
        save_csv_rows(
            history,
            output_dir / "training_history.csv",
        )

        print(
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_bal_acc={train_metrics['balanced_accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_incomplete_recall={val_metrics['incomplete_recall']:.4f} "
            f"score={current_score:.4f}"
        )

        if current_score > best_score + args.min_delta:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "view": view,
                    "epoch": epoch,
                    "architecture": "mobilenet_v3_small",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_checkpoint_score": best_score,
                    "class_to_index": CLASS_TO_INDEX,
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "pretrained": not args.no_pretrained,
                    "class_weights": (
                        class_weights.detach().cpu().tolist()
                    ),
                    "configuration": vars(args),
                },
                checkpoint_path,
            )

            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.early_stopping_patience
        ):
            print(
                "Early stopping after "
                f"{args.early_stopping_patience} "
                "epochs without improvement."
            )
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        validation_loss,
        validation_targets,
        validation_probabilities,
        _,
    ) = collect_probabilities(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        description="threshold tuning",
    )

    threshold_values = [
        round(value, 4)
        for value in np.arange(
            args.threshold_min,
            args.threshold_max + args.threshold_step / 2,
            args.threshold_step,
        )
    ]

    (
        best_threshold,
        best_validation_threshold_metrics,
        threshold_sweep_rows,
    ) = tune_probability_threshold(
        targets=validation_targets,
        incomplete_probabilities=validation_probabilities,
        thresholds=threshold_values,
        minimum_incomplete_recall=(
            args.minimum_validation_incomplete_recall
        ),
    )

    save_csv_rows(
        threshold_sweep_rows,
        output_dir / "validation_threshold_sweep.csv",
    )

    with (
        output_dir / "selected_threshold.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "view": view,
                "selected_threshold": best_threshold,
                "validation_loss": validation_loss,
                "selection_rule": (
                    "Maximize complete recall while requiring "
                    "minimum validation incomplete recall; "
                    "break ties using balanced accuracy."
                ),
                "minimum_validation_incomplete_recall": (
                    args.minimum_validation_incomplete_recall
                ),
                "validation_metrics": (
                    best_validation_threshold_metrics
                ),
            },
            file,
            indent=2,
        )

    test_metrics, test_prediction_rows = evaluate_test_set(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        probability_threshold=best_threshold,
    )

    test_metrics.update(
        {
            "view": view,
            "best_epoch": best_epoch,
            "best_checkpoint_score": best_score,
            "selected_probability_threshold": best_threshold,
            "dataset_counts": {
                "all": len(samples),
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
        }
    )

    with (
        output_dir / "test_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            test_metrics,
            file,
            indent=2,
        )

    save_csv_rows(
        test_prediction_rows,
        output_dir / "test_predictions.csv",
    )

    print(
        f"\n{view.upper()} TEST | "
        f"threshold={best_threshold:.4f} | "
        f"accuracy={test_metrics['accuracy']:.4f} | "
        f"balanced_accuracy="
        f"{test_metrics['balanced_accuracy']:.4f} | "
        f"incomplete_recall="
        f"{test_metrics['incomplete_recall']:.4f}"
    )

    print(
        json.dumps(
            test_metrics["confusion_matrix"],
            indent=2,
        )
    )

    return test_metrics


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    dataset_root = Path(
        args.dataset
    ).expanduser().resolve()

    output_root = Path(
        args.output
    ).expanduser().resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {dataset_root}"
        )

    samples, skipped = discover_samples(dataset_root)

    if not samples:
        raise RuntimeError(
            "No labeled images were discovered."
        )

    with (
        output_root / "skipped_images.txt"
    ).open("w", encoding="utf-8") as file:
        file.write("\n".join(skipped))

    device = resolve_device(args.device)

    print(f"Dataset: {dataset_root}")
    print(f"Output:  {output_root}")
    print(f"Device:  {device}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    print(f"Images:  {len(samples)}")
    print(f"Skipped: {len(skipped)}")

    selected_views = (
        args.views
        or ["front", "back", "left", "right"]
    )

    summary = {}

    for view in selected_views:
        view_samples = [
            sample
            for sample in samples
            if sample.view == view
        ]

        label_counts = Counter(
            sample.label
            for sample in view_samples
        )

        if len(view_samples) < 20:
            print(
                f"Skipping {view}: "
                f"only {len(view_samples)} images."
            )
            continue

        if len(label_counts) < 2:
            print(
                f"Skipping {view}: "
                "both classes are required."
            )
            continue

        summary[view] = train_one_view(
            view=view,
            samples=view_samples,
            args=args,
            device=device,
        )

    with (
        output_root / "all_views_test_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\nTraining complete. Summary: "
        f"{output_root / 'all_views_test_summary.json'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one improved MobileNetV3-Small "
            "completeness classifier per vehicle view."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Root of the labeled dataset.",
    )

    parser.add_argument(
        "--output",
        default="./mobilenet_completeness_models_v2",
        help="Output directory.",
    )

    parser.add_argument(
        "--views",
        nargs="+",
        choices=[
            "front",
            "back",
            "left",
            "right",
        ],
        default=None,
        help="Train selected views only.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--minimum-validation-incomplete-recall",
        type=float,
        default=0.90,
        help=(
            "Required validation recall for incomplete images "
            "during checkpoint and threshold selection."
        ),
    )

    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--device",
        default="auto",
        help='"auto", "cuda", or "cpu".',
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
