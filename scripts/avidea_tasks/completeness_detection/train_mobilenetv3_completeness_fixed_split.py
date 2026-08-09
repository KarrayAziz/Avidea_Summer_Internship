#!/usr/bin/env python3
"""
Train one MobileNetV3-Small completeness classifier per vehicle view using
an existing fixed train/validation/test folder split.

Expected structure:

dataset/
  train/
    front/{complete,incomplete}/
    back/{complete,incomplete}/
    left/{complete,incomplete}/
    right/{complete,incomplete}/
  val/
    ...
  test/
    ...

Rules:
- train may contain real and approved synthetic images;
- val and test remain real-only;
- no random resplitting;
- ordinary CrossEntropyLoss;
- no class weights;
- no WeightedRandomSampler;
- checkpoint selected by validation balanced accuracy;
- incomplete probability threshold tuned on validation;
- final evaluation performed on untouched test data.

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
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from tqdm import tqdm

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIEW_ALIASES = {"front": "front", "back": "back", "rear": "back", "left": "left", "right": "right"}
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


def infer_metadata(image_path: Path, split_root: Path) -> tuple[Optional[str], Optional[str]]:
    parts = image_path.relative_to(split_root).parts[:-1]
    tokens = [normalize_token(part) for part in parts]
    views = {VIEW_ALIASES[token] for token in tokens if token in VIEW_ALIASES}
    labels = {LABEL_ALIASES[token] for token in tokens if token in LABEL_ALIASES}
    view = next(iter(views)) if len(views) == 1 else None
    label_name = next(iter(labels)) if len(labels) == 1 else None
    return view, label_name


def discover_samples(split_root: Path) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    skipped: list[str] = []

    for path in sorted(split_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        view, label_name = infer_metadata(path, split_root)
        if view is None or label_name is None:
            skipped.append(str(path))
            continue
        samples.append(Sample(str(path), view, label_name, CLASS_TO_INDEX[label_name]))

    return samples, skipped


def discover_fixed_splits(dataset_root: Path):
    split_samples = {}
    split_skipped = {}
    for split_name in ("train", "val", "test"):
        split_root = dataset_root / split_name
        if not split_root.exists():
            raise FileNotFoundError(f"Missing required split folder: {split_root}")
        samples, skipped = discover_samples(split_root)
        if not samples:
            raise RuntimeError(f"No labeled images discovered in: {split_root}")
        split_samples[split_name] = samples
        split_skipped[split_name] = skipped
    return split_samples, split_skipped


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


def build_transforms(image_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def build_model(pretrained: bool, dropout: float) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[3] = nn.Linear(in_features, 2)
    return model


def create_loader(samples, transform, batch_size, num_workers, shuffle, pin_memory):
    return DataLoader(
        CompletenessDataset(samples, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def metrics_from_predictions(targets: list[int], predictions: list[int]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
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


def run_training_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    targets_all, predictions_all = [], []

    for images, targets, _ in tqdm(loader, desc="train", leave=False, unit="batch"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        targets_all.extend(targets.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())

    metrics = metrics_from_predictions(targets_all, predictions_all)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def collect_probabilities(model, loader, criterion, device, use_amp, description):
    model.eval()
    total_loss = 0.0
    targets_all, probs_all, paths_all = [], [], []

    with torch.no_grad():
        for images, targets, paths in tqdm(loader, desc=description, leave=False, unit="batch"):
            images = images.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets_device)
                probabilities = torch.softmax(logits, dim=1)

            total_loss += loss.item() * images.size(0)
            targets_all.extend(targets.tolist())
            probs_all.extend(probabilities[:, 1].cpu().tolist())
            paths_all.extend(paths)

    return total_loss / len(loader.dataset), targets_all, probs_all, paths_all


def tune_probability_threshold(targets, incomplete_probabilities, thresholds, minimum_incomplete_recall):
    rows = []
    for threshold in thresholds:
        predictions = [1 if probability >= threshold else 0 for probability in incomplete_probabilities]
        metrics = metrics_from_predictions(targets, predictions)
        rows.append({
            "threshold": threshold,
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            **metrics["confusion_matrix"],
        })

    valid = [row for row in rows if row["incomplete_recall"] >= minimum_incomplete_recall]
    pool = valid if valid else rows
    best = max(
        pool,
        key=lambda row: (
            row["balanced_accuracy"],
            row["incomplete_f1"],
            row["complete_recall"],
            row["threshold"],
        ),
    )

    best_threshold = float(best["threshold"])
    best_predictions = [1 if p >= best_threshold else 0 for p in incomplete_probabilities]
    return best_threshold, metrics_from_predictions(targets, best_predictions), rows


def save_csv_rows(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_split_manifest(output_path, train_samples, val_samples, test_samples):
    rows = []
    for split_name, samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        for sample in samples:
            rows.append({"split": split_name, **asdict(sample)})
    save_csv_rows(rows, output_path)


def evaluate_test_set(model, loader, criterion, device, use_amp, probability_threshold):
    start = time.perf_counter()
    loss, targets, probs, paths = collect_probabilities(
        model, loader, criterion, device, use_amp, "test"
    )
    elapsed = time.perf_counter() - start
    predictions = [1 if p >= probability_threshold else 0 for p in probs]
    metrics = metrics_from_predictions(targets, predictions)
    metrics.update({
        "count": len(targets),
        "loss": loss,
        "probability_threshold": probability_threshold,
        "classification_report": classification_report(
            targets, predictions, labels=[0, 1], target_names=["complete", "incomplete"],
            output_dict=True, zero_division=0
        ),
        "timing": {
            "total_seconds": elapsed,
            "average_seconds_per_image": elapsed / len(targets) if targets else 0.0,
        },
    })

    rows = []
    for path, target, prediction, probability in zip(paths, targets, predictions, probs):
        rows.append({
            "path": path,
            "true_label": INDEX_TO_CLASS[target],
            "predicted_label": INDEX_TO_CLASS[prediction],
            "probability_incomplete": probability,
            "probability_threshold": probability_threshold,
            "correct": target == prediction,
        })
    return metrics, rows


def train_one_view(view, train_samples, val_samples, test_samples, args, device):
    output_dir = Path(args.output) / view
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_manifest(output_dir / "split_manifest.csv", train_samples, val_samples, test_samples)

    train_transform, eval_transform = build_transforms(args.image_size)
    pin_memory = device.type == "cuda"
    train_loader = create_loader(train_samples, train_transform, args.batch_size, args.num_workers, True, pin_memory)
    val_loader = create_loader(val_samples, eval_transform, args.batch_size, args.num_workers, False, pin_memory)
    test_loader = create_loader(test_samples, eval_transform, args.batch_size, args.num_workers, False, pin_memory)

    model = build_model(not args.no_pretrained, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience, min_lr=args.min_learning_rate
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    checkpoint_path = output_dir / "best_mobilenet_v3_small.pth"

    print("\n" + "=" * 82)
    print(f"TRAINING VIEW: {view.upper()}")
    print("=" * 82)
    print(f"train={len(train_samples)} | val={len(val_samples)} | test={len(test_samples)}")
    print(
        f"train labels: complete={sum(s.label == 0 for s in train_samples)}, "
        f"incomplete={sum(s.label == 1 for s in train_samples)}"
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_training_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_targets, val_probs, _ = collect_probabilities(
            model, val_loader, criterion, device, use_amp, "validation"
        )
        val_predictions = [1 if p >= 0.5 else 0 for p in val_probs]
        val_metrics = metrics_from_predictions(val_targets, val_predictions)
        val_metrics["loss"] = val_loss
        current_score = val_metrics["balanced_accuracy"]
        scheduler.step(current_score)

        history.append({
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "checkpoint_score": current_score,
            **{f"train_{k}": v for k, v in train_metrics.items() if k != "confusion_matrix"},
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        })
        save_csv_rows(history, output_dir / "training_history.csv")

        print(
            f"train_loss={train_metrics['loss']:.4f} train_bal_acc={train_metrics['balanced_accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_incomplete_recall={val_metrics['incomplete_recall']:.4f}"
        )

        if current_score > best_score + args.min_delta:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "view": view,
                "epoch": epoch,
                "architecture": "mobilenet_v3_small",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_balanced_accuracy": best_score,
                "class_to_index": CLASS_TO_INDEX,
                "image_size": args.image_size,
                "dropout": args.dropout,
                "pretrained": not args.no_pretrained,
                "configuration": vars(args),
            }, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping after {args.early_stopping_patience} epochs without improvement.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    validation_loss, validation_targets, validation_probs, _ = collect_probabilities(
        model, val_loader, criterion, device, use_amp, "threshold tuning"
    )
    thresholds = [
        round(value, 4)
        for value in np.arange(args.threshold_min, args.threshold_max + args.threshold_step / 2, args.threshold_step)
    ]
    best_threshold, validation_threshold_metrics, sweep_rows = tune_probability_threshold(
        validation_targets, validation_probs, thresholds, args.minimum_validation_incomplete_recall
    )
    save_csv_rows(sweep_rows, output_dir / "validation_threshold_sweep.csv")

    with (output_dir / "selected_threshold.json").open("w", encoding="utf-8") as file:
        json.dump({
            "view": view,
            "selected_threshold": best_threshold,
            "validation_loss": validation_loss,
            "selection_rule": (
                "Maximize validation balanced accuracy among thresholds meeting the minimum "
                "incomplete recall; break ties using incomplete F1 and complete recall."
            ),
            "minimum_validation_incomplete_recall": args.minimum_validation_incomplete_recall,
            "validation_metrics": validation_threshold_metrics,
        }, file, indent=2)

    test_metrics, test_rows = evaluate_test_set(
        model, test_loader, criterion, device, use_amp, best_threshold
    )
    test_metrics.update({
        "view": view,
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_score,
        "selected_probability_threshold": best_threshold,
        "dataset_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
    })

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)
    save_csv_rows(test_rows, output_dir / "test_predictions.csv")

    print(
        f"\n{view.upper()} TEST | threshold={best_threshold:.4f} | "
        f"accuracy={test_metrics['accuracy']:.4f} | "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} | "
        f"incomplete_recall={test_metrics['incomplete_recall']:.4f}"
    )
    print(json.dumps(test_metrics["confusion_matrix"], indent=2))
    return test_metrics


def main(args):
    set_seed(args.seed)
    dataset_root = Path(args.dataset).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_root}")

    split_samples, split_skipped = discover_fixed_splits(dataset_root)
    device = resolve_device(args.device)

    print(f"Dataset: {dataset_root}")
    print(f"Output:  {output_root}")
    print(f"Device:  {device}")
    print(f"CUDA:    {torch.cuda.is_available()}")

    for split_name in ("train", "val", "test"):
        print(
            f"{split_name.upper():5} images={len(split_samples[split_name])} | "
            f"skipped={len(split_skipped[split_name])}"
        )
        with (output_root / f"skipped_{split_name}_images.txt").open("w", encoding="utf-8") as file:
            file.write("\n".join(split_skipped[split_name]))

    selected_views = args.views or ["front", "back", "left", "right"]
    summary = {}

    for view in selected_views:
        train_view = [s for s in split_samples["train"] if s.view == view]
        val_view = [s for s in split_samples["val"] if s.view == view]
        test_view = [s for s in split_samples["test"] if s.view == view]

        for split_name, samples in (("train", train_view), ("val", val_view), ("test", test_view)):
            counts = Counter(s.label for s in samples)
            if len(samples) < 2 or len(counts) < 2:
                raise RuntimeError(
                    f"{view}/{split_name} must contain both classes. "
                    f"Found {len(samples)} images with labels {dict(counts)}."
                )

        summary[view] = train_one_view(view, train_view, val_view, test_view, args, device)

    with (output_root / "all_views_test_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nTraining complete. Summary: {output_root / 'all_views_test_summary.json'}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3-Small per view using fixed train/val/test folders."
    )
    parser.add_argument("--dataset", required=True, help="Root containing train/, val/, and test/ folders.")
    parser.add_argument("--output", default="./mobilenet_completeness_fixed_split")
    parser.add_argument("--views", nargs="+", choices=["front", "back", "left", "right"], default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--minimum-validation-incomplete-recall", type=float, default=0.80)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
