#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MobileNetV3-Small for real-vs-toy car authenticity."
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/"
            "data/car_authenticity_dataset"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/"
            "models/car_authenticity_mobilenetv3"
        ),
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision on CUDA.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            )
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_datasets(root: Path, image_size: int):
    train_tf, eval_tf = build_transforms(image_size)
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)

    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise RuntimeError("Class mappings differ across train/val/test")

    if set(train_ds.classes) != {"real", "toy_scale"}:
        raise RuntimeError(f"Expected classes ['real', 'toy_scale'], found {train_ds.classes}")

    return train_ds, val_ds, test_ds


def build_loaders(train_ds, val_ds, test_ds, batch_size, num_workers, device):
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def build_model(dropout: float, num_classes: int = 2) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def compute_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_per_class": [float(x) for x in precision],
        "recall_per_class": [float(x) for x in recall],
        "f1_per_class": [float(x) for x in f1],
        "support_per_class": [int(x) for x in support],
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_enabled):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled, desc="Val"):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, probabilities = [], [], []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, y_true, y_pred, probabilities


def save_history(history, path: Path):
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_root = args.dataset.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    print("=" * 80)
    print("MOBILENETV3-SMALL CAR AUTHENTICITY TRAINING")
    print("=" * 80)
    print(f"Dataset       : {dataset_root}")
    print(f"Output        : {output_dir}")
    print(f"Device        : {device}")
    print(f"AMP           : {amp_enabled}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Learning rate : {args.lr}")
    print(f"Weight decay  : {args.weight_decay}")
    print(f"Image size    : {args.image_size}")
    print(f"Patience      : {args.patience}")
    print(f"Seed          : {args.seed}")

    train_ds, val_ds, test_ds = build_datasets(dataset_root, args.image_size)
    print(f"\nClass mapping: {train_ds.class_to_idx}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader, val_loader, test_loader = build_loaders(
        train_ds, val_ds, test_ds,
        args.batch_size, args.num_workers, device
    )

    model = build_model(args.dropout, len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    checkpoint_path = output_dir / "best_mobilenet_v3_small.pth"
    history_path = output_dir / "training_history.csv"

    best_val_bal_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    start_time = time.perf_counter()

    real_idx = train_ds.class_to_idx["real"]
    toy_idx = train_ds.class_to_idx["toy_scale"]

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch}/{args.epochs}")
        print("=" * 80)

        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp_enabled
        )
        val_m, _, _, _ = evaluate(
            model, val_loader, criterion, device, amp_enabled, desc="Val"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Train | loss={train_m['loss']:.4f} | "
            f"acc={train_m['accuracy']:.4f} | bal_acc={train_m['balanced_accuracy']:.4f}"
        )
        print(
            f"Val   | loss={val_m['loss']:.4f} | "
            f"acc={val_m['accuracy']:.4f} | bal_acc={val_m['balanced_accuracy']:.4f}"
        )
        print(
            f"Val recall | real={val_m['recall_per_class'][real_idx]:.4f} | "
            f"toy_scale={val_m['recall_per_class'][toy_idx]:.4f}"
        )

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_m["loss"],
            "train_accuracy": train_m["accuracy"],
            "train_balanced_accuracy": train_m["balanced_accuracy"],
            "train_real_recall": train_m["recall_per_class"][real_idx],
            "train_toy_recall": train_m["recall_per_class"][toy_idx],
            "val_loss": val_m["loss"],
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_real_recall": val_m["recall_per_class"][real_idx],
            "val_toy_recall": val_m["recall_per_class"][toy_idx],
        })
        save_history(history, history_path)
        scheduler.step(val_m["balanced_accuracy"])

        if val_m["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_m["balanced_accuracy"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "classes": train_ds.classes,
                "image_size": args.image_size,
                "dropout": args.dropout,
                "val_balanced_accuracy": best_val_bal_acc,
                "val_metrics": val_m,
                "seed": args.seed,
                "architecture": "mobilenet_v3_small",
                "weights": "IMAGENET1K_V1",
            }, checkpoint_path)
            print(f"✓ Saved new best checkpoint (val_bal_acc={best_val_bal_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement: {epochs_no_improve}/{args.patience}")

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    training_seconds = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_m, y_true, y_pred, probs = evaluate(
        model, test_loader, criterion, device, amp_enabled, desc="Test"
    )

    cm = confusion_matrix(y_true, y_pred, labels=[real_idx, toy_idx])
    print(f"Test accuracy          : {test_m['accuracy']:.4f}")
    print(f"Test balanced accuracy : {test_m['balanced_accuracy']:.4f}")
    print(f"Real recall            : {test_m['recall_per_class'][real_idx]:.4f}")
    print(f"Toy-scale recall       : {test_m['recall_per_class'][toy_idx]:.4f}")
    print("\nConfusion matrix (rows=true, cols=predicted; order=[real, toy_scale])")
    print(cm)

    pred_path = output_dir / "test_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "path", "true_class", "predicted_class",
            "probability_real", "probability_toy_scale", "correct"
        ])
        writer.writeheader()
        for sample, true_idx, pred_idx, p in zip(test_ds.samples, y_true, y_pred, probs):
            path, _ = sample
            writer.writerow({
                "path": path,
                "true_class": train_ds.classes[true_idx],
                "predicted_class": train_ds.classes[pred_idx],
                "probability_real": float(p[real_idx]),
                "probability_toy_scale": float(p[toy_idx]),
                "correct": bool(true_idx == pred_idx),
            })

    cm_path = output_dir / "confusion_matrix.csv"
    with cm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", "real", "toy_scale"])
        writer.writerow(["real", int(cm[0, 0]), int(cm[0, 1])])
        writer.writerow(["toy_scale", int(cm[1, 0]), int(cm[1, 1])])

    report = classification_report(
        y_true,
        y_pred,
        target_names=train_ds.classes,
        output_dict=True,
        zero_division=0,
    )

    test_metrics = {
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_bal_acc,
        "test_loss": test_m["loss"],
        "test_accuracy": test_m["accuracy"],
        "test_balanced_accuracy": test_m["balanced_accuracy"],
        "class_to_idx": train_ds.class_to_idx,
        "confusion_matrix_order": ["real", "toy_scale"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    summary = {
        "architecture": "mobilenet_v3_small",
        "pretrained_weights": "IMAGENET1K_V1",
        "dataset": str(dataset_root),
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "test_images": len(test_ds),
        "class_to_idx": train_ds.class_to_idx,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "initial_learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "amp": amp_enabled,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_bal_acc,
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint_path),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(f"  {checkpoint_path}")
    print(f"  {history_path}")
    print(f"  {output_dir / 'training_summary.json'}")
    print(f"  {pred_path}")
    print(f"  {output_dir / 'test_metrics.json'}")
    print(f"  {cm_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val balanced accuracy: {best_val_bal_acc:.4f}")


if __name__ == "__main__":
    main()
