#!/usr/bin/env python3
"""
Evaluate a MobileNetV3-Small completeness model on synthetic robustness sets
and copy all wrong predictions into labeled folders.

Because every robustness image is synthetically truncated, the ground-truth
label is always "incomplete". A wrong prediction therefore means the model
predicted "complete".

Expected robustness structure:

back_robustness_test/
  subtle/
    manifest.csv
    candidates/
  moderate/
    manifest.csv
    candidates/
  strong/
    manifest.csv
    candidates/

Output structure:

output/
  wrong_predictions/
    subtle/
      top/
      bottom/
      left/
      right/
      top_left/
      top_right/
      bottom_left/
      bottom_right/
    moderate/
      ...
    strong/
      ...
  wrong_predictions.csv
  summary_by_severity.csv
  summary_by_crop_type.csv
  summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RobustnessDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["candidate_path"])

        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Could not read image: {path}") from exc

        return (
            self.transform(image),
            str(path),
            row["severity"],
            row["crop_type"],
        )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested, but torch.cuda.is_available() returned False."
        )

    return torch.device(requested)


def build_model(dropout: float) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[3] = nn.Linear(in_features, 2)
    return model


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def normalize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def find_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    normalized = {normalize_name(name): name for name in fieldnames}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    return None


def discover_candidate_images(candidate_root: Path) -> list[Path]:
    return sorted(
        path
        for path in candidate_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def resolve_manifest_path(
    raw_value: str,
    manifest_path: Path,
    candidate_root: Path,
) -> Path | None:
    raw = Path(raw_value).expanduser()

    possibilities = [
        raw,
        manifest_path.parent / raw,
        candidate_root / raw,
        candidate_root / raw.name,
    ]

    for path in possibilities:
        if path.exists() and path.is_file():
            return path.resolve()

    matching_names = list(candidate_root.rglob(raw.name))
    if len(matching_names) == 1:
        return matching_names[0].resolve()

    return None


def load_severity_rows(
    severity_root: Path,
    severity_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    manifest_path = severity_root / "manifest.csv"
    candidate_root = severity_root / "candidates"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    if not candidate_root.exists():
        raise FileNotFoundError(
            f"Missing candidates directory: {candidate_root}"
        )

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise RuntimeError(f"Manifest has no columns: {manifest_path}")

        fieldnames = list(reader.fieldnames)

        path_column = find_column(
            fieldnames,
            [
                "candidate_path",
                "output_path",
                "generated_path",
                "crop_path",
                "image_path",
                "path",
                "candidate",
                "output",
                "generated_image",
                "filename",
                "file_name",
            ],
        )

        crop_type_column = find_column(
            fieldnames,
            [
                "crop_type",
                "crop_direction",
                "direction",
                "crop",
            ],
        )

        if crop_type_column is None:
            raise RuntimeError(
                f"Could not find crop-type column in {manifest_path}. "
                f"Columns: {fieldnames}"
            )

        manifest_rows = list(reader)

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    if path_column is not None:
        for row in manifest_rows:
            raw_path = str(row.get(path_column, "")).strip()

            if not raw_path:
                skipped.append("Manifest row has an empty candidate path.")
                continue

            resolved = resolve_manifest_path(
                raw_value=raw_path,
                manifest_path=manifest_path,
                candidate_root=candidate_root,
            )

            if resolved is None:
                skipped.append(raw_path)
                continue

            rows.append(
                {
                    "candidate_path": str(resolved),
                    "severity": severity_name,
                    "crop_type": str(row[crop_type_column]).strip(),
                }
            )
    else:
        files = discover_candidate_images(candidate_root)

        if len(files) != len(manifest_rows):
            raise RuntimeError(
                "Could not find a candidate-path column and cannot safely "
                "pair manifest rows with candidate files because counts differ: "
                f"{len(manifest_rows)} rows vs {len(files)} files. "
                f"Columns: {fieldnames}"
            )

        for row, path in zip(manifest_rows, files):
            rows.append(
                {
                    "candidate_path": str(path.resolve()),
                    "severity": severity_name,
                    "crop_type": str(row[crop_type_column]).strip(),
                }
            )

    return rows, skipped


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    probabilities = [float(row["probability_incomplete"]) for row in rows]
    detected = sum(row["predicted_label"] == "incomplete" for row in rows)
    count = len(rows)

    return {
        "count": count,
        "detected_incomplete": detected,
        "wrong_predictions": count - detected,
        "incomplete_recall": detected / count if count else 0.0,
        "mean_probability_incomplete": (
            float(np.mean(probabilities)) if probabilities else 0.0
        ),
    }


def unique_destination(directory: Path, source_name: str) -> Path:
    destination = directory / source_name

    if not destination.exists():
        return destination

    stem = Path(source_name).stem
    suffix = Path(source_name).suffix
    counter = 1

    while True:
        candidate = directory / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main(args: argparse.Namespace) -> None:
    robustness_root = Path(args.robustness_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    wrong_root = output_root / "wrong_predictions"

    output_root.mkdir(parents=True, exist_ok=True)

    if args.overwrite and wrong_root.exists():
        shutil.rmtree(wrong_root)

    wrong_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    image_size = int(checkpoint.get("image_size", args.image_size))
    dropout = float(checkpoint.get("dropout", args.dropout))

    model = build_model(dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold_path = checkpoint_path.parent / "selected_threshold.json"

        if not threshold_path.exists():
            raise FileNotFoundError(
                "No --threshold provided and selected_threshold.json was not "
                f"found beside the checkpoint: {threshold_path}"
            )

        with threshold_path.open("r", encoding="utf-8") as file:
            threshold = float(json.load(file)["selected_threshold"])

    all_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for severity in args.severities:
        rows, skipped_rows = load_severity_rows(
            robustness_root / severity,
            severity,
        )
        all_rows.extend(rows)
        skipped.extend(f"{severity}: {item}" for item in skipped_rows)

    if not all_rows:
        raise RuntimeError("No robustness images were discovered.")

    loader = DataLoader(
        RobustnessDataset(all_rows, build_transform(image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    use_amp = args.amp and device.type == "cuda"
    prediction_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for images, paths, severities, crop_types in tqdm(
            loader,
            desc="Evaluating",
            unit="batch",
        ):
            images = images.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)[:, 1]

            for path, severity, crop_type, probability in zip(
                paths,
                severities,
                crop_types,
                probabilities.cpu().tolist(),
            ):
                predicted_incomplete = probability >= threshold

                row = {
                    "source_path": path,
                    "severity": severity,
                    "crop_type": crop_type,
                    "true_label": "incomplete",
                    "predicted_label": (
                        "incomplete" if predicted_incomplete else "complete"
                    ),
                    "probability_incomplete": float(probability),
                    "threshold": threshold,
                    "correct": predicted_incomplete,
                    "copied_path": "",
                }

                if not predicted_incomplete:
                    destination_dir = (
                        wrong_root
                        / normalize_name(severity)
                        / normalize_name(crop_type)
                    )
                    destination_dir.mkdir(parents=True, exist_ok=True)

                    source = Path(path)
                    destination = unique_destination(
                        destination_dir,
                        source.name,
                    )

                    if args.copy_mode == "copy":
                        shutil.copy2(source, destination)
                    elif args.copy_mode == "hardlink":
                        try:
                            destination.hardlink_to(source)
                        except OSError:
                            shutil.copy2(source, destination)
                    elif args.copy_mode == "symlink":
                        destination.symlink_to(source)
                    else:
                        raise ValueError(
                            f"Unsupported copy mode: {args.copy_mode}"
                        )

                    row["copied_path"] = str(destination)
                    wrong_rows.append(row.copy())

                prediction_rows.append(row)

    severity_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    crop_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in prediction_rows:
        severity_groups[row["severity"]].append(row)
        crop_groups[(row["severity"], row["crop_type"])].append(row)

    severity_summary = [
        {
            "severity": severity,
            **summarize(severity_groups[severity]),
        }
        for severity in args.severities
    ]

    crop_summary = [
        {
            "severity": severity,
            "crop_type": crop_type,
            **summarize(rows),
        }
        for (severity, crop_type), rows in sorted(crop_groups.items())
    ]

    write_csv(output_root / "all_predictions.csv", prediction_rows)
    write_csv(output_root / "wrong_predictions.csv", wrong_rows)
    write_csv(output_root / "summary_by_severity.csv", severity_summary)
    write_csv(output_root / "summary_by_crop_type.csv", crop_summary)

    summary = {
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "total_images": len(prediction_rows),
        "wrong_predictions": len(wrong_rows),
        "overall_incomplete_recall": (
            1.0 - len(wrong_rows) / len(prediction_rows)
            if prediction_rows
            else 0.0
        ),
        "wrong_predictions_root": str(wrong_root),
        "by_severity": severity_summary,
        "by_crop_type": crop_summary,
        "skipped_count": len(skipped),
    }

    with (output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with (output_root / "skipped_images.txt").open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(skipped))

    print("\n" + "=" * 86)
    print("WRONG-PREDICTION EXPORT COMPLETE")
    print("=" * 86)
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Threshold:        {threshold:.4f}")
    print(f"Total images:     {len(prediction_rows)}")
    print(f"Wrong predictions:{len(wrong_rows)}")
    print(f"Output folders:   {wrong_root}")

    print("\nBY SEVERITY")
    for row in severity_summary:
        print(
            f"{row['severity']:9} "
            f"n={row['count']:4d} | "
            f"wrong={row['wrong_predictions']:3d} | "
            f"recall={row['incomplete_recall']:.4f}"
        )

    print("\nCreated:")
    print(f"  {output_root / 'all_predictions.csv'}")
    print(f"  {output_root / 'wrong_predictions.csv'}")
    print(f"  {output_root / 'summary_by_severity.csv'}")
    print(f"  {output_root / 'summary_by_crop_type.csv'}")
    print(f"  {output_root / 'summary.json'}")
    print(f"  {wrong_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate robustness crops and export all missed incomplete "
            "predictions into labeled folders."
        )
    )

    parser.add_argument("--robustness-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output",
        default="./robustness_wrong_predictions",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        default=["subtle", "moderate", "strong"],
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override the incomplete threshold. Otherwise the script reads "
            "selected_threshold.json beside the checkpoint."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the previous wrong_predictions folder before exporting.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
