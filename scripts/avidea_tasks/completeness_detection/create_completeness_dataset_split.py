#!/usr/bin/env python3
"""
Create stratified train/validation/test folders for vehicle completeness data.

The split is performed independently for every (view, label) group:
    front/complete
    front/incomplete
    back/complete
    back/incomplete
    left/complete
    left/incomplete
    right/complete
    right/incomplete

Output layout:

output/
  train/
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
  val/
    ...
  test/
    ...
  split_manifest.csv
  split_summary.json
  skipped_images.txt

Important workflow:
1. Run this script once on the original dataset.
2. Use only output/train as input to the synthetic crop generator.
3. Keep output/val and output/test untouched and real-only.
4. Train using the resulting split manifest or folders.

Supported input layouts:

dataset/
  front/
    complete/
    incomplete/
  back/
    complete/
    incomplete/
  ...

or:

dataset/
  complete/
    front/
    back/
    ...
  incomplete/
    front/
    back/
    ...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


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
    source_path: str
    view: str
    label: str


@dataclass
class SplitRecord:
    split: str
    source_path: str
    destination_path: str
    view: str
    label: str
    operation: str


def normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def infer_metadata(
    image_path: Path,
    dataset_root: Path,
) -> tuple[Optional[str], Optional[str]]:
    parts = image_path.relative_to(dataset_root).parts[:-1]
    tokens = [normalize_token(part) for part in parts]

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


def discover_samples(
    dataset_root: Path,
) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    skipped: list[str] = []

    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        view, label = infer_metadata(path, dataset_root)

        if view is None or label is None:
            skipped.append(str(path))
            continue

        samples.append(
            Sample(
                source_path=str(path),
                view=view,
                label=label,
            )
        )

    return samples, skipped


def validate_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be greater than zero.")

    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(
            "Train, validation, and test ratios must sum to exactly 1.0."
        )


def split_count(
    count: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int, int]:
    """
    Allocate counts using rounded targets while ensuring:
    - all images are assigned;
    - each split gets at least one image when the group has >= 3 images;
    """
    train_count = int(round(count * train_ratio))
    val_count = int(round(count * val_ratio))
    test_count = count - train_count - val_count

    if count >= 3:
        counts = [train_count, val_count, test_count]

        for index in range(3):
            if counts[index] == 0:
                donor = max(range(3), key=lambda i: counts[i])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1

        train_count, val_count, test_count = counts

    if test_count < 0:
        overflow = -test_count
        take_from_train = min(overflow, max(0, train_count - 1))
        train_count -= take_from_train
        overflow -= take_from_train

        if overflow > 0:
            take_from_val = min(overflow, max(0, val_count - 1))
            val_count -= take_from_val
            overflow -= take_from_val

        test_count = count - train_count - val_count

    return train_count, val_count, test_count


def build_stratified_split(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)

    for sample in samples:
        groups[(sample.view, sample.label)].append(sample)

    split_map = {
        "train": [],
        "val": [],
        "test": [],
    }

    for group_key in sorted(groups):
        group_samples = groups[group_key]

        group_seed_material = (
            f"{seed}:{group_key[0]}:{group_key[1]}".encode("utf-8")
        )
        group_seed = int(
            hashlib.sha256(group_seed_material).hexdigest()[:16],
            16,
        )

        rng = random.Random(group_seed)
        rng.shuffle(group_samples)

        train_count, val_count, test_count = split_count(
            count=len(group_samples),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        train_end = train_count
        val_end = train_count + val_count

        split_map["train"].extend(group_samples[:train_end])
        split_map["val"].extend(group_samples[train_end:val_end])
        split_map["test"].extend(group_samples[val_end:val_end + test_count])

    for split_name, split_samples in split_map.items():
        rng = random.Random(seed + {"train": 1, "val": 2, "test": 3}[split_name])
        rng.shuffle(split_samples)

    return split_map


def unique_destination_name(
    source_path: Path,
    used_names: set[str],
) -> str:
    candidate = source_path.name

    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    digest = hashlib.sha1(
        str(source_path).encode("utf-8")
    ).hexdigest()[:10]

    candidate = f"{source_path.stem}__{digest}{source_path.suffix.lower()}"
    used_names.add(candidate)
    return candidate


def transfer_file(
    source: Path,
    destination: Path,
    mode: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(source, destination)
        return

    if mode == "hardlink":
        os.link(source, destination)
        return

    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return

    raise ValueError(f"Unsupported mode: {mode}")


def create_split_folders(
    split_map: dict[str, list[Sample]],
    output_root: Path,
    mode: str,
    overwrite: bool,
) -> list[SplitRecord]:
    records: list[SplitRecord] = []
    used_names_by_folder: dict[str, set[str]] = defaultdict(set)

    for split_name, samples in split_map.items():
        for sample in samples:
            source_path = Path(sample.source_path)

            destination_folder = (
                output_root
                / split_name
                / sample.view
                / sample.label
            )

            folder_key = str(destination_folder)
            filename = unique_destination_name(
                source_path,
                used_names_by_folder[folder_key],
            )
            destination_path = destination_folder / filename

            if destination_path.exists() or destination_path.is_symlink():
                if not overwrite:
                    raise FileExistsError(
                        f"Destination already exists: {destination_path}\n"
                        "Use --overwrite or choose a new output directory."
                    )

                if destination_path.is_dir():
                    shutil.rmtree(destination_path)
                else:
                    destination_path.unlink()

            transfer_file(
                source=source_path,
                destination=destination_path,
                mode=mode,
            )

            records.append(
                SplitRecord(
                    split=split_name,
                    source_path=str(source_path),
                    destination_path=str(destination_path),
                    view=sample.view,
                    label=sample.label,
                    operation=mode,
                )
            )

    return records


def build_summary(
    split_map: dict[str, list[Sample]],
    source_count: int,
    skipped_count: int,
    args: argparse.Namespace,
) -> dict:
    summary = {
        "configuration": {
            "dataset": str(Path(args.dataset).expanduser().resolve()),
            "output": str(Path(args.output).expanduser().resolve()),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "mode": args.mode,
        },
        "source_images_discovered": source_count,
        "skipped_images": skipped_count,
        "splits": {},
    }

    for split_name, samples in split_map.items():
        counts = Counter(
            (sample.view, sample.label)
            for sample in samples
        )

        split_summary = {
            "total": len(samples),
            "by_view_and_label": {},
        }

        for view in ("front", "back", "left", "right"):
            split_summary["by_view_and_label"][view] = {
                "complete": counts.get((view, "complete"), 0),
                "incomplete": counts.get((view, "incomplete"), 0),
            }

        summary["splits"][split_name] = split_summary

    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 76)
    print("DATASET SPLIT SUMMARY")
    print("=" * 76)

    for split_name in ("train", "val", "test"):
        split = summary["splits"][split_name]
        print(f"\n{split_name.upper()} — {split['total']} images")

        for view in ("front", "back", "left", "right"):
            counts = split["by_view_and_label"][view]
            print(
                f"  {view:<5} "
                f"complete={counts['complete']:4d} | "
                f"incomplete={counts['incomplete']:4d}"
            )

    print("=" * 76)


def main(args: argparse.Namespace) -> None:
    validate_ratios(
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    dataset_root = Path(args.dataset).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {dataset_root}"
        )

    if dataset_root == output_root:
        raise ValueError(
            "The output directory must be different from the source dataset."
        )

    samples, skipped = discover_samples(dataset_root)

    if not samples:
        raise RuntimeError(
            "No labeled images were discovered. Folder names must contain "
            "one view and one completeness label."
        )

    split_map = build_stratified_split(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if args.dry_run:
        records: list[SplitRecord] = []
    else:
        output_root.mkdir(parents=True, exist_ok=True)

        records = create_split_folders(
            split_map=split_map,
            output_root=output_root,
            mode=args.mode,
            overwrite=args.overwrite,
        )

        manifest_path = output_root / "split_manifest.csv"

        with manifest_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(asdict(records[0]).keys()),
            )
            writer.writeheader()

            for record in records:
                writer.writerow(asdict(record))

        with (output_root / "skipped_images.txt").open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write("\n".join(skipped))

    summary = build_summary(
        split_map=split_map,
        source_count=len(samples),
        skipped_count=len(skipped),
        args=args,
    )

    print_summary(summary)

    if args.dry_run:
        print("\nDry run only: no folders or files were created.")
        return

    with (output_root / "split_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"\nOutput:   {output_root}")
    print(f"Manifest: {output_root / 'split_manifest.csv'}")
    print(f"Summary:  {output_root / 'split_summary.json'}")
    print(
        "\nUse this folder for synthetic generation:\n"
        f"  {output_root / 'train'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create stratified train/validation/test folders for "
            "vehicle completeness classification."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Root of the original labeled dataset.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Destination root for train/val/test folders.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
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
        "--mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help=(
            "copy: independent files; "
            "hardlink: saves disk space on the same filesystem; "
            "symlink: references original files."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed split without creating files.",
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
