#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import argparse
import csv
import os
import random
import shutil

import pandas as pd


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_TOY_ROOT = Path(
    "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/"
    "data/toy_cars/toy_scale_sources/toy_cars_yolo"
)

DEFAULT_REAL_ROOT = Path(
    "/home/aziz/Pictures/Internship_Images/"
    "no_duplicates_detection_des faces_500/"
    "real_scale_sources/avidea_real_yolo"
)

DEFAULT_OUTPUT = Path(
    "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/"
    "data/car_authenticity_dataset"
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Create fixed source-aware train/val/test split "
            "for real-vs-toy car authenticity classification."
        )
    )

    parser.add_argument(
        "--toy-root",
        type=Path,
        default=DEFAULT_TOY_ROOT,
    )

    parser.add_argument(
        "--real-root",
        type=Path,
        default=DEFAULT_REAL_ROOT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "hardlink",
            "copy",
            "symlink",
        ],
        default="hardlink",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# FILE EXPORT
# ============================================================

def export_file(
    source: Path,
    destination: Path,
    mode: str,
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        return

    if mode == "copy":

        shutil.copy2(
            source,
            destination,
        )

    elif mode == "symlink":

        destination.symlink_to(
            source.resolve()
        )

    elif mode == "hardlink":

        try:

            os.link(
                source,
                destination,
            )

        except OSError:

            print(
                f"\nWARNING: hardlink failed for:\n"
                f"  {source}\n"
                f"Falling back to copy."
            )

            shutil.copy2(
                source,
                destination,
            )


# ============================================================
# SPLIT SOURCE IDs
# ============================================================

def split_source_ids(
    source_ids,
    train_ratio,
    val_ratio,
    seed,
):

    source_ids = list(
        source_ids
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        source_ids
    )

    n = len(
        source_ids
    )

    n_train = round(
        n * train_ratio
    )

    n_val = round(
        n * val_ratio
    )

    # Everything remaining goes to test
    n_test = (
        n
        - n_train
        - n_val
    )

    train_ids = set(
        source_ids[
            :n_train
        ]
    )

    val_ids = set(
        source_ids[
            n_train:
            n_train + n_val
        ]
    )

    test_ids = set(
        source_ids[
            n_train + n_val:
        ]
    )

    assert (
        len(train_ids)
        + len(val_ids)
        + len(test_ids)
        == n
    )

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


# ============================================================
# TOY DATASET
# ============================================================

def load_toy_samples(
    toy_root: Path,
):

    manifest_path = (
        toy_root
        / "manifest_capped.csv"
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Toy manifest not found:\n"
            f"{manifest_path}"
        )

    df = pd.read_csv(
        manifest_path
    )

    required_columns = {
        "source_image",
        "crop_path",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Toy manifest is missing columns: "
            + ", ".join(missing)
        )

    samples = []

    for _, row in df.iterrows():

        source_image = str(
            row["source_image"]
        )

        source_id = Path(
            source_image
        ).stem

        crop_path = Path(
            str(row["crop_path"])
        )

        if not crop_path.is_absolute():

            crop_path = (
                toy_root
                / crop_path
            )

        if not crop_path.exists():

            # fallback
            fallback = (
                toy_root
                / "contextual_crops_capped"
                / Path(
                    str(row["crop_path"])
                ).name
            )

            if fallback.exists():

                crop_path = (
                    fallback
                )

            else:

                raise FileNotFoundError(
                    f"Toy crop not found:\n"
                    f"{crop_path}"
                )

        samples.append(
            {
                "source_id":
                    source_id,

                "source_image":
                    source_image,

                "crop_path":
                    crop_path,

                "class_name":
                    "toy_scale",

                "source_dataset":
                    "toy_cars_yolo",

                "view":
                    "",
            }
        )

    return samples


# ============================================================
# REAL DATASET
# ============================================================

def load_real_samples(
    real_root: Path,
):

    manifest_path = (
        real_root
        / "manifest.csv"
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Real manifest not found:\n"
            f"{manifest_path}"
        )

    df = pd.read_csv(
        manifest_path
    )

    required_columns = {
        "source_image",
        "crop_path",
        "view",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Real manifest is missing columns: "
            + ", ".join(missing)
        )

    samples = []

    for _, row in df.iterrows():

        source_image = str(
            row["source_image"]
        )

        view = str(
            row["view"]
        ).lower()

        # Include view in source ID because different
        # view folders could theoretically contain
        # identical filenames
        source_id = (
            f"{view}__"
            f"{Path(source_image).stem}"
        )

        crop_path = Path(
            str(row["crop_path"])
        )

        if not crop_path.is_absolute():

            crop_path = (
                real_root
                / crop_path
            )

        if not crop_path.exists():

            fallback = (
                real_root
                / "contextual_crops"
                / view
                / Path(
                    str(row["crop_path"])
                ).name
            )

            if fallback.exists():

                crop_path = (
                    fallback
                )

            else:

                raise FileNotFoundError(
                    f"Real crop not found:\n"
                    f"{crop_path}"
                )

        samples.append(
            {
                "source_id":
                    source_id,

                "source_image":
                    source_image,

                "crop_path":
                    crop_path,

                "class_name":
                    "real",

                "source_dataset":
                    "avidea",

                "view":
                    view,
            }
        )

    return samples


# ============================================================
# TOY SOURCE-AWARE SPLIT
# ============================================================

def assign_toy_splits(
    samples,
    train_ratio,
    val_ratio,
    seed,
):

    source_ids = sorted(
        {
            sample["source_id"]
            for sample in samples
        }
    )

    split_ids = split_source_ids(
        source_ids,
        train_ratio,
        val_ratio,
        seed,
    )

    source_to_split = {}

    for split_name, ids in split_ids.items():

        for source_id in ids:

            source_to_split[
                source_id
            ] = split_name

    for sample in samples:

        sample["split"] = (
            source_to_split[
                sample["source_id"]
            ]
        )

    return split_ids


# ============================================================
# REAL VIEW-STRATIFIED SPLIT
# ============================================================

def assign_real_splits(
    samples,
    train_ratio,
    val_ratio,
    seed,
):

    by_view = defaultdict(
        list
    )

    for sample in samples:

        by_view[
            sample["view"]
        ].append(
            sample
        )

    split_counts_by_view = {}

    for view_index, (
        view,
        view_samples,
    ) in enumerate(
        sorted(by_view.items())
    ):

        source_ids = [
            sample["source_id"]
            for sample in view_samples
        ]

        split_ids = split_source_ids(
            source_ids,
            train_ratio,
            val_ratio,
            seed + view_index,
        )

        source_to_split = {}

        for split_name, ids in split_ids.items():

            for source_id in ids:

                source_to_split[
                    source_id
                ] = split_name

        for sample in view_samples:

            sample["split"] = (
                source_to_split[
                    sample["source_id"]
                ]
            )

        split_counts_by_view[
            view
        ] = {
            split:
                len(ids)

            for split, ids
            in split_ids.items()
        }

    return split_counts_by_view


# ============================================================
# VERIFY NO TOY SOURCE LEAKAGE
# ============================================================

def verify_source_leakage(
    samples,
    class_name,
):

    source_splits = defaultdict(
        set
    )

    for sample in samples:

        source_splits[
            sample["source_id"]
        ].add(
            sample["split"]
        )

    leaking_sources = {
        source:
            splits

        for source, splits
        in source_splits.items()

        if len(splits) > 1
    }

    if leaking_sources:

        raise RuntimeError(
            f"Source leakage detected "
            f"in class '{class_name}': "
            f"{len(leaking_sources)} sources"
        )

    print(
        f"✓ No source leakage in "
        f"{class_name}"
    )


# ============================================================
# EXPORT DATASET
# ============================================================

def export_samples(
    samples,
    output_root,
    mode,
):

    manifest_rows = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        split = (
            sample["split"]
        )

        class_name = (
            sample["class_name"]
        )

        source_path = (
            sample["crop_path"]
        )

        # Prefix destination filenames so we don't
        # risk name collisions between sources

        if class_name == "real":

            filename = (
                f"avidea__"
                f"{sample['view']}__"
                f"{source_path.name}"
            )

        else:

            filename = (
                f"toy__"
                f"{source_path.name}"
            )

        destination = (
            output_root
            / split
            / class_name
            / filename
        )

        export_file(
            source_path,
            destination,
            mode,
        )

        manifest_rows.append(
            {
                "split":
                    split,

                "class_name":
                    class_name,

                "source_dataset":
                    sample[
                        "source_dataset"
                    ],

                "source_id":
                    sample[
                        "source_id"
                    ],

                "source_image":
                    sample[
                        "source_image"
                    ],

                "view":
                    sample[
                        "view"
                    ],

                "original_crop_path":
                    str(
                        source_path
                    ),

                "dataset_path":
                    str(
                        destination
                    ),
            }
        )

        print(
            f"\r"
            f"Exporting "
            f"{index:4d}/"
            f"{len(samples):4d}",
            end="",
        )

    print()

    return manifest_rows


# ============================================================
# SUMMARY
# ============================================================

def write_summary(
    manifest_rows,
    output_root,
):

    counts = defaultdict(
        int
    )

    source_sets = defaultdict(
        set
    )

    for row in manifest_rows:

        key = (
            row["split"],
            row["class_name"],
        )

        counts[key] += 1

        source_sets[key].add(
            row["source_id"]
        )

    summary_rows = []

    print("\n")
    print("=" * 80)
    print("FINAL DATASET SPLIT")
    print("=" * 80)

    print(
        f"{'Split':<10}"
        f"{'Class':<15}"
        f"{'Images':>10}"
        f"{'Sources':>12}"
    )

    print("-" * 47)

    for split in [
        "train",
        "val",
        "test",
    ]:

        for class_name in [
            "real",
            "toy_scale",
        ]:

            key = (
                split,
                class_name,
            )

            images = (
                counts[key]
            )

            sources = len(
                source_sets[key]
            )

            print(
                f"{split:<10}"
                f"{class_name:<15}"
                f"{images:>10}"
                f"{sources:>12}"
            )

            summary_rows.append(
                {
                    "split":
                        split,

                    "class_name":
                        class_name,

                    "images":
                        images,

                    "source_images":
                        sources,
                }
            )

    summary_path = (
        output_root
        / "split_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "class_name",
                "images",
                "source_images",
            ],
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    return summary_path


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    ratio_sum = (
        args.train_ratio
        + args.val_ratio
        + args.test_ratio
    )

    if abs(
        ratio_sum - 1.0
    ) > 1e-6:

        raise ValueError(
            "train + val + test ratios "
            "must equal 1.0"
        )

    output_root = (
        args.output
        .expanduser()
        .resolve()
    )

    toy_root = (
        args.toy_root
        .expanduser()
        .resolve()
    )

    real_root = (
        args.real_root
        .expanduser()
        .resolve()
    )

    if output_root.exists():

        if args.overwrite:

            shutil.rmtree(
                output_root
            )

        else:

            raise FileExistsError(
                f"Output already exists:\n"
                f"{output_root}\n\n"
                f"Use --overwrite if you want "
                f"to recreate it."
            )

    print("=" * 80)
    print("CAR AUTHENTICITY DATASET SPLIT")
    print("=" * 80)

    print(
        f"\nTrain / Val / Test: "
        f"{args.train_ratio:.2f} / "
        f"{args.val_ratio:.2f} / "
        f"{args.test_ratio:.2f}"
    )

    print(
        f"Seed              : "
        f"{args.seed}"
    )

    print(
        f"Export mode       : "
        f"{args.mode}"
    )

    print(
        f"\nToy root:\n"
        f"  {toy_root}"
    )

    print(
        f"\nReal root:\n"
        f"  {real_root}"
    )

    print(
        f"\nOutput:\n"
        f"  {output_root}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\nLoading toy samples..."
    )

    toy_samples = (
        load_toy_samples(
            toy_root
        )
    )

    print(
        f"Toy crops: "
        f"{len(toy_samples)}"
    )

    print(
        "\nLoading real samples..."
    )

    real_samples = (
        load_real_samples(
            real_root
        )
    )

    print(
        f"Real crops: "
        f"{len(real_samples)}"
    )

    # --------------------------------------------------------
    # Assign splits
    # --------------------------------------------------------

    print(
        "\nCreating source-aware "
        "toy split..."
    )

    toy_split_ids = (
        assign_toy_splits(
            toy_samples,
            args.train_ratio,
            args.val_ratio,
            args.seed,
        )
    )

    print(
        "\nCreating view-stratified "
        "real split..."
    )

    real_view_counts = (
        assign_real_splits(
            real_samples,
            args.train_ratio,
            args.val_ratio,
            args.seed + 100,
        )
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    print(
        "\nChecking source leakage..."
    )

    verify_source_leakage(
        toy_samples,
        "toy_scale",
    )

    verify_source_leakage(
        real_samples,
        "real",
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    all_samples = (
        real_samples
        + toy_samples
    )

    print(
        f"\nExporting "
        f"{len(all_samples)} "
        f"images..."
    )

    manifest_rows = (
        export_samples(
            all_samples,
            output_root,
            args.mode,
        )
    )

    # --------------------------------------------------------
    # Full manifest
    # --------------------------------------------------------

    manifest_path = (
        output_root
        / "split_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                manifest_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_path = (
        write_summary(
            manifest_rows,
            output_root,
        )
    )

    print("\n")
    print("=" * 80)
    print("CREATED")
    print("=" * 80)

    print(
        f"\nDataset:\n"
        f"  {output_root}"
    )

    print(
        f"\nManifest:\n"
        f"  {manifest_path}"
    )

    print(
        f"\nSummary:\n"
        f"  {summary_path}"
    )

    print(
        "\nNo source-image leakage "
        "was detected."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()