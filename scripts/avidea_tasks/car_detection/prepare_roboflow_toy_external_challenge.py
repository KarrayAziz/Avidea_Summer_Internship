#!/usr/bin/env python3
"""
Prepare an external Roboflow toy-car challenge set for evaluating the existing
real-vs-toy MobileNetV3-Small classifier.

Input structure expected:

toy_cars_roboflow/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
└── ...

Processing:
- Pool train + valid + test as ONE external evaluation source.
- Read standard YOLO normalized labels:
      class_id cx cy width height
- Generate a 25%-expanded contextual crop around each annotation.
- Keep at most 4 crops per source image.
- Never split into train/val/test; this whole dataset remains external test data.
- Save a manifest with source split and source image identity.

Output:

toy_cars_roboflow/
└── external_authenticity_challenge/
    ├── contextual_crops_all/
    ├── contextual_crops_capped/
    ├── manifest_all.csv
    └── manifest_capped.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_SPLITS = ("train", "valid", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Roboflow toy-car external challenge crops."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/"
            "data/toy_cars_roboflow"
        ),
        help="Root of the downloaded Roboflow dataset.",
    )

    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="external_authenticity_challenge",
    )

    parser.add_argument(
        "--context-expansion",
        type=float,
        default=0.25,
        help="Fraction of bbox width/height added on EACH side.",
    )

    parser.add_argument(
        "--max-per-source",
        type=int,
        default=4,
        help="Maximum contextual crops retained per original image.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )

    return parser.parse_args()


def parse_yolo_label_line(line: str):
    parts = line.strip().split()

    if len(parts) < 5:
        return None

    try:
        class_id = int(float(parts[0]))
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return None

    if w <= 0 or h <= 0:
        return None

    return class_id, cx, cy, w, h


def yolo_to_xyxy(cx, cy, w, h, image_width, image_height):
    cx_px = cx * image_width
    cy_px = cy * image_height
    w_px = w * image_width
    h_px = h * image_height

    x1 = cx_px - w_px / 2.0
    y1 = cy_px - h_px / 2.0
    x2 = cx_px + w_px / 2.0
    y2 = cy_px + h_px / 2.0

    return x1, y1, x2, y2


def expand_box(x1, y1, x2, y2, image_width, image_height, expansion):
    box_w = x2 - x1
    box_h = y2 - y1

    expand_x = box_w * expansion
    expand_y = box_h * expansion

    ex1 = max(0, int(round(x1 - expand_x)))
    ey1 = max(0, int(round(y1 - expand_y)))
    ex2 = min(image_width, int(round(x2 + expand_x)))
    ey2 = min(image_height, int(round(y2 + expand_y)))

    return ex1, ey1, ex2, ey2


def get_images(images_dir: Path):
    return sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def locate_label(labels_dir: Path, image_path: Path):
    return labels_dir / f"{image_path.stem}.txt"


def main():
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = dataset_root / args.output_dir_name
    all_crops_dir = output_root / "contextual_crops_all"
    capped_crops_dir = output_root / "contextual_crops_capped"
    manifest_all_path = output_root / "manifest_all.csv"
    manifest_capped_path = output_root / "manifest_capped.csv"

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output already exists:\n{output_root}\n\n"
                "Use --overwrite to recreate it."
            )

    all_crops_dir.mkdir(parents=True, exist_ok=True)
    capped_crops_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 86)
    print("ROBOFLOW TOY-CAR EXTERNAL CHALLENGE PREPARATION")
    print("=" * 86)
    print(f"Dataset root       : {dataset_root}")
    print(f"Output root        : {output_root}")
    print(f"Context expansion  : {args.context_expansion * 100:.0f}% per side")
    print(f"Max crops/source   : {args.max_per_source}")
    print(f"Seed               : {args.seed}")
    print()

    manifest_rows = []

    total_images = 0
    total_annotations = 0
    total_crops = 0
    missing_labels = 0
    empty_labels = 0
    failed_images = 0
    invalid_annotations = 0

    split_stats = {}

    for source_split in SOURCE_SPLITS:
        images_dir = dataset_root / source_split / "images"
        labels_dir = dataset_root / source_split / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images directory: {images_dir}")

        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

        images = get_images(images_dir)
        total_images += len(images)

        split_crops = 0
        split_annotations = 0

        print(f"{source_split}: {len(images)} images")

        for image_index, image_path in enumerate(images, start=1):
            label_path = locate_label(labels_dir, image_path)

            if not label_path.exists():
                missing_labels += 1
                continue

            raw_lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            if not raw_lines:
                empty_labels += 1
                continue

            parsed = []
            for line in raw_lines:
                item = parse_yolo_label_line(line)
                if item is None:
                    invalid_annotations += 1
                    continue
                parsed.append(item)

            if not parsed:
                empty_labels += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                failed_images += 1
                continue

            image_h, image_w = image.shape[:2]

            # Unique across pooled Roboflow splits, even if filenames repeat.
            source_image_id = f"{source_split}__{image_path.stem}"

            for object_index, (class_id, cx, cy, w, h) in enumerate(parsed, start=1):
                x1, y1, x2, y2 = yolo_to_xyxy(
                    cx, cy, w, h, image_w, image_h
                )

                ex1, ey1, ex2, ey2 = expand_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    image_w,
                    image_h,
                    args.context_expansion,
                )

                if ex2 <= ex1 or ey2 <= ey1:
                    invalid_annotations += 1
                    continue

                crop = image[ey1:ey2, ex1:ex2]

                if crop.size == 0:
                    invalid_annotations += 1
                    continue

                crop_filename = (
                    f"{source_split}__{image_path.stem}"
                    f"__toy_{object_index:03d}.jpg"
                )

                crop_path = all_crops_dir / crop_filename

                if not cv2.imwrite(str(crop_path), crop):
                    invalid_annotations += 1
                    continue

                original_box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                image_area = image_w * image_h
                bbox_area_ratio = (
                    original_box_area / image_area if image_area > 0 else 0.0
                )

                manifest_rows.append(
                    {
                        "source_dataset": "roboflow_toy_cars_hqi4o_v2",
                        "source_split": source_split,
                        "source_image_id": source_image_id,
                        "source_image": image_path.name,
                        "source_image_path": str(image_path),
                        "source_label_path": str(label_path),
                        "num_objects_in_source": len(parsed),
                        "object_index": object_index,
                        "class_id": class_id,
                        "image_width": image_w,
                        "image_height": image_h,
                        "yolo_cx": cx,
                        "yolo_cy": cy,
                        "yolo_width": w,
                        "yolo_height": h,
                        "bbox_x1": round(x1, 2),
                        "bbox_y1": round(y1, 2),
                        "bbox_x2": round(x2, 2),
                        "bbox_y2": round(y2, 2),
                        "bbox_area_ratio": round(bbox_area_ratio, 6),
                        "context_expansion": args.context_expansion,
                        "crop_x1": ex1,
                        "crop_y1": ey1,
                        "crop_x2": ex2,
                        "crop_y2": ey2,
                        "crop_width": ex2 - ex1,
                        "crop_height": ey2 - ey1,
                        "crop_filename": crop_filename,
                        "crop_path": str(crop_path),
                    }
                )

                total_annotations += 1
                total_crops += 1
                split_annotations += 1
                split_crops += 1

            print(
                f"\r  [{image_index:4d}/{len(images):4d}] "
                f"crops={split_crops:5d}",
                end="",
            )

        print()

        split_stats[source_split] = {
            "images": len(images),
            "annotations": split_annotations,
            "crops": split_crops,
        }

    if not manifest_rows:
        raise RuntimeError("No contextual crops were generated.")

    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest_all_path, index=False)

    # --------------------------------------------------------
    # Cap at max N crops per source image
    # --------------------------------------------------------

    rng = random.Random(args.seed)
    selected_indices = []

    for source_image_id, group in df.groupby("source_image_id", sort=False):
        indices = list(group.index)

        if len(indices) <= args.max_per_source:
            selected_indices.extend(indices)
        else:
            selected_indices.extend(
                rng.sample(indices, args.max_per_source)
            )

    capped_df = (
        df.loc[selected_indices]
        .copy()
        .sort_values(["source_split", "source_image_id", "object_index"])
        .reset_index(drop=True)
    )

    new_paths = []

    for _, row in capped_df.iterrows():
        src = Path(row["crop_path"])
        dst = capped_crops_dir / src.name
        shutil.copy2(src, dst)
        new_paths.append(str(dst))

    capped_df["original_all_crop_path"] = capped_df["crop_path"]
    capped_df["crop_path"] = new_paths
    capped_df.to_csv(manifest_capped_path, index=False)

    unique_sources = df["source_image_id"].nunique()
    capped_sources = capped_df["source_image_id"].nunique()
    sources_above_cap = int(
        (df.groupby("source_image_id").size() > args.max_per_source).sum()
    )

    print()
    print("=" * 86)
    print("SUMMARY")
    print("=" * 86)

    print(
        f"{'Split':<10}"
        f"{'Images':>10}"
        f"{'Annotations':>15}"
        f"{'Crops':>10}"
    )
    print("-" * 45)

    for source_split in SOURCE_SPLITS:
        stats = split_stats[source_split]
        print(
            f"{source_split:<10}"
            f"{stats['images']:>10}"
            f"{stats['annotations']:>15}"
            f"{stats['crops']:>10}"
        )

    print("-" * 45)
    print(f"Total images discovered       : {total_images}")
    print(f"Unique source images w/crops  : {unique_sources}")
    print(f"All contextual crops          : {len(df)}")
    print(f"Sources above cap             : {sources_above_cap}")
    print(f"Capped contextual crops       : {len(capped_df)}")
    print(f"Unique capped source images   : {capped_sources}")
    print(f"Dropped by cap                : {len(df) - len(capped_df)}")
    print(f"Missing labels                : {missing_labels}")
    print(f"Empty labels                  : {empty_labels}")
    print(f"Failed images                 : {failed_images}")
    print(f"Invalid annotations/crops     : {invalid_annotations}")

    print(f"\nAll-crop manifest:\n  {manifest_all_path}")
    print(f"\nCapped manifest:\n  {manifest_capped_path}")
    print(f"\nFinal challenge crops:\n  {capped_crops_dir}")

    print(
        "\nIMPORTANT: Keep this dataset external. "
        "Do not train or tune the classifier on these images."
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
