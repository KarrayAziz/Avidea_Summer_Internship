#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired Right V3 augmentations: rear/left incomplete + hard-complete."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--yolo",
        type=Path,
        default=Path(
            "/home/aziz/Aziz/DigiCover/usingGeminiApi/models/yolov8m.pt"
        ),
    )
    parser.add_argument("--incomplete-count", type=int, default=36)
    parser.add_argument("--complete-count", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--min-area-ratio", type=float, default=0.08)
    parser.add_argument("--min-output-size", type=int, default=224)
    parser.add_argument("--hard-margin-min", type=float, default=0.008)
    parser.add_argument("--hard-margin-max", type=float, default=0.025)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=1303)
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_original_complete_images(folder: Path) -> list[Path]:
    images = []
    for path in folder.rglob("*"):
        if not is_image(path):
            continue

        relative_parts = [
            part.lower() for part in path.relative_to(folder).parts
        ]

        if any("synthetic" in part for part in relative_parts):
            continue

        images.append(path)

    return sorted(images)


def largest_vehicle_box(
    model: YOLO,
    image: np.ndarray,
    conf: float,
    device: str,
    min_area_ratio: float,
) -> tuple[int, int, int, int] | None:
    height, width = image.shape[:2]
    image_area = float(height * width)

    results = model.predict(
        source=image,
        conf=conf,
        classes=VEHICLE_CLASSES,
        device=device,
        verbose=False,
    )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    best_box = None
    best_area = -1.0

    for raw_box in results[0].boxes.xyxy.detach().cpu().numpy():
        x1, y1, x2, y2 = map(float, raw_box)
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        area = box_width * box_height

        if area / image_area < min_area_ratio:
            continue

        if area > best_area:
            best_area = area
            best_box = (
                max(0, int(np.floor(x1))),
                max(0, int(np.floor(y1))),
                min(width, int(np.ceil(x2))),
                min(height, int(np.ceil(y2))),
            )

    return best_box


def sample_severity(rng: random.Random) -> tuple[float, str]:
    draw = rng.random()

    if draw < 0.50:
        return rng.uniform(0.01, 0.04), "01_04"
    if draw < 0.85:
        return rng.uniform(0.04, 0.08), "04_08"
    return rng.uniform(0.08, 0.15), "08_15"


def make_incomplete_crop(
    image_shape: tuple[int, int, int],
    box: tuple[int, int, int, int],
    rng: random.Random,
    min_output_size: int,
) -> tuple[tuple[int, int, int, int], str, float, str] | None:
    image_height, image_width = image_shape[:2]
    box_x1, _, box_x2, _ = box
    box_width = box_x2 - box_x1

    if box_width <= 0:
        return None

    crop_type = rng.choices(
        ["left_rear", "both"],
        weights=[80, 20],
        k=1,
    )[0]

    severity, severity_band = sample_severity(rng)
    removed_pixels = max(2, int(round(box_width * severity)))

    crop_x1 = 0
    crop_x2 = image_width

    if crop_type == "left_rear":
        crop_x1 = max(
            0,
            min(image_width - 1, box_x1 + removed_pixels),
        )

    else:
        left_removed = max(1, int(round(removed_pixels * 0.70)))
        right_removed = max(1, removed_pixels - left_removed)

        crop_x1 = max(
            0,
            min(image_width - 1, box_x1 + left_removed),
        )
        crop_x2 = max(
            1,
            min(image_width, box_x2 - right_removed),
        )

    if crop_x2 - crop_x1 < min_output_size:
        return None

    if crop_type == "left_rear" and not crop_x1 > box_x1:
        return None

    if crop_type == "both" and not (
        crop_x1 > box_x1 and crop_x2 < box_x2
    ):
        return None

    return (
        (crop_x1, 0, crop_x2, image_height),
        crop_type,
        severity,
        severity_band,
    )


def make_hard_complete_crop(
    image_shape: tuple[int, int, int],
    box: tuple[int, int, int, int],
    rng: random.Random,
    margin_min: float,
    margin_max: float,
    min_output_size: int,
) -> tuple[int, int, int, int] | None:
    image_height, image_width = image_shape[:2]
    box_x1, box_y1, box_x2, box_y2 = box

    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1

    if box_width <= 0 or box_height <= 0:
        return None

    margin_x = max(
        3,
        int(round(box_width * rng.uniform(margin_min, margin_max))),
    )

    # Keep more vertical context; the experiment targets horizontal tightness.
    margin_y = max(
        6,
        int(round(box_height * rng.uniform(0.03, 0.08))),
    )

    crop_x1 = max(0, box_x1 - margin_x)
    crop_x2 = min(image_width, box_x2 + margin_x)
    crop_y1 = max(0, box_y1 - margin_y)
    crop_y2 = min(image_height, box_y2 + margin_y)

    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    if crop_width < min_output_size or crop_height < min_output_size:
        return None

    if not (
        crop_x1 <= box_x1
        and crop_x2 >= box_x2
        and crop_y1 <= box_y1
        and crop_y2 >= box_y2
    ):
        return None

    horizontal_fill_ratio = box_width / float(crop_width)

    # Keep only genuinely tight horizontal crops.
    if horizontal_fill_ratio < 0.94:
        return None

    return crop_x1, crop_y1, crop_x2, crop_y2


def save_preview(
    image: np.ndarray,
    vehicle_box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    preview = image.copy()

    vx1, vy1, vx2, vy2 = vehicle_box
    cx1, cy1, cx2, cy2 = crop_box

    cv2.rectangle(preview, (vx1, vy1), (vx2, vy2), (0, 255, 0), 3)
    cv2.rectangle(preview, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    source_dir = args.dataset / "train" / "right" / "complete"

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    if not args.yolo.exists():
        raise FileNotFoundError(f"YOLO model not found: {args.yolo}")

    if args.overwrite and args.output.exists():
        shutil.rmtree(args.output)

    incomplete_dir = args.output / "candidates" / "incomplete" / "right"
    complete_dir = args.output / "candidates" / "complete" / "right"
    incomplete_preview_dir = args.output / "previews" / "incomplete" / "right"
    complete_preview_dir = args.output / "previews" / "complete" / "right"

    incomplete_dir.mkdir(parents=True, exist_ok=True)
    complete_dir.mkdir(parents=True, exist_ok=True)

    source_images = list_original_complete_images(source_dir)
    rng.shuffle(source_images)

    model = YOLO(str(args.yolo))

    detections: list[
        tuple[Path, np.ndarray, tuple[int, int, int, int]]
    ] = []

    for source_path in source_images:
        image = cv2.imread(str(source_path))
        if image is None:
            continue

        vehicle_box = largest_vehicle_box(
            model=model,
            image=image,
            conf=args.conf,
            device=args.device,
            min_area_ratio=args.min_area_ratio,
        )

        if vehicle_box is not None:
            detections.append((source_path, image, vehicle_box))

    if not detections:
        raise RuntimeError("No valid YOLO detections found.")

    manifest_rows: list[dict[str, object]] = []

    generated_incomplete = 0
    attempts = 0

    while (
        generated_incomplete < args.incomplete_count
        and attempts < max(500, args.incomplete_count * 40)
    ):
        source_path, image, vehicle_box = detections[
            attempts % len(detections)
        ]
        attempts += 1

        result = make_incomplete_crop(
            image_shape=image.shape,
            box=vehicle_box,
            rng=rng,
            min_output_size=args.min_output_size,
        )

        if result is None:
            continue

        crop_box, crop_type, severity, severity_band = result
        x1, y1, x2, y2 = crop_box
        cropped = image[y1:y2, x1:x2]

        filename = (
            f"{generated_incomplete + 1:04d}__{source_path.stem}"
            f"__{crop_type}__{severity_band}.jpg"
        )
        output_path = incomplete_dir / filename

        if not cv2.imwrite(str(output_path), cropped):
            continue

        generated_incomplete += 1

        if args.save_previews:
            save_preview(
                image=image,
                vehicle_box=vehicle_box,
                crop_box=crop_box,
                output_path=incomplete_preview_dir / filename,
            )

        manifest_rows.append(
            {
                "label": "incomplete",
                "source_path": str(source_path),
                "output_path": str(output_path),
                "crop_type": crop_type,
                "severity": round(severity, 6),
                "severity_band": severity_band,
                "vehicle_box": ",".join(map(str, vehicle_box)),
                "crop_box": ",".join(map(str, crop_box)),
            }
        )

    generated_complete = 0
    attempts = 0

    while (
        generated_complete < args.complete_count
        and attempts < max(400, args.complete_count * 50)
    ):
        source_path, image, vehicle_box = detections[
            attempts % len(detections)
        ]
        attempts += 1

        crop_box = make_hard_complete_crop(
            image_shape=image.shape,
            box=vehicle_box,
            rng=rng,
            margin_min=args.hard_margin_min,
            margin_max=args.hard_margin_max,
            min_output_size=args.min_output_size,
        )

        if crop_box is None:
            continue

        x1, y1, x2, y2 = crop_box
        cropped = image[y1:y2, x1:x2]

        filename = (
            f"{generated_complete + 1:04d}__{source_path.stem}"
            "__hard_complete.jpg"
        )
        output_path = complete_dir / filename

        if not cv2.imwrite(str(output_path), cropped):
            continue

        generated_complete += 1

        if args.save_previews:
            save_preview(
                image=image,
                vehicle_box=vehicle_box,
                crop_box=crop_box,
                output_path=complete_preview_dir / filename,
            )

        manifest_rows.append(
            {
                "label": "complete",
                "source_path": str(source_path),
                "output_path": str(output_path),
                "crop_type": "hard_complete",
                "severity": "",
                "severity_band": "",
                "vehicle_box": ",".join(map(str, vehicle_box)),
                "crop_box": ",".join(map(str, crop_box)),
            }
        )

    manifest_path = args.output / "manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "label",
            "source_path",
            "output_path",
            "crop_type",
            "severity",
            "severity_band",
            "vehicle_box",
            "crop_box",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print(f"Generated incomplete crops: {generated_incomplete}")
    print(f"Generated hard-complete:    {generated_complete}")
    print(f"Output:                     {args.output}")
    print(f"Manifest:                   {manifest_path}")


if __name__ == "__main__":
    main()
