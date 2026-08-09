#!/usr/bin/env python3
"""
Generate targeted synthetic incomplete vehicle images by cropping into YOLO detections.

Important:
- Use this only on TRAIN images.
- Generate from images labeled complete.
- Manually review candidates before training.
- Keep validation/test sets untouched.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm
from ultralytics import YOLO


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VIEW_ALIASES = {
    "front": "front",
    "back": "back",
    "rear": "back",
    "left": "left",
    "right": "right",
}

COMPLETE_ALIASES = {"complete", "completed", "full", "uncropped"}


@dataclass
class SourceSample:
    path: str
    view: str


@dataclass
class GeneratedRecord:
    source_path: str
    output_path: str
    preview_path: str
    view: str
    crop_type: str
    severity: float
    image_width: int
    image_height: int
    vehicle_x1: float
    vehicle_y1: float
    vehicle_x2: float
    vehicle_y2: float
    yolo_confidence: float
    yolo_class_id: int
    status: str


def normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def infer_view_and_complete(
    image_path: Path,
    dataset_root: Path,
) -> tuple[Optional[str], bool]:
    parts = image_path.relative_to(dataset_root).parts[:-1]
    tokens = [normalize_token(part) for part in parts]

    views = {VIEW_ALIASES[token] for token in tokens if token in VIEW_ALIASES}
    is_complete = any(token in COMPLETE_ALIASES for token in tokens)

    view = next(iter(views)) if len(views) == 1 else None
    return view, is_complete


def discover_complete_samples(
    dataset_root: Path,
    selected_views: list[str],
) -> tuple[list[SourceSample], list[str]]:
    samples: list[SourceSample] = []
    skipped: list[str] = []

    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        view, is_complete = infer_view_and_complete(path, dataset_root)

        if view is None or not is_complete or view not in selected_views:
            skipped.append(str(path))
            continue

        samples.append(SourceSample(path=str(path), view=view))

    return samples, skipped


def select_largest_vehicle_box(result) -> Optional[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return None

    xyxy = result.boxes.xyxy.cpu()
    confidences = result.boxes.conf.cpu()
    classes = result.boxes.cls.cpu()

    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    areas = widths * heights

    index = int(areas.argmax().item())

    return {
        "xyxy": [float(v) for v in xyxy[index].tolist()],
        "confidence": float(confidences[index].item()),
        "class_id": int(classes[index].item()),
        "area": float(areas[index].item()),
    }


def crop_types_for_view(view: str) -> list[str]:
    if view in {"front", "back"}:
        return [
            "left", "right", "top", "bottom",
            "top_left", "top_right", "bottom_left", "bottom_right",
        ]

    return [
        "left", "left", "right", "right",
        "top", "bottom",
        "top_left", "top_right", "bottom_left", "bottom_right",
    ]


def compute_crop(
    image_width: int,
    image_height: int,
    box: tuple[float, float, float, float],
    crop_type: str,
    severity: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    vehicle_width = max(1.0, x2 - x1)
    vehicle_height = max(1.0, y2 - y1)

    remove_x = max(1, int(round(vehicle_width * severity)))
    remove_y = max(1, int(round(vehicle_height * severity)))

    left, top, right, bottom = 0, 0, image_width, image_height

    if crop_type in {"left", "top_left", "bottom_left"}:
        left = int(round(x1 + remove_x))

    if crop_type in {"right", "top_right", "bottom_right"}:
        right = int(round(x2 - remove_x))

    if crop_type in {"top", "top_left", "top_right"}:
        top = int(round(y1 + remove_y))

    if crop_type in {"bottom", "bottom_left", "bottom_right"}:
        bottom = int(round(y2 - remove_y))

    left = max(0, min(left, image_width - 2))
    top = max(0, min(top, image_height - 2))
    right = max(left + 2, min(right, image_width))
    bottom = max(top + 2, min(bottom, image_height))

    return left, top, right, bottom


def resize_to_height(image, height: int):
    if image.shape[0] == height:
        return image

    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height))


def save_preview(
    original,
    cropped,
    box: tuple[int, int, int, int],
    crop_rect: tuple[int, int, int, int],
    output_path: Path,
    crop_type: str,
    severity: float,
) -> None:
    annotated = original.copy()

    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop_rect

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)

    cv2.putText(
        annotated,
        f"{crop_type} severity={severity:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    target_height = max(annotated.shape[0], cropped.shape[0])
    left_img = resize_to_height(annotated, target_height)
    right_img = resize_to_height(cropped, target_height)

    combined = cv2.hconcat([left_img, right_img])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)


def limit_sources_per_view(
    samples: list[SourceSample],
    max_sources_per_view: Optional[int],
    seed: int,
) -> list[SourceSample]:
    if max_sources_per_view is None:
        return samples

    rng = random.Random(seed)
    selected: list[SourceSample] = []

    for view in ("front", "back", "left", "right"):
        subset = [sample for sample in samples if sample.view == view]
        rng.shuffle(subset)
        selected.extend(subset[:max_sources_per_view])

    rng.shuffle(selected)
    return selected



def parse_weighted_crop_distribution(
    crop_types: list[str],
    crop_weights: list[float],
) -> tuple[list[str], list[float]]:
    valid_crop_types = {
        "left", "right", "top", "bottom",
        "top_left", "top_right", "bottom_left", "bottom_right",
    }

    if len(crop_types) != len(crop_weights):
        raise ValueError(
            "--crop-types and --crop-weights must have the same length."
        )

    if not crop_types:
        raise ValueError("At least one crop type is required.")

    invalid = [item for item in crop_types if item not in valid_crop_types]
    if invalid:
        raise ValueError(f"Unsupported crop types: {invalid}")

    if any(weight < 0 for weight in crop_weights):
        raise ValueError("Crop weights cannot be negative.")

    if sum(crop_weights) <= 0:
        raise ValueError("Crop weights must sum to more than zero.")

    return crop_types, crop_weights


def parse_severity_distribution(
    severity_bands: list[float],
    severity_weights: list[float],
) -> tuple[list[tuple[float, float]], list[float]]:
    if len(severity_bands) % 2 != 0:
        raise ValueError(
            "--severity-bands must contain min/max pairs, for example "
            "'0.01 0.04 0.04 0.08 0.08 0.15'."
        )

    bands = [
        (severity_bands[index], severity_bands[index + 1])
        for index in range(0, len(severity_bands), 2)
    ]

    if len(bands) != len(severity_weights):
        raise ValueError(
            "The number of severity bands must equal the number of "
            "--severity-weights."
        )

    for minimum, maximum in bands:
        if minimum <= 0:
            raise ValueError("Severity minimums must be greater than zero.")
        if maximum <= minimum:
            raise ValueError(
                "Every severity maximum must be greater than its minimum."
            )

    if any(weight < 0 for weight in severity_weights):
        raise ValueError("Severity weights cannot be negative.")

    if sum(severity_weights) <= 0:
        raise ValueError("Severity weights must sum to more than zero.")

    return bands, severity_weights


def sample_targeted_crop(
    rng: random.Random,
    crop_types: list[str],
    crop_weights: list[float],
    severity_bands: list[tuple[float, float]],
    severity_weights: list[float],
) -> tuple[str, float]:
    crop_type = rng.choices(
        population=crop_types,
        weights=crop_weights,
        k=1,
    )[0]

    minimum, maximum = rng.choices(
        population=severity_bands,
        weights=severity_weights,
        k=1,
    )[0]

    severity = rng.uniform(minimum, maximum)
    return crop_type, severity


def generate(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_root}")

    selected_views = args.views or ["front", "back", "left", "right"]

    crop_types, crop_weights = parse_weighted_crop_distribution(
        args.crop_types,
        args.crop_weights,
    )

    severity_bands, severity_weights = parse_severity_distribution(
        args.severity_bands,
        args.severity_weights,
    )

    samples, skipped = discover_complete_samples(
        dataset_root,
        selected_views,
    )

    samples = limit_sources_per_view(
        samples,
        args.max_sources_per_view,
        args.seed,
    )

    if not samples:
        raise RuntimeError("No complete images were discovered.")

    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    model = YOLO(args.yolo)

    records: list[GeneratedRecord] = []
    generated = 0

    print(f"Dataset: {dataset_root}")
    print(f"Output:  {output_root}")
    print(f"Sources: {len(samples)}")
    print(f"Views:   {selected_views}")
    print(f"Device:  {args.device}")
    print(
        "Crop distribution: "
        + ", ".join(
            f"{crop_type}={weight}"
            for crop_type, weight in zip(crop_types, crop_weights)
        )
    )
    print(
        "Severity distribution: "
        + ", ".join(
            f"{minimum:.3f}-{maximum:.3f}={weight}"
            for (minimum, maximum), weight
            in zip(severity_bands, severity_weights)
        )
    )

    for sample in tqdm(samples, desc="Generating", unit="source"):
        source_path = Path(sample.path)
        image = cv2.imread(str(source_path))

        if image is None:
            records.append(
                GeneratedRecord(
                    source_path=str(source_path),
                    output_path="",
                    preview_path="",
                    view=sample.view,
                    crop_type="",
                    severity=0.0,
                    image_width=0,
                    image_height=0,
                    vehicle_x1=0.0,
                    vehicle_y1=0.0,
                    vehicle_x2=0.0,
                    vehicle_y2=0.0,
                    yolo_confidence=0.0,
                    yolo_class_id=-1,
                    status="IMAGE_READ_FAILED",
                )
            )
            continue

        image_height, image_width = image.shape[:2]
        image_area = image_width * image_height

        results = model.predict(
            source=str(source_path),
            conf=args.conf,
            classes=args.classes,
            verbose=False,
            device=args.device,
        )

        selected = select_largest_vehicle_box(results[0]) if results else None

        if selected is None:
            records.append(
                GeneratedRecord(
                    source_path=str(source_path),
                    output_path="",
                    preview_path="",
                    view=sample.view,
                    crop_type="",
                    severity=0.0,
                    image_width=image_width,
                    image_height=image_height,
                    vehicle_x1=0.0,
                    vehicle_y1=0.0,
                    vehicle_x2=0.0,
                    vehicle_y2=0.0,
                    yolo_confidence=0.0,
                    yolo_class_id=-1,
                    status="NO_VEHICLE_DETECTED",
                )
            )
            continue

        if selected["area"] / image_area < args.min_area_ratio:
            records.append(
                GeneratedRecord(
                    source_path=str(source_path),
                    output_path="",
                    preview_path="",
                    view=sample.view,
                    crop_type="",
                    severity=0.0,
                    image_width=image_width,
                    image_height=image_height,
                    vehicle_x1=selected["xyxy"][0],
                    vehicle_y1=selected["xyxy"][1],
                    vehicle_x2=selected["xyxy"][2],
                    vehicle_y2=selected["xyxy"][3],
                    yolo_confidence=selected["confidence"],
                    yolo_class_id=selected["class_id"],
                    status="VEHICLE_TOO_SMALL",
                )
            )
            continue

        x1, y1, x2, y2 = selected["xyxy"]

        x1 = max(0.0, min(x1, image_width - 1))
        y1 = max(0.0, min(y1, image_height - 1))
        x2 = max(x1 + 1, min(x2, image_width))
        y2 = max(y1 + 1, min(y2, image_height))

        for variant in range(args.variants_per_source):
            crop_type, severity = sample_targeted_crop(
                rng=rng,
                crop_types=crop_types,
                crop_weights=crop_weights,
                severity_bands=severity_bands,
                severity_weights=severity_weights,
            )

            crop_rect = compute_crop(
                image_width,
                image_height,
                (x1, y1, x2, y2),
                crop_type,
                severity,
            )

            left, top, right, bottom = crop_rect
            cropped = image[top:bottom, left:right]

            if cropped.size == 0:
                continue

            if (
                cropped.shape[0] < args.min_output_size
                or cropped.shape[1] < args.min_output_size
            ):
                continue

            filename = (
                f"{source_path.stem}"
                f"__synthetic_incomplete"
                f"__{crop_type}"
                f"__severity_{severity:.3f}"
                f"__v{variant + 1}"
                f"{source_path.suffix.lower()}"
            )

            candidate_path = (
                output_root / "candidates" / sample.view / crop_type / filename
            )
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(candidate_path), cropped)

            preview_path = Path("")
            if args.save_previews:
                preview_path = (
                    output_root / "previews" / sample.view / crop_type / filename
                )
                save_preview(
                    original=image,
                    cropped=cropped,
                    box=(
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ),
                    crop_rect=crop_rect,
                    output_path=preview_path,
                    crop_type=crop_type,
                    severity=severity,
                )

            records.append(
                GeneratedRecord(
                    source_path=str(source_path),
                    output_path=str(candidate_path),
                    preview_path=str(preview_path) if args.save_previews else "",
                    view=sample.view,
                    crop_type=crop_type,
                    severity=severity,
                    image_width=image_width,
                    image_height=image_height,
                    vehicle_x1=x1,
                    vehicle_y1=y1,
                    vehicle_x2=x2,
                    vehicle_y2=y2,
                    yolo_confidence=selected["confidence"],
                    yolo_class_id=selected["class_id"],
                    status="GENERATED",
                )
            )

            generated += 1

    manifest_path = output_root / "manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(asdict(records[0]).keys()) if records else [
            field for field in GeneratedRecord.__dataclass_fields__
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))

    with (output_root / "skipped_source_paths.txt").open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(skipped))

    print("\nDone.")
    print(f"Generated:  {generated}")
    print(f"Candidates: {output_root / 'candidates'}")
    print(f"Manifest:   {manifest_path}")

    if args.save_previews:
        print(f"Previews:   {output_root / 'previews'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic incomplete vehicle crops using YOLO."
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output",
        default="./synthetic_incomplete_candidates",
    )
    parser.add_argument(
        "--yolo",
        default=(
            "/home/aziz/Aziz/DigiCover/usingGeminiApi/"
            "models/yolov8m.pt"
        ),
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=["front", "back", "left", "right"],
        default=None,
    )
    parser.add_argument("--variants-per-source", type=int, default=1)
    parser.add_argument("--max-sources-per-view", type=int, default=None)
    parser.add_argument(
        "--crop-types",
        nargs="+",
        default=["top", "bottom", "right", "left"],
        help=(
            "Crop directions to sample. Default targets single-edge failures."
        ),
    )
    parser.add_argument(
        "--crop-weights",
        type=float,
        nargs="+",
        default=[40, 25, 20, 15],
        help=(
            "Relative weights corresponding to --crop-types. "
            "Default: top=40, bottom=25, right=20, left=15."
        ),
    )
    parser.add_argument(
        "--severity-bands",
        type=float,
        nargs="+",
        default=[0.01, 0.04, 0.04, 0.08, 0.08, 0.15],
        help=(
            "Min/max pairs for severity bands. Default bands are "
            "1-4%%, 4-8%%, and 8-15%%."
        ),
    )
    parser.add_argument(
        "--severity-weights",
        type=float,
        nargs="+",
        default=[70, 20, 10],
        help=(
            "Relative weights corresponding to severity bands. "
            "Default: 70%%, 20%%, 10%%."
        ),
    )
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[2, 5, 7],
    )
    parser.add_argument("--min-area-ratio", type=float, default=0.15)
    parser.add_argument("--min-output-size", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-previews",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.variants_per_source <= 0:
        raise ValueError("--variants-per-source must be greater than zero.")

    generate(args)
