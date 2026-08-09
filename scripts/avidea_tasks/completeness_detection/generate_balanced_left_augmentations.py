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
DEFAULT_CLASSES = [2, 5, 7]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate balanced hard-complete and subtle-incomplete left-view crops."
    )
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--yolo",
        type=Path,
        default=Path("/home/aziz/Aziz/DigiCover/usingGeminiApi/models/yolov8m.pt"),
    )
    p.add_argument("--view", default="left")
    p.add_argument("--target-complete", type=int, default=450)
    p.add_argument("--target-incomplete", type=int, default=450)
    p.add_argument("--hard-margin-min", type=float, default=0.008)
    p.add_argument("--hard-margin-max", type=float, default=0.025)
    p.add_argument("--subtle-severity-min", type=float, default=0.01)
    p.add_argument("--subtle-severity-max", type=float, default=0.04)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--classes", nargs="+", type=int, default=DEFAULT_CLASSES)
    p.add_argument("--min-area-ratio", type=float, default=0.08)
    p.add_argument("--min-output-size", type=int, default=224)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=909)
    p.add_argument("--save-previews", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_original_images(folder: Path):
    out = []
    for path in folder.rglob("*"):
        if not is_image(path):
            continue
        rel_parts = [part.lower() for part in path.relative_to(folder).parts]
        if any("synthetic" in part for part in rel_parts):
            continue
        out.append(path)
    return sorted(out)


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if is_image(p))


def largest_vehicle_box(model, image, conf, classes, device, min_area_ratio):
    h, w = image.shape[:2]
    image_area = float(h * w)
    results = model.predict(
        source=image,
        conf=conf,
        classes=classes,
        device=device,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    best = None
    best_area = -1.0
    for raw in results[0].boxes.xyxy.detach().cpu().numpy():
        x1, y1, x2, y2 = map(float, raw)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area / image_area < min_area_ratio:
            continue
        if area > best_area:
            best_area = area
            best = (
                max(0, int(np.floor(x1))),
                max(0, int(np.floor(y1))),
                min(w, int(np.ceil(x2))),
                min(h, int(np.ceil(y2))),
            )
    return best


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def hard_complete_crop(shape, box, rng, margin_min, margin_max, min_output_size):
    h, w = shape[:2]
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    if bw <= 0 or bh <= 0:
        return None

    mx = max(3, int(round(bw * rng.uniform(margin_min, margin_max))))
    my = max(3, int(round(bh * rng.uniform(margin_min, margin_max))))

    x1 = clamp(bx1 - mx, 0, w)
    y1 = clamp(by1 - my, 0, h)
    x2 = clamp(bx2 + mx, 0, w)
    y2 = clamp(by2 + my, 0, h)

    if x2 - x1 < min_output_size or y2 - y1 < min_output_size:
        return None
    if not (x1 <= bx1 and y1 <= by1 and x2 >= bx2 and y2 >= by2):
        return None
    if ((x2 - x1) * (y2 - y1)) / float(w * h) > 0.97:
        return None
    return x1, y1, x2, y2


def subtle_incomplete_crop(shape, box, rng, severity_min, severity_max, min_output_size):
    h, w = shape[:2]
    bx1, by1, bx2, by2 = box
    bw = bx2 - bx1
    if bw <= 0:
        return None

    severity = rng.uniform(severity_min, severity_max)
    cut = max(2, int(round(bw * severity)))
    crop_type = rng.choices(["left", "right", "both"], weights=[40, 40, 20], k=1)[0]

    x1, y1, x2, y2 = 0, 0, w, h
    if crop_type == "left":
        x1 = clamp(bx1 + cut, 0, w - 1)
    elif crop_type == "right":
        x2 = clamp(bx2 - cut, 1, w)
    else:
        half = max(1, cut // 2)
        x1 = clamp(bx1 + half, 0, w - 1)
        x2 = clamp(bx2 - half, 1, w)

    if x2 - x1 < min_output_size or y2 - y1 < min_output_size:
        return None
    if not (x1 > bx1 or x2 < bx2):
        return None

    return (x1, y1, x2, y2), crop_type, severity


def save_preview(image, box, crop, path):
    preview = image.copy()
    bx1, by1, bx2, by2 = box
    cx1, cy1, cx2, cy2 = crop
    cv2.rectangle(preview, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
    cv2.rectangle(preview, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def safe_name(source: Path, index: int, label: str, subtype: str) -> str:
    return f"{index:04d}__{source.stem.replace(' ', '_')}__{label}__{subtype}.jpg"


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    complete_dir = args.dataset / "train" / args.view / "complete"
    incomplete_dir = args.dataset / "train" / args.view / "incomplete"

    if not complete_dir.exists():
        raise FileNotFoundError(complete_dir)
    if not args.yolo.exists():
        raise FileNotFoundError(args.yolo)

    original_complete = list_original_images(complete_dir)
    current_incomplete = count_images(incomplete_dir)

    need_complete = max(0, args.target_complete - len(original_complete))
    need_incomplete = max(0, args.target_incomplete - current_incomplete)

    print(f"Original complete: {len(original_complete)}")
    print(f"Current incomplete: {current_incomplete}")
    print(f"Generate hard-complete: {need_complete}")
    print(f"Generate subtle-incomplete: {need_incomplete}")

    if args.overwrite and args.output.exists():
        shutil.rmtree(args.output)

    hard_dir = args.output / "candidates" / "complete" / args.view
    inc_dir = args.output / "candidates" / "incomplete" / args.view
    hard_prev = args.output / "previews" / "complete" / args.view
    inc_prev = args.output / "previews" / "incomplete" / args.view
    hard_dir.mkdir(parents=True, exist_ok=True)
    inc_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.yolo))
    detections = []

    sources = original_complete.copy()
    rng.shuffle(sources)

    for source in sources:
        image = cv2.imread(str(source))
        if image is None:
            continue
        box = largest_vehicle_box(
            model, image, args.conf, args.classes, args.device, args.min_area_ratio
        )
        if box is not None:
            detections.append((source, image, box))

    if not detections:
        raise RuntimeError("No valid YOLO detections found.")

    manifest = []

    generated = 0
    attempts = 0
    while generated < need_complete and attempts < max(500, need_complete * 30):
        source, image, box = detections[attempts % len(detections)]
        attempts += 1
        crop = hard_complete_crop(
            image.shape, box, rng,
            args.hard_margin_min, args.hard_margin_max,
            args.min_output_size
        )
        if crop is None:
            continue

        x1, y1, x2, y2 = crop
        out = image[y1:y2, x1:x2]
        name = safe_name(source, generated + 1, "complete", "hard")
        out_path = hard_dir / name
        if not cv2.imwrite(str(out_path), out):
            continue

        generated += 1
        if args.save_previews:
            save_preview(image, box, crop, hard_prev / name)

        manifest.append({
            "label": "complete",
            "augmentation": "hard_complete",
            "source_path": str(source),
            "output_path": str(out_path),
            "crop_type": "tight_all",
            "severity": "",
            "vehicle_box": ",".join(map(str, box)),
            "crop_box": ",".join(map(str, crop)),
        })

    generated_inc = 0
    attempts = 0
    while generated_inc < need_incomplete and attempts < max(800, need_incomplete * 30):
        source, image, box = detections[attempts % len(detections)]
        attempts += 1
        result = subtle_incomplete_crop(
            image.shape, box, rng,
            args.subtle_severity_min, args.subtle_severity_max,
            args.min_output_size
        )
        if result is None:
            continue

        crop, crop_type, severity = result
        x1, y1, x2, y2 = crop
        out = image[y1:y2, x1:x2]
        name = safe_name(source, generated_inc + 1, "incomplete", crop_type)
        out_path = inc_dir / name
        if not cv2.imwrite(str(out_path), out):
            continue

        generated_inc += 1
        if args.save_previews:
            save_preview(image, box, crop, inc_prev / name)

        manifest.append({
            "label": "incomplete",
            "augmentation": "subtle_incomplete",
            "source_path": str(source),
            "output_path": str(out_path),
            "crop_type": crop_type,
            "severity": round(severity, 6),
            "vehicle_box": ",".join(map(str, box)),
            "crop_box": ",".join(map(str, crop)),
        })

    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "label", "augmentation", "source_path", "output_path",
            "crop_type", "severity", "vehicle_box", "crop_box"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    print()
    print(f"Generated hard-complete: {generated}")
    print(f"Generated subtle-incomplete: {generated_inc}")
    print(f"Expected final complete: {len(original_complete) + generated}")
    print(f"Expected final incomplete: {current_incomplete + generated_inc}")
    print(f"Output: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
