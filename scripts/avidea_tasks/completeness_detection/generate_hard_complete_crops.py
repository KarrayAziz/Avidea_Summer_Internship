#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck in COCO
LAYOUTS = ['tight_left', 'tight_right', 'tight_top', 'tight_bottom', 'tight_horizontal', 'tight_all']
LAYOUT_WEIGHTS = [25, 25, 15, 15, 10, 10]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate hard-complete vehicle crops.')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--yolo', type=Path,
                   default=Path('/home/aziz/Aziz/DigiCover/usingGeminiApi/models/yolov8m.pt'))
    p.add_argument('--views', nargs='+', default=['left'])
    p.add_argument('--max-sources-per-view', type=int, default=250)
    p.add_argument('--variants-per-source', type=int, default=1)
    p.add_argument('--conf', type=float, default=0.30)
    p.add_argument('--classes', nargs='+', type=int, default=DEFAULT_VEHICLE_CLASSES)
    p.add_argument('--min-area-ratio', type=float, default=0.08)
    p.add_argument('--min-output-size', type=int, default=224)
    p.add_argument('--device', default='0')
    p.add_argument('--seed', type=int, default=707)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--save-previews', action='store_true')
    return p.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def largest_vehicle_box(model: YOLO, image: np.ndarray, conf: float,
                        classes: list[int], device: str,
                        min_area_ratio: float):
    h, w = image.shape[:2]
    image_area = float(w * h)
    results = model.predict(source=image, conf=conf, classes=classes, device=device, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    best, best_area = None, 0.0
    for box in results[0].boxes.xyxy.detach().cpu().numpy():
        x1, y1, x2, y2 = map(float, box)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area / image_area < min_area_ratio:
            continue
        if area > best_area:
            best_area = area
            best = (max(0, int(np.floor(x1))), max(0, int(np.floor(y1))),
                    min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2))))
    return best


def sample_margin_fraction(rng: random.Random) -> float:
    r = rng.random()
    if r < 0.45:
        return rng.uniform(0.01, 0.03)
    if r < 0.80:
        return rng.uniform(0.03, 0.06)
    return rng.uniform(0.06, 0.12)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def create_crop(image_shape, box, layout: str, rng: random.Random, min_output_size: int):
    h, w = image_shape[:2]
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    if bw <= 0 or bh <= 0:
        return None

    # Protect against slightly inaccurate YOLO boxes.
    safety_x = max(4, int(round(bw * 0.008)))
    safety_y = max(4, int(round(bh * 0.008)))

    frac = sample_margin_fraction(rng)
    tight_x = max(safety_x, int(round(bw * frac)))
    tight_y = max(safety_y, int(round(bh * frac)))

    left = max(tight_x, int(round(bw * rng.uniform(0.12, 0.35))))
    right = max(tight_x, int(round(bw * rng.uniform(0.12, 0.35))))
    top = max(tight_y, int(round(bh * rng.uniform(0.12, 0.35))))
    bottom = max(tight_y, int(round(bh * rng.uniform(0.12, 0.35))))

    if layout == 'tight_left':
        left = tight_x
    elif layout == 'tight_right':
        right = tight_x
    elif layout == 'tight_top':
        top = tight_y
    elif layout == 'tight_bottom':
        bottom = tight_y
    elif layout == 'tight_horizontal':
        left = right = tight_x
    elif layout == 'tight_all':
        left = right = tight_x
        top = bottom = tight_y
    else:
        raise ValueError(layout)

    x1 = clamp(bx1 - left, 0, w)
    y1 = clamp(by1 - top, 0, h)
    x2 = clamp(bx2 + right, 0, w)
    y2 = clamp(by2 + bottom, 0, h)

    # Strictly keep the detected vehicle inside the crop.
    x1 = clamp(min(x1, bx1 - safety_x), 0, w)
    y1 = clamp(min(y1, by1 - safety_y), 0, h)
    x2 = clamp(max(x2, bx2 + safety_x), 0, w)
    y2 = clamp(max(y2, by2 + safety_y), 0, h)

    crop_w, crop_h = x2 - x1, y2 - y1
    if crop_w < min_output_size or crop_h < min_output_size:
        return None

    if not (x1 <= bx1 - safety_x and y1 <= by1 - safety_y and
            x2 >= bx2 + safety_x and y2 >= by2 + safety_y):
        return None

    # Skip crops that are effectively unchanged.
    if (crop_w * crop_h) / float(w * h) > 0.97:
        return None

    return x1, y1, x2, y2


def save_preview(image, box, crop, output_path: Path):
    preview = image.copy()
    bx1, by1, bx2, by2 = box
    x1, y1, x2, y2 = crop
    cv2.rectangle(preview, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.yolo.exists():
        raise FileNotFoundError(f'YOLO model not found: {args.yolo}')

    model = YOLO(str(args.yolo))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    for view in args.views:
        source_dir = args.dataset / 'train' / view / 'complete'
        if not source_dir.exists():
            print(f'[WARNING] Missing source folder: {source_dir}')
            continue

        sources = list_images(source_dir)
        # Do not recursively regenerate from earlier synthetic folders.
        sources = [p for p in sources if not any('synthetic' in part.lower() for part in p.parts)]
        rng.shuffle(sources)
        if args.max_sources_per_view > 0:
            sources = sources[:args.max_sources_per_view]

        out_dir = args.output / 'candidates' / view
        preview_dir = args.output / 'previews' / view
        out_dir.mkdir(parents=True, exist_ok=True)

        generated = no_detection = invalid = existing = 0

        for i, src in enumerate(sources, start=1):
            image = cv2.imread(str(src))
            if image is None:
                print(f'[WARNING] Could not read: {src}')
                continue

            box = largest_vehicle_box(model, image, args.conf, args.classes,
                                      args.device, args.min_area_ratio)
            if box is None:
                no_detection += 1
                continue

            for variant in range(1, args.variants_per_source + 1):
                layout = rng.choices(LAYOUTS, weights=LAYOUT_WEIGHTS, k=1)[0]
                crop_box = create_crop(image.shape, box, layout, rng, args.min_output_size)
                if crop_box is None:
                    invalid += 1
                    continue

                suffix = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTENSIONS else '.jpg'
                out = out_dir / f'{src.stem}__hard_complete__{layout}__v{variant:02d}{suffix}'
                if out.exists() and not args.overwrite:
                    existing += 1
                    continue

                x1, y1, x2, y2 = crop_box
                crop = image[y1:y2, x1:x2]
                if crop.size == 0 or not cv2.imwrite(str(out), crop):
                    invalid += 1
                    continue

                generated += 1
                if args.save_previews:
                    save_preview(image, box, crop_box,
                                 preview_dir / f'{out.stem}__preview.jpg')

                bx1, by1, bx2, by2 = box
                rows.append({
                    'view': view,
                    'source_path': str(src),
                    'output_path': str(out),
                    'layout': layout,
                    'vehicle_x1': bx1, 'vehicle_y1': by1,
                    'vehicle_x2': bx2, 'vehicle_y2': by2,
                    'crop_x1': x1, 'crop_y1': y1,
                    'crop_x2': x2, 'crop_y2': y2,
                    'source_width': image.shape[1],
                    'source_height': image.shape[0],
                    'crop_width': x2 - x1,
                    'crop_height': y2 - y1,
                })

            if i % 25 == 0:
                print(f'[{view}] processed={i}/{len(sources)} generated={generated}')

        print(f'\n[{view}] complete')
        print(f'  sources selected:       {len(sources)}')
        print(f'  generated:              {generated}')
        print(f'  skipped no detection:   {no_detection}')
        print(f'  skipped invalid crop:   {invalid}')
        print(f'  skipped existing:       {existing}')

    if rows:
        manifest = args.output / 'manifest.csv'
        with manifest.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nManifest: {manifest}')
    else:
        print('\nNo crops generated.')


if __name__ == '__main__':
    main()
