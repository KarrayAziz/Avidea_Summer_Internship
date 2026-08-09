#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, random, shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = [2, 5, 7]

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--yolo", type=Path, default=Path("/home/aziz/Aziz/DigiCover/usingGeminiApi/models/yolov8m.pt"))
    p.add_argument("--target-count", type=int, default=62)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--min-area-ratio", type=float, default=0.08)
    p.add_argument("--min-output-size", type=int, default=224)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=1202)
    p.add_argument("--save-previews", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def images(folder: Path):
    out = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            rel = [x.lower() for x in p.relative_to(folder).parts]
            if not any("synthetic" in x for x in rel):
                out.append(p)
    return sorted(out)

def largest_box(model, image, conf, device, min_area_ratio):
    h, w = image.shape[:2]
    results = model.predict(source=image, conf=conf, classes=CLASSES, device=device, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None
    best, best_area = None, -1.0
    for raw in results[0].boxes.xyxy.detach().cpu().numpy():
        x1, y1, x2, y2 = map(float, raw)
        area = max(0.0, x2-x1) * max(0.0, y2-y1)
        if area / float(h*w) < min_area_ratio:
            continue
        if area > best_area:
            best_area = area
            best = (max(0,int(np.floor(x1))), max(0,int(np.floor(y1))),
                    min(w,int(np.ceil(x2))), min(h,int(np.ceil(y2))))
    return best

def severity(rng):
    r = rng.random()
    if r < 0.65:
        return rng.uniform(0.01, 0.04), "01_04"
    if r < 0.90:
        return rng.uniform(0.04, 0.08), "04_08"
    return rng.uniform(0.08, 0.15), "08_15"

def make_crop(shape, box, rng, min_size):
    h, w = shape[:2]
    bx1, _, bx2, _ = box
    bw = bx2 - bx1
    if bw <= 0:
        return None

    crop_type = rng.choices(["right_front", "left_rear", "both"], weights=[45,35,20], k=1)[0]
    sev, band = severity(rng)
    cut = max(2, int(round(bw * sev)))
    x1, x2 = 0, w

    if crop_type == "right_front":
        x2 = max(1, min(w, bx2 - cut))
    elif crop_type == "left_rear":
        x1 = max(0, min(w-1, bx1 + cut))
    else:
        left_cut = max(1, cut // 2)
        right_cut = max(1, cut - left_cut)
        x1 = max(0, min(w-1, bx1 + left_cut))
        x2 = max(1, min(w, bx2 - right_cut))

    if x2-x1 < min_size or h < min_size:
        return None
    if crop_type == "right_front" and not x2 < bx2:
        return None
    if crop_type == "left_rear" and not x1 > bx1:
        return None
    if crop_type == "both" and not (x1 > bx1 and x2 < bx2):
        return None

    return (x1, 0, x2, h), crop_type, sev, band

def preview(image, vehicle_box, crop_box, path):
    img = image.copy()
    x1,y1,x2,y2 = vehicle_box
    cx1,cy1,cx2,cy2 = crop_box
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
    cv2.rectangle(img, (cx1,cy1), (cx2,cy2), (0,0,255), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

def main():
    a = args()
    rng = random.Random(a.seed)
    source_dir = a.dataset / "train" / "right" / "complete"
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not a.yolo.exists():
        raise FileNotFoundError(a.yolo)

    if a.overwrite and a.output.exists():
        shutil.rmtree(a.output)

    cand = a.output / "candidates" / "right"
    prev = a.output / "previews" / "right"
    cand.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(a.yolo))
    srcs = images(source_dir)
    rng.shuffle(srcs)

    detections = []
    for src in srcs:
        img = cv2.imread(str(src))
        if img is None:
            continue
        box = largest_box(model, img, a.conf, a.device, a.min_area_ratio)
        if box is not None:
            detections.append((src, img, box))

    rows = []
    generated = 0
    attempts = 0
    max_attempts = max(500, a.target_count * 30)

    while generated < a.target_count and attempts < max_attempts:
        src, img, box = detections[attempts % len(detections)]
        attempts += 1
        made = make_crop(img.shape, box, rng, a.min_output_size)
        if made is None:
            continue
        crop_box, crop_type, sev, band = made
        x1,y1,x2,y2 = crop_box
        cropped = img[y1:y2, x1:x2]
        name = f"{generated+1:04d}__{src.stem}__{crop_type}__{band}.jpg"
        out = cand / name
        if not cv2.imwrite(str(out), cropped):
            continue
        generated += 1
        if a.save_previews:
            preview(img, box, crop_box, prev / name)
        rows.append({
            "source_path": str(src),
            "output_path": str(out),
            "crop_type": crop_type,
            "severity": round(sev, 6),
            "severity_band": band,
            "vehicle_box": ",".join(map(str, box)),
            "crop_box": ",".join(map(str, crop_box)),
        })

    manifest = a.output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        fields = ["source_path","output_path","crop_type","severity","severity_band","vehicle_box","crop_box"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Generated: {generated}")
    print(f"Candidates: {cand}")
    print(f"Manifest: {manifest}")

if __name__ == "__main__":
    main()
