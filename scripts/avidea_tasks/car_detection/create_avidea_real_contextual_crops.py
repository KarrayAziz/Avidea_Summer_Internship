from pathlib import Path
import csv
import shutil

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "/home/aziz/Pictures/Internship_Images/"
    "no_duplicates_detection_des faces_500"
)

OUTPUT_ROOT = (
    DATASET_ROOT
    / "real_scale_sources"
    / "avidea_real_yolo"
)

ORIGINALS_DIR = OUTPUT_ROOT / "original_images"
CROPS_DIR = OUTPUT_ROOT / "contextual_crops"

MODEL_PATH = "yolov8m.pt"

CONF_THRESHOLD = 0.25

# COCO:
# 2 = car
# 7 = truck
TARGET_CLASS_IDS = [2, 7]

# Same minimum area rule used in your other pipeline
MIN_AREA_RATIO = 0.08

# Same contextual expansion as toy dataset
CONTEXT_EXPANSION = 0.25

VIEWS = [
    "back",
    "front",
    "left",
    "right",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def get_images(folder: Path):
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_largest_valid_vehicle(
    result,
    image_width,
    image_height,
):
    """
    Select largest YOLO car/truck box whose area is
    at least MIN_AREA_RATIO of the whole image.
    """

    if result.boxes is None or len(result.boxes) == 0:
        return None

    image_area = image_width * image_height

    candidates = []

    for box in result.boxes:

        class_id = int(box.cls.item())

        if class_id not in TARGET_CLASS_IDS:
            continue

        confidence = float(box.conf.item())

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .tolist()
        )

        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)

        area = box_width * box_height

        area_ratio = (
            area / image_area
            if image_area > 0
            else 0.0
        )

        if area_ratio < MIN_AREA_RATIO:
            continue

        candidates.append(
            {
                "class_id": class_id,
                "confidence": confidence,
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
                "area": area,
                "area_ratio": area_ratio,
            }
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda d: d["area"],
    )


def expand_box(
    bbox,
    image_width,
    image_height,
    expansion=0.25,
):
    """
    Expand bbox by 25% of bbox width/height on EACH side.
    """

    x1, y1, x2, y2 = bbox

    box_width = x2 - x1
    box_height = y2 - y1

    expand_x = box_width * expansion
    expand_y = box_height * expansion

    new_x1 = int(
        round(x1 - expand_x)
    )

    new_y1 = int(
        round(y1 - expand_y)
    )

    new_x2 = int(
        round(x2 + expand_x)
    )

    new_y2 = int(
        round(y2 + expand_y)
    )

    # Clip to image boundaries
    new_x1 = max(
        0,
        new_x1,
    )

    new_y1 = max(
        0,
        new_y1,
    )

    new_x2 = min(
        image_width,
        new_x2,
    )

    new_y2 = min(
        image_height,
        new_y2,
    )

    return (
        new_x1,
        new_y1,
        new_x2,
        new_y2,
    )


# ============================================================
# INITIALIZE MODEL
# ============================================================

print("=" * 85)
print("AVIDEA REAL-CAR CONTEXTUAL CROP GENERATION")
print("=" * 85)

print(f"\nDataset:")
print(f"  {DATASET_ROOT}")

print(f"\nOutput:")
print(f"  {OUTPUT_ROOT}")

print(f"\nModel:")
print(f"  {MODEL_PATH}")

print(
    f"\nContext expansion: "
    f"{CONTEXT_EXPANSION * 100:.0f}% per side"
)

print(
    f"Minimum YOLO area: "
    f"{MIN_AREA_RATIO * 100:.0f}%"
)

model = YOLO(MODEL_PATH)


# ============================================================
# PREPARE OUTPUT
# ============================================================

for view in VIEWS:

    (ORIGINALS_DIR / view).mkdir(
        parents=True,
        exist_ok=True,
    )

    (CROPS_DIR / view).mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# STATISTICS
# ============================================================

manifest_rows = []
skipped_rows = []

total_images = 0
total_crops = 0

view_stats = {}


# ============================================================
# PROCESS EACH VIEW
# ============================================================

for view in VIEWS:

    input_dir = DATASET_ROOT / view

    if not input_dir.exists():

        print(
            f"\nWARNING: missing folder: "
            f"{input_dir}"
        )

        continue

    images = get_images(
        input_dir
    )

    total_images += len(images)

    view_generated = 0
    view_skipped = 0

    print("\n" + "=" * 85)
    print(
        f"VIEW: {view.upper()} "
        f"({len(images)} images)"
    )
    print("=" * 85)

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            skipped_rows.append(
                {
                    "source_image":
                        image_path.name,

                    "view":
                        view,

                    "reason":
                        "IMAGE_READ_FAILED",
                }
            )

            view_skipped += 1
            continue

        image_height, image_width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        result = model.predict(
            source=str(image_path),

            conf=CONF_THRESHOLD,

            classes=TARGET_CLASS_IDS,

            verbose=False,

            device="cpu",
        )[0]

        detection = (
            select_largest_valid_vehicle(
                result,
                image_width,
                image_height,
            )
        )

        # ----------------------------------------------------
        # No usable vehicle
        # ----------------------------------------------------

        if detection is None:

            skipped_rows.append(
                {
                    "source_image":
                        image_path.name,

                    "view":
                        view,

                    "reason":
                        "NO_VALID_CAR_TRUCK_DETECTION",
                }
            )

            view_skipped += 1

            print(
                f"\r"
                f"[{index:4d}/{len(images):4d}] "
                f"generated={view_generated:4d} "
                f"skipped={view_skipped:3d}",
                end="",
            )

            continue

        # ----------------------------------------------------
        # Expand box
        # ----------------------------------------------------

        original_bbox = (
            detection["bbox"]
        )

        crop_bbox = expand_box(
            original_bbox,
            image_width,
            image_height,
            expansion=CONTEXT_EXPANSION,
        )

        x1, y1, x2, y2 = (
            crop_bbox
        )

        crop = image[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:

            skipped_rows.append(
                {
                    "source_image":
                        image_path.name,

                    "view":
                        view,

                    "reason":
                        "EMPTY_CONTEXT_CROP",
                }
            )

            view_skipped += 1
            continue

        # ----------------------------------------------------
        # Preserve original
        # ----------------------------------------------------

        original_destination = (
            ORIGINALS_DIR
            / view
            / image_path.name
        )

        shutil.copy2(
            image_path,
            original_destination,
        )

        # ----------------------------------------------------
        # Save contextual crop
        # ----------------------------------------------------

        crop_filename = (
            f"{image_path.stem}"
            f"__context25"
            f"{image_path.suffix.lower()}"
        )

        crop_path = (
            CROPS_DIR
            / view
            / crop_filename
        )

        success = cv2.imwrite(
            str(crop_path),
            crop,
        )

        if not success:

            skipped_rows.append(
                {
                    "source_image":
                        image_path.name,

                    "view":
                        view,

                    "reason":
                        "CROP_SAVE_FAILED",
                }
            )

            view_skipped += 1
            continue

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        ox1, oy1, ox2, oy2 = (
            original_bbox
        )

        manifest_rows.append(
            {
                "source_image":
                    image_path.name,

                "source_image_id":
                    image_path.stem,

                "view":
                    view,

                "source_dataset":
                    "avidea",

                "label":
                    "real",

                "yolo_class_id":
                    detection["class_id"],

                "yolo_class_name":
                    model.names[
                        detection["class_id"]
                    ],

                "yolo_confidence":
                    round(
                        detection[
                            "confidence"
                        ],
                        6,
                    ),

                "original_image_width":
                    image_width,

                "original_image_height":
                    image_height,

                "original_bbox_x1":
                    round(
                        ox1,
                        2,
                    ),

                "original_bbox_y1":
                    round(
                        oy1,
                        2,
                    ),

                "original_bbox_x2":
                    round(
                        ox2,
                        2,
                    ),

                "original_bbox_y2":
                    round(
                        oy2,
                        2,
                    ),

                "original_bbox_area_ratio":
                    round(
                        detection[
                            "area_ratio"
                        ],
                        6,
                    ),

                "context_expansion":
                    CONTEXT_EXPANSION,

                "crop_x1":
                    x1,

                "crop_y1":
                    y1,

                "crop_x2":
                    x2,

                "crop_y2":
                    y2,

                "crop_width":
                    x2 - x1,

                "crop_height":
                    y2 - y1,

                "crop_filename":
                    crop_filename,

                "crop_path":
                    str(
                        crop_path
                    ),
            }
        )

        view_generated += 1
        total_crops += 1

        print(
            f"\r"
            f"[{index:4d}/{len(images):4d}] "
            f"generated={view_generated:4d} "
            f"skipped={view_skipped:3d}",
            end="",
        )

    print()

    view_stats[view] = {
        "images": len(images),
        "generated": view_generated,
        "skipped": view_skipped,
    }


# ============================================================
# WRITE MANIFEST
# ============================================================

manifest_path = (
    OUTPUT_ROOT
    / "manifest.csv"
)

if manifest_rows:

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                manifest_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )


# ============================================================
# WRITE SKIPPED MANIFEST
# ============================================================

skipped_path = (
    OUTPUT_ROOT
    / "skipped.csv"
)

if skipped_rows:

    with skipped_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_image",
                "view",
                "reason",
            ],
        )

        writer.writeheader()
        writer.writerows(
            skipped_rows
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 85)
print("SUMMARY")
print("=" * 85)

print(
    f"{'View':<12}"
    f"{'Images':>10}"
    f"{'Generated':>12}"
    f"{'Skipped':>10}"
)

print("-" * 44)

for view in VIEWS:

    if view not in view_stats:
        continue

    stats = view_stats[
        view
    ]

    print(
        f"{view:<12}"
        f"{stats['images']:>10}"
        f"{stats['generated']:>12}"
        f"{stats['skipped']:>10}"
    )

print("-" * 44)

print(
    f"{'TOTAL':<12}"
    f"{total_images:>10}"
    f"{total_crops:>12}"
    f"{len(skipped_rows):>10}"
)

print("\nManifest:")
print(
    f"  {manifest_path}"
)

print("\nSkipped:")
print(
    f"  {skipped_path}"
)

print("\nOutput crops:")
print(
    f"  {CROPS_DIR}"
)

print("\nDone.")