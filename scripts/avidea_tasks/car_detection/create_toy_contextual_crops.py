from pathlib import Path
import csv
import shutil

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/data/toy_cars"
)

OUTPUT_ROOT = (
    DATASET_ROOT
    / "toy_scale_sources"
    / "toy_cars_yolo"
)

ORIGINALS_DIR = OUTPUT_ROOT / "original_images"
LABELS_DIR = OUTPUT_ROOT / "labels"
CROPS_DIR = OUTPUT_ROOT / "contextual_crops"

# Expand each side of the YOLO bounding box by 25%
CONTEXT_EXPANSION = 0.25

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

def find_images(dataset_root: Path):
    """
    Find images directly inside DATASET_ROOT.

    We intentionally do NOT recursively search because otherwise
    rerunning the script could pick up its own generated crops.
    """

    return sorted(
        p
        for p in dataset_root.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    )


def read_yolo_annotations(label_path: Path):
    """
    Read standard YOLO annotations:

        class_id center_x center_y width height

    Coordinates are normalized to [0, 1].
    """

    annotations = []

    with label_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"WARNING: malformed annotation "
                    f"{label_path}:{line_number}"
                )
                continue

            try:

                class_id = int(float(parts[0]))

                cx = float(parts[1])
                cy = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:

                print(
                    f"WARNING: invalid numbers in "
                    f"{label_path}:{line_number}"
                )
                continue

            annotations.append(
                {
                    "class_id": class_id,
                    "cx": cx,
                    "cy": cy,
                    "width": width,
                    "height": height,
                }
            )

    return annotations


def yolo_to_xyxy(
    annotation,
    image_width,
    image_height,
):
    """
    Convert normalized YOLO coordinates to absolute pixel xyxy.
    """

    cx = annotation["cx"] * image_width
    cy = annotation["cy"] * image_height

    box_width = (
        annotation["width"]
        * image_width
    )

    box_height = (
        annotation["height"]
        * image_height
    )

    x1 = cx - box_width / 2
    y1 = cy - box_height / 2

    x2 = cx + box_width / 2
    y2 = cy + box_height / 2

    return x1, y1, x2, y2


def expand_box(
    bbox,
    image_width,
    image_height,
    expansion=0.25,
):
    """
    Expand bounding box by a percentage of its width/height
    on EACH side.

    Example:
        original width = 200 px
        expansion = 0.25

        left  expands by 50 px
        right expands by 50 px

    Final width can therefore become up to 300 px,
    unless clipped by the image boundary.
    """

    x1, y1, x2, y2 = bbox

    box_width = x2 - x1
    box_height = y2 - y1

    expand_x = box_width * expansion
    expand_y = box_height * expansion

    expanded_x1 = x1 - expand_x
    expanded_y1 = y1 - expand_y

    expanded_x2 = x2 + expand_x
    expanded_y2 = y2 + expand_y

    # Clip to image boundaries

    expanded_x1 = max(
        0,
        int(round(expanded_x1)),
    )

    expanded_y1 = max(
        0,
        int(round(expanded_y1)),
    )

    expanded_x2 = min(
        image_width,
        int(round(expanded_x2)),
    )

    expanded_y2 = min(
        image_height,
        int(round(expanded_y2)),
    )

    return (
        expanded_x1,
        expanded_y1,
        expanded_x2,
        expanded_y2,
    )


# ============================================================
# PREPARE OUTPUT
# ============================================================

ORIGINALS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LABELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CROPS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATASET DISCOVERY
# ============================================================

images = find_images(DATASET_ROOT)

print("=" * 80)
print("TOY CAR CONTEXTUAL CROP GENERATION")
print("=" * 80)

print(f"\nDataset:")
print(f"  {DATASET_ROOT}")

print(f"\nOutput:")
print(f"  {OUTPUT_ROOT}")

print(
    f"\nContext expansion: "
    f"{CONTEXT_EXPANSION * 100:.0f}% per side"
)

print(f"\nFound {len(images)} images.")


# ============================================================
# STATISTICS
# ============================================================

single_car_images = 0
multi_car_images = 0

images_without_labels = 0
images_without_annotations = 0
failed_images = 0

total_annotations = 0
total_crops = 0

manifest_rows = []


# ============================================================
# PROCESS
# ============================================================

for image_index, image_path in enumerate(
    images,
    start=1,
):

    label_path = (
        DATASET_ROOT
        / f"{image_path.stem}.txt"
    )

    # --------------------------------------------------------
    # Check label exists
    # --------------------------------------------------------

    if not label_path.exists():

        print(
            f"\nWARNING: missing label for "
            f"{image_path.name}"
        )

        images_without_labels += 1
        continue

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"\nWARNING: could not read "
            f"{image_path.name}"
        )

        failed_images += 1
        continue

    image_height, image_width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # Read YOLO boxes
    # --------------------------------------------------------

    annotations = read_yolo_annotations(
        label_path
    )

    if not annotations:

        print(
            f"\nWARNING: no valid annotations in "
            f"{label_path.name}"
        )

        images_without_annotations += 1
        continue

    num_cars = len(annotations)

    total_annotations += num_cars

    if num_cars == 1:
        single_car_images += 1
    else:
        multi_car_images += 1

    # --------------------------------------------------------
    # Preserve original image + label
    # --------------------------------------------------------

    shutil.copy2(
        image_path,
        ORIGINALS_DIR / image_path.name,
    )

    shutil.copy2(
        label_path,
        LABELS_DIR / label_path.name,
    )

    # --------------------------------------------------------
    # Generate one contextual crop PER annotation
    # --------------------------------------------------------

    for car_index, annotation in enumerate(
        annotations,
        start=1,
    ):

        original_bbox = yolo_to_xyxy(
            annotation,
            image_width,
            image_height,
        )

        expanded_bbox = expand_box(
            original_bbox,
            image_width,
            image_height,
            expansion=CONTEXT_EXPANSION,
        )

        x1, y1, x2, y2 = expanded_bbox

        if x2 <= x1 or y2 <= y1:

            print(
                f"\nWARNING: invalid crop in "
                f"{image_path.name}, "
                f"car #{car_index}"
            )

            continue

        crop = image[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:

            print(
                f"\nWARNING: empty crop in "
                f"{image_path.name}, "
                f"car #{car_index}"
            )

            continue

        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        crop_filename = (
            f"{image_path.stem}"
            f"__car_{car_index:03d}"
            f"{image_path.suffix.lower()}"
        )

        crop_path = (
            CROPS_DIR
            / crop_filename
        )

        success = cv2.imwrite(
            str(crop_path),
            crop,
        )

        if not success:

            print(
                f"\nWARNING: failed to save "
                f"{crop_path}"
            )

            continue

        total_crops += 1

        # ----------------------------------------------------
        # Record metadata
        # ----------------------------------------------------

        ox1, oy1, ox2, oy2 = original_bbox

        original_box_width = (
            ox2 - ox1
        )

        original_box_height = (
            oy2 - oy1
        )

        original_box_area_ratio = (
            (
                original_box_width
                * original_box_height
            )
            /
            (
                image_width
                * image_height
            )
        )

        crop_width = x2 - x1
        crop_height = y2 - y1

        manifest_rows.append(
            {
                "source_image":
                    image_path.name,

                "source_label":
                    label_path.name,

                "source_image_id":
                    image_path.stem,

                "num_cars_in_source":
                    num_cars,

                "car_index":
                    car_index,

                "class_id":
                    annotation[
                        "class_id"
                    ],

                "yolo_center_x":
                    annotation["cx"],

                "yolo_center_y":
                    annotation["cy"],

                "yolo_width":
                    annotation["width"],

                "yolo_height":
                    annotation["height"],

                "original_image_width":
                    image_width,

                "original_image_height":
                    image_height,

                "original_bbox_x1":
                    round(ox1, 2),

                "original_bbox_y1":
                    round(oy1, 2),

                "original_bbox_x2":
                    round(ox2, 2),

                "original_bbox_y2":
                    round(oy2, 2),

                "original_bbox_area_ratio":
                    round(
                        original_box_area_ratio,
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
                    crop_width,

                "crop_height":
                    crop_height,

                "crop_filename":
                    crop_filename,

                "crop_path":
                    str(crop_path),
            }
        )

    print(
        f"\r"
        f"[{image_index:4d}/{len(images):4d}] "
        f"{image_path.name:<20} "
        f"cars={num_cars:<3} "
        f"total_crops={total_crops:<5}",
        end="",
    )


print()


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
            fieldnames=manifest_rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Images discovered          : "
    f"{len(images)}"
)

print(
    f"Single-car images          : "
    f"{single_car_images}"
)

print(
    f"Multi-car images           : "
    f"{multi_car_images}"
)

print(
    f"Total YOLO annotations     : "
    f"{total_annotations}"
)

print(
    f"Contextual crops generated : "
    f"{total_crops}"
)

print(
    f"Missing label files        : "
    f"{images_without_labels}"
)

print(
    f"Empty annotation files     : "
    f"{images_without_annotations}"
)

print(
    f"Failed images              : "
    f"{failed_images}"
)


print("\n" + "=" * 80)
print("OUTPUT STRUCTURE")
print("=" * 80)

print(
    f"""
{OUTPUT_ROOT}/
├── original_images/
├── labels/
├── contextual_crops/
└── manifest.csv
"""
)

print("Manifest:")
print(f"  {manifest_path}")

print("\nDone.")