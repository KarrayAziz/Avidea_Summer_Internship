from pathlib import Path
import csv
import shutil

from fiftyone.utils.openimages import download_open_images_split


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    "/home/aziz/Pictures/Internship_Images/is_car_openimages"
)

RAW_DIR = ROOT / "raw_openimages"
SELECTED_DIR = ROOT / "selected"

CLASSES = [
    "Motorcycle",
    "Bicycle",
    "Truck",
    "Train",
    "Boat",
    "Airplane",
]

NUM_IMAGES_PER_CLASS = 150

# Download a few extra candidates because:
# - some images may contain multiple requested classes
# - we don't want the same image copied into two class folders
DOWNLOAD_PER_CLASS = 180

SEED = 42


# ============================================================
# SETUP
# ============================================================

RAW_DIR.mkdir(parents=True, exist_ok=True)
SELECTED_DIR.mkdir(parents=True, exist_ok=True)

for class_name in CLASSES:
    (SELECTED_DIR / class_name.lower()).mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# STEP 1: DOWNLOAD USING ONE SHARED OPEN IMAGES CACHE
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — DOWNLOADING OPEN IMAGES DATA")
print("=" * 70)

for class_name in CLASSES:

    print("\n" + "=" * 70)
    print(
        f"Downloading candidates for {class_name} "
        f"(target={NUM_IMAGES_PER_CLASS})"
    )
    print("=" * 70)

    download_open_images_split(
        dataset_dir=str(RAW_DIR),
        split="train",
        version="v7",
        label_types=["detections"],
        classes=[class_name],
        max_samples=DOWNLOAD_PER_CLASS,
        shuffle=True,
        seed=SEED,
        num_workers=8,
    )


# ============================================================
# STEP 2: LOCATE OPEN IMAGES METADATA
# ============================================================

classes_csv = RAW_DIR / "metadata" / "classes.csv"
detections_csv = RAW_DIR / "labels" / "detections.csv"

if not classes_csv.exists():
    raise FileNotFoundError(
        f"Could not find class metadata:\n{classes_csv}"
    )

if not detections_csv.exists():
    raise FileNotFoundError(
        f"Could not find detection annotations:\n{detections_csv}"
    )


# ============================================================
# STEP 3: MAP CLASS NAMES -> OPEN IMAGES LABEL IDs
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — READING CLASS LABELS")
print("=" * 70)

name_to_label = {}

with open(classes_csv, "r", encoding="utf-8") as f:
    reader = csv.reader(f)

    for row in reader:
        if len(row) < 2:
            continue

        label_id = row[0].strip()
        display_name = row[1].strip()

        name_to_label[display_name] = label_id


class_label_ids = {}

for class_name in CLASSES:

    if class_name not in name_to_label:
        raise ValueError(
            f"Class '{class_name}' was not found in Open Images metadata"
        )

    class_label_ids[class_name] = name_to_label[class_name]

    print(
        f"{class_name:12} -> "
        f"{class_label_ids[class_name]}"
    )


# ============================================================
# STEP 4: READ WHICH IMAGE IDS BELONG TO EACH CLASS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — READING DETECTION ANNOTATIONS")
print("=" * 70)

label_to_class = {
    label_id: class_name
    for class_name, label_id in class_label_ids.items()
}

class_image_ids = {
    class_name: set()
    for class_name in CLASSES
}

with open(detections_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:

        image_id = row["ImageID"]
        label_id = row["LabelName"]

        if label_id in label_to_class:

            class_name = label_to_class[label_id]

            class_image_ids[class_name].add(image_id)


for class_name in CLASSES:
    print(
        f"{class_name:12}: "
        f"{len(class_image_ids[class_name])} annotated image IDs"
    )


# ============================================================
# STEP 5: INDEX DOWNLOADED IMAGE FILES
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — INDEXING DOWNLOADED IMAGES")
print("=" * 70)

image_files = {}

for path in RAW_DIR.rglob("*"):

    if not path.is_file():
        continue

    if path.suffix.lower() not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:
        continue

    # Open Images filenames normally use the ImageID
    image_files[path.stem] = path


print(f"Found {len(image_files)} downloaded image files")


# ============================================================
# STEP 6: EXPORT EXACTLY 150 UNIQUE IMAGES PER CLASS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — EXPORTING CLASS-SPECIFIC FOLDERS")
print("=" * 70)

# Prevent the same source image from appearing in two categories
used_image_ids = set()

summary = {}

for class_name in CLASSES:

    output_dir = SELECTED_DIR / class_name.lower()

    candidates = sorted(class_image_ids[class_name])

    copied = 0

    for image_id in candidates:

        if copied >= NUM_IMAGES_PER_CLASS:
            break

        # Avoid duplicate source images across classes
        if image_id in used_image_ids:
            continue

        source = image_files.get(image_id)

        if source is None:
            continue

        destination = output_dir / source.name

        shutil.copy2(
            source,
            destination,
        )

        used_image_ids.add(image_id)
        copied += 1

    summary[class_name] = copied

    print(
        f"{class_name:12}: "
        f"{copied}/{NUM_IMAGES_PER_CLASS} exported"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

total = 0

for class_name in CLASSES:

    count = summary[class_name]

    print(
        f"{class_name:12}: "
        f"{count} images"
    )

    total += count

print("-" * 70)
print(f"Total exported images: {total}")

print("\nOutput directory:")
print(SELECTED_DIR)

print("\nExpected structure:")

for class_name in CLASSES:
    print(
        f"  {class_name.lower()}/"
    )

print("\nDone.")