from pathlib import Path
from collections import Counter
import shutil
import csv

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "/home/aziz/Pictures/Internship_Images/is_car_openimages/selected"
)

OUTPUT_ROOT = Path(
    "/home/aziz/Pictures/Internship_Images/is_car_openimages/"
    "yolov8m_dominant_vehicle_results"
)

MODEL_PATH = "yolov8m.pt"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Ignore vehicle detections smaller than this fraction
# of the full image area
MIN_AREA_RATIO = 0.08


# ============================================================
# COCO TRANSPORT CLASSES
# ============================================================

# COCO class IDs used by YOLOv8
#
# 1 = bicycle
# 2 = car
# 3 = motorcycle
# 4 = airplane
# 5 = bus
# 6 = train
# 7 = truck
# 8 = boat

TRANSPORT_CLASS_IDS = {
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    4,  # airplane
    5,  # bus
    6,  # train
    7,  # truck
    8,  # boat
}

# A dominant detection belonging to either of these classes
# passes Stage 1 and moves to the future is_car verification model
CAR_CANDIDATE_IDS = {
    2,  # car
    7,  # truck
}


# ============================================================
# EXPECTED FOLDER BEHAVIOR
# ============================================================

# These should pass Stage 1
EXPECTED_POSITIVE_FOLDERS = {
    "real_car",
    "truck",
}

# These should be rejected by Stage 1
EXPECTED_NEGATIVE_FOLDERS = {
    "motorcycle",
    "bicycle",
    "train",
    "boat",
    "airplane",
    "bus",
}

FOLDERS = [
    "real_car",
    "truck",
    "motorcycle",
    "bicycle",
    "train",
    "boat",
    "airplane",
    "bus",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_images(folder: Path):
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def area_ratio(bbox, image_width, image_height):
    """
    Returns bounding-box area as fraction of full image area.
    """

    x1, y1, x2, y2 = bbox

    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)

    box_area = box_width * box_height
    image_area = image_width * image_height

    if image_area <= 0:
        return 0.0

    return box_area / image_area


def draw_detections(
    image,
    detections,
    dominant_detection=None,
):
    """
    Draw all valid transport detections.

    Dominant detection:
        thick rectangle

    Other valid transport detections:
        thinner rectangle
    """

    output = image.copy()

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        is_dominant = (
            dominant_detection is not None
            and det is dominant_detection
        )

        thickness = 4 if is_dominant else 2

        # White for normal boxes, green for dominant.
        # Colors are only for debug visualization.
        color = (
            (0, 255, 0)
            if is_dominant
            else (255, 255, 255)
        )

        prefix = "DOMINANT" if is_dominant else ""

        label = (
            f"{prefix} "
            f"{det['class_name']} "
            f"{det['confidence']:.2f} "
            f"area={det['area_ratio'] * 100:.1f}%"
        ).strip()

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        cv2.putText(
            output,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 85)
print("YOLOv8m DOMINANT VEHICLE FILTER EVALUATION")
print("=" * 85)

print(f"\nDataset       : {DATASET_ROOT}")
print(f"Model         : {MODEL_PATH}")
print(f"Confidence    : {CONF_THRESHOLD}")
print(f"Min area ratio: {MIN_AREA_RATIO:.2f} ({MIN_AREA_RATIO * 100:.0f}%)")

model = YOLO(MODEL_PATH)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

FAILURE_ROOT = OUTPUT_ROOT / "failures"

FALSE_POSITIVE_ROOT = FAILURE_ROOT / "false_positives"
FALSE_NEGATIVE_ROOT = FAILURE_ROOT / "false_negatives"

FALSE_POSITIVE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

FALSE_NEGATIVE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RESULTS
# ============================================================

folder_results = {}
prediction_rows = []


# ============================================================
# EVALUATION
# ============================================================

for folder_name in FOLDERS:

    folder_path = DATASET_ROOT / folder_name

    if not folder_path.exists():
        print(f"\nWARNING: folder does not exist:")
        print(folder_path)
        continue

    images = get_images(folder_path)

    expected_positive = (
        folder_name in EXPECTED_POSITIVE_FOLDERS
    )

    print("\n" + "=" * 85)
    print(f"FOLDER: {folder_name}")

    if expected_positive:
        print("EXPECTED: dominant vehicle = CAR or TRUCK")
    else:
        print("EXPECTED: dominant vehicle != CAR/TRUCK")

    print(f"Images: {len(images)}")
    print("=" * 85)

    correct = 0
    incorrect = 0

    false_positives = 0
    false_negatives = 0

    no_valid_vehicle = 0

    dominant_class_counter = Counter()

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        # ----------------------------------------------------
        # Read image to know dimensions
        # ----------------------------------------------------

        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"\nWARNING: could not read {image_path}"
            )
            continue

        image_height, image_width = image.shape[:2]

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        result = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=list(TRANSPORT_CLASS_IDS),
            verbose=False,
            device="cpu",
        )[0]

        all_transport_detections = []
        valid_transport_detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls.item()
                )

                confidence = float(
                    box.conf.item()
                )

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )

                bbox = (
                    x1,
                    y1,
                    x2,
                    y2,
                )

                ratio = area_ratio(
                    bbox,
                    image_width,
                    image_height,
                )

                detection = {
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": confidence,
                    "bbox": bbox,
                    "area_ratio": ratio,
                }

                all_transport_detections.append(
                    detection
                )

                # --------------------------------------------
                # Apply 8% threshold
                # --------------------------------------------

                if ratio >= MIN_AREA_RATIO:

                    valid_transport_detections.append(
                        detection
                    )

        # ----------------------------------------------------
        # Choose dominant transport object
        # ----------------------------------------------------

        dominant_detection = None

        if valid_transport_detections:

            dominant_detection = max(
                valid_transport_detections,
                key=lambda d: d["area_ratio"],
            )

            dominant_class_name = (
                dominant_detection["class_name"]
            )

            dominant_class_id = (
                dominant_detection["class_id"]
            )

            dominant_class_counter[
                dominant_class_name
            ] += 1

            predicted_positive = (
                dominant_class_id
                in CAR_CANDIDATE_IDS
            )

        else:

            dominant_class_name = "none"
            dominant_class_id = -1
            predicted_positive = False

            dominant_class_counter["none"] += 1
            no_valid_vehicle += 1

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        is_correct = (
            predicted_positive
            == expected_positive
        )

        if is_correct:
            correct += 1
        else:
            incorrect += 1

        # ----------------------------------------------------
        # FALSE POSITIVE
        #
        # Example:
        # motorcycle folder
        # dominant object -> car
        # ----------------------------------------------------

        if (
            not expected_positive
            and predicted_positive
        ):

            false_positives += 1

            fp_dir = (
                FALSE_POSITIVE_ROOT
                / folder_name
            )

            originals_dir = (
                fp_dir / "original"
            )

            annotated_dir = (
                fp_dir / "annotated"
            )

            originals_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            annotated_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                image_path,
                originals_dir / image_path.name,
            )

            annotated = draw_detections(
                image,
                valid_transport_detections,
                dominant_detection,
            )

            cv2.imwrite(
                str(
                    annotated_dir
                    / image_path.name
                ),
                annotated,
            )

        # ----------------------------------------------------
        # FALSE NEGATIVE
        #
        # Example:
        # real_car folder
        # dominant object absent or motorcycle/etc.
        # ----------------------------------------------------

        if (
            expected_positive
            and not predicted_positive
        ):

            false_negatives += 1

            fn_dir = (
                FALSE_NEGATIVE_ROOT
                / folder_name
            )

            originals_dir = (
                fn_dir / "original"
            )

            annotated_dir = (
                fn_dir / "annotated"
            )

            originals_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            annotated_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                image_path,
                originals_dir / image_path.name,
            )

            annotated = draw_detections(
                image,
                valid_transport_detections,
                dominant_detection,
            )

            cv2.imwrite(
                str(
                    annotated_dir
                    / image_path.name
                ),
                annotated,
            )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        all_classes = ";".join(
            d["class_name"]
            for d in all_transport_detections
        )

        all_confidences = ";".join(
            f"{d['confidence']:.6f}"
            for d in all_transport_detections
        )

        all_area_ratios = ";".join(
            f"{d['area_ratio']:.6f}"
            for d in all_transport_detections
        )

        valid_classes = ";".join(
            d["class_name"]
            for d in valid_transport_detections
        )

        prediction_rows.append(
            {
                "folder": folder_name,
                "filename": image_path.name,

                "expected_car_candidate":
                    int(expected_positive),

                "predicted_car_candidate":
                    int(predicted_positive),

                "correct":
                    int(is_correct),

                "dominant_class":
                    dominant_class_name,

                "dominant_class_id":
                    dominant_class_id,

                "dominant_confidence":
                    (
                        dominant_detection[
                            "confidence"
                        ]
                        if dominant_detection
                        else ""
                    ),

                "dominant_area_ratio":
                    (
                        dominant_detection[
                            "area_ratio"
                        ]
                        if dominant_detection
                        else ""
                    ),

                "num_all_transport_detections":
                    len(
                        all_transport_detections
                    ),

                "num_valid_transport_detections":
                    len(
                        valid_transport_detections
                    ),

                "all_transport_classes":
                    all_classes,

                "all_transport_confidences":
                    all_confidences,

                "all_transport_area_ratios":
                    all_area_ratios,

                "valid_transport_classes":
                    valid_classes,
            }
        )

        print(
            f"\r"
            f"[{index:4d}/{len(images):4d}] "
            f"correct={correct:4d} "
            f"errors={incorrect:4d}",
            end="",
        )

    print()

    accuracy = (
        correct / len(images)
        if images
        else 0.0
    )

    folder_results[folder_name] = {
        "total": len(images),
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "no_valid_vehicle": no_valid_vehicle,
        "dominant_classes": dominant_class_counter,
    }


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_ROOT
    / "predictions.csv"
)

if prediction_rows:

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=prediction_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            prediction_rows
        )


# ============================================================
# PER-FOLDER REPORT
# ============================================================

print("\n")
print("=" * 95)
print("PER-FOLDER RESULTS")
print("=" * 95)

print(
    f"{'Folder':<15}"
    f"{'N':>8}"
    f"{'Correct':>10}"
    f"{'Errors':>10}"
    f"{'Accuracy':>12}"
    f"{'FP':>8}"
    f"{'FN':>8}"
    f"{'No valid':>12}"
)

print("-" * 95)

for folder_name, stats in folder_results.items():

    print(
        f"{folder_name:<15}"
        f"{stats['total']:>8}"
        f"{stats['correct']:>10}"
        f"{stats['incorrect']:>10}"
        f"{stats['accuracy'] * 100:>11.2f}%"
        f"{stats['false_positives']:>8}"
        f"{stats['false_negatives']:>8}"
        f"{stats['no_valid_vehicle']:>12}"
    )


# ============================================================
# DOMINANT CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 95)
print("DOMINANT YOLO VEHICLE CLASS BY FOLDER")
print("=" * 95)

for folder_name, stats in folder_results.items():

    print(f"\n{folder_name.upper()}")

    counter = stats["dominant_classes"]

    total = stats["total"]

    for class_name, count in counter.most_common():

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"  {class_name:<12}: "
            f"{count:>4} "
            f"({percentage:>6.2f}%)"
        )


# ============================================================
# OVERALL REPORT
# ============================================================

total_images = sum(
    stats["total"]
    for stats in folder_results.values()
)

total_correct = sum(
    stats["correct"]
    for stats in folder_results.values()
)

total_fp = sum(
    stats["false_positives"]
    for stats in folder_results.values()
)

total_fn = sum(
    stats["false_negatives"]
    for stats in folder_results.values()
)

overall_accuracy = (
    total_correct / total_images
    if total_images
    else 0.0
)


print("\n")
print("=" * 95)
print("OVERALL RESULTS")
print("=" * 95)

print(
    f"Total images       : {total_images}"
)

print(
    f"Correct predictions: {total_correct}"
)

print(
    f"Errors             : "
    f"{total_images - total_correct}"
)

print(
    f"Overall accuracy   : "
    f"{overall_accuracy * 100:.2f}%"
)

print()

print(
    f"False positives    : {total_fp}"
)

print(
    f"False negatives    : {total_fn}"
)


# ============================================================
# NEGATIVE CLASS FALSE ACCEPTANCE RATE
# ============================================================

print("\n")
print("=" * 95)
print("NEGATIVE-CLASS FALSE CAR/TRUCK ACCEPTANCE RATE")
print("=" * 95)

for folder_name in FOLDERS:

    if folder_name not in EXPECTED_NEGATIVE_FOLDERS:
        continue

    if folder_name not in folder_results:
        continue

    stats = folder_results[folder_name]

    rate = (
        stats["false_positives"]
        / stats["total"]
        if stats["total"]
        else 0.0
    )

    print(
        f"{folder_name:<15}: "
        f"{stats['false_positives']:>3}/"
        f"{stats['total']:<3} "
        f"({rate * 100:.2f}%)"
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n")
print("=" * 95)
print("OUTPUT")
print("=" * 95)

print("\nPredictions CSV:")
print(f"  {csv_path}")

print("\nFalse positives:")
print(f"  {FALSE_POSITIVE_ROOT}")

print("\nFalse negatives:")
print(f"  {FALSE_NEGATIVE_ROOT}")

print("\nDone.")