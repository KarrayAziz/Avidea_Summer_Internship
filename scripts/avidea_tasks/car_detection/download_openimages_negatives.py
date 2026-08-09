from pathlib import Path
import fiftyone.zoo as foz

OUTPUT_ROOT = Path(
    "/home/aziz/Pictures/Internship_Images/is_car_openimages"
)

CLASSES = [
    "Bus",
    "Motorcycle",
    "Bicycle",
    "Truck",
    "Train",
    "Boat",
    "Airplane",
]

MAX_SAMPLES_PER_CLASS = 300
SPLIT = "train"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

for class_name in CLASSES:
    print("\n" + "=" * 70)
    print(f"Downloading class: {class_name}")
    print("=" * 70)

    dataset_name = (
        "openimages_"
        + class_name.lower().replace(" ", "_")
    )

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=SPLIT,
        label_types=["detections"],
        classes=[class_name],
        max_samples=MAX_SAMPLES_PER_CLASS,
        shuffle=True,
        seed=42,
        dataset_name=dataset_name,
    )

    print(
        f"{class_name}: downloaded/loaded "
        f"{len(dataset)} images"
    )

print("\nDone.")