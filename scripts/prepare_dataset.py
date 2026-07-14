import os
import shutil
from pathlib import Path
from PIL import Image
from transformers import pipeline

print("Initializing OWL-ViT detector to handle background extraction...")
detector = pipeline(
    model="google/owlvit-base-patch32", 
    task="zero-shot-object-detection"
)

# Configuration
BASE_SOURCE_DIR = Path("/home/aziz/Pictures/Internship_Images/detetction des faces")
TARGET_DATASET_DIR = Path("./custom_car_dataset")
VIEWS = ["front", "back", "left", "right"]
CROPS_PER_CLASS = 500
TRAIN_SPLIT_COUNT = 400  # 400 for train, remaining 100 for validation

def get_largest_car_box(predictions):
    largest_area = 0
    best_box = None
    for pred in predictions:
        if pred["label"] == "a car":
            box = pred["box"]
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            area = (xmax - xmin) * (ymax - ymin)
            if area > largest_area:
                largest_area = area
                best_box = [xmin, ymin, xmax, ymax]
    return best_box

# Create standard PyTorch folder hierarchy
for split in ["train", "val"]:
    for view in VIEWS:
        (TARGET_DATASET_DIR / split / view).mkdir(parents=True, exist_ok=True)

print("\n--- Beginning Crop Collection Phase ---")
valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for view in VIEWS:
    source_folder = BASE_SOURCE_DIR / view / "complete"
    print(f"\nScanning source folder for [{view.upper()}]: {source_folder}")
    
    if not source_folder.exists():
        print(f" ! Warning: Folder missing: {source_folder}. Skipping.")
        continue
        
    image_files = sorted([
        f for f in source_folder.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ])
    
    successful_crops = 0
    
    for img_path in image_files:
        if successful_crops >= CROPS_PER_CLASS:
            break
            
        try:
            image = Image.open(img_path).convert("RGB")
            predictions = detector(image, candidate_labels=["a car"], threshold=0.15)
            box = get_largest_car_box(predictions)
            
            if box:
                # Determine destination sub-folder based on split
                split_assignment = "train" if successful_crops < TRAIN_SPLIT_COUNT else "val"
                dest_path = TARGET_DATASET_DIR / split_assignment / view / f"crop_{successful_crops}_{img_path.name}"
                
                # Crop and save 
                cropped_img = image.crop(box)
                cropped_img.save(dest_path)
                successful_crops += 1
                
                if successful_crops % 20 == 0:
                    print(f" -> Logged {successful_crops}/{CROPS_PER_CLASS} crops for {view}")
            else:
                # Fallback: if detection completely misses but image is high priority, use full image
                pass
                
        except Exception as e:
            print(f" ! Failed to process {img_path.name}: {e}")

    print(f"✅ Class Completed! Collected total of {successful_crops} crops for '{view}'.")

print(f"\nDataset fully prepared and located at: {TARGET_DATASET_DIR.resolve()}")