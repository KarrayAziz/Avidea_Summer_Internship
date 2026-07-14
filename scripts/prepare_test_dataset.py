import os
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
TARGET_TEST_DIR = Path("./test_dataset_unseen")
VIEWS = ["front", "back", "left", "right"]
CROPS_TO_COLLECT = 200

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

# Create standard PyTorch folder hierarchy for test split
for view in VIEWS:
    (TARGET_TEST_DIR / view).mkdir(parents=True, exist_ok=True)

print("\n--- Beginning Crop Collection Phase for UNSEEN Test Images ---")
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
    
    # CRITICAL CHANGE: Grab only the last 200 images alphabetically!
    if len(image_files) > CROPS_TO_COLLECT:
        target_images = image_files[-CROPS_TO_COLLECT:]
        print(f" -> Selected the LAST {CROPS_TO_COLLECT} files from the {len(image_files)} available.")
    else:
        target_images = image_files
        print(f" -> ! Warning: Only {len(image_files)} files found. Collecting all of them.")

    successful_crops = 0
    
    for img_path in target_images:
        try:
            image = Image.open(img_path).convert("RGB")
            predictions = detector(image, candidate_labels=["a car"], threshold=0.15)
            box = get_largest_car_box(predictions)
            
            if box:
                dest_path = TARGET_TEST_DIR / view / f"unseen_crop_{successful_crops}_{img_path.name}"
                
                # Crop and save 
                cropped_img = image.crop(box)
                cropped_img.save(dest_path)
                successful_crops += 1
                
                if successful_crops % 40 == 0:
                    print(f" -> Logged {successful_crops} unseen crops for {view}")
            else:
                pass
                
        except Exception as e:
            print(f" ! Failed to process {img_path.name}: {e}")

    print(f"✅ Class Completed! Collected total of {successful_crops} unseen test crops for '{view}'.")

print(f"\nUnseen test dataset fully prepared and located at: {TARGET_TEST_DIR.resolve()}")