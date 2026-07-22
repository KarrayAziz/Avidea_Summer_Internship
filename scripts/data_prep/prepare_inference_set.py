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
TARGET_INFERENCE_DIR = Path("./custom_car_inference_set")
VIEWS = ["front", "back", "left", "right"]
NUM_UNSEEN_FILES = 94

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

# Create standard folder structure for the inference set
for view in VIEWS:
    (TARGET_INFERENCE_DIR / view).mkdir(parents=True, exist_ok=True)

print("\n--- Beginning Inference Crop Collection Phase ---")
valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for view in VIEWS:
    source_folder = BASE_SOURCE_DIR / view / "complete"
    print(f"\nScanning source folder for [{view.upper()}]: {source_folder}")
    
    if not source_folder.exists():
        print(f" ! Warning: Folder missing: {source_folder}. Skipping.")
        continue
        
    # 1. Gather and sort ALL files in standard ASCII order
    all_image_files = sorted([
        f for f in source_folder.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ])
    
    # 2. Slice the LAST 94 files to avoid the first 299 used in training/testing
    inference_files = all_image_files[-NUM_UNSEEN_FILES:]
    
    print(f" -> Found {len(all_image_files)} total files.")
    print(f" -> Selected the LAST {len(inference_files)} ASCII-sorted files for validation.")

    successful_crops = 0
    
    for idx, img_path in enumerate(inference_files):
        try:
            image = Image.open(img_path).convert("RGB")
            predictions = detector(image, candidate_labels=["a car"], threshold=0.15)
            box = get_largest_car_box(predictions)
            
            # Destination path inside the inference folder structure
            dest_path = TARGET_INFERENCE_DIR / view / f"crop_{idx}_{img_path.name}"
            
            if box:
                # Crop and save 
                cropped_img = image.crop(box)
                cropped_img.save(dest_path)
            else:
                # Fallback: copy full image if detector misses to keep a complete test batch
                shutil.copy(img_path, dest_path)
                
            successful_crops += 1
            if successful_crops % 20 == 0:
                print(f" -> Logged {successful_crops}/{len(inference_files)} inference images for {view}")
                
        except Exception as e:
            print(f" ! Failed to process {img_path.name}: {e}")

    print(f"✅ Completed '{view}'! Successfully processed {successful_crops} images.")

print(f"\nInference dataset successfully prepared at: {TARGET_INFERENCE_DIR.resolve()}")