import os
import random
import shutil
import numpy as np
from glob import glob
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

def split_and_crop_dataset(root_dir, train_count=240, val_count=59, confidence_threshold=0.30):
    root_path = Path(root_dir)
    # Creating a dedicated folder for the cropped pipeline alongside your input folder
    output_path = root_path.parent / "dataset_split_cropped"
    
    categories = ['back', 'front', 'left', 'right']
    splits = ['train', 'val', 'test']
    
    # 1. Initialize target directory layout
    for split in splits:
        for category in categories:
            (output_path / split / category).mkdir(parents=True, exist_ok=True)
            
    print(f"🚀 Loading YOLO model for preprocessing...")
    yolo_model = YOLO("yolov8m.pt")
    
    print(f"📁 Target cropped split directory: {output_path}")
    print("-" * 60)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for cat in categories:
        cat_dir = root_path / cat
        if not cat_dir.exists():
            print(f"⚠️ Warning: Class folder '{cat}' not found at {cat_dir}. Skipping.")
            continue
            
        # Gather images matching extension profile
        all_images = [
            f for f in cat_dir.glob("*") 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        
        # Keep split distributions identical using fixed random seed shuffling
        random.shuffle(all_images)
        total_imgs = len(all_images)
        
        target_total = train_count + val_count
        if total_imgs < target_total:
            print(f"❌ Error: Class '{cat}' only has {total_imgs} images. Need at least {target_total}!")
            continue
            
        # Segment images using the exact quotas
        split_assignments = {
            'train': all_images[:train_count],
            'val': all_images[train_count:target_total],
            'test': all_images[target_total:]
        }
        
        print(f"📦 Processing '{cat.upper()}': Train={train_count} | Val={val_count} | Test={len(split_assignments['test'])}")
        
        # 2. Process, Crop, and Route Images into Target Sets
        for split_name, img_list in split_assignments.items():
            desc_msg = f"  └─ [{split_name.upper()}] Cropping"
            for img_path in tqdm(img_list, desc=desc_msg, leave=False, unit="img"):
                try:
                    # Open pristine source image
                    img = Image.open(img_path).convert("RGB")
                    img_w, img_h = img.size
                    total_image_area = img_w * img_h
                    
                    # Target cars (2), buses (5), and trucks (7) to capture wide SUVs/vans
                    yolo_results = yolo_model(img_path, conf=confidence_threshold, classes=[2, 5, 7], verbose=False)
                    
                    use_full_image = True
                    dest_save_path = output_path / split_name / cat / img_path.name

                    # Run structural detection checking identical to inference logic
                    if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
                        boxes = yolo_results[0].boxes
                        xyxy = boxes.xyxy
                        
                        # Find the largest vehicle present in frame
                        widths = xyxy[:, 2] - xyxy[:, 0]
                        heights = xyxy[:, 3] - xyxy[:, 1]
                        areas = widths * heights
                        
                        largest_idx = areas.argmax().item()
                        largest_area = areas[largest_idx].item()
                        
                        # Calculate coverage percentage
                        area_percentage = (largest_area / total_image_area) * 100
                        
                        # --- 15% AREA CHECK CRITERIA ---
                        if area_percentage >= 15.0:
                            use_full_image = False
                            x1, y1, x2, y2 = map(int, xyxy[largest_idx].tolist())
                            
                            # Crop directly from original image asset without text overlay/boxes
                            cropped_img = img.crop((x1, y1, x2, y2))
                            cropped_img.save(dest_save_path)
                            
                    # --- FALLBACK: Use clean original image layout ---
                    if use_full_image:
                        shutil.copy2(img_path, dest_save_path)
                        
                except Exception as e:
                    print(f"\n❌ Error handling image {img_path.name}: {e}")
                    continue

    print("-" * 60)
    print("✅ Cropped balanced dataset complete! Ready for clean, background-free training.")

if __name__ == "__main__":
    # Fix the seed value to preserve identical dataset splits as the raw version
    random.seed(42)
    
    ROOT_FOLDER = "/home/aziz/Pictures/Internship_Images/no_duplicates_detection_des faces"
    
    split_and_crop_dataset(
        root_dir=ROOT_FOLDER, 
        train_count=240, 
        val_count=59, 
        confidence_threshold=0.30
    )