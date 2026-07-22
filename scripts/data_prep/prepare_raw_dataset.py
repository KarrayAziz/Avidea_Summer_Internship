import os
import random
import shutil
from glob import glob
from pathlib import Path

def split_dataset(root_dir, train_count=240, val_count=59):
    root_path = Path(root_dir)
    output_path = root_path.parent / "dataset_split"
    
    # Define our view categories
    categories = ['back', 'front', 'left', 'right']
    splits = ['train', 'val', 'test']
    
    # Create target directory structure
    for split in splits:
        for category in categories:
            (output_path / split / category).mkdir(parents=True, exist_ok=True)
            
    print(f"📁 Target directories created at: {output_path}")
    print("-" * 60)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for cat in categories:
        cat_dir = root_path / cat
        if not cat_dir.exists():
            print(f"⚠️ Warning: Folder for class '{cat}' not found at {cat_dir}. Skipping.")
            continue
            
        # Gather all valid images in the category folder
        all_images = [
            f for f in cat_dir.glob("*") 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        
        # Shuffle to ensure an unbiased, random distribution
        random.shuffle(all_images)
        total_imgs = len(all_images)
        
        # Verify we have enough images to fulfill the 299 requirement
        target_total = train_count + val_count
        if total_imgs < target_total:
            print(f"❌ Error: Class '{cat}' only has {total_imgs} images. Need at least {target_total}!")
            continue
            
        # Slice out the precise quotas
        train_imgs = all_images[:train_count]
        val_imgs = all_images[train_count:target_total]
        test_imgs = all_images[target_total:]
        
        print(f"📦 Processing '{cat}': Total={total_imgs} | Train={len(train_imgs)} | Val={len(val_imgs)} | Test={len(test_imgs)}")
        
        # Helper function to copy images to their respective destinations
        def copy_set(img_list, split_name):
            for img_path in img_list:
                dest = output_path / split_name / cat / img_path.name
                shutil.copy2(img_path, dest)

        # Execute the copying routine
        copy_set(train_imgs, 'train')
        copy_set(val_imgs, 'val')
        copy_set(test_imgs, 'test')

    print("-" * 60)
    print("✅ Dataset splitting complete! Your balanced training sets and unbiased test holdouts are ready.")

if __name__ == "__main__":
    # Fix the random seed so that if you run this again, it yields the exact same distribution split
    random.seed(42)
    
    ROOT_FOLDER = "/home/aziz/Pictures/Internship_Images/no_duplicates_detection_des faces"
    
    split_dataset(ROOT_FOLDER, train_count=240, val_count=59)