from pathlib import Path

# Paths to your generated folders
TRAIN_VAL_DIR = Path("./custom_car_dataset1")
INFERENCE_DIR = Path("./custom_car_inference_set")
VIEWS = ["front", "back", "left", "right"]

def extract_original_name(filename: str) -> str:
    """Strips the 'crop_X_' prefix to recover the original image name."""
    parts = filename.split("_", 2)
    # If it follows 'crop_{number}_{name}', return the remaining name string
    if len(parts) > 2 and parts[0] == "crop":
        return parts[2]
    return filename

print("--- Starting Dataset Overlap Verification ---")
has_overlap = False

for view in VIEWS:
    # Gather all original names from train and val splits
    train_val_originals = set()
    
    for split in ["train", "val"]:
        split_folder = TRAIN_VAL_DIR / split / view
        if split_folder.exists():
            for f in split_folder.iterdir():
                if f.is_file():
                    train_val_originals.add(extract_original_name(f.name))
                    
    # Gather all original names from the inference folder
    inference_folder = INFERENCE_DIR / view
    inference_originals = set()
    if inference_folder.exists():
        for f in inference_folder.iterdir():
            if f.is_file():
                inference_originals.add(extract_original_name(f.name))
                
    # Find intersection
    duplicates = train_val_originals.intersection(inference_originals)
    
    if duplicates:
        has_overlap = True
        print(f"⚠️ Overlap detected in [{view.upper()}] class! ({len(duplicates)} items leaked):")
        for item in duplicates:
            print(f"   - {item}")
    else:
        print(f"✅ Class [{view.upper()}]: Zero leakage detected.")

print("\n" + "="*45)
if has_overlap:
    print("❌ Critical: There is data leakage between your datasets!")
else:
    print("🎉 Clean Split! Your inference set contains completely unique images.")
print("="*45)