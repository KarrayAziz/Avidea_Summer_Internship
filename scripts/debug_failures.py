import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

def main():
    # 1. Configuration & Paths
    CLASS_NAMES = ['back', 'front', 'left', 'right']
    TEST_DATASET_DIR = Path("/home/aziz/Aziz/DigiCover/usingGeminiApi/test_dataset_unseen")
    FAILURE_DIR = Path("/home/aziz/Aziz/DigiCover/usingGeminiApi/failures")
    WEIGHTS_PATH = '/home/aziz/Aziz/DigiCover/usingGeminiApi/scripts/vehicle_orientation_resnet18_finetuned.pth'

    print("--- Debug Failures Script Started ---")

    # Re-create a clean failures directory
    if FAILURE_DIR.exists():
        print(f"Cleaning up old failures directory at: {FAILURE_DIR}")
        shutil.rmtree(FAILURE_DIR)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Setup Device & Model
    device = torch.device('cpu')
    print("Rebuilding model architecture on CPU...")
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"⚠️ Could not find weights at '{WEIGHTS_PATH}'! Please make sure it is in the root directory.")

    print(f"Loading weights from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 3. Transforms
    inference_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\n--- Scanning Unseen Dataset for Failures ---")
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    failure_count = 0
    scanned_count = 0

    if not TEST_DATASET_DIR.exists():
        print(f"⚠️ Error: Test dataset directory does not exist at: {TEST_DATASET_DIR}")
        return

    for expected_class in CLASS_NAMES:
        class_folder = TEST_DATASET_DIR / expected_class
        if not class_folder.exists():
            print(f" ! Folder missing: {class_folder}")
            continue
            
        image_paths = [f for f in class_folder.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        print(f"Scanning {expected_class.upper()} folder ({len(image_paths)} images)...")
        
        for img_path in image_paths:
            scanned_count += 1
            try:
                img = Image.open(img_path).convert("RGB")
                input_tensor = inference_transforms(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, pred_idx = torch.max(probabilities, 0)
                    predicted_class = CLASS_NAMES[pred_idx.item()]
                    
                # If the model guessed wrong, isolate the image
                if predicted_class != expected_class:
                    failure_count += 1
                    
                    # Create a clear descriptive filename for debugging
                    confidence_pct = confidence.item() * 100
                    new_filename = f"expected_{expected_class.upper()}_predicted_{predicted_class.upper()}_conf_{confidence_pct:.1f}pct_{img_path.name}"
                    dest_path = FAILURE_DIR / new_filename
                    
                    # Copy the file
                    shutil.copy2(img_path, dest_path)
                    print(f"  ❌ Mismatch: Real={expected_class.upper()} | Pred={predicted_class.upper()} ({confidence_pct:.1f}%)")
                    
            except Exception as e:
                print(f" ! Skipped processing {img_path.name} due to: {e}")

    print("\n" + "="*40 + " SCAN COMPLETE " + "="*40)
    print(f" Total images scanned: {scanned_count}")
    print(f" Isolated exactly {failure_count} failure cases.")
    print(f" You can inspect them visually inside: {FAILURE_DIR.resolve()}")
    print("="*95)

# This guarantees execution when running the file directly or as a module
if __name__ == "__main__":
    main()