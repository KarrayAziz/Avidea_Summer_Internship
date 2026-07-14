import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from pathlib import Path

# Map classes alphabetically (matching PyTorch's ImageFolder standard)
CLASS_NAMES = ['back', 'front', 'left', 'right']

# Same preprocessing transforms used during training/validation
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device('cpu')

# Rebuild model structure locally on CPU
print("Initializing model architecture...")
model = models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)

weights_path = '/home/aziz/Aziz/DigiCover/usingGeminiApi/scripts/vehicle_orientation_resnet18_finetuned.pth'
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"⚠️ Could not find '{weights_path}'! Please place it in this directory.")

print(f"Loading weights onto CPU...")
model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.to(device)
model.eval()
print("Model ready for evaluation on CPU!\n")

# Path to the newly created unseen test folder
TEST_DATASET_DIR = Path("/home/aziz/Aziz/DigiCover/usingGeminiApi/test_dataset_unseen")

if not TEST_DATASET_DIR.exists():
    raise FileNotFoundError(f"⚠️ Test dataset directory '{TEST_DATASET_DIR}' not found. Run prepare_test_dataset.py first!")

total_images = 0
correct_predictions = 0
class_stats = {cls: {"total": 0, "correct": 0} for cls in CLASS_NAMES}

print("--- Starting Batch Inference on Unseen Test Dataset ---")
valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Loop through each subdirectory (class folder)
for expected_class in CLASS_NAMES:
    class_folder = TEST_DATASET_DIR / expected_class
    if not class_folder.exists():
        continue
        
    print(f"\nEvaluating folder: {expected_class.upper()}")
    
    image_paths = [f for f in class_folder.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            input_tensor = inference_transforms(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                _, pred_idx = torch.max(outputs, 1)
                predicted_class = CLASS_NAMES[pred_idx.item()]
                
            total_images += 1
            class_stats[expected_class]["total"] += 1
            
            if predicted_class == expected_class:
                correct_predictions += 1
                class_stats[expected_class]["correct"] += 1
                
        except Exception as e:
            print(f" ! Skipped processing {img_path.name} due to: {e}")

# Display Final Accuracy Stats
print("\n" + "="*45 + " REAL WORLD PERFORMANCE REPORT " + "="*45)
for cls in CLASS_NAMES:
    total = class_stats[cls]["total"]
    correct = class_stats[cls]["correct"]
    acc = (correct / total) * 100 if total > 0 else 0.0
    print(f" * Class {cls.upper():<7} -> Total Tested: {total:<4} | Correct: {correct:<4} | Accuracy: {acc:.2f}%")
    
print("-" * 121)
overall_accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0.0
print(f" Total Unseen Images Evaluated: {total_images}")
print(f" Total Correct Predictions:      {correct_predictions}")
print(f" Overall Real-World Accuracy:    {overall_accuracy:.2f}%")
print("="*121)