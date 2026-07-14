import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# 1. Map classes alphabetically (matching PyTorch's ImageFolder standard)
CLASS_NAMES = ['back', 'front', 'left', 'right']

# 2. Preprocessing pipeline (Must be identical to the validation transforms used in training)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Force the device context to 'cpu'
device = torch.device('cpu')

# 4. Rebuild the ResNet-18 structure locally
print("Initializing model architecture...")
model = models.resnet18()  # No need for pretrained=True because we are loading our custom weights!
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)

# 5. Load your saved weights with CPU remapping
weights_path = '/home/aziz/Aziz/DigiCover/usingGeminiApi/scripts/vehicle_orientation_resnet18_finetuned.pth'

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"⚠️ Could not find weights at '{weights_path}'. Make sure it is in the same directory!")

print(f"Loading weights from '{weights_path}' onto CPU...")
# map_location=device (or 'cpu') forces GPU-trained weights to load correctly on your CPU!
model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.to(device)
model.eval()  # CRITICAL: Sets dropout/batchnorm layers to evaluation mode!
print("Model successfully loaded and ready on CPU!\n")


# 6. Prediction Function
def predict_car_orientation(image_path):
    try:
        # Open and preprocess the image
        img = Image.open(image_path).convert("RGB")
        input_tensor = inference_transforms(img).unsqueeze(0).to(device) # Shape becomes [1, 3, 224, 224]
        
        # Disable gradient calculations for faster, low-memory CPU tracking
        with torch.no_grad():
            outputs = model(input_tensor)
            # Convert raw outputs (logits) into probability percentages
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, class_idx = torch.max(probabilities, 0)
            
        predicted_class = CLASS_NAMES[class_idx.item()]
        
        print(f"🔮 Prediction Results for: {os.path.basename(image_path)}")
        print(f" -> Direction: {predicted_class.upper()} ({confidence.item()*100:.2f}% confidence)")
        print("-" * 50)
        for i, score in enumerate(probabilities):
            print(f" * {CLASS_NAMES[i]:<6}: {score.item()*100:.2f}%")
        print("="*50 + "\n")
        
        return predicted_class
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

# --- RUN A QUICK TEST ---
if __name__ == "__main__":
    # Put a path to any test image on your computer here!
    test_image_path = "/home/aziz/Pictures/Internship_Images/rearr.jpg" 
    
    if os.path.exists(test_image_path):
        predict_car_orientation(test_image_path)
    else:
        print(f"Point 'test_image_path' to a real image on your computer to run a test!")