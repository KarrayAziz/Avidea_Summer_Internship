import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import sys

# Add the model directory to Python path
sys.path.insert(0, os.path.abspath("./models/result"))

# Import from model.py
from model import ft_net, ft_net_dense, ft_net_swin, ft_net_efficient, ft_net_hr, ft_net_NAS

class CustomVRICExtractor:
    def __init__(self, checkpoint_name="net_19.pth"):
        # 1. Dynamically target GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Custom Vehicle Re-ID Model using {checkpoint_name} on [{self.device}]...")
        
        # Load training configuration options
        opts_path = "./models/result/opts.yaml"
        if not os.path.exists(opts_path):
            raise FileNotFoundError(f"Configuration file not found at: {opts_path}")
            
        with open(opts_path, 'r') as f:
            opts = yaml.safe_load(f)
        
        num_classes = opts.get('num_classes', 2811)
        
        # 2. Load checkpoint onto targeted device structure
        ckpt_path = os.path.join("./models/result", checkpoint_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")
            
        state_dict = torch.load(ckpt_path, map_location=self.device)
        
        # 3. Auto-detect if IBN was used by checking state_dict keys
        has_ibn = any('ibn' in key.lower() or 'IN.' in key for key in state_dict.keys())
        
        if has_ibn:
            print("Detected IBN layers in state_dict. Initializing ft_net with ibn=True...")
            self.model = ft_net(num_classes, stride=1, ibn=True)
        else:
            print("Standard ResNet detected. Initializing ft_net...")
            self.model = ft_net(num_classes, stride=1)
        
        # 4. Load weights
        self.model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {ckpt_path}")
        
        # 5. Strip classification head to get feature embeddings only
        if hasattr(self.model, 'classifier'):
            if hasattr(self.model.classifier, 'classifier'):
                self.model.classifier.classifier = nn.Sequential()
            else:
                self.model.classifier = nn.Sequential()
            
        # Push target model parameters directly onto GPU memory
        self.model.to(self.device)
        self.model.eval()
        
        # 6. Image transformation pipeline
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image asset {image_path}: {e}")
            
        # Send input image tensor straight onto GPU execution pipeline
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
            features = F.normalize(features, p=2, dim=1)
            
        # Return on GPU to keep downstream verification math completely on-device
        return features.squeeze(0)


def verify_four_views(view_paths, threshold=0.50):
    extractor = CustomVRICExtractor("net_19.pth")
    embeddings = {}
    
    print("\nExtracting feature vectors...")
    for view_name, path in view_paths.items():
        if not Path(path).exists():
            print(f"❌ File missing for view {view_name}: {path}")
            return False
        embeddings[view_name] = extractor.get_embedding(path)
        print(f" -> Embedded vector ready for [{view_name.upper()}]")
        
    keys = list(embeddings.keys())
    print("\n" + "="*20 + " VRIC WORKSPACE MATRIX " + "="*20)
    all_pairs_valid = True
    
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            v1, v2 = keys[i], keys[j]
            # This dot product calculation now executes instantly on GPU hardware
            similarity = torch.dot(embeddings[v1], embeddings[v2]).item()
            distance = 1.0 - similarity
            
            status = "✅ MATCH" if distance <= threshold else "❌ MISMATCH"
            print(f" * {v1.upper():<5} <-> {v2.upper():<5} | Cosine Distance: {distance:.4f} | {status}")
            
            if distance > threshold:
                all_pairs_valid = False
                
    print("="*63)
    if all_pairs_valid:
        print("\n🎉 SUCCESS: All views correspond to the SAME identity.")
    else:
        print("\n⚠️ WARNING: Spatial identity anomaly detected.")
    return all_pairs_valid

if __name__ == "__main__":
    my_vehicle_crops = {
        "front": "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/backk1.jpeg",
        "back":  "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/backk2.jpeg",
        "left":  "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/back3.jpg",
        "right": "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/back4.jpg"
    }
    
    verify_four_views(my_vehicle_crops, threshold=0.40)