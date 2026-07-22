import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class VehicleReIDExtractor:
    def __init__(self):
        self.device = torch.device("cpu")
        
        # Load a strong pre-trained feature extractor (ResNet-50 gives rich embeddings)
        print("Loading Re-ID Backbone network...")
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Strip the final classification layer (fc) to output raw 2048-dim embeddings
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        
        # Standard Re-ID Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, image_path):
        """Extracts a normalized feature vector from a cropped vehicle image."""
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.backbone(tensor)
            # Flatten to 1D vector: [1, 2048, 1, 1] -> [2048]
            embedding = torch.squeeze(embedding).cpu().numpy()
            
        # L2 Normalization makes Cosine Similarity equivalent to a simple Dot Product
        return embedding / np.linalg.norm(embedding)

def verify_four_views(view_paths, threshold=0.65):
    """
    Compares 4 views to verify if they match the same vehicle identity.
    view_paths: Dict containing {'front': path, 'back': path, 'left': path, 'right': path}
    threshold: Max allowable cosine distance to consider images a 'match'
    """
    extractor = VehicleReIDExtractor()
    embeddings = {}
    
    # 1. Extract signatures for all available views
    for view_name, path in view_paths.items():
        embeddings[view_name] = extractor.get_embedding(path)
        
    keys = list(embeddings.keys())
    num_views = len(keys)
    
    print("\n" + "="*20 + " PAIRWISE DISTANCE MATRIX " + "="*20)
    
    max_distance = 0.0
    all_pairs_valid = True
    
    # 2. Compute Cosine Distance across every distinct pair
    # Cosine Distance = 1.0 - Cosine Similarity
    for i in range(num_views):
        for j in range(i + 1, num_views):
            view1, view2 = keys[i], keys[j]
            emb1, emb2 = embeddings[view1], embeddings[view2]
            
            cosine_similarity = np.dot(emb1, emb2)
            cosine_dist = 1.0 - cosine_similarity
            
            max_distance = max(max_distance, cosine_dist)
            status = "✅ MATCH" if cosine_dist <= threshold else "❌ MISMATCH"
            
            print(f" * {view1.upper():<5} <-> {view2.upper():<5} | Distance: {cosine_dist:.4f} | {status}")
            
            if cosine_dist > threshold:
                all_pairs_valid = False

    print("="*66)
    print(f"Worst-case Pair Distance Observed: {max_distance:.4f} (Threshold: {threshold})")
    
    if all_pairs_valid:
        print("🎉 SUCCESS: All 4 views confidently correspond to the SAME vehicle identity!")
    else:
        print("⚠️ WARNING: Identity anomaly detected! One or more views do not match the system cluster.")
        
    return all_pairs_valid

# --- Example Usage ---
if __name__ == "__main__":
    # Point these paths to your test crops from a single vehicle sequence
    my_vehicle_crops = {
        "front": "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/front.jpg",
        "back":  "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/back.jpg",
        "left":  "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/left.jpg",
        "right": "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/fake_right2.jpg"
    }
    
    # Run verification execution
    verify_four_views(my_vehicle_crops, threshold=0.65)