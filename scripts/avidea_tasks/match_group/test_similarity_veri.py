import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

# This will now import cleanly from your local folder!
import torchreid

class VehicleReIDExtractor:
    def __init__(self):
        # Force CPU execution
        self.device = torch.device("cpu")
        
        print("Initializing specialized OSNet Vehicle Re-ID model (VeRi-776) on CPU...")
        
        # Passing dataset="veri" tells torchreid to download the weights 
        # trained specifically on the VeRi-776 vehicle dataset automatically!
        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=776,
            pretrained=True,
            dataset="veri"
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Standard input preprocessing for OSNet
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def get_embedding(self, image_path):
        """Loads an image, processes it, and extracts an L2-normalized embedding."""
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Could not load image at {image_path}: {e}")
            
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(input_tensor)
            # Normalize to unit sphere (L2 Norm) so Dot Product equals Cosine Similarity
            features = F.normalize(features, p=2, dim=1)
            
        return features.squeeze(0).cpu()


def verify_four_views(view_paths, threshold=0.45):
    """
    Compares 4 perspectives of a vehicle to confirm they share the same identity.
    - threshold: Max allowable Cosine Distance (1.0 - similarity)
    """
    extractor = VehicleReIDExtractor()
    embeddings = {}
    
    print("\nExtracting deep vehicle signatures...")
    for view_name, path in view_paths.items():
        if not Path(path).exists():
            print(f"❌ Error: Image for view '{view_name}' not found at: {path}")
            return False
        embeddings[view_name] = extractor.get_embedding(path)
        print(f" -> Signature extracted for [{view_name.upper()}]")
        
    keys = list(embeddings.keys())
    num_views = len(keys)
    
    print("\n" + "="*20 + " PAIRWISE DISTANCE MATRIX (OSNet) " + "="*20)
    
    max_distance = 0.0
    all_pairs_valid = True
    
    for i in range(num_views):
        for j in range(i + 1, num_views):
            view1, view2 = keys[i], keys[j]
            emb1, emb2 = embeddings[view1], embeddings[view2]
            
            # Distance = 1.0 - Cosine Similarity
            cosine_similarity = torch.dot(emb1, emb2).item()
            cosine_dist = 1.0 - cosine_similarity
            
            max_distance = max(max_distance, cosine_dist)
            status = "✅ MATCH" if cosine_dist <= threshold else "❌ MISMATCH"
            
            print(f" * {view1.upper():<5} <-> {view2.upper():<5} | Distance: {cosine_dist:.4f} | {status}")
            
            if cosine_dist > threshold:
                all_pairs_valid = False

    print("="*74)
    print(f"Worst-case Pair Distance Observed: {max_distance:.4f} (Threshold: {threshold})")
    
    if all_pairs_valid:
        print("\n🎉 SUCCESS: All 4 views confidently correspond to the SAME vehicle identity!")
    else:
        print("\n⚠️ WARNING: Identity anomaly detected! One or more views do not match.")
        
    return all_pairs_valid


# --- Local Run Block ---
if __name__ == "__main__":
    my_vehicle_crops = {
        "front": "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/front.jpg",
        "back":  "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/back.jpg",
        "left":  "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/left.jpg",
        "right": "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/dark_orange_kia_picanto/fake_right2.jpg"  # The white VW Golf
    }
    
    verify_four_views(my_vehicle_crops, threshold=0.45)