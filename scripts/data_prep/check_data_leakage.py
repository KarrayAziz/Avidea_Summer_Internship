import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
from PIL import Image
from glob import glob
from tqdm import tqdm

# Import your GPU-enabled extractor from your script
from scripts.test_vric_similarity_gpu import CustomVRICExtractor

def detect_dataset_overlap(train_dir, inference_dir, distance_threshold=0.05, output_leak_dir="leakage_results"):
    """
    Finds identical or near-identical images between train and inference sets
    using Re-ID feature embeddings entirely on the GPU, sorts them by distance,
    and saves flagged pairs side-by-side into a folder structure.
    """
    # 1. Initialize your GPU-enabled extractor
    extractor = CustomVRICExtractor("net_19.pth")
    target_device = extractor.device
    
    # 2. Gather image paths
    train_paths = sorted(glob(os.path.join(train_dir, "*.*")))
    inf_paths = sorted(glob(os.path.join(inference_dir, "*.*")))
    
    # Filter for standard image formats
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    train_paths = [p for p in train_paths if p.lower().endswith(valid_exts)]
    inf_paths = [p for p in inf_paths if p.lower().endswith(valid_exts)]
    
    if not train_paths or not inf_paths:
        print(f"❌ Empty directory detected. Train count: {len(train_paths)}, Inference count: {len(inf_paths)}")
        return

    print(f"\n📊 Scanning: {len(train_paths)} training images vs {len(inf_paths)} inference images.")

    # 3. Extract Train Embeddings directly on GPU memory
    print("\nEncoding training set...")
    train_embeddings = []
    for path in tqdm(train_paths):
        train_embeddings.append(extractor.get_embedding(path))
    train_tensor = torch.stack(train_embeddings).to(target_device)

    # 4. Extract Inference Embeddings directly on GPU memory
    print("\nEncoding inference set...")
    inf_embeddings = []
    for path in tqdm(inf_paths):
        inf_embeddings.append(extractor.get_embedding(path))
    inf_tensor = torch.stack(inf_embeddings).to(target_device)

    # 5. Compute Full Distance Matrix via Parallel GPU Matrix Multiplication
    print(f"\nComputing similarity matrix on [{target_device}]...")
    similarity_matrix = torch.mm(train_tensor, inf_tensor.t())
    distance_matrix = 1.0 - similarity_matrix

    # 6. Find Threshold Breaches
    overlap_indices = (distance_matrix <= distance_threshold).nonzero(as_tuple=False).cpu()
    distance_matrix = distance_matrix.cpu() 
    
    print("\n" + "="*25 + " LEAKAGE REPORT " + "="*25)
    if len(overlap_indices) == 0:
        print("🎉 SUCCESS: No data leakage detected! The datasets are cleanly separated.")
        print("="*66)
        return
        
    # --- NEW: Gather and Sort Matches by Distance ---
    detected_matches = []
    for idx in overlap_indices:
        train_idx = idx[0].item()
        inf_idx = idx[1].item()
        dist = distance_matrix[train_idx, inf_idx].item()
        
        detected_matches.append({
            'train_path': train_paths[train_idx],
            'inf_path': inf_paths[inf_idx],
            'distance': dist
        })
        
    # Sort in ascending order (0.0000 exact duplicates first)
    detected_matches = sorted(detected_matches, key=lambda x: x['distance'])
    # ------------------------------------------------
        
    print(f"⚠️ WARNING: Found {len(detected_matches)} duplicate/overlapping image pairs!")
    print(f"📁 Creating sorted side-by-side visual pairs inside directory: {output_leak_dir}\n")
    
    # Clear out older results if they exist to keep validation clean
    if os.path.exists(output_leak_dir):
        shutil.rmtree(output_leak_dir)
    os.makedirs(output_leak_dir, exist_ok=True)
    
    for match_counter, match in enumerate(detected_matches, start=1):
        orig_train_path = match['train_path']
        orig_inf_path = match['inf_path']
        dist = match['distance']
        
        # Get the real, original filenames
        train_filename = os.path.basename(orig_train_path)
        inf_filename = os.path.basename(orig_inf_path)
        
        # If the filenames happen to be exactly identical, append a prefix 
        # so they don't overwrite each other if they land in the same subfolder
        if train_filename == inf_filename:
            train_filename = f"TRAIN_{train_filename}"
            inf_filename = f"INF_{inf_filename}"
        
        # Create a unique subfolder name preserving the sorted index ranking
        pair_folder_name = f"pair_{match_counter:03d}_dist_{dist:.4f}"
        pair_folder_path = os.path.join(output_leak_dir, pair_folder_name)
        os.makedirs(pair_folder_path, exist_ok=True)
        
        # Define destination paths using the REAL filenames
        dest_train = os.path.join(pair_folder_path, train_filename)
        dest_inf = os.path.join(pair_folder_path, inf_filename)
        
        # Copy the images into the pair folder structure
        shutil.copy2(orig_train_path, dest_train)
        shutil.copy2(orig_inf_path, dest_inf)
        
        print(f"🔴 Leakage Match #{match_counter:03d} (Distance: {dist:.4f}):")
        print(f"   ├─ Train Source: {os.path.basename(orig_train_path)}")
        print(f"   └─ Inf Source:   {os.path.basename(orig_inf_path)}")
        
    print("="*66)
    print(f"💡 Done! Open your file manager and browse '{output_leak_dir}'. The worst leakage issues are ranked first.")

if __name__ == "__main__":
    TRAIN_FOLDER = "/home/aziz/Aziz/DigiCover/usingGeminiApi/custom_car_dataset1/train/right"
    INFERENCE_FOLDER = "/home/aziz/Aziz/DigiCover/usingGeminiApi/custom_car_inference_set/right"
    OUTPUT_RESULTS_FOLDER = "/home/aziz/Aziz/DigiCover/usingGeminiApi/leakage_results"
    
    # Run the detection and auto-grouping routine
    detect_dataset_overlap(
        TRAIN_FOLDER, 
        INFERENCE_FOLDER, 
        distance_threshold=0.14, 
        output_leak_dir=OUTPUT_RESULTS_FOLDER
    )