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

def detect_single_folder_duplicates(input_dir, distance_threshold=0.05, output_dup_dir="duplicate_results"):
    """
    Finds identical or near-identical images within a SINGLE folder
    using Re-ID feature embeddings entirely on the GPU, sorts them by distance,
    and saves flagged pairs side-by-side into a folder structure.
    """
    # 1. Initialize your GPU-enabled extractor
    extractor = CustomVRICExtractor("net_19.pth")
    target_device = extractor.device
    
    # 2. Gather image paths from the single folder
    img_paths = sorted(glob(os.path.join(input_dir, "*.*")))
    
    # Filter for standard image formats
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    img_paths = [p for p in img_paths if p.lower().endswith(valid_exts)]
    
    num_images = len(img_paths)
    if num_images < 2:
        print(f"❌ Not enough images to perform a comparison. Found: {num_images}")
        return

    print(f"\n📊 Scanning: {num_images} images within the folder for internal duplicates.")

    # 3. Extract Embeddings directly on GPU memory
    print("\nEncoding image database...")
    embeddings = []
    for path in tqdm(img_paths):
        embeddings.append(extractor.get_embedding(path))
    embeddings_tensor = torch.stack(embeddings).to(target_device)

    # 4. Compute Internal Distance Matrix (Self-Similarity)
    print(f"\nComputing similarity matrix on [{target_device}]...")
    similarity_matrix = torch.mm(embeddings_tensor, embeddings_tensor.t())
    distance_matrix = 1.0 - similarity_matrix

    # 5. Filter out Self-Matches and Mirror Duplicates using an Upper Triangle Mask
    # diagonal=1 eliminates the 0.0000 distance of an image compared to itself.
    # It also ensures we only process Pair(A, B) and ignore the mirrored Pair(B, A).
    mask = torch.triu(torch.ones_like(distance_matrix), diagonal=1)
    
    # Apply mask and find indices that satisfy the threshold
    valid_threshold_mask = (distance_matrix <= distance_threshold) & (mask == 1)
    overlap_indices = valid_threshold_mask.nonzero(as_tuple=False).cpu()
    distance_matrix = distance_matrix.cpu() 
    
    print("\n" + "="*25 + " DUPLICATE REPORT " + "="*25)
    if len(overlap_indices) == 0:
        print("🎉 SUCCESS: No internal duplicates detected! Your dataset is completely unique.")
        print("="*66)
        return
        
    # 6. Gather and Sort Matches by Distance
    detected_matches = []
    for idx in overlap_indices:
        idx_a = idx[0].item()
        idx_b = idx[1].item()
        dist = distance_matrix[idx_a, idx_b].item()
        
        detected_matches.append({
            'img_a_path': img_paths[idx_a],
            'img_b_path': img_paths[idx_b],
            'distance': dist
        })
        
    # Sort in ascending order (0.0000 exact duplicates first)
    detected_matches = sorted(detected_matches, key=lambda x: x['distance'])
        
    print(f"⚠️ WARNING: Found {len(detected_matches)} near-identical image pairs within this folder!")
    print(f"📁 Creating sorted side-by-side visual pairs inside directory: {output_dup_dir}\n")
    
    # Clear out older results if they exist to keep validation clean
    if os.path.exists(output_dup_dir):
        shutil.rmtree(output_dup_dir)
    os.makedirs(output_dup_dir, exist_ok=True)
    
    for match_counter, match in enumerate(detected_matches, start=1):
        orig_a_path = match['img_a_path']
        orig_b_path = match['img_b_path']
        dist = match['distance']
        
        # Get original filenames
        filename_a = os.path.basename(orig_a_path)
        filename_b = os.path.basename(orig_b_path)
        
        # Safe-guard filenames so they look clean side-by-side
        filename_a = f"A_{filename_a}"
        filename_b = f"B_{filename_b}"
        
        # Create a unique subfolder name preserving the sorted index ranking
        pair_folder_name = f"pair_{match_counter:03d}_dist_{dist:.4f}"
        pair_folder_path = os.path.join(output_dup_dir, pair_folder_name)
        os.makedirs(pair_folder_path, exist_ok=True)
        
        # Define destination paths
        dest_a = os.path.join(pair_folder_path, filename_a)
        dest_b = os.path.join(pair_folder_path, filename_b)
        
        # Copy the images into the pair folder structure
        shutil.copy2(orig_a_path, dest_a)
        shutil.copy2(orig_b_path, dest_b)
        
        print(f"🔴 Duplicate Match #{match_counter:03d} (Distance: {dist:.4f}):")
        print(f"   ├─ Image A: {os.path.basename(orig_a_path)}")
        print(f"   └─ Image B: {os.path.basename(orig_b_path)}")
        
    print("="*66)
    print(f"💡 Done! Browse '{output_dup_dir}' to isolate and purge redundancies.")

if __name__ == "__main__":
    # Point this to the folder you want to audit internally (e.g., your back training set)
    TARGET_FOLDER = "/home/aziz/Pictures/Internship_Images/no_duplicates_detection_des faces/back"
    OUTPUT_RESULTS_FOLDER = "/home/aziz/Aziz/DigiCover/usingGeminiApi/internal_duplicate_results"
    
    # Run the detection and auto-grouping routine
    # A threshold of 0.10 to 0.14 is highly effective for trapping rapid burst images
    detect_single_folder_duplicates(
        TARGET_FOLDER, 
        distance_threshold=0.09, 
        output_dup_dir=OUTPUT_RESULTS_FOLDER
    )