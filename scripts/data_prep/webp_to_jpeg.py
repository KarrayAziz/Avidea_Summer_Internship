import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 📌 Configuration paths
WEBP_DIR = Path("/home/aziz/Aziz/DigiCover/usingGeminiApi/Clean_Inference_Set/back/webp")
JPG_DIR = Path("/home/aziz/Aziz/DigiCover/usingGeminiApi/Clean_Inference_Set/back/jpg")

def convert_webp_to_jpg():
    # 1. Ensure target directory exists
    JPG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Gather all .webp images (case-insensitive)
    webp_files = [f for f in WEBP_DIR.glob("*") if f.suffix.lower() in {".webp"}]
    
    if not webp_files:
        print(f"⚠️ No .webp files found in {WEBP_DIR}")
        return

    print(f"🔄 Found {len(webp_files)} images. Converting to JPEG...")

    # 3. Process with progress bar
    converted_count = 0
    for webp_path in tqdm(webp_files, desc="Converting", unit="img"):
        try:
            # Open the webp image
            with Image.open(webp_path) as img:
                # Convert to RGB mode (webp can have alpha channels which JPEG doesn't support)
                rgb_img = img.convert("RGB")
                
                # Define output path with .jpg extension
                jpg_path = JPG_DIR / f"{webp_path.stem}.jpg"
                
                # Save as JPEG with high quality
                rgb_img.save(jpg_path, "JPEG", quality=95)
                converted_count += 1
                
        except Exception as e:
            print(f"\n❌ Failed to convert {webp_path.name}: {e}")

    print("-" * 50)
    print(f"✅ Success! Converted {converted_count}/{len(webp_files)} images to: {JPG_DIR}")

if __name__ == "__main__":
    convert_webp_to_jpg()