import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def flip_images_horizontally(input_dir, output_dir=None):
    input_path = Path(input_dir)
    
    # If no output directory is given, create a 'flipped_images' folder inside the input directory
    if output_dir is None:
        dest_path = input_path / "flipped_images"
    else:
        dest_path = Path(output_dir)
        
    dest_path.mkdir(parents=True, exist_ok=True)

    # Gather all images
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [f for f in input_path.glob("*") if f.is_file() and f.suffix.lower() in valid_extensions]

    if not image_files:
        print(f"❌ No valid images found in: {input_dir}")
        return

    print(f"🔄 Found {len(image_files)} images. Starting horizontal flip augmentation...")
    print(f"📁 Saving outputs to: {dest_path}")
    print("-" * 50)

    # Process with a progress bar
    progress_bar = tqdm(image_files, desc="Flipping Images", unit="img")
    
    success_count = 0
    for img_path in progress_bar:
        try:
            # Update progress bar subtext
            progress_bar.set_postfix(file=f"{img_path.name[:15]}")
            
            # Open, flip horizontally, and save
            with Image.open(img_path) as img:
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Append the _flipped suffix right before the file extension
                new_filename = f"{img_path.stem}_flipped{img_path.suffix}"
                save_path = dest_path / new_filename
                
                flipped_img.save(save_path)
                success_count += 1
                
        except Exception as e:
            progress_bar.write(f"❌ Error processing {img_path.name}: {e}")

    print("-" * 50)
    print(f"✅ Successfully flipped and saved {success_count}/{len(image_files)} images!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horizontally flip all images in a target directory.")
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="Path to the folder containing images to flip."
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help="Path to save the flipped images (defaults to creating a 'flipped_images' folder inside the input directory)."
    )
    args = parser.parse_args()

    flip_images_horizontally(args.input, args.output)