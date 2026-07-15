import time
import argparse
from pathlib import Path
from scripts.plate_extractor import TunisianPlateExtractor

def run_multi_image_test():
    parser = argparse.ArgumentParser(description="Run Tunisian License Plate OCR Benchmark on multiple images.")
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default="/home/aziz/Pictures/Internship_Images/detetction des faces/back/complete",
        help="Path to the directory containing images to process."
    )
    parser.add_argument(
        "--num_images", "-n",
        type=int,
        default=5,
        help="Number of images to process from the directory."
    )
    parser.add_argument(
        "--save_dir", "-s",
        type=str,
        default="output_crops",
        help="Directory where normal and inverted cropped plates will be saved."
    )
    args = parser.parse_args()

    # Initialize the extractor
    extractor = TunisianPlateExtractor()
    
    img_dir = Path(args.dir)
    if not img_dir.exists() or not img_dir.is_dir():
        print(f"⚠️ Directory does not exist or is invalid: {img_dir}")
        return

    # Gather target images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(img_dir.glob(ext))
    
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"⚠️ No images found in directory: {img_dir}")
        return

    target_images = image_paths[:args.num_images]

    print("\n" + "="*50)
    print(f"RUNNING REAL-WORLD MULTI-OCR BENCHMARK ON CPU")
    print(f" Target Directory : {img_dir}")
    print(f" Total Found      : {len(image_paths)} images")
    print(f" Processing Count : {len(target_images)} images")
    print(f" Crop Output Dir  : {args.save_dir}/")
    print("="*50 + "\n")

    results = []
    total_latency = 0.0

    for idx, img_path in enumerate(target_images, 1):
        print(f"📸 [{idx}/{len(target_images)}] Processing: {img_path.name}")
        
        start_time = time.perf_counter()
        # Pass the saving directory here
        parsed_plate = extractor.extract_plate(str(img_path), save_crops_dir=args.save_dir)
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        total_latency += latency
        
        results.append({
            "name": img_path.name,
            "output": parsed_plate if parsed_plate else "❌ FAILED",
            "latency": latency
        })
        print("-" * 50)

    # Print Aggregated Benchmark Report
    print("\n" + "═"*70)
    print("                      AGGREGATED OCR BENCHMARK REPORT                    ")
    print("═"*70)
    print(f" {'File Name':<35} | {'Latency':<10} | {'Extracted Output':<20}")
    print("-" * 70)
    for res in results:
        print(f" {res['name'][:35]:<35} | {res['latency']:.3f}s     | {res['output']:<20}")
    print("═"*70)
    
    avg_latency = total_latency / len(target_images) if target_images else 0
    print(f" Overall Processing Latency : {total_latency:.3f} seconds")
    print(f" Average Latency per Image  : {avg_latency:.3f} seconds")
    print("═"*70 + "\n")

if __name__ == "__main__":
    run_multi_image_test()