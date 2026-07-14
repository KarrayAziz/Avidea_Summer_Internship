import os
import argparse  # Added for command line argument parsing
import pprint
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

print("Loading OWL-ViT (Local Part Detector)...")
detector = pipeline(
    model="google/owlvit-base-patch32", 
    task="zero-shot-object-detection"
)

print("Loading SigLIP 2 (Official Prompt Classifier)...")
siglip_classifier = pipeline(
    task="zero-shot-image-classification",
    model="google/siglip2-base-patch16-224",
    device=-1  # Change to 0 if running on a GPU
)

pp = pprint.PrettyPrinter(indent=4, width=120)

# Exact Prompt Set Template from the online setup
PROMPT_MAP = {
    "front": "This is a photo of the front view of a passenger car.",
    "rear": "This is a photo of the rear view of a passenger car.",
    "null": "This is a photo of something else, no passenger car visible.",
}

def get_largest_car_crop(image, predictions):
    """
    Finds the bounding box of the largest 'car' detection,
    returns its crop and its global top-left coordinates.
    """
    largest_area = 0
    best_box = None
    
    for pred in predictions:
        if pred["label"] == "a car":
            box = pred["box"]
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            area = (xmax - xmin) * (ymax - ymin)
            
            if area > largest_area:
                largest_area = area
                best_box = [xmin, ymin, xmax, ymax]
                
    if best_box:
        cropped_img = image.crop(best_box)
        return cropped_img, best_box
        
    return None, None


def classify_direction_two_stage(image_path, output_path="debug_output.jpg"):
    image = Image.open(image_path).convert("RGB")
    
    # ----------------------------------------------------
    # STAGE 1: Detect and Crop the Main Car
    # ----------------------------------------------------
    car_predictions = detector(
        image,
        candidate_labels=["a car"],
        threshold=0.15
    )
    
    crop, global_box = get_largest_car_crop(image, car_predictions)
    
    if not crop:
        print(" ! Error: No main car detected in the image.")
        return "null"
        
    crop_width = global_box[2] - global_box[0]
    crop_height = global_box[3] - global_box[1]
    aspect_ratio = crop_width / crop_height
    
    print(f" -> Aspect Ratio: {aspect_ratio:.2f}")
    
    # Global offset coordinates of our crop
    crop_x_offset, crop_y_offset, _, _ = global_box
    
    # ----------------------------------------------------
    # STAGE 4: Physical Heuristic-Based Decision Tree
    # ----------------------------------------------------
    
    # CASE A: SQUARE CROP (Must be Front or Rear view)
    # -> Handled robustly by SigLIP 2 using PROMPT_SET_1
    if aspect_ratio < 1.5:
        candidate_labels = list(PROMPT_MAP.values())
        siglip_results = siglip_classifier(crop, candidate_labels=candidate_labels)
        
        # Determine winning prediction text
        winning_text = siglip_results[0]["label"]
        
        # Map back to final shorthand prediction (front, rear, null)
        final_decision = "null"
        for key, val in PROMPT_MAP.items():
            if val == winning_text:
                final_decision = key
                break
                
        # If SigLIP is unsure and picks 'null', resolve tie between front and rear
        if final_decision == "null":
            front_score = next((r["score"] for r in siglip_results if r["label"] == PROMPT_MAP["front"]), 0)
            rear_score = next((r["score"] for r in siglip_results if r["label"] == PROMPT_MAP["rear"]), 0)
            final_decision = "front" if front_score > rear_score else "rear"
        
        # Draw target details and prediction on image
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except IOError:
            font = None
        draw.rectangle(global_box, outline="blue", width=4)
        draw.text((global_box[0], max(0, global_box[1] - 15)), f"Target: {final_decision.upper()} (SigLIP2)", fill="blue", font=font)
        image.save(output_path)
        
        return final_decision
            
    # CASE B: RECTANGULAR CROP (Must be Side profile: Left or Right)
    # -> Handled deterministically by relative-coordinate light math
    else:
        part_predictions = detector(
            crop,
            candidate_labels=["a car headlight", "a car taillight"],
            threshold=0.08
        )
        
        # Filter overlapping boxes & top-region background clutter
        clean_detections = []
        resolved_indices = set()
        
        for i, det_a in enumerate(part_predictions):
            if i in resolved_indices:
                continue
            box_a = det_a["box"]
            ymin_relative = box_a["ymin"] / crop_height
            if ymin_relative < 0.35:
                continue
                
            best_det = det_a
            for j, det_b in enumerate(part_predictions):
                if i == j or j in resolved_indices:
                    continue
                box_b = det_b["box"]
                
                # Intersection check
                x_left = max(box_a["xmin"], box_b["xmin"])
                y_top = max(box_a["ymin"], box_b["ymin"])
                x_right = min(box_a["xmax"], box_b["xmax"])
                y_bottom = min(box_a["ymax"], box_b["ymax"])
                
                if x_right > x_left and y_bottom > y_top:
                    resolved_indices.add(j)
                    if det_b["score"] > best_det["score"]:
                        best_det = det_b
                        
            clean_detections.append(best_det)

        headlights = []
        taillights = []
        
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except IOError:
            font = None
            
        draw.rectangle(global_box, outline="blue", width=4)
        
        for det in clean_detections:
            label = det["label"]
            score = det["score"]
            box = det["box"]
            
            center_x_cropped = (box["xmin"] + box["xmax"]) / 2
            abs_xmin = box["xmin"] + crop_x_offset
            abs_ymin = box["ymin"] + crop_y_offset
            abs_xmax = box["xmax"] + crop_x_offset
            abs_ymax = box["ymax"] + crop_y_offset
            
            if label == "a car headlight":
                headlights.append(center_x_cropped)
                box_color = "green"
            elif label == "a car taillight":
                taillights.append(center_x_cropped)
                box_color = "red"
                
            draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline=box_color, width=3)
            draw.text((abs_xmin, max(0, abs_ymin - 15)), f"{label} ({score:.2f})", fill=box_color, font=font)

        image.save(output_path)
        
        if headlights and taillights:
            avg_headlights = np.mean(headlights)
            avg_taillights = np.mean(taillights)
            return "left" if avg_headlights < avg_taillights else "right"
                    
        elif headlights and not taillights:
            return "left" if np.mean(headlights) < (crop_width / 2) else "right"
                    
        elif taillights and not headlights:
            return "left" if np.mean(taillights) > (crop_width / 2) else "right"
                    
    return "null"


# ----------------------------------------------------
# DIRECTORY PROCESSING BLOCK WITH ARGUMENT PARSER
# ----------------------------------------------------
if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Run hybrid face detection model on a select number of images.")
    parser.add_argument(
        "-n", "--num_images",
        type=int,
        default=None,  # None defaults to processing all files
        help="Number of images to process from the target folder. Leave empty to process all."
    )
    args = parser.parse_args()

    target_dir = Path("/home/aziz/Pictures/Internship_Images/detetction des faces/left/complete")
    output_dir = Path("./debug_results")
    output_dir.mkdir(exist_ok=True)
    
    # Define Ground Truth (All images in this directory are assumed to be "rear")
    GROUND_TRUTH = "left"
    
    print(f"\n--- Starting Hybrid Pipeline Scan ---")
    print(f"Target folder: {target_dir}")
    print(f"Ground Truth Label: {GROUND_TRUTH.upper()}")
    print(f"Saving debug outputs to: {output_dir.resolve()}\n")
    
    # Supported image extensions (case-insensitive)
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    # Gather all valid files in alphabetical order
    all_images = sorted([
        f for f in target_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ])
    
    if not all_images:
        print(f" ! No matching image files found in {target_dir}")
    else:
        total_available = len(all_images)
        
        # Limit the list of images if the argument was provided
        if args.num_images is not None:
            # Prevent slices out of range
            limit = min(args.num_images, total_available)
            print(f"Limit argument passed: Processing {limit} of {total_available} total images.\n")
            all_images = all_images[:limit]
        else:
            print(f"No limit passed: Processing all {total_available} images.\n")
            
        results_summary = {}
        correct_count = 0
        total_count = 0
        
        for idx, img_path in enumerate(all_images, start=1):
            print(f"[{idx}/{len(all_images)}] Processing: {img_path.name}")
            
            # Destination path for the visual box validation
            debug_output_path = output_dir / f"debug_{img_path.name}"
            
            try:
                prediction = classify_direction_two_stage(str(img_path), str(debug_output_path))
                print(f" -> Predicted: {prediction.upper()} | Expected: {GROUND_TRUTH.upper()}")
                
                results_summary[img_path.name] = prediction
                
                # Check prediction against Ground Truth
                if prediction.lower() == GROUND_TRUTH:
                    correct_count += 1
                total_count += 1
                
            except Exception as e:
                print(f" ! Execution failed for {img_path.name}: {e}")
                results_summary[img_path.name] = f"Error: {e}"
                
            print("-" * 50)
            
        # Calculate final accuracy metrics
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
        
        # Print final run summary
        print("\n" + "="*40 + " EVALUATION REPORT " + "="*40)
        print("Individual Predictions:")
        for name, pred in results_summary.items():
            status = "✅ CORRECT" if pred.lower() == GROUND_TRUTH else "❌ WRONG"
            print(f" * {name:<35} -> Predicted: {pred.upper():<8} | {status}")
            
        print("-" * 98)
        print(f" Total Images Processed: {total_count}")
        print(f" Correct Predictions:    {correct_count}")
        print(f" Incorrect Predictions:  {total_count - correct_count}")
        print(f" Overall Evaluation Accuracy: {accuracy:.2f}%")
        print("="*98)