import numpy as np
import pprint
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
    device=-1  # Set to 0 if running on a GPU
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
    print(f"\n[Stage 1] Localizing the main car in: {image_path}")
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
    
    print(f" -> Crop Box Dimensions: {crop_width}x{crop_height} (Aspect Ratio: {aspect_ratio:.2f})")
    
    # Global offset coordinates of our crop
    crop_x_offset, crop_y_offset, _, _ = global_box
    
    # ----------------------------------------------------
    # STAGE 4: Physical Heuristic-Based Decision Tree
    # ----------------------------------------------------
    
    # CASE A: SQUARE CROP (Must be Front or Rear view)
    # -> Handled robustly by SigLIP 2 using PROMPT_SET_1
    if aspect_ratio < 1.5:
        print(" [Heuristic] Vehicle profile is square. Routing to SigLIP 2 with official prompts...")
        
        # Gather the template strings
        candidate_labels = list(PROMPT_MAP.values())
        
        # Run classification
        siglip_results = siglip_classifier(crop, candidate_labels=candidate_labels)
        
        print("\n--- [DEBUG] SigLIP 2 Official Template Prediction Scores ---")
        pp.pprint(siglip_results)
        print("-------------------------------------------------------------\n")
        
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
            print("   -> SigLIP 2 matched 'null' template. Forcing tie-breaker between Front and Rear...")
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
    # -> Handled deterministically by relative-coordinate light math to bypass spatial blindness
    else:
        print(" [Heuristic] Vehicle profile is rectangular. Running local-part coordinate scanning...")
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
        print(f" -> Visualization saved to: {output_path}")
        
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
# RUN THE COMPLETE PIPELINE
# ----------------------------------------------------
print("\n--- Starting Hybrid Pipeline (OWL-ViT + SigLIP 2 Templates) ---")

path_rear = "/home/aziz/Pictures/Internship_Images/rear.jpg"
print(f"Prediction for Rear:  {classify_direction_two_stage(path_rear, 'debug_rear_two_stage.jpg')}")

path_left = "/home/aziz/Pictures/Internship_Images/left.jpg"
print(f"Prediction for Left:  {classify_direction_two_stage(path_left, 'debug_left_two_stage.jpg')}")

path_right = "/home/aziz/Pictures/Internship_Images/right.jpg"
print(f"Prediction for Right: {classify_direction_two_stage(path_right, 'debug_right_two_stage.jpg')}")

path_front = "/home/aziz/Pictures/Internship_Images/front.jpg"
print(f"Prediction for Front: {classify_direction_two_stage(path_front, 'debug_front_two_stage.jpg')}")