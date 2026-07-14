import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

print("Loading OWL-ViT...")
detector = pipeline(
    model="google/owlvit-base-patch32", 
    task="zero-shot-object-detection"
)

def classify_direction_and_save_debug(image_path, output_path="debug_output.jpg"):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    # Run OWL-ViT
    predictions = detector(
        image,
        candidate_labels=["a car headlight", "a car taillight"],
        threshold=0.12  # Cleaned up threshold slightly
    )
    
    valid_detections = []
    
    # 1. First Pass: Filter out background clutter close to edges (ROI Filter)
    margin_x = width * 0.08  # Ignore things in the outer 8% of the left/right frame
    
    for pred in predictions:
        label = pred["label"]
        score = pred["score"]
        box = pred["box"]
        
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        center_x = (xmin + xmax) / 2
        
        # Check if the detection belongs to background cars near the edges
        if center_x < margin_x or center_x > (width - margin_x):
            continue  # Skip background cars!
            
        valid_detections.append({
            "label": label,
            "score": score,
            "box": [xmin, ymin, xmax, ymax],
            "center_x": center_x
        })

    # 2. Second Pass: Resolve duplicates (If headlight & taillight overlap, keep highest score)
    clean_detections = []
    resolved_indices = set()
    
    for i, det_a in enumerate(valid_detections):
        if i in resolved_indices:
            continue
            
        box_a = det_a["box"]
        best_det = det_a
        
        for j, det_b in enumerate(valid_detections):
            if i == j:
                continue
            box_b = det_b["box"]
            
            # Simple intersection over bounding boxes (overlapping check)
            x_left = max(box_a[0], box_b[0])
            y_top = max(box_a[1], box_b[1])
            x_right = min(box_a[2], box_b[2])
            y_bottom = min(box_a[3], box_b[3])
            
            if x_right > x_left and y_bottom > y_top:  # They overlap!
                resolved_indices.add(j)
                if det_b["score"] > best_det["score"]:
                    best_det = det_b
                    
        clean_detections.append(best_det)

    # 3. Process the Cleaned Detections
    headlights = []
    taillights = []
    
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    print(f"\nFiltered Detections for {image_path}:")
    for det in clean_detections:
        label = det["label"]
        score = det["score"]
        box = det["box"]
        center_x = det["center_x"]
        
        print(f" - Labeled: {label} ({score:.2f}) at center-X: {center_x:.1f}")
        
        if label == "a car headlight":
            headlights.append(center_x)
            box_color = "green"
        elif label == "a car taillight":
            taillights.append(center_x)
            box_color = "red"
            
        # Draw on image
        draw.rectangle(box, outline=box_color, width=3)
        draw.text((box[0], max(0, box[1] - 15)), f"{label} ({score:.2f})", fill=box_color, font=font)
        draw.ellipse([center_x - 4, box[1] - 4, center_x + 4, box[1] + 4], fill="yellow")

    image.save(output_path)
    print(f" Saved clean debug visualization to: {output_path}")

    # --- GEOMETRIC DECISION SYSTEM ---
    num_heads = len(headlights)
    num_tails = len(taillights)
    
    # Handle pure front/rear scenarios directly
    if num_heads >= 2 and num_tails == 0:
        return "front"
    if num_tails >= 2 and num_heads == 0:
        return "rear"
        
    # Side views
    if headlights and taillights:
        avg_headlights = np.mean(headlights)
        avg_taillights = np.mean(taillights)
        
        if avg_headlights < avg_taillights:
            return "left"
        else:
            return "right"
            
    elif headlights and not taillights:
        return "front"
    elif taillights and not headlights:
        return "rear"
        
    return "null"

# Run your tests
print(classify_direction_and_save_debug("/home/aziz/Pictures/Internship_Images/rear.jpg", "debug_rear_clean.jpg"))
print(classify_direction_and_save_debug("/home/aziz/Pictures/Internship_Images/left.jpg", "debug_left_clean.jpg"))
print(classify_direction_and_save_debug("/home/aziz/Pictures/Internship_Images/right.jpg", "debug_right_clean.jpg"))
print(classify_direction_and_save_debug("/home/aziz/Pictures/Internship_Images/front.jpg", "debug_front_clean.jpg"))