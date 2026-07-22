import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

def get_dominant_paint_color(image_path, yolo_model, confidence_threshold=0.30):
    """
    Crops the vehicle, filters out background/shadow/trim noise via Lab thresholds,
    and uses K-Means clustering to find the true dominant paint color in LAB space.
    """
    img_pil = Image.open(image_path).convert("RGB")
    img_w, img_h = img_pil.size
    total_image_area = img_w * img_h
    
    yolo_results = yolo_model(image_path, conf=confidence_threshold, classes=[2, 5, 7], verbose=False)
    
    img_cv = cv2.imread(str(image_path))
    if img_cv is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
        
    use_full_image = True
    cropped_img = img_cv
    area_percentage = 0.0

    if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
        boxes = yolo_results[0].boxes
        xyxy = boxes.xyxy
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        areas = widths * heights
        largest_idx = areas.argmax().item()
        area_percentage = (areas[largest_idx].item() / total_image_area) * 100
        
        if area_percentage >= 15.0:
            use_full_image = False
            x1, y1, x2, y2 = map(int, xyxy[largest_idx].tolist())
            # Shave 10% off edges to aggressively eliminate background bleeding
            h_b, w_b = y2 - y1, x2 - x1
            cropped_img = img_cv[y1+int(h_b*0.1):y2-int(h_b*0.1), x1+int(w_b*0.1):x2-int(w_b*0.1)]

    file_name = Path(image_path).name
    if not use_full_image:
        print(f"✂️  [{file_name}] Successfully isolated vehicle via YOLO crop.")
    else:
        print(f"⚠️  [{file_name}] Vehicle too small. Using full frame.")

    # 1. Convert to CIELAB color space (L=Lightness, A=Green-Red, B=Blue-Yellow)
    lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2Lab)
    
    # 2. Filter out explicit black trim, silver chrome, white reflections, and asphalt grays
    # In OpenCV LAB: L (0-255), A (0-255), B (0-255) where neutral gray/achromatic centers sit around 128
    lab_flat = lab.reshape(-1, 3)
    
    # Keep pixels that show clear chromatic variance (A or B channel pushing away from neutral 128)
    # OR if it's an achromatic car, keep the core middle lightness band
    a_dist = np.abs(lab_flat[:, 1] - 128)
    b_dist = np.abs(lab_flat[:, 2] - 128)
    color_pixels = lab_flat[(a_dist > 8) | (b_dist > 8)]
    
    # Fallback if the car is genuinely grayscale (Black/White/Gray)
    if len(color_pixels) < (lab_flat.shape[0] * 0.02):
        # Target the core body lightness, ignoring extreme glare (top) or deep shadow cavities (bottom)
        color_pixels = lab_flat[(lab_flat[:, 0] > 30) & (lab_flat[:, 0] < 220)]

    if len(color_pixels) == 0:
        color_pixels = lab_flat # absolute emergency fallback

    # 3. Apply K-Means to find the primary dominant color cluster
    color_pixels = np.float32(color_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    # We look for K=2 clusters (usually splits paint color from remaining window/tire tint fragments)
    _, labels, centers = cv2.kmeans(color_pixels, 2, None, criteria, 10, flags)
    
    # Find the cluster center that has the highest saturation/chromaticity profile
    chroma = np.abs(centers[:, 1] - 128) + np.abs(centers[:, 2] - 128)
    dominant_color_lab = centers[chroma.argmax()]
    
    return dominant_color_lab

def verify_insurance_upload_color(image_paths, max_distance_threshold=18.0):
    """
    Validates color consistency using Delta E distance of K-Means color centers.
    """
    if len(image_paths) != 4:
        raise ValueError("The fraud check requires exactly 4 view images.")

    print("🤖 Loading YOLOv8m for background-filtering...")
    yolo_model = YOLO("yolov8m.pt")
    print("-" * 65)
    
    # 1. Extract dominant LAB vectors
    lab_centers = []
    for path in image_paths:
        lab_centers.append(get_dominant_paint_color(path, yolo_model))
        
    print("-" * 65)
    print("🔄 Running pairwise Delta-E color distance matrix...")
    
    distances = []
    view_names = [Path(p).name for p in image_paths]
    
    for i in range(len(lab_centers)):
        for j in range(i + 1, len(lab_centers)):
            # Calculate standard Euclidean distance in LAB space (Approximates Delta E)
            # We down-weight the L (Lightness) component by 50% to stay invariant to shadows!
            lab1 = lab_centers[i]
            lab2 = lab_centers[j]
            
            dL = (lab1[0] - lab2[0]) * 0.50  # Minimize shadow brightness penalty directly
            da = lab1[1] - lab2[1]
            db = lab1[2] - lab2[2]
            
            delta_e = np.sqrt(dL**2 + da**2 + db**2)
            distances.append(delta_e)
            print(f"   ├─ {view_names[i]} 🆚 {view_names[j]} -> Color Distance (ΔE): {delta_e:.2f}")
            
    worst_distance = max(distances)
    avg_distance = sum(distances) / len(distances)
    
    print("-" * 65)
    print(f"📊 FINAL COLOR DISTANCE REPORT:")
    print(f"   ├─ Worst Pairwise Distance (ΔE): {worst_distance:.2f} (Max Allowed is {max_distance_threshold})")
    print(f"   └─ Average Distance (ΔE):        {avg_distance:.2f}")
    print("-" * 65)
    
    # Note: Smaller distance = closer color match!
    if worst_distance > max_distance_threshold:
        print("❌ FRAUD/ERROR ALERT: Significant color variance detected between views!")
        return False
        
    print("✅ VALIDATION PASSED: All 4 images confirmed to match core color family.")
    return True

if __name__ == "__main__":
    client_bundle = [
        "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/Orange_Minicooper/back.jpg",
        "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/Orange_Minicooper/front.jpg",
        "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/Orange_Minicooper/left.jpg",
        "/home/aziz/Pictures/Internship_Images/detetction des faces/Ready_Cars/Orange_Minicooper/right.jpg"
    ]
    
    # 18.0 is an industry-standard delta threshold allowing for heavy environmental lighting variance
    verify_insurance_upload_color(client_bundle, max_distance_threshold=18.0)