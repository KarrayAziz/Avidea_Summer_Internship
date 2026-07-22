import re
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from rapidocr_onnxruntime import RapidOCR  # EasyOCR import removed

class TunisianPlateExtractor:
    def __init__(self):
        print("Initializing Tunisian YOLOv8-Nano License Plate Detector...")
        try:
            model_path = hf_hub_download(
                repo_id="Malek-Messaoudi/Tunisian-licence-plate-detection", 
                filename="licence-plate-detection.pt"
            )
            self.detector = YOLO(model_path)
            print(f"  ✅ Tunisian Plate Model loaded successfully!")
        except Exception as e:
            print(f"  ⚠️ Error loading YOLO model: {e}")
            self.detector = None

        print("Initializing RapidOCR Reader...")
        # RapidOCR is lightweight, fast, and runs fully on CPU via ONNX
        self.reader = RapidOCR()
        print("  ✅ RapidOCR initialized successfully!")

    def localize_and_crop_plate(self, image_path: str, save_crops_dir: str = None):
        if not self.detector:
            return None
            
        results = self.detector(image_path, verbose=False)
        img = cv2.imread(image_path)
        h_img, w_img = img.shape[:2]

        if len(results) > 0 and len(results[0].boxes) > 0:
            best_box = sorted(results[0].boxes, key=lambda b: b.conf[0], reverse=True)[0]
            xyxy = best_box.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = map(int, xyxy)
            
            # Save the YOLO detection visualization
            annotated_img = img.copy()
            cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            # Label text with confidence
            label = f"Plate: {best_box.conf[0]:.2f}"
            cv2.putText(annotated_img, label, (x_min, max(0, y_min - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if save_crops_dir:
                os.makedirs(save_crops_dir, exist_ok=True)
                base_name = Path(image_path).stem
                annotated_path = os.path.join(save_crops_dir, f"{base_name}_detected.jpg")
                cv2.imwrite(annotated_path, annotated_img)
            else:
                cv2.imwrite("debug_yolo_detection.jpg", annotated_img)

            # Crop with a comfortable margin for OCR
            pad_w = int((x_max - x_min) * 0.05)
            pad_h = int((y_max - y_min) * 0.05)
            
            y_start = max(0, y_min - pad_h)
            y_end = min(h_img, y_max + pad_h)
            x_start = max(0, x_min - pad_w)
            x_end = min(w_img, x_max + pad_w)
            
            return img[y_start:y_end, x_start:x_end]
        return None

    def extract_plate(self, image_path: str, save_crops_dir: str = None) -> str:
        try:
            # 1. Localize plate using YOLOv11 and pass save_crops_dir
            plate_crop = self.localize_and_crop_plate(image_path, save_crops_dir)
            if plate_crop is None:
                print("  ⚠️ No plate localized by YOLOv11.")
                return None

            # Convert to Grayscale
            gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            crop_h, crop_w = gray_crop.shape[:2]

            # 2. Invert polarity (makes black-plate text clean black-on-white)
            inverted_crop = cv2.bitwise_not(gray_crop)
            
            # Prepare visualization copy for text bounding boxes
            segmented_visualization = plate_crop.copy()

            # 3. Feed the inverted crop to RapidOCR
            ocr_results, elapse = self.reader(inverted_crop)

            if not ocr_results:
                print("\n  ⚠️ OCR returned no results.")
                return None

            print("\n  🔍 Raw OCR Detections (Before Filtering):")
            all_raw_text_joined = ""
            for line in ocr_results:
                bbox = line[0]        # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                text = line[1]
                confidence = float(line[2])
                print(f"    - Found '{text}' | Conf: {confidence:.2f}")
                all_raw_text_joined += " " + text

                # Draw the detected segmentation polygon on the visualization image
                pts = np.array(bbox, np.int32)
                pts = pts.reshape((-1, 1, 2))  # Reshaped correctly for OpenCV
                cv2.polylines(segmented_visualization, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Overlay the recognized text above its respective box
                x_min_text = int(bbox[0][0])
                y_min_text = int(bbox[0][1])
                cv2.putText(segmented_visualization, text, (x_min_text, max(15, y_min_text - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            # Save crops, inverted, and segmented results
            if save_crops_dir:
                os.makedirs(save_crops_dir, exist_ok=True)
                base_name = Path(image_path).stem
                
                normal_path = os.path.join(save_crops_dir, f"{base_name}_crop.jpg")
                inverted_path = os.path.join(save_crops_dir, f"{base_name}_inverted.jpg")
                segmented_path = os.path.join(save_crops_dir, f"{base_name}_segmented.jpg")
                
                cv2.imwrite(normal_path, plate_crop)
                cv2.imwrite(inverted_path, inverted_crop)
                cv2.imwrite(segmented_path, segmented_visualization)
                print(f"  💾 Saved crops, inverted, and segmented visualization to '{save_crops_dir}/'")
            else:
                cv2.imwrite("debug_plate_crop.jpg", inverted_crop)
                cv2.imwrite("debug_plate_segmented.jpg", segmented_visualization)

            # Detect if this is an RS (Suspensive Diet) plate containing "ن ت" or "نت"
            is_rs_plate = False
            # Search for "ن ت" or "نت" with boundary words to avoid matching parts of "تونس"/"تونت"
            if re.search(r'\bن\s*ت\b', all_raw_text_joined) or re.search(r'\bنت\b', all_raw_text_joined):
                # Double check to prevent false positives from corrupted "Tunisia" reads like "تونت"
                # If "تو" (Teh + Waw) is right before it, it's likely a standard plate misread
                if not re.search(r'تو\s*ن\s*ت', all_raw_text_joined):
                    is_rs_plate = True

            digit_blocks = []
            for line in ocr_results:
                bbox = line[0]
                text = line[1]
                
                # RapidOCR format for bbox is: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                box_h = bbox[2][1] - bbox[0][1]
                height_ratio = box_h / crop_h
                
                # Keep only vertically significant text blocks
                if height_ratio > 0.35:
                    # Remove all Arabic Unicode characters, leaving only English numbers and spaces
                    clean_text = re.sub(r'[\u0600-\u06FF]+', ' ', text)
                    clean_text = " ".join(clean_text.split())
                    
                    if any(char.isdigit() for char in clean_text):
                        x_center = (bbox[0][0] + bbox[1][0]) / 2
                        digit_blocks.append((clean_text, x_center))

            # --- Route 1: RS Plate Processing ---
            if is_rs_plate:
                print("\n  ℹ️ Detected RS / Suspensive Diet Plate ('ن ت')")
                if len(digit_blocks) > 0:
                    # Sort digit blocks from left to right
                    digit_blocks.sort(key=lambda item: item[1])
                    # Join all numbers found across the blocks
                    rs_digits = "".join(["".join(re.findall(r'\d', block[0])) for block in digit_blocks])
                    print(f"  📍 Filtered Results -> Registration: '{rs_digits}' | Type: RS")
                    return f"{rs_digits} RS"
                else:
                    print("  ⚠️ Found 'ن ت' but could not isolate numeric digits.")
                    return None

            # --- Route 2: Standard "TU" Plate Processing ---
            # Resolve the Left (Series) and Right (Registration) blocks
            if len(digit_blocks) >= 2:
                digit_blocks.sort(key=lambda item: item[1])
                series = "".join(re.findall(r'\d', digit_blocks[0][0]))
                registration = "".join(re.findall(r'\d', digit_blocks[-1][0]))
                
                if len(series) > len(registration) and len(series) >= 4:
                    series, registration = registration, series

                print(f"\n  📍 Filtered Results -> Series: '{series}' | Registration: '{registration}'")
                return f"{registration}TU{series}"
                
            elif len(digit_blocks) == 1:
                cleaned_block = digit_blocks[0][0]
                parts = [ "".join(re.findall(r'\d', part)) for part in cleaned_block.split() if part.strip() ]
                parts = [p for p in parts if p]

                if len(parts) >= 2:
                    part_1, part_2 = parts[0], parts[1]
                    if len(part_1) > len(part_2):
                        registration = part_1
                        series = part_2
                    else:
                        registration = part_2
                        series = part_1
                        
                    print(f"\n  📍 Split Merged Block -> Series: '{series}' | Registration: '{registration}'")
                    return f"{registration}TU{series}"
                else:
                    all_digits = parts[0] if parts else "".join(re.findall(r'\d', cleaned_block))
                    if len(all_digits) >= 6:
                        series = all_digits[-3:]
                        registration = all_digits[:-3]
                        return f"{registration}TU{series}"
                    elif len(all_digits) == 5:
                        series = all_digits[-2:]
                        registration = all_digits[:-2]
                        return f"{registration}TU{series}"

            print("\n  ⚠️ Could not clearly resolve distinct large digit blocks.")
            return None

        except Exception as e:
            print(f"  ⚠️ Error processing plate OCR: {e}")
            return None