import os
import time
import argparse
import shutil
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from ultralytics import YOLO

# 📌 Pipeline Configurations (Applied dynamically after on-the-fly cropping)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

WEIGHTS_PATH = "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/models/vehicle_orientation_resnet18_finetuned.pth"
YOLO_PATH = "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/models/yolov8m.pt"

# 📁 Output Directory Structure for Classifications
CLASSIFIED_OUTPUT_DIR = "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/scripts/avidea_tasks/view_classification/classified_output"

BATCH_SIZE = 32
CLASS_NAMES = ['back', 'front', 'left', 'right']


class UnlabeledFolderDataset(Dataset):
    """
    Custom Dataset designed to read from a flat folder of unclassified images
    without requiring any ground-truth target labels.
    """
    def __init__(self, folder_path, transform=None, confidence_threshold=0.30):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Provided path does not exist: {folder_path}")

        self.folder_path = folder_path
        self.transform = transform
        
        print(f"🚀 Loading YOLOv8m from {YOLO_PATH} for on-the-fly preprocessing...")
        self.yolo_model = YOLO(YOLO_PATH)
        self.confidence_threshold = confidence_threshold
        
        # Discover all valid format files directly within the targeted folder scope
        self.samples = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in valid_extensions:
                self.samples.append(fpath)
                    
        print(f"📊 Discovered {len(self.samples)} images to classify within: {folder_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        total_image_area = img_w * img_h
        
        yolo_results = self.yolo_model(img_path, conf=self.confidence_threshold, classes=[2, 5, 7], verbose=False)
        
        if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes
            xyxy = boxes.xyxy
            
            widths = xyxy[:, 2] - xyxy[:, 0]
            heights = xyxy[:, 3] - xyxy[:, 1]
            areas = widths * heights
            
            largest_idx = areas.argmax().item()
            largest_area = areas[largest_idx].item()
            area_percentage = (largest_area / total_image_area) * 100
            
            if area_percentage >= 15.0:
                x1, y1, x2, y2 = map(int, xyxy[largest_idx].tolist())
                img = img.crop((x1, y1, x2, y2))
        
        if self.transform:
            img = self.transform(img)
            
        return img, img_path


def run_folder_classification(target_folder):
    # 1. Initialize parent target sorting directories upfront
    for view in CLASS_NAMES:
        os.makedirs(os.path.join(CLASSIFIED_OUTPUT_DIR, f"classified_as_{view}"), exist_ok=True)
    
    print(f"📁 Sorted classifications destination root: {CLASSIFIED_OUTPUT_DIR}")
    print("-" * 65)

    # 2. Build unlabeled dataset instance
    test_dataset = UnlabeledFolderDataset(target_folder, transform=inference_transforms, confidence_threshold=0.30)
    if len(test_dataset) == 0:
        print("🛑 Process complete: No valid image assets found to classify.")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 3. Initialize ResNet model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    processed_count = 0
    batch_times = []

    print("\n🔮 Running pipeline classification sweep...")
    
    pbar = tqdm(test_loader, desc="Classifying Batches", unit="batch")
    
    with torch.no_grad():
        for inputs, paths in pbar:
            inputs = inputs.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            elapsed_time = end_time - start_time
            batch_times.append(elapsed_time)
            
            pbar.set_description(f"Classifying Batches (Last Batch Inf: {elapsed_time:.4f}s)")
            
            preds_np = preds.cpu().numpy()
            
            # Route predictions to sorting folders
            for i in range(len(preds_np)):
                pred_idx = preds_np[i]
                pred_view = CLASS_NAMES[pred_idx]
                
                original_file_path = paths[i]
                base_filename = os.path.basename(original_file_path)
                stem, ext = os.path.splitext(base_filename)
                
                target_class_folder = os.path.join(CLASSIFIED_OUTPUT_DIR, f"classified_as_{pred_view}")
                
                # 📸 1. Copy the original pristine image into its corresponding predicted folder
                shutil.copy2(original_file_path, os.path.join(target_class_folder, base_filename))
                
                # 🤖 2. Query YOLO structure to check the crops and bounding box annotation state
                yolo_res = test_dataset.yolo_model(original_file_path, conf=0.30, classes=[2, 5, 7], verbose=False)
                
                was_cropped = False
                if len(yolo_res) > 0 and len(yolo_res[0].boxes) > 0:
                    boxes = yolo_res[0].boxes.xyxy
                    widths = boxes[:, 2] - boxes[:, 0]
                    heights = boxes[:, 3] - boxes[:, 1]
                    areas = widths * heights
                    l_idx = areas.argmax().item()
                    
                    raw_cv_img = cv2.imread(original_file_path)
                    h, w, _ = raw_cv_img.shape
                    
                    # Verify threshold alignment matches dataset logic exactly
                    if (areas[l_idx].item() / (w * h)) * 100 >= 15.0:
                        was_cropped = True
                        x1, y1, x2, y2 = map(int, boxes[l_idx].tolist())
                        cropped_cv_img = raw_cv_img[y1:y2, x1:x2]
                        
                        # Ensure the nested /crops subfolder explicitly exists right now
                        crops_destination_folder = os.path.join(target_class_folder, "crops")
                        os.makedirs(crops_destination_folder, exist_ok=True)
                        
                        # ✂️ 3. Save the exact crop used into the nested /crops subfolder
                        crop_name = f"CROP_{base_filename}"
                        cv2.imwrite(os.path.join(crops_destination_folder, crop_name), cropped_cv_img)
                
                # 🏷️ Adjust suffix dynamically to denote threshold state
                crop_suffix = "_cropped" if was_cropped else "_fullframe"
                box_name = f"YOLOBOX{crop_suffix}_{stem}{ext}"
                
                # 🖼️ 4. Save the bounding box layout tracking file unconditionally inside the parent class folder
                annotated_img = yolo_res[0].plot()
                cv2.imwrite(os.path.join(target_class_folder, box_name), annotated_img)
                
                processed_count += 1

    # 4. Final Pipeline Summary
    avg_batch_time = np.mean(batch_times) if batch_times else 0.0
    print(f"\n🎉 Classification Sweep Finished!")
    print(f"   ├─ Total Images Processed: {processed_count}")
    print(f"   ├─ Avg Inference / Batch:  {avg_batch_time:.4f} seconds")
    print(f"   └─ Sorted Views Saved To:  {CLASSIFIED_OUTPUT_DIR}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run orientation classification on an unlabeled flat directory of images.")
    parser.add_argument("--folder", type=str, required=True, help="Absolute path to the unclassified image directory target.")
    
    args = parser.parse_args()
    run_folder_classification(args.folder)