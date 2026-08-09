import os
import shutil
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from ultralytics import YOLO

# 📌 Configuration paths
TEST_DIR = "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/data/Clean_Inference_Set"
WEIGHTS_PATH = "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/models/vehicle_orientation_resnet18_finetuned.pth"
DEBUG_OUTPUT_DIR = "/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/scripts/avidea_tasks/view_classification/view_classification_results_debugging"
BATCH_SIZE = 32

# Pipeline (Applied dynamically after on-the-fly cropping)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class OnTheFlyCroppedDataset(Dataset):
    """
    Custom Dataset that mirrors the exact YOLO cropping logic from prep work
    dynamically at runtime without saving temporary data to disk.
    """
    def __init__(self, root_dir, transform=None, confidence_threshold=0.30):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = ['back', 'front', 'left', 'right']
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        # Load YOLO model (Matches preprocessing script setup)
        print(f"🚀 Loading YOLOv8m for on-the-fly preprocessing...")
        self.yolo_model = YOLO("/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/models/yolov8m.pt")
        self.confidence_threshold = confidence_threshold
        
        # Discover file paths manually to match ImageFolder functionality
        self.samples = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        for class_name in self.class_names:
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                continue
            for fname in os.listdir(class_folder):
                if os.path.splitext(fname)[1].lower() in valid_extensions:
                    fpath = os.path.join(class_folder, fname)
                    self.samples.append((fpath, self.class_to_idx[class_name]))
                    
        print(f"📊 Discovered {len(self.samples)} images across classes in: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Open pristine source image
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        total_image_area = img_w * img_h
        
        # Target cars (2), buses (5), and trucks (7)
        yolo_results = self.yolo_model(img_path, conf=self.confidence_threshold, classes=[2, 5, 7], verbose=False)
        
        # Exact preprocessing logic fallback check
        if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes
            xyxy = boxes.xyxy
            
            # Find largest vehicle in frame
            widths = xyxy[:, 2] - xyxy[:, 0]
            heights = xyxy[:, 3] - xyxy[:, 1]
            areas = widths * heights
            
            largest_idx = areas.argmax().item()
            largest_area = areas[largest_idx].item()
            
            # Calculate coverage percentage
            area_percentage = (largest_area / total_image_area) * 100
            
            # --- 15% AREA CHECK CRITERIA ---
            if area_percentage >= 8.0:
                x1, y1, x2, y2 = map(int, xyxy[largest_idx].tolist())
                img = img.crop((x1, y1, x2, y2))
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, img_path


def isolate_misclassifications():
    # 1. Setup target directory system
    class_names = ['back', 'front', 'left', 'right']
    for view in class_names:
        folder_path = os.path.join(DEBUG_OUTPUT_DIR, f"misclassified_{view}")
        os.makedirs(folder_path, exist_ok=True)
    
    print(f"📁 Initialized diagnostic root at: {DEBUG_OUTPUT_DIR}")
    print("-" * 65)

    # 2. Build custom pre-processing on-the-fly pipeline
    # NOTE: num_workers must be 0 when mixing Ultralytics YOLO with PyTorch Dataloaders to prevent CUDA fork errors
    test_dataset = OnTheFlyCroppedDataset(TEST_DIR, transform=test_transforms, confidence_threshold=0.30)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 3. Initialize ResNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    misclassified_count = 0

    print("\n🔮 Running inference across aligned, cropped data matrix...")
    
    with torch.no_grad():
        for inputs, labels, paths in tqdm(test_loader, desc="Evaluating Batches", unit="batch"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            preds_np = preds.cpu().numpy()
            labels_np = labels.numpy()
            
            all_preds.extend(preds_np)
            all_labels.extend(labels_np)
            
            # Identify alignment anomalies
            for i in range(len(labels_np)):
                true_idx = labels_np[i]
                pred_idx = preds_np[i]
                
                if true_idx != pred_idx:
                    misclassified_count += 1
                    true_view = class_names[true_idx]
                    pred_view = class_names[pred_idx]
                    
                    original_file_path = paths[i]
                    base_filename = os.path.basename(original_file_path)
                    destination_folder = os.path.join(DEBUG_OUTPUT_DIR, f"misclassified_{true_view}")
                    
                    # 📸 1. Save the Raw Pristine Image
                    raw_name = f"{true_view}_classified_as_{pred_view}_RAW_{base_filename}"
                    shutil.copy2(original_file_path, os.path.join(destination_folder, raw_name))
                    
                    # 🤖 2. Re-evaluate YOLO to generate the cropped asset and box labels for visualization
                    yolo_res = test_dataset.yolo_model(original_file_path, conf=0.30, classes=[2, 5, 7], verbose=False)
                    
                    # Save the marked box layout
                    annotated_img = yolo_res[0].plot()
                    box_name = f"{true_view}_classified_as_{pred_view}_YOLOBOX_{base_filename}"
                    cv2.imwrite(os.path.join(destination_folder, box_name), annotated_img)
                    
                    # Save the cropped asset that was sent directly to ResNet
                    if len(yolo_res) > 0 and len(yolo_res[0].boxes) > 0:
                        boxes = yolo_res[0].boxes.xyxy
                        widths = boxes[:, 2] - boxes[:, 0]
                        heights = boxes[:, 3] - boxes[:, 1]
                        areas = widths * heights
                        l_idx = areas.argmax().item()
                        
                        raw_cv_img = cv2.imread(original_file_path)
                        h, w, _ = raw_cv_img.shape
                        if (areas[l_idx].item() / (w * h)) * 100 >= 15.0:
                            x1, y1, x2, y2 = map(int, boxes[l_idx].tolist())
                            cropped_cv_img = raw_cv_img[y1:y2, x1:x2]
                            crop_name = f"{true_view}_classified_as_{pred_view}_CROP_{base_filename}"
                            cv2.imwrite(os.path.join(destination_folder, crop_name), cropped_cv_img)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. Generate Performance Metrics
    print("\n" + "="*20 + " UNBIASED TEST REPORT " + "="*20)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=2))
    print("="*62)

    # 5. Confusion Matrix Dashboard
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n📊 CONFUSION MATRIX DASHBOARD")
    print("-" * 45)
    header = f"{'True \\ Pred':<12}" + "".join([f"{name:<10}" for name in class_names])
    print(header)
    print("-" * 45)
    
    for i, class_name in enumerate(class_names):
        row_str = f"{class_name:<12}" + "".join([f"{cm[i][j]:<10}" for j in range(len(class_names))])
        print(row_str)
    print("-" * 45)
    
    print(f"\n🎉 Analysis Complete! Debugging pairs located inside folder: {DEBUG_OUTPUT_DIR}")


if __name__ == "__main__":
    isolate_misclassifications()