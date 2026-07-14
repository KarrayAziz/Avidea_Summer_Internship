import time
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def run_benchmark():
    # 1. Configuration
    BATCH_SIZE = 32  # You can change this to 1, 16, 32, or 64 to compare!
    CLASS_NAMES = ['back', 'front', 'left', 'right']
    TEST_DATASET_DIR = Path("/home/aziz/Aziz/DigiCover/usingGeminiApi/test_dataset_unseen")
    WEIGHTS_PATH = Path("/home/aziz/Aziz/DigiCover/usingGeminiApi/scripts/vehicle_orientation_resnet18_finetuned.pth")

    print("=" * 60)
    print(f"  CPU BENCHMARKING SESSION (Batch Size: {BATCH_SIZE})")
    print("=" * 60)

    # ==========================================
    # 2. MEASURE MODEL LOADING TIME (Cold Start)
    # ==========================================
    print("\n[1/4] Measuring Model Loading Time...")
    t_start_load = time.perf_counter()

    # Rebuild architecture
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)

    # Load weights specifically onto CPU
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"⚠️ Could not find weight file at: {WEIGHTS_PATH}")
        
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.eval()  # Crucial: turn off dropout/batchnorm updates

    t_end_load = time.perf_counter()
    load_time = t_end_load - t_start_load
    print(f"  ✓ Model loaded and initialized in: {load_time:.4f} seconds")

    # ==========================================
    # 3. PREPARE DATASET & DATALOADER
    # ==========================================
    print("\n[2/4] Preparing Unseen Test Dataset...")
    inference_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not TEST_DATASET_DIR.exists():
        raise FileNotFoundError(f"⚠️ Missing test folder at: {TEST_DATASET_DIR}")

    test_dataset = datasets.ImageFolder(TEST_DATASET_DIR, transform=inference_transforms)
    # num_workers=0 is best on CPU VPS to avoid multi-processing overhead
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    total_images = len(test_dataset)
    total_batches = len(dataloader)
    print(f"  ✓ Found {total_images} images across {total_batches} batches.")

    # ==========================================
    # 4. CPU WARM-UP PHASE
    # ==========================================
    # We do a brief run to prime CPU caches and allow PyTorch to optimize its graph path
    print("\n[3/4] Warming up CPU caches...")
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    print("  ✓ Warm-up complete.")

    # ==========================================
    # 5. MEASURE STEADY-STATE BATCH INFERENCE
    # ==========================================
    print("\n[4/4] Executing Batch Inference Benchmark...")
    total_inference_time = 0.0

    # Ensure gradients are completely turned off (massively saves CPU memory/ops)
    with torch.no_grad():
        for inputs, _ in dataloader:
            # If the last batch is smaller than BATCH_SIZE, it still runs fine
            t_start_batch = time.perf_counter()
            _ = model(inputs)
            t_end_batch = time.perf_counter()
            
            total_inference_time += (t_end_batch - t_start_batch)

    # ==========================================
    # 6. COMPUTE METRICS
    # ==========================================
    avg_batch_time_ms = (total_inference_time / total_batches) * 1000
    avg_image_time_ms = (total_inference_time / total_images) * 1000
    fps = total_images / total_inference_time

    print("\n" + "═"*50)
    print("                 BENCHMARK REPORT                 ")
    print("═"*50)
    print(f" Hardware Target           : CPU-Only")
    print(f" Model Load Time (Cold)    : {load_time:.4f} seconds")
    print(f" Total Inference Time      : {total_inference_time:.4f} seconds")
    print(f" Total Images Processed    : {total_images}")
    print(f" Batch Size Used           : {BATCH_SIZE}")
    print(f"--------------------------------------------------")
    print(f" Avg Latency per Batch     : {avg_batch_time_ms:.2f} ms")
    print(f" Avg Latency per Image     : {avg_image_time_ms:.2f} ms")
    print(f" Production Throughput     : {fps:.2f} frames/sec (FPS)")
    print("═"*50 + "\n")

if __name__ == "__main__":
    run_benchmark()