import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.001
DATASET_DIR = './custom_car_dataset'

# 1. Standard Preprocessing Transforms for ImageNet architectures
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        # WARNING: We intentionally exclude random horizontal flipping!
        # Flipping a left-view car creates a right-view vehicle profile, which breaks training data integrity.
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 2. PyTorch Data Loaders
print("Loading crop assets into PyTorch datasets...")
image_datasets = {x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"Detected target classes: {class_names}")
print(f"Training subset volume: {dataset_sizes['train']} images | Validation subset volume: {dataset_sizes['val']} images")

# Establish Execution Device Context
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running processing tasks inside runtime context: {device}")

# 3. Import Architecture and Reconfigure Heads
print("Building ResNet-18 model framework...")
# Using modern weights initialization standard parameters
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze convolutional structural layers to preserve underlying generic feature extraction
for param in model.parameters():
    param.requires_grad = False

# Attach custom 4-way linear prediction layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# 4. Core Model Optimization Execution Block
print("\n--- Optimization Pipeline Execution Initiated ---")
best_acc = 0.0

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    print('-' * 15)
    
    # Each epoch undergoes separate training and evaluation loops
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward computation track
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward optimization parameters calculation only during train sequence
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = (running_corrects.double() / dataset_sizes[phase]) * 100
        
        print(f'{phase.capitalize():<5} Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%')
        
        # Snapshot track model weights tracking performance peaks
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'vehicle_orientation_resnet18.pth')

print("\n" + "="*40 + " OPTIMIZATION COMPLETE " + "="*40)
print(f"Optimal Validation Subset Accuracy Score Achieved: {best_acc:.2f}%")
print("Target weights saved successfully as: 'vehicle_orientation_resnet18.pth'")
print("="*103)