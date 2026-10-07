import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

# -------------------------------
# 1. Paths
# -------------------------------
data_dir = r"C:\Users\ndivi\SafetyAlertProject\data\gender"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# -------------------------------
# 2. Transformations
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# -------------------------------
# 3. Load datasets
# -------------------------------
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# --------- Reduce dataset size for testing ---------
train_dataset = Subset(train_dataset, range(min(2000, len(train_dataset))))  # first 2000 images
val_dataset = Subset(val_dataset, range(min(500, len(val_dataset))))         # first 500 images

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -------------------------------
# 4. Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 5. Model
# -------------------------------
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # male/female
model = model.to(device)

# -------------------------------
# 6. Loss and optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# -------------------------------
# 7. Training loop
# -------------------------------
num_epochs = 2  # reduce for quick test

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.4f}")

# -------------------------------
# 8. Save the model
# -------------------------------
model_path = os.path.join(r"C:\Users\ndivi\SafetyAlertProject\models", "gender_classifier_test.pth")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved at: {model_path}")
