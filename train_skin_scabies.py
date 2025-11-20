import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ================================================
# 1️⃣ KONFIGURASI DASAR
# ================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "D:/Kuliah/SEMESTER 5/Project/AI BONE GAN/data/SkinDisNet"
WEIGHTS_DIR = "weights"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "skin_model.pth")

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ================================================
# 2️⃣ TRANSFORMASI DATA
# ================================================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================================================
# 3️⃣ DATASET & DATALOADER
# ================================================
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================================================
# 4️⃣ MODEL TRANSFER LEARNING (ResNet18)
# ================================================
model = models.resnet18(weights="IMAGENET1K_V1")  # pakai pretrained
for param in model.parameters():
    param.requires_grad = False  # freeze semua layer awal

# Ganti FC layer terakhir agar output = 2 kelas
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(DEVICE)

# ================================================
# 5️⃣ LOSS & OPTIMIZER
# ================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# ================================================
# 6️⃣ TRAINING LOOP
# ================================================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 30)

    # TRAINING
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_loss = running_loss / total

    # VALIDASI
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data).item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    val_loss = val_loss / val_total

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

# ================================================
# 7️⃣ SIMPAN MODEL
# ================================================
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ Model tersimpan di: {MODEL_PATH}")
