import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import time
import json

# ============================================================
# CUSTOM DATASET FOR YOLO FORMAT
# ============================================================
class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, class_names, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.transform = transform
        
        # Get all image files
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"📊 Found {len(self.image_files)} images in {os.path.basename(images_dir)}")
        
        # Verify corresponding label files exist
        self.valid_samples = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(labels_dir, base_name + '.txt')
            
            if os.path.exists(label_file):
                self.valid_samples.append((img_file, label_file))
            else:
                print(f"⚠️  Label file not found for: {img_file}")
        
        print(f"✅ Valid samples with labels: {len(self.valid_samples)}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        img_file, label_file = self.valid_samples[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        
        # Load label and get class from YOLO format
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # For classification, we use the first object's class
        if lines:
            # YOLO format: class x_center y_center width height
            first_line = lines[0].strip().split()
            if first_line:
                class_id = int(first_line[0])
                # Ensure class_id is within valid range
                class_id = min(class_id, len(self.class_names) - 1)
            else:
                class_id = 0  # Default to first class if no valid label
        else:
            class_id = 0  # Default to first class if no labels
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, class_id

# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), (correct / total) * 100

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation"):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Use autocast for validation too for consistency
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), (correct / total) * 100

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    # ============================================================
    # GPU OPTIMIZATION SETTINGS
    # ============================================================
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {DEVICE}")

    if torch.cuda.is_available():
        print(f"🎯 GPU: {torch.cuda.get_device_name()}")
        print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ============================================================
    # LOAD YAML & RESOLVE PATHS
    # ============================================================
    def load_yaml(yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return data

    yaml_path = "D:/Kuliah/SEMESTER 5/Project/AI BONE GAN/data/MRI/Bone Fractures Detection/data.yaml"
    data_cfg = load_yaml(yaml_path)

    # Get class names from YAML
    class_names = data_cfg['names']
    num_classes = len(class_names)
    print(f"🎯 Classes: {class_names}")
    print(f"🎯 Number of classes: {num_classes}")

    # Manual paths based on your structure
    base_dir = "D:/Kuliah/SEMESTER 5/Project/AI BONE GAN/data/MRI/Bone Fractures Detection"
    train_images_dir = os.path.join(base_dir, "train", "images")
    train_labels_dir = os.path.join(base_dir, "train", "labels")
    valid_images_dir = os.path.join(base_dir, "valid", "images") 
    valid_labels_dir = os.path.join(base_dir, "valid", "labels")
    test_images_dir = os.path.join(base_dir, "test", "images")
    test_labels_dir = os.path.join(base_dir, "test", "labels")

    print("📁 Dataset paths:")
    print(f"   Train Images: {train_images_dir} - Exists: {os.path.exists(train_images_dir)}")
    print(f"   Train Labels: {train_labels_dir} - Exists: {os.path.exists(train_labels_dir)}")
    print(f"   Valid Images: {valid_images_dir} - Exists: {os.path.exists(valid_images_dir)}")
    print(f"   Valid Labels: {valid_labels_dir} - Exists: {os.path.exists(valid_labels_dir)}")
    print(f"   Test Images:  {test_images_dir} - Exists: {os.path.exists(test_images_dir)}")
    print(f"   Test Labels:  {test_labels_dir} - Exists: {os.path.exists(test_labels_dir)}")

    # ============================================================
    # TRANSFORMS
    # ============================================================
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # ============================================================
    # CREATE DATASETS & DATALOADERS
    # ============================================================
    print("\n📁 Creating datasets...")
    train_dataset = YOLODataset(train_images_dir, train_labels_dir, class_names, transform=train_transforms)
    valid_dataset = YOLODataset(valid_images_dir, valid_labels_dir, class_names, transform=val_transforms)
    test_dataset = YOLODataset(test_images_dir, test_labels_dir, class_names, transform=val_transforms)

    print(f"📊 Dataset sizes:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(valid_dataset)} samples") 
    print(f"   Test: {len(test_dataset)} samples")

    # Check class distribution
    def check_class_distribution(dataset, name):
        class_counts = {class_id: 0 for class_id in range(len(class_names))}
        for _, label in dataset:
            class_counts[label] += 1
        
        print(f"\n📊 Class distribution in {name}:")
        for class_id, count in class_counts.items():
            class_name = class_names[class_id]
            print(f"   {class_name}: {count} samples ({count/len(dataset)*100:.1f}%)")

    check_class_distribution(train_dataset, "Training")
    check_class_distribution(valid_dataset, "Validation")
    check_class_distribution(test_dataset, "Test")

    # DataLoaders with FIXED multiprocessing settings for Windows
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0,  # ✅ FIXED: Set to 0 for Windows to avoid multiprocessing issues
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0,  # ✅ FIXED: Set to 0 for Windows
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0,  # ✅ FIXED: Set to 0 for Windows
        pin_memory=True
    )

    # ============================================================
    # MODEL
    # ============================================================
    def create_model(num_classes):
        model = models.resnet18(weights="IMAGENET1K_V1")
        
        # Freeze early layers
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace classifier
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        return model

    model = create_model(num_classes)
    model = model.to(DEVICE)

    # Mixed precision
    scaler = torch.amp.GradScaler(device=DEVICE.type)

    # Class weights for imbalanced data
    class_counts = [153, 75, 54, 21, 297, 48, 12, 66, 531, 90]  # From your training distribution
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(DEVICE)

    # Loss and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    print(f"📦 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📦 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    EPOCHS = 50
    best_val_acc = 0
    patience = 7
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("\n🚀 Starting Training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"🎯 Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*50}")

        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_acc = validate(model, valid_loader, criterion, DEVICE)
        epoch_time = time.time() - epoch_start

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"✅ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"✅ Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.2f}%")
        print(f"⏰ Epoch Time: {epoch_time:.2f}s | LR: {current_lr:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'classes': class_names,
                'class_names': class_names,
                'class_weights': class_weights.cpu()
            }, "resnet18_mri_best.pth")
            print(f"💾 Model updated! Best Val Acc: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ No improvement: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"\n🏁 Training completed in {total_time/60:.2f} minutes")

    # ============================================================
    # FINAL TESTING
    # ============================================================
    print("\n🧪 Loading best model for testing...")
    checkpoint = torch.load("resnet18_mri_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print("\n🎉 FINAL TEST RESULTS:")
    print(f"📊 Test Loss: {test_loss:.4f}")
    print(f"📊 Test Accuracy: {test_acc:.2f}%")
    print(f"📊 Best Validation Accuracy: {checkpoint['best_val_acc']:.2f}%")

    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_accuracy': test_acc,
        'best_val_accuracy': checkpoint['best_val_acc'],
        'classes': class_names,
        'training_time_minutes': total_time / 60
    }

    with open("training_history.json", "w") as f:
        json.dump(training_history, f, indent=4)

    print("💾 Training history saved to training_history.json")

    # Print final class-wise accuracy
    print("\n📈 Final Model Performance:")
    print(f"✅ Training completed successfully!")
    print(f"✅ Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    print(f"✅ Test accuracy: {test_acc:.2f}%")
    print(f"✅ Total training time: {total_time/60:.2f} minutes")

    torch.cuda.empty_cache()
    print("🧹 GPU memory cleaned up")

# ============================================================
# MAIN GUARD FOR WINDOWS MULTIPROCESSING
# ============================================================
if __name__ == '__main__':
    main()