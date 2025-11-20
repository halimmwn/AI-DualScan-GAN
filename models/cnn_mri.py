import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 1️⃣ LOAD MODEL MRI ======
def load_mri_model():
    """
    Load model MRI untuk klasifikasi 10 jenis fracture
    """
    # Load checkpoint
    checkpoint = torch.load("resnet18_mri_best.pth", map_location=DEVICE)
    
    # Get class names dari checkpoint
    class_names = checkpoint['classes']
    num_classes = len(class_names)
    
    # Create model architecture - SAMA PERSIS seperti saat training
    model = models.resnet18(weights=None)
    
    # Replace classifier dengan architecture yang sama
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"✅ MRI Model loaded successfully!")
    print(f"🎯 Number of classes: {num_classes}")
    print(f"🎯 Classes: {class_names}")
    print(f"📊 Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    return model, class_names

# ====== 2️⃣ PREPROSES GAMBAR DAN PREDIKSI ======
def predict_mri_condition(model, class_names, image):
    """
    Prediksi kondisi MRI untuk 10 jenis fracture
    
    Args:
        model: Model MRI yang sudah diload
        class_names: List nama kelas (10 jenis fracture)
        image: PIL Image
    
    Returns:
        tuple: (label, confidence, all_predictions)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # Preprocess image
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_t)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        
        # Get confidence untuk semua kelas
        all_confidences = {
            class_names[i]: f"{probabilities[0][i].item() * 100:.2f}%"
            for i in range(len(class_names))
        }

    label = class_names[pred.item()]
    confidence_value = confidence.item() * 100

    return label, confidence_value, all_confidences

# ====== 3️⃣ SIMPLE PREDICTION (Hanya label dan confidence) ======
def predict_mri_simple(model, class_names, image):
    """
    Prediksi sederhana - hanya return label dan confidence utama
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_t)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    label = class_names[pred.item()]
    confidence_value = confidence.item() * 100

    return label, confidence_value

# ====== 4️⃣ QUICK TEST FUNCTION ======
def test_mri_model():
    """
    Test function untuk memverifikasi model bekerja
    """
    print("🧪 Testing MRI Model...")
    
    # Load model
    model, class_names = load_mri_model()
    
    # Create dummy image untuk testing (hitam 224x224)
    dummy_image = Image.new('RGB', (224, 224), color='black')
    
    # Test prediction
    label, confidence, all_preds = predict_mri_condition(model, class_names, dummy_image)
    
    print(f"📊 Prediction: {label}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print(f"📊 All predictions: {all_preds}")
    
    # Test simple prediction
    simple_label, simple_conf = predict_mri_simple(model, class_names, dummy_image)
    print(f"📊 Simple Prediction: {simple_label} ({simple_conf:.2f}%)")

# ====== 5️⃣ INTEGRATION WITH DUALSCAN AI ======
def load_all_models():
    """
    Load semua model untuk DualScan AI
    """
    print("🚀 Loading all models for DualScan AI...")
    
    models_dict = {}
    
    try:
        # Load MRI Model
        mri_model, mri_classes = load_mri_model()
        models_dict['mri_model'] = mri_model
        models_dict['mri_classes'] = mri_classes
        
        print("✅ MRI Model loaded successfully!")
        return models_dict
        
    except Exception as e:
        print(f"❌ Error loading MRI model: {e}")
        return None

# ====== 6️⃣ EXAMPLE USAGE ======
if __name__ == "__main__":
    # Test the model
    test_mri_model()
    
    print("\n" + "="*50)
    print("🎯 EXAMPLE USAGE FOR DUALSCAN AI:")
    print("="*50)
    
    # Contoh penggunaan di app.py Anda:
    models = load_all_models()
    
    if models:
        # Load image (ganti dengan path image MRI Anda)
        try:
            image_path = "sample_mri.jpg"  # Ganti dengan path image MRI Anda
            image = Image.open(image_path).convert("RGB")
            
            # Prediction sederhana
            label, confidence = predict_mri_simple(
                models['mri_model'], 
                models['mri_classes'], 
                image
            )
            
            print(f"🎯 MRI Prediction: {label}")
            print(f"📊 Confidence: {confidence:.2f}%")
            
            # Prediction detail
            detail_label, detail_conf, all_preds = predict_mri_condition(
                models['mri_model'],
                models['mri_classes'], 
                image
            )
            
            print(f"📈 Detailed prediction: {detail_label} ({detail_conf:.2f}%)")
            print("📊 All class probabilities:")
            for class_name, conf in all_preds.items():
                print(f"   {class_name}: {conf}")
                
        except Exception as e:
            print(f"⚠️  Image not found, but model is ready!")
            print(f"💡 Replace 'sample_mri.jpg' with your MRI image path")