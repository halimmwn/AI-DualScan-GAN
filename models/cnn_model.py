import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 1️⃣ LOAD MODEL CNN ======
def load_cnn_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 kelas: normal / abnormal
    model.load_state_dict(torch.load("weights/mura_direct_model.pth", map_location=DEVICE))
    model.eval()
    return model

# ====== 2️⃣ PREPROSES GAMBAR DAN PREDIKSI ======
def predict_bone_condition(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][pred].item() * 100

    label = "Abnormal" if pred.item() == 1 else "Normal"
    return label, confidence
