import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import base64
from ultralytics import YOLO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1️⃣ LOAD CNN MODEL (Kulit)
# ============================================================
def load_skin_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Normal / Scabies
    model.load_state_dict(torch.load("weights/skin_model.pth", map_location=DEVICE))
    model.eval()
    return model

# ============================================================
# 2️⃣ PREDIKSI CNN (Scabies atau Normal)
# ============================================================
def predict_skin_condition(model, image):
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

    label = "Scabies Terdeteksi" if pred.item() == 1 else "Kulit Normal"
    return label, confidence

# ============================================================
# 3️⃣ LOAD YOLO MODEL UNTUK BOUNDING BOX SCABIES
# ============================================================
def load_yolo_scabies():
    model = YOLO("weights/scabies_yolo_best.pt")  # ganti nama model YOLO kamu
    return model

# ============================================================
# 4️⃣ YOLO DETECTION – Mengambil bounding box
# ============================================================
def detect_scabies_yolo(yolo_model, image_path, conf=0.3):
    results = yolo_model(image_path, conf=conf)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            det_conf = float(box.conf[0])
            label = yolo_model.names[cls]

            detections.append({
                "box": [x1, y1, x2, y2],
                "label": label,
                "confidence": det_conf
            })

    return detections

# ============================================================
# 5️⃣ DRAW BOUNDING BOX + CONVERT BASE64
# ============================================================
def draw_bounding_boxes(image_path, detections):
    img = cv2.imread(image_path)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    _, buffer = cv2.imencode(".png", img)
    encoded = base64.b64encode(buffer).decode()

    return "data:image/png;base64," + encoded

# ============================================================
# 6️⃣ PIPELINE FINAL: CNN + YOLO (Bounding Box)
# ============================================================
def process_skin_image(image_path, skin_model, yolo_model):
    image = Image.open(image_path).convert("RGB")

    # Step 1: CNN Prediksi
    label, confidence = predict_skin_condition(skin_model, image)

    # Step 2: Kalau NORMAL → tidak pakai YOLO
    if label == "Kulit Normal":
        return {
            "status": label,
            "confidence": confidence,
            "detections": [],
            "bbox_image": None
        }

    # Step 3: Kalau SCABIES → YOLO bounding box
    detections = detect_scabies_yolo(yolo_model, image_path)
    bbox_image = draw_bounding_boxes(image_path, detections)

    return {
        "status": label,
        "confidence": confidence,
        "detections": detections,
        "bbox_image": bbox_image
    }
