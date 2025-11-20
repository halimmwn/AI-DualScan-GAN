from ultralytics import YOLO

# Load model YOLOv8s (kecil dan cepat)
model = YOLO("yolov8s.pt")

# Train
model.train(
    data="D:/Kuliah/SEMESTER 5/Project/AI BONE GAN/data/Scabies-yolo/data.yaml",    # ganti ke lokasi dataset roboflow
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,    # gunakan GPU jika ada
)
