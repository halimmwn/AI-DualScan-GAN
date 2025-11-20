import cv2
from ultralytics import YOLO

# Load model hasil training
model = YOLO("runs/detect/train/weights/best.pt")  # ganti jika lokasi berbeda

# Buka webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam tidak ditemukan!")
    exit()

print("🎥 Webcam YOLO Scabies Detection Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Tidak bisa membaca frame.")
        break

    # Deteksi YOLO
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Confidence score
            conf = float(box.conf[0])
            conf_text = f"{conf*100:.1f}%"

            # Ambil label
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Warna bounding (merah)
            color = (0, 0, 255)

            # Gambar kotak
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Tampilkan label + confidence
            cv2.putText(frame, f"{label} {conf_text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("YOLO Scabies Detection - Webcam", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
