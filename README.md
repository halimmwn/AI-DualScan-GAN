# DualScan-AI-GAN

DualScan-AI-GAN adalah proyek deep learning yang menggabungkan teknik Super-Resolution, GAN-based enhancement, dan Object Detection untuk membantu analisis citra medis seperti MRI serta deteksi penyakit kulit dan scabies.  
Proyek ini mencakup beberapa komponen utama:

- Model Super Resolution (Real-ESRGAN)
- Model deteksi berbasis YOLO
- Sistem Grad-CAM untuk interpretasi model
- Web App berbasis Flask untuk inferensi real-time
- Training pipeline untuk MRI & Xray, Scabies YOLO, dan Skin Scabies

--------------------------------------------------------------------------------------------------------------------

## 📂 **Dataset & Weights**

Untuk menjaga ukuran repository tetap ringan, file dataset dan model weights disimpan di Google Drive.

### 🔗 **Download Weights**
Link weights berbagai model (Real-ESRGAN,XRAY & MRI,Scabies):
➡️ https://drive.google.com/drive/folders/1jz39mzhVijPl2prwttlarZJiBVAok7e3?usp=drive_link

### 🔗 **Download Folder Runs (Training Output)**
Link Runs Hasil Training dari Yolo
➡️ https://drive.google.com/drive/folders/1H54A8PZi8iS7udslCnybZT_iJoS2PIed?usp=drive_link

---------------------------------------------------------------------------------------------------------------------

##  Cara Menjalankan

```bash
pip install -r requirements.txt
python app.py

