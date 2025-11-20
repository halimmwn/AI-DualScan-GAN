import io
import os
import base64
import numpy as np
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import cv2

# ============================================================
# KONFIGURASI FLASK UNTUK VERCEL
# ============================================================
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Cek environment Vercel
IS_VERCEL = os.environ.get('VERCEL') is not None

# Device configuration untuk Vercel (CPU only)
if IS_VERCEL:
    DEVICE = torch.device("cpu")  # Vercel hanya support CPU
    print("Running on Vercel Environment - Using CPU")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Locally - Using device: {DEVICE}")

# ============================================================
# BUAT FOLDER PENYIMPANAN DI VERCEL
# ============================================================
if IS_VERCEL:
    # Di Vercel, gunakan /tmp untuk penyimpanan sementara
    UPLOAD_DIR = "/tmp/uploads"
else:
    UPLOAD_DIR = "uploads"

ORIGINAL_DIR = os.path.join(UPLOAD_DIR, "original")
os.makedirs(ORIGINAL_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL ESRGAN UNTUK SUPER RESOLUTION
# ============================================================
def load_esrgan_model():
    """
    Load model Real-ESRGAN untuk peningkatan kualitas gambar
    """
    try:
        model_path = "weights/realESRGANX2plus.pth"
        if not os.path.exists(model_path):
            print(f"ESRGAN model file not found: {model_path}")
            return None
            
        # Load model ESRGAN
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        
        # Load weights
        loadnet = torch.load(model_path, map_location=DEVICE)
        if 'params' in loadnet:
            model.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            model.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            model.load_state_dict(loadnet, strict=True)
            
        model.to(DEVICE)
        model.eval()
        
        print("ESRGAN Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading ESRGAN model: {e}")
        return None

def enhance_image_esrgan(model, image_pil):
    """
    Meningkatkan kualitas gambar menggunakan ESRGAN
    """
    try:
        if model is None:
            return image_pil
            
        # Convert PIL to numpy array
        img_np = np.array(image_pil)
        
        # Preprocess untuk ESRGAN
        img_np = img_np.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Inference dengan ESRGAN
            output = model(img_tensor)
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        
        # Convert kembali ke PIL
        enhanced_image = Image.fromarray(output)
        print("Image enhanced successfully with ESRGAN")
        return enhanced_image
        
    except Exception as e:
        print(f"Error in ESRGAN enhancement: {e}")
        return image_pil

def enhance_image_opencv(image_pil):
    """
    Fallback image enhancement menggunakan OpenCV jika ESRGAN tidak tersedia
    """
    try:
        # Convert PIL to OpenCV
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Enhance contrast menggunakan CLAHE
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced_cv = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Convert back to PIL
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB))
        print("Image enhanced with OpenCV CLAHE")
        return enhanced_pil
        
    except Exception as e:
        print(f"Error in OpenCV enhancement: {e}")
        return image_pil

# ============================================================
# LOAD MODEL DENGAN ERROR HANDLING UNTUK VERCEL
# ============================================================
def load_bone_model():
    """
    Load model tulang unified untuk klasifikasi 10 jenis fracture (X-Ray & MRI)
    """
    try:
        model_path = "weights/bone_best.pth"
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return create_fallback_model(), ['Healthy', 'Fracture']
            
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Get class names dari checkpoint
        class_names = checkpoint.get('classes', ['Healthy', 'Fracture'])  # Fallback classes
        num_classes = len(class_names)
        
        # Create model architecture
        model = models.resnet18(weights=None)
        
        # Replace classifier
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
        
        print("Bone Model (Unified) loaded successfully!")
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {class_names}")
        
        return model, class_names
    except Exception as e:
        print(f"Error loading Bone model: {e}")
        return create_fallback_model(), ['Healthy', 'Fracture']

def load_skin_model():
    try:
        model_path = "weights/skin_model.pth"
        if not os.path.exists(model_path):
            print(f"Skin model file not found: {model_path}")
            return create_fallback_model()
            
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Skin Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading Skin model: {e}")
        return create_fallback_model()

def load_yolo_model():
    try:
        model_path = "runs/detect/train/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"YOLO model file not found: {model_path}")
            return None
            
        model = YOLO(model_path)
        print("YOLO Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def create_fallback_model():
    """Membuat model fallback sederhana"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(DEVICE)
    model.eval()
    print("Using fallback model")
    return model

# ============================================================
# GRADCAM IMPLEMENTATION YANG DIPERBAIKI
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self.register_hooks()
    
    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            return None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            return None

        # Register hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        self.hooks = [forward_handle, backward_handle]
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def generate(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Create heatmap
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam

def generate_gradcam(model, image_pil, target_size=(224, 224)):
    """
    Generate GradCAM heatmap untuk model ResNet
    """
    try:
        # Nonaktifkan GradCAM di Vercel untuk menghemat memory jika diperlukan
        if IS_VERCEL:
            print("GradCAM disabled on Vercel to save memory")
            return None
            
        model.eval()
        
        # Transformasi untuk preprocessing
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
        
        # Preprocess image
        img_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad_(True)
        
        # Target layer untuk ResNet18 (layer4 terakhir)
        target_layer = model.layer4[-1]
        
        # Initialize GradCAM
        gradcam = GradCAM(model, target_layer)
        
        try:
            # Generate heatmap
            cam = gradcam.generate(img_tensor)
            
            # Resize heatmap ke ukuran gambar asli
            original_size = image_pil.size
            cam_resized = cv2.resize(cam, original_size)
            
            # Convert to heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            
            # Convert original image to numpy
            img_np = np.array(image_pil)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Resize heatmap to match original image
            heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            
            # Blend heatmap dengan gambar original
            alpha = 0.5
            overlay = cv2.addWeighted(heatmap, alpha, img_np, 1 - alpha, 0)
            
            # Convert back to base64
            _, buffer = cv2.imencode('.png', overlay)
            gradcam_b64 = base64.b64encode(buffer).decode('utf-8')
            
            result = f"data:image/png;base64,{gradcam_b64}"
            
        finally:
            # Pastikan hooks selalu diremove
            gradcam.remove_hooks()
        
        print("GradCAM generated successfully")
        return result
        
    except Exception as e:
        print(f"GradCAM Error: {e}")
        return None

# ============================================================
# PREDIKSI TULANG UNIFIED DENGAN GRADCAM
# ============================================================
def predict_bone_condition(model, class_names, image):
    """
    Prediksi kondisi tulang untuk berbagai jenis fracture dengan GradCAM
    """
    try:
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
    except Exception as e:
        print(f"Error in bone prediction: {e}")
        return "Error", 0.0, {}

# ============================================================
# YOLO DETECTION UNTUK SCABIES
# ============================================================
def detect_scabies_yolo(model, pil_image):
    try:
        if model is None:
            return [], None
            
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        results = model.predict(img_cv, conf=0.20, verbose=False)

        detections = []
        annotated = img_cv.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                detections.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "confidence": conf
                })

                # Draw bounding box
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated, f"Scabies {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode annotated image
        _, buffer = cv2.imencode('.png', annotated)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        annotated_b64 = f"data:image/png;base64,{img_b64}"

        return detections, annotated_b64
    
    except Exception as e:
        print(f"YOLO Detection Error: {e}")
        return [], None

# ============================================================
# CNN PREDIKSI KULIT
# ============================================================
def predict_skin(model, image):
    try:
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

        label = "Scabies Terdeteksi" if pred.item() == 1 else "Kulit Normal"
        confidence_value = confidence.item() * 100
        
        return label, confidence_value
    except Exception as e:
        print(f"Error in skin prediction: {e}")
        return "Error", 0.0

# ============================================================
# LOAD MODEL SEKALI SAJA
# ============================================================
print("Loading models...")
bone_model, bone_classes = load_bone_model()
skin_model = load_skin_model()
yolo_model = load_yolo_model()
esrgan_model = load_esrgan_model()  # Load ESRGAN model
print("All models initialized!")

# ============================================================
# HALAMAN UTAMA
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

# ============================================================
# ENDPOINT UTAMA UPLOAD - DENGAN IMAGE ENHANCEMENT DAN GRADCAM
# ============================================================
@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        print("Received request to process image")
        
        image_type = request.form.get("image_type")
        print(f"Image type: {image_type}")
        
        if not image_type:
            return jsonify({'error': 'Tipe gambar tidak ditentukan'}), 400

        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diupload'}), 400

        uploaded = request.files['file']
        if uploaded.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

        print(f"Processing file: {uploaded.filename}")
        
        # Buka dan konversi gambar
        original_image = Image.open(uploaded.stream).convert("RGB")
        print(f"Original image size: {original_image.size}")

        # SIMPAN GAMBAR ASLI
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_type}_{timestamp}.png"
        save_path = os.path.join(ORIGINAL_DIR, filename)
        original_image.save(save_path)

        # ENHANCE GAMBAR MENGGUNAKAN ESRGAN ATAU OPENCV
        print("Enhancing image quality...")
        if esrgan_model is not None:
            enhanced_image = enhance_image_esrgan(esrgan_model, original_image)
        else:
            enhanced_image = enhance_image_opencv(original_image)
        
        print(f"Enhanced image size: {enhanced_image.size}")

        # Convert enhanced image to base64 untuk response
        buf_enhanced = io.BytesIO()
        enhanced_image.save(buf_enhanced, format="PNG")
        enhanced_b64 = base64.b64encode(buf_enhanced.getvalue()).decode("utf-8")
        enhanced_b64 = f"data:image/png;base64,{enhanced_b64}"

        # Convert original image to base64 untuk perbandingan
        buf_original = io.BytesIO()
        original_image.save(buf_original, format="PNG")
        original_b64 = base64.b64encode(buf_original.getvalue()).decode("utf-8")
        original_b64 = f"data:image/png;base64,{original_b64}"

        # ====================== BONE XRAY & MRI UNIFIED ============================
        if image_type in ["bone_xray", "bone_mri"]:
            if bone_model is None:
                return jsonify({"error": "Bone model not loaded"}), 500
                
            # Gunakan enhanced image untuk analisis
            label, conf, all_predictions = predict_bone_condition(bone_model, bone_classes, enhanced_image)

            # Generate GradCAM untuk bone analysis
            gradcam_img = None
            try:
                print("Generating GradCAM for bone analysis...")
                gradcam_img = generate_gradcam(bone_model, enhanced_image)
                if gradcam_img:
                    print("GradCAM generated successfully")
                else:
                    print("GradCAM generation failed or disabled")
            except Exception as e:
                print(f"GradCAM generation error: {e}")
                gradcam_img = None

            response_data = {
                "original_image_base64": original_b64,
                "enhanced_image_base64": enhanced_b64,
                "label": label,
                "confidence": conf,
                "all_predictions": all_predictions,
                "image_type": image_type,
                "class_names": bone_classes,
                "enhancement_used": "realESRGAN" if esrgan_model is not None else "OpenCV"
            }
            
            # Tambahkan GradCAM jika berhasil di-generate
            if gradcam_img:
                response_data["gradcam"] = gradcam_img

            print(f"Bone analysis completed: {label} ({conf}%)")
            return jsonify(response_data)

        # ====================== SKIN UPLOAD =====================
        elif image_type == "skin_upload":
            # Gunakan enhanced image untuk analisis
            label, conf = predict_skin(skin_model, enhanced_image)
            detections, bbox_image_b64 = detect_scabies_yolo(yolo_model, enhanced_image)

            response_data = {
                "original_image_base64": original_b64,
                "enhanced_image_base64": enhanced_b64,
                "label": label,
                "confidence": conf,
                "image_type": image_type,
                "enhancement_used": "realESRGAN" if esrgan_model is not None else "OpenCV"
            }
            
            if detections and bbox_image_b64:
                response_data["yolo_boxes"] = detections
                response_data["yolo_annotated_image"] = bbox_image_b64

            print(f"Skin analysis completed: {label} ({conf}%)")
            return jsonify(response_data)

        return jsonify({"error": "Mode tidak dikenali"}), 400

    except Exception as e:
        print(f"Process image error: {e}")
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

# ============================================================
# ENDPOINT DETEKSI YOLO WEBCAM
# ============================================================
@app.route("/detect-webcam", methods=["POST"])
def detect_webcam():
    try:
        if "frame" not in request.files:
            return jsonify({"error": "Tidak ada frame yang diterima"}), 400

        file = request.files["frame"]
        img = Image.open(file.stream).convert("RGB")

        # Enhance gambar webcam sebelum analisis
        if esrgan_model is not None:
            enhanced_img = enhance_image_esrgan(esrgan_model, img)
        else:
            enhanced_img = enhance_image_opencv(img)

        # Deteksi dengan YOLO pada gambar yang sudah di-enhanced
        detections, annotated_img = detect_scabies_yolo(yolo_model, enhanced_img)

        # Buat preview gambar asli
        img_np = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)
        _, buff = cv2.imencode(".png", img_np)
        frame_b64 = base64.b64encode(buff).decode("utf-8")
        frame_b64 = f"data:image/png;base64,{frame_b64}"

        # Jika ADA deteksi
        if len(detections) > 0 and annotated_img:
            return jsonify({
                "detected": True,
                "boxes": detections,
                "annotated": annotated_img,
                "frame_preview": frame_b64,
                "enhancement_used": "realESRGAN" if esrgan_model is not None else "OpenCV"
            })
        else:
            # Jika TIDAK ADA deteksi, kirim gambar asli
            return jsonify({
                "detected": False,
                "boxes": [],
                "annotated": frame_b64,
                "frame_preview": frame_b64,
                "enhancement_used": "realESRGAN" if esrgan_model is not None else "OpenCV"
            })

    except Exception as e:
        print(f"Webcam detection error: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================
# HEALTH CHECK
# ============================================================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
        "vercel": IS_VERCEL,
        "models_loaded": {
            "bone": bone_model is not None,
            "skin": skin_model is not None, 
            "yolo": yolo_model is not None,
            "esrgan": esrgan_model is not None
        },
        "bone_classes": bone_classes if bone_model else []
    })

# ============================================================
# RUN SERVER UNTUK VERCEL
# ============================================================
if __name__ == '__main__':
    if IS_VERCEL:
        print("Running on Vercel...")
    else:
        print("Starting MedScan AI Server...")
        print(f"Device: {DEVICE}")
        print("Server running on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)