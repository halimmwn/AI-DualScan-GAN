import os
import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Fungsi load model Real-ESRGAN
def load_superres_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "..", "weights", "RealESRGAN_x2plus.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model weight tidak ditemukan di: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)
    
    upsampler = RealESRGANer(
        scale=2,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )

    return upsampler


# Fungsi enhance gambar X-ray
def enhance_image(model_sr, image):
    # Cek apakah image sudah berupa objek PIL.Image
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    # Konversi ke array NumPy
    img_np = np.array(image)
    
    try:
        output, _ = model_sr.enhance(img_np, outscale=2)
        result = Image.fromarray(output)
        return result
    except Exception as e:
        print(f"Error Real-ESRGAN: {e}")
        return image  # kalau gagal, kembalikan gambar asli
