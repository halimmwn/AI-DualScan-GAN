from realesrgan import RealESRGANer, RRDBNet
from realesrgan.utils import load_file_from_url
import torch
import os

def load_superres_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "weights/RealESRGAN_x4plus.pth"
    if not os.path.exists(model_path):
        os.makedirs("weights", exist_ok=True)
        model_path = load_file_from_url(
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/RealESRGAN_x2plus.pth",
            model_dir="weights"
        )
    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(scale=2, model_path=model_path, model=model, device=device)
    return upsampler
