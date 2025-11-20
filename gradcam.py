import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================
# GradCAM class
# ===========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer

        # Hook untuk menangkap output feature map
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.cpu().detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].cpu().detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax().item()

        # Backpropagate untuk target class
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # Ambil grad & aktivasi
        gradients = self.gradients
        activations = self.activations

        # Global Average Pooling
        weights = gradients.mean(dim=[1, 2])

        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[0, i]

        # ReLU
        cam = np.maximum(cam.cpu(), 0)

        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.numpy()


# ===========================================
# Generate heatmap overlay
# ===========================================
def overlay_heatmap_on_image(cam, original_image):
    cam = cv2.resize(cam, (original_image.width, original_image.height))
    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    original = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    overlay = heatmap * 0.4 + original * 0.6
    overlay = overlay.astype(np.uint8)

    return overlay
