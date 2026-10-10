# Grad-CAM visualization for the CGLE model (target layer: backbone.additional_conv).
# Usage: python interpretability/gradcam_visualize.py --class_idx <class_index> --num_images <N>

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import argparse

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cgle import LAG
from cgle.data import preprocess_dataset

# --- GradCAM Helper ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        return cam

# --- Visualization ---
def show_gradcam_on_image(img, mask, alpha=0.5, save_path=None):
    img = np.array(img)
    # Resize mask to image size
    from PIL import Image as PILImage
    mask_resized = np.array(PILImage.fromarray((mask * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), resample=PILImage.BILINEAR)) / 255.0
    heatmap = plt.get_cmap('jet')(mask_resized)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)
    overlay = img * (1 - alpha) + heatmap * alpha
    overlay = np.uint8(overlay)
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(overlay)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_idx', type=int, required=True, help='Class index to visualize')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to visualize')
    parser.add_argument('--output_dir', type=str, default='results/gradcam', help='Output directory')
    parser.add_argument('--ckpt', type=str, default='checkpoints/cgle.pth')
    parser.add_argument('--data', type=str, default='data/new_dataset_15')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LAG().to(device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Find the target layer for GradCAM (last conv layer of backbone)
    target_layer = model.backbone.additional_conv
    gradcam = GradCAM(model, target_layer)

    # Load dataset
    c2i = checkpoint.get('class_to_idx')
    class_order = [k for k, v in sorted(c2i.items(), key=lambda x: x[1])] if c2i else None
    data = preprocess_dataset(root_dir=args.data, class_order=class_order)
    val_dataset = data.validation_dataset
    os.makedirs(args.output_dir, exist_ok=True)

    # Find images of the specified class
    class_indices = [i for i, lbl in enumerate(val_dataset.labels) if lbl == args.class_idx]
    if len(class_indices) == 0:
        print(f'No images found for class {args.class_idx}')
        return
    for idx in class_indices[:args.num_images]:
        img_path = val_dataset.images[idx]
        img = Image.open(img_path).convert('RGB')
        input_tensor = val_dataset.transform(img).unsqueeze(0).to(device)
        mask = gradcam(input_tensor, class_idx=args.class_idx)
        show_gradcam_on_image(img, mask, save_path=os.path.join(args.output_dir, f'gradcam_{os.path.basename(img_path)}'))
        print(f'Saved GradCAM for {img_path}')
    gradcam.remove_hooks()

if __name__ == '__main__':
    main()
