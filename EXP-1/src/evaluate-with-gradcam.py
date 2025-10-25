import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/netscratch/aslam/TCAV/text-inflation/EXP1/20251016_001519/best_model.pth"
IMG_DIR = "/netscratch/aslam/TCAV/PetImages/train/CAT-CAT"
SAVE_DIR = "/netscratch/aslam/TCAV/text-inflation/EXP1/20251016_001519/grad-cam/train-cat-cat"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# MODEL LOADING
# ===============================
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # binary classifier
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)["model_state_dict"])
model.to(DEVICE).eval()

# ===============================
# GRADCAM HOOKS
# ===============================
# Choose the last convolutional layer
target_layer = model.layer4[-1].conv2

activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0].detach()

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ===============================
# PREPROCESSING
# ===============================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ===============================
# GRAD-CAM FUNCTION
# ===============================
def generate_gradcam(image_tensor, class_idx=None):
    model.zero_grad()
    output = model(image_tensor)
    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()
    score = output[0, class_idx]
    score.backward()

    # Compute weights
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # Weight the activations
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Mean over channels -> heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()

    heatmap = F.relu(heatmap)

    # Normalize heatmap safely
    heatmap = F.relu(heatmap)
    if torch.isnan(heatmap).any() or torch.isinf(heatmap).any() or torch.max(heatmap) == 0:
        heatmap = torch.zeros_like(heatmap)  # fallback if gradients vanished
    else:
        heatmap /= torch.max(heatmap)


    return heatmap.cpu().numpy()

# ===============================
# OVERLAY FUNCTION
# ===============================
def overlay_heatmap_with_colorbar(heatmap, img_path, alpha=0.5, save_path=None):
    """
    Overlay Grad-CAM heatmap on image and save with colorbar legend.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Blend heatmap with image
    overlay = np.uint8(alpha * heatmap_color + (1 - alpha) * img)
    
    # Plot with colorbar
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(overlay)
    ax.axis('off')
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap='jet'),
        ax=ax,
        fraction=0.046, pad=0.04
    )
    cbar.set_label('Importance (Grad-CAM Intensity)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # Save result
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)

# ===============================
# MAIN LOOP
# ===============================
for filename in os.listdir(IMG_DIR)[:20]:
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(IMG_DIR, filename)
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    # Generate Grad-CAM heatmap
    heatmap = generate_gradcam(img_tensor)

    # Save overlay with colorbar
    save_path = os.path.join(SAVE_DIR, f"cam_{filename}")
    overlay_heatmap_with_colorbar(heatmap, img_path, alpha=0.5, save_path=save_path)

print(f"Grad-CAM images with colorbars saved to: {SAVE_DIR}")