import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION (Edit paths if needed)
# ==========================================
DATA_PATH = ''
CHECKPOINT_PATH = os.path.join(DATA_PATH, 'check_.pth')
BANDS = "all"  # "all" (13 channels) or "rgb" (3 channels)

BAND_SETS = {
    "rgb": [3, 2, 1],
    "all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
}
band_idx = BAND_SETS[BANDS]
n_channels = len(band_idx)


# ==========================================
# DATASET DEFINITION
# ==========================================
class SentinelDataset(Dataset):
    def __init__(self, X, y, transform=None):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if X_tensor.max() > 1.0:
            X_tensor = X_tensor / 10000.0
        self.X = X_tensor
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]
        if self.transform:
            x_sample = self.transform(x_sample)
        return x_sample, y_sample


# ==========================================
# MODEL DEFINITION
# ==========================================
def get_resnet34(num_classes, in_channels):
    # No pre-trained ImageNet weights needed since we are loading saved checkpoint weights
    model = models.resnet34(weights=None)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    # Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS engine)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA engine)")
    else:
        device = torch.device("cpu")
        print("Using CPU engine")

    # Load Data & Encode Labels
    train_npz = np.load(os.path.join(DATA_PATH, 'train.npz'))
    X_raw = train_npz['X']
    y_raw = train_npz['y']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    original_class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
                            'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

    if X_raw.ndim == 4 and X_raw.shape[-1] == 13:
        X_raw = np.transpose(X_raw, (0, 3, 1, 2))

    X_raw = X_raw[:, band_idx, :, :]

    # Recreate the exact same 75/25 split configuration
    _, X_test, _, y_test = train_test_split(
        X_raw,
        y_encoded,
        test_size=0.25,
        random_state=25,
        stratify=y_encoded
    )

    test_dataset = SentinelDataset(X_test, y_test, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model and load checkpoint weights
    model = get_resnet34(num_classes=num_classes, in_channels=n_channels)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Successfully loaded model weights from '{CHECKPOINT_PATH}'.")

    # Find confusions between AnnualCrop and PermanentCrop
    annual_idx = list(original_class_names).index('AnnualCrop')
    perm_idx = list(original_class_names).index('PermanentCrop')

    all_preds = []
    all_labels = []

    print("Running inference on test set to isolate confusions...")
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    confusion_mask = ((all_labels == annual_idx) & (all_preds == perm_idx)) | \
                     ((all_labels == perm_idx) & (all_preds == annual_idx))
    confused_indices = np.where(confusion_mask)[0]
    print(f"Found {len(confused_indices)} total confusions between Annual and Permanent Crops!")

    # ==========================================
    # GRAD-CAM HOOKS & VISUALIZATION LOOP
    # ==========================================
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        global activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        global gradients
        gradients = grad_output[0].detach()

    target_layer = model.layer4[-1]
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    print("\nGenerating Grad-CAM visualizations for confused samples...")
    for idx in confused_indices[:10]:  # Change slice amount to view more/fewer
        img_tensor, true_label_enc = test_dataset[idx]
        pred_label_enc = all_preds[idx]

        input_tensor = img_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        model.zero_grad()
        output = model(input_tensor)

        class_score = output[0, pred_label_enc]
        class_score.backward()

        # Compute Grad-CAM weights and heatmap
        pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=device)

        for i, w in enumerate(pooled_grads):
            cam += w * activations[0, i, :, :]

        cam = F.relu(cam)
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(64, 64), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Brighten RGB extraction using scaling + gamma correction
        rgb_tensor = img_tensor[[3, 2, 1], :, :]
        rgb_img = rgb_tensor.permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img * 3.0, 0, 1)
        rgb_img = np.power(rgb_img, 0.45)

        # Plot Side-by-Side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Test Index {idx}\nTrue: {original_class_names[true_label_enc]}", fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(rgb_img)
        axes[1].imshow(cam, cmap='jet', alpha=0.5)
        axes[1].set_title(f"Grad-CAM Heatmap\nPred: {original_class_names[pred_label_enc]}", fontsize=10)
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    # Clean up hooks
    handle_fwd.remove()
    handle_bwd.remove()
    print("Visualization complete.")