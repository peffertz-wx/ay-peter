import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR

# ----------------------------------------------------------------------
# 1. Custom PyTorch Dataset (Optimized for Fast In-Memory Fetching)
# ----------------------------------------------------------------------
class SentinelDataset(Dataset):
    def __init__(self, X, y):
        # Cast and normalize ONCE during initialization to save CPU cycles per batch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if X_tensor.max() > 1.0:
            X_tensor = X_tensor / 10000.0  # Normalize Sentinel-2 reflectance values
            
        self.X = X_tensor
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------------------------------------------------
# 2. ResNet-34 Model Definition
# ----------------------------------------------------------------------
def get_resnet34_13ch(num_classes):
    model = models.resnet34(weights=None)
    
    # 13 channels input; 3x3 kernel, stride 1 keeps 64x64 resolution intact
    model.conv1 = nn.Conv2d(
        in_channels=13,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ----------------------------------------------------------------------
# 3. Main Execution Block
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Device Setup for Apple Silicon M2 GPU
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
    train_npz = np.load('train.npz')
    X_raw = train_npz['X']
    y_raw = train_npz['y']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)

    print(f"Detected {num_classes} unique classes: {label_encoder.classes_}")

    # Transpose channels to PyTorch format: (N, 64, 64, 13) -> (N, 13, 64, 64)
    if X_raw.ndim == 4 and X_raw.shape[-1] == 13:
        X_raw = np.transpose(X_raw, (0, 3, 1, 2))

    # Step 1: 80/20 Stratified Split
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X_raw,
        y_encoded,
        test_size=0.25,
        random_state=67,
        stratify=y_encoded
    )

    # Step 2: Augment ONLY X_train (Horizontal Reflection along width)
    X_train_reflected = np.flip(X_train_orig, axis=3)
    X_train = np.concatenate([X_train_orig, X_train_reflected], axis=0)
    y_train = np.concatenate([y_train_orig, y_train_orig], axis=0)

    print(f"Augmented Train Set: {X_train.shape[0]} samples (Doubled)")
    print(f"Clean Test Set:     {X_test.shape[0]} samples")

    # Fast DataLoaders (batch_size=64, num_workers=4 for parallel CPU fetching)
    train_dataset = SentinelDataset(X_train, y_train)
    test_dataset  = SentinelDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,       # Increased from 32 -> 64 for higher M2 GPU throughput
        shuffle=True,
        num_workers=4,       # Parallel CPU data loading threads
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    # Model, Loss, Optimizer, and Cosine Annealing Scheduler Setup
    model = get_resnet34_13ch(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    # Decays learning rate smoothly from 1e-3 to 1e-5 across 10 epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_test_acc = 0.0
    checkpoint_path = 'best_resnet34.pth'

    print("\nStarting Fast Training Loop...\n" + "-" * 65)

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        epoch_train_loss = running_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100
        
        # --- Testing Phase ---
        model.eval()
        test_correct = 0
        test_total = 0
        test_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
                
        epoch_test_loss = test_running_loss / test_total
        epoch_test_acc = (test_correct / test_total) * 100
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | LR: {current_lr:.6f} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%", end="")

        # --- Checkpoint Saving ---
        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(" --> Saved Best Weights!")
        else:
            print()

        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()

    print("-" * 65)
    print(f"Training Complete! Peak Test Accuracy: {best_test_acc:.2f}%")

    # Reload best model weights from disk
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Successfully reloaded best model weights from '{checkpoint_path}'.")
