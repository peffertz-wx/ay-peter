import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#setup for apple m2 GPU or CUDA for windows
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS engine)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA engine)")
else:
    device = torch.device("cpu")
    print("Using CPU engine")


#load data

npz_file = np.load('train.npz')

X_raw = npz_file['X']
y_raw = npz_file['y']

#map class labels to contiguous integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

print(f"Detected {num_classes} unique classes: {label_encoder.classes_}")

if X_raw.ndim == 4 and X_raw.shape[-1] == 13:
    X_raw = np.transpose(X_raw, (0, 3, 1, 2))

print(f"Dataset shape: Images = {X_raw.shape}, Labels = {y_encoded.shape}")

#75/25 test/train
X_train, X_test, y_train, y_test = train_test_split(
    X_raw,
    y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")


class SentinelDataset(Dataset):
    def __init__(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # Normalize pixel values if they are raw uint16 reflectance values
        if X_tensor.max() > 1.0:
            X_tensor = X_tensor / 10000.0
            
        self.X = X_tensor
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SentinelDataset(X_train, y_train)
test_dataset  = SentinelDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)


#ResNet-18 model def

def get_resnet18_13ch(num_classes):
    model = models.resnet18(weights=None)
    
    #13 channels input; kernel size 3, stride 1
    model.conv1 = nn.Conv2d(
        in_channels=13,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    model.maxpool = nn.Identity()
    
    #output layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = get_resnet18_13ch(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#7 epochs of training and evaluation on test data
num_epochs = 12

print("\nStarting Training...\n")

for epoch in range(num_epochs):
    #training
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
    
    print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | "
          f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
          f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")

print("-" * 55)
print(f"Final Test Accuracy after {num_epochs} Epochs: {epoch_test_acc:.2f}%")
