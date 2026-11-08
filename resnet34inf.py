import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Running inference on device: {device}")

#extract class names
train_npz = np.load('train.npz')
y_raw = train_npz['y']

label_encoder = LabelEncoder()
label_encoder.fit(y_raw)
class_names = label_encoder.classes_
num_classes = len(class_names)

print(f"Detected {num_classes} classes with exact target spellings: {list(class_names)}")


#load saved model checkpoint

def get_resnet34_13ch(num_classes):
    model = models.resnet34(weights=None)
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

model = get_resnet34_13ch(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('best_resnet34.pth', map_location=device))
model.eval()

#Load Test Data & Extract 'id' Array

test_npz = np.load('test.npz')
X_test = test_npz['X']
test_ids = test_npz['id']  # Pulls IDs in original test order

#transpose spatial dimensions: (N, 64, 64, 13) -> (N, 13, 64, 64)
if X_test.ndim == 4 and X_test.shape[-1] == 13:
    X_test = np.transpose(X_test, (0, 3, 1, 2))

class InferenceDataset(Dataset):
    def __init__(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if X_tensor.max() > 1.0:
            X_tensor = X_tensor / 10000.0  #normalize reflectance like in train file
        self.X = X_tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

test_loader = DataLoader(
    InferenceDataset(X_test),
    batch_size=64,
    shuffle=False  #keeps test IDs in order. Keep this at false
)

predictions = []

with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())

predicted_class_names = label_encoder.inverse_transform(predictions)


member_name = "PeterBraza1" # <-change this to your own name to name the CSV

submission_df = pd.DataFrame({
    'id': test_ids,            # Column 1
    'label': predicted_class_names  # Column 2: predicted class label
})

csv_filename = f"{member_name}.csv"
submission_df.to_csv(csv_filename, index=False)

print(f"\nSuccessfully generated submission file: '{csv_filename}'")
print(f"Total Rows: {len(submission_df)}")
print("\nFirst 5 Rows Preview:")
print(submission_df.head())
