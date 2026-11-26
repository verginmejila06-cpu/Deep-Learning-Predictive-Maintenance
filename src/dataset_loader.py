
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


# Custom Dataset class
class SensorDataset(Dataset):

    def __init__(self, features, labels):
        self.features = torch.tensor(
            features, dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels, dtype=torch.long
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# Load processed data
data = np.load("output/processed_data.npz")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]


# Create Dataset objects
train_dataset = SensorDataset(X_train, y_train)
test_dataset = SensorDataset(X_test, y_test)


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Selected device:", device)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=(device.type == "cuda")
)


# Check one training batch
features, labels = next(iter(train_loader))

print("\nDATASET INFORMATION")
print("-------------------")

print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))

print("\nFIRST TRAINING BATCH")
print("--------------------")

print("Feature batch shape:", features.shape)
print("Label batch shape:", labels.shape)

print("\nFirst 5 labels:")
print(labels[:5])

print("\nTotal training batches:", len(train_loader))
print("Total testing batches:", len(test_loader))

print("\nCustom Dataset and DataLoader ready!")
