
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from torch.utils.data import Dataset, DataLoader


# Custom Dataset
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


# Neural Network
class PredictiveMaintenanceModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.network(x)


# Load test data
data = np.load("output/processed_data.npz")

test_dataset = SensorDataset(
    data["X_test"], data["y_test"]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load trained model
model = PredictiveMaintenanceModel().to(device)

model.load_state_dict(
    torch.load(
        "output/predictive_maintenance_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# Evaluate model
all_predictions = []
all_labels = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)

        outputs = model(features)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(labels.numpy())


# Calculate metrics
accuracy = accuracy_score(
    all_labels, all_predictions
)

print("MODEL EVALUATION RESULTS")
print("------------------------")

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
        target_names=["Normal", "Warning", "Failure"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2]
    )
)
