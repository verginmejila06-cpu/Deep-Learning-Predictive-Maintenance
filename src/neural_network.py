
import torch
import torch.nn as nn


# Define neural network
class PredictiveMaintenanceModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # Input layer: 5 to 64
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Hidden layer: 64 to 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer: 32 to 3
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.network(x)


# Create model
model = PredictiveMaintenanceModel()

print("NEURAL NETWORK ARCHITECTURE")
print("---------------------------")

print(model)


# Test with sample input
sample_input = torch.randn(1, 5)

model.eval()

with torch.no_grad():
    output = model(sample_input)

print("\nSample input shape:", sample_input.shape)
print("Output shape:", output.shape)
print("Raw model output:", output)

print("\nNeural network created successfully!")
