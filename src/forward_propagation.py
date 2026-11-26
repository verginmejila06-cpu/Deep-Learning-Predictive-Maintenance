
import torch
import torch.nn as nn

# Input: 3 sensor values
input_data = torch.tensor([[25.0, 0.5, 100.0]])

# Create a simple neural network
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 2)
)

# Forward propagation
output = model(input_data)

print("Input data:")
print(input_data)

print("\nPrediction output:")
print(output)

print("\nOutput shape:")
print(output.shape)
