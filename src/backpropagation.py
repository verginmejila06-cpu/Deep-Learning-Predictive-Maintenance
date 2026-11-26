
import torch

# Create a weight that can learn
weight = torch.tensor(2.0, requires_grad=True)

# Input and expected output
x = torch.tensor(3.0)
target = torch.tensor(10.0)

# Forward propagation
prediction = weight * x

# Calculate loss
loss = (prediction - target) ** 2

# Backpropagation
loss.backward()

# Display results
print("Prediction:", prediction.item())
print("Loss:", loss.item())
print("Gradient:", weight.grad.item())
