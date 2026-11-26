import torch

# Initial weight
weight = torch.tensor(2.0, requires_grad=True)

# Input and target
x = torch.tensor(3.0)
target = torch.tensor(10.0)

# Create Adam optimizer
optimizer = torch.optim.Adam([weight], lr=0.1)

# Forward propagation
prediction = weight * x

# Calculate loss
loss = (prediction - target) ** 2

# Calculate gradients
loss.backward()

# Update weight
optimizer.step()

print("Initial weight: 2.0")
print("Gradient:", weight.grad.item())
print("Updated weight:", weight.item())
print("Loss before update:", loss.item())