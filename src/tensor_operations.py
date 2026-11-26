
import torch

# Create two tensors
a = torch.tensor([10.0, 20.0, 30.0])
b = torch.tensor([1.0, 2.0, 3.0])

# Addition
print("Addition:", a + b)

# Subtraction
print("Subtraction:", a - b)

# Multiplication
print("Multiplication:", a * b)

# Division
print("Division:", a / b)

# Mean
print("Mean:", torch.mean(a))

# Maximum
print("Maximum:", torch.max(a))
