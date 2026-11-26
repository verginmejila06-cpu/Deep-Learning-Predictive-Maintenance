
import torch

# 1. Create a scalar tensor
scalar = torch.tensor(10)

# 2. Create a 1D tensor
vector = torch.tensor([10, 20, 30, 40])

# 3. Create a 2D tensor
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

# Display tensors
print("Scalar:")
print(scalar)

print("\nVector:")
print(vector)

print("\nMatrix:")
print(matrix)

# Check tensor properties
print("\nTensor Shape:", matrix.shape)
print("Tensor Dimension:", matrix.ndim)
print("Tensor Data Type:", matrix.dtype)
