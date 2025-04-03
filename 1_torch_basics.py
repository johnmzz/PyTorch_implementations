import torch
import numpy as np

# directly from data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

# from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# print(f"Tensor: \n {x_np} \n")

# from another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
# print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")

# tensor dimensions
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)


# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

# Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3,4)

# move tensors to the accelerator
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor = tensor.to(device)

# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

# numpy-like indexing and slicing
tensor = torch.ones(4, 4)
# print(f"First row: {tensor[0]}")
# print(f"First column: {tensor[:, 0]}")
# print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
# print(tensor)

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

# Arithmetic operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single-element tensors, eg. aggregating all values of a tensor into one value
agg = tensor.sum()
agg_item = agg.item()   # convert into a python numerical value using item()

# print(agg_item, type(agg_item))

# In-place operations: operations that store the result into the operand, denoted by _ suffix
# print(f"{tensor} \n")
tensor.add_(5)
# print(tensor)

# Tensors on the CPU and NumPy arrays can share their underlying memory locations
# Tensor to numpy
t = torch.ones(5)
# print(f"t: {t}")

# A change in the tensor reflects the numpy array
n = t.numpy()
# print(f"n: {n}")

t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")

# Numpy to tensors
n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
# print(f"t: {t}")
# print(f"n: {n}")