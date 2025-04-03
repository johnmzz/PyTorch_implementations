# Back propagation: parameters (model weights) are adjusted according to the gradient of the loss function with respect to the given parameter.
# To compute those gradients, PyTorch has a built-in differentiation engine called torch.autograd. 
# It supports automatic computation of gradient for any computational graph.

import torch

# Consider the simplest one-layer neural network, with input x, parameters w and b, and some loss function.
x = torch.ones(5)   # input tensor
y = torch.zeros(3)  # expected output

w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# A function that we apply to tensors to construct computational graph is in fact an object of class Function. 
# This object knows how to compute the function in the forward direction, and also how to compute its derivative during the backward propagation step.
# A reference to the backward propagation function is stored in grad_fn property of a tensor. 

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# To optimize weights of parameters in the neural network, we need to compute the derivatives of our loss function with respect to parameters
# To compute those derivatives, we call loss.backward(), and then retrieve the values from w.grad and b.grad:
loss.backward()
print(w.grad)
print(b.grad)

# We can only obtain the grad properties for the leaf nodes of the computational graph, which have requires_grad property set to True. 
# For all other nodes in our graph, gradients will not be available.

# Can disable gradient tracking if we have trained the model and just want to apply it to some input data
# i.e. we only want to do forward computations through the network
# We can stop tracking computations by surrounding our computation code with torch.no_grad() block:
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Another way to achieve the same result is to use the detach() method on the tensor:
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# There are reasons you might want to disable gradient tracking:
#   - to mark some parameters in your neural network as frozen parameters
#   - to speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.