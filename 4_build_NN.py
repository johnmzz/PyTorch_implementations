# The torch.nn namespace provides all the building blocks you need to build your own neural network. 
# Every module in PyTorch subclasses the nn.Module. 

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# If the current accelerator is available, we will use it. Otherwise, we use the CPU.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# define our neural network by subclassing nn.Module
class NeuralNetwork(nn.Module):

    # initialize the nn layers
    def __init__(self):
        super().__init__()                          # 确保 PyTorch 的内部机制能正确注册你的模型
        self.flatten = nn.Flatten()                 # 把输入的图像从 2D (28×28) 摊平成一维向量（大小 784 = 28×28），方便送入全连接层，例如输入是 shape = (batch_size, 1, 28, 28)，会变成 (batch_size, 784)。

        # 顺序模型
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),                  # 全连接层
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),                     # 输出 10 个值，对应分类的 logits（如 MNIST 的 10 个数字类别）
        )

    # Every nn.Module subclass implements the operations on input data in the forward method
    def forward(self, x):
        x = self.flatten(x)                     # 把图片变成一维向量
        logits = self.linear_relu_stack(x)      # 依次通过上面定义的三层网络
        return logits                           # 输出的是最后一层的结果（未经过 softmax），通常称为 logits
    
# create an instance of nn
model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)        # 随机张量

logits = model(X)                               # 调用模型的 forward() 函数，得到原始分数（可以是正数、负数、任意大小，它们不需要加起来为1）
print(f"logits = {logits}")

pred_probab = nn.Softmax(dim=1)(logits)         # 转换成 归一化的概率
print(f"pred_probab = {pred_probab}")

y_pred = pred_probab.argmax(1)                  # 最终类别（概率最大的位置）
print(f"Predicted class: {y_pred}")

# 注意：
# 训练时，PyTorch 通常不会用 Softmax，而是直接将 logits 送入 CrossEntropyLoss()，它内部会自动处理 softmax。
# Softmax + argmax 这套流程是 测试时用来可视化或输出最终预测类别 的。

# Subclassing nn.Module automatically tracks all fields defined inside your model object, 
# and makes all parameters accessible using your model’s parameters() or named_parameters() methods.

# iterate over each parameter, and print its size and a preview of its values.
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")