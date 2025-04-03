# Data does not always come in its final processed form that is required for training machine learning algorithms. 
# We use transforms to perform some manipulation of the data and make it suitable for training.

# transform to modify the features and target_transform to modify the labels

# The FashionMNIST features are in PIL Image format, and the labels are integers. 
# For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors.

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))     # 创建一个长度为 10 的 0 向量, 在索引 y 的位置上设置为 1
)

