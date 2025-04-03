# An example of how to load the Fashion-MNIST dataset from TorchVision
# Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples. 
# Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader

# Load the FashionMNIST Dataset with the following parameters:
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# visualize some samples in our training data
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):     #  index Datasets manually like a list: training_data[index]
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.savefig("fashion_grid.png")

# Creating a Custom Dataset for your files
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
# eg. 
# the FashionMNIST images are stored in a directory img_dir
# their labels are stored separately in a CSV file annotations_file
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform                  # 图像的转换（如 resize, to tensor, normalize）
        self.target_transform = target_transform    # 标签的转换（如转为 one-hot）

    def __len__(self):
        return len(self.img_labels)

    # loads and returns a sample from the dataset at the given index 
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])     # 获取图像路径，例如 'images/shirt1.jpg'
        image = read_image(img_path)                                            # 加载图像并转成 PyTorch tensor，形状为 [C, H, W]（通常是 [3, H, W]
        label = self.img_labels.iloc[idx, 1]                                    # 获取图像对应的标签

        # 如果设置了 transform，对图像或标签进行变换
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Preparing your data for training with DataLoaders
# While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting
# DataLoader is an iterable that abstracts this complexity for us in an easy API.
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the DataLoader
train_features, train_labels = next(iter(train_dataloader))     # 取出一个 batch 的数据
print(f"Feature batch shape: {train_features.size()}")          # train_features: 一批图像张量，tensor shape = [batch_size, 1, 28, 28]
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()   # 取出第一个图像（[1, 28, 28]），并使用 .squeeze() 去掉 channel 维度，变成 [28, 28]，方便画图。
label = train_labels[0]             # 获取这个图像对应的标签（例如 9 → Ankle Boot）
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")