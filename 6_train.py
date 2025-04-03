# Train, validate and test our model by optimizing its parameters on our data.
# Training a model is an iterative process, in each iteration the model:
# - makes a guess about the output
# - calculates the error in its guess (loss)
# - collects the derivatives of the error with respect to its parameters
# - optimizes these parameters using gradient descent

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# Hyperparameters are adjustable parameters that let you control the model optimization process. 
# Number of Epochs - the number times to iterate over the dataset
# Batch Size - the number of data samples propagated through the network before the parameters are updated
# Learning Rate - how much to update models parameters at each batch/epoch.

learning_rate = 1e-3
batch_size = 64
epochs = 10

# Train and optimize our model with an optimization loop. Each iteration of the optimization loop is called an epoch.
# The Train Loop - iterate over the training dataset and try to converge to optimal parameters.
# The Validation/Test Loop - iterate over the test dataset to check if model performance is improving.

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Choose an optimization algorithm, all optimization logic is encapsulated in the optimizer object.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop: loops over our optimization code
# 每次从 dataloader 中读取一个 batch，送进模型进行训练
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)                  # 获取训练数据集的总样本数，用于日志打印

    model.train()   # 设置模型为“训练模式”, 对某些层（如 BatchNorm、Dropout）有影响

    for batch, (X, y) in enumerate(dataloader):     # 遍历所有的 batch。每个 batch 是 (X, y), X：输入图像数据, y：标签（正确答案）
        # Forward prop, compute loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        loss.backward()             # 反向传播，计算所有参数的梯度
        optimizer.step()            # 根据梯度，更新模型参数
        optimizer.zero_grad()       # 清空上一步残留的梯度，准备下一轮

        # 每训练 100 个 batch 打印一次 loss
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 评估模型性能的函数（验证集或测试集用）。不会更新参数，只是查看模型表现。
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()        # 设置模型为“评估模式”, 这会关闭 Dropout、BatchNorm 的训练行为

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:     # 遍历测试集
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # 预测值 pred 与 y 比较, pred.argmax(1) 得到每个样本的预测类别, 与 y 比较得到布尔值，再转为 float，最后 sum() 求正确个数。同时累加 loss 总和。
    
    test_loss /= num_batches    # avergae loss per batch
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Pass loss function and optimizer to train_loop and test_loop
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")







