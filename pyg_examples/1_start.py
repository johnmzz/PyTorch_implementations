import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.nn import GCNConv

# A simple example of an unweighted and undirected graph with three nodes and four edges. 
# Each node contains exactly one feature.

# edge list, 2 edges (both directions)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

# node features
x = torch.tensor([[-1], 
                  [0], 
                  [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
# print(data)

# edge_index, i.e. the tensor defining the source and target nodes of all edges, is not a list of index tuples. 
# If you want to write your indices this way, you should transpose and call contiguous on it before passing them to the data constructor:
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index.t().contiguous())
# print(data)

# The elements in edge_index only hold indices in the range { 0, ..., num_nodes - 1}. i.e. VID compact form
data.validate(raise_on_error=True)  # checks if follows this form

# utility functions of Data
# print(data.keys())
# print(data['x'])
# for key, item in data:
#     print(f'{key} found in data')
# print('edge_attr' in data)
# print(f"num_nodes = {data.num_nodes}")
# print(f"num_edges = {data.num_edges}")
# print(f"num_node_features = {data.num_node_features}")
# print(f"has isolated nodes = {data.has_isolated_nodes()}")
# print(f"has self loops = {data.has_self_loops()}")
# print(f"is directed = {data.is_directed()}")

# Transfer data object to GPU.
device = torch.device('cuda')
data = data.to(device)



# Common Datasets
# An initialization of a dataset will automatically download its raw files and process them to the previously described Data format.
# load the ENZYMES dataset (consisting of 600 graphs within 6 classes)
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
# print(len(dataset))
# print(dataset.num_classes)
# print(dataset.num_node_features)

# We now have access to all 600 graphs in the dataset, which contains 37 nodes, each one having 3 features. 
# There are 168/2 = 84 undirected edges and the graph is assigned to exactly one class.
data = dataset[0]
# print(data)
# print(data.is_undirected())

# to create a 90/10 train/test split
train_dataset = dataset[:540]
test_dataset = dataset[540:]

# If you are unsure whether the dataset is already shuffled before you split, you can randomly permute it by running
dataset = dataset.shuffle()

# Download Cora, the standard benchmark dataset for semi-supervised graph node classification
dataset = Planetoid(root='./dataset/Cora', name='Cora')
# print(len(dataset))
# print(dataset.num_classes)
# print(dataset.num_node_features)

# the dataset contains only a single, undirected citation graph
data = dataset[0]
# print(data)
# print(data.is_undirected())
# print(data.train_mask.sum().item())     # denotes against which nodes to train (140 nodes)
# print(data.val_mask.sum().item())       # denotes which nodes to use for validation, e.g., to perform early stopping (500 nodes)
# print(data.test_mask.sum().item())      # denotes against which nodes to test (1000 nodes)



# Mini-batches
# PyG 采用了特殊的批处理方式，将多个图整合到一起，形成一个较大的图，但同时保留各个图之间的区分信息
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    print(data)                # data.batch is a column vector which maps each node to its respective graph in the batch
    print(data.num_graphs)

    # average node features in the node dimension for each graph individually
    x = scatter(data.x, data.batch, dim=0, reduce='mean')
    print(x.size())
    break



# Data Transforms
# PyG Transforms expect a Data object as input and return a new transformed Data object.

# 略



# Implement a simple GCN layer and replicate the experiments on the Cora citation dataset.
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# 定义了一个图卷积神经网络（GCN）模型，用于节点分类任务
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)     # 第一层图卷积，输入是每个节点的特征向量（维度为 dataset.num_node_features），输出为 16 维隐藏表示。
        self.conv2 = GCNConv(16, dataset.num_classes)           # 第二层图卷积，输入维度为 16，输出维度为类别数（dataset.num_classes），用于分类。

    def forward(self, data):
        # data 是一个 torch_geometric.data.Data 对象，表示图。
        # x：所有节点的特征矩阵（形状：[num_nodes, num_node_features]）。
        # edge_index：边的连接信息，COO 格式（形状：[2, num_edges]）。
        x, edge_index = data.x, data.edge_index

        # 图卷积 + 激活 + Dropout + 输出
        x = self.conv1(x, edge_index)               # 应用第一层图卷积，将原始节点特征映射到一个新的空间（维度变为 16），自动对邻居进行信息聚合。
        x = F.relu(x)                               # 给图卷积后的结果添加非线性
        x = F.dropout(x, training=self.training)    # 正则化手段，防止过拟合。只在训练时生效（training=self.training）。
        x = self.conv2(x, edge_index)               # 再次图卷积，将中间特征（16维）映射为输出类别空间

        # 使用 log_softmax 得到每个节点的 对数概率分布。
        return F.log_softmax(x, dim=1)      # dim=1 表示按每个节点的特征维度进行 softmax（即：对每个节点的类别进行归一化）。
    
# 训练 GCN 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN().to(device)        # 创建一个图神经网络模型实例，将模型参数移动到指定设备（GPU）
data = dataset[0].to(device)    # 选取数据集中的第一张图（比如 Cora 只包含一张图，是节点分类任务），同样将这张图的所有数据（节点特征、边、标签等）移动到设备上。

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()   # 将模型设置为“训练模式”，使得 dropout、生BN 等操作生效。
for epoch in range(200):
    optimizer.zero_grad()       # 清除上一次梯度残留，否则会累加。
    out = model(data)           # 将图数据输入模型，返回的是 所有节点的分类概率（对数形式），out.shape = [num_nodes, num_classes]

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])         # 计算损失（仅用训练集节点），nll_loss（负对数似然）适用于 log_softmax 输出

    loss.backward()             # 自动计算每个参数的梯度
    optimizer.step()            # 更新模型参数


# 在测试集上评估模型的准确率
model.eval()        # 将模型设置为评估模式，Dropout 被关闭；BatchNorm 使用固定均值和方差；

# model(data) 会执行前向传播，返回的是每个节点的 log softmax 输出（对数概率分布），形状为 [num_nodes, num_classes]。
# .argmax(dim=1) 作用是：对每一行（每个节点）找到最大概率对应的类别索引
# 输出的 pred 是一个 shape 为 [num_nodes] 的一维向量，记录每个节点预测的类别
pred = model(data).argmax(dim=1)

correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()    # 统计预测正确的个数

acc = int(correct) / int(data.test_mask.sum())      # 正确率 = 正确数 / 总数
print(f'Accuracy: {acc:.4f}')

