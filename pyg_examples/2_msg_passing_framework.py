# The “MessagePassing” Base Class
# The user only has to define the functions: message() and update(), as well as the aggregation scheme to use, i.e. aggr="add", aggr="mean" or aggr="max"
import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU

# Implementing the GCN Layer
# Neighboring node features are first transformed by a weight matrix, normalized by their degree, and finally summed up.
# Lastly, we apply the bias vector to the aggregated output. 

# The following steps:
# 1. Add self-loops to the adjacency matrix.
# 2. Linearly transform node feature matrix.
# 3. Compute normalization coefficients.
# 4. Normalize node features
# 5. Sum up neighboring node features ("add" aggregation).
# 6. Apply a final bias vector.

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')                                # 指定消息聚合方式为加法
        self.lin = Linear(in_channels, out_channels, bias=False)    # 一个线性层，用于将输入特征从 in_channels 映射到 out_channels
        self.bias = Parameter(torch.empty(out_channels))            # 手动定义一个偏置参数，这样可以将偏置放到消息传递之后再加

        self.reset_parameters()

    def reset_parameters(self):     # 初始化权重矩阵（使用 PyTorch 默认的初始化策略）
        self.lin.reset_parameters()
        self.bias.data.zero_()
    
    # 核心部分，流程遵循 GCN 公式中的步骤。
    def forward(self, x, edge_index):
        # x size = [N, in_channels]， edge_index size = [2, E]
        
        # 1. 添加自环（self-loops）
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 2. Linear transform
        x = self.lin(x)         # x size = [N, out_channels]

        # 计算归一化系数（Normalization Coefficient）
        row, col = edge_index                           # # edge_index shape: [2, E]，row shape: [E]，col shape: [E]
        deg = degree(col, x.size(0), dtype=x.dtype)     # 计算每个节点的 in-degree，shape: [N]。这里使用的是 col，也就是边的源节点（默认 flow 是 'source_to_target'）
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]    # 所有目标节点的 deg * 所有源节点的 deg，shape: [E]
        
        # 消息传播（调用 propagate）
        # 这会调用：
        # message() → 对每条边提取 x_j 和 norm（调用 message）
        # aggregate() → 按节点收集邻居的消息（调用 aggregate，这里是加法）
        # update() →聚合完成后调用 update（默认不变）

        out = self.propagate(edge_index, x=x, norm=norm)    

        # 加上 bias
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j 是边源节点的特征（PyG 自动提取的），形状是 [E, out_channels]
        # norm 是每条边的归一化系数
        
        # 最终返回每条边的“消息”值，即 normalized feature = norm * x_j。
        # norm.view(-1, 1) 会把 norm 变成一个二维张量，形状 [E, 1]。这样 PyTorch 会自动将它 广播（broadcast） 成 [E, out_channels] 与 x_j 的形状一致。
        # -1 在 .view() 里的作用：自动推导维度。“你自己算吧，只要总元素数量别变就行！”
        return norm.view(-1,1)*x_j




# Implementing the Edge Convolution
# x_i^l = max MLP(x_i^(l-1), x_j^(l-1) - x_i^(l-1) ), where j in N(i)

# EdgeConv 是 GNN 的一种形式，通过对邻居节点进行差值计算再输入 MLP 来学习“边”的信息。
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') # 使用最大聚合

        # 一个两层的多层感知机（MLP）
        self.mlp = Seq(                 
            Linear(2 * in_channels, out_channels),  # 第一层输入是 2 * in_channels，因为后面我们要拼接两个特征（x_i 和 x_j - x_i）
            ReLU(),
            Linear(out_channels, out_channels)      # 第二层输出 out_channels，作为节点的新表示
        )

    def forward(self, x, edge_index):
        # x shape = [N, in_channels]， edge_index shape = [2, E]
        return self.propagate(edge_index, x=x)      # 调用 PyG 内置的 propagate 方法，自动处理消息传递机制
    
    # x_j 是源节点特征，x_i 是目标节点特征。x_j - x_i 表示边的“方向差异”——捕捉结构差异或相对位置
    def message(self, x_i, x_j):
        # 拼接成 [x_i || x_j - x_i]，形状为 [E, 2 * in_channels]
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)
    
# 这个类继承自 EdgeConv，不同之处在于它的图结构是动态构造的，即不是事先给定 edge_index，而是运行时用最近邻关系来生成。
# 略