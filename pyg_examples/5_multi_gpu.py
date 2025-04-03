# 在处理大规模图数据时，利用多 GPU 进行训练可以显著提升模型的训练效率。
# PyTorch Geometric（PyG）提供了与 PyTorch 的 torch.nn.parallel.DistributedDataParallel 模块集成的功能，
# 允许用户在不依赖其他第三方库（如 PyTorch Lightning）的情况下，实现多 GPU 的数据并行训练。

# 主要概念：
# 数据并行（Data Parallelism）：​每个 GPU 上运行相同的模型副本，但处理不同的数据子集。
# 各 GPU 独立计算梯度，随后在所有 GPU 之间同步梯度，并更新模型参数。

# eg. 在 Reddit 数据集上训练一个 GraphSAGE 图神经网络（GNN）模型。
# 我们将使用 torch.nn.parallel.DistributedDataParallel 来在所有可用的 GPU 上扩展训练过程。
# 我们将通过在 Python 代码中启动多个进程来实现这一点，这些进程将执行相同的函数。

# 在每个进程中，我们会单独设置一个模型实例，并通过使用 NeighborLoader 来向模型提供数据。
# 通过将模型封装在 torch.nn.parallel.DistributedDataParallel 中，可以实现梯度的同步。

from torch_geometric.datasets import Reddit
import torch.multiprocessing as mp


# 多 GPU 分布式训练的启动部分，为每张 GPU 启动一个独立的子进程。

# 在每一张可用 GPU 上并行地运行一个训练函数 run()，并给每个进程传入自己的 rank（编号）、总的 GPU 数量、和数据集 dataset。
# rank: 当前进程的编号（从 0 到 world_size - 1）
# world_size: 总共使用的进程（或 GPU）数量
# dataset: 主进程提前加载的 Reddit 数据集，并传给所有子进程（通过共享内存传递，不会复制多份）
# 在每个 GPU 上单独训练 GraphSAGE 模型的一份副本，利用 PyTorch 的 DistributedDataParallel（DDP）实现 分布式梯度同步训练。、
def run(rank: int, world_size: int, dataset: Reddit):
    # 初始化分布式通信环境，确保多个 GPU 能够互相同步梯度。
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # 获取图数据并划分训练节点
    data = dataset[0]                                   # Reddit 是单图任务，因此数据集中只有一个大图。
    train_index = data.train_mask.nonzero().view(-1)    # 找到所有训练节点的索引
    train_index = train_index.split(train_index.size(0) // world_size)[rank]    # 将训练节点平均分给所有进程，每个进程只处理一部分节点。

    # 构建 NeighborLoader
    train_loader = NeighborLoader(      # NeighborLoader 是 PyG 的 mini-batch 邻居采样器。
        data,
        input_nodes=train_index,
        num_neighbors=[25,10],          # num_neighbors=[25, 10] 表示从目标节点出发，分别在两跳中采样最多 25 和 10 个邻居。
        batch_size=[1024],              # 每个批次采样 1024 个训练节点及其邻域子图。
        num_workers=4,
        shuffle=True,
    )

    # 验证阶段不需要所有 GPU 同时进行，通常只让主进程（rank 0）做验证即可。
    if rank == 0:
        val_index = data.val_mask.nonzero().view(-1)
        val_loader = NeighborLoader(
            data,
            input_nodes=val_index,
            num_neighbors=[25, 10],
            batch_size=1024,
            num_workers=4,
            shuffle=False,
        )

    # 构建模型：
    torch.manual_seed(12345)
    model = GraphSAGE(                      # 使用 GraphSAGE 作为 GNN 模型结构。
        in_channels=dataset.num_features,
        hidden_channels=256,
        num_layers=2,
        out_channels=dataset.num_classes,
    ).to(rank)  # 将模型移动到当前进程对应的 GPU（rank）。

    # 用 DistributedDataParallel 包装，实现：多 GPU 同步梯度，自动广播参数
    model = DistributedDataParallel(model, device_ids=[rank])

    # 优化器设置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 模型训练循环
    for epoch in range(1,11):
        model.train()                   # 设置为训练模式
        for batch in train_loader:
            batch = batch.to(rank)      # 将当前子图移动到对应的 GPU。
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[:batch.batch_size]   # 模型会输出所有节点的预测，但只有前 batch_size 个是目标节点（中心节点），其他是邻居。
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()

        # 每个 epoch 的训练后，验证与同步
        dist.barrier()      # 等待所有进程同步，确保每个进程都完成了本轮 epoch 的训练后，再进入验证阶段。

        # 主进程打印 loss
        if rank == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        # 主进程执行验证逻辑
        if rank == 0:
            model.eval()                    # 进入评估模式（关闭 Dropout, BN 等）
            count = correct = 0
            with torch.no_grad():           # 不计算梯度，节省显存和计算资源
                for batch in val_loader:    # 遍历验证集（只在 rank 0 有构建）
                    batch = batch.to(rank)  # 将子图数据移动到当前 GPU

                    out = model(batch.x, batch.edge_index)[:batch.batch_size]   # 模型输出前 batch_size 个“中心节点”的预测

                    pred = out.argmax(dim=-1)       # 取预测概率最大类别作为结果

                    correct += (pred == batch.y[:batch.batch_size]).sum()   # 累加预测正确的节点数
                    count += batch.batch_size       # 记录总预测节点数量
            print(f'Validation Accuracy: {correct/count:.4f}')

        # 验证后再同步一次，确保主进程验证完成后，其他 GPU 再进入下一轮训练。
        dist.barrier()



# 让所有 GPU 进程同步，并由 rank=0 的主进程负责打印 loss 和 执行验证（Validation）

if __name__ == '__main__':
    dataset = Reddit('./data/Reddit')       # 在主进程中加载 Reddit 数据集（只加载一次）。
    world_size = torch.cuda.device_count()  # 获取当前机器上的 GPU 数量

    # run: 每个进程要执行的函数（上面定义的）
    # args=(world_size, dataset): 除了 rank 以外的其余参数（会传给每个进程）
    # nprocs=world_size: 启动的子进程数量，即使用的 GPU 数量
    # join=True: 主进程等待所有子进程结束后才退出
    mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)

    # 每个进程执行时将调用：run(rank=i, world_size=world_size, dataset=dataset)
