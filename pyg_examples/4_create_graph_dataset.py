# PyG 提供了两种抽象类来帮助创建自定义数据集：torch_geometric.data.Dataset 和 torch_geometric.data.InMemoryDataset。
# InMemoryDataset 继承自 Dataset，适用于整个数据集可以完全加载到内存的情况。

# 按照 torchvision 的惯例，每个数据集都需要指定一个根目录 root，用于存储数据集。
# ​该目录下通常包含两个子目录：raw_dir（存放原始数据）和 processed_dir（存放处理后的数据）。
# ​此外，数据集类还可以接受以下可选参数：
#   transform：一个函数，在每次访问数据对象时动态转换数据，适用于数据增强。
#   pre_transform：一个函数，在将数据对象保存到磁盘之前应用转换，适用于需要预先计算且只需执行一次的操作。
#   pre_filter：一个函数，用于在保存数据对象之前进行筛选，例如，仅保留特定类别的数据对象。

# 要创建一个自定义的 InMemoryDataset，需要实现以下方法：
#   raw_file_names()：返回一个列表，包含 raw_dir 中需要存在的文件名，以便跳过下载步骤。
#   rocessed_file_names()：返回一个列表，包含 processed_dir 中需要存在的文件名，以便跳过处理步骤。
#   download()：将原始数据下载到 raw_dir。
#   process()：处理原始数据并将结果保存到 processed_dir。

# 在 process() 方法中，通常需要读取原始数据，创建一个或多个 Data 对象，并将它们保存到 processed_dir。
# ​为了提高保存效率，可以使用 collate() 方法将多个 Data 对象合并为一个大的 Data 对象，并生成一个 slices 字典，以便在需要时重新构建单个数据对象。
# 从 PyG 2.4 版本开始，torch.save() 和 collate() 的功能被统一并封装在 save() 方法中，此外，load() 方法可以自动加载 self.data 和 self.slices。

import torch
from torch_geometric.data import InMemoryDataset, download_url

# InMemoryDataset 是 PyG 中用于将整个数据集一次性加载进内存的类，适用于中小型图数据集
# 以下是一个创建自定义 InMemoryDataset 的简化示例：
class MyOwnDataset(InMemoryDataset):
    # root: 存放数据的主目录。它会自动创建子目录：raw_dir = root/raw/ 和 processed_dir = root/processed/
    # transform: 每次访问数据对象时动态应用的转换（如数据增强）
    # pre_transform: 在 process() 中预先应用的转换（只执行一次）
    # pre_filter: 用于在 process() 中过滤不需要的数据
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)    # 会自动检查数据是否已经存在。如果没有，会自动调用 download() 和 process()。
        self.load(self.processed_paths[0])                              # 加载保存好的处理结果。

    # raw_file_names：表示原始文件名列表。
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]  # 如果这些文件存在于 raw_dir/，就会跳过 download()。否则会触发 download()。

    # processed_file_names：表示处理后的文件名。
    @property
    def processed_file_names(self):
        return ['data.pt']      # 如果文件 processed/data.pt 存在，就跳过 process()。否则执行 process()。
    
    # 下载原始数据到 self.raw_dir
    @property
    def download(self):
        download_url("your_url", self.raw_dir)     # 你可以使用 download_url()，或者手动写文件复制逻辑。

    # 数据处理，通常会读取 raw_dir/ 中的文件，解析成 torch_geometric.data.Data 对象的列表（data_list）
    @property
    def process(self):
        data_list = [...]   # Read data into huge `Data` list.

        # 可选的预处理
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]   # 对每个 Data 对象应用 pre_filter。可以过滤掉某些不符合条件的样本（如：图节点太少、无标签等）。

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]        # 对每个样本执行预处理（如归一化、特征生成）。

        # 保存处理后的数据
        self.save(data_list, self.processed_paths[0])




# For creating datasets which do not fit into memory，略