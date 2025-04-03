# This tutorial introduces how heterogeneous graphs are mapped to PyG and how they can be used as input to Graph Neural Network models.

# The heterogeneous ogbn-mag network:
#   1,939,743 nodes, 4 types: author, paper, institution and field of study.
#   21,111,007 edges, 4 types: 
#      - writes: An author writes a specific paper
#      - affiliated with: An author is affiliated with a specific institution
#      - cites: A paper cites another paper
#      - has topic: A paper has a topic of a specific field of study

# The task for this graph is to infer the venue of each paper (conference or journal).

from torch_geometric.data import HeteroData
from torch_geometric.datasets import OGB_MAG


# 使用 PyG 中的 HeteroData 来创建和表示一个异构图 (Heterogeneous Graph)，有多种节点类型（如作者、论文、机构），有多种边类型（如撰写、引用、隶属）
# PyG使用 HeteroData 提供了非常直观的方式去表示：
# - 节点特征: data[node_type].x
# - 边索引: data[(src_type, relation_type, dst_type)].edge_index
# - 边特征: data[(src_type, relation_type, dst_type)].edge_attr

# eg.
data = HeteroData()

# Node features
data['paper'].x = ...               # [num_papers, num_features_paper]
data['author'].x = ...              # [num_authors, num_features_author]
data['institution'].x = ...         # [num_institutions, num_features_institution]
data['field_of_study'].x = ...      # [num_field, num_features_field]

# Edge indices
data['paper', 'cites', 'paper'].edge_index = ...                        # [2, num_edges_cites]
data['author', 'writes', 'paper'].edge_index = ...                      # [2, num_edges_writes]
data['author', 'affiliated_with', 'institution'].edge_index = ...       # [2, num_edges_affiliated]
data['paper', 'has_topic', 'field_of_study'].edge_index = ...           # [2, num_edges_topic]

# Edge features
data['paper', 'cites', 'paper'].edge_attr = ...                         # [num_edges_cites, num_features_cites]
data['author', 'writes', 'paper'].edge_attr = ...                       # [num_edges_writes, num_features_writes]
data['author', 'affiliated_with', 'institution'].edge_attr = ...        # [num_edges_affiliated, num_features_affiliated]
data['paper', 'has_topic', 'field_of_study'].edge_attr = ...            # [num_edges_topic, num_features_topic]


# HeteroData 的默认设计是针对少量节点类型（例如作者、论文、机构）的场景：
# - 少数节点类型（比如2-10种），每个类型的节点数量较多。
# - 少数边类型（比如5-20种）。

# 由于不适用于知识图谱，以下部分略