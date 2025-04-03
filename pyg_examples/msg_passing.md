非常棒的问题！理解 `propagate()`, `message()`, `update()` 是掌握 PyTorch Geometric (PyG) 中 `MessagePassing` 构建图神经网络的关键！我会从整体逻辑上讲清楚这三个函数的 **流程、作用、输入机制**，然后再解释你问到的 `message(x_j, norm)` 和 `message(x_i, x_j)` 的不同来源。

---

## 🧠 一、PyG 中的核心流程：`propagate → message → aggregate → update`

PyG 使用 `MessagePassing` 抽象消息传递机制，核心是这几个步骤：

### 1. `propagate()` —— 入口点，发起消息传递

你只需要在 `forward()` 里调用它，它会自动：

- 提取每条边上的邻居节点对（如 `i ← j`）
- 调用你定义的 `message()` 来计算消息
- 自动执行 `aggregate()`（聚合邻居消息，默认是加法/最大值/平均）
- 调用你定义的 `update()` 来更新节点嵌入

### 2. `message()` —— 定义每条边上消息的计算方式

对每一条边 \( i \leftarrow j \)，会调用：

```python
message(x_i, x_j, ...)
```

它表示 **邻居节点 j 向中心节点 i 发送消息**。

- `x_j`：源节点 j 的特征
- `x_i`：目标节点 i 的特征
- 你可以定义任何你想要用的函数，如 `x_j - x_i`，或加上边的特征等

### 3. `aggregate()` —— 聚合所有邻居消息（PyG 自动处理）

这个部分你通常不用自己实现，除非你要写自定义聚合逻辑（默认是你在 `super().__init__(aggr=...)` 里设定的，比如 `'add'`, `'mean'`, `'max'`）

### 4. `update()` —— 聚合完后对中心节点特征的更新（默认可以跳过）

默认就是直接返回聚合后的向量，但你可以加一些激活函数、MLP 等。

---

## ✅ 二、变量从哪里来的？为什么是 `x_j` 还是 `x_i`？

PyG 的神奇之处是：它通过 **参数命名的后缀** 来自动从传给 `propagate()` 的 `**kwargs` 中抽取你想要的特征。

---

### 🔹 你在 `propagate(edge_index, x=x, norm=norm)` 中传了：

- `x` 是节点特征 `[N, F]`
- `norm` 是边特征（广播用）或额外信息 `[E]`

---

### PyG 根据 `edge_index = [source, target]`：

- `_j` 后缀表示源节点（source）特征，比如 `x_j`
- `_i` 后缀表示目标节点（target）特征，比如 `x_i`

---

## 🧪 三、对比 GCN 和 EdgeConv 的 `message()` 输入

### ✅ GCN 中：

```python
def message(self, x_j, norm):
    return norm.view(-1, 1) * x_j
```

- 你只需要源节点的特征 `x_j`（邻居的）
- `norm` 是边归一化系数
- 中心节点 `x_i` 没有用上，所以没写

📌 因为 GCN 本质上就是用邻居节点的特征乘以一个权重（D^(-0.5) * A * D^(-0.5)）

---

### ✅ EdgeConv 中：

```python
def message(self, x_i, x_j):
    tmp = torch.cat([x_i, x_j - x_i], dim=1)
    return self.mlp(tmp)
```

- 你需要 `x_i`（中心节点）和 `x_j`（邻居）来计算 `x_j - x_i`
- 所以 message 函数要两个参数：`x_i`, `x_j`

📌 EdgeConv 的核心是：
> 「邻居和中心节点之间的特征差异」用于刻画边的信息，所以需要同时访问 `x_i` 和 `x_j`

---

## 📝 四、完整流程回顾：

你在 `forward()` 中调用：
```python
self.propagate(edge_index, x=x, norm=norm)
```

- PyG 识别：
  - `x_i`, `x_j` 来自 `x` 和 `edge_index`
  - `norm` 没后缀，默认是 `E` 维，会直接广播

然后 PyG 会自动：

1. 拿到边列表 `[2, E]`
2. 调用你写的：
   ```python
   message(x_i, x_j, norm)   # 看你 message 需要什么变量
   ```
3. 用 `aggr='max'` 或 `add/mean` 聚合
4. 调用 `update(aggr_out)`（你可以重写）

---

## ✅ 总结重点记忆：

| 函数        | 作用                            | 是否手写 |
|-------------|----------------------------------|-----------|
| `propagate` | 调度整个流程，从 edge_index 开始 | ❌（调用） |
| `message`   | 定义边上消息计算方式             | ✅（要写） |
| `aggregate` | 如何合并邻居消息（加/平均/最大） | ❌（默认即可） |
| `update`    | 聚合后的最终输出处理              | 可选写 |

后缀规则：
- `x_j`：邻居（源节点）
- `x_i`：自己（目标节点）

---

如果你想我用一张图画出这个流程，或者帮你写个自定义 GNN 层来巩固这些概念，我可以继续帮你补充 😄