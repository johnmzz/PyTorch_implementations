# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange #pip install einops
from typing import List
import random
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from timm.utils import ModelEmaV3 #pip install timm 
from tqdm import tqdm #pip install tqdm
import matplotlib.pyplot as plt #pip install matplotlib
import torch.optim as optim
import numpy as np

# Sinusoidal Embedding - 用来为时间步 t 生成位置编码（positional encoding），类似于 Transformer 中的 encoder
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):     # time_steps = 总步数，embed_dim = embedding 维度
        super().__init__()
        
        # 生成一个 time_steps x 1 的列向量，表示每个时间步的索引, 如（1000步）：
        # [[0],
        #  [1],
        #  [2], 
        #  ...
        #  [999]]
        position = torch.arange(time_steps).unsqueeze(1).float()
        
        # 生成频率缩放因子, 其作用是让不同维度上的正余弦周期不同。
        # torch.arange(0, embed_dim, 2): 生成从 0 到 embed_dim 的偶数索引，比如如果 embed_dim = 8，结果是：tensor([0, 2, 4, 6])
        # 每两个维度共享一个频率组：[sin, cos]。
        # -(math.log(10000.0) / embed_dim): 这是一个缩放因子，用于调整频率的变化范围。
        # 整体乘积: [0, 2, 4, 6] * (-log(10000) / embed_dim), 得到的是一个逐渐变小的数列，比如：[0, -1.15, -2.30, -3.45]
        # torch.exp(...): 对上面的数列做指数运算，得到：[1.0, e^-1.15, e^-2.30, e^-3.45]
        # [1.0, e^-1.15, e^-2.30, e^-3.45] ≈ [1.0, 0.316, 0.1, 0.031]
        # 这个结果就是正余弦嵌入中不同比例的频率值。
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        # 使用 sin 和 cos 构建 embedding，结果是一个大小为 [time_steps, embed_dim] 的嵌入矩阵
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)

        self.embeddings = embeddings
    
    # 向前传播：x 是输入图像（仅用于获取设备信息），t 是一个 batch 的时间步 tensor，表示每个样本的当前扩散步数。
    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)    # 根据时间步 t 从预计算的正余弦嵌入中选取对应的嵌入向量, dim = [batch_size, emb_dim]

        return embeds[:, :, None, None]  # 加两个维度，变成 [B, embed_dim, 1, 1]，以便后续可以与图像一起 broadcast 到卷积网络中。


# 残差块（Residual Block），其中结合了 时间步嵌入（timestep embeddings）
class ResBlock(nn.Module):
    # C: channel 数量，num_groups: GroupNorm 中的组数，dropout_prob: dropout 的概率
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # 使用 Group Normalization（比 BatchNorm 更适合 batch 小的情况，比如生成模型）
        # BatchNorm	在 同一channel & 跨样本 (B) 上归一化，适合大 batch（分类等）
        # LayerNorm	在 每个样本的所有通道 + 空间维度 上归一化，适合NLP / Transformer
        # GroupNorm 在 每个样本中按组划分的通道 + 空间维度 上归一化，适合小 batch / 图像生成
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)

        # 两个 3x3 的卷积层，保持输入输出通道数不变。
        # 输入通道数 in_channels = C；输出通道数 out_channels = C；卷积核大小为 3x3，
        # 在输入图像的边缘填充 1 个像素，确保输出特征图大小与输入一样；省略了的 stride=1
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
    
    # 向前传播
    # x 是输入特征图，形状：[B, C, H, W]
    # embeddings 是时间步嵌入（broadcast 成 [B, C, 1, 1]），代表“当前扩散到了第几步”
    def forward(self, x, embeddings):
        # 时间步嵌入加到特征图上
        # 取出前 C 个通道的时间嵌入，形状是 [B, C, 1, 1]，自动 broadcast 成 [B, C, H, W]
        x = x + embeddings[:, :x.shape[1], :, :]    # 然后和 x 做逐元素加法，相当于 每个位置都加了一份“时间信息”

        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))

        return r + x    # 残差连接，输出加回原始输入


# 用于图像特征图的多头自注意力模块，被用于 U-Net 结构中间层或瓶颈层
class Attention(nn.Module):
    # C: 输入通道数，num_heads: 注意力头数，dropout_prob: dropout 概率
    def __init__(self, C: int, num_heads:int , dropout_prob: float):
        super().__init__()

        # 线性投影：用于生成 Query、Key、Value（简称 QKV）向量
        self.proj1 = nn.Linear(C, C*3)      # 对每个 [C] 维的特征向量，用一个权重矩阵 [C, 3C] 做一次线性变换，输出 [3C] 维。
        self.proj2 = nn.Linear(C, C)

        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]  # 记录原始空间维度

        # 原本 dimension = [batch_num, channel_num, h, w]
        # 变成 [b, HW, C]，也就是一个长度为 HW 的序列。每个像素视作一个 token。
        x = rearrange(x, 'b c h w -> b (h w) c')

        # 线性变换 → 得到 QKV
        x = self.proj1(x)    # shape: [B, HW, 3C]

        # 将拼在一起的 [Q | K | V] 向量拆分出来，并重排成适合多头注意力结构的格式。
        # B: batch size，L: token 数量，C: 每个Q/K/V 的 channel 数
        # H: attention head 数量，K: 3 (Q,V,K)
        # 把最后一维拆成 [C = C/num_heads, H = num_heads, K = 3]，再 rearrange
        # 每个 attention head 拥有原始通道数 C 被平均分配的部分，也就是 C / num_heads 个通道。
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q,k,v = x[0], x[1], x[2]    # q.shape = [batch_num, head_num, token_num, channel_per_head]

        # 执行多头注意力
        # is_causal = False，不是自回归模式，不遮挡未来信息（适合 encoder）
        # 在 softmax attention 权重上加一点 dropout
        # x.shape = [batch_num, head_num, token_num, channel_per_head]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)

        # 恢复空间格式
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)

        # 把 x 变回卷积风格的数据格式 [B, C, H, W]，方便跟其他 U-Net 卷积层连接。
        return rearrange(x, 'b h w C -> b C h w')
    

# U-Net 中的编码层或解码层模块，支持：
#   可选的下采样或上采样（downsample 或 upsample）
#   可选的 self-attention 机制；
#   两个残差块（ResBlocks）用于局部特征处理；
#   在 forward 中返回两个输出：下/上采样后的结果，以及当前层的 feature map（用于 skip connection）。 
class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool,          # 是否是 decoder（上采样）
            attention: bool,        # 是否加入 attention 模块
            num_groups: int,        # GroupNorm 的组数
            dropout_prob: float,    # Dropout 概率
            num_heads: int,         # attention 头数
            C: int):                # 输入 channel 数量
        super().__init__()

        # 每个 ResBlock 中包含了 GroupNorm + ReLU + Conv + Dropout；
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)

        # 下采样或上采样选择：编码器通道数逐层加倍，解码器通道数逐层减半
        if upscale:     # decoder阶段
            # 使用反卷积（ConvTranspose2d）进行 上采样，通道数减半：C -> C//2
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            # 使用 stride=2 的普通卷积进行 下采样，通道数翻倍：C -> C*2
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)
    
    # x: 当前层输入图像特征 [B, C, H, W]
    # embeddings: timestep embedding（形状通常为 [B, C, 1, 1]）
    def forward(self, x, embeddings):
        # 第一个 ResBlock
        x = self.ResBlock1(x, embeddings)   # 用于初步特征处理，注入时间步信息。

        # 可选的 Attention
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)

        # 第二个 ResBlock
        x = self.ResBlock2(x, embeddings)   # 再次处理并增强特征

        # 返回：采样后的输出（下一层的输入），本层特征图 x（用于 skip connection）
        return self.conv(x), x  


# 完整的 U-Net 网络结构，用于 DDPM 中的 denoise 模型
#   使用 SinusoidalEmbeddings 加入时间步 t 的条件信息
#   编码器 + 解码器对称结构（典型的 U-Net）
#   层中支持 Attention（可选）
#   支持上采样 / 下采样切换
#   融合 skip-connection（跳跃连接
class UNET(nn.Module):
    def __init__(self,
            Channels: List = [64, 128, 256, 512, 512, 384],
            Attentions: List = [False, True, False, False, False, True],
            Upscales: List = [False, False, False, True, True, True],
            num_groups: int = 32,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            input_channels: int = 1,
            output_channels: int = 1,
            time_steps: int = 1000):
        super().__init__()
        self.num_layers = len(Channels)

        # 输入图像通道数（如 1）-> 第一层 U-Net 的通道数（如 64），保持尺寸不变
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)

        # 在 forward 中最后一层输出的通道是由：
        # 最后一层 decoder 输出：Channels[-1] // 2
        # 第一层 encoder 的 residual：Channels[0]
        # 拼接后再变换维度
        out_channels = (Channels[-1]//2)+Channels[0]

        # 最后两层
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

        # time step embedding，嵌入维度选用 max(Channels) 保证能 broadcast 给所有层用。
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))

        # 构造各层的 UnetLayer（共 6 层）
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        # 初始卷积
        x = self.shallow_conv(x)
        residuals = []      # 初始化 residuals 用于 skip-connection。

        # Encoder 部分（前 3 层）
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)       # 时间嵌入随 x 更新而重新生成
            x, r = layer(x, embeddings)
            residuals.append(r)                      # 把 r 存到 residuals 中备用（跳跃连接）

        # Decoder 部分（后 3 层）
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')

            # 把当前输出和前面 encoder 层的输出 residual 拼接；
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)

        # 输出阶段
        return self.output_conv(self.relu(self.late_conv(x)))


# DDPM 的调度器模块，负责控制每个时间步加入多少噪声，也就是所谓的 噪声调度表
# 每一步扩散的噪声量 beta
# 每一步累计噪声的保留比例 alpha_bar
class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()

        # beta: 在第 t 步要加入的噪声比例，从小到大均匀增长（线性调度），从 0.0001 → 0.02
        # 每个时间步都加入一点噪声，逐步加重
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)

        # 每一步的 alpha 是在第 t 步保留原图像的比例
        alpha = 1 - self.beta

        # alpha_bar: 到第 t 步为止，原图像保留了多少
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    # 传入一个整数时间步 t（或一个张量），返回 beta 和 alpha_bar
    def forward(self, t):
        return self.beta[t], self.alpha[t]
    

# 设置整个训练或推理过程中的“随机种子”，使结果可以重复（可复现性）
# 参数 seed 是你想要设置的随机种子，默认值是 42（经典的梗，来自《银河系漫游指南》，表示“宇宙终极答案”
def set_seed(seed: int = 42):
    torch.manual_seed(seed)                     # 设置 PyTorch 的 CPU 上的随机种子（用于如初始化权重、dropout 等）
    torch.cuda.manual_seed_all(seed)            # 设置所有 GPU 的种子，如果你用多卡（多 GPU），这可以确保每块卡上随机行为一致。
    torch.backends.cudnn.deterministic = True   # 强制使用确定性算法（即便速度慢），保证每次运行结果一致；
    torch.backends.cudnn.benchmark = False      # 禁用 cuDNN 的自动算法搜索
    np.random.seed(seed)                        # 设置 NumPy 的随机种子
    random.seed(seed)                           # 设置 Python 标准库 random 的种子


# 训练主函数：数据加载 → 训练 → 计算损失 → 权重保存 
def train(batch_size: int=64,           # 每个 batch 中的图片数
          num_time_steps: int=1000,     # 扩散步数
          num_epochs: int=15,           # 训练轮数
          seed: int=-1,                 # 随机种子
          ema_decay: float=0.9999,      # Exponential Moving Average 的衰减因子（用于参数平滑）
          lr=2e-5,                      # 学习率
          checkpoint_path: str=None):   # 预训练模型路径（可恢复训练）

    # 若传入 seed == -1，则使用随机 seed；否则用用户指定的 seed，确保训练可复现。
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    # 加载 MNIST 手写数字图像（1x28x28 灰度图），ToTensor() 转为 [0, 1] 范围的张量，drop_last=True: 保证最后一批完整， num_workers=4: 多线程加载数据加速
    train_dataset = datasets.MNIST(root='./data', train=True, download=False,transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    #  初始化模型相关组件
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)   # 扩散过程的 beta 和 alpha_bar 时间表
    model = UNET().cuda()                                       # U-Net 架构用于反向预测噪声 epsilon
    optimizer = optim.Adam(model.parameters(), lr=lr)           # Adam 优化器
    ema = ModelEmaV3(model, decay=ema_decay)                    # 维护模型参数的滑动平均，提升生成稳定性
    
    # 如果传入了 checkpoint 路径，恢复：模型权重，优化器状态，EMA 权重
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 定义损失函数
    criterion = nn.MSELoss(reduction='mean')    # 使用均方误差损失（MSE）来训练模型预测 epsilon（加噪声）

    # 主训练循环
    for i in range(num_epochs):
        total_loss = 0

        # 遍历所有 batch 图像，我们只用图像 x 不用 label
        for bidx, (x,_) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):     # tqdm: 进度条显示
            x = x.cuda()                # x.size() = [batch_size, 1, 28, 28]
            x = F.pad(x, (2,2,2,2))     # 左右上下各填充2像素 → 变成 32x32

            # 为每个样本 随机采样一个时间步 t
            t = torch.randint(0,num_time_steps,(batch_size,))   # t.size() = [batch_siz]

            # 生成噪声 epsilon：创建一个和 x 形状相同的张量，每个元素是从标准正态分布中采样的噪声
            e = torch.randn_like(x, requires_grad=False)

            # 给出每张图像在时间步 t 对应的 alpha_bar → reshape 为 [B, 1, 1, 1]
            a = scheduler.alpha[t].view(batch_size,1,1,1).cuda()

            # 加噪声：forward noising process，得到 xt
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)

            # 模型预测 + 计算损失
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()

            # 梯度更新 + EMA 更新
            loss.backward()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / (60000/batch_size):.5f}')
    
    # 最终保存模型
    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    torch.save(checkpoint, 'checkpoints/ddpm_checkpoint')


# 显示反扩散过程中生成图像的中间结果，比如 DDPM 从纯噪声一步步变成图像的过程。
def display_reverse(images: List):
    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()


# DDPM 的推理（inference）函数，从纯噪声一步步还原出图像
def inference(checkpoint_path: str=None,    # 加载训练好的模型和 EMA 参数；
              num_time_steps: int=1000,     # 反扩散的总步数（通常 1000）
              ema_decay: float=0.9999, ):   # EMA 平滑参数（同训练时一致）

    # 加载模型与调度器
    checkpoint = torch.load(checkpoint_path)
    model = UNET().cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

    # 记录中间图像
    times = [0,15,50,100,200,300,400,550,700,999]
    images = []

    # 关闭梯度，进入 eval 模式
    with torch.no_grad():
        model = ema.module.eval()

        # 主循环：采样 10 张图像
        for i in range(10):
            z = torch.randn(1, 1, 32, 32)    # 初始纯噪声
            
            # 从 t=T 到 t=1 开始反扩散
            for t in reversed(range(1, num_time_steps)):
                t = [t]     # 包装成列表以便支持 batch 操作

                # DDPM 原始推理均值公式的近似/简化形式，实践中生成效果差别不大
                # temp: 简化公式中控制噪声 epsilon 贡献的比例项
                temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t])) ))
                z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*model(z.cuda(),t).cpu())

                # 保存中间图像，用于展示最终的变化过程
                if t[0] in times:
                    images.append(z)

                # 添加新噪声（用于采样多样性）
                e = torch.randn(1, 1, 32, 32)
                z = z + (e*torch.sqrt(scheduler.beta[t]))

            # 最后一步 从 t=1 到 t=0，不用加噪音
            temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])) )
            x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*model(z.cuda(),[0]).cpu())

            # 展示结果图
            images.append(x)
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
            x = x.numpy()
            plt.imshow(x)
            plt.show()
            display_reverse(images)
            images = []


def main():
    train(checkpoint_path='checkpoints/ddpm_checkpoint', lr=2e-5, num_epochs=75)
    inference('checkpoints/ddpm_checkpoint')

if __name__== '__main__':
    main()