import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 创建一个强化学习环境：CartPole（小车平衡一根杆子）
# 状态空间（Observation）：4维连续变量
# 动作空间（Action）：2个离散动作（向左推 or 向右推）
env = gym.make("CartPole-v1")

# 设置 matplotlib (for jupyter)
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# 设置设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# 实现 DQN（Deep Q-Network）中经验回放机制（Replay Memory） 
# 我们经常会用一个经验池来存储每一步的交互数据，然后从中随机采样 batch 来训练神经网络，这有助于打破时间相关性，稳定训练过程。

# namedtuple 是 Python 的轻量级结构体（类似简化版 class）
# Transition 表示一个完整的交互步骤（也叫经验），包括：
# state：当前状态（环境观测）
# action：采取的动作
# next_state：执行动作后的新状态
# reward：获得的奖励
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# 定义经验回放池（ReplayMemory 类）
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)    # 使用 deque（双向队列）存储经验

    def push(self, *args):
        self.memory.append(Transition(*args))       # 添加一个新经验

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)   # 随机采样一批经验
    
    def __len__(self):
        return len(self.memory)

# Our aim will be to train a policy that tries to maximize the discounted cumulative reward.
# The main idea behind Q-learning is that if we had a function Q* could tell us what our return would be, 
# if we were to take an action in a given state, then we could easily construct a policy that maximizes our rewards.
# Since neural networks are universal function approximators, we can simply create one and train it to resemble Q*.
# For our training update rule, we’ll use a fact that every Q function for some policy obeys the Bellman equation.
# The difference between the two sides of the equality is known as the temporal difference error.
# To minimize this error, we will use the Huber loss.

# DQN（Deep Q-Network） 的核心模型结构，目的是让神经网络 学习一个状态-动作值函数 Q(s, a)，从而根据当前状态预测每个动作的期望价值（Q-value）
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):  # n_observations: 状态空间的维度（state 的特征数量）, n_actions：动作空间的维度（可选动作数量）
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)    # 输入：当前状态（shape: [batch, n_observations]）
        self.layer2 = nn.Linear(128, 128)               # 中间层：两个 128 单元的隐藏层 + ReLU 激活
        self.layer3 = nn.Linear(128, n_actions)         # 输出：Q 值向量（shape: [batch, n_actions]）

    # 定义前向传播
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# epsilon-greedy: 
# The probability of choosing a random action will start at EPS_START and will decay exponentially towards EPS_END. 
# EPS_DECAY controls the rate of the decay.
# 超参数设置
BATCH_SIZE = 128          # 每次从 replay buffer 中采样 128 个 transition
GAMMA = 0.99              # 折扣因子 γ，用于延迟奖励计算（未来奖励的重要性）
EPS_START = 0.9           # epsilon 的起始值（刚开始大部分随机探索）
EPS_END = 0.05            # epsilon 的最小值（随着训练减少随机性）
EPS_DECAY = 1000          # epsilon 衰减速度，越大越慢
TAU = 0.005               # 目标网络更新速率（软更新）
LR = 1e-4                 # 学习率，AdamW 优化器的步长

# 提取环境信息
n_actions = env.action_space.n      # 当前环境允许的离散动作数（如 CartPole 是 2）

state, info = env.reset()
n_observations = len(state)         # 状态的维度（如 CartPole 是 4）
    
# 创建 Q 网络（behavior 和 target）
policy_net = DQN(n_observations, n_actions).to(device)      # target: 当前训练的主网络
target_net = DQN(n_observations, n_actions).to(device)      # behavior: 用于生成 target Q 值的稳定网络（不直接训练）(代码中叫 target 但实际上是 behavior network)
target_net.load_state_dict(policy_net.state_dict())         # 同步两者初始权重（后续会慢慢分化）

# optimizer + replay buffer
optimizer = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)    # 使用 AdamW 优化器更新策略网络
memory = ReplayMemory(10000)                                            # 使用 ReplayMemory 缓存过去经验（最多 10,000 条）

steps_done = 0

# 动作选择策略 (epsilon-greedy)
def select_action(state):
    # 计算 epsilon 值
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        return policy_net(state).max(1).indices.view(1, 1)  # greedy action, 返回 tensor size = [1, 1] 为了方便后续把多个状态组成 batch 时能够直接拼接
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)  # random action, 返回 tensor size = [1, 1]

# 在每个 episode 结束时 实时画出游戏持续的时间（通常用来衡量训练效果）
# 在像 CartPole 这样的环境中，episode 的“持续时间”越长，说明 agent 学得越好，因为它能更久地保持杆子不倒。
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# 模型优化函数
# 这是 DQN 中最关键的一步：从经验池中采样、计算 TD 目标值、再通过神经网络优化 Q 值函数。
def optimize_model():
    # 如果经验回放（memory）中存储的转换（Transition）数量还没达到一个 batch 的大小，则不执行优化更新。
    if len(memory) < BATCH_SIZE:
        return

    # 从经验回放中采样 batch 数据,（每个转换包含 state、action、next_state、reward）
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))      # converts batch-array of Transitions to Transition of batch-arrays

    # 对于终止状态（final state），通常不计算下一状态的价值。这里代码处理这一情况：
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)   # 创建一个布尔张量，表示 batch 中哪些样本的 next_state 不是 None（即不是终止状态）
    # eg. 假设 batch 中有3个transition，non_final_mask = tensor([True, False, True], dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])   # 将所有非终止状态（next_state）拼接成一个张量，便于后续计算 Q 值。

    # 对当前状态、动作、奖励也做拼接处理
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # eg, state_batch = tensor([            action_batch = tensor([         reward_batch = tensor([1.0, 0.0, 1.0])
    #        [1,0, 2.0],                        [0],
    #        [3.0, 4.0],                        [1],
    #        [5.0, 6.0]                         [0]
    #    ])                                 ])

    # 计算当前 (s,a) 的 Q 值
    state_action_values = policy_net(state_batch).gather(1, action_batch)   # state_action_values 是一个 [BATCH_SIZE, 1] 张量，表示每个样本当前动作的 Q 值。

    # 计算下一个状态的 Q 值（目标值）
    next_state_values = torch.zeros(BATCH_SIZE, device=device)  # 首先初始化所有下一状态的 Q 值为 0（默认对于终止状态，下一个状态价值为 0）

    with torch.no_grad():   # 利用目标网络计算非终止状态的 Q 值
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # 根据 Bellman 方程，计算目标 Q 值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算损失
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # 反向传播和参数更新    
    optimizer.zero_grad()   # 清空上一次梯度
    loss.backward()         # 计算梯度
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)   # 对梯度进行裁剪，防止梯度爆炸（将梯度最大值限制为 100）
    optimizer.step()        # 根据计算出的梯度更新策略网络参数


# 完整的 DQN 训练循环
# 强化学习的整个闭环过程：从初始化环境、选择动作、存储经验、训练网络，到更新目标网络
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # 初始化状态
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)    # 把 shape 从 [4] → [1, 4]，表示 batch=1 的状态张量

    for t in count():   # 无限循环，使用 itertools.count() 计数器来跟踪当前是第几个 timestep
        # 用 epsilon-greedy 策略选择动作
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())     # 与环境交互（step），获取：下一个状态, 即时奖励，是否“成功结束”，是否“时间到/强制终止”
        reward = torch.tensor([reward], device=device)      # 获得的即时奖励
        done = terminated or truncated                      # done = episode 是否结束

        # 如果是终止状态，不需要记录 next_state。否则转为 tensor
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 存储 transition 进 Replay Memory
        memory.push(state, action, next_state, reward)

        # 状态更新
        state = next_state

        # 训练网络（优化一步）
        optimize_model()

        # 用 TAU 进行 软更新：让 target network 缓慢接近 policy network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        # 如果游戏结束，记录 episode 长度并绘图
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

# 全部训练完成后，显示结果
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()





