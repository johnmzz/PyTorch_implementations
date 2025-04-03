import argparse
import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# parse arguments, eg: python reinforce.py --gamma 0.98 --render

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

# --gamma 折扣因子，用于计算 Gt (discounted reward)
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
# --seed 随机种子，保证结果可复现
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
# --render 如果设置，会渲染游戏画面
parser.add_argument('--render', action='store_true', help='render the environment')
# --log-interval 每隔多少回合打印一次状态
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')

args = parser.parse_args()




# 创建 Gym 中的 CartPole 环境（平衡小车）
env = gym.make('CartPole-v1')

# 用设定的随机种子初始化环境（确保每次运行轨迹一致）
env.reset(seed=args.seed)

# 设置 PyTorch 随机数种子，确保模型参数初始化一致
torch.manual_seed(args.seed)




# 策略网络类 Policy:
# 输入: 当前 state（状态）
# 输出: 当前 state 下每个 action 的概率分布
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)    # 全连接层，输入维度是 4（CartPole 的状态维度），输出 128
        self.dropout = nn.Dropout(p=0.6)    # Dropout 层，训练时随机丢弃 60% 神经元，防止过拟合
        self.affine2 = nn.Linear(128, 2)    # 全连接层，输入 128，输出 2（动作维度，CartPole 有两个动作：左或右）

        # 维护一些用于 REINFORCE 算法的中间变量
        self.saved_log_probs = []   # agent 每一步所采样的 action 的 log 概率
        self.rewards = []           # agent 每一步收到的 reward

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)  # 返回 [p(left), p(right)]


policy = Policy()                                       # 实例化 policy 网络
optimizer = optim.Adam(policy.parameters(), lr=1e-2)    # 使用 Adam optimizer
eps = np.finfo(np.float32).eps.item()                   # # 是一个极小的数值（≈1e-7），用于标准化 returns 时避免除以 0


# 策略采样（select_action）函数: agent 在某一时刻 根据当前状态 state 选择动作 action 的方式
def select_action(state):
    # 提取 state
    state = torch.from_numpy(state).float().unsqueeze(0)    # tensor dim = [1,4]
    # forward prop
    probs = policy(state)   # eg. probs = tensor([[0.73, 0.27]]) -> [左的概率，右的概率]

    m = Categorical(probs)  # 创建一个多项分布对象, 用于根据概率分布进行采样
    # 从这个分布中采样一个 action
    action = m.sample() 
    # 将这一步采样的 action 的 log probability 保存，稍后训练时会乘上对应 reward 作为梯度信号
    policy.saved_log_probs.append(m.log_prob(action))

    return action.item()    # 返回 action（0 或 1）
    

# 策略更新：在一个 episode（完整轨迹）结束后，根据所记录的 reward 和 log 概率 来更新策略网络的参数。
def finish_episode():
    R = 0               # 临时变量，用于从后往前计算 Gt（discounted return）
    policy_loss = []    # 存储每个时间步的 loss，用于 backprop
    returns = deque()   # 存储每个时间步的 Gt（从后往前算)

    # 计算 discounted return (从后往前)
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)

    # 标准化 normalize return, 转成均值为0、方差为1的分布
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # loss function
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)       #  因为我们用的是 PyTorch 的最小化 loss 机制，所以 -log_prob * R 实际上等于最大化 log_prob × R。

    # backprop + 更新参数
    optimizer.zero_grad()                       # 清空之前的梯度
    policy_loss = torch.cat(policy_loss).sum()  # total loss
    policy_loss.backward()                      # 自动微分，得到梯度
    optimizer.step()                            # 更新参数 

    # 清空缓存（为下一次 episode 做准备）
    del policy.rewards[:]
    del policy.saved_log_probs[:]


# 整个训练过程的主循环: 状态重置 - 动作采样 - 与环境交互 - 累计奖励 - 更新策略
def main():
    running_reward = 10     # 初始化一个滑动平均奖励，用于判断训练是否成功（reward 趋势是否变高）

    for i_episode in count(1):  # 无限训练循环，直到手动 break, count(1) 作用是每轮 episode 自增编号 i_episode = 1, 2, 3, ... 
        state, _ = env.reset()      # 重置环境，拿到初始状态 (state, info)
        ep_reward = 0               # 这一轮累计的 reward

        # agent 和环境互动（玩游戏 + 收集数据）
        for t in range(1, 10000):
            action = select_action(state)                   # 策略网络根据当前 state，采样一个 action
            state, reward, done, _, _ = env.step(action)    # 环境根据这个 action 返回下一 state、reward、是否结束等

            if args.render:
                env.render()    # 如果 --render 被设置，就在屏幕上可视化训练过程
            
            policy.rewards.append(reward)   # 将当前步得到的 reward 存入 policy.rewards（用于训练）
            ep_reward += reward             # 累计这轮的总 reward

            if done:    # 环境判断 episode 是否结束，例如小车倒了、杆子倾倒了等
                break

        # 计算滑动平均 reward，用于可视化和 early stopping
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # 策略更新: 执行一次 REINFORCE 参数优化
        finish_episode()

        # 每隔 log_interval 打印一次训练情况
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # 成功条件：Gym 环境内置的“成功判定”, 对于 CartPole-v1，这个值是 475
        if running_reward > env.spec.reward_threshold:      # 一旦滑动平均 reward 超过这个值，就宣布训练完成，退出循环
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()