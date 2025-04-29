"""
@File    : custom_PPO.py
@Author  : roc
@Time    : 2024/3/26 21:11
"""
'''
2p是pkl里的p值在计算时乘2
1000epoch-1step_epoch
ICM内在好奇心机制
固定高斯噪声1
经验回放，存每个小action,设置10000的buf容量
reward只取覆盖比率
reward取max
15卫星数据(tuple_list.pkl是gf1b,satellite_15list.pkl是gf3c)
lr=0.005
50个traj更新一次
一个traj包含所有卫星条带
# 加入每隔5小轮评估一次，用好用的ppo继续训练
改GPU
actor网络拆分成层
critic网络拆分成层
选择beta或gaussian概率分布作为网络输出
默认entropy
Tanh激活函数
-actloss+entropy
criticloss
-0.1Gauss-dist-entropy
0.02lr
在get_dist中加入噪声noise
'''
# import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
# from torch.distributions import Categorical
# from torch.distributions import  MultivariateNormal
import torch.nn.functional as F
from tqdm import tqdm
# import custom_env
import custom_env_v3
import pickle
import copy
# import queue

from torch.utils.tensorboard import SummaryWriter
# import torch.distributions
from torch.distributions import Beta, Normal
import random
import time
random.seed(2)

# para_name = 'PPOV3-(15_gf1)--加0.005entropy_custom--(tanh-beta-G-noise)--(epoch100-50-lr0001)--(envi1)--bufcapacity--ICM'
para_name = 'PPOV3-Dongting(5-9)-2p-maxreward-noICM-1000-20-40-criticloss'
writer_reward = SummaryWriter(log_dir='./custom1/'+para_name+'-reward')
writer_loss = SummaryWriter(log_dir='./custom1/'+para_name+'-loss')
writer_actloss = SummaryWriter(log_dir='./custom1/'+para_name+'-actloss')
writer_critloss = SummaryWriter(log_dir='./custom1/'+para_name+'-critloss')
writer_entropy = SummaryWriter(log_dir='./custom1/'+para_name+'-entropy')
# 定义神经网络模型
class Actor_Beta(nn.Module):
    def __init__(self, obs_size, act_size, std=None):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.alpha_layer = nn.Linear(64, act_size)
        self.beta_layer = nn.Linear(64, act_size)
        self.activate_func = nn.Tanh()

    def forward(self, state):
        s = self.activate_func(self.fc1(state))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, state, std_noise, decay_factor=0.1):
        alpha, beta = self.forward(state)
        # std_ = std_noise  # 噪音的标准差
        decay_ = torch.tensor(decay_factor)
        std_ = std_noise #* torch.exp(-decay_)  # 标准差固定衰减
        noise = torch.randn_like(alpha)* std_   # 从标准正态分布中采样噪音
        noisy_alpha = alpha + noise  # 添加噪音到alpha参数
        noisy_alpha = torch.max(noisy_alpha, torch.tensor(1.0))  # 将noisy_alpha限制为大于等于1
        noisy_beta = beta + noise   # 添加噪音到beta参数
        noisy_beta = torch.max(noisy_beta, torch.tensor(1.0))  # 将noisy_beta限制为大于等于1
        dist = Beta(noisy_alpha, noisy_beta)
        return dist

    def mean(self, state):
        alpha, beta = self.forward(state)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean

# class Actor_Gaussian(nn.Module):
#     def __init__(self, obs_size, act_size, std=None):
#         super(Actor_Gaussian, self).__init__()
#         # self.max_action = args.max_action
#         self.fc1 = nn.Linear(obs_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.mean_layer = nn.Linear(64, act_size)
#         self.log_std = nn.Parameter(torch.zeros(1, act_size))  # We use 'nn.Parameter' to train log_std automatically
#         self.activate_func = nn.Tanh()  # Trick10: use tanh
#
#     def forward(self, state):
#         s = self.activate_func(self.fc1(state))
#         s = self.activate_func(self.fc2(s))
#         mean = torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
#         return mean
#
#     def get_dist(self, state):
#         mean = self.forward(state)
#         noise = torch.randn_like(mean)  # 从标准正态分布中采样噪音
#         log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
#         std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
#         noisy_mean = mean + noise * std  # 添加噪音到均值参数
#         dist = Normal(noisy_mean, std)  # Get the Gaussian distribution
#         return dist
class Critic(nn.Module):
    def __init__(self, obs_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.activate_func = nn.Tanh() # Trick10: use tanh

    def forward(self, state):
        s = self.activate_func(self.fc1(state))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class RolloutBuffer:
    """
    用于存储回合数据的缓冲区
    """
    def __init__(self,capacity: int):
        self.capacity = capacity
        self.act_buf = deque(maxlen=self.capacity)  # 动作
        self.obs_buf = deque(maxlen=self.capacity)  # 状态
        self.logp_buf = deque(maxlen=self.capacity)  # 动作的对数概率
        self.rew_buf = deque(maxlen=self.capacity)  # 奖励
        self.is_done = deque(maxlen=self.capacity)  # 终止标志

    # def __init__(self):
    #     self.act_buf = []  # 动作
    #     self.obs_buf = []  # 状态
    #     self.logp_buf = []  # 动作的对数概率
    #     self.rew_buf = []  # 奖励
    #     self.is_done = []  # 终止标志

    def clear(self):
        """
        清空缓冲区
        """
        del self.act_buf[:]
        del self.obs_buf[:]
        del self.logp_buf[:]
        del self.rew_buf[:]
        del self.is_done[:]

class ICMModel(nn.Module):
    """ICM model for non-vision based tasks"""
    def __init__(self, input_size, output_size):
        super(ICMModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.resnet_time = 4
        self.device = device

        self.feature = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.output_size)
        )

        self.residual = [nn.Sequential(
            nn.Linear(self.output_size + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        ).to(self.device)] * 2 * self.resnet_time

        self.forward_net_1 = nn.Sequential(
            nn.Linear(self.output_size + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(self.output_size + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

    def forward(self, state, next_state, action):

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(self.resnet_time):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action
class ICM(nn.Module):
    def __init__(self, obs_size, act_size, std=None,learning_rate = 1e-4,
        eta = 0.01):
        super(ICM, self).__init__()
        # self.fc0 = nn.Linear(obs_size + act_size, 64)
        # self.fc0_1 = nn.Linear(obs_size*2, 64)
        # self.fc1 = nn.Linear(obs_size, 64)
        # self.fc2 = nn.Linear(64, 64)
        # # self.alpha_layer = nn.Linear(64, act_size)
        # # self.beta_layer = nn.Linear(64, act_size)
        # self.state_next = nn.Linear(64, obs_size)
        # self.invpred_act = nn.Linear(64, act_size)
        # self.activate_func = nn.Tanh()
        self.model = ICMModel(obs_size, act_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.Invpred_act = nn.Sequential(
            nn.Linear(obs_size*2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_size)
        )
        self.Pred_next_state = nn.Sequential(
            nn.Linear(obs_size + act_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, obs_size)

        )
        self.eta = eta
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
    # # St的特征
    # def forward(self, state):
    #     s0 = self.activate_func(self.fc1(state))
    #     s0 = self.activate_func(self.fc2(s0))
    #     # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
    #     alpha0 = F.softplus(self.alpha_layer(s0)) + 1.0
    #     beta0 = F.softplus(self.beta_layer(s0)) + 1.0
    #     return alpha0, beta0
    # # St+1的特征
    # def forward1(self, state1):
    #     s1 = self.activate_func(self.fc1(state1))
    #     s1 = self.activate_func(self.fc2(s1))
    #     # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
    #     alpha1 = F.softplus(self.alpha_layer(s1)) + 1.0
    #     beta1 = F.softplus(self.beta_layer(s1)) + 1.0
    #     return alpha1, beta1
    def forward(self, state, next_state, action_params):
        cur_state = state
        real_next_state = next_state
        # get pred action
        x = torch.cat((cur_state, real_next_state), dim=1)
        pred_action = self.Invpred_act(x)
        # get pred next state
        y = torch.cat((state, action_params), dim=1)
        pred_next_state = self.Pred_next_state(y)
        pred_next_state = torch.round(pred_next_state)
        return real_next_state, pred_next_state, pred_action

    def compute_icm(self,state, next_state, action_param):
        cur_state = state
        real_next_state = next_state
        real_next_state, pred_next_state, pred_action = self.forward(cur_state, real_next_state, action_param)
        intrinsic_reward = self.eta * 2 * torch.norm(pred_next_state - real_next_state)
        return intrinsic_reward

    def Train(self, state, next_state, action_param):
        cur_state = state
        real_next_state = next_state
        real_next_state, pred_next_state, pred_action=self.forward(cur_state, real_next_state, action_param)
        inverse_loss = self.ce(pred_action, action_param)

        forward_loss = self.mse(pred_next_state, real_next_state)
        loss = 0.5*inverse_loss + 0.5*forward_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    # 预测下一状态forward
    # def pred_next_state(self,state, action_params):
    #     # action_params_reshape = action_params.unsqueeze(0)  # 在维度0上进行扩展，变为二维张量
    #     x = torch.cat((state, action_params), dim=1)
    #     x = self.activate_func(self.fc0(x))
    #     x = self.activate_func(self.fc2(x))
    #     next_state = torch.round(self.state_next(x)) # 舍入
    #     return next_state
    # # 反预测动作forward
    # def Invpred_act(self, current_state, next_state):
    #     x = torch.cat((current_state, next_state), dim=1)
    #     x = self.activate_func(self.fc0_1(x))
    #     x = self.activate_func(self.fc2(x))
    #     invpred_act = self.invpred_act(x)
    #     return invpred_act

    # def get_dist(self, state, std_noise, decay_factor=0.1):
    #     alpha, beta = self.forward(state)
    #     # std_ = std_noise  # 噪音的标准差
    #     decay_ = torch.tensor(decay_factor)
    #     std_ = std_noise #* torch.exp(-decay_)  # 标准差固定衰减
    #     noise = torch.randn_like(alpha)* std_   # 从标准正态分布中采样噪音
    #     noisy_alpha = alpha + noise  # 添加噪音到alpha参数
    #     noisy_alpha = torch.max(noisy_alpha, torch.tensor(1.0))  # 将noisy_alpha限制为大于等于1
    #     noisy_beta = beta + noise   # 添加噪音到beta参数
    #     noisy_beta = torch.max(noisy_beta, torch.tensor(1.0))  # 将noisy_beta限制为大于等于1
    #     dist = Beta(noisy_alpha, noisy_beta)
    #     return dist

    # def predict(self, obs_size, dist):


# 定义PPO算法
class PPO:
    def __init__(self, obs_size, act_size, lr, gamma, clip_ratio, device , buf_capacity, std_noise, distribution="Beta"):
        # self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)  # 定义优化器
        self.gamma = gamma  # 定义折扣因子
        self.clip_ratio = clip_ratio  # 定义PPO中的超参数
        # self.noise_std = noise_std
        self.policy_dist = distribution #"Gaussian"
        self.device = device
        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(obs_size, act_size).to(device)  # 初始化神经网络模型
        # else:
        #     self.actor = Actor_Gaussian(obs_size, act_size).to(device)
        self.critic = Critic(obs_size).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(buf_capacity)  # 回合缓冲区 todo 构建经验缓冲区
        self.std_noise = std_noise
        self.ICM = ICM(obs_size, act_size, std_noise).to(device)

    def evaluate(self, state,std_noise):  # 训练完用评价动作代替选择动作
        s = torch.tensor(state, dtype=torch.float).to(self.device)
        if self.policy_dist == "Beta":
            dist = self.actor.get_dist(s,std_noise)
            a = dist.sample()  # Sample the action according to the probability distribution
            a_logprob = dist.log_prob(a)
        else:
            dist = self.actor.get_dist(s,std_noise)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -1, 1)  # [-max,max]
            a_logprob = dist.log_prob(a)
        return a, a_logprob

    def choose_action(self, state,std_noise):
        s = torch.tensor(state, dtype=torch.float).to(self.device)
        # s = state.clone().detach().to(self.device)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s,std_noise)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s,std_noise)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -1, 1)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        # return a.numpy(), a_logprob.numpy()
        return a, a_logprob

    def update(self, rollouts):
        obs = rollouts[0]
        act = rollouts[1]
        rew = rollouts[2]
        logp_old = rollouts[3]
        val_old = rollouts[4]
        # obs_cuda = pad_sequence([torch.tensor(seq) for seq in obs], batch_first=True).to(device)
        obs_cuda = torch.stack(obs).to(device)
        # act_cuda = torch.tensor([torch.stack(act[item]) for item in act]).to(device)
        # act_cuda = [torch.stack(tensor).to(device) for tensor in act]
        act_cuda = torch.stack(act)
        # rew_cuda = torch.tensor(rew).to(device).unsqueeze(1).permute(0, 2, 1)
        rew_cuda = torch.tensor(rew).to(device).unsqueeze(1)
        # logp_old_cuda = [torch.stack(tensor).to(device) for tensor in logp_old]
        logp_old_cuda = torch.stack(logp_old).unsqueeze(1)
        # val_old_cuda = torch.tensor(val_old).to(device).unsqueeze(1).permute(0, 2, 1)
        val_old_cuda = torch.tensor(val_old).to(device).unsqueeze(1)
        # 计算价值函数
        returns = np.zeros_like(rew)  # 初始化returns
        for t in reversed(range(len(rew))):
            if t == len(rew) - 1:
                returns[t] = rew[t]
            else:
                returns[t] = rew[t] + self.gamma * returns[t + 1]  # 计算returns
        returns_cuda = torch.tensor(returns).to(device).unsqueeze(1)
        # for i in range(len(rew)):
        #     for t in reversed(range(len(rew[i]))):
        #         if t == len(rew[i]) - 1:
        #             returns[i][t] = rew[i][t]
        #         else:
        #             returns[i][t] = rew[i][t] + self.gamma * returns[i][t + 1]  # 计算returns
        # returns_cuda = torch.tensor(returns).to(device).unsqueeze(-1)
        # values = self.actor_critic(torch.tensor(obs).float())[1].detach().numpy()  # 得到状态的价值
        # values = self.critic(torch.tensor(obs).float()).detach().numpy()
        values = self.critic(obs_cuda.float()).detach().squeeze(2)
        # adv = returns - np.sum(values, axis=1)
        adv = returns_cuda - values
        '''
          计算优势估计，所以adv是一个形状(T,)一维数组，
          其中每个元素表示在当前状态下执行到第t步时的优势估计值，
          即从当前状态开始执行完整个轨迹所获得的累计奖励值减去在当前状态下执行所有动作的状态价值函数的估计值之和，
          表示执行某个动作相对于执行平均动作的优势程度。
        '''
        # 计算旧策略的动作概率和对数概率
        '''已经在上面转换成tensor了'''
        # act = np.array(act)
        # act = torch.tensor(act).float()  # 将动作转换为Tensor类型
        # # logp_old = np.array(logp_old)
        # # logp_old = torch.stack(logp_old)  # 将对数概率转换为Tensor类型
        # logp_old = torch.tensor(logp_old).float()

        act_para, act_logp = self.choose_action(obs_cuda, self.std_noise)
        # act_para = torch.tensor(act_para).float() # 将动作转换为Tensor类型
        # logp_now = torch.tensor(act_logp).float() # 将对数概率转换为Tensor类型  此为旧策略才去的的动作概率
        logp_now_cuda = act_logp
        '''输出即时tensor'''
        # act_para = self.actor_critic(torch.tensor(obs).float())[0]  # 网络输出动作
        # pi_old = self.actor_critic(torch.tensor(obs).float())[2]  # 网络输出的log  # 得到旧策略的动作概率
        #这个piold是根据输入的obs产生的动作，在下一步计算

        '''
        这句话的意思，是在当前状态obs下，执行动作act的概率。这个值是一个浮点数。下面我们详细分解其中的过程
        self.policy(obs)：得到动作概率分布，它是一个形状为(1，act_size)的一维tensor。
        act.unsqueeze(-1)，将动作act转换成一个形状为(1,)的一维数组，并在最后一个维度上添加一个新维度，变成一个形状为(1,1)的二维数组，用来在gather函数时与self.policy的输出对应。
        gather(1, act.unsqueeze(-1))，在self.policy(obs)的第一个维度（即动作维度）上，选取act.unsqueeze(-1)中指定的动作，返回一个形状为(1,)的一维数组，表示执行动作act的概率。
        .squeeze(-1)，将这个一维数组中的最后一个维度压缩掉，即将形状从(1,)变为( )，得到一个标量值，表示在当前状态obs下，执行动作act的概率。
        '''
        # ratio = pi_old / (torch.exp(logp_old) + 1e-8)
        # ratio = pi_old / (logp_old + 1e-8)
        # ratio = torch.exp(torch.sum(pi_old / logp_old, dim=1))
        # ratio = torch.sum(logp_now / logp_old, dim=1)
        ratio = torch.exp(logp_now_cuda) / torch.exp(logp_old_cuda)
        # ratio = ratio.sum(dim=-1, keepdim=True).squeeze(2)
        # ratio = torch.exp(logp_now - logp_old)
        # ratio = torch.exp(torch.log(pi_old) - logp_old)  # 计算比率
        # adv = torch.from_numpy(adv).float().unsqueeze(1)
        # adv_expanded = adv.repeat(1, 1, 2)
        surr1 = ratio * adv.unsqueeze(2)  # 第一项损失
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv.unsqueeze(2)  # 第二项损失
        # actor_loss = -torch.min(surr1, surr2).mean()  # actor损失函数

        # dist_now = self.actor.get_dist(torch.tensor(obs).float())
        dist_now = self.actor.get_dist(obs_cuda.float(),self.std_noise)
        # entropy_custom = self.compute_entropy(logp_now_cuda)#.mean()
        en_loss = dist_now.entropy()#.sum(-1, keepdim=True).squeeze(2)

        # actor_loss = torch.min(surr1, surr2).mean()- 0.1 * en_loss  # actor损失函数-熵 # TODO 连续动作的策略熵，可以用熵描述策略的动作概率分布
        actor_loss = -torch.min(surr1, surr2)
        actor_loss = actor_loss + 0.005 * en_loss # actor损失函数-熵
        actloss = actor_loss.mean()
        self.optimizer_actor.zero_grad()
        actor_loss.mean().backward()

        # 计算critic损失函数
        # val_old = torch.tensor(val_old).float()
        val_old = val_old_cuda.float()
        val = self.critic(obs_cuda.float()).squeeze(2)
        # critic_loss = nn.MSELoss()(val.squeeze(-1), torch.tensor(returns).float())
        critic_loss = nn.MSELoss()(val, returns_cuda.float())
        # Update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        # 计算策略熵需要的是概率probs,之前计算的都是对数概率logp,这里转换一下

        # 用Beta分布的entropy,最大化熵损失：计算所有熵值的负值，并取平均。这种方法鼓励动作具有更高的不确定性和多样性。
        # entropy = dist.entropy()
        # # entropy_loss = -torch.mean(entropy)
        # entropy_loss = torch.mean(entropy)
        # 更新神经网络参数
        # 总损失函数 = 动作损失 + c1*价值损失 + c2策略的熵
        # loss = actor_loss + 0.5 * critic_loss - 0.5 * self.entropy(logp_old2probs)
        loss = actloss + 0.5 * critic_loss
        en_loss = en_loss.mean()
        return loss,actloss,critic_loss,en_loss

    def compute_entropy(self,logprobs):
        probs = torch.exp(logprobs)
        entropy =probs * logprobs
        entropy.requires_grad = True
        entropy_result = -torch.sum(entropy, dim=0)
        return entropy_result


# 定义训练函数
def train(env_para, epochs, steps_per_epoch, mult_sat_param, lr, gamma, clip_ratio, buf_capacity, device, std_noise):
    workspace, input_layer, scale_layer, shp_output = env_para[:4]
    env = custom_env_v3.TargetEnv(workspace, input_layer, mult_sat_param, scale_layer)
    # env = custom_env.TargetEnv(workspace, input_layer, mult_sat_param, scale_layer, shp_output)
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]*len(mult_sat_param)
    ppo = PPO(obs_size, act_size, lr, gamma, clip_ratio, device , buf_capacity, std_noise,  distribution="Beta")  # 初始化PPO算法
    # normal = Normal(act_size)
    # ep_reward = deque(maxlen=10)  # 初始化双端队列
    best_ep_reward = 0  # 记录最大回合奖励
    output_agent = None  # 用于储存表现最好的智能体
    # ep_reward = []  # 初始化双端队列

    print('Started!')
    allepoch_sat_scheduled_result=[]
    # 把所有过程记录在下面
    per_epoch_loss = []
    per_epoch_actloss = []
    per_epoch_critloss = []
    per_epoch_entropy = []
    steps_per_sat_scheduled_result = []  # 很多traj的卫星规划结果
    for epoch in range(epochs):
        print(epoch)
        # print(ep_reward)
        ep_reward = []
        obs_buf, act_buf, rew_buf, logp_buf = [], [], [], []  # 初始化存储buffer TODO 这些buffer存放一整个回合所有的transition（obs,act,rew,log）

        # sat_obs_buf, sat_act_buf, sat_rew_buf, sat_logp_buf = [], [], [], []  # 初始化所有卫星一次选完的存储buffer
        # q = queue.Queue() # 卫星队列，按照时间排入队列
        # for _ in range(len(mult_sat_param)):
        #     q.put(_)
        # steps_per_epoch_loss = []
        # obs = env.reset()  # 重置环境
        # ep_reward.append(0)  # 初始化episode奖励
        steps_per_epoch_loss = []
        steps_per_epoch_actloss = []
        steps_per_epoch_critloss = []
        steps_per_epoch_entropy = []
        sat_result_traj = []  # 一个traj卫星规划结果
        for _ in tqdm(range(steps_per_epoch)):
            # per_ep = copy.deepcopy(_)  # 创建一个值相同的副本
            obs = env.reset()
            ep_reward.append(0)
            step_sat_result_traj = []
            # max_ep_reward = 0
            for t in range(10):
                # obs = env.reset()
                # obs_traj, act_traj, rew_traj, logp_traj = [], [], [], []  # 参与卫星的traject 的buffer
                # ICM_loss = []
            # for t in range(1): # 弃用此循环
                # 网络输出概率分布
                # act_para, logp = ppo.choose_action(torch.tensor(obs).float())
                all_obs_0 = [0]*len(obs)
                obs_tensor = torch.tensor(all_obs_0, dtype=torch.float)
                obs_tensor_reshaped = obs_tensor.unsqueeze(0).to(device)
                act_para, act_logp = ppo.choose_action(obs_tensor_reshaped,std_noise)
                act_para_copy = copy.deepcopy(act_para)
                # 将[[_,_]]->[_,_]
                act_para = act_para[0]
                act_logp = act_logp[0]

                # 计算第一个元素的缩放
                if ppo.policy_dist == 'Beta': # beta分布是[0,1]
                    # 将CUDA张量移回到CPU
                    act_para_cpu = act_para.cpu()
                    # 处理张量
                    result_tensor = act_para_cpu.clone()
                    # 计算
                    result_tensor[::2] *= 60
                    result_tensor[1::2] = result_tensor[1::2] * 2 - 1
                    act_para_gpu = result_tensor.to(device)

                    # # 从 [0, 1] 缩放到 [0, 100]，可以使用 x * 100
                    # scaled_a_1 =act_para[0] * 60
                    # # 计算第二个元素的缩放
                    # # 从 [0, 1] 缩放到 [-1, 1]，可以使用 (x - 0.5) * 2
                    # scaled_a_2 = (act_para[1] - 0.5) * 2

                    # 提取每个卫星参数元组的第7个元素（偏转角）
                    new_list = [item[6] for item in mult_sat_param]
                    # 扩展列表，在每个元素之间插入1，每个卫星的开关机时间，偏转角，时间，偏转角（因为时间已经在前面乘过60，所以这里乘1）
                    allSat_TimP_param = [val for sublist in [[1.0, val] for val in new_list] for val in sublist]
                    # 将allSat_TimP_param转换为PyTorch张量
                    allSat_TimP_param = torch.tensor(allSat_TimP_param).to(device)
                    # 与生成的所有卫星时间，角度的tensor点乘
                    act = act_para_gpu * allSat_TimP_param



                # obs_buf.append(obs)  # 存储状态
                # act_buf.append(act_para)  # 存储动作 # 这里存在buff里的是生成出的原始动作参数，下面参与step计算的是对应转换成具体范围的动作
                # rew_buf.append(0)  # 存储奖励
                # logp_buf.append(act_logp)  # 存储对数概率

                # ppo.buffer.obs_buf.append(obs_tensor)
                # ppo.buffer.act_buf.append(act_para)
                # ppo.buffer.rew_buf.append(0)
                # ppo.buffer.logp_buf.append(act_logp)
                # cur_sat_param = mult_sat_param[t]
                # obs是动作后的环境，obs0是重置环境为0
                obs, rew, done, geo = env.step(act, mult_sat_param)  # 执行动作,这里step传入动作（time,P）和卫星t的parameter还有当前第几个卫星

                # TODO icm内在好奇心机制
                # ------------------- Train Intrinsic Curiosity Module ------------------- #
                next_obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(device)# 下一状态
                # ppo.ICM.Train(obs_tensor_reshaped, next_obs_tensor, act_para_copy) #current_state,next_state,action_params

                # ICM_reward = ppo.ICM.compute_icm(obs_tensor_reshaped, next_obs_tensor, act_para_copy)

                rew = rew + np.random.uniform(4.9, 5.0)#ICM_reward.item() #/len(mult_sat_param) # 把icm的reward缩小 参与计算卫星的数量 倍，确保累加不会出现太大的数字
                # ppo.buffer.rew_buf[-1] += rew  # 更新奖励buffer
                ep_reward[-1] += rew  # 更新episode奖励 TODO 这个奖励是当前卫星选出的最好的结果对应的所有奖励和
                # ep_reward.append(rew)
                obs_buf.append(next_obs_tensor)
                act_buf.append(act_para)
                rew_buf.append(rew)
                logp_buf.append(act_logp)
                step_sat_result_traj.append(geo) # 这里存每个卫星规划的结果，选出的卫星条带四点坐标,是把这一traj卫星的结果存

                if done:  # 如果True，update
                    break

            sat_result_traj.append(step_sat_result_traj)
            # obs_buf_list = [x for x in ppo.buffer.obs_buf]
            # act_buf_list = [x for x in ppo.buffer.act_buf]
            # rew_buf_list = [x for x in ppo.buffer.rew_buf]
            # logp_buf_list = [x for x in ppo.buffer.logp_buf]
            # loss, actloss, critic_loss, entropy_custom = ppo.update(
            #     (obs_buf_list, act_buf_list, rew_buf_list, logp_buf_list, np.zeros_like(rew_buf_list)))  # 更新策略
            loss, actloss, critic_loss, entropy_custom = ppo.update(
                (obs_buf, act_buf, rew_buf, logp_buf, np.zeros_like(rew_buf)))  # 更新策略
            steps_per_epoch_loss.append(loss.item())
            steps_per_epoch_actloss.append(actloss.item())
            steps_per_epoch_critloss.append(critic_loss.item())
            steps_per_epoch_entropy.append(entropy_custom.item())
        # 折线图里显示最大值
        per_epoch_loss.append(np.max(steps_per_epoch_loss))
        per_epoch_actloss.append(np.max(steps_per_epoch_actloss))
        per_epoch_critloss.append(np.max(steps_per_epoch_critloss))
        per_epoch_entropy.append(np.max(steps_per_epoch_entropy))

        max_ep_reward_index = ep_reward.index(max(ep_reward))
        # max_reward = ep_reward.index(max(ep_reward))
        steps_per_sat_scheduled_result.append(sat_result_traj[max_ep_reward_index])  # 这是把很多traj卫星的结果存入buffer
        # 最大值
        max_ep_reward = max(ep_reward)
        # 均值
        writer_reward.add_scalar(tag='reward', scalar_value=max_ep_reward, global_step=epoch)
        writer_loss.add_scalar(tag='loss', scalar_value=np.mean(per_epoch_loss), global_step=epoch)
        writer_actloss.add_scalar(tag='loss', scalar_value=np.mean(per_epoch_actloss), global_step=epoch)
        writer_critloss.add_scalar(tag='loss', scalar_value=np.mean(per_epoch_critloss), global_step=epoch)
        writer_entropy.add_scalar(tag='loss', scalar_value=np.mean(per_epoch_entropy), global_step=epoch)
        # allepoch_sat_scheduled_result.append(steps_per_sat_scheduled_result)
        # 每次只要这此epoch的卫星条带结果
        # temp = [[steps_per_sat_scheduled_result[epoch]]]
        env.epoch_create_shapefile_from_deque_file(steps_per_sat_scheduled_result, shp_output, epoch)
        print("Epoch: {}, Avg Reward: {:.2f}".format(epoch, max_ep_reward))  # 打印平均奖励

        # #最大值
        # max_reward = max(ep_reward)
        # reward_index = ep_reward.index(max_reward)
        # reward_max_schedule = sat_scheduled_result[reward_index]
        # reward_max_loss = steps_per_epoch_loss[reward_index] # 一个epoch的loss
        # writer_reward.add_scalar(tag='reward', scalar_value=max_reward, global_step=epoch)
        # writer_loss.add_scalar(tag='loss', scalar_value=reward_max_loss, global_step=epoch)
        # writer_actloss.add_scalar(tag='loss', scalar_value=np.mean(steps_per_epoch_actloss), global_step=epoch)
        # writer_critloss.add_scalar(tag='loss', scalar_value=np.mean(steps_per_epoch_critloss), global_step=epoch)
        # allepoch_sat_scheduled_result.append([reward_max_schedule])  # 保存每个epoch最大的那组卫星组合
        # # env.create_shapefile_from_deque_file(allepoch_sat_scheduled_result, shp_output)
        # print("Epoch: {}, Avg Reward: {:.2f}".format(epoch, max_reward))  # 打印最大奖励


        # print(ep_reward)
        # ep_reward = []
    output_agent = copy.deepcopy(ppo)
    writer_reward.close()
    writer_loss.close()
    writer_actloss.close()
    writer_critloss.close()
    writer_entropy.close()
    # 把最后一个epoch的所有steps_per_epoch的卫星条带结果输出,最大值时
    # env.create_shapefile_from_deque_file(allepoch_sat_scheduled_result, shp_output)
    # # 把最后一个epoch的所有steps_per_epoch的卫星条带结果输出,均值时
    # last_epoch_sat_scheduled_result = allepoch_sat_scheduled_result[-1]
    # env.create_shapefile_from_deque(last_epoch_sat_scheduled_result, shp_output)

    return output_agent

def test_ppo(env_para, epochs,steps_per_epoch, mult_sat_param, lr, gamma, clip_ratio, buf_capacity, device, std_noise, agent):
    workspace, input_layer, scale_layer, shp_output = env_para[:4]
    env = custom_env_v3.TargetEnv(workspace, input_layer, mult_sat_param, scale_layer)
    # env = custom_env.TargetEnv(workspace, input_layer, mult_sat_param, scale_layer, shp_output)
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0] * len(mult_sat_param)
    ppo = PPO(obs_size, act_size, lr, gamma, clip_ratio, device, buf_capacity, std_noise,
              distribution="Beta")  # 初始化PPO算法
    # normal = Normal(act_size)
    # ep_reward = deque(maxlen=10)  # 初始化双端队列
    best_ep_reward = 0  # 记录最大回合奖励
    output_agent = None  # 用于储存表现最好的智能体
    # ep_reward = []  # 初始化双端队列

    print('Started!')
    allepoch_sat_scheduled_result = []
    # 把所有过程记录在下面
    per_epoch_loss = []
    per_epoch_actloss = []
    per_epoch_critloss = []
    per_epoch_entropy = []
    steps_per_sat_scheduled_result = []  # 很多traj的卫星规划结果
    for epoch in range(1):
        print(epoch)
        # print(ep_reward)
        # ep_reward = deque(maxlen=10)
        ep_reward = []
        obs_buf, act_buf, rew_buf, logp_buf = [], [], [], []  # 初始化存储buffer TODO 这些buffer存放一整个回合所有的transition（obs,act,rew,log）

        steps_per_epoch_loss = []
        steps_per_epoch_actloss = []
        steps_per_epoch_critloss = []
        steps_per_epoch_entropy = []
        sat_result_traj = []  # 一个traj卫星规划结果
        for _ in tqdm(range(10)):
            # per_ep = copy.deepcopy(_)  # 创建一个值相同的副本
            obs = env.reset()
            ep_reward.append(0)
            step_sat_result_traj = []

            for t in range(1):

                all_obs_0 = [0] * len(obs)
                obs_tensor = torch.tensor(all_obs_0, dtype=torch.float)
                obs_tensor_reshaped = obs_tensor.unsqueeze(0).to(device)
                # act_para, act_logp = ppo.choose_action(obs_tensor_reshaped, std_noise)
                act_para, act_logp = ppo.evaluate(obs_tensor_reshaped, agent, std_noise)
                act_para_copy = copy.deepcopy(act_para)

                act_para = act_para[0]
                act_logp = act_logp[0]

                # 计算第一个元素的缩放
                if ppo.policy_dist == 'Beta':  # beta分布是[0,1]
                    # 将CUDA张量移回到CPU
                    act_para_cpu = act_para.cpu()
                    # 处理张量
                    result_tensor = act_para_cpu.clone()
                    # 计算
                    result_tensor[::2] *= 60
                    result_tensor[1::2] = result_tensor[1::2] * 2 - 1
                    act_para_gpu = result_tensor.to(device)

                    # 提取每个卫星参数元组的第7个元素（偏转角）
                    new_list = [item[6] for item in mult_sat_param]
                    # 扩展列表，在每个元素之间插入1，每个卫星的开关机时间，偏转角，时间，偏转角（因为时间已经在前面乘过60，所以这里乘1）
                    allSat_TimP_param = [val for sublist in [[1.0, val] for val in new_list] for val in sublist]
                    # 将allSat_TimP_param转换为PyTorch张量
                    allSat_TimP_param = torch.tensor(allSat_TimP_param).to(device)
                    # 与生成的所有卫星时间，角度的tensor点乘
                    act = act_para_gpu * allSat_TimP_param
                # obs是动作后的环境，obs0是重置环境为0
                obs, rew, done, geo = env.step(act, mult_sat_param)  # 执行动作,这里step传入动作（time,P）和卫星t的parameter还有当前第几个卫星

                # ------------------- Train Intrinsic Curiosity Module ------------------- #
                next_obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(device)  # 下一状态
                ppo.ICM.Train(obs_tensor_reshaped, next_obs_tensor,
                              act_para_copy)  # current_state,next_state,action_params
                ICM_reward = ppo.ICM.compute_icm(obs_tensor_reshaped, next_obs_tensor, act_para_copy)
                rew = rew + ICM_reward.item()  # /len(mult_sat_param) # 把icm的reward缩小 参与计算卫星的数量 倍，确保累加不会出现太大的数字
                ep_reward[-1] += rew  # 更新episode奖励 TODO 这个奖励是当前卫星选出的最好的结果对应的所有奖励和
                obs_buf.append(next_obs_tensor)
                act_buf.append(act_para)
                rew_buf.append(rew)
                logp_buf.append(act_logp)
                step_sat_result_traj.append(geo)  # 这里存每个卫星规划的结果，选出的卫星条带四点坐标,是把这一traj卫星的结果存
                if done:  # 如果True，update
                    break
            sat_result_traj.append(step_sat_result_traj)
        max_ep_reward_index = ep_reward.index(max(ep_reward))
        # max_reward = ep_reward.index(max(ep_reward))
        steps_per_sat_scheduled_result.append(sat_result_traj[max_ep_reward_index])  # 这是把很多traj卫星的结果存入buffer
        env.epoch_create_shapefile_from_deque_file(steps_per_sat_scheduled_result, shp_output, epoch)


# 训练模型
if __name__ == '__main__':
    start_time = time.time()
    print('Go!Go!Go!')
    # workspace = "E:/arcpy-/wuhan"
    # input_layer = "E:/arcpy-/monggulia_fire/maybe_fire_area1_fishnet.shp"
    # scale_layer = "E:/arcpy-/monggulia_fire/New_Shapefile.shp"
    workspace = "E:/Satellite_para/target"
    input_layer = "E:/Satellite_para/target/Dongting_Lake_fishnet.shp"
    scale_layer = "E:/Satellite_para/target/Dongting_Lake.shp"
    model_dir = "models/Dongting-ppo.pt"
    # select_layers = ["E:/arcpy-/wuhan/stripe_test.shp", "E:/arcpy-/wuhan/stripe_test1.shp"]
    # shp_output = "E:/arcpy-/wuhan/selected_feature.shp"
    shp_output = "E:/Satellite_para/shp_output/"+para_name+"_sat_output" # "E:/arcpy-/monggulia_fire/"+para_name+"_sat_output.shp"
    env_para=(workspace, input_layer, scale_layer, shp_output)
    # file_path = "tuple_list.pkl"
    file_path = "Dongting_satellites_param.pkl"
    with open(file_path, "rb") as file:
        loaded_data = pickle.load(file)
    mult_sat_param = loaded_data
    # 确保CUDA可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train(env_para, epochs=100, steps_per_epoch=200, mult_sat_param=mult_sat_param, lr=0.002, gamma=0.99, clip_ratio=0.2)
    # train(env_para, epochs=1000, steps_per_epoch=40, mult_sat_param=mult_sat_param, lr=0.001, gamma=0.99,clip_ratio=0.2, buf_capacity=10000, device=device,std_noise=1)# 经验回放区的容量
    # train('CartPole-v0', epochs=50, steps_per_epoch=4000, batch_size=128, lr=0.002, gamma=0.99, clip_ratio=0.2)
    ppo_model = torch.load(model_dir)
    test_ppo(env_para, epochs=100, steps_per_epoch=40, mult_sat_param=mult_sat_param, lr=0.001, gamma=0.99,
             clip_ratio=0.2,
             buf_capacity=10000, device=device, std_noise=1, agent=ppo_model)
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    print(f"代码运行时间：{elapsed_time:.6f} 秒")
