import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch
from torch import nn
import argparse
# import os
# import yaml
import time
import seaborn as sns
from transformer import ActorTransformer
# from soft_attention import ActorAttention
# from diffusion import Diffusion
# from implicit import Implicit

#
import h5py
import pandas as pd
from pathlib import Path
import numpy as np

dim_goal = 4
dim_obs = 7
dim_act = 5
history_size = 128
horizon_size = 8
total_time_step = 1
num_sample = 64

hidden_size = 128
batch_size = 128
num_epochs = 200
learning_rate = 1e-4

val_interval = 10
num_val_epochs = 1

device = "cuda" 
log_path = Path("data/")


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_goal = dim_goal
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.history_size = history_size
        self.horizon_size = horizon_size

        self.goal_mean = nn.Parameter(torch.zeros(dim_goal), requires_grad=False)
        self.goal_std = nn.Parameter(torch.ones(dim_goal), requires_grad=False)
        self.obs_mean = nn.Parameter(torch.zeros(dim_obs), requires_grad=False)
        self.obs_std = nn.Parameter(torch.ones(dim_obs), requires_grad=False)
        self.act_mean = nn.Parameter(torch.zeros(dim_act), requires_grad=False)
        self.act_std = nn.Parameter(torch.ones(dim_act), requires_grad=False)
        # self.net = nn.Sequential(
        #     nn.Linear(dim_obs, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU()
        # )

        # self.net_td = nn.Sequential(
        #     nn.Linear((dim_obs - 1) * history_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, dim_act)
        # )
        # self.net_ba = ActorTransformer(dim_obs - 1, dim_act)
        # self.net_ba = ActorTransformer(dim_obs - 1, hidden_size)
        self.net = ActorTransformer(dim_goal, dim_obs, dim_act, history_size, horizon_size)
        self.actor = ActorTransformer(dim_goal, dim_obs, dim_act, history_size, horizon_size)
        # self.net = ActorAttention(dim_goal, dim_obs, dim_act, history_size)

        self.diffusion = Diffusion(self.net, total_time_step, horizon_size)
        self.implicit = Implicit(dim_obs, dim_act, history_size, num_sample, self.net)

        # self.act_pred = torch.zeros((1, horizon_size, dim_act)).to(device)
        # self.mean = nn.Linear(hidden_size, dim_act)
        # self.log_std = nn.Linear(hidden_size, dim_act)

        # self.net_vc = nn.Sequential(
        #     nn.Linear(dim_obs * history_size, hidden_size),
        #     # nn.Linear(hidden_size + 1, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1)
        #     # nn.Linear(hidden_size, 1 * history_size)
        #     # nn.Linear(hidden_size, 4)
        # )
        # # self.net_vc = ActorTransformer(dim_obs, 1)

    # def forward(self, x):
    #     x_obs = x[:, :, :-1]
    #     x_obs = x_obs.reshape(x_obs.shape[0], -1)
    #     x = x.reshape(x.shape[0], -1)
    #     a = self.net_td(x_obs)
    #     a_k = self.net_vc(x)
    #     # mean = self.mean(x)
    #     action = a.clone()
    #     action[:, :3] *= 1 + a_k
    #     # action = torch.tanh(mean)
    #     self.action = torch.cat((a[0, :3], a_k[0], action[0])).cpu().detach().numpy()
    #     return action

    # def forward(self, x):
    #     x_obs = x[:, :, :-1]
    #     # x_time = x[:, -1, -1].unsqueeze(-1)
    #     # x_obs = x_obs.reshape(x_obs.shape[0], -1)
    #     x = x.reshape(x.shape[0], -1)
    #     a = self.net_ba(x_obs)
    #     # z = torch.cat((_z, x_time), dim=1)
    #     # a = self.net_vc(z)
    #     a_k = self.net_vc(x)
    #     # a_k = a_k.unsqueeze(-1)
    #     action = a.clone()
    #     # a_norm = torch.norm(a[:, :3], dim=1).unsqueeze(-1)
    #     # action[:, :3] *= a_k / a_norm
    #     # action[:, :, :3] *= 1 + a_k
    #     # action[:, :3] *= a_k
    #     action = torch.cat((action, a_k), dim=1)
    #     # self.action = torch.cat((a[0, :3], a_k[0], action[0])).cpu().detach().numpy()
    #     return action

    # def forward(self, goal, obs):
    #     goal = (goal - self.goal_mean) / self.goal_std
    #     obs = (obs - self.obs_mean) / self.obs_std

    #     act = self.net(goal, obs)

    #     act = self.act_std * act + self.act_mean
    #     return act

    # def forward(self, goal, obs, act, noised_act, noised_obs, time_step):
    #     # goal = (goal - self.goal_mean) / self.goal_std
    #     # obs = (obs - self.obs_mean) / self.obs_std
    #     # act = (act - self.act_mean) / self.act_std

    #     act_mean, act_log_std, obs_mean, obs_log_std = self.net(goal, obs, act, noised_act, noised_obs, time_step)

    #     # obs_mean = self.obs_std * obs_mean + self.obs_mean
    #     # obs_log_std += self.obs_std.log()
    #     # act_mean = self.act_std * act_mean + self.act_mean
    #     # act_log_std += self.act_std.log()
    #     return act_mean, act_log_std, obs_mean, obs_log_std
    
    def predict(self, goal, obs, act): #通过gaol, his_obs, his_act预测下一个动作
        goal = (goal - self.goal_mean) / self.goal_std
        obs = (obs - self.obs_mean) / self.obs_std
        # act = (act - self.act_mean) / self.act_std

        # act_pred, obs_pred = self.diffusion.sampler(goal, obs, act)
        # act_pred, obs_pred = self.implicit.sampler(goal, obs, act)
        # act_pred, _, obs_pred, _ = self.actor(goal, obs, act)
        act_pred = self.actor(goal, obs)
        # act_pred, obs_pred = self.actor(goal, obs, act)
        # act_pred, obs_pred, f_act_pred, f_obs_pred = self.actor(goal, obs, act)
        act_pred = act_pred[0, -1] #取最后一个动作(时间t)
        # obs_pred = obs_pred[0, -1]
        # self.act_pred[:, :-1] = self.act_pred[:, 1:] * 0.01
        # self.act_pred += act_pred * 0.99
        # # print(self.act_pred)
        # act_pred = self.act_pred
        # act_pred = act_pred[0, 0]
        # obs_pred = obs_pred[0, 0]
        
        act_pred = self.act_std * act_pred + self.act_mean
        # act_log_std += self.act_std.log()
        # obs_pred = self.obs_std * obs_pred + self.obs_mean
        # obs_log_std += self.obs_std.log()
        # return act_pred, obs_pred
        return act_pred
    
    # def sample(self, x):
    #     x = self.net(x)
    #     mean = self.mean(x)
    #     log_std = self.log_std(x)

    #     noise = torch.randn_like(mean)
    #     action = mean + noise * log_std.exp()

    #     action = torch.tanh(action)
    #     return action

    # def output_act(self):
    #     return self.action

class MakeDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        
        file_path = Path("hdf_data/hdf_dataset.hdf5") #没问题，可以读取
        ee_vel = []
        grip = []
        ee_pos = []
        obj_pos = []
        t_pos = []
        episode = []
        
        #处理action
        #处理末端速度模长
        with h5py.File(file_path, 'r') as f:
            for i in range(10):
                key = f'data/demo_{i}/obs/ee_velocity_abs'
                ee_vel_data = f[key]
                ee_vel.append(pd.DataFrame(ee_vel_data))
        
        ee_vel = pd.concat(ee_vel,axis=0)
        ee_vel = ee_vel.values #将DataFrame转换为numpy数组
        ee_vel_direc= np.array([np.linalg.norm(ee_vel, axis=1)]) 
        ee_vel_direc[ee_vel_direc == 0] = 1e-8 #防止除0错误
        ee_vel_direc = ee_vel_direc.reshape(-1, 1) #将ee_vel_direc转换为二维数组，方便广播
        ee_vel_abs= ee_vel/ee_vel_direc       #计算归一化后的末端速度   
        
        with h5py.File(file_path, 'r') as f:
            for i in range(10):
                key = f'data/demo_{i}/obs/actions'
                grip_data = f[key]#[:args.data_size]
                grip_data = grip_data[:,-1]
                grip.append(pd.DataFrame(grip_data))
        
        grip = pd.concat(grip,axis=0)
        grip = grip.values #将DataFrame转换为numpy数组 

        #总的action,将ee_vel_abs,ee_vel_direc,grip合并
        expert_actions = np.concatenate([ee_vel_abs,ee_vel_direc,grip],axis=1) #axis=1表示横向合并

        # print(expert_actions)
        # print(expert_actions.shape)
        # print(expert_actions.dtype)

        #处理observation
        with h5py.File(file_path, 'r') as f:
            for i in range(10):
                key = f'data/demo_{i}/obs/ee_position'
                ee_pos_data = f[key]#[:args.data_size]
                ee_pos.append(pd.DataFrame(ee_pos_data))

        ee_pos = pd.concat(ee_pos,axis=0)     
        ee_pos = ee_pos.values

        with h5py.File(file_path, 'r') as f:
            for i in range(10):
                key = f'data/demo_{i}/obs/object_position'
                obj_pos_data = f[key]#[:args.data_size]
                obj_pos.append(pd.DataFrame(obj_pos_data))
        
        obj_pos = pd.concat(obj_pos,axis=0)
        obj_pos = obj_pos.values

        #总的observation,将ee_pos,grip,obj_pos合并
        expert_obs = np.concatenate([ee_pos,grip,obj_pos],axis=1)

        #处理goal
        with h5py.File(file_path, 'r') as f:
            for i in range(10):
                key = f'data/demo_{i}/obs/target_object_position'
                t_pos_data = f[key][:,:3]
                target_time = np.random.randint(60, 140 + 1)
                t_vel = 10 / target_time
                target_time_column = np.full((t_pos_data.shape[0], 1), t_vel)
                t_pos_data_with_time = np.hstack((t_pos_data, target_time_column))
                t_pos.append(pd.DataFrame(t_pos_data_with_time))

        t_pos = pd.concat(t_pos,axis=0)
        goal = t_pos.values

        #处理episode
        with h5py.File(file_path, 'r') as f:
            for i in range(10):
                key = f'data/demo_{i}/actions'
                episode_data = f[key]
                episode_data = np.full((episode_data.shape[0], 1), i)
                episode.append(pd.DataFrame(episode_data))

        episode = pd.concat(episode,axis=0)
        episode = episode.values






        self.acts= torch.tensor(expert_actions).float().to(device)
        self.obss= torch.tensor(expert_obs).float().to(device)
        self.goals= torch.tensor(goal).float().to(device)
        episode_index = list(episode)

        print(self.acts)
        # print(self.obss)
        # print(self.goals.shape)
        
        # # 打开HDF5文件
        # with h5py.File(file_path, 'r') as f:
        #     for i in range(10):
        #         key = f'data/demo_{i}/actions'
        #         actions_data = f[key][:args.data_size]  # 读取指定数量的数据
        #         df = pd.DataFrame(actions_data)  # 将数据转换为DataFrame
        #         df['i'] = i  # 添加新的列表示i值
        #         acts_list.append(df)

        # # 将所有DataFrame拼接成一个
        # expert_actions = pd.concat(acts_list)

        # with h5py.File(file_path, 'r') as f:
        #     for i in range(10):
        #         key = f'data/demo_{i}/obs/target_object_position'
        #         goals_data = f[key][:args.data_size]  # 读取指定数量的数据
        #         goals_list.append(pd.DataFrame(goals_data))  # 将数据转换为DataFrame
        

        # # 将所有DataFrame拼接成一个
        # expert_actions = pd.concat(acts_list)
        # expert_goals = pd.concat(goals_list)


        # expert_data = pd.read_csv(log_path / 'demo.csv')[:args.data_size] 
        # # expert_data = expert_data[expert_data.iloc[:, -1] % 2 == 0]
        
        # self.goals = torch.tensor(expert_data.iloc[:, :dim_goal].values).float().to(device)
        # self.obss = torch.tensor(expert_data.iloc[:, dim_goal:dim_goal + dim_obs].values).float().to(device)
        # self.acts = torch.tensor(expert_data.iloc[:, dim_goal + dim_obs:-1].values).float().to(device)
        # episode_index = list(expert_data.iloc[:, -1])

        # self.obss[:, -1] = 80
        # self.obss[:, -1] = 10 / self.obss[:, -1]
        # self.goals[:, -1] = 10 / 120
        # self.acts[:, :3] *= self.acts[:, 3].unsqueeze(-1)
        # self.acts[:, 3] = 1

        # target_norm = torch.norm(self.acts[:, :3], dim=1).unsqueeze(-1)
        # self.acts[:, :3] /= target_norm
        # self.acts = torch.cat((self.acts[:, :3], target_norm, self.acts[:, -1].unsqueeze(-1)), dim=1)

        # # delete the no-action data
        # index = []
        # for i in range(len(expert_data)):
        #     if torch.any(self.acts[i, :6]):
        #     # if torch.any(self.acts[i, :6]) and torch.sum(self.obss[i]) != float("-inf"):
        #         index.append(i)
        # self.obss = self.obss[index]
        # self.acts = self.acts[index]

        self.data_size = self.obss.shape[0]
        # self.data_size = len(self.history_obss)

        self.mean = {}
        self.std = {}
        self.mean["goal"] = torch.mean(self.goals, dim=0)
        self.std["goal"] = torch.std(self.goals, dim=0)
        self.mean["obs"] = torch.mean(self.obss, dim=0) #
        self.std["obs"] = torch.std(self.obss, dim=0)
        self.mean["act"] = torch.mean(self.acts, dim=0)
        self.std["act"] = torch.std(self.acts, dim=0)

        for key in self.std:
            self.std[key][self.std[key] == 0] = 1

        # self.goals = (self.goals - self.mean["goal"]) / self.std["goal"]
        # self.obss = (self.obss - self.mean["obs"]) / self.std["obs"]
        # self.acts = (self.acts - self.mean["act"]) / self.std["act"]

        # self.history_obss = []
        # self.history_acts = []
        # start_index = 0
        # for i in range(self.obss.shape[0]):
        #     if torch.sum(self.obss[i]) != float("-inf"):
        #         self.history_obss.append(self.obss[max(i - history_size + 1, start_index):i + 1])
        #         self.history_acts.append(self.acts[max(i - history_size + 1, start_index):i + 1])
        #     else:
        #         start_index = i + 1

        # self.history_obss = nn.utils.rnn.pad_sequence(self.history_obss, batch_first=True, padding_value=0)
        # self.history_acts = nn.utils.rnn.pad_sequence(self.history_acts, batch_first=True, padding_value=0)

        self.history_obss = []
        self.history_acts = []
        self.history_masks = []
        episode_value = -1
        for i in range(self.data_size):
            if episode_index[i] == episode_value:
                self.history_obss.append(torch.cat([self.history_obss[-1][:, 1:], self.obss[i].view(1, 1, -1)], dim=1))
                self.history_acts.append(torch.cat([self.history_acts[-1][:, 1:], self.acts[i].view(1, 1, -1)], dim=1))
                self.history_masks.append(torch.cat([self.history_masks[-1][:, 1:], torch.ones(1, 1, 1)], dim=1))
            else:
                # self.history_obss.append(self.obss[i].repeat(1, history_size, 1))
                self.history_obss.append(torch.cat([torch.zeros(history_size - 1, dim_obs).to(device), self.obss[i].view(1, -1)]).unsqueeze(0))
                self.history_acts.append(torch.cat([torch.zeros(history_size - 1, dim_act).to(device), self.acts[i].view(1, -1)]).unsqueeze(0))
                self.history_masks.append(torch.cat([torch.zeros(history_size - 1, 1), torch.ones(1, 1)]).unsqueeze(0))
                episode_value = episode_index[i]

        self.horizon_obss = []
        self.horizon_acts = []
        self.horizon_masks = []
        episode_value = -1
        for i in reversed(range(self.data_size)):
            if episode_index[i] == episode_value:
                self.horizon_obss.append(torch.cat([self.obss[i].view(1, 1, -1), self.horizon_obss[-1][:, :-1]], dim=1))
                self.horizon_acts.append(torch.cat([self.acts[i].view(1, 1, -1), self.horizon_acts[-1][:, :-1]], dim=1))
                self.horizon_masks.append(torch.cat([torch.ones(1, 1, 1), self.horizon_masks[-1][:, :-1]], dim=1))
            else:
                self.horizon_obss.append(torch.cat([self.obss[i].view(1, -1), torch.zeros(horizon_size - 1, dim_obs).to(device)]).unsqueeze(0))
                self.horizon_acts.append(torch.cat([self.acts[i].view(1, -1), torch.zeros(horizon_size - 1, dim_act).to(device)]).unsqueeze(0))
                self.horizon_masks.append(torch.cat([torch.ones(1, 1), torch.zeros(horizon_size - 1, 1)]).unsqueeze(0))
                episode_value = episode_index[i]

        self.goals = self.goals.unsqueeze(1)
        self.history_obss = torch.cat(self.history_obss)
        self.history_acts = torch.cat(self.history_acts)
        self.history_masks = torch.cat(self.history_masks).to(device)
        self.horizon_obss = torch.cat(list(reversed(self.horizon_obss)))
        self.horizon_acts = torch.cat(list(reversed(self.horizon_acts)))
        self.horizon_masks = torch.cat(list(reversed(self.horizon_masks))).to(device)

        self.goals = (self.goals - self.mean["goal"]) / self.std["goal"]
        self.history_obss = (self.history_obss - self.mean["obs"]) / self.std["obs"]
        self.history_acts = (self.history_acts - self.mean["act"]) / self.std["act"]
        self.horizon_obss = (self.horizon_obss - self.mean["obs"]) / self.std["obs"]
        self.horizon_acts = (self.horizon_acts - self.mean["act"]) / self.std["act"]

    

        # for i, episode in enumerate(episode_index):
        #     if episode == episode_index[-1]:
        #         self.val_index = i
        #         break
        # self.train_index = self.val_index
        # if args.num_episode is not None:
        #     for i, episode in enumerate(episode_index):
        #         if episode == args.num_episode:
        #             self.train_index = i
        #             break
        for index, episode in enumerate(episode_index):
            if episode != 0:
                self.val_index = index - 1
                break
        self.train_index = self.data_size
        if args.num_episode is not None:
            for index, episode in enumerate(episode_index):
                if episode == args.num_episode + 1:
                    self.train_index = index - 1
                    break
        # print(self.val_index, self.train_index)
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, i):
        # # return self.obss[i], self.acts[i]
        # # return self.goals[i], self.history_obss[i], self.acts[i]
        # # return self.goals[i], self.history_obss[i], self.history_acts[i]
        # return self.goals[i], self.history_obss[i], self.history_acts[i], self.history_masks[i]
        # return self.goals[i], self.history_obss[i], self.history_acts[i], self.horizon_obss[i], self.horizon_acts[i], self.history_masks[i]
        return self.goals[i], self.history_obss[i], self.history_acts[i], self.horizon_obss[i], self.horizon_acts[i], self.history_masks[i], self.horizon_masks[i]
        # 3 change points in trainer.py and 2 change points in controller.py for shifting Transformer model and simple NN model.


class Trainer:
    def __init__(self):
        dataset = MakeDataset()
        # train_indices = range(dataset.train_index)
        # val_indices = range(dataset.val_index, dataset.data_size)
        val_indices = range(dataset.val_index)
        train_indices = range(dataset.val_index, dataset.train_index)
        # print(val_indices, train_indices)
        self.train_data_size = len(train_indices)
        self.val_data_size = len(val_indices)

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        self.train_data = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
        self.val_data = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

        # self.model = ActorTransformer(dim_obs, dim_act)
        self.model = NN()
        self.model.to(device)

        # create a PyTorch optimizer
        self.optimizer_critic = torch.optim.AdamW(self.model.net.parameters(), lr=learning_rate)
        self.optimizer_actor = torch.optim.AdamW(self.model.actor.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction="none")

        self.model.goal_mean = nn.Parameter(dataset.mean["goal"], requires_grad=False)
        self.model.goal_std = nn.Parameter(dataset.std["goal"], requires_grad=False)
        self.model.obs_mean = nn.Parameter(dataset.mean["obs"], requires_grad=False)
        self.model.obs_std = nn.Parameter(dataset.std["obs"], requires_grad=False)
        self.model.act_mean = nn.Parameter(dataset.mean["act"], requires_grad=False)
        self.model.act_std = nn.Parameter(dataset.std["act"], requires_grad=False)

        self.mode = "implicit"

    # def loss(self, mean, log_std, target, mask=None):
    def loss(self, mean, target, mask=None):
        if mask is None:
            mask = torch.ones_like(mean).to(device)
            
        loss = self.criterion(mean, target)
        # else:
        #     # negative log liklihood
        #     # # loss = (((target - mean) / log_std.exp()).pow(2) / 2 + log_std).mean()
        #     loss = (((target - mean) / log_std.exp()).pow(2) / 2 + log_std)
        loss = (loss * mask).sum()
        # loss /= batch_size * (history_size * dim_act)
        return loss
    
    def loss_step(self, goals, obss, acts, f_obss, f_acts, masks, f_masks):
        # obss = obss * masks
        # acts = acts * masks
        if self.mode == "implicit":
            # time_steps = torch.randint(1, total_time_step + 1, (acts.shape[0], 1, 1)).to(device)
            # time_steps = torch.rand(acts.shape[0], 1, 1).to(device)
            # p_obs_noises = torch.randn_like(obss).to(device)
            # p_act_noises = torch.randn_like(acts[:, :-1]).to(device)
            # act_noises = torch.randn_like(acts).to(device)
            # # act_noises = self.model.diffusion.additional_noise(act_noises, time_steps)
            # obs_noises = torch.randn_like(obss[:, 1:]).to(device)
            # # obs_noises = self.model.diffusion.additional_noise(obs_noises, time_steps)
            # batch_data = goals, obss, acts[:, :-1], acts, obss[:, 1:]
            # batch_data = acts, obss[:, 1:]
            batch_data = f_acts, f_obss[:, 1:]
            # noised_acts, noised_obss, value_targets = self.model.implicit.noise_distribution(batch_data)
            noised_acts, noised_obss, value_act_targets, value_obs_targets = self.model.implicit.noise_distribution(batch_data)
            # noised_acts = noised_acts * masks
            # noised_obss = noised_obss * masks[:, 1:]
            # noised_acts = self.model.diffusion.forward_process(acts, act_noises, time_steps)
            # noised_acts = f_acts + act_noises
            # noised_obss = f_obss[:, 1:] + obs_noises
            # target_acts = self.model.diffusion.forward_process(acts, act_noises, time_steps - 1)

            # outputs = self.model(goals, obss)
            # outputs = outputs * masks
            # time_steps = time_steps / total_time_step
            # act_means, act_log_stds, obs_means, obs_log_stds = self.model(goals, obss, acts[:, :-1], noised_acts, noised_obss, time_steps)
            # value_means, value_log_stds = self.model.net(*noised_data)
            # value_means, value_log_stds = self.model.net(goals, obss, acts[:, :-1], noised_acts, noised_obss)
            value_act_means, value_obs_means = self.model.net(goals, obss, acts[:, :-1], noised_acts, noised_obss)
            # act_means = act_means * masks
            # act_log_stds = act_log_stds * masks
            # obs_means = obs_means * masks[:, 1:]
            # obs_log_stds = obs_log_stds * masks[:, 1:]

            # loss = self.loss(act_means, act_log_stds, acts, masks)
            # loss += self.loss(obs_means, obs_log_stds, obss[:, 1:], masks[:, 1:])
            # loss = self.loss(act_means, act_log_stds, act_noises, masks)
            # loss += self.loss(obs_means, obs_log_stds, obs_noises, masks[:, 1:])
            # loss = self.loss(value_means, value_log_stds, value_targets)
            # loss = self.loss(value_act_means, value_act_means, value_act_targets, masks)
            # loss += self.loss(value_obs_means, value_obs_means, value_obs_targets, masks[:, :-1])
            loss = self.loss(value_act_means, value_act_means, value_act_targets, f_masks)
            loss += self.loss(value_obs_means, value_obs_means, value_obs_targets, f_masks[:, 1:])
            # loss /= batch_size * (history_size * dim_act + (history_size - 1) * dim_obs)
            loss /= batch_size * (horizon_size * dim_act + (horizon_size - 1) * dim_obs)
            # loss /= batch_size * (2 * history_size - 1)
        elif self.mode == "actor": # 训练时采用的是actor模式
            # noise_scale = torch.randn(goals[0].shape[0], 1, 1).to(device) * 0.1
            # goals += torch.randn_like(goals).to(device) * noise_scale
            # obss += torch.randn_like(obss).to(device) * noise_scale * masks
            # acts += torch.randn_like(acts).to(device) * noise_scale * masks
        
            # act_means, act_log_stds, obs_means, obs_log_stds = self.model.actor(goals, obss, acts[:, :-1])
            act_means = self.model.actor(goals, obss)
            # act_means, obs_means = self.model.actor(goals, obss, acts[:, :-1])
            # act_means, obs_means, f_act_means, f_obs_means = self.model.actor(goals, obss, acts[:, :-1])
            # act_means = act_means * masks
            # obs_means = obs_means * masks[:, 1:]
            # act_target, obs_target = self.model.implicit.sample_target(goals, obss, acts[:, :-1], act_means, act_log_stds, obs_means, obs_log_stds)
            # value_act_means, value_obs_means = self.model.net(goals, obss, acts[:, :-1], act_means, obs_means)
            # with torch.no_grad():
            #     value_targets, _ = self.model.net(goals, obss, acts[:, :-1], acts, obss[:, 1:])
            
            # loss = self.loss(act_means, act_log_stds, act_target, masks)
            # loss += self.loss(obs_means, obs_log_stds, obs_target, masks[:, 1:])
            loss = self.loss(act_means, acts, masks)
            # loss += self.loss(obs_means, obss[:, 1:], masks[:, :-1])
            # loss += self.loss(f_act_means, f_acts[:, 1:], f_masks[:, 1:])
            # loss += self.loss(f_obs_means, f_obss[:, 1:], f_masks[:, 1:])
            # loss = value_means.sum() / batch_size * (2 * history_size - 1)
            # loss = (value_act_means * masks).sum()
            # loss += (value_obs_means * masks[:, 1:]).sum()
            # loss = (value_act_means * masks).pow(2).sum()
            # loss += (value_obs_means * masks[:, :-1]).pow(2).sum()
            # loss = (value_act_means * f_masks).pow(2).sum()
            # loss += (value_obs_means * f_masks[:, 1:]).pow(2).sum()
            loss /= batch_size * history_size * dim_act
            # loss /= batch_size * (history_size * dim_act + (history_size - 1) * dim_obs)
            # loss /= batch_size * ((history_size * dim_act + (history_size - 1) * dim_obs) + (horizon_size - 1) * (dim_act + dim_obs))
            # loss /= batch_size * (horizon_size * dim_act + (horizon_size - 1) * dim_obs)
            # loss /= batch_size * (2 * history_size - 1)
            # loss = self.loss(value_means, value_log_stds, value_targets) / batch_size
            # loss = - (- value_means).exp().sum() / batch_size
        return loss
        
    def train(self):
        # print the number of parameters in the model
        print(sum(p.numel() for p in self.model.parameters()) / 1e3, 'K parameters.')

        start_time = time.time()
        print("Training starts with {} data.".format(self.train_data_size))
        
        #定义两个list，用于存储训练和验证集的loss
        train_loss = []
        val_loss = []
    
        for epoch in range(num_epochs): #训练num_epochs次
            losses = [] #存储每个epoch的loss
            for goals, obss, acts, f_obss, f_acts, masks, f_masks in self.train_data:
                loss = self.loss_step(goals, obss, acts, f_obss, f_acts, masks, f_masks)

                if self.mode == "implicit":
                    self.optimizer_critic.zero_grad(set_to_none=True)
                    loss.backward() 
                    self.optimizer_critic.step()
                else:
                    self.optimizer_actor.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer_actor.step()

                losses.append(loss.item()) 

            epoch_loss = sum(losses) / (self.train_data_size / batch_size)
            train_loss.append(epoch_loss)

            epoch_val_loss = self.validate()
            val_loss.append(epoch_val_loss)
            # every once in a while evaluate the loss on train and val sets
            if epoch == 0 or (epoch + 1) % val_interval == 0:
                print(f"epoch {epoch + 1}: train loss {epoch_loss:.6f}, val loss {epoch_val_loss:.6f}")

        end_time = time.time()
        print("Training is finished in {} min.".format((end_time - start_time) / 60))

        self.visualize(train_loss, val_loss)
        torch.save(self.model.state_dict(), log_path / 'model')

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        losses = []
        for epoch in range(num_val_epochs):
            for goals, obss, acts, f_obss, f_acts, masks, f_masks in self.val_data:
                loss = self.loss_step(goals, obss, acts, f_obss, f_acts, masks, f_masks)

                losses.append(loss.item())
        val_loss = sum(losses) / (self.val_data_size / batch_size)
        self.model.train()
        return val_loss
    
    def visualize(self, train_loss, val_loss):
        sns.set(font_scale=2)
        plt.figure(figsize=(11, 6))
        plt.plot(range(1, num_epochs + 1), train_loss, label='train loss', color="darkblue")
        plt.plot(range(1, num_epochs + 1), val_loss, label='val loss', color="darkorange")
        # plt.xlim([0, num_epochs])
        # plt.ylim([0., 0.1])
        plt.ylim([0., 0.2])
        # plt.ylim([-3., 0.])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(log_path / 'loss')
        # plt.savefig(log_path / 'loss_{}'.format(self.mode))
        plt.close()

# def train():
#     dataset = MakeDataset()
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

#     model = NN(dim_obs, dim_act)
#     model.to(device)
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.MSELoss()

#     start_time = time.time()
#     print("Training starts with {} data.".format(dataset.data_size))
#     epoch_loss = []
#     for epoch in range(num_epochs):
#         losses = []
#         for obss, acts in dataloader:
#             # outputs = model.sample(obss)
#             outputs = model(obss)
#             loss = criterion(outputs, acts)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             losses.append(loss.item())
#             # batch_loss += loss.detach()

#         epoch_loss = sum(losses) / len(losses)
#         epoch_loss.append(epoch_loss)
#         # epoch_loss.append(batch_loss.cpu().numpy())

#     end_time = time.time()
#     print("Training is finished in {} sec.".format(end_time - start_time))

#     sns.set()
#     plt.plot(range(num_epochs), epoch_loss)
#     # plt.xlim([0, num_epochs])
#     # plt.ylim([0., 3.])
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     # plt.legend()
#     plt.savefig(log_path / 'loss')
#     # pd.DataFrame(epoch_loss).to_csv(log_path / 'loss', index=False)
#     plt.close()

#     torch.save(model.state_dict(), log_path / 'model')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--data_size", type=int)
    p.add_argument("--num_episode", type=int)
    p.add_argument("--mode", type=str)
    args = p.parse_args()

    # torch.manual_seed(0)

    trainer = Trainer() 
    # if args.mode != "actor":
    #     trainer.mode = "implicit"
    #     num_epochs = 100
    #     trainer.train()
    # else:
    #     trainer.model.load_state_dict(torch.load(log_path / "model"))
    #     trainer.model.actor.apply(trainer.model.actor._init_weights)
    #     trainer.model.to(device)
    trainer.mode = "actor"
    # num_epochs = 100
    trainer.train()
