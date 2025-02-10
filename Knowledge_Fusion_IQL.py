#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
import matplotlib.pyplot as plt
import torch.nn.utils as nnutils

STATE_DIM = 10
NUM_ACTIONS = 19
HIDDEN_SIZE = 256

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
GAMMA = 0.95
TAU = 0.7
EMA_ALPHA = 0.7
MAX_EPOCHS = 300

REWARD_SCALE = 0.1
ETA = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
torch.backends.cudnn.benchmark = True

class OfflineDataset(Dataset):
    def __init__(self, data_array):
        self.data = data_array
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.data[idx]
        state = row[:STATE_DIM]
        a_float = row[STATE_DIM:STATE_DIM+1]
        reward = row[STATE_DIM+1:STATE_DIM+2]
        next_state = row[STATE_DIM+2:STATE_DIM+2+STATE_DIM]
        action = int(a_float[0])
        return (torch.FloatTensor(state),
                action,
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state))

class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions=19, hidden_size=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, state):
        return self.net(state)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.net(state)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions=19, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, state):
        return self.net(state)

    def get_action_probs(self, state):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        return probs

def iql_update(q_net, q_opt,
               v_net, v_opt,
               pi_net, pi_opt,
               batch,
               gamma=0.95,
               tau=0.7,
               reward_scale=0.1):
    states, actions, rewards, next_states = batch
    bsz = states.shape[0]
    rewards = rewards * reward_scale
    q_all = q_net(states)
    q_pred = q_all[torch.arange(bsz), actions]
    with torch.no_grad():
        v_next = v_net(next_states).squeeze(-1)
        td_target = rewards.squeeze(-1) + gamma * v_next
    q_loss = (q_pred - td_target) ** 2
    q_loss = q_loss.mean()
    q_opt.zero_grad()
    q_loss.backward()
    nnutils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    q_opt.step()
    with torch.no_grad():
        q_all_s = q_net(states)
        a_star = q_all_s.argmax(dim=1)
        q_star = q_all_s[torch.arange(bsz), a_star]
    v_pred = v_net(states).squeeze(-1)
    diff = q_star - v_pred
    v_loss = torch.where(diff > 0, tau * diff, (1.0 - tau) * (-diff)).mean()
    v_opt.zero_grad()
    v_loss.backward()
    nnutils.clip_grad_norm_(v_net.parameters(), max_norm=10.0)
    v_opt.step()
    with torch.no_grad():
        q_all_s = q_net(states)
        v_s = v_net(states).squeeze(-1)
        adv = q_all_s - v_s.unsqueeze(-1)
        beta = 1.0
        weights = torch.exp(adv / beta)
        eps = 0.01
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8
        target_dist = (weights * (1 - eps)) / weights_sum + eps / weights.shape[1]
    logits = pi_net(states)
    log_probs = torch.log_softmax(logits, dim=-1)
    pi_loss = - torch.sum(target_dist * log_probs, dim=1).mean()
    pi_opt.zero_grad()
    pi_loss.backward()
    nnutils.clip_grad_norm_(pi_net.parameters(), max_norm=10.0)
    pi_opt.step()
    return {
        "q_loss": q_loss.item(),
        "v_loss": v_loss.item(),
        "pi_loss": pi_loss.item()
    }

def ema_fuse_params(params1, params2, alpha=0.7):
    fused_sd = {}
    for (k1, v1), (k2, v2) in zip(params1.items(), params2.items()):
        if k1 != k2:
            raise ValueError(f"Keys not match: {k1} vs {k2}")
        fused_sd[k1] = alpha * v1 + (1 - alpha) * v2
    return fused_sd

def main():
    path_original = "original_data.pkl"
    path_relabel = "relabel_data.pkl"
    with open(path_original, "rb") as f:
        data_original_list = pickle.load(f)
    with open(path_relabel, "rb") as f:
        data_relabel_list = pickle.load(f)
    data_original = np.array(data_original_list, dtype=np.float32)
    data_relabel = np.array(data_relabel_list, dtype=np.float32)
    dataset_original = OfflineDataset(data_original)
    dataset_relabel = OfflineDataset(data_relabel)
    loader_original = DataLoader(dataset_original, batch_size=BATCH_SIZE, shuffle=True)
    loader_relabel = DataLoader(dataset_relabel, batch_size=BATCH_SIZE, shuffle=True)
    q_net_1 = QNetwork(STATE_DIM, NUM_ACTIONS, HIDDEN_SIZE).to(device)
    v_net_1 = ValueNetwork(STATE_DIM, HIDDEN_SIZE).to(device)
    pi_net_1 = PolicyNetwork(STATE_DIM, NUM_ACTIONS, HIDDEN_SIZE).to(device)
    q_net_2 = QNetwork(STATE_DIM, NUM_ACTIONS, HIDDEN_SIZE).to(device)
    v_net_2 = ValueNetwork(STATE_DIM, HIDDEN_SIZE).to(device)
    pi_net_2 = PolicyNetwork(STATE_DIM, NUM_ACTIONS, HIDDEN_SIZE).to(device)
    q_net_2.load_state_dict(copy.deepcopy(q_net_1.state_dict()))
    v_net_2.load_state_dict(copy.deepcopy(v_net_1.state_dict()))
    pi_net_2.load_state_dict(copy.deepcopy(pi_net_1.state_dict()))
    q_opt_1 = optim.Adam(q_net_1.parameters(), lr=LEARNING_RATE)
    v_opt_1 = optim.Adam(v_net_1.parameters(), lr=LEARNING_RATE)
    pi_opt_1 = optim.Adam(pi_net_1.parameters(), lr=LEARNING_RATE)
    q_opt_2 = optim.Adam(q_net_2.parameters(), lr=LEARNING_RATE)
    v_opt_2 = optim.Adam(v_net_2.parameters(), lr=LEARNING_RATE)
    pi_opt_2 = optim.Adam(pi_net_2.parameters(), lr=LEARNING_RATE)
    epoch_q_loss_1, epoch_v_loss_1, epoch_pi_loss_1 = [], [], []
    epoch_q_loss_2, epoch_v_loss_2, epoch_pi_loss_2 = [], [], []
    
    for epoch in range(MAX_EPOCHS):
        all_q_1, all_v_1, all_pi_1 = [], [], []
        for batch_data in loader_original:
            batch_data = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch_data]
            states, actions, rewards, next_states = batch_data
            actions = actions.long().to(device)
            loss_dict_1 = iql_update(
                q_net_1, q_opt_1,
                v_net_1, v_opt_1,
                pi_net_1, pi_opt_1,
                (states, actions, rewards, next_states),
                gamma=GAMMA,
                tau=TAU,
                reward_scale=REWARD_SCALE
            )
            all_q_1.append(loss_dict_1["q_loss"])
            all_v_1.append(loss_dict_1["v_loss"])
            all_pi_1.append(loss_dict_1["pi_loss"])
        avg_q_1 = np.mean(all_q_1)
        avg_v_1 = np.mean(all_v_1)
        avg_pi_1 = np.mean(all_pi_1)
        epoch_q_loss_1.append(avg_q_1)
        epoch_v_loss_1.append(avg_v_1)
        epoch_pi_loss_1.append(avg_pi_1)
        
        all_q_2, all_v_2, all_pi_2 = [], [], []
        for batch_data in loader_relabel:
            batch_data = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch_data]
            states, actions, rewards, next_states = batch_data
            actions = actions.long().to(device)
            loss_dict_2 = iql_update(
                q_net_2, q_opt_2,
                v_net_2, v_opt_2,
                pi_net_2, pi_opt_2,
                (states, actions, rewards, next_states),
                gamma=GAMMA,
                tau=TAU,
                reward_scale=REWARD_SCALE
            )
            all_q_2.append(loss_dict_2["q_loss"])
            all_v_2.append(loss_dict_2["v_loss"])
            all_pi_2.append(loss_dict_2["pi_loss"])
        avg_q_2 = np.mean(all_q_2)
        avg_v_2 = np.mean(all_v_2)
        avg_pi_2 = np.mean(all_pi_2)
        epoch_q_loss_2.append(avg_q_2)
        epoch_v_loss_2.append(avg_v_2)
        epoch_pi_loss_2.append(avg_pi_2)

        if (epoch + 1) % ETA == 0:
            q1_sd = q_net_1.state_dict()
            v1_sd = v_net_1.state_dict()
            pi1_sd = pi_net_1.state_dict()
            q2_sd = q_net_2.state_dict()
            v2_sd = v_net_2.state_dict()
            pi2_sd = pi_net_2.state_dict()
            q_fused_sd = ema_fuse_params(q1_sd, q2_sd, alpha=EMA_ALPHA)
            v_fused_sd = ema_fuse_params(v1_sd, v2_sd, alpha=EMA_ALPHA)
            pi_fused_sd = ema_fuse_params(pi1_sd, pi2_sd, alpha=EMA_ALPHA)
            q_net_1.load_state_dict(q_fused_sd)
            q_net_2.load_state_dict(q_fused_sd)
            v_net_1.load_state_dict(v_fused_sd)
            v_net_2.load_state_dict(v_fused_sd)
            pi_net_1.load_state_dict(pi_fused_sd)
            pi_net_2.load_state_dict(pi_fused_sd)
            print(f"Epoch {epoch+1}: EMA fusion performed.")

        if (epoch + 1) % 100 == 0:
            save_path = f"model_epoch{epoch+1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "q_net_1": q_net_1.state_dict(),
                "v_net_1": v_net_1.state_dict(),
                "pi_net_1": pi_net_1.state_dict(),
                "q_net_2": q_net_2.state_dict(),
                "v_net_2": v_net_2.state_dict(),
                "pi_net_2": pi_net_2.state_dict()
            }, save_path)
    epochs = np.arange(1, MAX_EPOCHS + 1)


if __name__ == "__main__":
    main()
