import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrajectoryDataset(Dataset):
    def __init__(self, data_list, is_labeled=True):
        super(TrajectoryDataset, self).__init__()
        self.data_list = data_list
        self.is_labeled = is_labeled

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        trajectory = self.data_list[idx]
        return trajectory

def collate_fn(batch):
    return batch

def load_datasets(labeled_path, unlabeled_path):
    with open(labeled_path, "rb") as f:
        labeled_dataset = pickle.load(f)
    with open(unlabeled_path, "rb") as f:
        unlabeled_dataset = pickle.load(f)
    return labeled_dataset, unlabeled_dataset

class RewardModel(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64):
        super(RewardModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class RewardModelEnsemble(nn.Module):
    def __init__(self, n_models=3, input_dim=21, hidden_dim=64):
        super(RewardModelEnsemble, self).__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([RewardModel(input_dim, hidden_dim) for _ in range(n_models)])

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        return torch.mean(outputs, dim=0)

    def individual_forward(self, x):
        return [m(x) for m in self.models]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff_output = self.linear2(self.dropout(nn.ReLU()(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=64, num_layers=2):
        super(SimpleTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MaskedAutoEncoder(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=64, num_layers=2, state_dim=10):
        super(MaskedAutoEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.type_embed = nn.Embedding(num_embeddings=3, embedding_dim=embed_dim)
        self.value_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Linear(1, embed_dim)
        self.reward_embed = nn.Linear(1, embed_dim)
        self.encoder = SimpleTransformerEncoder(embed_dim, num_heads, ff_dim, num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, state_dim)
        )

    def forward(self, tokens, mask_positions):
        enc_out = self.encoder(tokens)
        masked_token_features = enc_out[mask_positions]
        reconstructed_state = self.decoder(masked_token_features)
        return reconstructed_state

    def embed_trajectory(self, trajectory):
        embeddings = []
        for item in trajectory:
            val, t = item
            if t == "state":
                type_emb = self.type_embed(torch.tensor([0], dtype=torch.long, device=device))
                value_emb = self.value_embed(torch.tensor(val, dtype=torch.float32, device=device).unsqueeze(0))
                emb = type_emb + value_emb
            elif t == "action":
                type_emb = self.type_embed(torch.tensor([1], dtype=torch.long, device=device))
                value_emb = self.action_embed(torch.tensor([val], dtype=torch.float32, device=device).unsqueeze(0))
                emb = type_emb + value_emb
            elif t == "reward":
                type_emb = self.type_embed(torch.tensor([2], dtype=torch.long, device=device))
                value_emb = self.reward_embed(torch.tensor([val], dtype=torch.float32, device=device).unsqueeze(0))
                emb = type_emb + value_emb
            else:
                raise ValueError("Unknown type in trajectory!")
            embeddings.append(emb)
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.unsqueeze(1)
        return embeddings

def insert_predicted_rewards(trajectory, ensemble_model, device):
    new_traj = []
    i = 0
    while i < len(trajectory):
        token = trajectory[i]
        new_traj.append(token)
        if token[1] == "action" and i > 0 and trajectory[i - 1][1] == "state":
            if (i + 1) < len(trajectory) and trajectory[i + 1][1] == "reward":
                if trajectory[i + 1][0] == -1:
                    if (i + 2) < len(trajectory) and trajectory[i + 2][1] == "state":
                        s = trajectory[i - 1][0]
                        a = token[0]
                        s_next = trajectory[i + 2][0]
                        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                        a_t = torch.tensor([a], dtype=torch.float32, device=device).unsqueeze(0)
                        s_next_t = torch.tensor(s_next, dtype=torch.float32, device=device).unsqueeze(0)
                        sa_concat = torch.cat([s_t, a_t, s_next_t], dim=1)
                        with torch.no_grad():
                            pred_r = ensemble_model(sa_concat)
                        pred_r_val = pred_r.item()
                        new_traj.append((pred_r_val, "reward"))
                        i += 1
            else:
                if (i + 1) < len(trajectory) and trajectory[i + 1][1] == "state":
                    s = trajectory[i - 1][0]
                    a = token[0]
                    s_next = trajectory[i + 1][0]
                    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    a_t = torch.tensor([a], dtype=torch.float32, device=device).unsqueeze(0)
                    s_next_t = torch.tensor(s_next, dtype=torch.float32, device=device).unsqueeze(0)
                    sa_concat = torch.cat([s_t, a_t, s_next_t], dim=1)
                    with torch.no_grad():
                        pred_r = ensemble_model(sa_concat)
                    pred_r_val = pred_r.item()
                    new_traj.append((pred_r_val, "reward"))
        i += 1
    return new_traj

def mask_states_in_trajectory(trajectory, mask_ratio=0.2):
    state_positions = []
    for i, item in enumerate(trajectory):
        val, t = item
        if t == "state":
            state_positions.append(i)
    n_states = len(state_positions)
    n_to_mask = int(n_states * mask_ratio)
    mask_positions_chosen = random.sample(state_positions, n_to_mask) if n_to_mask > 0 else []
    seq_len = len(trajectory)
    mask_positions_bool = [False] * seq_len
    for pos in mask_positions_chosen:
        mask_positions_bool[pos] = True
    masked_trajectory = []
    for i, (val, t) in enumerate(trajectory):
        if t == "state" and mask_positions_bool[i]:
            masked_trajectory.append(([0.0] * len(val), t))
        else:
            masked_trajectory.append((val, t))
    return mask_positions_bool, masked_trajectory

def make_transition_batch(trajectory):
    states = []
    actions = []
    next_states = []
    rewards = []
    i = 0
    while i < (len(trajectory) - 2):
        if trajectory[i][1] == "state" and trajectory[i + 1][1] == "action" and trajectory[i + 2][1] == "reward":
            s = trajectory[i][0]
            a = [trajectory[i + 1][0]]
            r = trajectory[i + 2][0]
            if (i + 3) < len(trajectory) and trajectory[i + 3][1] == "state":
                s_next = trajectory[i + 3][0]
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(s_next)
            i += 3
        else:
            i += 1
    return states, actions, next_states, rewards

def train_step_labeled(mae_model, ensemble_model, trajectories, mae_optimizer, ensemble_optimizer, mask_ratio=0.2, device="cpu"):
    mae_model.train()
    ensemble_model.train()
    mae_optimizer.zero_grad()
    ensemble_optimizer.zero_grad()
    total_reward_loss = 0.0
    total_mae_loss = 0.0
    mse_loss_fn = nn.MSELoss()
    for trajectory in trajectories:
        states, actions, next_states, rewards = make_transition_batch(trajectory)
        if len(states) == 0:
            continue
        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        sa_concat = torch.cat([states_t, actions_t, next_states_t], dim=1)
        pred_r = ensemble_model(sa_concat).squeeze(-1)
        reward_loss = mse_loss_fn(pred_r, rewards_t)
        total_reward_loss += reward_loss.item()
        mask_positions_bool, masked_trajectory = mask_states_in_trajectory(trajectory, mask_ratio)
        embeddings = mae_model.embed_trajectory(masked_trajectory).to(device)
        mask_positions_idx = torch.nonzero(torch.tensor(mask_positions_bool, device=device), as_tuple=True)[0]
        reconstructed_state = mae_model(embeddings, (mask_positions_idx, 0))
        gt_states = []
        for i in range(len(trajectory)):
            if mask_positions_bool[i] and trajectory[i][1] == "state":
                gt_states.append(trajectory[i][0])
        if len(gt_states) == 0:
            loss = reward_loss
            loss.backward()
            continue
        gt_states_t = torch.tensor(gt_states, dtype=torch.float32, device=device)
        mae_loss = mse_loss_fn(reconstructed_state, gt_states_t)
        total_mae_loss += mae_loss.item()
        loss = reward_loss + mae_loss
        loss.backward()
    mae_optimizer.step()
    ensemble_optimizer.step()
    return total_reward_loss / len(trajectories), total_mae_loss / len(trajectories)

def train_step_unlabeled(mae_model, ensemble_model, trajectories, mae_optimizer, mask_ratio=0.2, device="cpu"):
    mae_model.train()
    ensemble_model.eval()
    mae_optimizer.zero_grad()
    total_mae_loss = 0.0
    mse_loss_fn = nn.MSELoss()
    for trajectory in trajectories:
        trajectory_with_reward = insert_predicted_rewards(trajectory, ensemble_model, device)
        mask_positions_bool, masked_trajectory = mask_states_in_trajectory(trajectory_with_reward, mask_ratio)
        embeddings = mae_model.embed_trajectory(masked_trajectory).to(device)
        mask_positions_idx = torch.nonzero(torch.tensor(mask_positions_bool, device=device), as_tuple=True)[0]
        reconstructed_state = mae_model(embeddings, (mask_positions_idx, 0))
        gt_states = []
        for i in range(len(trajectory_with_reward)):
            if mask_positions_bool[i] and trajectory_with_reward[i][1] == "state":
                gt_states.append(trajectory_with_reward[i][0])
        if len(gt_states) == 0:
            continue
        gt_states_t = torch.tensor(gt_states, dtype=torch.float32, device=device)
        mae_loss = mse_loss_fn(reconstructed_state, gt_states_t)
        total_mae_loss += mae_loss.item()
        mae_loss.backward()
    mae_optimizer.step()
    return total_mae_loss / len(trajectories)

def main():
    device_local = "cuda:1" if torch.cuda.is_available() else "cpu"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    labeled_path = "labeled_dataset.pkl"
    unlabeled_path = "unlabeled_dataset.pkl"
    batch_size = 32
    num_epochs = 1000
    mask_ratio = 0.3
    lr = 1e-3
    info_str = (
        f"labeled dataset: {labeled_path}\n"
        f"unlabeled dataset: {unlabeled_path}\n"
        f"batch_size={batch_size}, num_epochs={num_epochs}\n"
        f"mask_ratio={mask_ratio}, learning_rate={lr}\n"
        f"device={device_local}\n"
    )
    labeled_dataset_list, unlabeled_dataset_list = load_datasets(labeled_path, unlabeled_path)
    labeled_ds = TrajectoryDataset(labeled_dataset_list, is_labeled=True)
    unlabeled_ds = TrajectoryDataset(unlabeled_dataset_list, is_labeled=False)
    labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    mae_model = MaskedAutoEncoder(embed_dim=32, num_heads=4, ff_dim=64, num_layers=2, state_dim=10).to(device_local)
    ensemble_model = RewardModelEnsemble(n_models=3, input_dim=21, hidden_dim=64).to(device_local)
    mae_optimizer = optim.Adam(mae_model.parameters(), lr=lr)
    ensemble_optimizer = optim.Adam(ensemble_model.parameters(), lr=lr)
    reward_loss_over_epochs = []
    mae_loss_over_epochs = []
    unlabeled_mae_loss_over_epochs = []
    model_save_dir = "checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)
    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch+1}/{num_epochs} ===")
        labeled_reward_loss_epoch = 0.0
        labeled_mae_loss_epoch = 0.0
        labeled_batches_count = 0
        for _, batch_trajectories in enumerate(labeled_loader):
            reward_loss, mae_loss = train_step_labeled(
                mae_model,
                ensemble_model,
                batch_trajectories,
                mae_optimizer,
                ensemble_optimizer,
                mask_ratio=mask_ratio,
                device=device_local
            )
            labeled_reward_loss_epoch += reward_loss
            labeled_mae_loss_epoch += mae_loss
            labeled_batches_count += 1
        if labeled_batches_count > 0:
            labeled_reward_loss_epoch /= labeled_batches_count
            labeled_mae_loss_epoch /= labeled_batches_count
        print(f"[Labeled] RewardLoss={labeled_reward_loss_epoch:.4f}, MAELoss={labeled_mae_loss_epoch:.4f}")
        unlabeled_mae_loss_epoch = 0.0
        unlabeled_batches_count = 0
        for _, batch_trajectories in enumerate(unlabeled_loader):
            mae_loss = train_step_unlabeled(
                mae_model,
                ensemble_model,
                batch_trajectories,
                mae_optimizer,
                mask_ratio=mask_ratio,
                device=device_local
            )
            unlabeled_mae_loss_epoch += mae_loss
            unlabeled_batches_count += 1
        if unlabeled_batches_count > 0:
            unlabeled_mae_loss_epoch /= unlabeled_batches_count
        print(f"[Unlabeled] MAELoss={unlabeled_mae_loss_epoch:.4f}")
        reward_loss_over_epochs.append(labeled_reward_loss_epoch)
        mae_loss_over_epochs.append(labeled_mae_loss_epoch)
        unlabeled_mae_loss_over_epochs.append(unlabeled_mae_loss_epoch)
        if (epoch + 1) % 200 == 0:
            mae_temp_path = os.path.join(model_save_dir, f"mae_model_epoch{epoch+1}.pth")
            ensemble_temp_path = os.path.join(model_save_dir, f"ensemble_model_epoch{epoch+1}.pth")
            torch.save(mae_model.state_dict(), mae_temp_path)
            torch.save(ensemble_model.state_dict(), ensemble_temp_path)
            print(f"Checkpoint saved at epoch {epoch+1}.")
    final_mae_path = os.path.join(model_save_dir, "mae_model_final.pth")
    final_ensemble_path = os.path.join(model_save_dir, "ensemble_model_final.pth")
    torch.save(mae_model.state_dict(), final_mae_path)
    torch.save(ensemble_model.state_dict(), final_ensemble_path)

    info_file = os.path.join(model_save_dir, "training_info.txt")
    with open(info_file, "w") as f:
        f.write(info_str)
    epochs_range = range(1, num_epochs + 1)


if __name__ == "__main__":
    main()
