
import os
import sys
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from models import TrajMAE, RewardEnsemble, mask_trajectory_states, extract_transitions_from_trajectory, extract_transitions_unlabeled
from utils import set_seed, get_device, setup_logging, save_metrics


class TrajectoryDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return batch


def train_step_labeled(mae, ensemble, trajectories, mae_opt, ensemble_opt, mask_ratio, device):

    mae.train()
    ensemble.train()
    mae_opt.zero_grad()
    ensemble_opt.zero_grad()

    total_reward_loss = 0.0
    total_mae_loss = 0.0
    mse = nn.MSELoss()
    valid_count = 0

    for trajectory in trajectories:

        states, actions, next_states, rewards = extract_transitions_from_trajectory(trajectory)
        if len(states) == 0:
            continue
        valid_count += 1

        s_t = torch.tensor(states, dtype=torch.float32, device=device)
        a_t = torch.tensor(actions, dtype=torch.float32, device=device)
        ns_t = torch.tensor(next_states, dtype=torch.float32, device=device)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        sa = torch.cat([s_t, a_t, ns_t], dim=1)

        reward_loss = ensemble.individual_losses(sa, r_t)
        total_reward_loss += reward_loss.item()


        mask_bool, masked_traj = mask_trajectory_states(trajectory, mask_ratio)
        embeddings = []
        for i, (val, t) in enumerate(masked_traj):
            v = torch.tensor(val, dtype=torch.float32, device=device) if not isinstance(val, torch.Tensor) else val
            embeddings.append(mae.embed_token(v, t, device))
        if len(embeddings) == 0:
            reward_loss.backward()
            continue
        token_emb = torch.cat(embeddings, dim=0).unsqueeze(1)
        mask_indices = [i for i, m in enumerate(mask_bool) if m and trajectory[i][1] == "state"]

        if len(mask_indices) == 0:
            reward_loss.backward()
            continue

        recon = mae(token_emb, mask_indices)
        gt = torch.tensor(
            [trajectory[i][0] for i in mask_indices],
            dtype=torch.float32, device=device
        )
        mae_loss = mse(recon, gt)
        total_mae_loss += mae_loss.item()

        loss = reward_loss + mae_loss
        loss.backward()

    if valid_count > 0:
        mae_opt.step()
        ensemble_opt.step()
    return (total_reward_loss / max(valid_count, 1),
            total_mae_loss / max(valid_count, 1))


def train_step_unlabeled(mae, ensemble, trajectories, mae_opt, ensemble_opt, mask_ratio, device):

    mae.train()
    ensemble.train()
    mae_opt.zero_grad()
    ensemble_opt.zero_grad()

    total_mae_loss = 0.0
    mse = nn.MSELoss()
    valid_count = 0

    for trajectory in trajectories:
        states, actions, next_states = extract_transitions_unlabeled(trajectory)
        if len(states) == 0:
            continue

        s_t = torch.tensor(states, dtype=torch.float32, device=device)
        a_t = torch.tensor(actions, dtype=torch.float32, device=device)
        ns_t = torch.tensor(next_states, dtype=torch.float32, device=device)
        sa = torch.cat([s_t, a_t, ns_t], dim=1)


        pred_rewards = ensemble(sa).squeeze(-1)


        augmented_traj = []
        reward_idx = 0
        for val, t in trajectory:
            if t == "state":
                augmented_traj.append((val, t))
            elif t == "action":
                augmented_traj.append((val, t))

                if reward_idx < len(pred_rewards):
                    augmented_traj.append((pred_rewards[reward_idx], "reward"))
                    reward_idx += 1
            elif t == "reward":

                if reward_idx <= len(pred_rewards):
                    pass
                else:
                    augmented_traj.append((val, t))


        if len(augmented_traj) < 3:

            augmented_traj = []
            r_idx = 0
            for val, t in trajectory:
                if t == "reward" and r_idx < len(pred_rewards):
                    augmented_traj.append((pred_rewards[r_idx], "reward"))
                    r_idx += 1
                else:
                    augmented_traj.append((val, t))


        state_positions = [i for i, (_, t) in enumerate(augmented_traj) if t == "state"]
        if len(state_positions) < 2:
            continue
        valid_count += 1

        n_mask = max(1, int(len(state_positions) * mask_ratio))
        chosen = set(random.sample(state_positions, min(n_mask, len(state_positions))))
        mask_indices = sorted(chosen)

        embeddings = []
        for i, (val, t) in enumerate(augmented_traj):
            if t == "state" and i in chosen:
                zero_val = torch.zeros(C.MAE_STATE_DIM, device=device)
                embeddings.append(mae.embed_token(zero_val, t, device))
            else:
                if isinstance(val, torch.Tensor):
                    embeddings.append(mae.embed_token(val, t, device))
                else:
                    v = torch.tensor(val if isinstance(val, list) else [val],
                                     dtype=torch.float32, device=device)
                    embeddings.append(mae.embed_token(v, t, device))

        if len(embeddings) == 0:
            continue

        token_emb = torch.cat(embeddings, dim=0).unsqueeze(1)
        recon = mae(token_emb, list(range(len(mask_indices))))


        recon = mae(token_emb, mask_indices)

        gt_states = []
        for mi in mask_indices:
            if mi < len(augmented_traj) and augmented_traj[mi][1] == "state":
                val = augmented_traj[mi][0]
                if isinstance(val, torch.Tensor):
                    gt_states.append(val)
                else:
                    gt_states.append(torch.tensor(val, dtype=torch.float32, device=device))

        if len(gt_states) == 0:
            continue

        gt = torch.stack(gt_states)
        mae_loss = mse(recon, gt)
        total_mae_loss += mae_loss.item()
        mae_loss.backward()

    if valid_count > 0:
        mae_opt.step()
        ensemble_opt.step()
    return total_mae_loss / max(valid_count, 1)


def train_semi_mae(driver_key, gpu_id, seed=42):
    set_seed(seed)
    device = get_device(gpu_id)

    out_dir = os.path.join(C.get_output_dir(driver_key), "semi_mae")
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(out_dir, "semi_mae")

    data_dir = os.path.join(C.get_output_dir(driver_key), "data")
    labeled_path = os.path.join(data_dir, "labeled_traj_latent.pkl")
    unlabeled_path = os.path.join(data_dir, "unlabeled_traj_latent.pkl")

    with open(labeled_path, 'rb') as f:
        labeled_data = pickle.load(f)
    with open(unlabeled_path, 'rb') as f:
        unlabeled_data = pickle.load(f)
    logger.info(f"Labeled trajectories: {len(labeled_data)}, Unlabeled: {len(unlabeled_data)}")

    labeled_loader = DataLoader(
        TrajectoryDataset(labeled_data), batch_size=C.MAE_BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn
    )
    unlabeled_loader = DataLoader(
        TrajectoryDataset(unlabeled_data), batch_size=C.MAE_BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn
    )

    mae = TrajMAE(C.MAE_EMBED_DIM, C.MAE_NUM_HEADS, C.MAE_FF_DIM,
                   C.MAE_NUM_LAYERS, C.MAE_STATE_DIM).to(device)
    ensemble = RewardEnsemble(C.ENSEMBLE_N_MODELS, C.ENSEMBLE_INPUT_DIM,
                               C.ENSEMBLE_HIDDEN_DIM).to(device)

    mae_opt = optim.Adam(mae.parameters(), lr=C.MAE_LR)
    ensemble_opt = optim.Adam(ensemble.parameters(), lr=C.MAE_LR)

    history = {"reward_loss": [], "labeled_mae": [], "unlabeled_mae": []}

    for epoch in range(1, C.MAE_EPOCHS + 1):

        rl_total, ml_total, lc = 0, 0, 0
        for batch in labeled_loader:
            rl, ml = train_step_labeled(mae, ensemble, batch, mae_opt, ensemble_opt,
                                        C.MAE_MASK_RATIO, device)
            rl_total += rl
            ml_total += ml
            lc += 1


        ul_total, uc = 0, 0
        for batch in unlabeled_loader:
            ul = train_step_unlabeled(mae, ensemble, batch, mae_opt, ensemble_opt,
                                      C.MAE_MASK_RATIO, device)
            ul_total += ul
            uc += 1

        rl_avg = rl_total / max(lc, 1)
        ml_avg = ml_total / max(lc, 1)
        ul_avg = ul_total / max(uc, 1)
        history["reward_loss"].append(rl_avg)
        history["labeled_mae"].append(ml_avg)
        history["unlabeled_mae"].append(ul_avg)

        if epoch % 50 == 0:
            logger.info(f"Epoch {epoch}/{C.MAE_EPOCHS} | R_loss={rl_avg:.4f} "
                        f"L_MAE={ml_avg:.4f} U_MAE={ul_avg:.4f}")

        if epoch % 200 == 0:
            torch.save(mae.state_dict(), os.path.join(out_dir, f"mae_epoch{epoch}.pth"))
            torch.save(ensemble.state_dict(), os.path.join(out_dir, f"ensemble_epoch{epoch}.pth"))

    torch.save(mae.state_dict(), os.path.join(out_dir, "mae_final.pth"))
    torch.save(ensemble.state_dict(), os.path.join(out_dir, "ensemble_final.pth"))

    save_metrics(history, os.path.join(out_dir, "training_history.json"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes,
                               ["reward_loss", "labeled_mae", "unlabeled_mae"],
                               ["Reward Ensemble Loss", "Labeled MAE Loss", "Unlabeled MAE Loss"]):
        ax.plot(history[key])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "semi_mae_loss.png"), dpi=150)
    plt.close()

    logger.info(f"Semi-MAE training complete for {driver_key}")
    return os.path.join(out_dir, "ensemble_final.pth")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True, choices=list({**C.DRIVERS, **C.DRIVERS_EXPANDED}.keys()))
    parser.add_argument("--gpu", type=int, default=C.GPU_PRIMARY)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_semi_mae(args.driver, args.gpu, args.seed)
