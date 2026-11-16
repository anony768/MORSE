import os
import sys
import pickle
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from models import (TrajMAE, RewardEnsemble, sample_masked_states,
                    extract_transitions_from_trajectory, extract_transitions_unlabeled)
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


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def reconstruction_loss(mae, trajectory, mask_ratio, device):
    masked_positions = sample_masked_states(trajectory, mask_ratio)
    token_emb, positions, targets = mae.embed_trajectory(trajectory, masked_positions, device)
    recon = mae(token_emb, positions)
    return sum(((targets[t] - recon[t]) ** 2).sum(dim=-1).mean() for t in targets)


def insert_predicted_rewards(ensemble, trajectory, device):
    states, actions, next_states = extract_transitions_unlabeled(trajectory)
    if len(states) == 0:
        return None
    sa = torch.cat([to_tensor(states, device), to_tensor(actions, device),
                    to_tensor(next_states, device)], dim=1)
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
    return augmented_traj


def train_step(mae, ensemble, labeled_batch, unlabeled_batch, mae_opt, ensemble_opt,
               mask_ratio, device):
    mae.train()
    ensemble.train()
    mae_opt.zero_grad()
    ensemble_opt.zero_grad()

    reward_losses, labeled_recon, unlabeled_recon = [], [], []

    for trajectory in labeled_batch:
        states, actions, next_states, rewards = extract_transitions_from_trajectory(trajectory)
        if len(states) == 0:
            continue
        sa = torch.cat([to_tensor(states, device), to_tensor(actions, device),
                        to_tensor(next_states, device)], dim=1)
        reward_losses.append(ensemble.individual_losses(sa, to_tensor(rewards, device)))
        labeled_recon.append(reconstruction_loss(mae, trajectory, mask_ratio, device))

    for trajectory in unlabeled_batch:
        augmented_traj = insert_predicted_rewards(ensemble, trajectory, device)
        if augmented_traj is None:
            continue
        unlabeled_recon.append(reconstruction_loss(mae, augmented_traj, mask_ratio, device))

    terms = [torch.stack(losses).mean()
             for losses in (reward_losses, labeled_recon, unlabeled_recon) if losses]
    if terms:
        total_loss = sum(terms)
        total_loss.backward()
        mae_opt.step()
        ensemble_opt.step()

    def avg(losses):
        return sum(l.item() for l in losses) / len(losses) if losses else 0.0
    return avg(reward_losses), avg(labeled_recon), avg(unlabeled_recon)


def train_semi_mae(task, gpu_id, seed=42):
    set_seed(seed)
    device = get_device(gpu_id)

    out_dir = os.path.join(C.get_output_dir(task), "semi_mae")
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(out_dir, "semi_mae")

    data_dir = os.path.join(C.get_output_dir(task), "data")
    labeled_path = os.path.join(data_dir, "labeled_traj_latent.pkl")
    unlabeled_path = os.path.join(data_dir, "unlabeled_traj_latent.pkl")

    with open(labeled_path, 'rb') as f:
        labeled_data = pickle.load(f)
    with open(unlabeled_path, 'rb') as f:
        unlabeled_data = pickle.load(f)
    logger.info(f"Labeled trajectories: {len(labeled_data)}, Unlabeled: {len(unlabeled_data)}")
    if len(labeled_data) == 0:
        raise ValueError(f"No labeled trajectories for {task}")

    labeled_loader = DataLoader(
        TrajectoryDataset(labeled_data), batch_size=C.MAE_BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn
    )
    unlabeled_loader = None
    if len(unlabeled_data) > 0:
        unlabeled_loader = DataLoader(
            TrajectoryDataset(unlabeled_data), batch_size=C.MAE_BATCH_SIZE,
            shuffle=True, collate_fn=collate_fn
        )

    mae = TrajMAE(C.MAE_EMBED_DIM, C.MAE_NUM_HEADS, C.MAE_FF_DIM,
                   C.MAE_NUM_LAYERS, C.MAE_STATE_DIM, C.MAE_MAX_TIMESTEPS).to(device)
    ensemble = RewardEnsemble(C.ENSEMBLE_N_MODELS, C.ENSEMBLE_INPUT_DIM,
                               C.ENSEMBLE_HIDDEN_DIM).to(device)

    mae_opt = optim.Adam(mae.parameters(), lr=C.MAE_LR)
    ensemble_opt = optim.Adam(ensemble.parameters(), lr=C.MAE_LR)

    labeled_iter = cycle(labeled_loader)
    steps_per_epoch = len(unlabeled_loader) if unlabeled_loader is not None else len(labeled_loader)
    logger.info(f"Steps per epoch: {steps_per_epoch}, mask ratio: {C.MAE_MASK_RATIO}")

    history = {"reward_loss": [], "labeled_mae": [], "unlabeled_mae": []}

    for epoch in range(1, C.MAE_EPOCHS + 1):
        rl_total, ml_total, ul_total, n_steps = 0.0, 0.0, 0.0, 0
        unlabeled_batches = unlabeled_loader if unlabeled_loader is not None else [[]] * steps_per_epoch
        for unlabeled_batch in unlabeled_batches:
            labeled_batch = next(labeled_iter)
            rl, ml, ul = train_step(mae, ensemble, labeled_batch, unlabeled_batch,
                                    mae_opt, ensemble_opt, C.MAE_MASK_RATIO, device)
            rl_total += rl
            ml_total += ml
            ul_total += ul
            n_steps += 1

        rl_avg = rl_total / max(n_steps, 1)
        ml_avg = ml_total / max(n_steps, 1)
        ul_avg = ul_total / max(n_steps, 1)
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

    logger.info(f"Semi-MAE training complete for {task}")
    return os.path.join(out_dir, "ensemble_final.pth")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_semi_mae(args.task, args.gpu, args.seed)
