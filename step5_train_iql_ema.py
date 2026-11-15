"""Step 5: IQL with Adaptive Knowledge Fusion.
Two policies trained on labeled/relabeled data, fused via EMA every eta epochs.
"""
import os
import sys
import copy
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from models import QNetwork, ValueNetwork, PolicyNetwork
from utils import set_seed, get_device, setup_logging, save_metrics


class LatentTransitionDataset(Dataset):
    """Dataset for 22-dim latent transitions: [s(10), a(1), r(1), s'(10)]."""
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        s = torch.FloatTensor(row[:10])
        a = torch.FloatTensor(row[10:11])
        r = torch.FloatTensor(row[11:12])
        s_next = torch.FloatTensor(row[12:22])
        return s, a, r, s_next


def iql_update(q_net, q_opt, v_net, v_opt, pi_net, pi_opt,
               batch, gamma=0.99, tau=0.7, beta=3.0):
    """IQL update (Kostrikov et al., 2021) with correct expectile regression."""
    states, actions, rewards, next_states = batch

    # 1) Q-network: L_Q = E[(Q(s,a) - (r + gamma*V(s')))^2]
    with torch.no_grad():
        v_next = v_net(next_states).squeeze(-1)
        td_target = rewards.squeeze(-1) + gamma * v_next

    q_pred = q_net(states, actions).squeeze(-1)
    q_loss = ((q_pred - td_target) ** 2).mean()

    q_opt.zero_grad()
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    q_opt.step()

    # 2) V-network: expectile regression on Q(s, a_data)
    with torch.no_grad():
        q_data = q_net(states, actions).squeeze(-1)
    v_pred = v_net(states).squeeze(-1)
    diff = q_data - v_pred
    # L_V = E[|tau - 1(diff < 0)| * diff^2]
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    v_loss = (weight * (diff ** 2)).mean()

    v_opt.zero_grad()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
    v_opt.step()

    # 3) Policy: advantage-weighted regression
    with torch.no_grad():
        advantage = q_net(states, actions).squeeze(-1) - v_net(states).squeeze(-1)
        exp_adv = torch.exp(beta * advantage)
        exp_adv = torch.clamp(exp_adv, max=100.0)

    log_prob = pi_net.log_prob(states, actions).squeeze(-1)
    log_prob = torch.clamp(log_prob, min=-100.0)  # Prevent explosion after EMA fusion
    pi_loss = -(exp_adv * log_prob).mean()

    pi_opt.zero_grad()
    pi_loss.backward()
    torch.nn.utils.clip_grad_norm_(pi_net.parameters(), 1.0)
    pi_opt.step()

    return {"q_loss": q_loss.item(), "v_loss": v_loss.item(), "pi_loss": pi_loss.item()}


def ema_fuse(sd_labeled, sd_relabeled, alpha):
    """EMA fusion: omega_t = alpha * omega_{t-1} + (1-alpha) * sigma_t."""
    fused = {}
    for key in sd_labeled:
        fused[key] = alpha * sd_labeled[key] + (1 - alpha) * sd_relabeled[key]
    return fused


def train_iql_ema(task, gpu_id, seed=42):
    set_seed(seed)
    device = get_device(gpu_id)

    out_dir = C.get_output_dir(task, seed)
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(out_dir, "iql_ema")

    data_dir = os.path.join(C.get_output_dir(task), "data")
    labeled_path = os.path.join(data_dir, "labeled_buffer_latent.pkl")
    relabeled_path = os.path.join(data_dir, "relabeled_buffer_latent.pkl")

    with open(labeled_path, 'rb') as f:
        labeled_data = pickle.load(f)
    with open(relabeled_path, 'rb') as f:
        relabeled_data = pickle.load(f)

    logger.info(f"Labeled: {len(labeled_data)} transitions, Relabeled: {len(relabeled_data)} transitions")
    logger.info(f"alpha={C.EMA_ALPHA}, eta={C.EMA_ETA}, tau={C.IQL_TAU}, beta={C.IQL_BETA}")

    loader_labeled = DataLoader(
        LatentTransitionDataset(labeled_data), batch_size=C.IQL_BATCH_SIZE, shuffle=True
    )
    loader_relabeled = DataLoader(
        LatentTransitionDataset(relabeled_data), batch_size=C.IQL_BATCH_SIZE, shuffle=True
    )

    SD, AD, HS = C.IQL_STATE_DIM, C.IQL_ACTION_DIM, C.IQL_HIDDEN_SIZE

    # Policy 1: pi_labeled (trained on D_L)
    q1 = QNetwork(SD, AD, HS).to(device)
    v1 = ValueNetwork(SD, HS).to(device)
    pi1 = PolicyNetwork(SD, AD, HS).to(device)
    q1_opt = optim.Adam(q1.parameters(), lr=C.IQL_LR)
    v1_opt = optim.Adam(v1.parameters(), lr=C.IQL_LR)
    pi1_opt = optim.Adam(pi1.parameters(), lr=C.IQL_LR)

    # Policy 2: pi_relabeled (trained on D_relabeled)
    q2 = QNetwork(SD, AD, HS).to(device)
    v2 = ValueNetwork(SD, HS).to(device)
    pi2 = PolicyNetwork(SD, AD, HS).to(device)
    q2.load_state_dict(copy.deepcopy(q1.state_dict()))
    v2.load_state_dict(copy.deepcopy(v1.state_dict()))
    pi2.load_state_dict(copy.deepcopy(pi1.state_dict()))
    q2_opt = optim.Adam(q2.parameters(), lr=C.IQL_LR)
    v2_opt = optim.Adam(v2.parameters(), lr=C.IQL_LR)
    pi2_opt = optim.Adam(pi2.parameters(), lr=C.IQL_LR)

    history = {k: [] for k in ["q1", "v1", "pi1", "q2", "v2", "pi2"]}

    for epoch in range(1, C.IQL_EPOCHS + 1):
        # Train pi_labeled on D_L
        losses1 = []
        for batch in loader_labeled:
            batch = [x.to(device) for x in batch]
            l = iql_update(q1, q1_opt, v1, v1_opt, pi1, pi1_opt,
                           batch, C.IQL_GAMMA, C.IQL_TAU, C.IQL_BETA)
            losses1.append(l)

        # Train pi_relabeled on D_relabeled
        losses2 = []
        for batch in loader_relabeled:
            batch = [x.to(device) for x in batch]
            l = iql_update(q2, q2_opt, v2, v2_opt, pi2, pi2_opt,
                           batch, C.IQL_GAMMA, C.IQL_TAU, C.IQL_BETA)
            losses2.append(l)

        avg1 = {k: np.mean([l[k] for l in losses1]) for k in losses1[0]} if losses1 else {"q_loss": 0, "v_loss": 0, "pi_loss": 0}
        avg2 = {k: np.mean([l[k] for l in losses2]) for k in losses2[0]} if losses2 else {"q_loss": 0, "v_loss": 0, "pi_loss": 0}
        history["q1"].append(avg1["q_loss"])
        history["v1"].append(avg1["v_loss"])
        history["pi1"].append(avg1["pi_loss"])
        history["q2"].append(avg2["q_loss"])
        history["v2"].append(avg2["v_loss"])
        history["pi2"].append(avg2["pi_loss"])

        # Adaptive Knowledge Fusion: EMA every eta epochs
        if epoch % C.EMA_ETA == 0:
            q_fused = ema_fuse(q1.state_dict(), q2.state_dict(), C.EMA_ALPHA)
            v_fused = ema_fuse(v1.state_dict(), v2.state_dict(), C.EMA_ALPHA)
            pi_fused = ema_fuse(pi1.state_dict(), pi2.state_dict(), C.EMA_ALPHA)

            q1.load_state_dict(q_fused)
            v1.load_state_dict(v_fused)
            pi1.load_state_dict(pi_fused)
            # Note: only pi_labeled is updated via EMA. pi_relabeled continues independently.

        if epoch % 50 == 0:
            logger.info(f"Epoch {epoch}/{C.IQL_EPOCHS} | "
                        f"L: Q={avg1['q_loss']:.4f} V={avg1['v_loss']:.4f} Pi={avg1['pi_loss']:.4f} | "
                        f"R: Q={avg2['q_loss']:.4f} V={avg2['v_loss']:.4f} Pi={avg2['pi_loss']:.4f}"
                        f"{' [EMA]' if epoch % C.EMA_ETA == 0 else ''}")

        if epoch % 200 == 0 or epoch == C.IQL_EPOCHS:
            torch.save({
                'epoch': epoch,
                'q_net_1': q1.state_dict(), 'v_net_1': v1.state_dict(), 'pi_net_1': pi1.state_dict(),
                'q_net_2': q2.state_dict(), 'v_net_2': v2.state_dict(), 'pi_net_2': pi2.state_dict(),
            }, os.path.join(out_dir, f"iql_epoch{epoch}.pth"))

    save_metrics(history, os.path.join(out_dir, "iql_history.json"))

    logger.info(f"IQL+EMA training complete for {task} seed={seed}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_iql_ema(args.task, args.gpu, args.seed)
