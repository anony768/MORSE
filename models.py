"""MORSE Model Definitions - All neural network architectures."""
import torch
import torch.nn as nn
import random

# ================================================================
#  1. VAE for Urban State Compression
# ================================================================
class VAE(nn.Module):
    """Variational Autoencoder with KL divergence regularization.
    L_Compress = ||S - S_hat||^2 + beta * KL(q(s|S) || p(s))
    """
    def __init__(self, input_dim=125, hidden_dim=64, latent_dim=10):
        super().__init__()
        self.fc_hidden = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.fc_hidden(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode_deterministic(self, x):
        """For inference: use mu as the latent representation."""
        mu, _ = self.encode(x)
        return mu

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=0.1):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ================================================================
#  2. Reward Ensemble Network
# ================================================================
class RewardModel(nn.Module):
    """Single reward predictor: (s, a, s') -> r"""
    def __init__(self, input_dim=21, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)


class RewardEnsemble(nn.Module):
    """Ensemble of m reward models with trimmed mean aggregation.
    Remove highest and lowest, average the rest.
    """
    def __init__(self, n_models=7, input_dim=21, hidden_dim=64):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            RewardModel(input_dim, hidden_dim) for _ in range(n_models)
        ])

    def forward(self, x):
        """Trimmed mean: remove max and min, average the rest."""
        outputs = torch.stack([m(x) for m in self.models], dim=0)  # [m, B, 1]
        if self.n_models >= 5:
            sorted_out, _ = torch.sort(outputs, dim=0)
            return sorted_out[1:-1].mean(dim=0)  # [B, 1]
        return outputs.mean(dim=0)

    def individual_losses(self, x, target):
        """Compute MSE loss for each model independently."""
        total = 0.0
        for m in self.models:
            pred = m(x).squeeze(-1)
            total = total + nn.functional.mse_loss(pred, target)
        return total


# ================================================================
#  3. Trajectory Masked Autoencoder - TrajMAE
# ================================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, src):
        attn_out, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.drop1(attn_out))
        src = self.norm2(src + self.drop2(self.ff(src)))
        return src


class TrajMAE(nn.Module):
    """Trajectory Masked AutoEncoder.
    Encodes trajectory tokens (state/action/reward) with a Transformer,
    reconstructs masked state tokens.
    """
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=64, num_layers=2, state_dim=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim

        self.type_embed = nn.Embedding(3, embed_dim)  # 0=state, 1=action, 2=reward
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Linear(1, embed_dim)
        self.reward_embed = nn.Linear(1, embed_dim)

        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, state_dim)
        )

    def embed_token(self, value, token_type, device):
        """Embed a single token. value must be a tensor for gradient flow."""
        type_idx = {"state": 0, "action": 1, "reward": 2}[token_type]
        type_emb = self.type_embed(torch.tensor([type_idx], device=device))

        if token_type == "state":
            val_emb = self.state_embed(value.unsqueeze(0) if value.dim() == 1 else value)
        elif token_type == "action":
            v = value if isinstance(value, torch.Tensor) else torch.tensor([value], dtype=torch.float32, device=device)
            val_emb = self.action_embed(v.view(1, 1))
        else:  # reward
            v = value if isinstance(value, torch.Tensor) else torch.tensor([value], dtype=torch.float32, device=device)
            val_emb = self.reward_embed(v.view(1, 1))
        return type_emb + val_emb  # [1, embed_dim]

    def forward(self, token_embeddings, mask_indices):
        """
        token_embeddings: [seq_len, 1, embed_dim]
        mask_indices: list of int positions where states were masked
        Returns: reconstructed states [n_masked, state_dim]
        """
        x = token_embeddings
        for layer in self.encoder:
            x = layer(x)

        if len(mask_indices) == 0:
            return torch.zeros(0, self.state_dim, device=x.device)

        masked_features = x[mask_indices, 0, :]  # [n_masked, embed_dim]
        return self.decoder(masked_features)  # [n_masked, state_dim]


# ================================================================
#  4. IQL Networks
# ================================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, state):
        return self.net(state)


class PolicyNetwork(nn.Module):
    """Gaussian policy for continuous actions."""
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        feat = self.net(state)
        mean = self.mean_head(feat)
        log_std = torch.clamp(self.log_std_head(feat), -20, 2)
        return mean, log_std

    def sample_action(self, state):
        mean, log_std = self.forward(state)
        return mean + torch.exp(log_std) * torch.randn_like(mean)

    def log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        return -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + 1.8378770664093453)


# ================================================================
#  Helper functions
# ================================================================
def mask_trajectory_states(trajectory, mask_ratio=0.3):
    """Mask a proportion of state tokens in a trajectory.
    Returns (mask_positions_bool, masked_trajectory).
    """
    state_positions = [i for i, (_, t) in enumerate(trajectory) if t == "state"]
    n_to_mask = max(1, int(len(state_positions) * mask_ratio))
    chosen = set(random.sample(state_positions, min(n_to_mask, len(state_positions))))

    mask_bool = [i in chosen for i in range(len(trajectory))]
    masked_traj = []
    for i, (val, t) in enumerate(trajectory):
        if t == "state" and i in chosen:
            masked_traj.append(([0.0] * len(val), t))
        else:
            masked_traj.append((val, t))
    return mask_bool, masked_traj


def extract_transitions_from_trajectory(trajectory):
    """Extract (s, a, s', r) tuples from a [value, type] trajectory."""
    states, actions, next_states, rewards = [], [], [], []
    i = 0
    while i < len(trajectory) - 2:
        if (trajectory[i][1] == "state" and
            trajectory[i+1][1] == "action" and
            trajectory[i+2][1] == "reward"):
            if i + 3 < len(trajectory) and trajectory[i+3][1] == "state":
                states.append(trajectory[i][0])
                actions.append([trajectory[i+1][0]])
                rewards.append(trajectory[i+2][0])
                next_states.append(trajectory[i+3][0])
            i += 3
        else:
            i += 1
    return states, actions, next_states, rewards


def extract_transitions_unlabeled(trajectory):
    """Extract (s, a, s') from unlabeled trajectory (no rewards)."""
    states, actions, next_states = [], [], []
    i = 0
    while i < len(trajectory) - 1:
        if trajectory[i][1] == "state" and trajectory[i+1][1] == "action":
            if i + 2 < len(trajectory) and trajectory[i+2][1] == "state":
                states.append(trajectory[i][0])
                actions.append([trajectory[i+1][0]])
                next_states.append(trajectory[i+2][0])
                i += 2
            elif (i + 2 < len(trajectory) and trajectory[i+2][1] == "reward"
                  and i + 3 < len(trajectory) and trajectory[i+3][1] == "state"):
                states.append(trajectory[i][0])
                actions.append([trajectory[i+1][0]])
                next_states.append(trajectory[i+3][0])
                i += 3
            else:
                i += 1
        else:
            i += 1
    return states, actions, next_states
