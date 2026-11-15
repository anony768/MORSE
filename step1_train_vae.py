"""Step 1: Train VAE for Urban State Compression."""
import os
import sys
import pickle
import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from models import VAE
from utils import set_seed, get_device, setup_logging


class StateDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)
        # AE buffer rows have no reward column: [S(125), A(1), S'(125)] = 251 dims
        data = np.array(raw, dtype=np.float32)
        S = data[:, :125]
        S_prime = data[:, 126:251]
        all_states = np.concatenate([S, S_prime], axis=0)
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(all_states).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        return x, x


def train_vae(gpu_id=0, seed=42):
    set_seed(seed)
    device = get_device(gpu_id)

    out_dir = os.path.join(C.OUTPUT_DIR, "vae")
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(out_dir, "vae")
    logger.info(f"Training VAE on {device}, beta={C.VAE_BETA}, lr={C.VAE_LR}, epochs={C.VAE_EPOCHS}")

    dataset = StateDataset(C.AE_DATA)
    loader = DataLoader(dataset, batch_size=C.VAE_BATCH_SIZE, shuffle=True)
    logger.info(f"Dataset size: {len(dataset)}")

    # Save scaler for later use
    joblib.dump(dataset.scaler, os.path.join(out_dir, "state_scaler.pkl"))

    model = VAE(C.VAE_INPUT_DIM, C.VAE_HIDDEN_DIM, C.VAE_LATENT_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.VAE_LR)

    best_loss = float('inf')
    patience_counter = 0
    losses = []

    for epoch in range(1, C.VAE_EPOCHS + 1):
        model.train()
        total_loss, total_recon, total_kl, n = 0, 0, 0, 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            recon, mu, logvar = model(batch_x)
            loss, recon_loss, kl_loss = VAE.loss_function(recon, batch_x, mu, logvar, beta=C.VAE_BETA)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total_recon += recon_loss.item() * bs
            total_kl += kl_loss.item() * bs
            n += bs

        avg_loss = total_loss / n
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "vae_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= C.VAE_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}/{C.VAE_EPOCHS} | Loss={avg_loss:.6f} "
                        f"Recon={total_recon/n:.6f} KL={total_kl/n:.6f}")

        if epoch % 500 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"vae_epoch{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(out_dir, "vae_final.pth"))

    logger.info(f"VAE training complete. Best loss={best_loss:.6f}")
    return os.path.join(out_dir, "vae_best.pth")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_vae(args.gpu, args.seed)
