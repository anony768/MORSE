"""Step 4: Relabel unlabeled transitions using trained reward ensemble."""
import os
import sys
import pickle
import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from models import VAE, RewardEnsemble
from utils import get_device, setup_logging


def relabel(task, gpu_id=0):
    device = get_device(gpu_id)
    out_dir = os.path.join(C.get_output_dir(task))
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(out_dir, "relabel")

    # Load VAE encoder
    vae_dir = os.path.join(C.OUTPUT_DIR, "vae")
    vae = VAE(C.VAE_INPUT_DIM, C.VAE_HIDDEN_DIM, C.VAE_LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(os.path.join(vae_dir, "vae_best.pth"), map_location=device))
    vae.eval()
    scaler = joblib.load(os.path.join(vae_dir, "state_scaler.pkl"))

    # Load ensemble
    ensemble_path = os.path.join(out_dir, "semi_mae", "ensemble_final.pth")
    ensemble = RewardEnsemble(C.ENSEMBLE_N_MODELS, C.ENSEMBLE_INPUT_DIM,
                               C.ENSEMBLE_HIDDEN_DIM).to(device)
    ensemble.load_state_dict(torch.load(ensemble_path, map_location=device))
    ensemble.eval()

    # Load unlabeled transitions (252-dim: S125 + A1 + R1 + S'125)
    paths = C.get_data_paths(task)
    with open(paths["unlabeled_transitions"], 'rb') as f:
        transitions = pickle.load(f)
    logger.info(f"Loaded {len(transitions)} unlabeled transitions for {task}")

    data = np.array(transitions, dtype=np.float32)
    S = data[:, :125]
    A = data[:, 125:126]
    S_prime = data[:, 127:252]

    S_scaled = scaler.transform(S).astype(np.float32)
    SP_scaled = scaler.transform(S_prime).astype(np.float32)

    relabeled = []
    batch_size = 4096

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            s_batch = torch.from_numpy(S_scaled[i:i+batch_size]).to(device)
            sp_batch = torch.from_numpy(SP_scaled[i:i+batch_size]).to(device)
            a_batch = torch.from_numpy(A[i:i+batch_size]).to(device)

            s_lat = vae.encode_deterministic(s_batch)
            sp_lat = vae.encode_deterministic(sp_batch)

            sa = torch.cat([s_lat, a_batch, sp_lat], dim=1)  # [B, 21]
            pred_r = ensemble(sa)  # [B, 1]

            # Build relabeled transitions: [s_lat(10), a(1), r_pred(1), s'_lat(10)] = 22 dims
            batch_relabeled = torch.cat([s_lat, a_batch, pred_r, sp_lat], dim=1)
            relabeled.append(batch_relabeled.cpu().numpy())

            if (i // batch_size) % 10 == 0:
                logger.info(f"Relabeled {min(i+batch_size, len(data))}/{len(data)}")

    relabeled_data = np.concatenate(relabeled, axis=0)
    output_path = os.path.join(out_dir, "data", "relabeled_buffer_latent.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(relabeled_data.tolist(), f)
    logger.info(f"Saved {len(relabeled_data)} relabeled transitions to {output_path}")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    relabel(args.task, args.gpu)
