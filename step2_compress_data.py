import os
import sys
import pickle
import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from models import VAE
from utils import get_device, setup_logging


def compress_transitions(data_path, vae, scaler, device, output_path):
    with open(data_path, 'rb') as f:
        transitions = pickle.load(f)
    data = np.array(transitions, dtype=np.float32)
    n = len(data)

    S = data[:, :125]
    A = data[:, 125:126]
    R = data[:, 126:127]
    S_prime = data[:, 127:252]

    S_scaled = scaler.transform(S).astype(np.float32)
    S_prime_scaled = scaler.transform(S_prime).astype(np.float32)

    batch_size = 4096
    s_latent_list, sp_latent_list = [], []

    vae.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            s_batch = torch.from_numpy(S_scaled[i:i+batch_size]).to(device)
            sp_batch = torch.from_numpy(S_prime_scaled[i:i+batch_size]).to(device)
            s_latent_list.append(vae.encode_deterministic(s_batch).cpu().numpy())
            sp_latent_list.append(vae.encode_deterministic(sp_batch).cpu().numpy())

    S_lat = np.concatenate(s_latent_list, axis=0)
    SP_lat = np.concatenate(sp_latent_list, axis=0)

    compressed = np.concatenate([S_lat, A, R, SP_lat], axis=1)
    with open(output_path, 'wb') as f:
        pickle.dump(compressed.tolist(), f)
    print(f"Compressed {n} transitions -> {output_path} (dim={compressed.shape[1]})")


def compress_trajectories(data_path, vae, scaler, device, output_path):
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)

    vae.eval()
    compressed_trajs = []
    for traj in trajectories:
        new_traj = []
        for val, t in traj:
            if t == "state":
                s = np.array(val, dtype=np.float32).reshape(1, -1)
                if s.shape[1] == 125:
                    s_scaled = scaler.transform(s).astype(np.float32)
                    with torch.no_grad():
                        s_lat = vae.encode_deterministic(
                            torch.from_numpy(s_scaled).to(device)
                        ).cpu().numpy().flatten().tolist()
                    new_traj.append((s_lat, t))
                else:
                    new_traj.append((val, t))
            else:
                new_traj.append((val, t))
        compressed_trajs.append(new_traj)

    with open(output_path, 'wb') as f:
        pickle.dump(compressed_trajs, f)
    print(f"Compressed {len(trajectories)} trajectories -> {output_path}")


def main(task, gpu_id=0):
    device = get_device(gpu_id)
    vae_dir = os.path.join(C.OUTPUT_DIR, "vae")
    logger = setup_logging(vae_dir, "compress")

    vae = VAE(C.VAE_INPUT_DIM, C.VAE_HIDDEN_DIM, C.VAE_LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(os.path.join(vae_dir, "vae_best.pth"), map_location=device))
    vae.eval()
    scaler = joblib.load(os.path.join(vae_dir, "state_scaler.pkl"))

    paths = C.get_data_paths(task)
    out_dir = os.path.join(C.get_output_dir(task), "data")
    os.makedirs(out_dir, exist_ok=True)

    compress_transitions(paths["labeled_buffer"], vae, scaler, device,
                         os.path.join(out_dir, "labeled_buffer_latent.pkl"))
    compress_trajectories(paths["labeled_traj"], vae, scaler, device,
                          os.path.join(out_dir, "labeled_traj_latent.pkl"))
    compress_trajectories(paths["unlabeled_traj"], vae, scaler, device,
                          os.path.join(out_dir, "unlabeled_traj_latent.pkl"))
    logger.info(f"Compressed data for {task}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args.task, args.gpu)
