"""MORSE configuration: paths and hyperparameters."""
import os

PROJECT_ROOT = os.environ.get("MORSE_ROOT", os.path.dirname(os.path.abspath(__file__)))

# ========== Paths ========== #
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AE_DATA = os.path.join(DATA_DIR, "ae_buffer.pkl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def get_data_paths(task):
    d = os.path.join(DATA_DIR, task)
    return {
        "labeled_buffer": os.path.join(d, "labeled_buffer.pkl"),
        "labeled_traj": os.path.join(d, "labeled_traj.pkl"),
        "unlabeled_traj": os.path.join(d, "unlabeled_traj.pkl"),
        "unlabeled_transitions": os.path.join(d, "unlabeled_transitions.pkl"),
    }


def get_output_dir(task, seed=None):
    if seed is not None:
        return os.path.join(OUTPUT_DIR, task, f"seed_{seed}")
    return os.path.join(OUTPUT_DIR, task)


# ========== VAE ========== #
VAE_INPUT_DIM = 125
VAE_HIDDEN_DIM = 64
VAE_LATENT_DIM = 10
VAE_EPOCHS = 2000
VAE_BATCH_SIZE = 2048
VAE_LR = 2e-4
VAE_BETA = 0.1           # KL weight
VAE_PATIENCE = 500

# ========== Reward ensemble ========== #
ENSEMBLE_N_MODELS = 7
ENSEMBLE_INPUT_DIM = 21  # s(10) + a(1) + s'(10)
ENSEMBLE_HIDDEN_DIM = 64

# ========== Semi-supervised TrajMAE ========== #
MAE_EMBED_DIM = 32
MAE_NUM_HEADS = 4
MAE_FF_DIM = 64
MAE_NUM_LAYERS = 2
MAE_STATE_DIM = 10
MAE_MASK_RATIO = 0.3
MAE_EPOCHS = 1000
MAE_BATCH_SIZE = 32
MAE_LR = 1e-3

# ========== IQL + EMA fusion ========== #
IQL_STATE_DIM = 10
IQL_ACTION_DIM = 1
IQL_HIDDEN_SIZE = 256
IQL_BATCH_SIZE = 64
IQL_LR = 1e-3
IQL_GAMMA = 0.99
IQL_TAU = 0.7            # expectile
IQL_BETA = 3.0           # AWR temperature
EMA_ALPHA = 0.5
EMA_ETA = 3              # fusion interval (epochs)
IQL_EPOCHS = 1000

# ========== Quick mode (MORSE_QUICK=1): few epochs, to check that everything runs ========== #
if os.environ.get("MORSE_QUICK") == "1":
    VAE_EPOCHS = 200
    MAE_EPOCHS = 3
    IQL_EPOCHS = 10
