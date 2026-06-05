# MORSE
# MORSE

Reference implementation of **MORSE** (Latent Multi-Task Offline RL with
Mask-Empowered Reward Learning) for optimizing urban human strategies
(taxi passenger-seeking).

The pipeline (1) compresses the raw urban state with a VAE, (2) learns a
reward function from sparse labels via a semi-supervised Trajectory MAE plus a
trimmed-mean reward ensemble, (3) relabels unlabeled transitions, and
(4) trains an IQL policy with an adaptive EMA knowledge-fusion scheme.


## Environment

- Python 3.8
- PyTorch 1.13.1 (CUDA 11.7)
- numpy, scikit-learn, matplotlib

```bash
conda create -n morse python=3.8 -y
conda activate morse
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy scikit-learn matplotlib
```

Data format conventions:

- `*_buffer.pkl`: transition `[S(125), A(1), R(1), S'(125)]` (252 dims)
- `*_buffer_latent.pkl`: latent transition `[s(10), a(1), r(1), s'(10)]` (22 dims)
- `*_dataset_*.pkl`: trajectory format for the semi-supervised MAE


## Key hyperparameters


| Parameter         | Value        | Parameter         | Value |
| ----------------- | ------------ | ----------------- | ----- |
| Latent dim        | 10           | Reward ensemble m | 7     |
| VAE beta (KL)     | 0.1          | MAE mask ratio    | 0.3   |
| VAE lr / epochs   | 2e-4 / 2000  | MAE epochs        | 1000  |
| IQL expectile τ   | 0.7          | IQL epochs        | 1000  |
| AWR temperature β | 3.0          | EMA alpha         | 0.5   |
| EMA interval η    | 3            | Eval rollouts     | 20    |
| Seeds             | 42, 123, 456 |                   |       |
