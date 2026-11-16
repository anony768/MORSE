# MORSE

Reference implementation of **MORSE** (Latent Multi-Task Offline RL with
Mask-Empowered Reward Learning) for optimizing urban human strategies
(taxi passenger-seeking).

The pipeline has three stages. (1) Urban State Compression: a VAE compresses
the 125-dimensional urban state into a 10-dimensional latent state.
(2) Mask-Empowered Reward Learning: a trajectory masked autoencoder (TrajMAE)
and a trimmed-mean reward ensemble are trained jointly on the labeled
trajectories of the target task and the unlabeled trajectories of the other
tasks, and the ensemble then relabels the unlabeled transitions.
(3) Adaptive Knowledge Fusion: two IQL learners are trained on the labeled and
the relabeled data, and the policy parameters of the relabeled learner are
fused into the labeled one by EMA every η epochs.


## Environment

- Python 3.8
- PyTorch 1.13.1 (CUDA 11.7)
- numpy, scikit-learn

```bash
conda create -n morse python=3.8 -y
conda activate morse
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy scikit-learn
```


## Data format

- `data/ae_buffer.pkl`: rows `[S(125), A(1), S'(125)]` from all tasks (251 dims)
- `data/<task>/labeled_buffer.pkl`: transitions `[S(125), A(1), R(1), S'(125)]` (252 dims)
- `data/<task>/labeled_traj.pkl`: trajectories as token lists `[[S, "state"], [a, "action"], [r, "reward"], [S', "state"], ...]`
- `data/<task>/unlabeled_traj.pkl`, `unlabeled_transitions.pkl`: the other tasks' data in the same two formats, with the reward set to `-1`
- `output/<task>/data/*_buffer_latent.pkl`: latent transitions `[s(10), a(1), r(1), s'(10)]` (22 dims)
- `output/<task>/data/*_traj_latent.pkl`: trajectories with each state compressed to 10 dims


## Key hyperparameters


| Parameter         | Value       | Parameter         | Value        |
| ----------------- | ----------- | ----------------- | ------------ |
| Latent dim        | 10          | Reward ensemble m | 7            |
| VAE beta (KL)     | 0.1         | Masking ratio p   | 0.3          |
| VAE lr / epochs   | 2e-4 / 2000 | MAE epochs        | 1000         |
| IQL expectile τ   | 0.7         | IQL epochs        | 1000         |
| AWR temperature β | 3.0         | EMA coefficient α | 0.7          |
| Fusion interval η | 3 (epochs)  | Seeds             | 42, 123, 456 |
