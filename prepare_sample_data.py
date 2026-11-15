"""Build each task's training files from sample_data/trajectories.json.gz.

The sample holds a few real trajectories per driver. For a target driver, its own
trajectories are the labeled data and the other drivers' trajectories, with rewards
removed (R = -1), are the unlabeled data.
"""
import gzip
import json
import os
import pickle

import config as C
from models import extract_transitions_from_trajectory

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "trajectories.json.gz")


def to_rows(trajectories):
    """Flatten trajectories to rows [S(125), A, R, S'(125)]."""
    rows = []
    for traj in trajectories:
        for s, a, sp, r in zip(*extract_transitions_from_trajectory(traj)):
            rows.append(s + a + [r] + sp)
    return rows


def remove_rewards(traj):
    return [[-1, t] if t == "reward" else [v, t] for v, t in traj]


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():
    with gzip.open(SAMPLE, 'rt') as f:
        data = json.load(f)

    all_trajs = [t for trajs in data.values() for t in trajs]
    # VAE rows have no reward column: [S(125), A, S'(125)]
    dump([row[:126] + row[127:] for row in to_rows(all_trajs)], C.AE_DATA)

    for key, labeled in data.items():
        unlabeled = [remove_rewards(t) for other, trajs in data.items() if other != key for t in trajs]
        paths = C.get_data_paths(key)
        dump(labeled, paths["labeled_traj"])
        dump(to_rows(labeled), paths["labeled_buffer"])
        dump(unlabeled, paths["unlabeled_traj"])
        dump(to_rows(unlabeled), paths["unlabeled_transitions"])
    print(f"Prepared {len(data)} tasks, {len(all_trajs)} trajectories -> {C.DATA_DIR}")


if __name__ == "__main__":
    main()
