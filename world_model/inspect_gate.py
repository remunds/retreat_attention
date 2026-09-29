"""Check whether a trained gate's attention pattern reflects a real,
game-relevant dependency or a spurious correlation.

Concretely: does the ball predictor's attention on "enemy" concentrate near
the enemy paddle's side of the court (where a bounce is actually possible),
or is it roughly constant everywhere (which would suggest it's exploiting
`enemy_y` as a lagged proxy for recent ball position via the enemy AI's own
ball-tracking, rather than learning genuine bounce-proximity relevance)?
"""

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from world_model.gated_model import gate_forward
from world_model.objects import OBJECT_DIMS
from world_model.windows import build_joint_dataset

DEFAULT_DATA = Path("artifacts/rollout.npz")


def recompute_weights(entry, dataset):
    own = dataset["own"]
    other_names = entry["other_names"]
    candidate_names = entry["candidate_names"]
    include_action = entry["include_action"]
    dim_t = len(entry["y_mean"])

    own_norm = (own - entry["own_mean"]) / entry["own_std"]
    own_current_norm = own_norm[:, -dim_t:]

    candidates_norm = {}
    for name in other_names:
        mean, std = entry["other_stats"][name]
        candidates_norm[name] = (dataset[name] - mean) / std
    if include_action:
        candidates_norm["action"] = jax.nn.one_hot(dataset["action"], entry["num_actions"])

    _, weights, _ = gate_forward(
        entry["gate_params"], own_current_norm, candidates_norm, candidate_names, entry["temperature"]
    )
    return np.array(weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--target", default="ball")
    args = parser.parse_args()

    raw = np.load(args.data)
    data = {k: raw[k] for k in raw.files}
    with open(args.params, "rb") as f:
        checkpoint = pickle.load(f)
    entry = checkpoint["objects"][args.target]

    dataset = build_joint_dataset(
        data, args.target, checkpoint["window"], list(OBJECT_DIMS), include_action=entry["include_action"]
    )
    weights = recompute_weights(entry, dataset)
    candidate_names = entry["candidate_names"] + ["null"]

    dim_t = OBJECT_DIMS[args.target]
    own_current_raw = dataset["own"][:, -dim_t:]

    if args.target == "ball":
        ball_x = own_current_raw[:, 0]  # enemy paddle sits near x=16, player near x=140
        buckets = [
            ("near enemy  (x<40)", ball_x < 40),
            ("mid-court (40<=x<=120)", (ball_x >= 40) & (ball_x <= 120)),
            ("near player (x>120)", ball_x > 120),
        ]
    else:
        buckets = [("all", np.ones(len(weights), dtype=bool))]

    print(f"Attention breakdown for '{args.target}' (n={len(weights)} examples):")
    for label, mask in buckets:
        if mask.sum() == 0:
            continue
        mean_w = weights[mask].mean(axis=0)
        weight_str = ", ".join(f"{n}={w:.2f}" for n, w in zip(candidate_names, mean_w))
        print(f"  {label:24s} (n={int(mask.sum()):5d}): {weight_str}")


if __name__ == "__main__":
    main()
