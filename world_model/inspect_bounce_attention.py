"""Check whether a null-prior strong enough to kill a confound also kills
real, sparse, bounce-timed dependency.

`inspect_gate.py` buckets attention by static court position, a coarse proxy
for "is a bounce possible here". This is the direct version: `bounce_stats.py`
finds the exact raw timesteps where the ball actually bounced off each
paddle, and this recomputes a trained gate's attention specifically at the
row that predicts each such bounce (row t-1, predicting the flip at t) versus
every other row. A real, conditional dependency should show a spike right at
the relevant side's bounce rows and nowhere near as much elsewhere; a
collapsed-to-null gate should show ~no difference anywhere.
"""

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from world_model.bounce_stats import analyze
from world_model.gated_model import gate_forward
from world_model.objects import OBJECT_DIMS, distance
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
    distances = {}
    for name in other_names:
        mean, std = entry["other_stats"][name]
        candidates_norm[name] = (dataset[name] - mean) / std
        distances[name] = dataset[f"{name}_distance"] / entry["dist_scale"][name]
    if include_action:
        candidates_norm["action"] = jax.nn.one_hot(dataset["action"], entry["num_actions"])

    _, weights, _ = gate_forward(
        entry["gate_params"],
        own_current_norm,
        candidates_norm,
        candidate_names,
        entry["temperature"],
        distances,
        hard=entry.get("hard", False),
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
        data,
        args.target,
        checkpoint["window"],
        list(OBJECT_DIMS),
        include_action=entry["include_action"],
        distance_fn=distance,
    )
    weights = recompute_weights(entry, dataset)
    candidate_names = entry["candidate_names"] + ["null"]

    # A bounce detected at raw index `event_t` (ball_vel_x flips there) is
    # "predicted" by the row whose window ends at event_t - 1.
    row_t = dataset["t"]
    t_to_row = {int(t): i for i, t in enumerate(row_t)}

    bounces, _ = analyze(data)
    rows_by_side = {"player": [], "enemy": []}
    for event in bounces:
        row = t_to_row.get(event["t"] - 1)
        if row is not None:
            rows_by_side[event["side"]].append(row)

    all_rows = set(range(len(row_t)))
    print(f"Attention breakdown for '{args.target}' (n={len(row_t)} rows, params={args.params.name}):")
    for side, rows in rows_by_side.items():
        if not rows:
            print(f"  {side:6s} bounce rows: none found")
            continue
        rows = np.array(rows)
        elsewhere = np.array(sorted(all_rows - set(rows.tolist())))
        mean_bounce = weights[rows].mean(axis=0)
        mean_elsewhere = weights[elsewhere].mean(axis=0)
        bounce_str = ", ".join(f"{n}={w:.2f}" for n, w in zip(candidate_names, mean_bounce))
        elsewhere_str = ", ".join(f"{n}={w:.2f}" for n, w in zip(candidate_names, mean_elsewhere))
        print(f"  {side:6s} bounce rows (n={len(rows):3d}): {bounce_str}")
        print(f"  {side:6s} elsewhere   (n={len(elsewhere):5d}): {elsewhere_str}")


if __name__ == "__main__":
    main()
