"""Open-loop rollout evaluation of the trained per-object baseline (no gating).

Feeds the model's own predictions back in as history (autoregressive), using
the recorded ground-truth action sequence to drive the player object, and
compares against the true rollout to see how prediction error compounds over
time. This is the "non-robust baseline" other approaches get compared against.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from world_model.model import forward

DEFAULT_DATA = Path("artifacts/rollout.npz")
DEFAULT_PARAMS = Path("artifacts/baseline_params.pkl")


def predict_next(entry, x):
    x_norm = (x - entry["x_mean"]) / entry["x_std"]
    y_norm = forward(entry["params"], x_norm)
    return np.array(y_norm) * np.array(entry["y_std"]) + np.array(entry["y_mean"])


def rollout(checkpoint, data, start: int, horizon: int):
    window = checkpoint["window"]
    objects = checkpoint["objects"]
    actions = data["actions"]

    history = {name: [data[name][start - window + 1 + i] for i in range(window)] for name in objects}
    predicted = {name: [] for name in objects}
    truth = {name: [] for name in objects}

    for step in range(horizon):
        t = start + step
        for name, entry in objects.items():
            x = np.concatenate(history[name])
            if entry["include_action"]:
                x = np.concatenate([x, [actions[t]]])
            pred = predict_next(entry, x[None, :])[0]
            predicted[name].append(pred)
            truth[name].append(data[name][t + 1])
            history[name] = history[name][1:] + [pred]

    return (
        {name: np.stack(v) for name, v in predicted.items()},
        {name: np.stack(v) for name, v in truth.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=200)
    args = parser.parse_args()

    raw = np.load(args.data)
    data = {k: raw[k] for k in raw.files}
    with open(args.params, "rb") as f:
        checkpoint = pickle.load(f)

    predicted, truth = rollout(checkpoint, data, args.start, args.horizon)

    print(f"Open-loop rollout MSE over {args.horizon} steps (start={args.start}):")
    for name in checkpoint["objects"]:
        err = (predicted[name] - truth[name]) ** 2
        first_half = err[: args.horizon // 2].mean()
        second_half = err[args.horizon // 2 :].mean()
        print(f"  {name:6s} | first-half MSE {first_half:8.3f} | second-half MSE {second_half:8.3f}")


if __name__ == "__main__":
    main()
