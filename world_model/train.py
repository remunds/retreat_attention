"""Train independent per-object one-step predictors on ground-truth rollout data (no gating).

This is the plain baseline from solutions.md: each object's predictor only ever
sees its own history (plus its own action, for the player, since that's the
agent's exogenous control input rather than a dependency on another object).
"""

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from world_model.model import forward, init_params
from world_model.objects import OBJECT_DIMS
from world_model.windows import build_dataset, train_val_split

WINDOW = 4
HIDDEN_DIM = 64
DEFAULT_DATA = Path("artifacts/rollout.npz")
DEFAULT_OUT = Path("artifacts/baseline_params.pkl")


def normalize_stats(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    return mean, std


def make_loss_fn(x_mean, x_std, y_mean, y_std):
    def loss_fn(params, x, y):
        x_norm = (x - x_mean) / x_std
        pred = forward(params, x_norm) * y_std + y_mean
        return jnp.mean((pred - y) ** 2)

    return loss_fn


def train_object(object_name: str, data: dict, key, epochs: int, batch_size: int, lr: float):
    dim = OBJECT_DIMS[object_name]
    include_action = object_name == "player"
    x, y = build_dataset(data, object_name, WINDOW, include_action=include_action)
    (x_train, y_train), (x_val, y_val) = train_val_split(x, y)

    x_mean, x_std = normalize_stats(x_train)
    y_mean, y_std = normalize_stats(y_train)
    loss_fn = make_loss_fn(x_mean, x_std, y_mean, y_std)

    in_dim = dim * WINDOW + (1 if include_action else 0)
    params = init_params(key, in_dim=in_dim, hidden_dim=HIDDEN_DIM, out_dim=dim)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng = np.random.default_rng(0)
    num_train = len(x_train)
    for epoch in range(epochs):
        perm = rng.permutation(num_train)
        for i in range(0, num_train, batch_size):
            batch_idx = perm[i : i + batch_size]
            params, opt_state, _ = step(params, opt_state, x_train[batch_idx], y_train[batch_idx])

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            val_loss = loss_fn(params, x_val, y_val)
            print(f"  [{object_name}] epoch {epoch + 1:3d}/{epochs} | val MSE {float(val_loss):.4f}")

    return {
        "params": params,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "include_action": include_action,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    raw = np.load(args.data)
    data = {k: raw[k] for k in raw.files}

    key = jax.random.PRNGKey(0)
    results = {}
    for object_name in OBJECT_DIMS:
        print(f"Training {object_name} predictor...")
        key, subkey = jax.random.split(key)
        results[object_name] = train_object(object_name, data, subkey, args.epochs, args.batch_size, args.lr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"window": WINDOW, "objects": results}, f)
    print(f"Saved trained baseline to {args.out}")


if __name__ == "__main__":
    main()
