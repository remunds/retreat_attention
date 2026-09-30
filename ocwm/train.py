"""Train the gated object-centric world model on collected trajectories.

Usage:
    uv run python -m ocwm.train --data data/train.npz --out runs/base
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from ocwm import data as D
from ocwm.env import OBJECTS
from ocwm.model import GATE_MASK, SOURCE_NAMES, ModelConfig, Stat, WorldModel, save


def delta_stats(pos, idx):
    delta = pos[idx[:, 0], idx[:, 1] + 1] - pos[idx[:, 0], idx[:, 1]]  # [N, 3, 2]
    return delta.mean(0), np.maximum(delta.std(0), 0.1)


def make_steps(graphdef, stats, tx, history, rollout, temperature):
    def loss_fn(params, pos, action, idx, key, sparsity):
        """Autoregressive rollout of `rollout` steps on the model's own predictions (BPTT).

        Position errors are normalized by the per-object Δ std, so step 1 is the one-step loss.
        """
        model = nnx.merge(graphdef, params, stats)
        hist, acts, future = D.gather_sequences(pos, action, idx, history, rollout)
        std, mean = model.delta_std[...], model.delta_mean[...]

        def step(hist, inp):
            act, true_next, key = inp
            delta, gates, logits = model(hist, act, key=key, temperature=temperature)
            nxt = hist[:, -1] + delta * std + mean
            err = optax.huber_loss((nxt - true_next) / std).mean()
            open_prob = (jax.nn.sigmoid(logits) * GATE_MASK).sum((1, 2)).mean() / GATE_MASK.sum()
            return jnp.concatenate([hist[:, 1:], nxt[:, None]], 1), (err, open_prob)

        keys = jax.random.split(key, rollout)
        _, (errs, open_prob) = jax.lax.scan(step, hist, (acts.T, future.swapaxes(0, 1), keys))
        pred_loss, open_prob = errs.mean(), open_prob.mean()
        loss = pred_loss + sparsity * open_prob
        return loss, dict(loss=loss, pred=pred_loss, open=open_prob, first=errs[0])

    @jax.jit
    def train_step(params, opt_state, pos, action, idx, key, sparsity):
        grads, metrics = jax.grad(loss_fn, has_aux=True)(params, pos, action, idx, key, sparsity)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    @jax.jit
    def eval_step(params, pos, action, idx):
        model = nnx.merge(graphdef, params, stats)
        hist, act, nxt = D.gather_windows(pos, action, idx, history)
        pred, gates = model.next_positions(hist, act)
        return jnp.abs(pred - nxt).mean(0), gates.mean(0)  # [3, 2] pixel MAE, [3, 4] open rate

    return train_step, eval_step


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="run directory, model saved to <out>/model.pkl")
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rollout", type=int, default=8, help="training rollout length (1 = one-step loss)")
    p.add_argument("--surprise-frac", type=float, default=0.5, help="batch fraction drawn from surprising windows")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sparsity", type=float, default=0.003, help="weight of the gate-opening penalty")
    p.add_argument("--sparsity-warmup", type=float, default=0.3, help="fraction of steps to ramp the penalty in")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = D.load(args.data)
    num_envs = d["pos"].shape[0]
    idx = D.valid_indices(d["done"], args.history, args.rollout)
    is_val = idx[:, 0] >= int(num_envs * 0.9)
    train_idx, val_idx = idx[~is_val], idx[is_val]
    # One pool per object: windows where that object does something a constant-velocity guess misses.
    surprise = D.surprise_masks(d["pos"], train_idx, args.rollout)
    pools = [train_idx[surprise[:, i]] for i in range(len(OBJECTS)) if surprise[:, i].any()]
    print(f"{len(train_idx)} train / {len(val_idx)} val windows; surprising per object: {surprise.mean(0).round(3)}")

    config = ModelConfig(history=args.history, hidden=args.hidden)
    model = WorldModel(config, nnx.Rngs(args.seed))
    mean, std = delta_stats(d["pos"], train_idx)
    model.delta_mean[...] = jnp.asarray(mean)
    model.delta_std[...] = jnp.asarray(std)

    graphdef, params, stats = nnx.split(model, nnx.Param, Stat)
    tx = optax.adamw(optax.cosine_decay_schedule(args.lr, args.steps))
    opt_state = tx.init(params)
    train_step, eval_step = make_steps(graphdef, stats, tx, args.history, args.rollout, args.temperature)

    pos, action = jnp.asarray(d["pos"]), jnp.asarray(d["action"])
    val_batch = jnp.asarray(val_idx[np.random.default_rng(0).permutation(len(val_idx))[:8192]])
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    for step in range(1, args.steps + 1):
        key, k = jax.random.split(key)
        n_sur = int(args.batch * args.surprise_frac) // len(pools)
        parts = [train_idx[rng.integers(0, len(train_idx), args.batch - n_sur * len(pools))]]
        parts += [pool[rng.integers(0, len(pool), n_sur)] for pool in pools]
        batch = jnp.asarray(np.concatenate(parts))
        # Ramp the sparsity penalty in, so dependencies are learned before they are pruned.
        sparsity = args.sparsity * min(1.0, step / (args.sparsity_warmup * args.steps))
        params, opt_state, m = train_step(params, opt_state, pos, action, batch, k, sparsity)
        if step % args.eval_every == 0 or step == args.steps:
            mae, open_rate = eval_step(params, pos, action, val_batch)
            mae_str = " ".join(f"{o}={float(mae[i].mean()):.3f}" for i, o in enumerate(OBJECTS))
            print(f"step {step:6d} | loss {float(m['loss']):.4f} pred {float(m['pred']):.4f} (1-step {float(m['first']):.4f}) open {float(m['open']):.3f} | val MAE px: {mae_str}", flush=True)

    print("gate open rate on val (rows: target, cols: " + ", ".join(SOURCE_NAMES) + ")")
    print(np.round(np.asarray(open_rate), 3))
    os.makedirs(args.out, exist_ok=True)
    nnx.update(model, params)
    save(model, os.path.join(args.out, "model.pkl"))
    print(f"saved {args.out}/model.pkl")


if __name__ == "__main__":
    main()
