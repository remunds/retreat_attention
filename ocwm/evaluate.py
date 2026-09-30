"""World-model prediction error and gate usage, with and without dynamics interventions (mods).

Usage:
    uv run python -m ocwm.evaluate --ckpt runs/base/model.pkl
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from ocwm import data as D
from ocwm import model as M
from ocwm.env import OBJECTS, make_env


def evaluate(model, mods, num_envs, num_steps, rollout, seed):
    H = model.config.history
    d = D.collect(make_env(mods), jax.random.PRNGKey(seed), num_envs, num_steps)
    idx = D.valid_indices(d["done"], H)
    idx = idx[idx[:, 1] + rollout < num_steps]  # room for the open-loop rollout
    idx = jnp.asarray(idx[np.random.default_rng(seed).permutation(len(idx))[:4096]])
    pos, action = jnp.asarray(d["pos"]), jnp.asarray(d["action"])

    @jax.jit
    def run(idx):
        hist, _, nxt = D.gather_windows(pos, action, idx, H)
        acts = jax.vmap(lambda i: jax.lax.dynamic_slice_in_dim(action[i[0]], i[1], rollout))(idx)
        true = jax.vmap(lambda i: jax.lax.dynamic_slice_in_dim(pos[i[0]], i[1] + 1, rollout))(idx)
        pred, gates = M.imagine(model, hist, acts)
        one_step = jnp.abs(pred[:, 0] - nxt).mean((0, 2))  # [3]
        multi = jnp.abs(pred[:, -1] - true[:, -1]).mean((0, 2))  # [3]
        return one_step, multi, gates[:, 0].mean(0)

    return run(idx)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mods", nargs="*", default=["none", "lazy_enemy"], help="'none' = unmodified game")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=2048)
    p.add_argument("--rollout", type=int, default=16, help="open-loop rollout length for the multi-step error")
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    model = M.load(args.ckpt)
    header = " | ".join(f"{o:>7}" for o in OBJECTS)
    print(f"{'mod':>14} | 1-step MAE px: {header} || {args.rollout}-step MAE px: {header}")
    for mod in args.mods:
        mods = [] if mod == "none" else [mod]
        one, multi, gates = evaluate(model, mods, args.num_envs, args.num_steps, args.rollout, args.seed)
        fmt = lambda v: " | ".join(f"{float(x):7.3f}" for x in v)
        print(f"{mod:>14} |                {fmt(one)} ||                 {fmt(multi)}")
        print(f"{'':>14}   gate open rate (rows: target, cols: {', '.join(M.SOURCE_NAMES)}):")
        for i, o in enumerate(OBJECTS):
            print(f"{'':>16}{o:>7}: " + "  ".join(f"{float(g):.2f}" for g in gates[i]))


if __name__ == "__main__":
    main()
