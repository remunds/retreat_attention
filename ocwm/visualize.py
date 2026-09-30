"""Plot which sources each object's prediction depended on, at every step of a rollout.

Usage:
    uv run python -m ocwm.visualize --ckpt runs/base/model.pkl --mods lazy_enemy --out runs/base/gates.png
"""

import argparse

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ocwm import model as M
from ocwm.data import collect
from ocwm.env import BALL, OBJECTS, PLAYER_X, make_env

OPEN_COLOR = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def rollout_gates(model, mods, num_steps, seed):
    """One game with a near-greedy tracking policy. Returns pos [T+1, 3, 2] and gates [T', 3, 4]."""
    d = collect(make_env(mods), jax.random.PRNGKey(seed), 1, num_steps, eps_range=(0.05, 0.05))
    pos, action = d["pos"][0], d["action"][0]
    H = model.config.history
    hist = np.stack([pos[t - H + 1 : t + 1] for t in range(H - 1, num_steps)])
    _, gates = jax.jit(lambda h, a: model.next_positions(h, a))(jnp.asarray(hist), jnp.asarray(action[H - 1 :]))
    return pos, np.asarray(gates), H - 1


def plot(pos, gates, t0, title, out):
    steps = np.arange(t0, t0 + len(gates))
    fig, axes = plt.subplots(
        1 + len(OBJECTS), 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.6] + [1] * len(OBJECTS)},
    )
    ax = axes[0]
    ax.plot(np.arange(len(pos)), pos[:, BALL, 0], color=MUTED, lw=1.5)
    for x, name in [(PLAYER_X, "player paddle"), (pos[0, 1, 0], "enemy paddle")]:
        ax.axhline(x, color=GRID, lw=1, ls="--", zorder=0)
        ax.text(steps[-1], x, f" {name}", va="center", fontsize=8, color=MUTED)
    ax.set_ylabel("ball x", color=MUTED)
    ax.set_title(title, loc="left", color=INK)

    for i, (ax, target) in enumerate(zip(axes[1:], OBJECTS)):
        sources = [s for s in range(M.NUM_SOURCES) if s != i]
        img = np.stack([gates[:, i, s] for s in sources])  # [S, T]
        ax.imshow(
            img, aspect="auto", interpolation="nearest", cmap=matplotlib.colors.ListedColormap(["#ffffff", OPEN_COLOR]),
            extent=(steps[0] - 0.5, steps[-1] + 0.5, len(sources) - 0.5, -0.5), vmin=0, vmax=1,
        )
        ax.set_yticks(range(len(sources)), [M.SOURCE_NAMES[s] for s in sources], fontsize=8, color=MUTED)
        ax.set_ylabel(f"{target}\nattends to", color=INK, fontsize=9)
    axes[-1].set_xlabel("step", color=MUTED)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.99, 0.01, "filled = gate open (dependency used)", ha="right", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mods", nargs="*", default=[])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = M.load(args.ckpt)
    pos, gates, t0 = rollout_gates(model, args.mods, args.steps, args.seed)
    plot(pos, gates, t0, f"Per-step dependencies, mods={args.mods or 'none'}", args.out)


if __name__ == "__main__":
    main()
