"""What each of the player's planning decisions relied on: the world model's dependency weights.

At every decision the planner imagines each candidate hit offset and picks the best. For the
imagined rollout it picked, we average each object's dependency weights over the imagined steps:
  - neural model: soft gate probabilities sigmoid(gate logit), its attention weights;
  - program model: 1 if the program branch that ran read that source, else 0.
So a weight of 0.3 for ball <- player means: in 30% of the imagined steps behind this decision,
the ball's prediction used the player paddle (or, for the neural model, that gate's average
probability).

Usage:
    uv run python -m ocwm.decisions --ckpts runs/programs/programs.json runs/v3/model.pkl --out runs/decisions
"""

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from ocwm import actor as A
from ocwm import model as M
from ocwm.env import BALL, EVAL_MOD, OBJECTS, PLAYER_X, make_env, object_positions

PAIRS = [(t, s) for t in range(len(OBJECTS)) for s in range(M.NUM_SOURCES) if s != t]
PAIR_NAMES = [f"{OBJECTS[t]} ← {M.SOURCE_NAMES[s]}" for t, s in PAIRS]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def step_with_weights(model):
    """(hist [B,H,3,2], action [B]) -> (next positions [B,3,2], dependency weights [B,3,4])."""
    if isinstance(model, nnx.Module):
        graphdef, params = nnx.split(model)

        def f(hist, action):
            m = nnx.merge(graphdef, params)
            delta, _, logits = m(hist, action)
            nxt = hist[:, -1] + delta * m.delta_std[...] + m.delta_mean[...]
            return nxt, jax.nn.sigmoid(logits) * M.GATE_MASK

        return f
    return model.next_positions


def trace_game(env, model, key, num_decisions, horizon, replan_every=8):
    """One game with the world-model planner, recording every decision's dependency weights."""
    step_fn, H = step_with_weights(model), model.config.history

    def rollout(hist, k):
        def step(hist, _):
            action = A.steer(hist[-1], hist[-2], k)
            nxt, w = step_fn(hist[None], action[None])
            return jnp.concatenate([hist[1:], nxt]), (nxt[0], w[0])

        return jax.lax.scan(step, hist, None, length=horizon)[1]

    def decide(hist):
        trajs, weights = jax.vmap(lambda k: rollout(hist, k))(A.HIT_OFFSETS)  # [K,T,3,2], [K,T,3,4]
        use_enemy = "enemy" not in getattr(model, "excluded", ())
        scores = jax.vmap(A.score_trajectory, in_axes=(0, None, None))(trajs, hist[-1], use_enemy)
        best = jnp.argmax(scores - 1e-3 * jnp.abs(A.HIT_OFFSETS - A.CENTER_OFFSET))
        return A.HIT_OFFSETS[best], weights[best].mean(0)

    def env_step(carry, key, k):
        state, hist, prev_action = carry
        action = A.steer(hist[-1], hist[-2], k)
        action = jnp.where(jax.random.uniform(key) < A.STICKY_PROB, prev_action, action)
        obs, state, *_ = env.step(state, action)
        hist = jnp.concatenate([hist[1:], object_positions(obs)[None]])
        return (state, hist, action), hist[-1, BALL, 0]

    def chunk(carry, key):
        k, weights = decide(carry[1])
        carry, ball_x = jax.lax.scan(lambda c, kk: env_step(c, kk, k), carry, jax.random.split(key, replan_every))
        return carry, (k, weights, ball_x)

    @jax.jit
    def run(key):
        k_reset, k_run = jax.random.split(key)
        obs, state = env.reset(k_reset)
        hist = jnp.repeat(object_positions(obs)[None], H, 0)
        carry = (state, hist, jnp.int32(0))
        (state, _, _), (ks, weights, ball_x) = jax.lax.scan(chunk, carry, jax.random.split(k_run, num_decisions))
        return ks, weights, ball_x.reshape(-1), state.player_score, state.enemy_score

    return run(key)


def plot_trace(name, condition, ks, weights, ball_x, score, replan_every, out):
    steps = np.arange(len(ks)) * replan_every
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1.2, 0.8, 2.6]})
    ax = axes[0]
    ax.plot(np.arange(len(ball_x)), ball_x, color=MUTED, lw=1.2)
    ax.axhline(PLAYER_X, color=GRID, lw=1, ls="--", zorder=0)
    ax.text(len(ball_x), PLAYER_X, " player", va="center", fontsize=8, color=MUTED)
    ax.set_ylabel("ball x", color=MUTED)
    ax.set_title(f"{name} — {condition} — final score {score[0]}–{score[1]}", loc="left", color=INK)

    ax = axes[1]
    ax.step(steps, ks, where="post", color="#2a78d6", lw=1.5)
    ax.axhline(A.CENTER_OFFSET, color=GRID, lw=1, ls="--", zorder=0)
    ax.set_ylabel("chosen hit\noffset k", color=MUTED)

    ax = axes[2]
    mat = np.stack([weights[:, t, s] for t, s in PAIRS])  # [pairs, decisions]
    im = ax.imshow(
        mat, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1,
        extent=(steps[0] - replan_every / 2, steps[-1] + replan_every / 2, len(PAIRS) - 0.5, -0.5),
    )
    ax.set_yticks(range(len(PAIRS)), PAIR_NAMES, fontsize=8, color=MUTED)
    for y in (2.5, 5.5):
        ax.axhline(y, color="white", lw=2)
    ax.set_xlabel("step (one column per planning decision)", color=MUTED)
    cb = fig.colorbar(im, ax=axes, shrink=0.35, anchor=(0, 0), pad=0.01)
    cb.set_label("weight in imagined rollout", color=MUTED, fontsize=8)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(results, out):
    """Mean dependency matrix per model (rows) and condition (columns)."""
    names = list(dict.fromkeys(r[0] for r in results))
    conds = list(dict.fromkeys(r[1] for r in results))
    fig, axes = plt.subplots(len(names), len(conds), figsize=(4.2 * len(conds), 3.2 * len(names)), squeeze=False)
    for name, cond, weights in results:
        ax = axes[names.index(name), conds.index(cond)]
        mat = np.asarray(weights).mean(0)
        masked = np.ma.masked_where(np.asarray(M.GATE_MASK) == 0, mat)
        ax.imshow(masked, cmap="Blues", vmin=0, vmax=1)
        for t in range(mat.shape[0]):
            for s in range(mat.shape[1]):
                if s == t:
                    ax.text(s, t, "self", ha="center", va="center", fontsize=8, color=MUTED)
                else:
                    ax.text(s, t, f"{mat[t, s]:.2f}", ha="center", va="center", fontsize=9, color="white" if mat[t, s] > 0.55 else INK)
        ax.set_xticks(range(M.NUM_SOURCES), M.SOURCE_NAMES, fontsize=8, color=MUTED)
        ax.set_yticks(range(len(OBJECTS)), [f"{o} uses" for o in OBJECTS], fontsize=8, color=MUTED)
        ax.set_title(f"{name}\n{cond}", fontsize=9, color=INK)
    fig.suptitle("Mean dependency weight behind the player's decisions", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--decisions", type=int, default=250, help="planning decisions per game (x8 env steps)")
    p.add_argument("--horizon", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = []
    for ckpt in args.ckpts:
        model = M.load(ckpt)
        name = "program synthesis" if ckpt.endswith(".json") else "neural (gated)"
        for cond, mods in [("training game", []), (f"evaluation ({EVAL_MOD})", [EVAL_MOD])]:
            ks, weights, ball_x, ps, es = map(
                np.asarray, trace_game(make_env(mods), model, jax.random.PRNGKey(args.seed), args.decisions, args.horizon)
            )
            tag = f"{'programs' if ckpt.endswith('.json') else 'neural'}_{'train' if not mods else 'eval'}"
            plot_trace(name, cond, ks, weights, ball_x, (int(ps), int(es)), 8, os.path.join(args.out, f"{tag}.png"))
            results.append((name, cond, weights))
            mean = weights.mean(0)
            print(f"{name:18s} | {cond:24s} | score after {len(ball_x)} steps {int(ps)}-{int(es)} | "
                  + "  ".join(f"{n}={mean[t, s]:.2f}" for n, (t, s) in zip(PAIR_NAMES, PAIRS)), flush=True)
    plot_summary(results, os.path.join(args.out, "summary.png"))
    print(f"saved plots to {args.out}/")


if __name__ == "__main__":
    main()
