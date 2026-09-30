"""Video of one player: the game, the player's world-model dependencies, and (for program
players) the program branches executing at each step.

The player is the planner using the world model in --ckpt (a program model .json or the neural
model .pkl). Next to the game: the model's dependency matrix at this step (program: sources the
executed branch read; neural: gate probabilities), the ball's recent x, and on the right either
the path through each program (conditions taken, expression used) or, for the neural player, its
gate probabilities over the last 240 steps.

Usage:
    uv run python -m ocwm.video --ckpt runs/programs/programs.json --out runs/videos/programs_train.mp4
    uv run python -m ocwm.video --ckpt runs/v3/model.pkl --mods lazy_enemy --out runs/videos/neural_eval.mp4
"""

import argparse

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ocwm import actor as A
from ocwm import model as M
from ocwm import programs as P
from ocwm.decisions import step_with_weights
from ocwm.env import OBJECTS, make_env, object_positions

HISTORY = 4  # recorded; each model uses the last `config.history` steps
SHOWN_PROGRAMS = [("player", "dy"), ("enemy", "dy"), ("ball", "dx"), ("ball", "dy")]
MAX_CONDITIONS = 4
INK, MUTED = "#0b0b0b", "#52514e"


def record(env, actor_model, key, num_steps, horizon, replan_every=8):
    """Play with the world-model planner; return per-step env states, histories [T,H,3,2], actions, k."""
    plan = A.model_planner(actor_model, horizon)
    H_actor = actor_model.config.history

    def env_step(carry, key, k):
        state, hist, prev_action = carry
        action = A.steer(hist[-1], hist[-2], k)
        action = jnp.where(jax.random.uniform(key) < A.STICKY_PROB, prev_action, action)
        obs, new_state, *_ = env.step(state, action)
        new_hist = jnp.concatenate([hist[1:], object_positions(obs)[None]])
        return (new_state, new_hist, action), (state, hist, action, k)

    def chunk(carry, key):
        k = plan(carry[0], carry[1][-H_actor:])
        return jax.lax.scan(lambda c, kk: env_step(c, kk, k), carry, jax.random.split(key, replan_every))

    @jax.jit
    def run(key):
        k_reset, k_run = jax.random.split(key)
        obs, state = env.reset(k_reset)
        hist = jnp.repeat(object_positions(obs)[None], HISTORY, 0)
        _, out = jax.lax.scan(chunk, (state, hist, jnp.int32(0)), jax.random.split(k_run, num_steps // replan_every))
        return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), out)

    return run(key)


def program_path(node, f):
    """Conditions on the executed path [(text, taken)] and the expression used, for one sample."""
    path = []
    while node[0] == "ite":
        taken = bool(P.evaluate(node[1], f, np)[0])
        path.append((P.to_str(node[1]), taken))
        node = node[2] if taken else node[3]
    return path, P.to_str(node)


def program_text(model, hist, action):
    f = P.features(np.asarray(hist)[None, -2:], np.asarray(action)[None], np)
    lines = [f"(locked away, never read: {', '.join(model.excluded)})", ""] if model.excluded else []
    for o, d in SHOWN_PROGRAMS:
        if o in model.excluded:
            continue
        path, leaf = program_path(model.programs[(o, d)], f)
        lines.append(f"{o}.{d}")
        if len(path) > MAX_CONDITIONS:
            lines.append(f"   … {len(path) - MAX_CONDITIONS} earlier conditions")
        for cond, taken in path[-MAX_CONDITIONS:]:
            lines.append(f"   {'✓' if taken else '✗'} {cond}")
        lines.append(f"   → {leaf}")
        lines.append("")
    return "\n".join(lines)


def player_name(path, model):
    if not path.endswith(".json"):
        return "neural (gated) player"
    return "program player, enemy locked away" if "enemy" in model.excluded else "program player"


class Frame:
    """Reusable matplotlib figure for one player; draw() returns an RGB frame."""

    def __init__(self, title, is_program, excluded=(), timeline=240):
        self.is_program, self.excluded, self.timeline = is_program, excluded, timeline
        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1.25, 1.1, 2.3], height_ratios=[1, 1], wspace=0.28, hspace=0.45, left=0.03, right=0.98, top=0.86, bottom=0.07)
        self.fig.suptitle(title, x=0.03, ha="left", color=INK, fontsize=12)
        self.sub = self.fig.text(0.03, 0.905, "", color=MUTED, fontsize=10)
        ax = self.fig.add_subplot(gs[:, 0])
        self.game = ax.imshow(np.zeros((210, 160, 3), np.uint8), interpolation="nearest")
        ax.axis("off")

        ax = self.fig.add_subplot(gs[0, 1])
        self.mat = ax.imshow(np.zeros((len(OBJECTS), M.NUM_SOURCES)), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(M.NUM_SOURCES), M.SOURCE_NAMES, fontsize=8, color=MUTED)
        ax.set_yticks(range(len(OBJECTS)), [f"{o} uses" for o in OBJECTS], fontsize=8, color=MUTED)
        ax.set_title("branch reads (this step)" if is_program else "gate probability (this step)", fontsize=9, color=INK)
        self.cells = [[ax.text(s, t, "", ha="center", va="center", fontsize=8) for s in range(M.NUM_SOURCES)] for t in range(len(OBJECTS))]

        ax = self.fig.add_subplot(gs[1, 1])
        self.ball_ax = ax
        (self.ball_line,) = ax.plot([], [], color=MUTED, lw=1.2)
        ax.set_xlim(-timeline, 0)
        ax.set_ylim(0, 160)
        ax.axhline(140, color="#e4e3df", lw=1, ls="--")
        ax.set_title("ball x, last steps", fontsize=9, color=INK)
        ax.tick_params(colors=MUTED, labelsize=7)

        ax = self.fig.add_subplot(gs[:, 2])
        if is_program:
            ax.axis("off")
            ax.set_title("programs: executed branch this step", loc="left", fontsize=9, color=INK)
            self.prog = ax.text(0, 1, "", va="top", ha="left", family="monospace", fontsize=10, color=INK, transform=ax.transAxes)
        else:
            from ocwm.decisions import PAIR_NAMES

            self.pairs = [(t, s) for t in range(len(OBJECTS)) for s in range(M.NUM_SOURCES) if s != t]
            self.tl = ax.imshow(np.zeros((len(self.pairs), timeline)), aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest", extent=(-timeline, 0, len(self.pairs) - 0.5, -0.5))
            ax.set_yticks(range(len(self.pairs)), PAIR_NAMES, fontsize=8, color=MUTED)
            ax.set_xlabel("steps ago", fontsize=8, color=MUTED)
            ax.set_title("gate probability over the last steps", loc="left", fontsize=9, color=INK)
            ax.tick_params(colors=MUTED, labelsize=7)

    def draw(self, t, frame, sub, mats, ball_x, text=None):
        self.game.set_data(frame)
        self.sub.set_text(sub)
        mat = mats[t]
        locked = [OBJECTS.index(o) for o in self.excluded]
        shown = np.where(np.asarray(M.GATE_MASK) > 0, mat, np.nan)
        for i in locked:
            shown[i, :] = np.nan
            shown[:, i] = np.nan
        self.mat.set_data(shown)
        for r in range(len(OBJECTS)):
            for c in range(M.NUM_SOURCES):
                label = "self" if r == c else ("locked" if (r in locked or c in locked) else f"{mat[r, c]:.2f}")
                self.cells[r][c].set_text(label)
                self.cells[r][c].set_color(MUTED if label in ("self", "locked") else ("white" if mat[r, c] > 0.55 else INK))
        lo = max(0, t - self.timeline + 1)
        self.ball_line.set_data(np.arange(lo, t + 1) - t, ball_x[lo : t + 1])
        if self.is_program:
            self.prog.set_text(text)
        else:
            window = np.zeros((len(self.pairs), self.timeline))
            recent = mats[lo : t + 1]
            window[:, self.timeline - len(recent) :] = np.stack([recent[:, a, b] for a, b in self.pairs])
            self.tl.set_data(window)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="world model of the player (.json program or .pkl neural)")
    p.add_argument("--mods", nargs="*", default=[])
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--horizon", type=int, default=256)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    env = make_env(args.mods)
    model = M.load(args.ckpt)
    is_program = args.ckpt.endswith(".json")
    states, hist, action, ks = record(env, model, jax.random.PRNGKey(args.seed), args.steps, args.horizon)
    frames = np.asarray(jax.jit(jax.vmap(env.render))(states))
    mats = np.asarray(step_with_weights(model)(hist[:, -model.config.history :], action)[1])
    hist, action, ks = np.asarray(hist), np.asarray(action), np.asarray(ks)
    ps, es = np.asarray(states.player_score), np.asarray(states.enemy_score)
    ball_x = hist[:, -1, 2, 0]

    game = f"evaluation game ({', '.join(args.mods)})" if args.mods else "training game (normal enemy)"
    view = Frame(f"{player_name(args.ckpt, model)} — {game}", is_program, getattr(model, "excluded", ()))
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=7, macro_block_size=1) as writer:
        for t in range(len(frames)):
            sub = f"step {t:5d}   score {ps[t]}–{es[t]}   chosen hit offset k = {ks[t]:.1f} (8 = paddle center)"
            text = program_text(model, hist[t], action[t]) if is_program else None
            writer.append_data(view.draw(t, frames[t], sub, mats, ball_x, text))
    print(f"saved {args.out}: {len(frames)} frames at {args.fps} fps")


if __name__ == "__main__":
    main()
