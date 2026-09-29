"""Render a video of an actor playing *inside* a learned world model (imagination rollout).

Works for any experiment module that exposes the interface used by `evaluate_lazy_enemy.py`:
`build_model(cfg)`, `wm_step(model, params, hist, action) -> (pred, delta, reward_logits, gates)`
and `ACT_GREEDY` / `ACT_SAMPLE(params, obs, key)`. Checkpoints are the pickles written by the
experiment scripts (`round*.pkl`, `best.pkl`, or `wm_only.pkl` without an actor, in which case the
world model is driven by random actions).

The rollout mirrors the imagined environment of `experiment_sparse_object_wm.make_imagination`:
it starts from real 16-frame histories of *unmodified* Pong, the world model predicts every step,
and a predicted point (or 256 steps) ends the imagined rally and restarts from another real history.

Each video frame shows the imagined game (left) with lines from every object to the objects its
prediction uses (line opacity = gate value), and the full gate matrix plus the recent history of
the ball's gates (right).

Runs on CPU only (sets `CUDA_VISIBLE_DEVICES=""`), so it never takes GPU memory from training jobs.

Usage:
    CUDA_VISIBLE_DEVICES= uv run dashboard/render_rollout.py --module experiment_sparse_object_wm \
        --ckpt runs/sow_v1/best.pkl --out dashboard/media/sow_v1.mp4
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import importlib
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import common  # noqa: E402

WALL_BOUNDS = jnp.array([[0.0, 0.0], [160.0, 210.0]])
TOKENS = ("player", "enemy", "ball", "action")
SIZES = {0: (4, 16), 1: (4, 16), 2: (2, 4)}  # (w, h) in pixels of player, enemy, ball
COLORS = {0: (92, 186, 92), 1: (213, 130, 74), 2: (236, 236, 236)}
BG = (24, 26, 30)
FIELD = (144, 72, 17)
TEXT = (230, 230, 225)
MUTED = (150, 152, 158)
SCALE = 2
GAME_W, GAME_H = 160 * SCALE, 210 * SCALE
PANEL_W = 320
W, H = GAME_W + PANEL_W, GAME_H + 44  # multiple of 8 for the video codec
ACTION_NAMES = ("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE")


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fake_obs(pos_hist):
    """(F, 3, 2) positions -> (F, 26) OC frames with only the positions filled in (all an actor reads)."""
    obs = jnp.zeros(pos_hist.shape[:-2] + (26,))
    for i in range(3):
        obs = obs.at[..., common.POS_IDX[i, 0]].set(pos_hist[..., i, 0])
        obs = obs.at[..., common.POS_IDX[i, 1]].set(pos_hist[..., i, 1])
    return obs


def real_starts(ac_params, act, K, key, n_envs=16, steps=600):
    """Real K-frame histories from unmodified Pong (actor or random policy) with the ball in play."""
    env = common.make_env()
    fn = act if ac_params is not None else common.random_act
    traj = common.collect(env, fn, steps, ac_params, jax.random.split(key, n_envs))
    buf = common.ReplayBuffer()
    buf.add(traj)
    env_idx, t_idx = buf.valid_starts(K, 0)
    frames = buf.arrays()["frames"]
    wins = frames[env_idx[:, None], t_idx[:, None] + np.arange(-K + 1, 1)[None]]
    moving = np.any(wins[:, -1, 2] != wins[:, -2, 2], axis=-1)
    return jnp.asarray(wins[moving] if moving.sum() > 16 else wins)


def rollout(mod, ck, num_steps, key, imag_len=256):
    cfg = ck["cfg"]
    K = cfg["K"]
    model = mod.build_model(cfg)
    wm_params = ck["wm_params"]
    ac_params = ck.get("ac_params")
    greedy = bool(ck.get("greedy", False))
    act = (mod.ACT_GREEDY if greedy else mod.ACT_SAMPLE) if ac_params is not None else None
    k_start, k_run = jax.random.split(key)
    starts = real_starts(ac_params, act, K, k_start)
    round_int = bool(cfg.get("round_int", 1))

    def step(carry, k):
        hist, t = carry
        k_act, k_reset = jax.random.split(k)
        if act is None:
            a = jax.random.randint(k_act, (), 0, common.NUM_ACTIONS)
        else:
            a = act(ac_params, fake_obs(hist[-common.FRAME_STACK:]), k_act)
        pred, _, rlog, gates = mod.wm_step(model, wm_params, hist[None], a[None])
        pred, rlog, gates = pred[0], rlog[0], gates[0]
        if round_int:
            pred = pred.at[1:].set(jnp.round(pred[1:]))
        pred = jnp.clip(pred, WALL_BOUNDS[0], WALL_BOUNDS[1])
        pred = pred.at[:2, 0].set(hist[-1, :2, 0])
        cls = jnp.argmax(rlog)
        reward = jnp.where(cls != 1, (cls - 1).astype(jnp.float32), 0.0)
        t = t + 1
        done = (cls != 1) | (t >= imag_len)
        next_hist = jnp.concatenate([hist[1:], pred[None]], 0)
        fresh = starts[jax.random.randint(k_reset, (), 0, starts.shape[0])]
        new_hist = jnp.where(done, fresh, next_hist)
        out = dict(pos=hist[-1], next=pred, action=a, reward=reward, done=done, gates=gates)
        return (new_hist, jnp.where(done, 0, t)), out

    init = (starts[0], jnp.array(0))
    _, traj = jax.lax.scan(jax.jit(step), init, jax.random.split(k_run, num_steps))
    return jax.device_get(traj), greedy, act is not None


def _to_px(xy, i):
    w, h = SIZES[i]
    x, y = float(xy[0]), float(xy[1])
    return (x * SCALE, y * SCALE + 40, (x + w) * SCALE, (y + h) * SCALE + 40)


def _center(xy, i):
    x0, y0, x1, y1 = _to_px(xy, i)
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def draw_frame(tr, t, title, score, flash, fonts):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    f_small, f_med, f_big = fonts
    d.text((10, 12), title, fill=TEXT, font=f_med)
    d.rectangle((0, 40, GAME_W, H), fill=FIELD)
    d.rectangle((0, 40 + 24 * SCALE, GAME_W, 40 + 26 * SCALE), fill=(236, 236, 236, 110))
    d.rectangle((0, 40 + 194 * SCALE, GAME_W, 40 + 196 * SCALE), fill=(236, 236, 236, 110))
    d.text((GAME_W * 0.25, 48), f"{int(score[1])}", fill=COLORS[1], font=f_big, anchor="ma")
    d.text((GAME_W * 0.75, 48), f"{int(score[0])}", fill=COLORS[0], font=f_big, anchor="ma")
    pos, gates = tr["pos"][t], tr["gates"][t]
    # ball trail
    for s in range(max(0, t - 12), t):
        if tr["done"][s:t].any():
            continue
        cx, cy = _center(tr["pos"][s][2], 2)
        a = int(25 + 10 * (s - t + 12))
        d.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(236, 236, 236, a))
    # dependency lines: receiver i -> token j with opacity = gate
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            g = float(gates[i, j])
            if g < 0.05:
                continue
            p0, p1 = _center(pos[i], i), _center(pos[j], j)
            col = COLORS[i] + (int(40 + 200 * g),)
            d.line((p0, p1), fill=col, width=max(1, int(1 + 3 * g)))
    for i in range(3):
        d.rectangle(_to_px(pos[i], i), fill=COLORS[i])
    if flash:
        msg, col = flash
        d.text((GAME_W / 2, H / 2), msg, fill=col, font=f_big, anchor="mm")

    # right panel: gate matrix
    x0 = GAME_W + 20
    d.text((x0, 52), "gates: row uses column", fill=TEXT, font=f_small)
    cell = 52
    gy = 92
    for j, n in enumerate(TOKENS):
        d.text((x0 + 60 + j * cell + cell / 2, gy - 6), n, fill=MUTED, font=f_small, anchor="md")
    for i in range(3):
        d.text((x0 + 54, gy + i * cell + cell / 2), common.OBJECT_NAMES[i], fill=COLORS[i], font=f_small, anchor="rm")
        for j in range(4):
            g = float(gates[i, j])
            box = (x0 + 60 + j * cell + 2, gy + i * cell + 2, x0 + 60 + (j + 1) * cell - 2, gy + (i + 1) * cell - 2)
            if i == j:
                d.rectangle(box, outline=(80, 82, 88), width=1)
                d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), "self", fill=(110, 112, 118), font=f_small, anchor="mm")
                continue
            shade = int(40 + 200 * g)
            d.rectangle(box, fill=(shade, int(shade * 0.85), int(60 + 120 * g)))
            d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), f"{g:.2f}",
                   fill=(20, 20, 20) if g > 0.55 else TEXT, font=f_small, anchor="mm")
    # timeline of the ball's gates
    ty0, ty1 = gy + 3 * cell + 40, gy + 3 * cell + 150
    tw = PANEL_W - 40
    d.text((x0, ty0 - 22), "ball gates, last 120 steps", fill=TEXT, font=f_small)
    d.rectangle((x0, ty0, x0 + tw, ty1), outline=(70, 72, 78))
    span = 120
    lo = max(0, t - span + 1)
    for j, col in ((0, COLORS[0]), (1, COLORS[1]), (3, (120, 170, 235))):
        pts = [(x0 + (s - lo) / (span - 1) * tw, ty1 - float(tr["gates"][s][2, j]) * (ty1 - ty0)) for s in range(lo, t + 1)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
    lx = x0
    for j, col in ((0, COLORS[0]), (1, COLORS[1]), (3, (120, 170, 235))):
        d.rectangle((lx, ty1 + 10, lx + 10, ty1 + 20), fill=col)
        d.text((lx + 14, ty1 + 15), f"ball<-{TOKENS[j]}", fill=MUTED, font=f_small, anchor="lm")
        lx += 96
    a = int(tr["action"][t])
    d.text((x0, ty1 + 44), f"step {t:4d}   action {ACTION_NAMES[a] if a < len(ACTION_NAMES) else a}", fill=TEXT, font=f_small)
    return np.asarray(img)


def render(module, ckpt, out, steps=450, fps=15, seed=0, label=None):
    import imageio_ffmpeg

    mod = importlib.import_module(module)
    with open(ckpt, "rb") as f:
        ck = pickle.load(f)
    tr, greedy, has_actor = rollout(mod, ck, steps, jax.random.PRNGKey(seed))
    who = ("greedy actor" if greedy else "sampling actor") if has_actor else "random actions (no actor)"
    title = f"{label or ckpt}  |  imagined Pong, {who}"
    fonts = (_font(12), _font(14), _font(28))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio_ffmpeg.write_frames(out, (W, H), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=8, output_params=["-crf", "28", "-movflags", "+faststart"])
    writer.send(None)
    score = np.zeros(2)
    flash, flash_left = None, 0
    for t in range(steps):
        if flash_left > 0:
            flash_left -= 1
        else:
            flash = None
        writer.send(draw_frame(tr, t, title, score, flash, fonts).tobytes())
        r = float(tr["reward"][t])
        if r > 0.5:
            score[0] += 1
            flash, flash_left = ("+1 agent", COLORS[0]), 8
        elif r < -0.5:
            score[1] += 1
            flash, flash_left = ("+1 enemy", COLORS[1]), 8
    writer.close()
    return dict(steps=steps, fps=fps, greedy=greedy, has_actor=has_actor,
                imagined_points=dict(agent=int(score[0]), enemy=int(score[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(render(args.module, args.ckpt, args.out, args.steps, args.fps, args.seed))


if __name__ == "__main__":
    main()
