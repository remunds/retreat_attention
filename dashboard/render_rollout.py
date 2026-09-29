"""Render the three dashboard videos of a checkpoint (world model + actor).

Views (`VIEWS`):
- `real_lazy`: the actor playing the *real* game with the `lazy_enemy` mod (held-out evaluation
  only; nothing here trains or selects anything). The gates are the world model's gates computed
  on the real history at every step, i.e. what the model would use to predict the next step.
- `wm_pong`: the actor playing *inside* the world model (imagination), started from real histories
  of unmodified Pong, as in training.
- `wm_lazy`: the same imagination, started from real `lazy_enemy` histories, which shows whether
  the model continues the enemy's changed behaviour or falls back to what it learned in Pong.

Imagination mirrors `experiment_sparse_object_wm.make_imagination`: the world model predicts every
step, and a predicted point (or 256 steps) ends the imagined rally and restarts from another real
history of the same environment.

Every frame shows the game with lines from each object to the objects its prediction uses (line
opacity = gate value), and below it the full gate matrix and the recent history of the ball's gates.

Works for any experiment module with the interface used by `evaluate_lazy_enemy.py`:
`build_model(cfg)`, `wm_step(model, params, hist, action) -> (pred, delta, reward_logits, gates)`
and `ACT_GREEDY` / `ACT_SAMPLE(params, obs, key)`. Checkpoints without an actor (`wm_only.pkl`)
are driven by random actions.

Runs on CPU only (sets `CUDA_VISIBLE_DEVICES=""`), so it never takes GPU memory from training jobs.

Usage:
    CUDA_VISIBLE_DEVICES= uv run dashboard/render_rollout.py --module experiment_sparse_object_wm \
        --ckpt runs/sow_v1/best.pkl --out_prefix dashboard/media/experiment_sparse_object_wm
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

VIEWS = {
    "real_lazy": ("REAL GAME · lazy_enemy", ["lazy_enemy"], False),
    "wm_pong": ("WORLD MODEL · Pong starts", None, True),
    "wm_lazy": ("WORLD MODEL · lazy_enemy starts", ["lazy_enemy"], True),
}
WALL_BOUNDS = jnp.array([[0.0, 0.0], [160.0, 210.0]])
TOKENS = ("player", "enemy", "ball", "action")
SIZES = {0: (4, 16), 1: (4, 16), 2: (2, 4)}  # (w, h) in pixels of player, enemy, ball
COLORS = {0: (92, 186, 92), 1: (213, 130, 74), 2: (236, 236, 236)}
ACTION_COLOR = (120, 170, 235)
BG = (24, 26, 30)
FIELD = (144, 72, 17)
TEXT = (230, 230, 225)
MUTED = (150, 152, 158)
SCALE = 2
W = 160 * SCALE  # 320
HEAD = 56
GAME_H = 210 * SCALE  # 420
H = 696  # header + game + gate panel, multiple of 8 for the video codec
ACTION_NAMES = ("NOOP", "FIRE", "RIGHT", "LEFT", "RFIRE", "LFIRE")


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"):
        if os.path.exists(os.path.join(d, name)):
            return ImageFont.truetype(os.path.join(d, name), size)
    return ImageFont.load_default()


def fake_obs(pos_hist):
    """(F, 3, 2) positions -> (F, 26) OC frames with only the positions filled in (all an actor reads)."""
    obs = jnp.zeros(pos_hist.shape[:-2] + (26,))
    for i in range(3):
        obs = obs.at[..., common.POS_IDX[i, 0]].set(pos_hist[..., i, 0])
        obs = obs.at[..., common.POS_IDX[i, 1]].set(pos_hist[..., i, 1])
    return obs


def load(module, ckpt):
    mod = importlib.import_module(module)
    with open(ckpt, "rb") as f:
        ck = pickle.load(f)
    ac_params = ck.get("ac_params")
    greedy = bool(ck.get("greedy", False))
    act = (mod.ACT_GREEDY if greedy else mod.ACT_SAMPLE) if ac_params is not None else None
    return mod, ck, act, greedy


def collect_real(env_mods, act, ac_params, steps, key, n_envs):
    env = common.make_env(env_mods)
    fn = act if act is not None else common.random_act
    traj = common.collect(env, fn, steps, ac_params, jax.random.split(key, n_envs))
    return jax.device_get(traj)


def real_starts(env_mods, act, ac_params, K, key, n_envs=16, steps=600):
    """Real K-frame histories (ball in play) from the given environment, played by the actor."""
    buf = common.ReplayBuffer()
    buf.add(collect_real(env_mods, act, ac_params, steps, key, n_envs))
    env_idx, t_idx = buf.valid_starts(K, 0)
    frames = buf.arrays()["frames"]
    wins = frames[env_idx[:, None], t_idx[:, None] + np.arange(-K + 1, 1)[None]]
    moving = np.any(wins[:, -1, 2] != wins[:, -2, 2], axis=-1)
    return jnp.asarray(wins[moving] if moving.sum() > 16 else wins)


def imagine(mod, ck, act, env_mods, num_steps, key, imag_len=256):
    """Actor inside the world model, (re)started from real histories of `env_mods`."""
    cfg = ck["cfg"]
    K = cfg["K"]
    model = mod.build_model(cfg)
    wm_params, ac_params = ck["wm_params"], ck.get("ac_params")
    k_start, k_run = jax.random.split(key)
    starts = real_starts(env_mods, act, ac_params, K, k_start)
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
        return (new_hist, jnp.where(done, 0, t)), dict(pos=hist[-1], action=a, reward=reward, done=done, gates=gates)

    _, traj = jax.lax.scan(jax.jit(step), (starts[0], jnp.array(0)), jax.random.split(k_run, num_steps))
    return jax.device_get(traj)


def real_game(mod, ck, act, env_mods, num_steps, key):
    """Actor in the real environment; gates = world model's gates on the real history at each step."""
    cfg = ck["cfg"]
    K = cfg["K"]
    tr = collect_real(env_mods, act, ck.get("ac_params"), num_steps, key, 1)
    pos = tr["pos"][0]  # (T, 3, 2)
    idx = np.clip(np.arange(num_steps)[:, None] + np.arange(-K + 1, 1)[None], 0, None)  # pad with the first frame
    hists = jnp.asarray(pos[idx])
    model = mod.build_model(cfg)
    _, _, _, gates = jax.jit(lambda h, a: mod.wm_step(model, ck["wm_params"], h, a))(hists, jnp.asarray(tr["action"][0]))
    return dict(pos=pos, action=tr["action"][0], reward=np.sign(tr["reward"][0]), done=tr["done"][0],
                gates=np.asarray(gates))


# ----------------------------------------------------------------------------- drawing


def _to_px(xy, i):
    w, h = SIZES[i]
    x, y = float(xy[0]), float(xy[1])
    return (x * SCALE, y * SCALE + HEAD, (x + w) * SCALE, (y + h) * SCALE + HEAD)


def _center(xy, i):
    x0, y0, x1, y1 = _to_px(xy, i)
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def draw_frame(tr, t, title, subtitle, score, flash, fonts):
    f_small, f_med, f_bold, f_big = fonts
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    d.text((10, 8), title, fill=TEXT, font=f_bold)
    d.text((10, 32), subtitle, fill=MUTED, font=f_small)
    # game
    d.rectangle((0, HEAD, W, HEAD + GAME_H), fill=FIELD)
    for yw in (24, 194):
        d.rectangle((0, HEAD + yw * SCALE, W, HEAD + (yw + 2) * SCALE), fill=(236, 236, 236, 110))
    d.text((W * 0.25, HEAD + 6), f"{int(score[1])}", fill=COLORS[1], font=f_big, anchor="ma")
    d.text((W * 0.75, HEAD + 6), f"{int(score[0])}", fill=COLORS[0], font=f_big, anchor="ma")
    pos, gates = tr["pos"][t], tr["gates"][t]
    for s in range(max(0, t - 12), t):  # ball trail within the current rally
        if tr["done"][s:t].any():
            continue
        cx, cy = _center(tr["pos"][s][2], 2)
        a = int(25 + 10 * (s - t + 12))
        d.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(236, 236, 236, a))
    for i in range(3):  # dependency lines: receiver i -> object j, opacity = gate
        for j in range(3):
            g = float(gates[i, j])
            if i == j or g < 0.05:
                continue
            d.line((_center(pos[i], i), _center(pos[j], j)), fill=COLORS[i] + (int(40 + 200 * g),), width=max(1, int(1 + 3 * g)))
    for i in range(3):
        d.rectangle(_to_px(pos[i], i), fill=COLORS[i])
    if flash:
        d.text((W / 2, HEAD + GAME_H / 2), flash[0], fill=flash[1], font=f_big, anchor="mm")

    # gate matrix: row = receiving object, column = token it uses
    y0 = HEAD + GAME_H + 8
    lx, cw, ch = 62, 62, 30
    for j, n in enumerate(TOKENS):
        d.text((lx + j * cw + cw / 2, y0 + 8), n, fill=ACTION_COLOR if j == 3 else COLORS[j], font=f_small, anchor="mm")
    for i in range(3):
        yy = y0 + 18 + i * ch
        d.text((lx - 6, yy + ch / 2), common.OBJECT_NAMES[i], fill=COLORS[i], font=f_small, anchor="rm")
        for j in range(4):
            box = (lx + j * cw + 2, yy + 2, lx + (j + 1) * cw - 2, yy + ch - 2)
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if i == j:
                d.rectangle(box, outline=(80, 82, 88), width=1)
                d.text((cx, cy), "self", fill=(110, 112, 118), font=f_small, anchor="mm")
                continue
            g = float(gates[i, j])
            shade = int(40 + 200 * g)
            d.rectangle(box, fill=(shade, int(shade * 0.85), int(60 + 120 * g)))
            d.text((cx, cy), f"{g:.2f}", fill=(20, 20, 20) if g > 0.55 else TEXT, font=f_small, anchor="mm")
    # ball's gates over the last 100 steps
    ty0 = y0 + 18 + 3 * ch + 10
    ty1 = ty0 + 60
    tx0, tx1 = 10, W - 10
    d.rectangle((tx0, ty0, tx1, ty1), outline=(70, 72, 78))
    span, lo = 100, max(0, t - 99)
    for j, col in ((0, COLORS[0]), (1, COLORS[1]), (3, ACTION_COLOR)):
        pts = [(tx0 + (s - lo) / (span - 1) * (tx1 - tx0), ty1 - float(tr["gates"][s][2, j]) * (ty1 - ty0)) for s in range(lo, t + 1)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
    d.rectangle((tx0 + 1, ty0 + 1, tx0 + 8 + d.textlength("ball's gates", font=f_small), ty0 + 17), fill=BG)
    d.text((tx0 + 4, ty0 + 2), "ball's gates", fill=MUTED, font=f_small)
    a = int(tr["action"][t])
    d.text((10, ty1 + 8), f"step {t:3d} · {ACTION_NAMES[a] if a < len(ACTION_NAMES) else a}", fill=TEXT, font=f_small)
    x = W - 10
    for j, col in ((3, ACTION_COLOR), (1, COLORS[1]), (0, COLORS[0])):
        tw = d.textlength(TOKENS[j], font=f_small)
        d.text((x - tw, ty1 + 8), TOKENS[j], fill=col, font=f_small)
        x -= tw + 10
    return np.asarray(img)


def write_video(tr, out, title, subtitle, fps):
    import imageio_ffmpeg

    fonts = (_font(12), _font(14), _font(15, bold=True), _font(28))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio_ffmpeg.write_frames(out, (W, H), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=8, output_params=["-crf", "28", "-movflags", "+faststart"])
    writer.send(None)
    score, flash, flash_left = np.zeros(2), None, 0
    for t in range(len(tr["action"])):
        flash_left = max(0, flash_left - 1)
        flash = flash if flash_left else None
        writer.send(draw_frame(tr, t, title, subtitle, score, flash, fonts).tobytes())
        r = float(tr["reward"][t])
        if r > 0.5:
            score[0] += 1
            flash, flash_left = ("+1 agent", COLORS[0]), 8
        elif r < -0.5:
            score[1] += 1
            flash, flash_left = ("+1 enemy", COLORS[1]), 8
    writer.close()
    return dict(agent=int(score[0]), enemy=int(score[1]))


def render(module, ckpt, out_prefix, steps=450, fps=15, seed=0, label=None, views=tuple(VIEWS)):
    """Writes `<out_prefix>__<view>.mp4` for every view; returns metadata per view."""
    mod, ck, act, greedy = load(module, ckpt)
    who = ("greedy actor" if greedy else "sampling actor") if act is not None else "random actions"
    key = jax.random.PRNGKey(seed)
    meta = dict(greedy=greedy, has_actor=act is not None, views={})
    for v in views:
        title, env_mods, in_wm = VIEWS[v]
        k = jax.random.fold_in(key, list(VIEWS).index(v))
        tr = imagine(mod, ck, act, env_mods, steps, k) if in_wm else real_game(mod, ck, act, env_mods, steps, k)
        out = f"{out_prefix}__{v}.mp4"
        pts = write_video(tr, out, title, f"{label or ckpt} · {who}", fps)
        meta["views"][v] = dict(file=os.path.basename(out), title=title, points=pts)
    meta.update(steps=steps, fps=fps)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(render(args.module, args.ckpt, args.out_prefix, args.steps, args.fps, args.seed, views=tuple(args.views)))


if __name__ == "__main__":
    main()
