"""Render the dashboard videos of a Seaquest checkpoint: the agent in the real game.

Views (`VIEWS`):
- `real_base`: the agent playing the base game (the training environment).
- `real_gravity`: the agent playing with the held-out `gravity` mod (evaluation only; nothing here
  trains or selects anything).

Every frame is the game's own rendering (2x), with a header showing the step, the divers currently
carried, successful rescues so far (surfacing with 6 divers), oxygen, lives and score, and a flash
when a rescue happens or a life is lost.

Works for any experiment module that exposes `ACT_GREEDY` / `ACT_SAMPLE(params, obs, key)` and saves
`ac_params` (and optionally `greedy`) in its checkpoints. Runs on CPU only.

Usage:
    CUDA_VISIBLE_DEVICES= uv run dashboard/render_seaquest.py --module experiment_seaquest_ppo \
        --ckpt runs/sq_ppo_a/best.pkl --out_prefix dashboard/media/experiment_seaquest_ppo
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

import jaxatari  # noqa: E402
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper  # noqa: E402

import seaquest_common as sc  # noqa: E402

VIEWS = {
    "real_base": ("REAL GAME · base Seaquest", None),
    "real_gravity": ("REAL GAME · gravity (held out)", ["gravity"]),
}
# extra view for world-model experiments (modules with `wm_predict` and `build_model`)
WM_VIEWS = {"wm_base": ("WORLD MODEL · imagined from base-game starts", None)}
OBJ_COLORS = {"player": (187, 187, 53), "diver": (66, 72, 200), "shark": (92, 186, 92), "sub": (170, 170, 170),
              "surface": (220, 220, 220), "torpedo": (187, 187, 53), "missile": (236, 120, 120)}
SLOT_KIND = ["player"] + ["diver"] * 4 + ["shark"] * 12 + ["sub"] * 12 + ["surface"] + ["torpedo"] + ["missile"] * 4
SCALE = 2
W = 160 * SCALE  # 320
HEAD = 80
H = HEAD + 210 * SCALE + 4  # 504, multiple of 8
BG = (24, 26, 30)
TEXT = (230, 230, 225)
MUTED = (150, 152, 158)
GOOD = (90, 200, 120)
BAD = (235, 110, 90)


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"):
        if os.path.exists(os.path.join(d, name)):
            return ImageFont.truetype(os.path.join(d, name), size)
    return ImageFont.load_default()


def load(module, ckpt):
    mod = importlib.import_module(module)
    with open(ckpt, "rb") as f:
        ck = pickle.load(f)
    greedy = bool(ck.get("greedy", False))
    return mod, ck, (mod.ACT_GREEDY if greedy else mod.ACT_SAMPLE), greedy


def play(act, params, mods, steps, key):
    """Agent in the real game for `steps` agent steps; returns rendered frames and HUD data."""
    base = jaxatari.make("seaquest", mods=mods) if mods else jaxatari.make("seaquest")
    env = ObjectCentricWrapper(AtariWrapper(base), frame_stack_size=sc.FRAME_STACK, frame_skip=sc.FRAME_SKIP)
    k_reset, k_run = jax.random.split(key)
    obs, st = env.reset(k_reset)

    def body(c, k):
        obs, st = c
        a = act(params, obs, k)
        obs2, st2, r, term, trunc, info = env.step(st, a)
        g = sc.game_state(st2)
        hud = dict(divers=g.divers_collected, rescues=g.successful_rescues, oxygen=g.oxygen, lives=g.lives,
                   score=g.score, done=info["env_done"], action=a)
        return (obs2, st2), (g, hud)

    _, (gs, hud) = jax.jit(lambda o, s, ks: jax.lax.scan(body, (o, s), ks))(obs, st, jax.random.split(k_run, steps))
    frames = jax.jit(jax.vmap(base.render))(gs)
    return np.asarray(frames), {k: np.asarray(v) for k, v in hud.items()}


def draw_oc(frame):
    """Draw a flattened object-centric frame (284,) as boxes (for imagined frames)."""
    img = Image.new("RGB", (160, 210), (0, 0, 139))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 160, 45), fill=(60, 100, 190))  # sky / surface
    d.rectangle((0, 195, 160, 210), fill=(150, 150, 150))  # sea floor
    ox = float(frame[sc.OXYGEN_IDX]) / 64.0
    d.rectangle((40, 200, 40 + int(80 * ox), 204), fill=(214, 214, 214))
    o = 0
    for start, n in ((0, 1), (8, 4), (40, 25), (240, 5)):
        for i in range(n):
            f = [float(frame[start + k * n + i]) for k in range(8)]
            x, y, w, h, active = f[:5]
            if active > 0:
                d.rectangle((x, y, x + max(w, 2), y + max(h, 1)), fill=OBJ_COLORS[SLOT_KIND[o + i]])
        o += n
    return np.asarray(img)


def imagine(mod, ck, act, steps, key):
    """The agent inside its world model, started from a real base-game history; restarts after a death."""
    model = mod.build_model(ck["cfg"])
    env = ObjectCentricWrapper(AtariWrapper(jaxatari.make("seaquest")), frame_stack_size=sc.FRAME_STACK,
                               frame_skip=sc.FRAME_SKIP)
    k_reset, k_warm, k_run = jax.random.split(key, 3)
    obs, st = env.reset(k_reset)

    def warm(c, k):  # a real start history: the agent plays the real game for a while
        obs, st = c
        obs, st, *_ = env.step(st, act(ck["ac_params"], obs, k))
        return (obs, st), obs

    _, starts = jax.lax.scan(warm, (obs, st), jax.random.split(k_warm, 300))
    starts = starts[60::20]

    def body(c, k):
        stack, i = c
        k_a, k_w = jax.random.split(k)
        a = act(ck["ac_params"], stack, k_a)
        nxt, score_d, rescue, died = mod.wm_predict(model, ck["wm_params"], stack[None], a[None], k_w)
        new = jnp.concatenate([stack[1:], nxt], 0)
        i2 = jnp.where(died[0], (i + 1) % starts.shape[0], i)
        new = jnp.where(died[0], starts[i2], new)
        hud = dict(divers=nxt[0, sc.DIVERS_IDX], rescues=rescue[0], oxygen=nxt[0, sc.OXYGEN_IDX],
                   lives=nxt[0, sc.LIVES_IDX], score=nxt[0, sc.SCORE_IDX], done=died[0], action=a)
        return (new, i2), (nxt[0], hud)

    _, (frames, hud) = jax.jit(lambda s0, ks: jax.lax.scan(body, (s0, jnp.array(0)), ks))(starts[0], jax.random.split(k_run, steps))
    hud = {k: np.asarray(v) for k, v in hud.items()}
    hud["rescues"] = np.cumsum(hud["rescues"])  # imagined rescues so far
    hud["lives"] = 3 - np.cumsum(hud["done"])  # count imagined deaths (restart from a real history)
    return np.stack([draw_oc(f) for f in np.asarray(frames)]), hud


def draw(frame, hud, t, title, subtitle, flash, fonts):
    f_small, f_bold, f_big = fonts
    img = Image.new("RGB", (W, H), BG)
    game = Image.fromarray(frame.astype(np.uint8)).resize((W, 210 * SCALE), Image.NEAREST)
    img.paste(game, (0, HEAD))
    d = ImageDraw.Draw(img, "RGBA")
    d.text((10, 6), title, fill=TEXT, font=f_bold)
    d.text((10, 26), subtitle, fill=MUTED, font=f_small)
    d.text((10, 44), f"step {t:4d} · divers {int(hud['divers'][t])}/6 · rescues {int(hud['rescues'][t])}",
           fill=TEXT, font=f_small)
    d.text((10, 60), f"oxygen {int(hud['oxygen'][t])}/64 · lives {int(hud['lives'][t])} · score {int(hud['score'][t])}",
           fill=MUTED, font=f_small)
    if flash:
        d.rectangle((0, HEAD + 190, W, HEAD + 240), fill=(0, 0, 0, 150))
        d.text((W / 2, HEAD + 215), flash[0], fill=flash[1], font=f_big, anchor="mm")
    return np.asarray(img)


def write_video(frames, hud, out, title, subtitle, fps):
    import imageio_ffmpeg

    fonts = (_font(12), _font(15, bold=True), _font(24, bold=True))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio_ffmpeg.write_frames(out, (W, H), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=8, output_params=["-crf", "28", "-movflags", "+faststart"])
    writer.send(None)
    flash, left = None, 0
    rescues = lives_lost = 0
    for t in range(len(frames)):
        if t > 0 and hud["rescues"][t] > hud["rescues"][t - 1]:
            rescues += 1
            flash, left = ("6 DIVERS RESCUED", GOOD), 20
        elif t > 0 and hud["lives"][t] < hud["lives"][t - 1]:
            lives_lost += 1
            flash, left = ("LIFE LOST", BAD), 12
        left = max(0, left - 1)
        writer.send(draw(frames[t], hud, t, title, subtitle, flash if left else None, fonts).tobytes())
    writer.close()
    return dict(rescues=int(hud["rescues"].max()), lives_lost=lives_lost, score=int(hud["score"].max()),
                game_over=bool(hud["done"].any()))


def render(module, ckpt, out_prefix, steps=900, fps=20, seed=0, label=None, views=tuple(VIEWS)):
    """Writes `<out_prefix>__<view>.mp4` for every view; returns metadata per view."""
    mod, ck, act, greedy = load(module, ckpt)
    who = "greedy agent" if greedy else "sampling agent"
    meta = dict(greedy=greedy, has_actor=True, game="seaquest", views={})
    key = jax.random.PRNGKey(seed)
    if hasattr(mod, "wm_predict") and "wm_params" in ck:
        views = tuple(views) + tuple(WM_VIEWS)
    for v in views:
        if v in WM_VIEWS:
            title = WM_VIEWS[v][0]
            frames, hud = imagine(mod, ck, act, steps, key)
        else:
            title, mods = VIEWS[v]
            frames, hud = play(act, ck["ac_params"], mods, steps, key)  # same seed for both real views
        out = f"{out_prefix}__{v}.mp4"
        stats = write_video(frames, hud, out, title, f"{label or ckpt} · {who}", fps)
        meta["views"][v] = dict(file=os.path.basename(out), title=title, stats=stats)
    meta.update(steps=steps, fps=fps)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(render(args.module, args.ckpt, args.out_prefix, args.steps, args.fps, args.seed, views=tuple(args.views)))


if __name__ == "__main__":
    main()
