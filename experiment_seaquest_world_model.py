"""Experiment W1 (Seaquest): agent trained entirely inside a learned object-centric world model.

The Pong method (a world model fitted on real data, and an actor trained purely in imagination)
applied to Seaquest. The model-free agents S1-S5 were trained on the real game. Here the real base game
is used **only** to collect data for the world model and to evaluate / select checkpoints. The
agent never learns from a real transition.

World model (`SeaquestWM`): an object-centric transformer over one token per object slot (player,
4 divers, 12 sharks, 12 enemy subs, surface sub, torpedo, 4 enemy missiles), one global token
(oxygen, divers carried, lives) and one action token. Each token holds its slot's last 4 frames.
For every slot it predicts whether the object exists in the next frame, and its next position (a
displacement for moving objects, an absolute position for newly spawned ones), orientation and
visual id. The global token predicts the next oxygen, divers carried and lives, the score gain, a
rescue event (surfacing with 6 divers) and a lost life. In imagination, spawns (inactive -> active)
are sampled from the predicted probability. Everything else is decoded deterministically, and
positions are rounded to integers as in the game.

Actor: S3's object-attention agent (`experiment_seaquest_object_attention.py`) on the (imagined)
object-centric observation. PPO on 2048 imagined games. An imagined episode starts from a real
4-frame history of the replay buffer and ends at a predicted lost life, or after `imag_len` steps
(value bootstrap). The reward is computed from the world model's predicted *state* (a first version
that used the event heads directly was exploited: many imagined rescues, none in the real game):
kill score gain (capped at 90) / 100 + 2 per diver above the highest count reached in the imagined
episode + 10 per rescue (6 divers drop to 5 at the surface, without a death; the game then counts
the divers down while converting oxygen into points, and during this countdown no pickups or second
rescue are counted) + potential shaping towards the surface with 6
divers or low oxygen.

Data (Dreamer-style rounds): real base games with the start-state diver curriculum of S5 (at game
start and after a lost life, divers carried := k ~ U{0..6} with probability `p_gift`). Transitions
that a gift changes are excluded from world-model training, so the model never learns "magic" diver
changes. Imagined games can also start with gifted divers (`p_imag_gift`), a start-state
intervention inside the world model. Round 0 collects with random actions, later rounds with the
current actor (10 % random actions).

Selection uses base-game rescues from the normal start. The held-out `gravity` mod is only
evaluated afterwards on the selected checkpoint with `evaluate_gravity.py`.
"""

import argparse
import json
import os
import pickle
import time
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import experiment_seaquest_object_attention as s3
import experiment_seaquest_ppo as s1
import seaquest_common as sc

AC = s3.AC
ACT_SAMPLE, ACT_GREEDY = s3.ACT_SAMPLE, s3.ACT_GREEDY
FD = sc.FRAME_DIM
GROUPS = [(0, 1), (8, 4), (40, 25), (240, 5)]  # (start, n) of player, divers, enemies, projectiles
N_OBJ = 35
ORIENTS = jnp.array([0.0, 90.0, 270.0])


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def slot_fields(frames):
    """(..., F, 284) -> dict of (..., F, 35) arrays (x, y, w, h, active, vid, state, orient)."""
    names = ("x", "y", "w", "h", "active", "vid", "state", "orient")
    out = {}
    for i, k in enumerate(names):
        out[k] = jnp.concatenate([frames[..., s + i * n:s + (i + 1) * n] for s, n in GROUPS], axis=-1)
    return out


def set_slot_fields(frame, fields):
    """Write (..., 35) slot fields back into a flattened (..., 284) frame."""
    names = ("x", "y", "w", "h", "active", "vid", "state", "orient")
    for i, k in enumerate(names):
        o = 0
        for s, n in GROUPS:
            frame = frame.at[..., s + i * n:s + (i + 1) * n].set(fields[k][..., o:o + n])
            o += n
    return frame


# ----------------------------------------------------------------------------- world model


class WMBlock(nn.Module):
    d: int
    heads: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.heads, qkv_features=self.d)(h, h)
        h = nn.LayerNorm()(x)
        return x + nn.Dense(self.d)(nn.gelu(nn.Dense(2 * self.d)(h)))


N_DIV_CLS, N_LIVES_CLS = 8, 5  # divers -1..6, lives 0..4
OBJ_OUT = 1 + 2 + 2 + 3 + 6  # active, delta (x, y), absolute (x, y), orientation (3), visual id (6)
GLB_OUT = 1 + N_DIV_CLS + N_LIVES_CLS + 3  # oxygen, divers, lives, score (symlog), rescue, life lost


class SeaquestWM(nn.Module):
    d: int = 128
    heads: int = 4
    layers: int = 4

    @nn.compact
    def __call__(self, stack, action):
        """stack: (B, 4, 284) raw frames, action: (B,) -> (obj (B, 35, OBJ_OUT), glb (B, GLB_OUT))."""
        f = slot_fields(stack)  # (B, 4, 35)
        act = f["active"]
        r = jnp.deg2rad(f["orient"])
        per = jnp.stack([f["x"] / 160.0 * act, f["y"] / 210.0 * act, act, jnp.sin(r) * act, jnp.cos(r) * act,
                         f["vid"] / 5.0 * act], axis=-1)  # (B, 4, 35, 6)
        per = per.transpose(0, 2, 1, 3).reshape(stack.shape[0], N_OBJ, -1)  # (B, 35, 24)
        mov = act[:, -1] * act[:, -2]
        vel = jnp.stack([jnp.clip(f["x"][:, -1] - f["x"][:, -2], -8, 8) / 8.0 * mov,
                         jnp.clip(f["y"][:, -1] - f["y"][:, -2], -8, 8) / 8.0 * mov], -1)
        obj = nn.Dense(self.d, name="emb_obj")(jnp.concatenate([per, vel], -1))
        glb_in = jnp.concatenate([stack[..., sc.OXYGEN_IDX] / 64.0, stack[..., sc.DIVERS_IDX] / 6.0,
                                  stack[..., sc.LIVES_IDX] / 4.0], -1)  # (B, 12)
        glb = nn.Dense(self.d, name="emb_glb")(glb_in)[:, None]
        a = nn.Embed(sc.NUM_ACTIONS, self.d, name="emb_act")(action)[:, None]
        x = jnp.concatenate([obj, glb, a], axis=1)
        x = x + self.param("slot_emb", nn.initializers.normal(0.02), (N_OBJ + 2, self.d))
        for _ in range(self.layers):
            x = WMBlock(self.d, self.heads)(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(OBJ_OUT, name="head_obj")(x[:, :N_OBJ]), nn.Dense(GLB_OUT, name="head_glb")(x[:, N_OBJ])


def targets(stack, nxt):
    """Supervised targets for the transition stack[-1] -> nxt (both raw)."""
    c, n = slot_fields(stack[:, -1:])  , slot_fields(nxt[:, None])
    c = {k: v[:, 0] for k, v in c.items()}
    n = {k: v[:, 0] for k, v in n.items()}
    orient_cls = jnp.argmin(jnp.abs(n["orient"][..., None] - ORIENTS), -1)
    return c, n, orient_cls


def wm_loss(model, params, stack, action, nxt, score_d, rescue, life_lost, cfg):
    obj, glb = model.apply(params, stack, action)
    c, n, orient_cls = targets(stack, nxt)
    act_c, act_n = c["active"] > 0, n["active"] > 0
    l_active = optax.sigmoid_binary_cross_entropy(obj[..., 0], act_n.astype(jnp.float32)).mean()
    moving, spawn = act_c & act_n, (~act_c) & act_n
    d_tgt = jnp.stack([(n["x"] - c["x"]) / 8.0, (n["y"] - c["y"]) / 8.0], -1)
    a_tgt = jnp.stack([n["x"] / 160.0, n["y"] / 210.0], -1)
    se_d = jnp.sum((obj[..., 1:3] - d_tgt) ** 2, -1)
    se_a = jnp.sum((obj[..., 3:5] - a_tgt) ** 2, -1) * 100.0
    l_pos = (jnp.sum(se_d * moving) + jnp.sum(se_a * spawn)) / jnp.maximum(jnp.sum(act_n), 1.0)
    ce_o = optax.softmax_cross_entropy_with_integer_labels(obj[..., 5:8], orient_cls)
    ce_v = optax.softmax_cross_entropy_with_integer_labels(obj[..., 8:14], jnp.clip(n["vid"], 0, 5).astype(jnp.int32))
    l_cat = jnp.sum((ce_o + ce_v) * act_n) / jnp.maximum(jnp.sum(act_n), 1.0)
    o = 1
    oxy = glb[:, 0]
    div_l, liv_l = glb[:, o:o + N_DIV_CLS], glb[:, o + N_DIV_CLS:o + N_DIV_CLS + N_LIVES_CLS]
    sc_p, res_p, life_p = glb[:, -3], glb[:, -2], glb[:, -1]
    l_oxy = jnp.mean((oxy - nxt[:, sc.OXYGEN_IDX] / 64.0) ** 2) * 100.0
    l_div = optax.softmax_cross_entropy_with_integer_labels(
        div_l, jnp.clip(nxt[:, sc.DIVERS_IDX] + 1, 0, N_DIV_CLS - 1).astype(jnp.int32)).mean()
    l_liv = optax.softmax_cross_entropy_with_integer_labels(
        liv_l, jnp.clip(nxt[:, sc.LIVES_IDX], 0, N_LIVES_CLS - 1).astype(jnp.int32)).mean()
    l_sc = jnp.mean((sc_p - symlog(score_d)) ** 2)
    w_res = jnp.where(rescue > 0, cfg.event_weight, 1.0)
    w_life = jnp.where(life_lost > 0, cfg.event_weight, 1.0)
    l_res = jnp.mean(w_res * optax.sigmoid_binary_cross_entropy(res_p, rescue))
    l_life = jnp.mean(w_life * optax.sigmoid_binary_cross_entropy(life_p, life_lost))
    loss = l_active + l_pos + l_cat + l_oxy + l_div + l_liv + l_sc + l_res + l_life
    # accuracy metrics
    pred_act = obj[..., 0] > 0
    pos_err_px = jnp.sqrt(jnp.sum(jnp.where(moving[..., None], (obj[..., 1:3] - d_tgt) * 8.0, 0.0) ** 2)
                          / jnp.maximum(jnp.sum(moving), 1.0))
    grp = {"player": slice(0, 1), "divers": slice(1, 5), "enemies": slice(5, 30), "proj": slice(30, 35)}
    err2 = jnp.sum((obj[..., 1:3] - d_tgt) ** 2, -1) * 64.0
    per_grp = {f"rmse_{g}": jnp.sqrt(jnp.sum((err2 * moving)[:, sl]) / jnp.maximum(jnp.sum(moving[:, sl]), 1.0))
               for g, sl in grp.items()}
    div_changed = nxt[:, sc.DIVERS_IDX] != stack[:, -1, sc.DIVERS_IDX]
    aux_extra = dict(div_change_acc=jnp.sum((jnp.argmax(div_l, -1) - 1 == nxt[:, sc.DIVERS_IDX]) & div_changed)
                     / jnp.maximum(jnp.sum(div_changed), 1.0), **per_grp)
    aux = dict(loss=loss, active=l_active, pos=l_pos, cat=l_cat, oxy=l_oxy, div=l_div, lives=l_liv, score=l_sc,
               rescue=l_res, life=l_life, act_acc=jnp.mean(pred_act == act_n),
               spawn_recall=jnp.sum(pred_act & spawn) / jnp.maximum(jnp.sum(spawn), 1.0),
               moving_rmse_px=pos_err_px,
               div_acc=jnp.mean(jnp.argmax(div_l, -1) - 1 == nxt[:, sc.DIVERS_IDX]),
               rescue_recall=jnp.sum((res_p > 0) & (rescue > 0)) / jnp.maximum(jnp.sum(rescue), 1.0),
               life_recall=jnp.sum((life_p > 0) & (life_lost > 0)) / jnp.maximum(jnp.sum(life_lost), 1.0),
               life_false=jnp.sum((life_p > 0) & (life_lost == 0)) / jnp.maximum(jnp.sum(life_lost == 0), 1.0),
               **aux_extra)
    return loss, aux


def wm_predict(model, params, stack, action, key):
    """Sample the next frame and events. Returns (next_frame (B, 284), score_delta, rescue, life_lost)."""
    obj, glb = model.apply(params, stack, action)
    cur = stack[:, -1]
    c = {k: v[:, 0] for k, v in slot_fields(cur[:, None]).items()}
    p_act = jax.nn.sigmoid(obj[..., 0])
    act_c = c["active"] > 0
    spawn = jax.random.uniform(key, p_act.shape) < p_act
    act_n = jnp.where(act_c, p_act > 0.5, spawn)
    act_n = act_n.at[:, 0].set(True)  # the player slot is always active
    x = jnp.where(act_c, c["x"] + jnp.round(obj[..., 1] * 8.0), jnp.round(obj[..., 3] * 160.0))
    y = jnp.where(act_c, c["y"] + jnp.round(obj[..., 2] * 8.0), jnp.round(obj[..., 4] * 210.0))
    x = jnp.clip(x, 0, 160) * act_n
    y = jnp.clip(y, 0, 210) * act_n
    fields = dict(x=x, y=y, w=c["w"], h=c["h"], active=act_n.astype(jnp.float32),
                  vid=jnp.argmax(obj[..., 8:14], -1).astype(jnp.float32) * act_n,
                  state=jnp.zeros_like(x), orient=ORIENTS[jnp.argmax(obj[..., 5:8], -1)] * act_n)
    nxt = set_slot_fields(cur, fields)
    o = 1
    score_d = jnp.maximum(jnp.round(symexp(glb[:, -3])), 0.0)
    nxt = nxt.at[:, sc.OXYGEN_IDX].set(jnp.clip(jnp.round(glb[:, 0] * 64.0), 0, 64))
    nxt = nxt.at[:, sc.DIVERS_IDX].set(jnp.argmax(glb[:, o:o + N_DIV_CLS], -1) - 1.0)
    nxt = nxt.at[:, sc.LIVES_IDX].set(jnp.argmax(glb[:, o + N_DIV_CLS:o + N_DIV_CLS + N_LIVES_CLS], -1) * 1.0)
    nxt = nxt.at[:, sc.SCORE_IDX].set(cur[:, sc.SCORE_IDX] + score_d)
    return nxt, score_d, glb[:, -2] > 0, glb[:, -1] > 0


# ----------------------------------------------------------------------------- real data


def make_collect(env, cfg, act_fn, steps):
    """Collect `steps` agent steps in each env (base game + diver curriculum). jitted."""

    def gift(st, key, apply):
        k1, k2 = jax.random.split(key)
        use = apply & (jax.random.uniform(k1) < cfg.p_gift)
        g = st.atari_state.env_state
        g = g.replace(divers_collected=jnp.where(use, jax.random.randint(k2, (), 0, 7), g.divers_collected)
                      .astype(g.divers_collected.dtype))
        return st.replace(atari_state=st.atari_state.replace(env_state=g)), use

    def run_one(params, key):
        k_reset, k_g, k_run = jax.random.split(key, 3)
        obs, st = env.reset(k_reset)
        st, g_init = gift(st, k_g, jnp.array(True))

        def body(carry, k):
            obs, st = carry
            k_a, k_gift = jax.random.split(k)
            g0 = sc.game_state(st)
            a = act_fn(params, obs, k_a)
            obs2, st2, r, term, trunc, info = env.step(st, a)
            g1 = sc.game_state(st2)
            done = info["env_done"]
            lost = (g1.lives < g0.lives) & ~done  # a new life starts (end of the death animation)
            died = (g1.death_counter > 0) & (g0.death_counter == 0) & ~done  # collision / no oxygen
            st2, gifted = gift(st2, k_gift, lost | done)
            out = dict(frame=obs[-1], action=a, score_d=jnp.where(done, 0, jnp.maximum(g1.score - g0.score, 0)),
                       rescue=(g1.successful_rescues > g0.successful_rescues) & ~done, life_lost=died,
                       in_death=g0.death_counter > 0, done=done, gift=gifted, next=obs2[-1])
            return (obs2, st2), out

        _, tr = jax.lax.scan(body, (obs, st), jax.random.split(k_run, steps))
        tr["gift_init"] = g_init
        return tr

    return jax.jit(jax.vmap(run_one, in_axes=(None, 0)))


class Buffer:
    """Real transitions, (env, t) layout per chunk. invalid[s]: transition s -> s+1 must not be used."""

    def __init__(self, max_chunks):
        self.chunks, self.max_chunks = [], max_chunks

    def add(self, tr):
        tr = jax.device_get(tr)
        frames = np.concatenate([tr["frame"], tr["next"][:, -1:]], 1).astype(np.float32)
        inv = tr["done"] | tr["in_death"]  # the death animation is never needed in imagination
        inv[:, 1:] |= tr["gift"][:, :-1]  # a gift after step s changes transition s+1
        inv[:, 0] |= tr["gift_init"]
        self.chunks.append(dict(frames=frames, action=tr["action"].astype(np.int32),
                                score_d=tr["score_d"].astype(np.float32), rescue=tr["rescue"].astype(np.float32),
                                life_lost=tr["life_lost"].astype(np.float32), invalid=inv.astype(bool)))
        self.chunks = self.chunks[-self.max_chunks:]

    def arrays(self):
        return {k: np.concatenate([c[k] for c in self.chunks], 0) for k in self.chunks[0]}

    @staticmethod
    def valid(inv, hist, horizon):
        """(env, t) with frames[t-hist+1 .. t] a clean history and transitions t .. t+horizon-1 usable."""
        n_env, T = inv.shape
        cs = np.concatenate([np.zeros((n_env, 1), np.int64), np.cumsum(inv, 1)], 1)
        ts = np.arange(hist - 1, T - horizon + 1)
        bad = cs[:, ts + horizon] - cs[:, ts - hist + 1]
        e, i = np.nonzero(bad == 0)
        return e.astype(np.int32), ts[i].astype(np.int32)


def windows(arr, e, t, K=4):
    fo = np.arange(-K + 1, 1)
    return arr["frames"][e[:, None], t[:, None] + fo[None]]


# ----------------------------------------------------------------------------- training


def train_wm(model, params, opt, tx, buf, cfg, key, log):
    arr = buf.arrays()
    e, t = Buffer.valid(arr["invalid"], 4, 1)
    val = (e % 16) == 0
    D = {k: jnp.asarray(v) for k, v in arr.items()}
    tr_idx = (jnp.asarray(e[~val]), jnp.asarray(t[~val]))
    va_idx = (jnp.asarray(e[val]), jnp.asarray(t[val]))
    fo = jnp.arange(-3, 1)

    def batch(D, idx, key, n):
        i = jax.random.randint(key, (n,), 0, idx[0].shape[0])
        ee, tt = idx[0][i], idx[1][i]
        stack = D["frames"][ee[:, None], tt[:, None] + fo[None]]
        return (stack, D["action"][ee, tt], D["frames"][ee, tt + 1], D["score_d"][ee, tt], D["rescue"][ee, tt],
                D["life_lost"][ee, tt])

    @jax.jit
    def step(params, opt, key, D, idx):
        b = batch(D, idx, key, cfg.wm_batch)
        (loss, aux), g = jax.value_and_grad(lambda p: wm_loss(model, p, *b, cfg), has_aux=True)(params)
        upd, opt = tx.update(g, opt, params)
        return optax.apply_updates(params, upd), opt, aux

    @jax.jit
    def val_step(params, key, D, idx):
        return wm_loss(model, params, *batch(D, idx, key, 8192), cfg)[1]

    t0 = time.time()
    for it in range(cfg.wm_steps):
        key, k = jax.random.split(key)
        params, opt, aux = step(params, opt, k, D, tr_idx)
        if it % 2500 == 0 or it == cfg.wm_steps - 1:
            key, k = jax.random.split(key)
            v = val_step(params, k, D, va_idx)
            log(f"  wm it {it:6d} train {float(aux['loss']):.4f} val {float(v['loss']):.4f} "
                + " ".join(f"{k_}={float(x):.3f}" for k_, x in v.items() if k_ != "loss") + f" ({time.time() - t0:.0f}s)")
    return params, opt, {k_: float(x) for k_, x in v.items()}


def potential(cfg, frame):
    h = (141.0 - frame[:, sc.PLAYER_Y_IDX]) / 95.0
    return h * (cfg.c_surf * (frame[:, sc.DIVERS_IDX] >= 6) + cfg.c_ox * (frame[:, sc.OXYGEN_IDX] < cfg.ox_low))


def diver_bookkeeping(cur, nxt, lost, ist):
    """Rescue and diver-pickup counting from (predicted) consecutive frames.

    In Seaquest a rescue is the step where 6 divers drop to 5 at the surface; afterwards the game counts
    the divers down one per step while it converts oxygen into points. Pickups are only counted above
    the highest number of divers reached so far in the episode (`m`), so an imagined lose-and-regain
    oscillation pays nothing. During the post-rescue countdown (`count`) no pickups or second rescue
    are counted. Returns (rescue, picked, m, count).
    """
    d0 = jnp.clip(cur[:, sc.DIVERS_IDX], 0, 6)
    d1 = jnp.clip(nxt[:, sc.DIVERS_IDX], 0, 6)
    count = ist["count"]
    rescue = (d0 >= 6) & (d1 <= 5) & (nxt[:, sc.PLAYER_Y_IDX] <= 50) & ~lost & ~count
    in_count = count | rescue
    picked = jnp.where(in_count, 0.0, jnp.maximum(d1 - ist["m"], 0.0)) * ~lost
    m = jnp.where(in_count, d1, jnp.maximum(ist["m"], d1))
    return rescue, picked, m, in_count & (d1 > 0)


def make_imagination(model, cfg):
    def sample_start(starts, key, n):
        k1, k2, k3 = jax.random.split(key, 3)
        st = starts[jax.random.randint(k1, (n,), 0, starts.shape[0])]
        gift = jax.random.uniform(k2, (n,)) < cfg.p_imag_gift
        k = jax.random.randint(k3, (n,), 0, 7).astype(jnp.float32)
        return st.at[:, :, sc.DIVERS_IDX].set(jnp.where(gift[:, None], k[:, None], st[:, :, sc.DIVERS_IDX]))

    def init_state(stack):
        return dict(t=jnp.zeros(stack.shape[0], jnp.int32), m=jnp.clip(stack[:, -1, sc.DIVERS_IDX], 0, 6),
                    count=jnp.zeros(stack.shape[0], bool))

    def step(wm_params, starts, stack, ist, action, key):
        k_wm, k_reset = jax.random.split(key)
        nxt, score_d, _, lost = wm_predict(model, wm_params, stack, action, k_wm)
        cur = stack[:, -1]
        rescue, picked, m, count = diver_bookkeeping(cur, nxt, lost, ist)
        score_d = jnp.minimum(score_d, 90.0)
        t = ist["t"]
        t = t + 1
        trunc = (t >= cfg.imag_len) & ~lost
        done = lost | trunc
        phi1 = jnp.where(lost, 0.0, potential(cfg, nxt))
        r = (score_d / 100.0 + cfg.diver_bonus * picked + cfg.rescue_bonus * rescue
             + cfg.gamma * phi1 - potential(cfg, cur))
        next_stack = jnp.concatenate([stack[:, 1:], nxt[:, None]], 1)
        reset = jnp.where(done[:, None, None], sample_start(starts, k_reset, stack.shape[0]), next_stack)
        fresh = init_state(reset)
        ist = dict(t=jnp.where(done, 0, t), m=jnp.where(done, fresh["m"], m), count=jnp.where(done, False, count))
        return next_stack, reset, ist, r, done, trunc, (rescue, picked)

    return sample_start, init_state, step


def make_ppo(cfg, tx, imag_step):
    apply = s3.apply_fn

    def rollout(ac, wm, starts, stack, t, key):
        def body(carry, k):
            stack, t = carry
            k1, k2 = jax.random.split(k)
            logits, v = apply(ac, stack)
            a = jax.random.categorical(k1, logits)
            logp = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a]
            nstack, reset, t, r, done, trunc, (res, picked) = imag_step(wm, starts, stack, t, a, k2)
            _, v_next = apply(ac, nstack)
            r = r + jnp.where(trunc, cfg.gamma * v_next, 0.0)
            return (reset, t), dict(obs=stack, a=a, logp=logp, v=v, r=r, done=done, res=res, picked=picked)

        (stack, t), tr = jax.lax.scan(body, (stack, t), jax.random.split(key, cfg.ppo_T))
        _, last_v = apply(ac, stack)
        return stack, t, tr, last_v

    def gae(tr, last_v):
        def body(carry, x):
            adv, v_next = carry
            nt = 1.0 - x["done"].astype(jnp.float32)
            delta = x["r"] + cfg.gamma * v_next * nt - x["v"]
            adv = delta + cfg.gamma * cfg.lam * nt * adv
            return (adv, x["v"]), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), tr, reverse=True)
        return adv, adv + tr["v"]

    def loss_fn(ac, b):
        logits, v = apply(ac, b["obs"])
        lp_all = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp_all, b["a"][:, None], -1)[:, 0]
        ratio = jnp.exp(lp - b["logp"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vl = 0.5 * ((v - b["ret"]) ** 2).mean()
        ent = -(jnp.exp(lp_all) * lp_all).sum(-1).mean()
        return pg + cfg.vf_coef * vl - cfg.ent_coef * ent, dict(pg=pg, v=vl, ent=ent)

    @jax.jit
    def update(ac, opt, wm, starts, stack, t, key):
        key, kr = jax.random.split(key)
        stack, t, tr, last_v = rollout(ac, wm, starts, stack, t, kr)
        adv, ret = gae(tr, last_v)
        n = cfg.ppo_T * cfg.num_imag_envs
        data = dict(obs=tr["obs"].reshape(n, 4, FD), a=tr["a"].reshape(n), logp=tr["logp"].reshape(n),
                    adv=adv.reshape(n), ret=ret.reshape(n))

        def epoch(carry, k):
            ac, opt = carry
            perm = jax.random.permutation(k, n).reshape(cfg.ppo_minibatches, -1)

            def mb(carry, idx):
                ac, opt = carry
                b = jax.tree.map(lambda x: x[idx], data)
                (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(ac, b)
                upd, opt = tx.update(g, opt, ac)
                return (optax.apply_updates(ac, upd), opt), aux

            return jax.lax.scan(mb, (ac, opt), perm)

        (ac, opt), aux = jax.lax.scan(epoch, (ac, opt), jax.random.split(key, cfg.ppo_epochs))
        stats = dict(r_per_env=tr["r"].sum() / cfg.num_imag_envs, rescues=tr["res"].sum() / cfg.num_imag_envs,
                     divers=tr["picked"].sum() / cfg.num_imag_envs, episodes=tr["done"].sum(),
                     **jax.tree.map(lambda x: x.mean(), aux))
        return ac, opt, stack, t, stats

    return update


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="seaquest_world_model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--collect_envs", type=int, default=512)
    ap.add_argument("--collect_steps", type=int, default=2000)
    ap.add_argument("--collect_eps", type=float, default=0.1)
    ap.add_argument("--p_gift", type=float, default=0.5)
    ap.add_argument("--max_chunks", type=int, default=5, help="replay buffer: keep the last N collections")
    ap.add_argument("--wm_d", type=int, default=128)
    ap.add_argument("--wm_layers", type=int, default=4)
    ap.add_argument("--wm_batch", type=int, default=512)
    ap.add_argument("--wm_steps", type=int, default=20000)
    ap.add_argument("--wm_lr", type=float, default=3e-4)
    ap.add_argument("--event_weight", type=float, default=5.0)
    ap.add_argument("--wm_only", type=int, default=0, help="only collect round-0 data and fit the world model")
    ap.add_argument("--imag_len", type=int, default=256)
    ap.add_argument("--p_imag_gift", type=float, default=0.5)
    ap.add_argument("--n_starts", type=int, default=200000, help="real start histories for imagination")
    ap.add_argument("--num_imag_envs", type=int, default=2048)
    ap.add_argument("--ppo_T", type=int, default=64)
    ap.add_argument("--ppo_updates", type=int, default=300)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--ppo_minibatches", type=int, default=8)
    ap.add_argument("--ppo_lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--diver_bonus", type=float, default=2.0)
    ap.add_argument("--rescue_bonus", type=float, default=10.0)
    ap.add_argument("--c_surf", type=float, default=2.0)
    ap.add_argument("--c_ox", type=float, default=1.0)
    ap.add_argument("--ox_low", type=float, default=16.0)
    ap.add_argument("--eval_games", type=int, default=32)
    ap.add_argument("--eval_max_steps", type=int, default=10000)
    return ap


def build_model(cfg):
    return SeaquestWM(d=cfg["wm_d"], layers=cfg["wm_layers"])


def main():
    cfg = get_parser().parse_args()
    out_dir = os.path.join("runs", cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "log.txt"), "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    cfg_d = vars(cfg) | dict(experiment="experiment_seaquest_world_model", game="seaquest")
    log(f"config: {cfg_d}")
    env = sc.make_env()  # base game only
    key = jax.random.PRNGKey(cfg.seed)
    model = SeaquestWM(d=cfg.wm_d, layers=cfg.wm_layers)
    key, k1, k2 = jax.random.split(key, 3)
    wm_params = model.init(k1, jnp.zeros((1, 4, FD)), jnp.zeros((1,), jnp.int32))
    tx_wm = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg.wm_lr, weight_decay=1e-4))
    wm_opt = tx_wm.init(wm_params)
    ac_params = s3.init_fn(k2)
    tx_ac = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.ppo_lr, eps=1e-5))
    ac_opt = tx_ac.init(ac_params)
    sample_start, init_state, imag_step = make_imagination(model, cfg)
    ppo_update = make_ppo(cfg, tx_ac, imag_step)

    def collect_act(params, obs, key):
        k1, k2, k3 = jax.random.split(key, 3)
        a = ACT_SAMPLE(params, obs, k1)
        return jnp.where(jax.random.uniform(k2) < cfg.collect_eps, jax.random.randint(k3, (), 0, sc.NUM_ACTIONS), a)

    collect_random = make_collect(env, cfg, lambda p, o, k: jax.random.randint(k, (), 0, sc.NUM_ACTIONS),
                                  cfg.collect_steps)
    collect_actor = make_collect(env, cfg, collect_act, cfg.collect_steps)
    buf = Buffer(cfg.max_chunks)
    history, best = [], None
    for rnd in range(cfg.rounds):
        log(f"=== collect round {rnd} ===")
        key, k = jax.random.split(key)
        keys = jax.random.split(k, cfg.collect_envs)
        tr = (collect_random if rnd == 0 else collect_actor)(ac_params, keys)
        buf.add(tr)
        log(f"collected {int(np.prod(tr['action'].shape))} real steps; rescues {int(np.asarray(tr['rescue']).sum())}, "
            f"lives lost {int(np.asarray(tr['life_lost']).sum())}, gifts {int(np.asarray(tr['gift']).sum())}")
        del tr
        key, k = jax.random.split(key)
        wm_params, wm_opt, wm_stats = train_wm(model, wm_params, wm_opt, tx_wm, buf, cfg, k, log)
        if cfg.wm_only:
            with open(os.path.join(out_dir, "wm_only.pkl"), "wb") as f:
                pickle.dump(dict(wm_params=jax.device_get(wm_params), cfg=cfg_d), f)
            with open(os.path.join(out_dir, "wm_only.json"), "w") as f:
                json.dump(dict(config=cfg_d, wm=wm_stats), f, indent=1)
            return
        arr = buf.arrays()
        e, t = Buffer.valid(arr["invalid"], 4, 0)
        key, k = jax.random.split(key)
        sel = np.asarray(jax.random.choice(k, len(e), (min(cfg.n_starts, len(e)),), replace=False))
        starts = jnp.asarray(windows(arr, e[sel], t[sel]))
        del arr
        key, k = jax.random.split(key)
        stack = sample_start(starts, k, cfg.num_imag_envs)
        tt = init_state(stack)
        t0 = time.time()
        for u in range(cfg.ppo_updates):
            key, k = jax.random.split(key)
            ac_params, ac_opt, stack, tt, st = ppo_update(ac_params, ac_opt, wm_params, starts, stack, tt, k)
            if u % 50 == 0 or u == cfg.ppo_updates - 1:
                log(f"  ppo {u:4d} " + " ".join(f"{k_}={float(v):.3f}" for k_, v in st.items()) + f" ({time.time() - t0:.0f}s)")
        del starts
        key, k = jax.random.split(key)
        ev = {g: s1.evaluate(env, ac_params, k, cfg.eval_games, cfg.eval_max_steps, g, (ACT_SAMPLE, ACT_GREEDY))
              for g in (False, True)}
        log(f"=== round {rnd} === update {rnd + 1}: base-game eval sample rescues {ev[False]['rescues']:.2f} "
            f"divers {ev[False]['divers']:.1f} score {ev[False]['score']:.0f} | greedy rescues {ev[True]['rescues']:.2f} "
            f"divers {ev[True]['divers']:.1f} score {ev[True]['score']:.0f}")
        history.append(dict(round=rnd, update=rnd + 1, wm=wm_stats, base_sample=ev[False], base_greedy=ev[True],
                            imag=jax.tree.map(float, st), transitions=len(buf.chunks) * cfg.collect_envs * cfg.collect_steps))
        ck = dict(ac_params=jax.device_get(ac_params), wm_params=jax.device_get(wm_params), cfg=cfg_d)
        with open(os.path.join(out_dir, f"round{rnd}.pkl"), "wb") as f:
            pickle.dump(ck, f)
        for g in (False, True):
            crit = (ev[g]["rescues"], ev[g]["score"])
            if best is None or crit > best[0]:
                best = (crit, rnd, g)
                with open(os.path.join(out_dir, "best.pkl"), "wb") as f:
                    pickle.dump(ck | dict(round=rnd, greedy=g), f)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(dict(config=cfg_d, rounds=history, selected=dict(round=best[1], greedy=best[2],
                                                                        base_rescues=best[0][0], base_score=best[0][1])), f, indent=1)
    log(f"selected round {best[1]} (greedy={best[2]}) by base-game rescues {best[0][0]:.2f}, score {best[0][1]:.0f}")


if __name__ == "__main__":
    main()
