"""Experiment 1: sparse-gated object-interaction world model + PPO actor trained purely in imagination.

Approach (first experiment in this repo, no earlier approach to compare against):

World model ("SparseObjectWM"), trained on unmodified Pong only:
- Input: the last K frames of JAXAtari's object-centric (OC) state, as (x, y) of player, enemy
  and ball, plus the agent's action.
- Every object has its *own* encoder over its *own* history (positions + velocities).
- Every object i then receives messages from the other tokens j (the two other objects and an
  action token). Each message m_ij is layer-normalised and multiplied by a scalar gate
  g_ij = sigmoid(f(e_i, e_j)) in [0, 1] that is recomputed at every step. An L1 penalty on the
  gates pushes them towards 0, so an object only "looks at" another object when needed
  (e.g. the ball at a paddle only around a bounce). The gates are the dependency graph that we
  visualise.
- Every object has its own head that predicts its next (x, y) displacement. The reward (point
  scored / conceded) is predicted by the ball's head, since goals are a property of the ball.
- Trained with an open-loop multi-step loss (predictions are fed back for H steps).

Actor, trained *only* inside the world model:
- PPO on a batched "imagined" environment whose step function is the world model. Imagined
  episodes start from real histories sampled from the replay buffer (unmodified Pong) and end
  when the world model predicts a point (one rally = one imagined episode) or after L steps.
- The actor never sees a real transition; real interaction is only used to collect data for the
  world model (Dreamer-style rounds: collect -> fit world model -> train actor in imagination).

Model selection uses unmodified Pong only. `lazy_enemy` is evaluated separately on the final,
selected checkpoint with `evaluate_lazy_enemy.py`.
"""

import argparse
import json
import os
import pickle
import time
from functools import partial
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import common

# ----------------------------------------------------------------------------- world model

WALL_BOUNDS = jnp.array([[0.0, 0.0], [160.0, 210.0]])


class MLP(nn.Module):
    features: Sequence[int]
    act_last: bool = False

    @nn.compact
    def __call__(self, x):
        for i, f in enumerate(self.features):
            x = nn.Dense(f)(x)
            if i < len(self.features) - 1 or self.act_last:
                x = nn.gelu(x)
        return x


class SparseObjectWM(nn.Module):
    d: int = 128

    @nn.compact
    def __call__(self, hist, action):
        """hist: (B, K, 3, 2) pixel positions, action: (B,) int.

        Returns delta (B, 3, 2) in units of 8 px, reward logits (B, 3) for rewards (-1, 0, +1)
        and gates (B, 3, 4): gates[:, i, j] = how much object i uses token j
        (tokens: player, enemy, ball, action; the diagonal is 1 = own history).
        """
        B, K = hist.shape[:2]
        p = hist / common.POS_SCALE
        v = (hist[:, 1:] - hist[:, :-1]) / 8.0
        per_obj = jnp.concatenate(
            [p.transpose(0, 2, 1, 3).reshape(B, 3, -1), v.transpose(0, 2, 1, 3).reshape(B, 3, -1)], -1
        )
        enc = [MLP([self.d, self.d], name=f"enc_{n}")(per_obj[:, i]) for i, n in enumerate(common.OBJECT_NAMES)]
        e_act = nn.Embed(common.NUM_ACTIONS, self.d, name="act_embed")(action)
        tokens = jnp.stack(enc + [e_act], axis=1)  # (B, 4, d)

        deltas, gates = [], []
        reward_logits = None
        for i, n in enumerate(common.OBJECT_NAMES):
            others = [j for j in range(4) if j != i]
            e_i = tokens[:, i]
            e_j = tokens[:, others]
            pair = jnp.concatenate([jnp.broadcast_to(e_i[:, None], e_j.shape), e_j], -1)
            g = nn.sigmoid(MLP([self.d, 1], name=f"gate_{n}")(pair)[..., 0])  # (B, 3)
            m = nn.LayerNorm(name=f"msg_ln_{n}")(MLP([self.d, self.d], name=f"msg_{n}")(pair))
            msg = jnp.sum(g[..., None] * m, axis=1)
            h = MLP([2 * self.d, self.d], act_last=True, name=f"dec_{n}")(jnp.concatenate([e_i, msg], -1))
            deltas.append(nn.Dense(2, name=f"out_{n}")(h))
            if n == "ball":
                reward_logits = nn.Dense(3, name="reward")(h)
            full = jnp.ones((B, 4)).at[:, jnp.array(others)].set(g)
            gates.append(full)
        return jnp.stack(deltas, 1), reward_logits, jnp.stack(gates, 1)


def build_model(cfg):
    """World model from a config dict (used by evaluate_lazy_enemy.py)."""
    return SparseObjectWM(d=cfg["d"])


def wm_step(model, params, hist, action):
    delta, rlog, gates = model.apply(params, hist, action)
    pred = hist[:, -1] + delta * 8.0
    return pred, delta, rlog, gates


def make_wm_loss(model, K, H, gate_coef):
    def loss_fn(params, frames, actions, rewards):
        # frames (B, K+H, 3, 2), actions (B, H), rewards (B, H)
        def body(hist, inp):
            a, target, r = inp
            pred, delta, rlog, gates = wm_step(model, params, hist, a)
            tgt_delta = (target - hist[:, -1]) / 8.0
            mse = jnp.mean(jnp.sum((delta - tgt_delta) ** 2, axis=-1), axis=0)  # per object
            ce = optax.softmax_cross_entropy_with_integer_labels(rlog, (r + 1).astype(jnp.int32)).mean()
            gate_l1 = jnp.mean(jnp.sum(gates, axis=-1) - 1.0)  # off-diagonal gates only
            new_hist = jnp.concatenate([hist[:, 1:], pred[:, None]], axis=1)
            return new_hist, (mse, ce, gate_l1)

        inputs = (actions.T, frames[:, K:].transpose(1, 0, 2, 3), rewards.T)
        _, (mse, ce, gl1) = jax.lax.scan(body, frames[:, :K], inputs)
        loss = mse.sum(-1).mean() + ce.mean() + gate_coef * gl1.mean()
        return loss, dict(mse_player=mse[:, 0].mean(), mse_enemy=mse[:, 1].mean(), mse_ball=mse[:, 2].mean(),
                          mse_step1=mse[0].sum(), reward_ce=ce.mean(), gate_mean=gl1.mean() / 3.0)

    return loss_fn


def gather_windows(frames, action, reward, env_idx, t_idx, K, H):
    """Windows frames[t-K+1 .. t+H], actions/rewards[t .. t+H-1]."""
    fo = jnp.arange(-K + 1, H + 1)
    ao = jnp.arange(0, H)
    f = frames[env_idx[:, None], t_idx[:, None] + fo[None]]
    a = action[env_idx[:, None], t_idx[:, None] + ao[None]]
    r = reward[env_idx[:, None], t_idx[:, None] + ao[None]]
    return f, a, r


def train_world_model(model, params, opt_state, tx, buf, cfg, key, log):
    arr = buf.arrays()
    n_env = arr["frames"].shape[0]
    val_envs = np.arange(n_env) % 16 == 0  # held-out envs for validation
    env_idx, t_idx = buf.valid_starts(cfg.K, cfg.H)
    is_val = val_envs[env_idx]
    tr = (jnp.asarray(env_idx[~is_val]), jnp.asarray(t_idx[~is_val]))
    va = (jnp.asarray(env_idx[is_val]), jnp.asarray(t_idx[is_val]))
    frames, action, reward = (jnp.asarray(arr[k]) for k in ("frames", "action", "reward"))
    loss_fn = make_wm_loss(model, cfg.K, cfg.H, cfg.gate_coef)

    data = (frames, action, reward, tr, va)

    @jax.jit
    def train_step(params, opt_state, key, data):
        frames, action, reward, tr, va = data
        i = jax.random.randint(key, (cfg.wm_batch,), 0, tr[0].shape[0])
        f, a, r = gather_windows(frames, action, reward, tr[0][i], tr[1][i], cfg.K, cfg.H)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, f, a, r)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    @jax.jit
    def val_step(params, key, data):
        frames, action, reward, tr, va = data
        i = jax.random.randint(key, (4096,), 0, va[0].shape[0])
        f, a, r = gather_windows(frames, action, reward, va[0][i], va[1][i], cfg.K, cfg.H)
        return loss_fn(params, f, a, r)

    t0 = time.time()
    for it in range(cfg.wm_steps):
        key, k = jax.random.split(key)
        params, opt_state, loss, aux = train_step(params, opt_state, k, data)
        if it % 1000 == 0 or it == cfg.wm_steps - 1:
            key, k = jax.random.split(key)
            vloss, vaux = val_step(params, k, data)
            log(f"  wm it {it:6d} train {float(loss):.4f} val {float(vloss):.4f} "
                + " ".join(f"{k_}={float(v):.4f}" for k_, v in vaux.items()) + f" ({time.time() - t0:.0f}s)")
    return params, opt_state, {k_: float(v) for k_, v in vaux.items()} | {"val_loss": float(vloss)}


# ----------------------------------------------------------------------------- actor-critic


class ActorCritic(nn.Module):
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        def trunk(x, name):
            for i in range(2):
                x = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), name=f"{name}_{i}")(x)
                x = nn.relu(x)
            return x

        logits = nn.Dense(common.NUM_ACTIONS, kernel_init=nn.initializers.orthogonal(0.01), name="pi")(trunk(x, "a"))
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0), name="v")(trunk(x, "c"))[..., 0]
        return logits, value


AC = ActorCritic()


def make_real_act_fn(greedy):
    def act(params, obs, key):
        x = common.policy_features(common.frames_to_pos(obs))
        logits, _ = AC.apply(params, x)
        return jnp.where(greedy, jnp.argmax(logits), jax.random.categorical(key, logits))

    return act


ACT_SAMPLE = make_real_act_fn(False)
ACT_GREEDY = make_real_act_fn(True)


def make_collect_act_fn(eps):
    def act(params, obs, key):
        k1, k2, k3 = jax.random.split(key, 3)
        a = ACT_SAMPLE(params, obs, k1)
        return jnp.where(jax.random.uniform(k2) < eps, jax.random.randint(k3, (), 0, common.NUM_ACTIONS), a)

    return act


# ----------------------------------------------------------------------------- imagination + PPO


def make_imagination(model, cfg):
    """Batched imagined environment. starts: (S, K, 3, 2) real start histories (unmodified Pong)."""

    def sample_starts(starts, key, n):
        return starts[jax.random.randint(key, (n,), 0, starts.shape[0])]

    def step(wm_params, starts, hist, t, action, key):
        pred, _, rlog, gates = wm_step(model, wm_params, hist, action)
        if cfg.round_int:
            # Ball and enemy positions are integers in Pong; the player's y is continuous.
            pred = pred.at[:, 1:].set(jnp.round(pred[:, 1:]))
        pred = jnp.clip(pred, WALL_BOUNDS[0], WALL_BOUNDS[1])
        pred = pred.at[:, :2, 0].set(hist[:, -1, :2, 0])  # paddles never move horizontally
        cls = jnp.argmax(rlog, axis=-1)
        event = cls != 1
        reward = jnp.where(event, (cls - 1).astype(jnp.float32), 0.0)
        next_hist = jnp.concatenate([hist[:, 1:], pred[:, None]], axis=1)
        t = t + 1
        trunc = jnp.logical_and(t >= cfg.imag_len, jnp.logical_not(event))
        done = jnp.logical_or(event, trunc)
        fresh = sample_starts(starts, key, hist.shape[0])
        reset_hist = jnp.where(done[:, None, None, None], fresh, next_hist)
        t = jnp.where(done, 0, t)
        return next_hist, reset_hist, t, reward, done, trunc, gates

    return sample_starts, step


def make_ppo(model, cfg, tx_ac):
    def obs_of(hist):
        return common.policy_features(hist[:, -common.FRAME_STACK:])

    def rollout(ac_params, wm_params, starts, hist, t, key, imag_step):
        def body(carry, k):
            hist, t = carry
            k1, k2 = jax.random.split(k)
            obs = obs_of(hist)
            logits, value = AC.apply(ac_params, obs)
            a = jax.random.categorical(k1, logits)
            logp = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a]
            next_hist, new_hist, t, r, done, trunc, _ = imag_step(wm_params, starts, hist, t, a, k2)
            # Bootstrap truncated imagined episodes with the value of the pre-reset state.
            _, v_next = AC.apply(ac_params, obs_of(next_hist))
            r = r + jnp.where(trunc, cfg.gamma * v_next, 0.0)
            return (new_hist, t), dict(obs=obs, a=a, logp=logp, v=value, r=r, done=done)

        (hist, t), traj = jax.lax.scan(body, (hist, t), jax.random.split(key, cfg.ppo_T))
        _, last_v = AC.apply(ac_params, obs_of(hist))
        return hist, t, traj, last_v

    def gae(traj, last_v):
        def body(carry, x):
            adv, v_next = carry
            r, v, d = x["r"], x["v"], x["done"]
            nonterm = 1.0 - d.astype(jnp.float32)
            delta = r + cfg.gamma * v_next * nonterm - v
            adv = delta + cfg.gamma * cfg.lam * nonterm * adv
            return (adv, v), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), traj, reverse=True)
        return adv, adv + traj["v"]

    def ppo_loss(ac_params, batch):
        logits, v = AC.apply(ac_params, batch["obs"])
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, batch["a"][:, None], axis=-1)[:, 0]
        ratio = jnp.exp(logp - batch["logp"])
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vloss = 0.5 * ((v - batch["ret"]) ** 2).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        return pg + cfg.vf_coef * vloss - cfg.ent_coef * ent, dict(pg=pg, v=vloss, ent=ent)

    def update(ac_params, opt_state, wm_params, starts, hist, t, key, imag_step):
        key, k_roll = jax.random.split(key)
        hist, t, traj, last_v = rollout(ac_params, wm_params, starts, hist, t, k_roll, imag_step)
        adv, ret = gae(traj, last_v)
        n = cfg.ppo_T * cfg.num_imag_envs
        data = dict(obs=traj["obs"].reshape(n, -1), a=traj["a"].reshape(n), logp=traj["logp"].reshape(n),
                    adv=adv.reshape(n), ret=ret.reshape(n))

        def epoch(carry, k):
            ac_params, opt_state = carry
            perm = jax.random.permutation(k, n).reshape(cfg.ppo_minibatches, -1)

            def mb(carry, idx):
                ac_params, opt_state = carry
                batch = jax.tree.map(lambda x: x[idx], data)
                (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(ac_params, batch)
                upd, opt_state = tx_ac.update(grads, opt_state, ac_params)
                return (optax.apply_updates(ac_params, upd), opt_state), aux

            (ac_params, opt_state), aux = jax.lax.scan(mb, (ac_params, opt_state), perm)
            return (ac_params, opt_state), aux

        (ac_params, opt_state), aux = jax.lax.scan(epoch, (ac_params, opt_state), jax.random.split(key, cfg.ppo_epochs))
        n_done = traj["done"].sum()
        stats = dict(
            reward_per_episode=traj["r"].sum() / jnp.maximum(n_done, 1),
            win_frac=(traj["r"] > 0.5).sum() / jnp.maximum((jnp.abs(traj["r"]) > 0.5).sum(), 1),
            episodes=n_done,
            **jax.tree.map(lambda x: x.mean(), aux),
        )
        return ac_params, opt_state, hist, t, stats

    return update


# ----------------------------------------------------------------------------- main loop


def evaluate_real(env, ac_params, key, n_games, max_steps, greedy):
    act = ACT_GREEDY if greedy else ACT_SAMPLE
    ps, es, over = common.evaluate(env, act, max_steps, ac_params, jax.random.split(key, n_games))
    return dict(player=float(ps.mean()), enemy=float(es.mean()), player_std=float(ps.std()),
                finished=float(over.mean()), player_per_game=[float(x) for x in ps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sparse_object_wm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--collect_envs", type=int, default=256)
    ap.add_argument("--collect_steps", type=int, default=2000)
    ap.add_argument("--collect_eps", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--gate_coef", type=float, default=0.01)
    ap.add_argument("--wm_batch", type=int, default=512)
    ap.add_argument("--wm_steps", type=int, default=10000)
    ap.add_argument("--wm_lr", type=float, default=3e-4)
    ap.add_argument("--round_int", type=int, default=1)
    ap.add_argument("--imag_len", type=int, default=256)
    ap.add_argument("--num_imag_envs", type=int, default=2048)
    ap.add_argument("--ppo_T", type=int, default=128)
    ap.add_argument("--ppo_updates", type=int, default=300)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--ppo_minibatches", type=int, default=8)
    ap.add_argument("--ppo_lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--eval_games", type=int, default=32)
    ap.add_argument("--eval_max_steps", type=int, default=10000)
    cfg = ap.parse_args()

    out_dir = os.path.join("runs", cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "log.txt"), "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    log(f"config: {vars(cfg)}")
    key = jax.random.PRNGKey(cfg.seed)
    env = common.make_env()  # unmodified Pong: the only environment used for training / selection

    model = SparseObjectWM(d=cfg.d)
    key, k = jax.random.split(key)
    wm_params = model.init(k, jnp.zeros((1, cfg.K, 3, 2)), jnp.zeros((1,), jnp.int32))
    tx_wm = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg.wm_lr, weight_decay=1e-4))
    wm_opt = tx_wm.init(wm_params)

    key, k = jax.random.split(key)
    ac_params = AC.init(k, jnp.zeros((1, common.POLICY_FEATURE_DIM)))
    tx_ac = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.ppo_lr, eps=1e-5))
    ac_opt = tx_ac.init(ac_params)
    sample_starts, imag_step = make_imagination(model, cfg)
    ppo_update = jax.jit(partial(make_ppo(model, cfg, tx_ac), imag_step=imag_step))

    buf = common.ReplayBuffer()
    history, best = [], None
    for rnd in range(cfg.rounds):
        log(f"=== round {rnd} ===")
        # 1) collect real data on unmodified Pong (random policy first, then the current actor)
        key, k = jax.random.split(key)
        keys = jax.random.split(k, cfg.collect_envs)
        if rnd == 0:
            traj = common.collect(env, common.random_act, cfg.collect_steps, None, keys)
        else:
            traj = common.collect(env, make_collect_act_fn(cfg.collect_eps), cfg.collect_steps, ac_params, keys)
        buf.add(traj)
        r = np.asarray(traj["reward"])
        log(f"collected {r.size} steps, total {buf.num_transitions}; points won {int((r > 0).sum())} lost {int((r < 0).sum())}")

        # 2) fit the world model on all real data so far
        key, k = jax.random.split(key)
        wm_params, wm_opt, wm_stats = train_world_model(model, wm_params, wm_opt, tx_wm, buf, cfg, k, log)

        # 3) train the actor purely in imagination
        env_idx, t_idx = buf.valid_starts(cfg.K, 0)
        arr = buf.arrays()
        fo = np.arange(-cfg.K + 1, 1)
        starts = jnp.asarray(arr["frames"][env_idx[:, None], t_idx[:, None] + fo[None]])
        key, k = jax.random.split(key)
        hist = sample_starts(starts, k, cfg.num_imag_envs)
        t = jnp.zeros((cfg.num_imag_envs,), jnp.int32)
        t0 = time.time()
        for u in range(cfg.ppo_updates):
            key, k = jax.random.split(key)
            ac_params, ac_opt, hist, t, st = ppo_update(ac_params, ac_opt, wm_params, starts, hist, t, k)
            if u % 25 == 0 or u == cfg.ppo_updates - 1:
                log(f"  ppo {u:4d} " + " ".join(f"{k_}={float(v):.3f}" for k_, v in st.items()) + f" ({time.time() - t0:.0f}s)")

        # 4) evaluate on real, unmodified Pong (used for model selection)
        key, k = jax.random.split(key)
        ev = {g: evaluate_real(env, ac_params, k, cfg.eval_games, cfg.eval_max_steps, g) for g in (False, True)}
        log(f"  real Pong eval sample: {ev[False]['player']:.2f} : {ev[False]['enemy']:.2f} | "
            f"greedy: {ev[True]['player']:.2f} : {ev[True]['enemy']:.2f}")
        rec = dict(round=rnd, wm=wm_stats, pong_sample=ev[False], pong_greedy=ev[True], transitions=buf.num_transitions)
        history.append(rec)
        with open(os.path.join(out_dir, f"round{rnd}.pkl"), "wb") as f:
            pickle.dump(dict(wm_params=jax.device_get(wm_params), ac_params=jax.device_get(ac_params),
                             cfg=vars(cfg)), f)
        score = max(ev[False]["player"] - ev[False]["enemy"], ev[True]["player"] - ev[True]["enemy"])
        if best is None or score > best[0]:
            best = (score, rnd, bool(ev[True]["player"] - ev[True]["enemy"] >= ev[False]["player"] - ev[False]["enemy"]))
            with open(os.path.join(out_dir, "best.pkl"), "wb") as f:
                pickle.dump(dict(wm_params=jax.device_get(wm_params), ac_params=jax.device_get(ac_params),
                                 cfg=vars(cfg), round=rnd, greedy=best[2]), f)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(dict(config=vars(cfg), rounds=history,
                           selected=dict(round=best[1], greedy=best[2], pong_score_diff=best[0])), f, indent=1)
    log(f"selected round {best[1]} (greedy={best[2]}) by unmodified-Pong score diff {best[0]:.2f}")


if __name__ == "__main__":
    main()
