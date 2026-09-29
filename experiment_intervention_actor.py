"""Experiment 2: L0-gated object world model + actor trained in imagination under object interventions.

Motivation (from experiment 1, `experiment_sparse_object_wm.md`): the imagination-trained actor won
unmodified Pong 21 : 2, but lost with `lazy_enemy` (12.6 : 21). Two causes were visible, both
measurable on unmodified Pong:
  1. The world model is not object-contained: the ball's prediction relies on the enemy's position
     as a shortcut (in Pong the enemy tracks the ball), even far from any bounce (counterfactual
     leakage `ball<-enemy_far` = 4.3 px, see `common.leakage`).
  2. The actor has only ever seen an enemy that tracks the ball, so any other enemy behaviour is
     out of distribution for the policy.

Changes relative to experiment 1:
- World model gates are *hard-concrete (L0) gates* (Louizos et al. 2018) instead of sigmoid gates
  with an L1 penalty: during training each gate is a stochastic, exactly-0-or-1-capable variable,
  and an expected-L0 penalty counts how many other tokens each object uses. A message only
  survives if it is needed; a shortcut that saves little loss is closed.
- The actor is trained in imagination under random **interventions on the enemy object's
  mechanism**, which the factored world model makes possible: in each imagined rally the enemy's
  next position either comes from the world model (no intervention), or is replaced by a generic
  intervention: `freeze` (do(enemy_y = current value), switched on and off at random times),
  `scale` (the model's enemy displacement times a random factor in [0, 1.5]), or `random walk`.
  The other objects (ball, player, reward) keep coming from the world model and react to the
  intervened enemy. These interventions are generic perturbations of one object's dynamics; they
  are not the `lazy_enemy` rule, and `lazy_enemy` is never used for training or selection.
- Data collection, PPO, the Dreamer-style rounds and model selection (on unmodified Pong only) are
  unchanged from experiment 1 and reused from `experiment_sparse_object_wm.py`.
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

import common
import experiment_sparse_object_wm as e1

# hard-concrete constants (Louizos et al. 2018)
HC_BETA, HC_GAMMA, HC_ZETA = 2.0 / 3.0, -0.1, 1.1


class L0ObjectWM(nn.Module):
    d: int = 128

    @nn.compact
    def __call__(self, hist, action, train: bool = False):
        """Like `e1.SparseObjectWM`, with hard-concrete gates.

        Returns delta (B, 3, 2), reward logits (B, 3), gates (B, 3, 4) and the expected number of
        open gates per receiver `l0` (B, 3).
        """
        B = hist.shape[0]
        p = hist / common.POS_SCALE
        v = (hist[:, 1:] - hist[:, :-1]) / 8.0
        per_obj = jnp.concatenate(
            [p.transpose(0, 2, 1, 3).reshape(B, 3, -1), v.transpose(0, 2, 1, 3).reshape(B, 3, -1)], -1
        )
        enc = [e1.MLP([self.d, self.d], name=f"enc_{n}")(per_obj[:, i]) for i, n in enumerate(common.OBJECT_NAMES)]
        e_act = nn.Embed(common.NUM_ACTIONS, self.d, name="act_embed")(action)
        tokens = jnp.stack(enc + [e_act], axis=1)

        deltas, gates, l0s = [], [], []
        reward_logits = None
        for i, n in enumerate(common.OBJECT_NAMES):
            others = [j for j in range(4) if j != i]
            e_i = tokens[:, i]
            e_j = tokens[:, others]
            pair = jnp.concatenate([jnp.broadcast_to(e_i[:, None], e_j.shape), e_j], -1)
            alpha = e1.MLP([self.d, 1], name=f"gate_{n}")(pair)[..., 0]  # (B, 3) log-odds
            if train:
                u = jax.random.uniform(self.make_rng("gate"), alpha.shape, minval=1e-6, maxval=1 - 1e-6)
                s = nn.sigmoid((jnp.log(u) - jnp.log1p(-u) + alpha) / HC_BETA)
            else:
                s = nn.sigmoid(alpha)
            g = jnp.clip(s * (HC_ZETA - HC_GAMMA) + HC_GAMMA, 0.0, 1.0)
            l0s.append(nn.sigmoid(alpha - HC_BETA * jnp.log(-HC_GAMMA / HC_ZETA)).sum(-1))
            m = nn.LayerNorm(name=f"msg_ln_{n}")(e1.MLP([self.d, self.d], name=f"msg_{n}")(pair))
            msg = jnp.sum(g[..., None] * m, axis=1)
            h = e1.MLP([2 * self.d, self.d], act_last=True, name=f"dec_{n}")(jnp.concatenate([e_i, msg], -1))
            deltas.append(nn.Dense(2, name=f"out_{n}")(h))
            if n == "ball":
                reward_logits = nn.Dense(3, name="reward")(h)
            gates.append(jnp.ones((B, 4)).at[:, jnp.array(others)].set(g))
        return jnp.stack(deltas, 1), reward_logits, jnp.stack(gates, 1), jnp.stack(l0s, 1)


ACT_SAMPLE, ACT_GREEDY = e1.ACT_SAMPLE, e1.ACT_GREEDY  # MLP actor of experiment 1


def build_model(cfg):
    """World model from a config dict (used by evaluate_lazy_enemy.py)."""
    return L0ObjectWM(d=cfg["d"])


def wm_step(model, params, hist, action):
    """Deterministic (test-time gates) one-step prediction; same interface as experiment 1."""
    delta, rlog, gates, _ = model.apply(params, hist, action)
    return hist[:, -1] + delta * 8.0, delta, rlog, gates


def make_wm_loss(model, K, H, gate_coef):
    def loss_fn(params, frames, actions, rewards, key):
        def body(carry, inp):
            hist, key = carry
            a, target, r = inp
            key, k = jax.random.split(key)
            delta, rlog, gates, l0 = model.apply(params, hist, a, train=True, rngs={"gate": k})
            pred = hist[:, -1] + delta * 8.0
            tgt_delta = (target - hist[:, -1]) / 8.0
            mse = jnp.mean(jnp.sum((delta - tgt_delta) ** 2, axis=-1), axis=0)
            ce = optax.softmax_cross_entropy_with_integer_labels(rlog, (r + 1).astype(jnp.int32)).mean()
            new_hist = jnp.concatenate([hist[:, 1:], pred[:, None]], axis=1)
            return (new_hist, key), (mse, ce, l0.mean(), (jnp.sum(gates, -1) - 1).mean() / 3.0)

        inputs = (actions.T, frames[:, K:].transpose(1, 0, 2, 3), rewards.T)
        _, (mse, ce, l0, gm) = jax.lax.scan(body, (frames[:, :K], key), inputs)
        loss = mse.sum(-1).mean() + ce.mean() + gate_coef * l0.mean()
        return loss, dict(mse_player=mse[:, 0].mean(), mse_enemy=mse[:, 1].mean(), mse_ball=mse[:, 2].mean(),
                          mse_step1=mse[0].sum(), reward_ce=ce.mean(), open_gates=l0.mean(), gate_mean=gm.mean())

    return loss_fn


def train_world_model(model, params, opt_state, tx, buf, cfg, key, log):
    arr = buf.arrays()
    n_env = arr["frames"].shape[0]
    val_envs = np.arange(n_env) % 16 == 0
    env_idx, t_idx = buf.valid_starts(cfg.K, cfg.H)
    is_val = val_envs[env_idx]
    tr = (jnp.asarray(env_idx[~is_val]), jnp.asarray(t_idx[~is_val]))
    va = (jnp.asarray(env_idx[is_val]), jnp.asarray(t_idx[is_val]))
    frames, action, reward = (jnp.asarray(arr[k]) for k in ("frames", "action", "reward"))
    data = (frames, action, reward, tr, va)
    loss_fn = make_wm_loss(model, cfg.K, cfg.H, cfg.gate_coef)

    @jax.jit
    def train_step(params, opt_state, key, data):
        frames, action, reward, tr, va = data
        k1, k2 = jax.random.split(key)
        i = jax.random.randint(k1, (cfg.wm_batch,), 0, tr[0].shape[0])
        f, a, r = e1.gather_windows(frames, action, reward, tr[0][i], tr[1][i], cfg.K, cfg.H)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, f, a, r, k2)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    @jax.jit
    def val_step(params, key, data):
        frames, action, reward, tr, va = data
        k1, k2 = jax.random.split(key)
        i = jax.random.randint(k1, (4096,), 0, va[0].shape[0])
        f, a, r = e1.gather_windows(frames, action, reward, va[0][i], va[1][i], cfg.K, cfg.H)
        return loss_fn(params, f, a, r, k2)

    t0 = time.time()
    for it in range(cfg.wm_steps):
        key, k = jax.random.split(key)
        params, opt_state, loss, aux = train_step(params, opt_state, k, data)
        if it % 2000 == 0 or it == cfg.wm_steps - 1:
            key, k = jax.random.split(key)
            vloss, vaux = val_step(params, k, data)
            log(f"  wm it {it:6d} train {float(loss):.4f} val {float(vloss):.4f} "
                + " ".join(f"{k_}={float(v):.4f}" for k_, v in vaux.items()) + f" ({time.time() - t0:.0f}s)")

    # counterfactual leakage on held-out real windows (unmodified Pong)
    key, k1, k2 = jax.random.split(key, 3)
    i = jax.random.randint(k1, (8192,), 0, va[0].shape[0])
    f, a, _ = e1.gather_windows(frames, action, reward, va[0][i], va[1][i], cfg.K, 1)
    pf = jax.jit(lambda h, a_: wm_step(model, params, h, a_)[0])
    leak = common.leakage(pf, f[:, :cfg.K], a[:, 0], k2)
    log("  leakage (px): " + " ".join(f"{k_}={v:.2f}" for k_, v in leak.items()))
    return params, opt_state, {k_: float(v) for k_, v in vaux.items()} | {"val_loss": float(vloss), "leakage": leak}


# ----------------------------------------------------------------------------- imagination with interventions

MODE_NONE, MODE_FREEZE, MODE_SCALE, MODE_WALK = 0, 1, 2, 3


def make_imagination(model, cfg, enemy_range):
    """Imagined env whose enemy mechanism is randomly intervened on (per imagined rally).

    The imagination state `istate` is a dict (step counter, intervention mode and parameters);
    it is carried through experiment 1's PPO code in place of the step counter.
    """
    probs = jnp.array([cfg.p_none, cfg.p_freeze, cfg.p_scale, cfg.p_walk])
    probs = probs / probs.sum()

    def sample_istate(key, n):
        k1, k2, k3 = jax.random.split(key, 3)
        return dict(
            t=jnp.zeros((n,), jnp.int32),
            mode=jax.random.choice(k1, 4, (n,), p=probs),
            frozen=jax.random.bernoulli(k2, 0.5, (n,)),
            scale=jax.random.uniform(k3, (n,), minval=0.0, maxval=1.5),
        )

    def sample_starts(starts, key, n):
        return starts[jax.random.randint(key, (n,), 0, starts.shape[0])]

    def step(wm_params, starts, hist, istate, action, key):
        k_sw, k_walk, k_reset, k_is = jax.random.split(key, 4)
        pred, _, rlog, gates = wm_step(model, wm_params, hist, action)
        last_e = hist[:, -1, 1, 1]
        d_model = pred[:, 1, 1] - last_e
        n = hist.shape[0]
        switch = jax.random.uniform(k_sw, (n,)) < cfg.freeze_switch_p
        frozen = jnp.logical_xor(istate["frozen"], switch)
        d_walk = jax.random.choice(k_walk, jnp.array([-8.0, -4.0, 0.0, 4.0, 8.0]), (n,))
        mode = istate["mode"]
        d_e = jnp.select(
            [mode == MODE_NONE, mode == MODE_FREEZE, mode == MODE_SCALE],
            [d_model, jnp.where(frozen, 0.0, d_model), d_model * istate["scale"]],
            d_walk,
        )
        pred = pred.at[:, 1, 1].set(jnp.clip(last_e + d_e, enemy_range[0], enemy_range[1]))
        if cfg.round_int:
            pred = pred.at[:, 1:].set(jnp.round(pred[:, 1:]))
        pred = jnp.clip(pred, e1.WALL_BOUNDS[0], e1.WALL_BOUNDS[1])
        pred = pred.at[:, :2, 0].set(hist[:, -1, :2, 0])
        cls = jnp.argmax(rlog, axis=-1)
        event = cls != 1
        reward = jnp.where(event, (cls - 1).astype(jnp.float32), 0.0)
        next_hist = jnp.concatenate([hist[:, 1:], pred[:, None]], axis=1)
        t = istate["t"] + 1
        trunc = jnp.logical_and(t >= cfg.imag_len, jnp.logical_not(event))
        done = jnp.logical_or(event, trunc)
        reset_hist = jnp.where(done[:, None, None, None], sample_starts(starts, k_reset, n), next_hist)
        fresh = sample_istate(k_is, n)
        new_istate = dict(t=t, mode=mode, frozen=frozen, scale=istate["scale"])
        new_istate = jax.tree.map(lambda f, o: jnp.where(done, f, o), fresh, new_istate)
        return next_hist, reset_hist, new_istate, reward, done, trunc, gates

    return sample_starts, sample_istate, step


# ----------------------------------------------------------------------------- main loop


def get_parser(default_name="intervention_actor"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=default_name)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--collect_envs", type=int, default=256)
    ap.add_argument("--collect_steps", type=int, default=2000)
    ap.add_argument("--collect_eps", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--gate_coef", type=float, default=0.05)
    ap.add_argument("--wm_batch", type=int, default=512)
    ap.add_argument("--wm_steps", type=int, default=20000)
    ap.add_argument("--wm_lr", type=float, default=3e-4)
    ap.add_argument("--wm_only", type=int, default=0, help="only fit the world model on round-0 data")
    ap.add_argument("--round_int", type=int, default=1)
    ap.add_argument("--imag_len", type=int, default=256)
    ap.add_argument("--p_none", type=float, default=0.25)
    ap.add_argument("--p_freeze", type=float, default=0.35)
    ap.add_argument("--p_scale", type=float, default=0.2)
    ap.add_argument("--p_walk", type=float, default=0.2)
    ap.add_argument("--freeze_switch_p", type=float, default=0.1)
    ap.add_argument("--num_imag_envs", type=int, default=2048)
    ap.add_argument("--ppo_T", type=int, default=128)
    ap.add_argument("--ppo_updates", type=int, default=400)
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
    return ap


def main():
    run(get_parser().parse_args(), L0ObjectWM)


def run(cfg, model_cls):
    """Full pipeline; `model_cls(d=...)` must follow the `L0ObjectWM` interface."""
    out_dir = os.path.join("runs", cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "log.txt"), "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    log(f"config: {vars(cfg)}")
    key = jax.random.PRNGKey(cfg.seed)
    env = common.make_env()  # unmodified Pong only

    model = model_cls(d=cfg.d)
    key, k = jax.random.split(key)
    wm_params = model.init(k, jnp.zeros((1, cfg.K, 3, 2)), jnp.zeros((1,), jnp.int32))
    tx_wm = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg.wm_lr, weight_decay=1e-4))
    wm_opt = tx_wm.init(wm_params)

    key, k = jax.random.split(key)
    ac_params = e1.AC.init(k, jnp.zeros((1, common.POLICY_FEATURE_DIM)))
    tx_ac = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.ppo_lr, eps=1e-5))
    ac_opt = tx_ac.init(ac_params)

    buf = common.ReplayBuffer()
    history, best = [], None
    for rnd in range(cfg.rounds):
        log(f"=== round {rnd} ===")
        key, k = jax.random.split(key)
        keys = jax.random.split(k, cfg.collect_envs)
        if rnd == 0:
            traj = common.collect(env, common.random_act, cfg.collect_steps, None, keys)
        else:
            traj = common.collect(env, e1.make_collect_act_fn(cfg.collect_eps), cfg.collect_steps, ac_params, keys)
        buf.add(traj)
        r = np.asarray(traj["reward"])
        log(f"collected {r.size} steps, total {buf.num_transitions}; points won {int((r > 0).sum())} lost {int((r < 0).sum())}")

        key, k = jax.random.split(key)
        wm_params, wm_opt, wm_stats = train_world_model(model, wm_params, wm_opt, tx_wm, buf, cfg, k, log)
        if cfg.wm_only:
            with open(os.path.join(out_dir, "wm_only.pkl"), "wb") as f:
                pickle.dump(dict(wm_params=jax.device_get(wm_params), cfg=vars(cfg)), f)
            with open(os.path.join(out_dir, "wm_only.json"), "w") as f:
                json.dump(dict(config=vars(cfg), wm=wm_stats), f, indent=1)
            return

        arr = buf.arrays()
        env_idx, t_idx = buf.valid_starts(cfg.K, 0)
        fo = np.arange(-cfg.K + 1, 1)
        starts = jnp.asarray(arr["frames"][env_idx[:, None], t_idx[:, None] + fo[None]])
        enemy_range = (float(arr["frames"][..., 1, 1].min()), float(arr["frames"][..., 1, 1].max()))
        sample_starts, sample_istate, imag_step = make_imagination(model, cfg, enemy_range)
        ppo_update = jax.jit(partial(e1.make_ppo(model, cfg, tx_ac), imag_step=imag_step))
        key, k1, k2 = jax.random.split(key, 3)
        hist = sample_starts(starts, k1, cfg.num_imag_envs)
        istate = sample_istate(k2, cfg.num_imag_envs)
        t0 = time.time()
        for u in range(cfg.ppo_updates):
            key, k = jax.random.split(key)
            ac_params, ac_opt, hist, istate, st = ppo_update(ac_params, ac_opt, wm_params, starts, hist, istate, k)
            if u % 50 == 0 or u == cfg.ppo_updates - 1:
                log(f"  ppo {u:4d} " + " ".join(f"{k_}={float(v):.3f}" for k_, v in st.items()) + f" ({time.time() - t0:.0f}s)")

        key, k = jax.random.split(key)
        ev = {g: e1.evaluate_real(env, ac_params, k, cfg.eval_games, cfg.eval_max_steps, g) for g in (False, True)}
        log(f"  real Pong eval sample: {ev[False]['player']:.2f} : {ev[False]['enemy']:.2f} | "
            f"greedy: {ev[True]['player']:.2f} : {ev[True]['enemy']:.2f}")
        rec = dict(round=rnd, wm=wm_stats, pong_sample=ev[False], pong_greedy=ev[True], transitions=buf.num_transitions,
                   imag_final=jax.tree.map(float, st))
        history.append(rec)
        ck = dict(wm_params=jax.device_get(wm_params), ac_params=jax.device_get(ac_params), cfg=vars(cfg))
        with open(os.path.join(out_dir, f"round{rnd}.pkl"), "wb") as f:
            pickle.dump(ck, f)
        diffs = {g: ev[g]["player"] - ev[g]["enemy"] for g in (False, True)}
        g_best = diffs[True] >= diffs[False]
        if best is None or diffs[g_best] > best[0]:
            best = (diffs[g_best], rnd, bool(g_best))
            with open(os.path.join(out_dir, "best.pkl"), "wb") as f:
                pickle.dump(ck | dict(round=rnd, greedy=bool(g_best)), f)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(dict(config=vars(cfg), rounds=history,
                           selected=dict(round=best[1], greedy=best[2], pong_score_diff=best[0])), f, indent=1)
    log(f"selected round {best[1]} (greedy={best[2]}) by unmodified-Pong score diff {best[0]:.2f}")


if __name__ == "__main__":
    main()
