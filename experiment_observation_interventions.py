"""Experiment 6: actor trained in imagination with interventions on its *observation* of the enemy.

History (see the experiment Markdown files):
- Exp. 1: imagination-trained MLP actor wins Pong 21 : 2 but loses with `lazy_enemy` (final
  score -8.4). The policy relies on the enemy's position, which in Pong always tracks the ball.
- Exp. 2: training the actor under interventions on the enemy's *dynamics* inside the world model
  helped (lazy final score +0.2) but not enough: the world model's ball prediction depends on the
  enemy even far from a bounce (counterfactual leakage ~5-13 px), so under intervened enemies the
  imagined ball misbehaves and the actor learns a false ball-enemy dependence.
- Exp. 4/5 and an ablation: gating variants (receiver-driven L0, soft and straight-through binary
  gates) did not remove this leakage, although removing ball<-enemy entirely costs only ~0.2 px of
  1-step ball accuracy.

This experiment avoids needing a causally correct world model for the enemy:
- Imagination uses experiment 1's world model unchanged (dynamics exactly as learned from
  unmodified Pong).
- In each imagined rally, with probability 1 - p_none, the enemy position that the *actor sees* is
  replaced by an intervened track: `freeze` (the observed enemy stops and restarts at random
  times), `scale` (it moves with a random fraction/multiple of the true enemy's motion), or
  `random walk`. The world model, the ball, the reward and the true enemy are not changed. The
  actor therefore learns that its enemy input can be unreliable and has to win using the ball and
  its own paddle, while it can still use the enemy when its track looks normal.
- These are generic interventions on one object's observed dynamics; `lazy_enemy` is never used
  for training or model selection. The choice to intervene on the enemy (the object not controlled
  by the agent and not the ball it must hit) is a design decision of this experiment.
- Everything else (world model, data collection rounds, PPO, MLP actor, selection on unmodified
  Pong) is identical to experiment 1.
"""

import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import common
import experiment_sparse_object_wm as e1

F = common.FRAME_STACK
MODE_NONE, MODE_FREEZE, MODE_SCALE, MODE_WALK = 0, 1, 2, 3

# interfaces for evaluate_lazy_enemy.py
build_model = e1.build_model
wm_step = e1.wm_step
ACT_SAMPLE, ACT_GREEDY = e1.ACT_SAMPLE, e1.ACT_GREEDY
AC = e1.AC


def make_imagination(model, cfg, enemy_range):
    """Wraps experiment 1's imagined env and adds the intervened enemy observation to the state."""
    _, e1_step = e1.make_imagination(model, cfg)
    probs = jnp.array([cfg.p_none, cfg.p_freeze, cfg.p_scale, cfg.p_walk])
    probs = probs / probs.sum()

    def sample_istate(key, hist):
        n = hist.shape[0]
        k1, k2, k3 = jax.random.split(key, 3)
        return dict(
            t=jnp.zeros((n,), jnp.int32),
            mode=jax.random.choice(k1, 4, (n,), p=probs),
            frozen=jax.random.bernoulli(k2, 0.5, (n,)),
            scale=jax.random.uniform(k3, (n,), minval=0.0, maxval=1.5),
            obs_e=hist[:, -F:, 1, 1],  # observed enemy y over the last F frames
        )

    def step(wm_params, starts, hist, ist, action, key):
        k_env, k_sw, k_walk, k_is = jax.random.split(key, 4)
        next_hist, reset_hist, t, r, done, trunc, gates = e1_step(wm_params, starts, hist, ist["t"], action, k_env)
        n = hist.shape[0]
        true_d = next_hist[:, -1, 1, 1] - hist[:, -1, 1, 1]
        frozen = jnp.logical_xor(ist["frozen"], jax.random.uniform(k_sw, (n,)) < cfg.freeze_switch_p)
        d_walk = jax.random.choice(k_walk, jnp.array([-8.0, -4.0, 0.0, 4.0, 8.0]), (n,))
        mode = ist["mode"]
        d = jnp.select(
            [mode == MODE_NONE, mode == MODE_FREEZE, mode == MODE_SCALE],
            [true_d, jnp.where(frozen, 0.0, true_d), jnp.round(true_d * ist["scale"])],
            d_walk,
        )
        new_e = jnp.where(mode == MODE_NONE, next_hist[:, -1, 1, 1],
                          jnp.clip(ist["obs_e"][:, -1] + d, enemy_range[0], enemy_range[1]))
        obs_e_next = jnp.concatenate([ist["obs_e"][:, 1:], new_e[:, None]], axis=1)
        cont = dict(t=t, mode=mode, frozen=frozen, scale=ist["scale"], obs_e=obs_e_next)
        fresh = sample_istate(k_is, reset_hist)
        new_ist = jax.tree.map(lambda f, o: jnp.where(done.reshape((-1,) + (1,) * (o.ndim - 1)), f, o), fresh, cont)
        return next_hist, reset_hist, new_ist, r, done, trunc, obs_e_next

    return sample_istate, step


def obs_of(hist, obs_e):
    """Policy input from the WM history, with the enemy replaced by the (possibly intervened) view."""
    return common.policy_features(hist[:, -F:].at[:, :, 1, 1].set(obs_e))


def make_ppo(cfg, tx_ac, imag_step):
    AC_ = e1.AC

    def rollout(ac_params, wm_params, starts, hist, ist, key):
        def body(carry, k):
            hist, ist = carry
            k1, k2 = jax.random.split(k)
            obs = obs_of(hist, ist["obs_e"])
            logits, value = AC_.apply(ac_params, obs)
            a = jax.random.categorical(k1, logits)
            logp = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a]
            next_hist, new_hist, new_ist, r, done, trunc, obs_e_next = imag_step(wm_params, starts, hist, ist, a, k2)
            _, v_next = AC_.apply(ac_params, obs_of(next_hist, obs_e_next))
            r = r + jnp.where(trunc, cfg.gamma * v_next, 0.0)
            return (new_hist, new_ist), dict(obs=obs, a=a, logp=logp, v=value, r=r, done=done,
                                             intervened=ist["mode"] != MODE_NONE)

        (hist, ist), traj = jax.lax.scan(body, (hist, ist), jax.random.split(key, cfg.ppo_T))
        _, last_v = AC_.apply(ac_params, obs_of(hist, ist["obs_e"]))
        return hist, ist, traj, last_v

    def gae(traj, last_v):
        def body(carry, x):
            adv, v_next = carry
            nonterm = 1.0 - x["done"].astype(jnp.float32)
            delta = x["r"] + cfg.gamma * v_next * nonterm - x["v"]
            adv = delta + cfg.gamma * cfg.lam * nonterm * adv
            return (adv, x["v"]), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), traj, reverse=True)
        return adv, adv + traj["v"]

    def ppo_loss(ac_params, batch):
        logits, v = AC_.apply(ac_params, batch["obs"])
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, batch["a"][:, None], axis=-1)[:, 0]
        ratio = jnp.exp(logp - batch["logp"])
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vloss = 0.5 * ((v - batch["ret"]) ** 2).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        return pg + cfg.vf_coef * vloss - cfg.ent_coef * ent, dict(pg=pg, v=vloss, ent=ent)

    def update(ac_params, opt_state, wm_params, starts, hist, ist, key):
        key, k_roll = jax.random.split(key)
        hist, ist, traj, last_v = rollout(ac_params, wm_params, starts, hist, ist, k_roll)
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

            return jax.lax.scan(mb, (ac_params, opt_state), perm)

        (ac_params, opt_state), aux = jax.lax.scan(epoch, (ac_params, opt_state), jax.random.split(key, cfg.ppo_epochs))
        r, iv = traj["r"], traj["intervened"]
        pts = jnp.abs(r) > 0.5

        def win(m):
            return ((r > 0.5) & m).sum() / jnp.maximum((pts & m).sum(), 1)

        stats = dict(win_frac=win(jnp.ones_like(iv)), win_frac_intervened=win(iv), win_frac_clean=win(~iv),
                     **jax.tree.map(lambda x: x.mean(), aux))
        return ac_params, opt_state, hist, ist, stats

    return update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="observation_interventions")
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
    ap.add_argument("--wm_steps", type=int, default=20000)
    ap.add_argument("--wm_lr", type=float, default=3e-4)
    ap.add_argument("--round_int", type=int, default=1)
    ap.add_argument("--imag_len", type=int, default=256)
    ap.add_argument("--p_none", type=float, default=0.4)
    ap.add_argument("--p_freeze", type=float, default=0.3)
    ap.add_argument("--p_scale", type=float, default=0.15)
    ap.add_argument("--p_walk", type=float, default=0.15)
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
    env = common.make_env()  # unmodified Pong only

    model = e1.SparseObjectWM(d=cfg.d)
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
        wm_params, wm_opt, wm_stats = e1.train_world_model(model, wm_params, wm_opt, tx_wm, buf, cfg, k, log)

        arr = buf.arrays()
        env_idx, t_idx = buf.valid_starts(cfg.K, 0)
        fo = np.arange(-cfg.K + 1, 1)
        starts = jnp.asarray(arr["frames"][env_idx[:, None], t_idx[:, None] + fo[None]])
        enemy_range = (float(arr["frames"][..., 1, 1].min()), float(arr["frames"][..., 1, 1].max()))
        sample_istate, imag_step = make_imagination(model, cfg, enemy_range)
        ppo_update = jax.jit(make_ppo(cfg, tx_ac, imag_step))
        key, k1, k2 = jax.random.split(key, 3)
        hist = starts[jax.random.randint(k1, (cfg.num_imag_envs,), 0, starts.shape[0])]
        ist = sample_istate(k2, hist)
        t0 = time.time()
        for u in range(cfg.ppo_updates):
            key, k = jax.random.split(key)
            ac_params, ac_opt, hist, ist, st = ppo_update(ac_params, ac_opt, wm_params, starts, hist, ist, k)
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
