"""Experiment 3: object-gated actor (sparse attention over objects) trained in imagination.

Motivation (experiment 1, `experiment_sparse_object_wm.md`): the imagination-trained MLP actor won
unmodified Pong 21 : 2 but lost with `lazy_enemy` 12.6 : 21. The MLP policy reads every object at
every step, including the enemy, whose position in unmodified Pong is always close to the ball's
height. When the enemy behaves differently, its position is out of distribution for the policy.

The project idea (each prediction should use only the objects it needs, decided per step) is
applied here to the *actor* as well:
- The actor has one token per object (player, ball, enemy) built from that object's last 4
  frames. The player's own token is always used. The ball and enemy tokens reach the policy only
  through per-step hard-concrete (L0) gates g_j = HC(f(e_player, e_j)), and an expected-L0 penalty
  (`--gate_coef`) is added to the PPO loss, so the actor only looks at an object when that improves
  its return. Gates start open and the penalty is switched on after `--gate_warmup` PPO updates
  (a first try with the penalty from the start closed all gates before the actor learned anything). During PPO updates the gates are stochastic (object dropout); rollouts and
  evaluation use the deterministic gates.
- The critic is an ordinary MLP over all objects (it is not used when acting).

The world model, data collection, imagination (no interventions), rounds and model selection on
unmodified Pong are exactly those of experiment 1 (`experiment_sparse_object_wm.py`).
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

HC_BETA, HC_GAMMA, HC_ZETA = 2.0 / 3.0, -0.1, 1.1
F = common.FRAME_STACK
# order of tokens inside the actor: player (always on), then gated ball and enemy
GATED = (2, 1)
GATED_NAMES = ("ball", "enemy")


def object_tokens(x):
    """Flat policy features (see common.policy_features) -> (B, 3, per-object features)."""
    B = x.shape[0]
    n_p = F * 3 * 2
    p = x[:, :n_p].reshape(B, F, 3, 2).transpose(0, 2, 1, 3).reshape(B, 3, -1)
    v = x[:, n_p:].reshape(B, F - 1, 3, 2).transpose(0, 2, 1, 3).reshape(B, 3, -1)
    return jnp.concatenate([p, v], -1)


class GatedActorCritic(nn.Module):
    hidden: int = 256
    d: int = 64
    gate_init: float = 1.0  # gates start (mostly) open; the penalty closes the ones the actor does not need

    @nn.compact
    def __call__(self, x, train: bool = False):
        orth = nn.initializers.orthogonal
        tok = object_tokens(x)
        e_self = nn.relu(nn.Dense(self.d, name="enc_player")(tok[:, 0]))
        gates, l0, msgs = [], [], []
        for j, n in zip(GATED, GATED_NAMES):
            e_j = nn.relu(nn.Dense(self.d, name=f"enc_{n}")(tok[:, j]))
            pair = jnp.concatenate([e_self, e_j], -1)
            alpha = nn.Dense(1, bias_init=nn.initializers.constant(self.gate_init), name=f"gate_{n}_out")(
                nn.relu(nn.Dense(self.d, name=f"gate_{n}")(pair)))[:, 0]
            if train:
                u = jax.random.uniform(self.make_rng("gate"), alpha.shape, minval=1e-6, maxval=1 - 1e-6)
                s = nn.sigmoid((jnp.log(u) - jnp.log1p(-u) + alpha) / HC_BETA)
            else:
                s = nn.sigmoid(alpha)
            g = jnp.clip(s * (HC_ZETA - HC_GAMMA) + HC_GAMMA, 0.0, 1.0)
            gates.append(g)
            l0.append(nn.sigmoid(alpha - HC_BETA * jnp.log(-HC_GAMMA / HC_ZETA)))
            m = nn.relu(nn.Dense(self.d, name=f"msg_{n}")(pair))
            msgs.append(g[:, None] * nn.Dense(self.d, name=f"msg_{n}_out")(m))
        h = jnp.concatenate([e_self] + msgs, -1)
        for i in range(2):
            h = nn.relu(nn.Dense(self.hidden, kernel_init=orth(np.sqrt(2)), name=f"a_{i}")(h))
        logits = nn.Dense(common.NUM_ACTIONS, kernel_init=orth(0.01), name="pi")(h)
        c = x
        for i in range(2):
            c = nn.relu(nn.Dense(self.hidden, kernel_init=orth(np.sqrt(2)), name=f"c_{i}")(c))
        value = nn.Dense(1, kernel_init=orth(1.0), name="v")(c)[:, 0]
        return logits, value, jnp.stack(gates, -1), jnp.stack(l0, -1).sum(-1)


AC = GatedActorCritic()  # replaced in main() if --gate_init differs from the default


def _act(params, obs, key, greedy):
    x = common.policy_features(common.frames_to_pos(obs))[None]
    logits, _, _, _ = AC.apply(params, x)
    return jnp.where(greedy, jnp.argmax(logits[0]), jax.random.categorical(key, logits[0]))


def ACT_SAMPLE(params, obs, key):
    return _act(params, obs, key, False)


def ACT_GREEDY(params, obs, key):
    return _act(params, obs, key, True)


def make_collect_act_fn(eps):
    def act(params, obs, key):
        k1, k2, k3 = jax.random.split(key, 3)
        a = ACT_SAMPLE(params, obs, k1)
        return jnp.where(jax.random.uniform(k2) < eps, jax.random.randint(k3, (), 0, common.NUM_ACTIONS), a)

    return act


# world-model interface for evaluate_lazy_enemy.py (same world model as experiment 1)
build_model = e1.build_model
wm_step = e1.wm_step


def make_ppo(cfg, tx_ac, imag_step):
    def obs_of(hist):
        return common.policy_features(hist[:, -F:])

    def rollout(ac_params, wm_params, starts, hist, t, key):
        def body(carry, k):
            hist, t = carry
            k1, k2 = jax.random.split(k)
            obs = obs_of(hist)
            logits, value, gates, _ = AC.apply(ac_params, obs)
            a = jax.random.categorical(k1, logits)
            logp = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a]
            next_hist, new_hist, t, r, done, trunc, _ = imag_step(wm_params, starts, hist, t, a, k2)
            _, v_next, _, _ = AC.apply(ac_params, obs_of(next_hist))
            r = r + jnp.where(trunc, cfg.gamma * v_next, 0.0)
            return (new_hist, t), dict(obs=obs, a=a, logp=logp, v=value, r=r, done=done, gates=gates)

        (hist, t), traj = jax.lax.scan(body, (hist, t), jax.random.split(key, cfg.ppo_T))
        _, last_v, _, _ = AC.apply(ac_params, obs_of(hist))
        return hist, t, traj, last_v

    def gae(traj, last_v):
        def body(carry, x):
            adv, v_next = carry
            nonterm = 1.0 - x["done"].astype(jnp.float32)
            delta = x["r"] + cfg.gamma * v_next * nonterm - x["v"]
            adv = delta + cfg.gamma * cfg.lam * nonterm * adv
            return (adv, x["v"]), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), traj, reverse=True)
        return adv, adv + traj["v"]

    def ppo_loss(ac_params, batch, key, gate_coef):
        logits, v, _, l0 = AC.apply(ac_params, batch["obs"], train=True, rngs={"gate": key})
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, batch["a"][:, None], axis=-1)[:, 0]
        ratio = jnp.exp(logp - batch["logp"])
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vloss = 0.5 * ((v - batch["ret"]) ** 2).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        loss = pg + cfg.vf_coef * vloss - cfg.ent_coef * ent + gate_coef * l0.mean()
        return loss, dict(pg=pg, v=vloss, ent=ent, l0=l0.mean())

    def update(ac_params, opt_state, wm_params, starts, hist, t, key, gate_coef):
        key, k_roll = jax.random.split(key)
        hist, t, traj, last_v = rollout(ac_params, wm_params, starts, hist, t, k_roll)
        adv, ret = gae(traj, last_v)
        n = cfg.ppo_T * cfg.num_imag_envs
        data = dict(obs=traj["obs"].reshape(n, -1), a=traj["a"].reshape(n), logp=traj["logp"].reshape(n),
                    adv=adv.reshape(n), ret=ret.reshape(n))

        def epoch(carry, k):
            ac_params, opt_state = carry
            k_perm, k_gate = jax.random.split(k)
            perm = jax.random.permutation(k_perm, n).reshape(cfg.ppo_minibatches, -1)
            gkeys = jax.random.split(k_gate, cfg.ppo_minibatches)

            def mb(carry, inp):
                ac_params, opt_state = carry
                idx, kg = inp
                batch = jax.tree.map(lambda x: x[idx], data)
                (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(ac_params, batch, kg, gate_coef)
                upd, opt_state = tx_ac.update(grads, opt_state, ac_params)
                return (optax.apply_updates(ac_params, upd), opt_state), aux

            return jax.lax.scan(mb, (ac_params, opt_state), (perm, gkeys))

        (ac_params, opt_state), aux = jax.lax.scan(epoch, (ac_params, opt_state), jax.random.split(key, cfg.ppo_epochs))
        n_done = traj["done"].sum()
        g = traj["gates"].reshape(n, 2)
        stats = dict(
            reward_per_episode=traj["r"].sum() / jnp.maximum(n_done, 1),
            win_frac=(traj["r"] > 0.5).sum() / jnp.maximum((jnp.abs(traj["r"]) > 0.5).sum(), 1),
            gate_ball=g[:, 0].mean(), gate_enemy=g[:, 1].mean(),
            **jax.tree.map(lambda x: x.mean(), aux),
        )
        return ac_params, opt_state, hist, t, stats

    return update


def evaluate_real(env, ac_params, key, n_games, max_steps, greedy):
    act = ACT_GREEDY if greedy else ACT_SAMPLE
    ps, es, over = common.evaluate(env, act, max_steps, ac_params, jax.random.split(key, n_games))
    return dict(player=float(ps.mean()), enemy=float(es.mean()), player_std=float(ps.std()),
                finished=float(over.mean()), player_per_game=[float(x) for x in ps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gated_actor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--collect_envs", type=int, default=256)
    ap.add_argument("--collect_steps", type=int, default=2000)
    ap.add_argument("--collect_eps", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--gate_coef_wm", type=float, default=0.01, help="world-model gate L1 (experiment 1)")
    ap.add_argument("--wm_batch", type=int, default=512)
    ap.add_argument("--wm_steps", type=int, default=20000)
    ap.add_argument("--wm_lr", type=float, default=3e-4)
    ap.add_argument("--round_int", type=int, default=1)
    ap.add_argument("--imag_len", type=int, default=256)
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
    ap.add_argument("--gate_coef", type=float, default=0.05, help="actor gate expected-L0 penalty")
    ap.add_argument("--gate_init", type=float, default=1.0, help="initial gate log-odds")
    ap.add_argument("--gate_warmup", type=int, default=400, help="PPO updates without gate penalty")
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
    global AC
    AC = GatedActorCritic(gate_init=cfg.gate_init)
    key = jax.random.PRNGKey(cfg.seed)
    env = common.make_env()  # unmodified Pong only

    model = e1.SparseObjectWM(d=cfg.d)
    key, k = jax.random.split(key)
    wm_params = model.init(k, jnp.zeros((1, cfg.K, 3, 2)), jnp.zeros((1,), jnp.int32))
    tx_wm = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg.wm_lr, weight_decay=1e-4))
    wm_opt = tx_wm.init(wm_params)
    wm_cfg = argparse.Namespace(**(vars(cfg) | dict(gate_coef=cfg.gate_coef_wm)))

    key, k = jax.random.split(key)
    ac_params = AC.init(k, jnp.zeros((1, common.POLICY_FEATURE_DIM)))
    tx_ac = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.ppo_lr, eps=1e-5))
    ac_opt = tx_ac.init(ac_params)
    sample_starts, imag_step = e1.make_imagination(model, cfg)
    ppo_update = jax.jit(make_ppo(cfg, tx_ac, imag_step))

    buf = common.ReplayBuffer()
    history, best = [], None
    n_ppo = 0  # PPO updates so far (for the gate-penalty warm-up)
    for rnd in range(cfg.rounds):
        log(f"=== round {rnd} ===")
        key, k = jax.random.split(key)
        keys = jax.random.split(k, cfg.collect_envs)
        if rnd == 0:
            traj = common.collect(env, common.random_act, cfg.collect_steps, None, keys)
        else:
            traj = common.collect(env, make_collect_act_fn(cfg.collect_eps), cfg.collect_steps, ac_params, keys)
        buf.add(traj)
        r = np.asarray(traj["reward"])
        log(f"collected {r.size} steps, total {buf.num_transitions}; points won {int((r > 0).sum())} lost {int((r < 0).sum())}")

        key, k = jax.random.split(key)
        wm_params, wm_opt, wm_stats = e1.train_world_model(model, wm_params, wm_opt, tx_wm, buf, wm_cfg, k, log)

        arr = buf.arrays()
        env_idx, t_idx = buf.valid_starts(cfg.K, 0)
        fo = np.arange(-cfg.K + 1, 1)
        starts = jnp.asarray(arr["frames"][env_idx[:, None], t_idx[:, None] + fo[None]])
        key, k = jax.random.split(key)
        hist = sample_starts(starts, k, cfg.num_imag_envs)
        t = jnp.zeros((cfg.num_imag_envs,), jnp.int32)
        t0 = time.time()
        for u in range(cfg.ppo_updates):
            key, k = jax.random.split(key)
            coef = 0.0 if n_ppo < cfg.gate_warmup else cfg.gate_coef
            ac_params, ac_opt, hist, t, st = ppo_update(ac_params, ac_opt, wm_params, starts, hist, t, k, coef)
            n_ppo += 1
            if u % 50 == 0 or u == cfg.ppo_updates - 1:
                log(f"  ppo {u:4d} " + " ".join(f"{k_}={float(v):.3f}" for k_, v in st.items()) + f" ({time.time() - t0:.0f}s)")

        key, k = jax.random.split(key)
        ev = {g: evaluate_real(env, ac_params, k, cfg.eval_games, cfg.eval_max_steps, g) for g in (False, True)}
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
