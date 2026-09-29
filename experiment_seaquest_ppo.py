"""Experiment S1 (Seaquest): model-free PPO on the base game with a task-shaped reward.

First Seaquest experiment. The target behaviour (collect 6 divers, then surface with them, twice
per game) is a long-horizon exploration problem: the game's own reward mostly pays for shooting
enemies, and surfacing with fewer than 6 divers costs a diver. So we shape the reward on the base
game, from the base game's own state:
  r = score gain / 100  (enemies, rescue bonus)
    + `diver_bonus` per diver picked up
    + `rescue_bonus` per successful rescue (surfacing with 6 divers)
    - `life_penalty` per life lost
    - `surface_penalty` when surfacing costs a diver (surfacing with 1-5 divers).

The agent is an MLP actor-critic on the object-centric observation (4 stacked frames, fixed feature
scaling) and is trained with PPO (PureJaxRL-style, fully jitted) on many parallel copies of the
**base** game. The `gravity` mod is never used here: selection uses rescues in the base game, and
`evaluate_gravity.py` evaluates the selected checkpoint afterwards.
"""

import argparse
import json
import os
import pickle
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import seaquest_common as sc


class ActorCritic(nn.Module):
    hidden: int = 512

    @nn.compact
    def __call__(self, x):
        orth = nn.initializers.orthogonal

        def trunk(x, name):
            for i in range(2):
                x = nn.relu(nn.Dense(self.hidden, kernel_init=orth(np.sqrt(2)), name=f"{name}_{i}")(x))
            return x

        logits = nn.Dense(sc.NUM_ACTIONS, kernel_init=orth(0.01), name="pi")(trunk(x, "a"))
        value = nn.Dense(1, kernel_init=orth(1.0), name="v")(trunk(x, "c"))[..., 0]
        return logits, value


AC = ActorCritic()


def ACT_SAMPLE(params, obs, key):
    logits, _ = AC.apply(params, sc.obs_features(obs))
    return jax.random.categorical(key, logits)


def ACT_GREEDY(params, obs, key):
    logits, _ = AC.apply(params, sc.obs_features(obs))
    return jnp.argmax(logits)


def shaped_reward(cfg, g0, g1, env_done):
    """Task-shaped reward from two consecutive base-game states (see module docstring)."""
    d_score = jnp.maximum(g1.score - g0.score, 0).astype(jnp.float32)
    d_div = g1.divers_collected - g0.divers_collected
    rescued = (g1.successful_rescues > g0.successful_rescues).astype(jnp.float32)
    lost_life = (g1.lives < g0.lives).astype(jnp.float32)
    picked = jnp.maximum(d_div, 0).astype(jnp.float32)
    # a diver lost without a rescue or death: surfacing with 1-5 divers
    surf_loss = ((d_div < 0) & (rescued == 0) & (lost_life == 0)).astype(jnp.float32)
    r = (d_score / 100.0 + cfg.diver_bonus * picked + cfg.rescue_bonus * rescued
         - cfg.life_penalty * lost_life - cfg.surface_penalty * surf_loss)
    return jnp.where(env_done, 0.0, r)  # the transition into an automatic reset carries no reward


def make_train(cfg, env):
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                     optax.adam(optax.linear_schedule(cfg.lr, cfg.lr * 0.1, cfg.updates * cfg.epochs * cfg.minibatches),
                                eps=1e-5))

    def env_step(carry, key):
        params, obs, st = carry
        k_act, _ = jax.random.split(key)
        logits, value = AC.apply(params, jax.vmap(sc.obs_features)(obs))
        a = jax.random.categorical(k_act, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a]
        g0 = sc.game_state(st)
        obs2, st2, r, term, trunc, info = jax.vmap(env.step)(st, a)
        g1 = sc.game_state(st2)
        rs = shaped_reward(cfg, g0, g1, info["env_done"])
        done = jnp.logical_or(term, info["env_done"])  # life loss ends the value horizon
        tr = dict(obs=obs, a=a, logp=logp, v=value, r=rs, done=done,
                  env_r=info["env_reward"], rescue=(g1.successful_rescues > g0.successful_rescues) & ~info["env_done"],
                  picked=jnp.maximum(g1.divers_collected - g0.divers_collected, 0) * ~info["env_done"],
                  game_over=info["env_done"])
        return (params, obs2, st2), tr

    def gae(traj, last_v):
        def body(carry, x):
            adv, v_next = carry
            nonterm = 1.0 - x["done"].astype(jnp.float32)
            delta = x["r"] + cfg.gamma * v_next * nonterm - x["v"]
            adv = delta + cfg.gamma * cfg.lam * nonterm * adv
            return (adv, x["v"]), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), traj, reverse=True)
        return adv, adv + traj["v"]

    def loss_fn(params, b):
        logits, v = AC.apply(params, jax.vmap(sc.obs_features)(b["obs"]))
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, b["a"][:, None], -1)[:, 0]
        ratio = jnp.exp(logp - b["logp"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        v_clip = b["v"] + jnp.clip(v - b["v"], -cfg.clip, cfg.clip)
        vloss = 0.5 * jnp.maximum((v - b["ret"]) ** 2, (v_clip - b["ret"]) ** 2).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        return pg + cfg.vf_coef * vloss - cfg.ent_coef * ent, dict(pg=pg, v=vloss, ent=ent)

    def update(params, opt_state, obs, st, key):
        key, k_roll = jax.random.split(key)
        (params, obs, st), traj = jax.lax.scan(env_step, (params, obs, st), jax.random.split(k_roll, cfg.T))
        _, last_v = AC.apply(params, jax.vmap(sc.obs_features)(obs))
        adv, ret = gae(traj, last_v)
        n = cfg.T * cfg.num_envs
        data = dict(obs=traj["obs"].reshape(n, *traj["obs"].shape[2:]), a=traj["a"].reshape(n),
                    logp=traj["logp"].reshape(n), v=traj["v"].reshape(n), adv=adv.reshape(n), ret=ret.reshape(n))

        def epoch(carry, k):
            params, opt_state = carry
            perm = jax.random.permutation(k, n).reshape(cfg.minibatches, -1)

            def mb(carry, idx):
                params, opt_state = carry
                b = jax.tree.map(lambda x: x[idx], data)
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
                upd, opt_state = tx.update(grads, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), aux

            return jax.lax.scan(mb, (params, opt_state), perm)

        (params, opt_state), aux = jax.lax.scan(epoch, (params, opt_state), jax.random.split(key, cfg.epochs))
        stats = dict(shaped_r=traj["r"].sum() / cfg.num_envs, env_r=traj["env_r"].sum() / cfg.num_envs,
                     rescues=traj["rescue"].sum() / cfg.num_envs, divers=traj["picked"].sum() / cfg.num_envs,
                     games_over=traj["game_over"].sum(), **jax.tree.map(lambda x: x.mean(), aux))
        return params, opt_state, obs, st, stats

    return tx, jax.jit(update)


def evaluate(env, params, key, games, max_steps, greedy):
    act = ACT_GREEDY if greedy else ACT_SAMPLE
    out = sc.evaluate(env, act, max_steps, params, jax.random.split(key, games))
    out = {k: np.asarray(v) for k, v in out.items()}
    return dict(rescues=float(out["rescues"].mean()), rescues_per_game=out["rescues"].tolist(),
                divers=float(out["divers"].mean()), score=float(out["score"].mean()),
                lives_lost=float(out["lives_lost"].mean()), steps=float(out["steps"].mean()),
                finished=float(out["finished"].mean()))


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="seaquest_ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_envs", type=int, default=512)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--diver_bonus", type=float, default=1.0)
    ap.add_argument("--rescue_bonus", type=float, default=10.0)
    ap.add_argument("--life_penalty", type=float, default=2.0)
    ap.add_argument("--surface_penalty", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_games", type=int, default=32)
    ap.add_argument("--eval_max_steps", type=int, default=20000)
    return ap


def main(cfg=None, make_train_env=None, experiment=None):
    """Train; `make_train_env(cfg)` builds the training environment (default: the base game)."""
    cfg = cfg or get_parser().parse_args()
    out_dir = os.path.join("runs", cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "log.txt"), "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    cfg_d = vars(cfg) | dict(experiment=experiment or os.path.splitext(os.path.basename(__file__))[0], game="seaquest")
    log(f"config: {cfg_d}")
    env = sc.make_env()  # base game only (evaluation during training)
    train_env = make_train_env(cfg) if make_train_env else env
    key = jax.random.PRNGKey(cfg.seed)
    key, k_init, k_env = jax.random.split(key, 3)
    params = AC.init(k_init, jnp.zeros((1, sc.OBS_DIM)))
    tx, update = make_train(cfg, train_env)
    opt_state = tx.init(params)
    obs, st = jax.vmap(train_env.reset)(jax.random.split(k_env, cfg.num_envs))

    history, best = [], None
    t0 = time.time()
    for u in range(cfg.updates):
        key, k = jax.random.split(key)
        params, opt_state, obs, st, stats = update(params, opt_state, obs, st, k)
        if u % 25 == 0:
            steps = (u + 1) * cfg.num_envs * cfg.T
            log(f"  upd {u:5d} frames {4 * steps / 1e6:7.1f}M " + " ".join(f"{k_}={float(v):.3f}" for k_, v in stats.items())
                + f" ({time.time() - t0:.0f}s)")
        if (u + 1) % cfg.eval_every == 0 or u == cfg.updates - 1:
            key, k = jax.random.split(key)
            ev = {g: evaluate(env, params, k, cfg.eval_games, cfg.eval_max_steps, g) for g in (False, True)}
            rnd = len(history)
            log(f"=== round {rnd} === update {u + 1}: base-game eval sample rescues {ev[False]['rescues']:.2f} "
                f"divers {ev[False]['divers']:.1f} score {ev[False]['score']:.0f} | greedy rescues {ev[True]['rescues']:.2f} "
                f"divers {ev[True]['divers']:.1f} score {ev[True]['score']:.0f}")
            history.append(dict(round=rnd, update=u + 1, frames=4 * (u + 1) * cfg.num_envs * cfg.T,
                                base_sample=ev[False], base_greedy=ev[True], train=jax.tree.map(float, stats)))
            ck = dict(ac_params=jax.device_get(params), cfg=cfg_d)
            with open(os.path.join(out_dir, f"round{rnd}.pkl"), "wb") as f:
                pickle.dump(ck, f)
            # model selection on the BASE game only: most rescues, then score
            for g in (False, True):
                crit = (ev[g]["rescues"], ev[g]["score"])
                if best is None or crit > best[0]:
                    best = (crit, rnd, g)
                    with open(os.path.join(out_dir, "best.pkl"), "wb") as f:
                        pickle.dump(ck | dict(round=rnd, greedy=g), f)
            with open(os.path.join(out_dir, "results.json"), "w") as f:
                json.dump(dict(config=cfg_d, rounds=history,
                               selected=dict(round=best[1], greedy=best[2], base_rescues=best[0][0],
                                             base_score=best[0][1])), f, indent=1)
    log(f"selected round {best[1]} (greedy={best[2]}) by base-game rescues {best[0][0]:.2f}, score {best[0][1]:.0f}")


if __name__ == "__main__":
    main()
