"""Evaluate a finished, already-selected checkpoint on unmodified Pong and on the `lazy_enemy` mod.

This script is for validation only: it never trains or selects anything. The checkpoint (and its
greedy/sampling mode) must have been chosen beforehand on unmodified Pong by the experiment script.

It reports:
- points per game of the actor (player and enemy), over `--games` full games with different seeds,
- per-object world-model prediction error (1-step and open-loop) on trajectories of the actor
  in both environments,
- average interaction gates of the world model, and a plot of the gates along one trajectory.

Usage:
    CUDA_VISIBLE_DEVICES=4 uv run evaluate_lazy_enemy.py --module experiment_sparse_object_wm \
        --ckpt runs/sow_v1/best.pkl
"""

import argparse
import importlib
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import common


def wm_errors(mod, cfg, wm_params, traj, horizons=(1, 5, 10, 20)):
    """Per-object RMSE (pixels) of open-loop world-model predictions on real trajectories."""
    buf = common.ReplayBuffer()
    buf.add(traj)
    K, Hmax = cfg["K"], max(horizons)
    env_idx, t_idx = buf.valid_starts(K, Hmax)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(env_idx), size=min(8192, len(env_idx)), replace=False)
    arr = buf.arrays()
    fo = np.arange(-K + 1, Hmax + 1)
    f = jnp.asarray(arr["frames"][env_idx[sel, None], t_idx[sel, None] + fo[None]])
    a = jnp.asarray(arr["action"][env_idx[sel, None], t_idx[sel, None] + np.arange(Hmax)[None]])
    model = mod.build_model(cfg)

    @jax.jit
    def rollout(f, a):
        def body(hist, x):
            act = x
            pred, _, _, gates = mod.wm_step(model, wm_params, hist, act)
            return jnp.concatenate([hist[:, 1:], pred[:, None]], 1), (pred, gates)

        _, (preds, gates) = jax.lax.scan(body, f[:, :K], a.T)
        return preds.transpose(1, 0, 2, 3), gates.transpose(1, 0, 2, 3)

    preds, gates = rollout(f, a)
    target = f[:, K:]
    err = np.sqrt(np.mean(np.sum((np.asarray(preds) - np.asarray(target)) ** 2, -1), axis=0))  # (H, 3)
    res = {f"h{h}": {n: float(err[h - 1, i]) for i, n in enumerate(common.OBJECT_NAMES)} for h in horizons}
    g1 = np.asarray(gates[:, 0]).mean(0)  # gates at the first (teacher-forced history) step
    res["mean_gates_step1"] = {n: {m: float(g1[i, j]) for j, m in enumerate(common.OBJECT_NAMES + ("action",))}
                               for i, n in enumerate(common.OBJECT_NAMES)}
    return res


def plot_gates(mod, cfg, wm_params, traj, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = cfg["K"]
    frames = np.concatenate([np.asarray(traj["pos"][0]), np.asarray(traj["next"][0, -1:])], 0)
    actions = np.asarray(traj["action"][0])
    T = 300
    model = mod.build_model(cfg)
    hists = np.stack([frames[t - K + 1:t + 1] for t in range(K - 1, K - 1 + T)])
    _, _, _, gates = mod.wm_step(model, wm_params, jnp.asarray(hists), jnp.asarray(actions[K - 1:K - 1 + T]))
    gates = np.asarray(gates)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ts = np.arange(T)
    axes[0].plot(ts, hists[:, -1, 2, 0], label="ball x")
    axes[0].plot(ts, hists[:, -1, 2, 1], label="ball y")
    axes[0].plot(ts, hists[:, -1, 0, 1], label="player y")
    axes[0].plot(ts, hists[:, -1, 1, 1], label="enemy y")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_ylabel("pixels")
    tok = common.OBJECT_NAMES + ("action",)
    for i, n in enumerate(common.OBJECT_NAMES):
        for j, m in enumerate(tok):
            if i != j:
                axes[i + 1].plot(ts, gates[:, i, j], label=f"{n} <- {m}")
        axes[i + 1].set_ylim(-0.05, 1.05)
        axes[i + 1].set_ylabel(f"gates of {n}")
        axes[i + 1].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("agent step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="experiment_sparse_object_wm")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    mod = importlib.import_module(args.module)
    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    cfg, greedy = ck["cfg"], bool(ck.get("greedy", False))
    act = mod.ACT_GREEDY if greedy else mod.ACT_SAMPLE
    out_dir = os.path.dirname(args.ckpt)
    results = dict(ckpt=args.ckpt, round=ck.get("round"), greedy=greedy, games=args.games, seed=args.seed)
    key = jax.random.PRNGKey(args.seed)
    for name, mods in (("pong", None), ("lazy_enemy", ["lazy_enemy"])):
        env = common.make_env(mods)
        k_eval, k_col = jax.random.split(jax.random.fold_in(key, 0))  # same seeds for both envs
        ps, es, over = common.evaluate(env, act, args.max_steps, ck["ac_params"], jax.random.split(k_eval, args.games))
        ps, es = np.asarray(ps), np.asarray(es)
        traj = common.collect(env, act, 2000, ck["ac_params"], jax.random.split(k_col, 32))
        results[name] = dict(
            player_mean=float(ps.mean()), player_std=float(ps.std()), enemy_mean=float(es.mean()),
            finished=float(np.asarray(over).mean()), player_per_game=ps.tolist(), enemy_per_game=es.tolist(),
            wm_rmse_px=wm_errors(mod, cfg, ck["wm_params"], traj),
        )
        plot_gates(mod, cfg, ck["wm_params"], traj, os.path.join(out_dir, f"gates_{name}.png"),
                   f"{args.ckpt} on {name}")
        print(f"{name}: player {ps.mean():.2f} +- {ps.std():.2f}, enemy {es.mean():.2f}, finished {np.asarray(over).mean():.2f}")
        print(json.dumps(results[name]["wm_rmse_px"], indent=1))
    for name in ("pong", "lazy_enemy"):
        results[name]["final_score_mean"] = results[name]["player_mean"] - results[name]["enemy_mean"]
    ok = results["lazy_enemy"]["final_score_mean"] >= 10
    print(f"final score (player - enemy): pong {results['pong']['final_score_mean']:+.2f}, "
          f"lazy_enemy {results['lazy_enemy']['final_score_mean']:+.2f}")
    with open(os.path.join(out_dir, "lazy_enemy_eval.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"lazy_enemy criterion (final score >= +10 on average over {args.games} games): {'MET' if ok else 'NOT met'}")


if __name__ == "__main__":
    main()
