"""Compute the model-dependent data behind the paper's figures (trajectories, gates, rollouts).

Writes `paper/data/*.npz` / `*.json`; `make_figures.py` turns them (and `runs/*`) into PDFs.
`lazy_enemy` is used here only to *evaluate / visualise* finished, already-selected checkpoints.

Usage (from the repository root):
    CUDA_VISIBLE_DEVICES=4 uv run paper/compute_figure_data.py
"""

import importlib
import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS = {
    # name: (module, checkpoint)
    "ours": ("experiment_observation_interventions", "runs/oi_p01_s0/robust/best.pkl"),
    "baseline": ("experiment_sparse_object_wm", "runs/sow_v1/robust/best.pkl"),
}
T_TRACE = 600


def load(name):
    mod_name, path = MODELS[name]
    mod = importlib.import_module(mod_name)
    with open(path, "rb") as f:
        ck = pickle.load(f)
    act = mod.ACT_GREEDY if ck.get("greedy", False) else mod.ACT_SAMPLE
    return mod, ck, act


def traces():
    """One game segment per (model, env): object positions, actions, rewards and WM gates."""
    out = {}
    for name in MODELS:
        mod, ck, act = load(name)
        model = mod.build_model(ck["cfg"])
        K = ck["cfg"]["K"]
        for env_name, mods in (("pong", None), ("lazy_enemy", ["lazy_enemy"])):
            env = common.make_env(mods)
            traj = common.collect(env, act, T_TRACE + K, ck["ac_params"], jax.random.split(jax.random.PRNGKey(3), 1))
            frames = np.concatenate([np.asarray(traj["pos"][0]), np.asarray(traj["next"][0, -1:])], 0)
            actions = np.asarray(traj["action"][0])
            hists = np.stack([frames[t - K + 1:t + 1] for t in range(K - 1, K - 1 + T_TRACE)])
            _, _, _, gates = mod.wm_step(model, ck["wm_params"], jnp.asarray(hists), jnp.asarray(actions[K - 1:K - 1 + T_TRACE]))
            out[f"{name}_{env_name}_pos"] = hists[:, -1]
            out[f"{name}_{env_name}_reward"] = np.asarray(traj["reward"][0, K - 1:K - 1 + T_TRACE])
            out[f"{name}_{env_name}_gates"] = np.asarray(gates)
    np.savez(os.path.join(OUT, "traces.npz"), **out)


def rollouts():
    """Open-loop imagination vs. reality for our world model, on Pong and lazy_enemy trajectories."""
    mod, ck, act = load("ours")
    model = mod.build_model(ck["cfg"])
    K, H = ck["cfg"]["K"], 40
    out = {}
    for env_name, mods in (("pong", None), ("lazy_enemy", ["lazy_enemy"])):
        env = common.make_env(mods)
        traj = common.collect(env, act, 2000, ck["ac_params"], jax.random.split(jax.random.PRNGKey(5), 16))
        buf = common.ReplayBuffer()
        buf.add(traj)
        ei, ti = buf.valid_starts(K, H)
        arr = buf.arrays()
        rng = np.random.default_rng(0)
        sel = rng.choice(len(ei), size=min(4096, len(ei)), replace=False)
        fo = np.arange(-K + 1, H + 1)
        f = jnp.asarray(arr["frames"][ei[sel, None], ti[sel, None] + fo[None]])
        a = jnp.asarray(arr["action"][ei[sel, None], ti[sel, None] + np.arange(H)[None]])

        def body(hist, act_t):
            pred = mod.wm_step(model, ck["wm_params"], hist, act_t)[0]
            return jnp.concatenate([hist[:, 1:], pred[:, None]], 1), pred

        _, preds = jax.lax.scan(body, f[:, :K], a.T)
        preds = np.asarray(preds.transpose(1, 0, 2, 3))
        real = np.asarray(f[:, K:])
        err = np.sqrt(np.mean(np.sum((preds - real) ** 2, -1), axis=0))  # (H, 3)
        out[f"{env_name}_rmse"] = err
        # one example window where the ball travels towards the enemy (for an illustration)
        bx = np.asarray(f[:, K - 1, 2, 0])
        vx = np.asarray(f[:, K - 1, 2, 0] - f[:, K - 2, 2, 0])
        idx = int(np.nonzero((bx > 100) & (vx < 0))[0][0])
        out[f"{env_name}_example_real"] = np.asarray(f[idx])
        out[f"{env_name}_example_pred"] = preds[idx]
    np.savez(os.path.join(OUT, "rollouts.npz"), **out)


def leakage_by_gap():
    """Counterfactual enemy swap for experiment 1's round-0 world model, binned by the gap."""
    mod = importlib.import_module("experiment_sparse_object_wm")
    with open("runs/sow_v1/round0.pkl", "rb") as f:
        ck = pickle.load(f)
    model = mod.build_model(ck["cfg"])
    env = common.make_env()
    traj = common.collect(env, common.random_act, 2000, None, jax.random.split(jax.random.PRNGKey(7), 32))
    buf = common.ReplayBuffer()
    buf.add(traj)
    ei, ti = buf.valid_starts(16, 1)
    sel = np.random.default_rng(0).choice(len(ei), 16384, replace=False)
    arr = buf.arrays()
    fo = np.arange(-15, 1)
    f = jnp.asarray(arr["frames"][ei[sel, None], ti[sel, None] + fo[None]])
    a = jnp.asarray(arr["action"][ei[sel], ti[sel]])
    pf = jax.jit(lambda h, a_: mod.wm_step(model, ck["wm_params"], h, a_)[0])
    base = np.asarray(pf(f, a))
    perm = np.random.default_rng(1).permutation(len(sel))
    sw = f.at[:, :, 1].set(f[perm][:, :, 1])
    d = np.linalg.norm(np.asarray(pf(sw, a)) - base, axis=-1)[:, 2]
    bx = np.asarray(f[:, -1, 2, 0])
    moving = np.any(np.asarray(f[:, -1, 2] != f[:, -3, 2]), axis=-1)
    gap_real = np.abs(np.asarray(f[:, -1, 1, 1] - f[:, -1, 2, 1]))
    gap_sw = np.abs(np.asarray(sw[:, -1, 1, 1] - f[:, -1, 2, 1]))
    far = moving & (bx > 56)
    edges = [0, 10, 20, 40, 60, 100, 200]
    res = dict(edges=edges, leak_rmse=[], n=[], real_gap_hist=np.histogram(gap_real[far], bins=edges)[0].tolist())
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = far & (gap_sw >= lo) & (gap_sw < hi)
        res["leak_rmse"].append(float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else float("nan"))
        res["n"].append(int(m.sum()))
    res["exp1_leakage"] = common.leakage(pf, f, a, jax.random.PRNGKey(1))
    with open(os.path.join(OUT, "leakage_by_gap.json"), "w") as f_:
        json.dump(res, f_, indent=1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["traces", "rollouts", "leakage"]
    if "traces" in which:
        traces()
    if "rollouts" in which:
        rollouts()
    if "leakage" in which:
        leakage_by_gap()
    print("done", which)
