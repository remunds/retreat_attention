"""Select a checkpoint using only unmodified Pong: normal play plus a frozen-enemy-view check.

For every round checkpoint of the given runs and both action modes (greedy / sampled), this plays
`--games` full games of unmodified Pong normally and `--games` games in which the actor sees the
enemy frozen at a random height (`common.evaluate_frozen_enemy_view`). The selection score is the
mean final score (player minus enemy points) over both. `lazy_enemy` is not used.

The selected checkpoint is written to `<first run>/best_robust.pkl` (or `--out`) together with a JSON
summary, and can then be evaluated once with `evaluate_lazy_enemy.py`.

Usage:
    CUDA_VISIBLE_DEVICES=4 uv run select_robust_checkpoint.py --module experiment_observation_interventions \
        --runs runs/oi_p01_s0 runs/oi_p01_s1 --out runs/oi_p01_selected.pkl
"""

import argparse
import glob
import importlib
import json
import os
import pickle
import re

import jax
import numpy as np

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    mod = importlib.import_module(args.module)
    env = common.make_env()  # unmodified Pong only
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.games)
    rows, best = [], None
    for run in args.runs:
        for path in sorted(glob.glob(os.path.join(run, "round*.pkl")), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
            with open(path, "rb") as f:
                ck = pickle.load(f)
            for greedy in (True, False):
                act = mod.ACT_GREEDY if greedy else mod.ACT_SAMPLE
                ps, es, _ = common.evaluate(env, act, args.max_steps, ck["ac_params"], keys)
                fps, fes, _ = common.evaluate_frozen_enemy_view(env, act, args.max_steps, ck["ac_params"], keys)
                normal = float(np.mean(ps - es))
                frozen = float(np.mean(fps - fes))
                row = dict(ckpt=path, greedy=greedy, pong_final=normal, frozen_view_final=frozen,
                           score=(normal + frozen) / 2)
                rows.append(row)
                print(f"{path} greedy={greedy}: pong {normal:+.2f}  frozen-view {frozen:+.2f}  score {row['score']:+.2f}", flush=True)
                if best is None or row["score"] > best[0]["score"]:
                    best = (row, ck)
    row, ck = best
    out = args.out or os.path.join(args.runs[0], "best_robust.pkl")
    with open(out, "wb") as f:
        pickle.dump(ck | dict(greedy=row["greedy"], selected_from=row["ckpt"]), f)
    with open(os.path.splitext(out)[0] + "_selection.json", "w") as f:
        json.dump(dict(selected=row, candidates=rows, args=vars(args)), f, indent=1)
    print(f"selected {row['ckpt']} greedy={row['greedy']} -> {out}")


if __name__ == "__main__":
    main()
