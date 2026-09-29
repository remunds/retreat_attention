"""Evaluate a finished, already-selected Seaquest checkpoint on the base game and the `gravity` mod.

Validation only: nothing is trained or selected here. The checkpoint (and its greedy/sampled mode)
must have been chosen beforehand on the base game by the experiment script.

Reports per environment, over `--games` full games with different seeds (the same seeds for both):
successful rescues per game (surfacing with 6 divers; the goal is >= 2 on average with `gravity`),
divers picked up, score, lives lost and game length. Writes `gravity_eval.json` next to the
checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES=4 uv run evaluate_gravity.py --module experiment_seaquest_ppo --ckpt runs/sq_ppo_a/best.pkl
"""

import argparse
import importlib
import json
import os
import pickle

import jax
import numpy as np

import seaquest_common as sc

TARGET_RESCUES = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--max_steps", type=int, default=27000)  # 108k frames, the usual Atari cap
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    mod = importlib.import_module(args.module)
    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    greedy = bool(ck.get("greedy", False))
    act = mod.ACT_GREEDY if greedy else mod.ACT_SAMPLE
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.games)
    results = dict(ckpt=args.ckpt, round=ck.get("round"), greedy=greedy, games=args.games, seed=args.seed,
                   max_steps=args.max_steps, game="seaquest", target_rescues=TARGET_RESCUES)
    for name, mods in (("base", None), ("gravity", ["gravity"])):
        out = sc.evaluate(sc.make_env(mods), act, args.max_steps, ck["ac_params"], keys)
        out = {k: np.asarray(v) for k, v in out.items()}
        results[name] = dict(
            rescues_mean=float(out["rescues"].mean()), rescues_per_game=out["rescues"].tolist(),
            games_with_2_rescues=int((out["rescues"] >= 2).sum()),
            divers_mean=float(out["divers"].mean()), score_mean=float(out["score"].mean()),
            score_per_game=out["score"].tolist(), lives_lost_mean=float(out["lives_lost"].mean()),
            steps_mean=float(out["steps"].mean()), finished=float(out["finished"].mean()))
        r = results[name]
        print(f"{name}: rescues/game {r['rescues_mean']:.2f} (games with >=2: {r['games_with_2_rescues']}/{args.games}), "
              f"divers {r['divers_mean']:.1f}, score {r['score_mean']:.0f}, steps {r['steps_mean']:.0f}, "
              f"finished {r['finished']:.2f}", flush=True)
    out_path = os.path.join(os.path.dirname(args.ckpt), "gravity_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    ok = results["gravity"]["rescues_mean"] >= TARGET_RESCUES
    print(f"gravity criterion (>= {TARGET_RESCUES:.0f} rescues per game on average over {args.games} games): "
          f"{'MET' if ok else 'NOT met'} -> {out_path}")


if __name__ == "__main__":
    main()
