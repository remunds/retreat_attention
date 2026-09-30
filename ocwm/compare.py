"""Compare players with per-game statistics, confidence intervals and significance tests.

Step 1 (one process per player, can run in parallel): play N games on the training game and on
the evaluation game, save per-game results.
    uv run python -m ocwm.compare run --ckpt runs/programs/programs.json --name programs --out runs/compare
Step 2: summarize all saved players.
    uv run python -m ocwm.compare summarize --out runs/compare
"""

import argparse
import glob
import itertools
import os

import jax
import numpy as np
from scipy import stats

from ocwm import actor as A
from ocwm import model as M
from ocwm.env import EVAL_MOD, make_env

GAMES = {"train": [], "eval": [EVAL_MOD]}
METRICS = {
    "won (reached 21)": lambda d: (d["player"] >= 21).astype(float),
    "player points / 1000 frames": lambda d: 1000 * d["player"] / d["frames"],
    "enemy points / 1000 frames": lambda d: 1000 * d["enemy"] / d["frames"],
    "winners per hit": lambda d: d["player"] / np.maximum(d["hits"], 1),
}
HIGHER_IS_BETTER = {"enemy points / 1000 frames": False}


def run(args):
    model = M.load(args.ckpt)
    out = {}
    for game, mods in GAMES.items():
        env = make_env(mods)
        res = A.play(env, A.model_planner(model, args.horizon), jax.random.PRNGKey(args.seed), args.episodes, model.config.history, args.max_steps)
        for key, v in zip(("player", "enemy", "hits", "frames"), res):
            out[f"{game}/{key}"] = np.asarray(v)
        print(f"{args.name} {game}: " + "  ".join(f"{a}-{b}" for a, b in zip(out[f"{game}/player"], out[f"{game}/enemy"])), flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, f"{args.name}.npz"), **out)


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x))).mean(1)
    return np.percentile(means, [2.5, 97.5])


def summarize(args):
    players = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in sorted(glob.glob(os.path.join(args.out, "*.npz")))}
    for game in GAMES:
        print(f"\n=== {game} game ({'unmodified' if not GAMES[game] else ', '.join(GAMES[game])}) — mean [95% bootstrap CI], n games per player")
        per = {}
        for name, d in players.items():
            g = {k.split("/")[1]: v for k, v in d.items() if k.startswith(game + "/")}
            per[name] = {m: f(g) for m, f in METRICS.items()}
            n = len(g["player"])
            print(f"  {name} (n={n})")
            for m, v in per[name].items():
                lo, hi = bootstrap_ci(v)
                print(f"    {m:28s} {v.mean():7.3f}  [{lo:.3f}, {hi:.3f}]")
        print(f"  pairwise (Mann-Whitney U, two-sided; Fisher exact for wins)")
        for a, b in itertools.combinations(per, 2):
            for m in METRICS:
                x, y = per[a][m], per[b][m]
                if m.startswith("won"):
                    table = [[x.sum(), len(x) - x.sum()], [y.sum(), len(y) - y.sum()]]
                    p = stats.fisher_exact(table)[1]
                else:
                    p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
                diff = x.mean() - y.mean()
                better = a if (diff > 0) == HIGHER_IS_BETTER.get(m, True) else b
                verdict = f"{better} better" if p < 0.05 else "no significant difference"
                print(f"    {a} vs {b} | {m:28s} diff {diff:+7.3f}  p={p:.2g}  -> {verdict}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--episodes", type=int, default=32)
    r.add_argument("--max-steps", type=int, default=30000)
    r.add_argument("--horizon", type=int, default=256)
    r.add_argument("--seed", type=int, default=1, help="differs from the benchmark's seed 0: fresh games")
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    args = p.parse_args()
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()
