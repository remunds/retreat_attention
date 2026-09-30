"""Program search: find a DSL program (see programs.py) for each object's next displacement.

Divide-and-conquer enumerative synthesis (in the spirit of EUSolver):
  1. Enumerate all terms up to --max-size bottom-up over projections, action flags, small
     constants, + - * neg sign. Terms with identical outputs on the data are merged (observational
     equivalence), keeping the smallest.
  2. At a node, leaf candidates are the enumerated terms plus terms with fitted constants
     (a*t + b and a*t1 + b*t2 + c, least squares, rounded to 3 decimals). A candidate "explains"
     a transition if it is within --tol of the true displacement.
  3. If no candidate explains (almost) all transitions at the node, split on a condition
     `term <= threshold` (term from the small enumerated terms, threshold from data quantiles)
     that maximizes the number of transitions the two children can explain, and recurse.
The result is one `ite` program per coordinate. Transitions no program can explain (hidden
state: the enemy's pause every 8th frame, the ball held 60 frames after a goal) end up as the
minority inside a leaf.

Usage:
    uv run python -m ocwm.synthesis --data data/train.npz --out runs/programs
"""

import argparse
import os
import time

import numpy as np

from ocwm import data as D
from ocwm.env import OBJECTS
from ocwm.programs import ACTIONS, ATTRS, TARGETS, ProgramModel, features, evaluate, size, to_str

CONSTANTS = (0.0, 1.0, 2.0, -1.0)


def build_dataset(path, n, seed, surprise_frac=0.5, history=2):
    """Features and targets for n transitions. Half are drawn evenly from each object's 'surprising'
    transitions (bounces, goals, enemy pauses, paddle accelerations), so rare events are present."""
    d = D.load(path)
    idx = D.valid_indices(d["done"], history)
    rng = np.random.default_rng(seed)
    sur = D.surprise_masks(d["pos"], idx)
    pools = [np.nonzero(sur[:, i])[0] for i in range(sur.shape[1]) if sur[:, i].any()]
    n_each = int(n * surprise_frac) // len(pools)
    pick = np.concatenate([rng.choice(pool, n_each) for pool in pools] + [rng.choice(len(idx), n - n_each * len(pools))])
    idx = idx[pick]
    hist = np.stack([d["pos"][idx[:, 0], idx[:, 1] - 1], d["pos"][idx[:, 0], idx[:, 1]]], 1).astype(np.float32)
    action = d["action"][idx[:, 0], idx[:, 1]]
    delta = d["pos"][idx[:, 0], idx[:, 1] + 1] - hist[:, -1]
    targets = {(o, dd): delta[:, i, j].astype(np.float32) for i, o in enumerate(OBJECTS) for j, dd in enumerate(("dx", "dy"))}
    return features(hist, action, np), targets


def enumerate_terms(f, max_size, excluded=()):
    """Bottom-up enumeration with observational equivalence. Returns [(node, values, size)]."""
    seen, by_size = set(), {}

    def add(node, vals, s):
        if not np.all(np.isfinite(vals)):
            return
        key = np.round(vals, 4).tobytes()
        if key in seen:
            return
        seen.add(key)
        by_size.setdefault(s, []).append((node, vals.astype(np.float32)))

    for c in CONSTANTS:  # constants first, so they represent their value class
        add(("const", c), np.full_like(f[("ball", "x")], c), 1)
    for o in OBJECTS:
        if o in excluded:
            continue
        for a in ATTRS:
            if np.ptp(f[(o, a)]) > 0:  # constant projections (e.g. player.x) carry no information
                add(("proj", o, a), f[(o, a)], 1)
    for name in ACTIONS:
        add(("act", name), f[name], 1)

    for s in range(2, max_size + 1):
        for node, v in by_size.get(s - 1, []):
            add(("neg", node), -v, s)
            add(("sign", node), np.sign(v), s)
        for s1 in range(1, s - 1):
            s2 = s - 1 - s1
            for n1, v1 in by_size.get(s1, []):
                for n2, v2 in by_size.get(s2, []):
                    if s1 <= s2:  # add and mul are commutative
                        add(("add", n1, n2), v1 + v2, s)
                        add(("mul", n1, n2), v1 * v2, s)
                    add(("sub", n1, n2), v1 - v2, s)
    return [(node, v, s) for s, items in sorted(by_size.items()) for node, v in items]


def trimmed_lstsq(A, t, tol, iters=3):
    """Least squares, refit on the points the current fit explains (robust to rare exceptions)."""
    coef = np.linalg.lstsq(A, t, rcond=None)[0]
    for _ in range(iters):
        inl = np.abs(A @ coef - t) <= max(tol, 0.5 * np.median(np.abs(A @ coef - t)))
        if inl.sum() < A.shape[1] + 2:
            break
        coef = np.linalg.lstsq(A[inl], t[inl], rcond=None)[0]
    return coef


def affine(a, t, b):
    """DSL node for a*t + b, simplified."""
    node = t if a == 1 else ("neg", t) if a == -1 else ("mul", ("const", a), t)
    return node if b == 0 else ("add", node, ("const", b))


class Synthesizer:
    def __init__(self, f, terms, tol, max_depth, min_leaf, fit_size=4, split_size=3, n_thresholds=32, top_k=150):
        self.f, self.tol, self.max_depth, self.min_leaf = f, tol, max_depth, min_leaf
        self.nodes = [t[0] for t in terms]
        self.values = np.stack([t[1] for t in terms])  # [T, N]
        sizes = np.array([t[2] for t in terms])
        self.fit_ids = np.nonzero(sizes <= fit_size)[0]
        split_ids = [i for i in np.nonzero(sizes <= split_size)[0] if np.unique(self.values[i]).size > 1]
        self.split_ids = np.array(split_ids)
        self.split_sizes = sizes[self.split_ids]
        self.n_thresholds, self.top_k = n_thresholds, top_k

    def candidates(self, y, rows):
        """Leaf candidates at a node: (nodes, predictions [K, n]) for enumerated and fitted terms."""
        V, t = self.values[:, rows], y[rows]
        hit = np.abs(V - t) <= self.tol  # [T, n]
        # Greedy cover: each pick explains the most transitions the earlier picks miss, so rare
        # behaviours (e.g. -ball.vx at a bounce) are candidates next to the common ones.
        top, covered = [], np.zeros(len(t), bool)
        for _ in range(self.top_k // 3):
            gain = (hit & ~covered).sum(1)
            i = int(np.argmax(gain))
            if gain[i] == 0:
                break
            top.append(i)
            covered |= hit[i]
        top += [i for i in np.argsort(-hit.sum(1)) if i not in top][: self.top_k - len(top)]
        nodes, preds = [self.nodes[i] for i in top], [V[i] for i in top]

        X = self.values[self.fit_ids][:, rows]  # single-term fits a*x + b, vectorized over terms
        a, b = np.zeros(len(X)), np.zeros(len(X))
        w = np.ones_like(X, dtype=bool)
        for _ in range(4):  # trimmed: refit on the points the current fit explains
            n_w = np.maximum(w.sum(1), 1)
            xm, tm = (X * w).sum(1) / n_w, (t * w).sum(1) / n_w
            var = (((X - xm[:, None]) ** 2) * w).sum(1)
            a = np.where(var > 1e-9, (((X - xm[:, None]) * (t - tm[:, None])) * w).sum(1) / np.maximum(var, 1e-9), 0.0)
            b = tm - a * xm
            res = np.abs(a[:, None] * X + b[:, None] - t)
            w = res <= np.maximum(self.tol, 0.5 * np.median(res, 1, keepdims=True))
        a, b = np.round(a, 3), np.round(b, 3)
        P = a[:, None] * X + b[:, None]
        good = (np.abs(P - t) <= self.tol).sum(1)
        best1 = np.argsort(-good)[:40]
        for j in best1:
            nodes.append(affine(float(a[j]), self.nodes[self.fit_ids[j]], float(b[j])))
            preds.append(P[j])
        for i, j in [(i, j) for ii, i in enumerate(best1[:30]) for j in best1[ii + 1 : 30]]:  # two-term fits
            A = np.stack([X[i], X[j], np.ones_like(t)], 1)
            coef = np.round(trimmed_lstsq(A, t, self.tol), 3)
            n1, n2 = self.nodes[self.fit_ids[i]], self.nodes[self.fit_ids[j]]
            nodes.append(affine(1.0, ("add", affine(float(coef[0]), n1, 0.0), affine(float(coef[1]), n2, 0.0)), float(coef[2])))
            preds.append(A @ coef)
        nodes.append(("const", float(np.round(np.median(t), 3))))
        preds.append(np.full_like(t, np.round(np.median(t), 3)))
        return nodes, np.stack(preds)

    def solve(self, y, rows, depth=0):
        nodes, preds = self.candidates(y, rows)
        E = (np.abs(preds - y[rows]) <= self.tol).astype(np.float32)  # [K, n]
        counts = E.sum(1)
        best = int(np.argmax(counts - 1e-3 * np.array([size(n) for n in nodes])))  # ties -> smaller program
        if counts[best] >= 0.995 * len(rows) or depth >= self.max_depth or len(rows) < 2 * self.min_leaf:
            return nodes[best]

        F = self.values[self.split_ids][:, rows]  # [S, n]
        # Thresholds: half at mass quantiles, half evenly over the value range, so that rare regions
        # (e.g. the ball at the paddle column) can be split off too.
        half = self.n_thresholds // 2
        lo, hi = F.min(1, keepdims=True), F.max(1, keepdims=True)
        qs = np.concatenate(
            [np.quantile(F, np.linspace(0.02, 0.98, half), axis=1).T, lo + (hi - lo) * np.linspace(0.01, 0.99, self.n_thresholds - half)],
            1,
        )  # [S, Q]
        masks = (F[:, None, :] <= qs[:, :, None]).reshape(-1, len(rows)).astype(np.float32)  # [S*Q, n]
        # Information gain on "which candidate explains this transition" labels (first explaining
        # candidate in greedy-cover order). Unlike counting explained transitions, this rewards a
        # split that only pays off after a further split (e.g. ball at the paddle column, then
        # ball level with the paddle); reduced-error pruning removes splits that never pay off.
        label = np.where(E.any(0), E.argmax(0), len(E))
        _, label = np.unique(label, return_inverse=True)
        onehot = np.eye(label.max() + 1, dtype=np.float32)[label]  # [n, L]
        n = len(rows)

        def entropy(c):
            p = c / np.maximum(c.sum(-1, keepdims=True), 1)
            return -(p * np.log(np.where(p > 0, p, 1))).sum(-1)

        def split_gain(masks):
            left = masks @ onehot  # [M, L]
            n_left = left.sum(1)
            right = onehot.sum(0) - left
            gain = entropy(onehot.sum(0)) - (n_left * entropy(left) + (n - n_left) * entropy(right)) / n
            gain[(n_left < self.min_leaf) | (n_left > n - self.min_leaf)] = -1
            return gain

        penalty = 1e-6 * self.split_sizes  # ties -> smaller term
        gain = split_gain(masks) - np.repeat(penalty, self.n_thresholds)
        if gain.max() < 1e-3:
            return nodes[best]
        # Refine the best few terms: try every data value near their best grid threshold.
        per_term = gain.reshape(-1, self.n_thresholds)
        best_s, best_thr, best_gain = None, None, -np.inf
        for s in np.argsort(-per_term.max(1))[:5]:
            q = int(np.argmax(per_term[s]))
            grid = np.sort(qs[s])
            j = np.searchsorted(grid, qs[s, q])
            lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
            vals = np.unique(np.append(F[s][(F[s] >= lo) & (F[s] <= hi)], qs[s, q]))
            fine = split_gain((F[s][None, :] <= vals[:, None]).astype(np.float32)) - penalty[s]
            if fine.max() > best_gain:
                best_s, best_thr, best_gain = s, float(vals[int(np.argmax(fine))]), fine.max()
        s = best_s
        term, thr = self.nodes[self.split_ids[s]], best_thr
        go_left = F[s] <= thr
        then = self.solve(y, rows[go_left], depth + 1)
        other = self.solve(y, rows[~go_left], depth + 1)
        return then if then == other else ("ite", ("le", term, thr), then, other)


def prune(node, f, y, tol, rows):
    """Reduced-error pruning on held-out data: replace an `ite` by one of its branches when that
    explains at least as many of the held-out transitions reaching it."""
    if node[0] != "ite" or rows.sum() == 0:
        return node
    cond = evaluate(node[1], f, np)
    then = prune(node[2], f, y, tol, rows & cond)
    other = prune(node[3], f, y, tol, rows & ~cond)
    node = ("ite", node[1], then, other)
    hits = lambda n: int((np.abs(evaluate(n, f, np) - y) <= tol)[rows].sum())
    best = max([node, then, other], key=lambda n: (hits(n), -size(n)))
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="directory; writes programs.json and programs.txt")
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--max-size", type=int, default=4, help="max enumerated term size")
    p.add_argument("--max-depth", type=int, default=12, help="max nesting of if-conditions")
    p.add_argument("--min-leaf", type=int, default=10)
    p.add_argument("--exclude", nargs="*", default=[], help="objects to lock away, e.g. enemy: no program may read them, and they are not modeled")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    f, targets = build_dataset(args.data, args.samples, args.seed)
    f_val, targets_val = build_dataset(args.data, args.samples, args.seed + 1)  # for pruning
    f_test, targets_test = build_dataset(args.data, args.samples, args.seed + 2)  # for reporting
    t0 = time.time()
    terms = enumerate_terms(f, args.max_size, args.exclude)
    print(f"enumerated {len(terms)} distinct terms up to size {args.max_size} in {time.time() - t0:.1f}s", flush=True)

    programs = {}
    for o, d in TARGETS:
        if o in args.exclude:
            programs[(o, d)] = ("const", 0.0)  # not modeled: stays where it was last seen
            continue
        tol = 0.05 if o == "player" else 0.5  # player y is continuous, ball and enemy are on the pixel grid
        synth = Synthesizer(f, terms, tol, args.max_depth, args.min_leaf)
        t0 = time.time()
        prog = synth.solve(targets[(o, d)], np.arange(args.samples))
        prog = prune(prog, f_val, targets_val[(o, d)], tol, np.ones(args.samples, bool))
        programs[(o, d)] = prog
        err = np.abs(evaluate(prog, f_test, np) - targets_test[(o, d)])
        print(
            f"{o}.{d}: size {size(prog)}, {time.time() - t0:.1f}s | held-out: exact {np.mean(err <= tol):.1%}, MAE {err.mean():.3f} px",
            flush=True,
        )

    model = ProgramModel(programs, args.exclude)
    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "programs.json"))
    with open(os.path.join(args.out, "programs.txt"), "w") as fh:
        fh.write(model.to_text() + "\n")
    print("\n" + model.to_text())
    print(f"\nsaved {args.out}/programs.json and programs.txt")


if __name__ == "__main__":
    main()
