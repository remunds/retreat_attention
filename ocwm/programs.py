"""A small DSL for object dynamics, and a world model made of one program per object coordinate.

Programs are nested tuples (JSON-serializable as lists):
    ("proj", obj, attr)     obj in player/enemy/ball, attr in x/y/vx/vy (v = pos_t - pos_{t-1})
    ("act", name)           1.0 if the action moves up / down / fires, else 0.0
    ("const", c)
    ("add"|"sub"|"mul", a, b), ("neg"|"sign", a)
    ("le", term, threshold) boolean condition
    ("ite", cond, then, else)

A program for obj.dx / obj.dy predicts the object's next displacement. Which sources a program
reads on the branch it actually executes is that object's dependency set at that step.
"""

import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from ocwm.env import OBJECTS

ATTRS = ("x", "y", "vx", "vy")
ACTIONS = {"up": (2, 4), "down": (3, 5), "fire": (1, 4, 5)}
SOURCES = OBJECTS + ("action",)
TARGETS = [(o, d) for o in OBJECTS for d in ("dx", "dy")]


def features(hist, action, xp=np):
    """Projections from history [..., H>=2, 3, 2] and action [...]."""
    pos, prev = hist[..., -1, :, :], hist[..., -2, :, :]
    f = {}
    for i, o in enumerate(OBJECTS):
        f[(o, "x")], f[(o, "y")] = pos[..., i, 0], pos[..., i, 1]
        f[(o, "vx")], f[(o, "vy")] = pos[..., i, 0] - prev[..., i, 0], pos[..., i, 1] - prev[..., i, 1]
    for name, ids in ACTIONS.items():
        f[name] = sum((action == a) for a in ids).astype(pos.dtype)
    return f


def evaluate(node, f, xp=np):
    kind = node[0]
    if kind == "proj":
        return f[(node[1], node[2])]
    if kind == "act":
        return f[node[1]]
    if kind == "const":
        return xp.full_like(f[("ball", "x")], node[1])
    if kind == "le":
        return evaluate(node[1], f, xp) <= node[2]
    if kind == "ite":
        return xp.where(evaluate(node[1], f, xp), evaluate(node[2], f, xp), evaluate(node[3], f, xp))
    args = [evaluate(a, f, xp) for a in node[1:]]
    return {
        "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
        "neg": lambda a: -a, "sign": lambda a: xp.sign(a),
    }[kind](*args)


def static_sources(node):
    """Sources a program reads anywhere in it (all branches)."""
    kind = node[0]
    if kind == "proj":
        return {node[1]}
    if kind == "act":
        return {"action"}
    if kind == "const":
        return set()
    if kind == "le":
        return static_sources(node[1])
    return set().union(*(static_sources(a) for a in node[1:]))


def used_sources(node, f, xp=np):
    """[..., 4] mask of the sources read on the branch each sample actually executes."""
    shape = f[("ball", "x")].shape
    if node[0] == "ite":
        cond = evaluate(node[1], f, xp)
        cond_src = xp.asarray([s in static_sources(node[1]) for s in SOURCES], dtype=xp.float32)
        branch = xp.where(cond[..., None], used_sources(node[2], f, xp), used_sources(node[3], f, xp))
        return xp.maximum(branch, cond_src)
    src = xp.asarray([s in static_sources(node) for s in SOURCES], dtype=xp.float32)
    return xp.broadcast_to(src, shape + (len(SOURCES),))


def size(node):
    if node[0] in ("proj", "act", "const"):
        return 1
    if node[0] == "le":
        return 1 + size(node[1])
    return 1 + sum(size(a) for a in node[1:])


def to_str(node, indent=0):
    kind, pad = node[0], "    " * indent
    if kind == "proj":
        return f"{node[1]}.{node[2]}"
    if kind == "act":
        return node[1]
    if kind == "const":
        return f"{node[1]:g}"
    if kind == "le":
        return f"{to_str(node[1])} <= {node[2]:g}"
    if kind == "ite":
        return (
            f"if {to_str(node[1])}:\n{pad}    {to_str(node[2], indent + 1)}\n"
            f"{pad}else:\n{pad}    {to_str(node[3], indent + 1)}"
        )
    if kind in ("neg", "sign"):
        return f"{'-' if kind == 'neg' else 'sign'}({to_str(node[1])})"
    symbol = {"add": "+", "sub": "-", "mul": "*"}[kind]
    return f"({to_str(node[1])} {symbol} {to_str(node[2])})"


def _linear(node):
    """{atom: coef} with atom None for the constant, if node is linear in non-linear atoms."""
    kind = node[0]
    if kind == "const":
        return {None: node[1]}
    if kind in ("add", "sub"):
        a, b = _linear(node[1]), _linear(node[2])
        sign = 1.0 if kind == "add" else -1.0
        return {k: a.get(k, 0.0) + sign * b.get(k, 0.0) for k in set(a) | set(b)}
    if kind == "neg":
        return {k: -v for k, v in _linear(node[1]).items()}
    if kind == "mul" and node[1][0] == "const":
        return {k: node[1][1] * v for k, v in _linear(node[2]).items()}
    return {node: 1.0}


def simplify(node):
    """Collect linear combinations (e.g. -1.725*(vy - down) + 2.425*vy -> 0.7*vy + 1.725*down)."""
    if node[0] == "ite":
        return ("ite", node[1], simplify(node[2]), simplify(node[3]))
    if node[0] == "le":
        return node
    terms = {k: round(v, 4) for k, v in _linear(node).items() if abs(round(v, 4)) > 1e-4}
    const = terms.pop(None, 0.0)
    out = None
    for atom, coef in sorted(terms.items(), key=lambda kv: to_str(kv[0])):
        t = atom if coef == 1 else ("neg", atom) if coef == -1 else ("mul", ("const", coef), atom)
        out = t if out is None else ("add", out, t)
    if out is None:
        return ("const", const)
    return out if const == 0 else ("add", out, ("const", round(const, 4)))


def _tuplify(x):
    return tuple(_tuplify(v) for v in x) if isinstance(x, list) else x


class ProgramModel:
    """World model from programs {(obj, "dx"|"dy"): node}; same interface as model.WorldModel."""

    config = SimpleNamespace(history=2)

    def __init__(self, programs, excluded=()):
        self.programs = {k: simplify(v) for k, v in programs.items()}
        self.excluded = tuple(excluded)  # objects locked away: never read, not modeled

    def next_positions(self, hist, action):
        """hist [B, H, 3, 2] -> next positions [B, 3, 2] and per-step dependencies [B, 3, 4]."""
        f = features(hist, action, jnp)
        delta = jnp.stack(
            [jnp.stack([evaluate(self.programs[(o, d)], f, jnp) for d in ("dx", "dy")], -1) for o in OBJECTS], 1
        )
        deps = []
        for i, o in enumerate(OBJECTS):
            used = jnp.maximum(*(used_sources(self.programs[(o, d)], f, jnp) for d in ("dx", "dy")))
            deps.append(used.at[..., i].set(0.0))  # own history is not a dependency
        return hist[:, -1] + delta, jnp.stack(deps, 1)

    def to_text(self):
        header = f"# locked away (never read, not modeled): {', '.join(self.excluded)}\n\n" if self.excluded else ""
        return header + "\n\n".join(f"{o}.{d} =\n    {to_str(self.programs[(o, d)], 1)}" for o, d in TARGETS)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"excluded": list(self.excluded), "programs": [[o, d, self.programs[(o, d)]] for o, d in TARGETS]}, fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            data = json.load(fh)
        if isinstance(data, list):  # older format: just the programs
            data = {"excluded": [], "programs": data}
        return cls({(o, d): _tuplify(node) for o, d, node in data["programs"]}, data["excluded"])
