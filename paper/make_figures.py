"""Make all paper figures (vector PDFs in paper/figures/) from runs/* and paper/data/*.

Usage (from the repository root):  uv run paper/make_figures.py
Also writes paper/data/results_table.json with the numbers quoted in the paper.
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
DATA = os.path.join(ROOT, "paper", "data")
RUNS = os.path.join(ROOT, "runs")

# --- palette (validated with the dataviz validator; see paper/README.md) --------------------
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
ENV = {"pong": "#2a78d6", "lazy_enemy": "#eb6834"}  # environment colours (Pong / lazy_enemy)
OBJ = {"player": "#1baf7a", "enemy": "#e87ba4", "ball": "#4a3aa7", "action": "#eda100"}
OBJ_LS = {"player": "-", "enemy": (0, (4, 2)), "ball": "-", "action": (0, (1, 1.5))}
GROUP = {"baselines": "#8a8984", "ours60": "#eb6834", "ours90": "#2a78d6"}

METHODS = [  # key, label, run glob, group
    ("wm_mlp", "WM + MLP actor\n(Exp. 1)", ["sow_v1", "sow_s*"], "baselines"),
    ("dyn_int", "Dynamics\ninterventions (Exp. 2)", ["ia_v1"], "baselines"),
    ("gated", "Gated actor\n(Exp. 3)", ["ga_v3"], "baselines"),
    ("obs60", "Obs. interventions\n60% (ours)", ["oi_s0", "oi_s1"], "ours60"),
    ("obs90", "Obs. interventions\n90% (ours)", ["oi_p01_s*"], "ours90"),
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.4, "legend.frameon": False, "pdf.fonttype": 42,
})


def runs_of(patterns):
    out = []
    for p in patterns:
        out += sorted(glob.glob(os.path.join(RUNS, p)))
    return [r for r in out if os.path.exists(os.path.join(r, "robust", "lazy_enemy_eval.json"))]


def load_eval(run):
    with open(os.path.join(run, "robust", "lazy_enemy_eval.json")) as f:
        return json.load(f)


def final_scores(ev, env):
    return np.array(ev[env]["player_per_game"]) - np.array(ev[env]["enemy_per_game"])


def collect_results():
    res = {}
    for key, label, pats, group in METHODS:
        rows = []
        for run in runs_of(pats):
            ev = load_eval(run)
            with open(os.path.join(run, "robust", "best_selection.json")) as f:
                sel = json.load(f)["selected"]
            rows.append(dict(run=os.path.basename(run), pong=float(final_scores(ev, "pong").mean()),
                             lazy=float(final_scores(ev, "lazy_enemy").mean()),
                             lazy_games=final_scores(ev, "lazy_enemy").tolist(),
                             pong_games=final_scores(ev, "pong").tolist(),
                             frozen_view=sel["frozen_view_final"], selected=sel["ckpt"], greedy=sel["greedy"],
                             lazy_won=int((final_scores(ev, "lazy_enemy") > 0).sum()),
                             wm_h1_pong=ev["pong"]["wm_rmse_px"]["h1"], wm_h1_lazy=ev["lazy_enemy"]["wm_rmse_px"]["h1"]))
        res[key] = dict(label=label, group=group, runs=rows)
    with open(os.path.join(DATA, "results_table.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def fig_main(res):
    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    keys = [k for k, *_ in METHODS if res[k]["runs"]]
    x = np.arange(len(keys))
    w = 0.36
    for j, env in enumerate(("pong", "lazy_enemy")):
        field = "pong" if env == "pong" else "lazy"
        means = [np.mean([r[field] for r in res[k]["runs"]]) for k in keys]
        xs = x + (j - 0.5) * w
        ax.bar(xs, means, width=w - 0.04, color=ENV[env], alpha=0.85, zorder=2,
               label="Pong (training env.)" if env == "pong" else "lazy_enemy (held out)")
        for xi, k in zip(xs, keys):
            vals = [r[field] for r in res[k]["runs"]]
            jit = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter(xi + np.array(jit), vals, s=9, color="white", edgecolor=INK, linewidth=0.6, zorder=3)
        for xi, m, k in zip(xs, means, keys):
            vals = [r[field] for r in res[k]["runs"]] + [m]
            y = max(vals) + 1.2 if m >= 0 else min(vals) - 1.2
            ax.text(xi, y, f"{m:+.1f}", ha="center", va="bottom" if m >= 0 else "top", fontsize=6.3, color=INK)
    ax.axhline(10, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.text(len(keys) - 0.35, 10, "goal\n(+10)", ha="left", va="center", fontsize=6.3, color=INK2, clip_on=False)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(x, [res[k]["label"] for k in keys])
    ax.set_ylabel("final score (agent $-$ enemy)")
    ax.set_ylim(-16, 25)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "main_results.pdf"))
    plt.close(fig)


def fig_per_game(res):
    fig, ax = plt.subplots(figsize=(5.5, 1.9))
    keys = [k for k, *_ in METHODS if res[k]["runs"]]
    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        vals = np.concatenate([r["lazy_games"] for r in res[k]["runs"]])
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(vals)), vals, s=5, color=ENV["lazy_enemy"], alpha=0.55,
                   linewidth=0, zorder=2)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([i - 0.33, i + 0.33], [med, med], color=INK, lw=1.4, zorder=3)
        ax.plot([i, i], [q1, q3], color=INK, lw=0.8, zorder=3)
        won = (vals > 0).mean() * 100
        ax.text(i, 23.5, f"{won:.0f}% won", ha="center", fontsize=6.5, color=INK2)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axhline(10, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(range(len(keys)), [res[k]["label"] for k in keys])
    ax.set_ylabel("final score per game\n(lazy_enemy)")
    ax.set_ylim(-22, 26)
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "per_game.pdf"))
    plt.close(fig)


def fig_training(res):
    fig, ax = plt.subplots(figsize=(2.7, 2.1))
    cols = {"wm_mlp": "#8a8984", "dyn_int": "#1baf7a", "gated": "#4a3aa7", "obs60": "#eb6834", "obs90": "#2a78d6"}
    labels = {"wm_mlp": "Exp. 1", "dyn_int": "Exp. 2", "gated": "Exp. 3", "obs60": "ours 60%", "obs90": "ours 90%"}
    for k, *_ in METHODS:
        curves = []
        for r in res[k]["runs"]:
            with open(os.path.join(RUNS, r["run"], "results.json")) as f:
                rounds = json.load(f)["rounds"]
            curves.append([x["pong_greedy"]["player"] - x["pong_greedy"]["enemy"] for x in rounds])
        if not curves:
            continue
        c = np.array(curves)
        rr = np.arange(c.shape[1])
        for cc in c:
            ax.plot(rr, cc, color=cols[k], lw=0.5, alpha=0.35)
        ax.plot(rr, c.mean(0), color=cols[k], lw=1.5, marker="o", ms=2.5, label=labels[k])
    ax.set_xlabel("round (real data collection + WM fit + PPO in imagination)")
    ax.set_ylabel("final score on real Pong")
    ax.set_xticks(range(5))
    ax.legend(loc="lower right", fontsize=6.2, handlelength=1.5)
    fig.savefig(os.path.join(FIG, "training.pdf"))
    plt.close(fig)


def fig_frozen_proxy(res):
    fig, ax = plt.subplots(figsize=(2.7, 2.1))
    marks = {"baselines": "s", "ours60": "^", "ours90": "o"}
    names = {"baselines": "baselines (Exp. 1-3)", "ours60": "ours, 60%", "ours90": "ours, 90%"}
    seen = set()
    for k, *_ in METHODS:
        g = res[k]["group"]
        for r in res[k]["runs"]:
            ax.scatter(r["frozen_view"], r["lazy"], s=22, marker=marks[g], color=GROUP[g], edgecolor="white",
                       linewidth=0.6, zorder=3, label=None if g in seen else names[g])
            seen.add(g)
    ax.axhline(10, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("final score, Pong with frozen enemy view\n(selection proxy, no lazy_enemy)")
    ax.set_ylabel("final score, lazy_enemy")
    ax.legend(loc="upper left", fontsize=6.2, handletextpad=0.3)
    fig.savefig(os.path.join(FIG, "frozen_proxy.pdf"))
    plt.close(fig)


def _arrivals(pos):
    """Steps at which the ball reaches the player's side (x turns around at x >= 128 or a point)."""
    bx = pos[:, 2, 0]
    return [t for t in range(1, len(bx) - 1) if bx[t] >= 128 and bx[t + 1] < bx[t] and bx[t - 1] < bx[t]]


def fig_traces():
    path = os.path.join(DATA, "traces.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    # the baseline's first lost point, and our agent's first return with the enemy >= 40 px away
    base_pos, base_rew = d["baseline_lazy_enemy_pos"], d["baseline_lazy_enemy_reward"]
    ours_pos = d["ours_lazy_enemy_pos"]
    t_miss = int(np.nonzero(base_rew < 0)[0][0])
    t_hit = next(t for t in _arrivals(ours_pos) if abs(ours_pos[t, 2, 1] - ours_pos[t, 1, 1]) >= 40)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), sharey=True, constrained_layout=True)
    panels = ((axes[0], base_pos, t_miss, "WM + MLP actor (Exp. 1): misses"),
              (axes[1], ours_pos, t_hit, "Observation interventions (ours): returns"))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    for ax, pos, t0, title in panels:
        a, b = t0 - 45, t0 + 6
        t = np.arange(a, b)
        seg = pos[a:b]
        ax.fill_between(t, seg[:, 1, 1], seg[:, 1, 1] + 16, color=OBJ["enemy"], alpha=0.35, lw=0)
        ax.fill_between(t, seg[:, 0, 1], seg[:, 0, 1] + 16, color=OBJ["player"], alpha=0.45, lw=0)
        ax.plot(t, seg[:, 2, 1] + 2, color=OBJ["ball"], lw=1.4)
        ax.axvline(t0, color=MUTED, lw=0.7, ls=(0, (2, 2)))
        yb = seg[t0 - a, 2, 1] + 2
        ax.plot([t0], [yb], marker="x" if "misses" in title else "o", ms=5, color=INK, mew=1.2)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("agent step (lazy_enemy)")
    handles = [Patch(color=OBJ["player"], alpha=0.45, label="player paddle"),
               Patch(color=OBJ["enemy"], alpha=0.35, label="enemy paddle (lazy, stale)"),
               Line2D([], [], color=OBJ["ball"], lw=1.4, label="ball"),
               Line2D([], [], color=MUTED, lw=0.7, ls=(0, (2, 2)), label="ball reaches the player"),
               Line2D([], [], color=INK, marker="x", lw=0, label="miss"),
               Line2D([], [], color=INK, marker="o", lw=0, ms=4, label="return")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=6,
               handlelength=1.4, columnspacing=0.9)
    axes[0].set_ylabel("height (px), top of screen up")
    axes[0].set_ylim(212, 20)
    fig.savefig(os.path.join(FIG, "traces_lazy.pdf"))
    plt.close(fig)


def fig_gates():
    path = os.path.join(DATA, "traces.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    T = 220
    pos = d["ours_pong_pos"][:T]
    g = d["ours_pong_gates"][:T]  # (T, receiver, token)
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 3.3), sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1, 1]))
    ax = axes[0]
    ax.plot(pos[:, 2, 0], color=OBJ["ball"], lw=1.3)
    ax.axhspan(16, 20, color=OBJ["enemy"], alpha=0.25, lw=0)
    ax.axhspan(140, 144, color=OBJ["player"], alpha=0.25, lw=0)
    ax.text(100, 23, "enemy paddle", ha="left", va="bottom", fontsize=6, color=INK2)
    ax.text(62, 147, "player paddle", ha="left", va="bottom", fontsize=6, color=INK2)
    ax.set_ylabel("ball x (px)")
    ax.set_ylim(0, 165)
    ax.set_xlim(0, T)
    tok = ("player", "enemy", "ball", "action")
    for ax, (recv, ri) in zip(axes[1:], (("ball", 2), ("enemy", 1))):
        for tj, t in enumerate(tok):
            if tj == ri:
                continue
            ax.plot(g[:, ri, tj], color=OBJ[t], ls=OBJ_LS[t], lw=1.1, label=f"{recv} $\\leftarrow$ {t}")
        ax.set_ylim(-0.05, 1.45)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(f"gates of\n{recv}")
        ax.legend(loc="upper left", ncol=3, fontsize=6.2, bbox_to_anchor=(0.0, 1.02), borderaxespad=0.1)
    axes[-1].set_xlabel("agent step (unmodified Pong)")
    fig.align_ylabels(axes)
    fig.savefig(os.path.join(FIG, "gates.pdf"))
    plt.close(fig)


def fig_wm_error():
    path = os.path.join(DATA, "rollouts.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75), sharey=True)
    h = np.arange(1, d["pong_rmse"].shape[0] + 1)
    for i, (ax, obj) in enumerate(zip(axes, ("player", "enemy", "ball"))):
        for env in ("pong", "lazy_enemy"):
            ax.plot(h, d[f"{env}_rmse"][:, i], color=ENV[env], label="Pong" if env == "pong" else "lazy_enemy")
        ax.set_title(obj, loc="left", color=INK)
        ax.set_xlabel("open-loop horizon (steps)")
        ax.set_xlim(0, h[-1] + 1)
    axes[0].legend(loc="upper left", fontsize=6.3)
    axes[0].set_ylabel("RMSE (px)")
    fig.savefig(os.path.join(FIG, "wm_error.pdf"))
    plt.close(fig)


def parse_ablation():
    """Ball 1-step error per region from runs/ablation_ball_enemy.log (with / without ball<-enemy).

    Regions: ball moving towards the enemy at x < 30 (bounce zone); 30 <= x < 60 and 60 <= x < 120
    (count-weighted over both directions); x >= 120 (count-weighted over both directions).
    """
    import re
    blocks, cur = [], None
    with open(os.path.join(RUNS, "ablation_ball_enemy.log")) as f:
        for line in f:
            if line.startswith("mask_ball_enemy"):
                cur = {}
                blocks.append(cur)
            m = re.search(r"x\[(\d+),(\d+)\) dir([+-]1) n= *(\d+) err=([\d.]+)", line)
            if m and cur is not None:
                cur[(int(m.group(1)), int(m.group(3)))] = (int(m.group(4)), float(m.group(5)))

    def agg(b, lo, dirs):
        n = sum(b[(lo, d)][0] for d in dirs)
        return sum(b[(lo, d)][0] * b[(lo, d)][1] for d in dirs) / n

    out = []
    for b in blocks:
        out.append([agg(b, 0, [-1]), agg(b, 30, [-1, 1]), agg(b, 60, [-1, 1]), agg(b, 120, [-1, 1])])
    return out[0], out[1]


def fig_leakage():
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.35), gridspec_kw=dict(width_ratios=[1.0, 1.1, 1.4]),
                             constrained_layout=True)
    # (a) ablation: ball error by region with / without the ball<-enemy message
    regions = ["<30", "30-\n60", "60-\n120", "$\\geq$\n120"]
    with_e, without = parse_ablation()
    ax = axes[0]
    x = np.arange(4)
    ax.bar(x - 0.19, with_e, 0.34, color=INK2, label="with enemy messages")
    ax.bar(x + 0.19, without, 0.34, color="#1baf7a", label="without")
    ax.set_xticks(x, regions, fontsize=5.8)
    ax.set_xlabel("ball x (px)")
    ax.set_ylabel("ball 1-step error (px)")
    ax.set_title("(a) ablation", loc="left", color=INK, pad=20)
    ax.legend(fontsize=5.8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, borderaxespad=0.2,
              handlelength=1.0)
    ax.grid(axis="x", visible=False)
    # (b) leakage vs enemy-ball gap after swapping the enemy's history
    with open(os.path.join(DATA, "leakage_by_gap.json")) as f:
        lk = json.load(f)
    ax = axes[1]
    e = lk["edges"]
    mids = [(a + b) / 2 for a, b in zip(e[:-1], e[1:])]
    real = np.array(lk["real_gap_hist"]) / max(1, sum(lk["real_gap_hist"]))
    support = e[int(np.nonzero(real > 0)[0].max()) + 1]  # upper edge of the last bin with real data
    ax.axvspan(0, support, color=GRID, alpha=0.8, lw=0, zorder=0)
    ax.text(support + 4, 2.0, "$\\leftarrow$ gaps seen\n    in real data", ha="left", va="bottom", fontsize=5.6,
            color=INK2)
    ax.plot(mids, lk["leak_rmse"], color=OBJ["ball"], marker="o", ms=3, zorder=3)
    ax.set_xlim(0, e[-1])
    ax.set_xlabel("|enemy y $-$ ball y| after swap (px)")
    ax.set_ylabel("ball prediction change (px)")
    ax.set_ylim(0, 24)
    ax.set_title("(b) leakage vs. gap", loc="left", color=INK, pad=20)
    # (c) leakage by world-model variant (far from the enemy, ball moving)
    variants = [("Exp. 1: pair sigmoid", None), ("pair $L_0$ (0.05)", "wmonly_g0.05"),
                ("pair $L_0$ (0.2)", "wmonly_g0.2"), ("receiver $L_0$", "rg_wmonly"),
                ("receiver soft (0.1)", "rsg_wmonly_g0.1"), ("receiver binary (0.1)", "rste_wmonly_g0.1")]
    names, far, near = [], [], []
    for name, run in variants:
        if run is None:
            lkg = lk["exp1_leakage"]
        else:
            with open(os.path.join(RUNS, run, "wm_only.json")) as fh:
                lkg = json.load(fh)["wm"]["leakage"]
        names.append(name)
        far.append(lkg["ball<-enemy_far"])
        near.append(lkg["ball<-enemy_near"])
    ax = axes[2]
    y = np.arange(len(names))
    ax.barh(y - 0.19, far, 0.34, color=OBJ["ball"], label="far from enemy")
    ax.barh(y + 0.19, near, 0.34, color=OBJ["enemy"], label="near enemy")
    ax.set_yticks(y, names, fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlabel("ball $\\leftarrow$ enemy leakage (px)")
    ax.set_title("(c) gating variants", loc="left", color=INK, pad=20)
    ax.legend(fontsize=5.8, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, borderaxespad=0.2,
              handlelength=1.0, columnspacing=0.8)
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(FIG, "leakage.pdf"))
    plt.close(fig)


def fig_imagination():
    path = os.path.join(DATA, "rollouts.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.9), sharey=True)
    for ax, env in zip(axes, ("pong", "lazy_enemy")):
        real = d[f"{env}_example_real"]
        pred = d[f"{env}_example_pred"]
        K = real.shape[0] - pred.shape[0]
        for obj, i in (("ball", 2), ("enemy", 1), ("player", 0)):
            ax.plot(np.arange(real.shape[0]) - K + 1, real[:, i, 1], color=OBJ[obj], ls=OBJ_LS[obj], lw=1.0,
                    alpha=0.9, label=f"{obj} (real)")
            ax.plot(np.arange(1, pred.shape[0] + 1), pred[:, i, 1], color=OBJ[obj], lw=0, marker="o", ms=1.6,
                    label=f"{obj} (imagined)")
        ax.axvline(0.5, color=MUTED, lw=0.6)
        ax.set_title("Pong" if env == "pong" else "lazy_enemy", loc="left", color=INK)
        ax.set_xlabel("step (context before 0, open-loop after)")
        ax.set_ylim(215, 15)
    axes[0].set_ylabel("height (px)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=5.8, handlelength=1.2,
               columnspacing=0.8)
    fig.savefig(os.path.join(FIG, "imagination.pdf"))
    plt.close(fig)


def write_numbers(res):
    """LaTeX macros and the main results table, generated from the evaluation files."""
    macros = []

    def stats(key, field):
        v = np.array([r[field] for r in res[key]["runs"]], dtype=float)
        return v.mean(), (v.std(ddof=0) if len(v) > 1 else 0.0), len(v)

    names = {"wm_mlp": "Base", "dyn_int": "Dyn", "gated": "Gated", "obs60": "OursSixty", "obs90": "Ours"}
    for key, tag in names.items():
        if not res[key]["runs"]:  # placeholders so that drafts compile before all runs are done
            for suffix in ("Pong", "PongStd", "Lazy", "LazyStd", "Frozen", "FrozenStd", "Seeds", "Won",
                           "LazyMin", "LazyMax", "WorstGame"):
                macros.append(f"\\newcommand{{\\{tag}{suffix}}}{{\\textbf{{??}}}}")
            continue
        for field, ftag in (("pong", "Pong"), ("lazy", "Lazy"), ("frozen_view", "Frozen")):
            m, sd, n = stats(key, field)
            macros.append(f"\\newcommand{{\\{tag}{ftag}}}{{{m:+.1f}}}")
            macros.append(f"\\newcommand{{\\{tag}{ftag}Std}}{{{sd:.1f}}}")
        macros.append(f"\\newcommand{{\\{tag}Seeds}}{{{n}}}")
        won = sum(r["lazy_won"] for r in res[key]["runs"])
        games = sum(len(r["lazy_games"]) for r in res[key]["runs"])
        macros.append(f"\\newcommand{{\\{tag}Won}}{{{100 * won / games:.0f}}}")
        lo = min(r["lazy"] for r in res[key]["runs"])
        hi = max(r["lazy"] for r in res[key]["runs"])
        macros.append(f"\\newcommand{{\\{tag}LazyMin}}{{{lo:+.1f}}}")
        macros.append(f"\\newcommand{{\\{tag}LazyMax}}{{{hi:+.1f}}}")
        worst = min(min(r["lazy_games"]) for r in res[key]["runs"])
        macros.append(f"\\newcommand{{\\{tag}WorstGame}}{{{worst:+.0f}}}")
    with open(os.path.join(ROOT, "paper", "numbers.tex"), "w") as f:
        f.write("% generated by make_figures.py -- do not edit\n" + "\n".join(macros) + "\n")

    rows = []
    for key, label, *_ in METHODS:
        if not res[key]["runs"]:
            continue
        cells = []
        for field in ("pong", "frozen_view", "lazy"):
            m, sd, n = stats(key, field)
            cells.append(f"${m:+.1f}$" + (f" {{\\scriptsize$\\pm${sd:.1f}}}" if n > 1 else ""))
        won = sum(r["lazy_won"] for r in res[key]["runs"])
        games = sum(len(r["lazy_games"]) for r in res[key]["runs"])
        name = label.replace("\n", " ").replace("%", "\\%")
        if key == "obs90":
            name = "\\textbf{" + name + "}"
            cells[-1] = "\\textbf{" + cells[-1].split(" ")[0] + "} " + " ".join(cells[-1].split(" ")[1:])
        rows.append(f"{name} & {len(res[key]['runs'])} & " + " & ".join(cells) + f" & {100 * won / games:.0f}\\% \\\\")
    with open(os.path.join(ROOT, "paper", "numbers.tex"), "a") as f:
        f.write("\\newcommand{\\MainTableRows}{%\n" + "\n".join(rows) + "\n}\n")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    res = collect_results()
    for k, v in res.items():
        print(k, [(r["run"], round(r["pong"], 2), round(r["lazy"], 2)) for r in v["runs"]])
    write_numbers(res)
    fig_main(res)
    fig_per_game(res)
    fig_training(res)
    fig_frozen_proxy(res)
    fig_traces()
    fig_gates()
    fig_wm_error()
    fig_leakage()
    fig_imagination()
    print("figures written to", FIG)
