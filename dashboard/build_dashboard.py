"""Build the experiment dashboard (`dashboard/index.html` + `dashboard/data.json`) from the repo.

Nothing has to be registered by hand. The builder finds:
- **experiments**: every `experiment_*.py` in the repo root with its same-named Markdown file.
  From the Markdown it reads the H1 title, `## Summary` (optional, 1-3 sentences on the approach;
  falls back to the script's docstring), `## Goal`, `## Research question` and `## Results`.
- **runs** (`runs/<name>/`) and which experiment they belong to, from, in this order:
  1. `"experiment"` in the run's config (`results.json` / `wm_only.json`), if a script records it,
  2. `uv run experiment_x.py ... --name <run>` commands and `--module experiment_x --ckpt runs/<run>/...`
     commands in the experiment's Markdown file,
  3. running processes (`ps`), so that runs show up while they are still training.
  Runs that match no experiment are listed separately as "unassigned".
- **statistics** of every run: per-round world-model validation loss and real (unmodified) Pong
  scores (`results.json`, or `log.txt` while a run is in progress), world-model-only results
  (`wm_only.json`), and held-out evaluations (`lazy_enemy_eval.json`, written by
  `evaluate_lazy_enemy.py`, also when the checkpoint was saved in `runs/` itself).
- **videos**: for every experiment, the newest checkpoint of its most recently updated run
  (latest `round*.pkl`, else `best.pkl`, else `wm_only.pkl`) is rolled out in imagination by
  `dashboard/render_rollout.py` (CPU only) and saved to `dashboard/media/<experiment>.mp4`. Videos
  are only re-rendered when that checkpoint changes (`dashboard/media/manifest.json`).

The dashboard only *displays* `lazy_enemy` results; it never trains, tunes or selects anything.

Outputs: `dashboard/index.html` (open it directly or serve with `python -m http.server -d dashboard`),
`dashboard/data.json` (all collected numbers, machine-readable), `dashboard/media/*.mp4`, and
`dashboard/artifact.html` (the same page without the document skeleton, for republishing the hosted
copy; see `dashboard/README.md`).

Usage (from the repo root):
    CUDA_VISIBLE_DEVICES= uv run dashboard/build_dashboard.py              # update everything
    CUDA_VISIBLE_DEVICES= uv run dashboard/build_dashboard.py --no-video   # statistics only
    CUDA_VISIBLE_DEVICES= uv run dashboard/build_dashboard.py --force-video
"""

import argparse
import ast
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time

DASH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DASH)
RUNS = os.path.join(ROOT, "runs")
MEDIA = os.path.join(DASH, "media")
TARGET = 10.0  # lazy_enemy criterion: final score (player - enemy) >= +10
RUNNING_WINDOW_S = 15 * 60  # a run whose log changed this recently counts as running


# ----------------------------------------------------------------------------- parsing helpers


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def md_sections(text):
    """{'_title': H1, '<lowercase h2>': body} for a Markdown document."""
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith("# ") and "_title" not in out:
            out["_title"] = line[2:].strip()
            continue
        if line.startswith("## "):
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = line[3:].strip().lower(), []
            continue
        if cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def find_section(sections, *names):
    for n in names:
        for k, v in sections.items():
            if k.startswith(n):
                return v
    return ""


def docstring(path):
    try:
        return ast.get_docstring(ast.parse(read(path))) or ""
    except SyntaxError:
        return ""


def doc_summary(doc):
    """Fallback approach summary from a docstring: the paragraph that describes this experiment."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc) if p.strip()]
    for p in paras[1:]:
        if re.match(r"(This experiment|Approach|Idea|Here we)\b", p):
            return p
    return paras[1] if len(paras) >= 2 else (paras[0] if paras else "")


def exp_number(*texts):
    for t in texts:
        m = re.search(r"Experiment\s+(\d+)", t or "")
        if m:
            return int(m.group(1))
    return None


def run_names_from_md(text):
    """{run_name: module} from the commands in an experiment's Markdown file."""
    found = {}
    for m in re.finditer(r"uv run (?:python )?([\w/]+)\.py([^\n`]*)", text):
        script, args = os.path.basename(m.group(1)), m.group(2)
        n = re.search(r"--name[ =](\S+)", args)
        mod = re.search(r"--module[ =](\S+)", args)
        ck = re.search(r"--ckpt[ =]runs/([^/\s]+)/", args)
        if script.startswith("experiment_") and n:
            found.setdefault(n.group(1), script)
        if mod and ck:
            found.setdefault(ck.group(1), mod.group(1))
    return found


def processes():
    """[(elapsed_s, args)] of running experiment / evaluation processes."""
    try:
        out = subprocess.run(["ps", "-eo", "etimes=,args="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    procs = []
    for line in out.splitlines():
        line = line.strip()
        if ".py" not in line or "uv run" in line:  # skip the `uv run` wrappers, keep the python child
            continue
        el, _, args = line.partition(" ")
        if re.search(r"(experiment_\w+|evaluate_\w+|select_\w+)\.py", args):
            procs.append((int(el), args))
    return procs


# ----------------------------------------------------------------------------- runs


def final(p, e):
    return None if p is None or e is None else float(p) - float(e)


def parse_log(path):
    """Config, per-round progress and current activity from a run's log.txt."""
    info = dict(config=None, rounds=[], last_line="", wm_only=False)
    try:
        lines = read(path).splitlines()
    except OSError:
        return info
    rnd = None
    for line in lines:
        if line.startswith("config: ") and info["config"] is None:
            try:
                info["config"] = ast.literal_eval(line[len("config: "):])
            except (ValueError, SyntaxError):
                pass
        m = re.match(r"=== round (\d+) ===", line)
        if m:
            rnd = dict(round=int(m.group(1)))
            info["rounds"].append(rnd)
        m = re.search(r"wm it\s+\d+ train \S+ val (\S+)", line)
        if m and rnd is not None:
            rnd["val_loss"] = float(m.group(1))
        m = re.search(r"real Pong eval sample: (\S+) : (\S+) \| greedy: (\S+) : (\S+)", line)
        if m and rnd is not None:
            a, b, c, d = map(float, m.groups())
            rnd["pong_sample"], rnd["pong_greedy"] = final(a, b), final(c, d)
            rnd["done"] = True
    info["last_line"] = lines[-1].strip() if lines else ""
    return info


def lazy_evals():
    """{run_dir_name: [eval dict]} from every lazy_enemy_eval*.json below runs/."""
    out = {}
    for path in glob.glob(os.path.join(RUNS, "**", "*lazy_enemy_eval*.json"), recursive=True):
        ev = load_json(path)
        if not ev or "lazy_enemy" not in ev:
            continue
        ckpt = ev.get("ckpt", "")
        parts = os.path.normpath(ckpt).split(os.sep)
        run = parts[1] if len(parts) >= 3 and parts[0] == "runs" else None
        if run is None:  # checkpoint stored in runs/ itself (e.g. by select_robust_checkpoint.py)
            sel = load_json(os.path.join(ROOT, os.path.splitext(ckpt)[0] + "_selection.json")) or {}
            src = os.path.normpath(sel.get("selected", {}).get("ckpt", "")).split(os.sep)
            run = src[1] if len(src) >= 3 else None
        if run is None:
            continue
        lz, pg = ev["lazy_enemy"], ev.get("pong", {})
        won = sum(p > e for p, e in zip(lz.get("player_per_game", []), lz.get("enemy_per_game", [])))
        rm_l, rm_p = lz.get("wm_rmse_px", {}), pg.get("wm_rmse_px", {})
        out.setdefault(run, []).append(dict(
            file=os.path.relpath(path, ROOT), ckpt=ckpt, round=ev.get("round"), greedy=ev.get("greedy"),
            games=ev.get("games"), seed=ev.get("seed"),
            pong_player=pg.get("player_mean"), pong_enemy=pg.get("enemy_mean"),
            pong_final=final(pg.get("player_mean"), pg.get("enemy_mean")),
            lazy_player=lz.get("player_mean"), lazy_enemy=lz.get("enemy_mean"),
            lazy_final=final(lz.get("player_mean"), lz.get("enemy_mean")), lazy_won=won,
            rmse_ball_h1=[rm_p.get("h1", {}).get("ball"), rm_l.get("h1", {}).get("ball")],
            rmse_enemy_h1=[rm_p.get("h1", {}).get("enemy"), rm_l.get("h1", {}).get("enemy")],
            mtime=os.path.getmtime(path)))
    return out


def run_info(name, evals, procs):
    d = os.path.join(RUNS, name)
    files = [os.path.join(d, f) for f in os.listdir(d)]
    mtime = max([os.path.getmtime(f) for f in files] + [os.path.getmtime(d)])
    res, wmo, log = load_json(os.path.join(d, "results.json")), load_json(os.path.join(d, "wm_only.json")), \
        parse_log(os.path.join(d, "log.txt"))
    cfg = (res or {}).get("config") or (wmo or {}).get("config") or log["config"] or {}
    info = dict(name=name, mtime=mtime, config=cfg, experiment=cfg.get("experiment") or cfg.get("module"))
    rounds = []
    if res:
        for r in res.get("rounds", []):
            ps, pg = r.get("pong_sample", {}), r.get("pong_greedy", {})
            rounds.append(dict(round=r["round"], val_loss=r.get("wm", {}).get("val_loss"),
                               pong_sample=final(ps.get("player"), ps.get("enemy")),
                               pong_greedy=final(pg.get("player"), pg.get("enemy")),
                               transitions=r.get("transitions")))
        info["selected"] = res.get("selected")
    else:
        rounds = [r for r in log["rounds"] if r.get("done")]
    info["rounds"] = rounds
    info["rounds_total"] = cfg.get("rounds")
    if wmo:
        wm = wmo.get("wm", {})
        info["kind"] = "wm_only"
        info["wm_only"] = dict(val_loss=wm.get("val_loss"), mse_ball=wm.get("mse_ball"),
                               open_gates=wm.get("open_gates"), leakage=wm.get("leakage", {}))
    elif res or rounds:
        info["kind"] = "full"
    elif cfg.get("wm_only"):
        info["kind"] = "wm_only"
    else:
        info["kind"] = "started"
    info["evals"] = sorted(evals.get(name, []), key=lambda e: e["mtime"])
    pat = re.compile(rf"--name[ =]{re.escape(name)}(\s|$)|runs/{re.escape(name)}/")
    live = [(el, a) for el, a in procs if pat.search(a)]
    info["process"] = [dict(elapsed_s=el, cmd=a[re.search(r"[\w/]+\.py", a).start():]) for el, a in live]
    info["running"] = bool(live)
    done_rounds = len(rounds)
    if info["running"] and any(re.match(r"(evaluate|select)_", p["cmd"]) for p in info["process"]):
        info["activity"] = "held-out evaluation" if any(p["cmd"].startswith("evaluate_") for p in info["process"]) \
            else "checkpoint selection (unmodified Pong)"
    elif info["running"] and info["kind"] != "wm_only":
        cur = log["last_line"]
        m = re.search(r"ppo\s+(\d+)", cur)
        stage = f"PPO update {m.group(1)}/{cfg.get('ppo_updates', '?')}" if m else (
            "world model" if "wm it" in cur else "collecting / evaluating")
        info["activity"] = f"round {min(done_rounds, (info['rounds_total'] or 99) - 1)}: {stage}"
    info["checkpoint"] = latest_checkpoint(d)
    return info


def latest_checkpoint(d):
    rounds = glob.glob(os.path.join(d, "round*.pkl"))
    if rounds:
        path = max(rounds, key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
        return dict(path=os.path.relpath(path, ROOT), label=f"round {re.findall(r'[0-9]+', os.path.basename(path))[0]} (latest)")
    for f, label in (("best.pkl", "selected checkpoint"), ("wm_only.pkl", "world model only")):
        if os.path.exists(os.path.join(d, f)):
            return dict(path=os.path.relpath(os.path.join(d, f), ROOT), label=label)
    return None


# ----------------------------------------------------------------------------- experiments


def collect():
    procs = processes()
    evals = lazy_evals()
    run_names = sorted(n for n in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, n))) if os.path.isdir(RUNS) else []
    runs = {n: run_info(n, evals, procs) for n in run_names}

    experiments = {}
    for py in sorted(glob.glob(os.path.join(ROOT, "experiment_*.py"))):
        module = os.path.splitext(os.path.basename(py))[0]
        md_path = os.path.splitext(py)[0] + ".md"
        md = read(md_path) if os.path.exists(md_path) else ""
        sec = md_sections(md)
        doc = docstring(py)
        experiments[module] = dict(
            module=module, py=os.path.basename(py), md=os.path.basename(md_path) if md else None,
            title=sec.get("_title") or (doc.splitlines()[0] if doc else module),
            number=exp_number(sec.get("_title"), doc),
            summary=find_section(sec, "summary") or doc_summary(doc),
            summary_source="md" if find_section(sec, "summary") else "docstring",
            goal=find_section(sec, "goal"), question=find_section(sec, "research question"),
            results=find_section(sec, "results"), setup=find_section(sec, "setup", "approach"),
            md_runs=run_names_from_md(md), flags=set(re.findall(r"add_argument\(\s*[\"']--(\w+)", read(py))),
            mtime=os.path.getmtime(py), runs=[])

    def owner(run):
        if run["experiment"] in experiments:
            return run["experiment"]
        for m, e in experiments.items():
            if e["md_runs"].get(run["name"]) == m:
                return m
        for m, e in experiments.items():  # evaluation commands in another experiment's md
            if run["name"] in e["md_runs"] and e["md_runs"][run["name"]] in experiments:
                return e["md_runs"][run["name"]]
        for p in run["process"]:
            mm = re.search(r"(experiment_\w+)\.py|--module[ =](\w+)", p["cmd"])
            if mm and (mm.group(1) or mm.group(2)) in experiments:
                return mm.group(1) or mm.group(2)
        # last resort: the only experiment whose command-line flags are exactly the run's config keys
        keys = set(run["config"])
        match = [m for m, e in experiments.items() if keys and e["flags"] == keys]
        return match[0] if len(match) == 1 else None

    unassigned = []
    for r in runs.values():
        m = owner(r)
        (experiments[m]["runs"] if m else unassigned).append(r)

    out = []
    for e in experiments.values():
        e["runs"].sort(key=lambda r: r["mtime"])
        if not e["md"] and not e["runs"]:
            continue  # e.g. a scratch script that is not an experiment (yet)
        e["status"] = status(e)
        e["updated"] = max([r["mtime"] for r in e["runs"]] + [e["mtime"]])
        del e["md_runs"], e["flags"]
        out.append(e)
    out.sort(key=lambda e: (e["number"] is None, e["number"] or 0, e["mtime"]))
    return out, unassigned


def status(e):
    runs = e["runs"]
    evals = [ev for r in runs for ev in r["evals"]]
    if any(ev["lazy_final"] is not None and ev["lazy_final"] >= TARGET and (ev["games"] or 0) >= 10 for ev in evals):
        return "goal_met"
    if any(r["running"] for r in runs):
        return "running"
    if evals:
        return "evaluated"
    if runs and all(r["kind"] == "wm_only" for r in runs):
        return "wm_only"
    if any(r["kind"] == "full" for r in runs):
        return "trained"
    return "no_runs"


# ----------------------------------------------------------------------------- videos


def update_videos(experiments, force=False):
    os.makedirs(MEDIA, exist_ok=True)
    man_path = os.path.join(MEDIA, "manifest.json")
    manifest = load_json(man_path) or {}
    for e in experiments:
        cands = [r for r in e["runs"] if r["checkpoint"]]
        if not cands:
            e["video"] = None
            continue
        run = max(cands, key=lambda r: os.path.getmtime(os.path.join(ROOT, r["checkpoint"]["path"])))
        ck = run["checkpoint"]
        ck_path = os.path.join(ROOT, ck["path"])
        ck_mtime = os.path.getmtime(ck_path)
        out = os.path.join(MEDIA, f"{e['module']}.mp4")
        prev = manifest.get(e["module"])
        fresh = prev and prev["ckpt"] == ck["path"] and abs(prev["ckpt_mtime"] - ck_mtime) < 1 and os.path.exists(out)
        if time.time() - ck_mtime < 30:  # still being written
            fresh = fresh or (prev is not None and os.path.exists(out))
        if force or not fresh:
            print(f"rendering {e['module']} <- {ck['path']}", flush=True)
            try:
                import render_rollout

                meta = render_rollout.render(e["module"], ck_path, out, label=f"{run['name']} · {ck['label']}")
            except Exception as exc:  # a broken checkpoint must not break the dashboard
                print(f"  video failed: {exc!r}", flush=True)
                e["video"] = dict(error=repr(exc), run=run["name"], ckpt=ck["path"])
                continue
            manifest[e["module"]] = prev = dict(ckpt=ck["path"], ckpt_mtime=ck_mtime, run=run["name"],
                                                label=ck["label"], rendered=time.time(), **meta)
        e["video"] = dict(src=f"media/{e['module']}.mp4", **prev)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1)


# ----------------------------------------------------------------------------- html


def git_head():
    try:
        return subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%h %s"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def write_outputs(experiments, unassigned, fragment_path=None):
    data = dict(generated=time.time(), generated_iso=dt.datetime.now().isoformat(timespec="seconds"),
                git=git_head(), target=TARGET, experiments=experiments, unassigned=unassigned)
    with open(os.path.join(DASH, "data.json"), "w") as f:
        json.dump(data, f, indent=1)
    template = read(os.path.join(DASH, "template.html"))
    payload = json.dumps(data).replace("</", "<\\/")
    body = template.replace("/*__DATA__*/null", payload)
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + body + "\n</html>\n")
    with open(os.path.join(DASH, "index.html"), "w") as f:
        f.write(page)
    if fragment_path:  # the same page without the document skeleton, for hosts that add their own
        with open(fragment_path, "w") as f:
            f.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true", help="skip rendering, reuse existing videos")
    ap.add_argument("--force-video", action="store_true", help="re-render every video")
    ap.add_argument("--fragment", default=os.path.join(DASH, "artifact.html"),
                    help="also write the page without <html>/<head> here (for publishing as a claude.ai Artifact)")
    args = ap.parse_args()
    sys.path.insert(0, DASH)
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    experiments, unassigned = collect()
    if args.no_video:
        manifest = load_json(os.path.join(MEDIA, "manifest.json")) or {}
        for e in experiments:
            v = manifest.get(e["module"])
            e["video"] = dict(src=f"media/{e['module']}.mp4", **v) if v else None
    else:
        update_videos(experiments, force=args.force_video)
    write_outputs(experiments, unassigned, args.fragment)
    for e in experiments:
        print(f"{e['number'] or '-':>3} {e['module']:<40} {e['status']:<10} runs: {', '.join(r['name'] for r in e['runs'])}")
    if unassigned:
        print("unassigned runs:", ", ".join(r["name"] for r in unassigned))
    print(f"wrote {os.path.relpath(os.path.join(DASH, 'index.html'), ROOT)}")


if __name__ == "__main__":
    main()
