# Experiment dashboard

A grid with one card per experiment: status, research question, approach summary, key numbers
(held-out `lazy_enemy` and unmodified-Pong final scores, world-model losses, leakage), a
per-round training curve, all runs, and three videos of the newest checkpoint: the actor in the
real `lazy_enemy` game, and the actor inside the world model started from real Pong or real
`lazy_enemy` histories, with the per-step gates drawn as dependency lines.

- Local page: `dashboard/index.html` (open in a browser; the videos are in `dashboard/media/`).
- Hosted copy (private claude.ai Artifact): https://claude.ai/artifact/SKykRqLKbAwdPm8apsUHRv

## Updating it

```
CUDA_VISIBLE_DEVICES= uv run dashboard/build_dashboard.py              # rescan the repo, re-render changed videos
CUDA_VISIBLE_DEVICES= uv run dashboard/build_dashboard.py --no-video   # numbers only (fast)
```

It runs on CPU only (about 20 s per re-rendered experiment), so it never uses a GPU. Rebuild whenever a
run finishes or an experiment's Markdown file changes, and commit `dashboard/` together with the
experiment.

To update the hosted copy, an agent with the Artifact tool republishes `dashboard/artifact.html` to
the URL above, with every video as a supporting file
(`files: {"media/<experiment>__<view>.mp4": "dashboard/media/<experiment>__<view>.mp4", ...}`).

## How experiments are found (no registration needed)

- Every `experiment_*.py` in the repo root with its same-named `.md` file becomes a card. The card
  shows the Markdown's `# Title`, `## Summary` (1-3 sentences: the approach and its outcome; if it
  is missing, a paragraph of the script's docstring is used), `## Research question`, and, when
  expanded, `## Goal`, `## Setup` and `## Results`.
- A run directory `runs/<name>/` belongs to an experiment if the experiment's Markdown contains its
  command (`uv run experiment_x.py ... --name <name>`, or `--module experiment_x --ckpt runs/<name>/...`),
  if a running process trains or evaluates it, if its config contains `"experiment": "experiment_x"`,
  or if the run's config keys are exactly the script's command-line flags.
- Numbers come from the files the experiment scripts already write: `results.json` (or `log.txt`
  while a run is in progress), `wm_only.json`, and `lazy_enemy_eval.json` from
  `evaluate_lazy_enemy.py` (also in subdirectories of a run, or next to a checkpoint in `runs/`
  selected by `select_robust_checkpoint.py`).
- The three videos show the newest checkpoint (latest `round*.pkl`, else `best.pkl`, else
  `wm_only.pkl`) of the experiment's most recently updated run: (1) the actor in the real
  `lazy_enemy` game, with the world model's gates on the real history, (2) the actor inside the
  world model started from real Pong histories, (3) the same started from real `lazy_enemy`
  histories. It needs the module interface used by
  `evaluate_lazy_enemy.py`: `build_model(cfg)`, `wm_step(...)`, `ACT_GREEDY` / `ACT_SAMPLE`.
- Status: *goal met* (some held-out evaluation reached a final score of at least +10 over at least 10
  games), *running*, *evaluated*, *trained*, *world model only*, or *no runs yet*.

The dashboard only displays `lazy_enemy` results. Never use it to choose checkpoints or
hyperparameters (see `AGENTS.md`).
