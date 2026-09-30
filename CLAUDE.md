# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research project: an **object-centric world model for Pong** built on [JAXAtari](https://github.com/k4ntz/JAXAtari). The full goals are in [project.md](project.md). Read it before making design decisions. In short:

- The model predicts the next state of each object (player paddle, enemy paddle, ball) **independently**, not whole frames.
- Each object's prediction uses its own history plus the history of the objects it **attends to at the current step**. These dependencies are decided again at every time step (e.g. the ball depends on a paddle only near a bounce), not fixed.
- The per-step dependencies must be **visualizable**: for any object and any rollout step, show which objects it depended on.
- Robustness test: train on the unmodified game, evaluate with the enemy's dynamics changed by **JAXAtari's own `lazy_enemy` mod** (`EVAL_MOD` in `ocwm/env.py`). The user wants JAXAtari's mod, not a custom one. It makes the enemy track the ball only while the ball moves toward it (and still pause every 8th frame); after a goal the enemy resets to y=115.
- Success criteria: `ocwm/benchmark.py` reports three candidates side by side (the user asked for all three): **A** the original ">10 points"; **B** actor + model checks (win ≥90%, winners per hit above the best model-free baseline and ≥90% of the oracle planner); **C** counterfactual containment of the ball/player predictions.

## Setup and commands

Dependencies are managed with `uv` (Python ≥3.12). `jaxatari` is installed from its GitHub repo (see `[tool.uv.sources]` in `pyproject.toml`). JAX runs on CPU on this machine.

```bash
uv sync
.venv/bin/install-sprites    # one-time; needed before any env can be made. Answering "n" installs replacement sprites (no ROM ownership claim)

uv run pytest                                    # all tests (~5 s)
uv run pytest tests/test_pipeline.py::test_frozen_enemy_never_moves   # single test

uv run python -m ocwm.data --out data/train.npz --num-envs 128 --num-steps 4096   # trajectories, unmodified game
uv run python -m ocwm.train --data data/train.npz --out runs/v3 --sparsity 0.2    # neural model, ~20 min on CPU
uv run python -m ocwm.synthesis --data data/train.npz --out runs/programs         # program model, ~1 min
uv run python -m ocwm.synthesis --data data/train.npz --out runs/programs_noenemy --exclude enemy   # program player that only knows ball + itself
uv run python -m ocwm.evaluate --ckpt runs/programs/programs.json                # 1-step / 16-step error + dependency rates per mod
uv run python -m ocwm.visualize --ckpt runs/programs/programs.json --mods lazy_enemy --out runs/programs/gates.png
uv run python -m ocwm.actor --policy planner --predictor model --ckpt runs/programs/programs.json --mods   # planner, unmodified game
uv run python -m ocwm.actor --policy planner --predictor oracle      # same planner on the true env (upper bound); default mod lazy_enemy
uv run python -m ocwm.actor --policy fixed --offset 8                # model-free baseline
uv run python -m ocwm.benchmark --ckpt runs/programs/programs.json   # all success criteria (~15 min)
uv run python -m ocwm.video --ckpt runs/programs/programs.json --out runs/videos/programs_train.mp4   # one player's video: game, its dependencies, program branches or neural gate timeline (~2 min); --mods lazy_enemy for eval
uv run python -m ocwm.decisions --ckpts runs/programs/programs.json runs/v3/model.pkl --out runs/decisions   # dependency weights behind each planner decision
```

`data/` and `runs/` are gitignored. Every `--ckpt` accepts either a neural `.pkl` or a program `.json` (`model.load` dispatches); both expose `next_positions(hist, action) -> (next_pos, deps)` and `config.history`, which is all the actor, evaluate, benchmark and visualize need. Pass `--mods` with no values for the unmodified game. With zsh, split a variable holding several args with `${=args}`.

## Architecture (`ocwm/`)

- `env.py`: `make_env(mods)` and `object_positions(obs)` → `[3, 2]` (x, y) array in fixed order player, enemy, ball. It also registers the custom **`frozen_enemy`** mod into JAXAtari's `PongEnvMod.REGISTRY` as an import side effect, so any `make_env` call can use it.
- `data.py`: jitted `vmap`+`scan` collection with a noisy ball-tracking policy. The random policy almost never returns the ball, so it gives too few bounces. Arrays are `pos [N, T+1, 3, 2]`, `action [N, T]`, `done [N, T]`. `valid_indices` drops steps after a game ends, and `gather_windows` slices `(history, action, next_pos)` batches on-device.
- `model.py`: `WorldModel` (flax **nnx**).
  - **Gates.** For each target object and each source (other objects + the action), a gate MLP on pairwise features decides a binary gate at every step.
  - **Prediction.** The target's Δposition = head(self_enc(own history) + Σ gate·message(target, source)).
  - **Invariant:** a closed gate makes the prediction exactly independent of that source (tested). Keep all cross-object information flowing through gated pair messages only.
  - **Training vs evaluation.** Training uses straight-through Gumbel-sigmoid gates, and evaluation thresholds the logits. Gates start open (bias +3), and `train.py` ramps the sparsity penalty in over the first 30% of steps. Without that, rare but needed dependencies get pruned before they are learned.
  - **Normalization and helpers.** Δ is normalized per object by `delta_mean/std` (`Stat` variables, not trained, saved in the checkpoint). `imagine()` rolls the model out autoregressively, and `save`/`load` pickle the config plus the pure-dict state.
- `train.py`: functional loop (`nnx.split` into `Param`/`Stat`, optax, `jax.jit`). Loss is an 8-step autoregressive rollout (`--rollout`). Half of each batch comes from per-object "surprising" windows (`data.surprise_masks`: deviation from constant velocity), otherwise rare bounces get no weight. Envs in the last 10% are validation.
- `programs.py` + `synthesis.py`: the second approach, **program synthesis**. The DSL has projections (`ball.x`, `ball.vy`, …, from the last two positions), action flags, constants, `+ − * neg sign`, and `ite(term <= θ, …)`. Search is divide-and-conquer enumerative synthesis:
  - Enumerate terms bottom-up (size ≤ 4, deduplicated by their outputs on the data).
  - Leaf candidates add fitted constants (a·t + b and two-term fits, trimmed least squares).
  - Leaf candidates are chosen by greedy set cover. Top-by-count alone picks only variants of the common case, so e.g. `-ball.vx` (a bounce) would never be a candidate.
  - Splits are chosen by **information gain**, not by the number of transitions explained. Bounces need two nested conditions, and the first one alone explains nothing more.
  - Split thresholds are refined to exact data values.
  - Reduced-error pruning on a second sample removes splits that don't help.
  - `ProgramModel` computes per-step dependencies from the sources read on the branch actually executed.
- `actor.py`: the paddle follows the ball with a hit offset k (which fifth of the paddle is hit, and so the rebound angle). `planner` imagines each of 7 candidate k in closed loop for 256 steps and scores the imagined outcome (`score_trajectory`: goals detected by the ball passing a paddle column or jumping back to x=78). `fixed` is the model-free baseline. Real steps use sticky actions (p=0.25); `play` counts player hits for winners per hit.
- `benchmark.py`: the success criteria (see top). Criterion C is counterfactual: on unmodified-game windows where the ball stays at x > 40, the enemy's trajectory is replaced by the one JAXAtari's lazy enemy would take (`lazy_enemy_track`), and the ball/player rollout errors must not rise >10%. Comparing games played under the mod would be confounded, because the whole game changes.

## Current findings (as of 2026-09-29)

- **Players under JAXAtari's `lazy_enemy`** (benchmark, 16 games, sticky actions; "won" = reached 21 before the 30k-frame cap):

  | Player | Won | Score | Winners per hit | Criterion C (containment) |
  |---|---|---|---|---|
  | Program, all objects (`runs/programs`) | 75% | 20.5–6.2 | 0.21 | FAIL (16-step ball +14%) |
  | Program, enemy locked away (`runs/programs_noenemy`, `--exclude enemy`) | 69% | 20.3–2.3 | 0.21 | PASS (identical errors) |
  | Neural v3 | 6% | 17.2 avg | 0.16 | FAIL (ball +16%, player +13%) |
  | Model-free center hit (offset 8) | 100% | 21–2.1 | 0.33 | – |
  | Oracle planner | 100% | 21–4.4 | 0.42 | – |

  The enemy-locked player's ball programs learn "the ball bounces back at the enemy column" (the normal enemy returns almost everything), so it never imagines scoring and plays safe. It is the most robust (same results on both games) but scores slowly. No model-based planner beats the model-free center hit yet.

- **Enemy speed decides whether the game is a contest** (from an earlier, now removed, custom enemy-slowdown mod). Model-free center follower vs enemy moving on every frame: 0–4. Base game (7 of 8 frames): 21–2. Every 2nd frame: 21–1. Every 3rd/4th frame: 21–0; at every 4th frame the player never touches the ball (the enemy can't reach a serve), so no score-based criterion could separate models. JAXAtari's `lazy_enemy` does not have this problem: with a competent player the ball comes back and the enemy returns shots (center follower 21–2, winners per hit 0.33; oracle planner 0.42).
- **Actors on the unmodified game** (16 games, sticky actions):

  | Actor | Score (player–enemy) | Winners per hit |
  |---|---|---|
  | Oracle planner | 21–4.4 | 0.42 |
  | Best fixed offset (8) | 21–2 | 0.33 |
  | Program-model planner | 21–4.6 | 0.29 |
  | Neural v2 planner | 15.6–19.5 | 0.21 |

  The planner is only as good as its bounce predictions.
- **Program model** (`runs/programs`): about 99.4% exact on ball transitions, including 91% of player-paddle bounces for x and 85% for y. `enemy.dy` top level is JaxPong's exact rule (±2 / 0 by `enemy.y - ball.y`). `player.dy` recovers the paddle physics `0.7*vy ± 1.725`. 1-step error is 2–4× lower than neural, 16-step ball error is worse. Dependencies are sparse and match the game (ball reads paddles only at bounces), but deep `enemy.dy` splits chasing the hidden 8-frame pause read the player spuriously.
- **Neural gates:** with the 8-step loss, sparsity 0.02 left all gates open (v2). With 0.2 (v3), the enemy and ball drop the action, but the player still reads enemy and ball about 99% of the time.
- Hidden state (enemy pause every 8th frame, 60-frame ball hold after a goal) caps achievable accuracy: `enemy.dy` ≈ 87–89% exact.

## JAXAtari usage

- `env.reset(key)` → `(obs, state)`, `env.step(state, action)` → `(obs, state, reward, done, info)`. Both are pure, so wrap them in `jax.jit`/`vmap`.
- Actions (index into `ACTION_SET`): 0 NOOP, 1 FIRE, 2 RIGHT (= paddle **up**), 3 LEFT (= **down**), 4 RIGHTFIRE, 5 LEFTFIRE. FIRE on a paddle hit speeds the ball up.
- Hidden state that object positions don't reveal: `step_counter` (enemy pauses every 8th step; the ball is held at center for 60 steps after a goal) and `player_speed` (inferable from position history). This sets a floor on 1-step error.
- Observed ball motion lags the hit: at the step the ball reaches the paddle column its displacement is still +vx; the reversal shows in the next step.
- Pong mods live in `.venv/lib/python3.13/site-packages/jaxatari/games/mods/pong/pong_mod_plugins.py`. Pass them as `jaxatari.make("pong", mods=[...])`.
