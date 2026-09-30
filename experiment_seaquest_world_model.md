# Experiment W1: Seaquest agent trained entirely in a learned object-centric world model

Code: `experiment_seaquest_world_model.py` (agent from `experiment_seaquest_object_attention.py`,
base-game evaluation from `experiment_seaquest_ppo.py`), held-out evaluation with `evaluate_gravity.py`.

## Summary

The Pong world-model method applied to Seaquest. An object-centric transformer world model is
fitted on real base-game data, with one token per object slot plus global and action tokens. The
object-attention agent is trained with PPO purely on imagined games from real start histories. Real
interaction only collects world-model data and evaluates checkpoints. S5's diver curriculum shapes
the real data and the imagined start states.

## Goal

An agent trained **inside a learned world model** of the base Seaquest game that, with the held-out
`gravity` mod, collects 6 divers and surfaces with them at least twice per game
(`successful_rescues` >= 2), on average over at least 10 games.

## Research question

Can an object-centric world model of Seaquest (35 object slots, stochastic spawns, oxygen, divers,
surfacing and death rules) be accurate enough that an agent trained only in imagination learns the
long-horizon rescue behaviour, and is that agent robust to `gravity`?

## Setup

- World model: tokens = 35 object slots (4 frames of position, existence, orientation, visual id,
  plus last velocity), 1 global token (oxygen, divers, lives over 4 frames), 1 action token, learned
  slot embeddings; 4 pre-LayerNorm transformer blocks (d = 128, 4 heads). Heads per slot: existence
  (BCE), displacement for objects that stay (MSE, units of 8 px), absolute position for spawns, and
  orientation and visual id (CE). Global head: oxygen (MSE), divers carried and lives (CE), score
  gain (symlog MSE), rescue and death events (BCE, positives weighted 5x). Imagination samples spawns
  and decodes the rest deterministically; positions are rounded.
- Deaths: the terminal event is the start of the death animation (collision or no oxygen).
  Death-animation frames are excluded from world-model training.
- Real data per round: 512 base games x 2000 steps with the diver curriculum (probability 0.5 at
  game start / new life). Transitions changed by a gift are excluded. The replay keeps the last 5
  rounds (5M transitions). World model: 20k updates per round (batch 512, AdamW 3e-4), continued
  across rounds.
- Imagination: 2048 imagined games, 64-step PPO rollouts, 300 PPO updates per round; episodes start
  from 200k real 4-frame histories (with probability 0.5 the divers carried are set to k ~ U{0..6})
  and end at a predicted death or after 256 steps (value bootstrap). Reward: score gain / 100 +
  2 x divers picked up + 10 x rescue + potential shaping (as S3/S5).
- 8 rounds. Selection: base-game rescues (32 games from the normal start), then score. `gravity` only
  afterwards.

## Experiments

World-model-only checks on round-0 data (random actions + curriculum, 1M transitions; run
directories not kept):

```
CUDA_VISIBLE_DEVICES=4 uv run experiment_seaquest_world_model.py --name sqwm_wmonly --wm_only 1 --wm_steps 10001
```

- First version (terminal event = life counter decrement, death animation included): moving-object
  error 5.8 px RMSE, lost lives almost never predicted (recall ~0; the counter drops at the end of
  the animation, whose timing is hidden).
- Final version (death start as event, animation excluded): moving objects 2.2 px RMSE (player 1.1,
  divers 0.7, enemies 1.0, projectiles 6.0), death recall 0.77 with 0.9 % false alarms per step,
  diver-count changes 56 % correct, rescue recall 0.5-0.67.

First full run `sqwm_a` (stopped after round 1; run directory removed): the imagined reward came
from the world model's event heads (rescue event, score gain, positive diver changes). By round 1
the agent collected 1.12 imagined rescues per game per 64 imagined steps, but 0 in the real game
(4.1 divers per game). It had learned to trigger predicted rescue events without the matching state
change. Fix: rewards are computed from the predicted state change (rescue = 6 divers -> 0 without a
death; signed diver changes; kill score capped at 90).

Full runs with the fix:

```
CUDA_VISIBLE_DEVICES=4 uv run experiment_seaquest_world_model.py --name sqwm_b
CUDA_VISIBLE_DEVICES=5 uv run experiment_seaquest_world_model.py --name sqwm_c --imag_len 64 --ppo_updates 400
```

## Results

Runs in progress.
