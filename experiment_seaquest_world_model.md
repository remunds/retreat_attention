# Experiment W1: Seaquest agent trained entirely in a learned object-centric world model

Code: `experiment_seaquest_world_model.py` (agent from `experiment_seaquest_object_attention.py`,
base-game evaluation from `experiment_seaquest_ppo.py`), held-out evaluation with `evaluate_gravity.py`.

## Summary

**Goal met with the world-model method:** 8.88 rescues per game under the held-out `gravity` mod (all
32 games with at least 2), with an agent trained only in imagination. The Pong world-model method
applied to Seaquest. An object-centric transformer world model is
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

Further fixes found while running (each time the runs were restarted and their directories
removed):

1. The rescue rule "6 divers -> 0" never fired. In the real game a rescue is the step where 6 divers
   drop to 5 at the surface, and the divers are then counted down one per step while oxygen turns
   into points. The signed diver reward would also have punished that countdown. New rule: rescue =
   6 -> 5 at the surface (player y <= 50) without a death. Pickups are paid only above the highest
   count reached in the imagined episode, so a lose-and-regain pays nothing.
2. The running maximum was reset to 0 at a rescue, so the countdown values (5, 4, ...) were paid as
   pickups: 7.4 imagined pickups and 1.0 rescues per 64 imagined steps, with 0 real rescues. Fix: a
   countdown flag; while it runs, no pickups and no second rescue are counted. The bookkeeping
   (`diver_bookkeeping`) was validated on 3000 real steps of a real game: it counts 7 / 7 rescues
   and 43 / 44 pickups (the missed one re-collects a diver lost at a surfacing, which is
   deliberately not paid).
3. Deaths were taken as "predicted probability > 0.5". Because death events are up-weighted 5x in
   training, this gave about 0.9 % false deaths per imagined step (an imagined life ended after about
   110 steps on average, versus about 180 real steps). The agent could not tell real dangers from
   random false alarms (after 3 rounds: 0 real rescues, about 5 divers and 150 points per game,
   3 lives lost within 550-870 steps). Fix: deaths are sampled from the calibrated probability
   (logit - log 5).
4. After round 1 the agent sat at the surface 67 % of the time in the real game (0.08 pickups per 64
   steps) but was credited with 1.2 imagined pickups per 64 steps from the same real states. In
   imagination it hovered just below the surface (y 50-60, where real pickups from the top diver lane
   are possible but rare), and the world model predicted far too many diver collisions there.
   Diagnostics (all base game): diving from the surface is predicted correctly; one-step action
   distributions on real vs predicted frames differ by only 0.045 (total variation); inactive enemy
   slots were decoded with visual id 0 instead of 4 / 5 (masked everywhere, fixed anyway). Fix: a
   pickup is only rewarded with evidence in the predicted frames (a diver that was active next to the
   submarine disappears), and diver-count changes are weighted 10x in the world-model loss. The
   bookkeeping still counts 43 / 44 real pickups and 7 / 7 real rescues on a real game.
5. With that check, imagined pickups dropped to exactly 0 while imagined rescues stayed high (0.32 per
   game per 64 steps). On real pickup transitions the world model predicted the diver count +1 in 56 %
   of cases, but the picked-up diver disappearing in 0 % (real frames: 98.6 %): despawns are rare, so
   the existence loss was dominated by "stays active". Fixes: slots whose existence changes (pickups,
   kills, spawns) are weighted 10x in the loss; the pickup evidence is a diver right next to the
   submarine in the current frame; the held count only rises with evidence, and a rescue requires
   a legitimately held 6 (so a hallucinated jump of the counter to 6 cannot be cashed in). The
   bookkeeping still counts 43 / 44 real pickups and 7 / 7 real rescues.
6. Still 4-5x more imagined than real pickups from the agent's own real states (e.g. 1.09 vs 0.24 per
   64 steps). One step ahead on the agent's own real states only 12 % of the model's rewarded pickups
   were real (recall 86 %): the 10x up-weighting of changes in the loss inflates the predicted odds of
   a change, the same issue as with deaths. With calibrated logits (subtract log 10 for changes of the
   divers carried and of slot existence) and argmax decoding: precision 64 %, recall 15 %; with
   calibrated *sampling*: 627 predicted vs 335 real pickups, precision 16 %. So the model could not
   discriminate pickup states well. Change: each slot token also gets its position relative to the
   submarine in all 4 frames (collisions depend on relative positions), and the divers carried are
   sampled from the calibrated distribution. The runs were restarted.

Full runs with all fixes:

```
CUDA_VISIBLE_DEVICES=4 uv run experiment_seaquest_world_model.py --name sqwm_b
CUDA_VISIBLE_DEVICES=5 uv run experiment_seaquest_world_model.py --name sqwm_c --imag_len 64 --ppo_updates 400
CUDA_VISIBLE_DEVICES=4 uv run experiment_seaquest_world_model.py --name sqwm_c_s1 --seed 1 --imag_len 64 --ppo_updates 400
```

`sqwm_b` was stopped after round 2 (base-game rescues 0.09 per game) in favour of a second seed of
the better `sqwm_c` configuration (`sqwm_c_s1`). Long imagined episodes (256 steps) let model errors
accumulate further.

## Results

Base-game evaluation after each round of `sqwm_c` (32 real games from the normal start, at most
10,000 agent steps each, sampled / greedy actions). Each round adds 1.02M real steps (the replay
keeps the last 5M):

| round | real frames so far | rescues per game | divers per game | score |
|---|---|---|---|---|
| 0 | 4.1M | 0.00 / 0.00 | 5.9 / 5.0 | 274 / 207 |
| 1 | 8.2M | 1.38 / 1.34 | 16.8 / 17.9 | 2909 / 2882 |
| 2 | 12.3M | 3.53 / 3.50 | 31.2 / 30.7 | 10343 / 10248 |
| 3 | 16.4M | 4.94 / 5.03 | 41.3 / 41.1 | 17881 / 18298 |
| 4 | 20.5M | 4.94 / 5.34 | 45.4 / 43.9 | 16942 / 18955 |
| 5 | 24.6M | 5.22 / 5.38 | 46.4 / 49.2 | 20569 / 21342 |
| 6 | 28.7M | 10.97 / **12.12** (selected) | 74.3 / 81.4 | 53266 / 62934 |
| 7 | 32.8M | 7.47 / 10.25 | 58.3 / 72.8 | 33704 / 54048 |

**Held-out evaluation** of the selected checkpoint (round 6, greedy; `evaluate_gravity.py`, 32 full
games per environment with the same seeds from `PRNGKey(12345)`, at most 27,000 agent steps per game),
evaluated once:

| environment | rescues per game | games with >= 2 | divers per game | score | steps per game |
|---|---|---|---|---|---|
| base game | 11.12 | 30 / 32 | 78.8 | 58396 | 6689 (2 games hit the step cap) |
| **`gravity`** | **8.88** | **32 / 32** | 63.6 | 39425 | 4870 |

```
CUDA_VISIBLE_DEVICES=5 uv run evaluate_gravity.py --module experiment_seaquest_world_model --ckpt runs/sqwm_c/best.pkl
```

Rescues per game under `gravity`: 11, 6, 10, 16, 4, 5, 13, 11, 11, 16, 5, 4, 12, 3, 13, 14, 5, 7, 13, 12,
5, 3, 4, 12, 18, 2, 5, 16, 4, 10, 4, 10 (minimum 2).

**The goal is met with the world-model method**: the agent was trained only on imagined transitions
of a world model fitted to base-game data, and with the held-out `gravity` mod it rescues 6 divers
8.88 times per game on average, at least twice in every one of the 32 games.

### Interpretation

- **The world model has to be made unexploitable before imagination training works.** Each failure
  above was the agent finding a way to collect imagined reward that the real game does not give:
  event heads that fire without the matching state change, a countdown paid as pickups, uncalibrated
  up-weighted events (false deaths, 7x too many pickups), and pickups without a diver nearby. The
  fixes were to compute rewards from predicted *states* with evidence checks, to validate the reward
  bookkeeping against the real game's counters (7 / 7 rescues, 43 / 44 pickups), to calibrate and sample
  every up-weighted prediction, and to give the model relative positions for collisions. After that,
  real rescues appeared within 2 rounds and reached 12 per base game.
- **Short imagined episodes work better** (64 steps: 1.38 rescues per game after round 1; 256 steps:
  0.09, the run was stopped), as model errors compound over long rollouts.
- **Robustness to `gravity`:** the imagination-trained agent loses only about 20 % of its base-game
  rescues under gravity (11.1 -> 8.9), versus about 47 % for the model-free S5 agents (15.7 -> 8.4).
  No perturbation of the submarine's dynamics was used here. The imagined dynamics are noisier than
  the real game (sampled spawns, deaths and diver counts, rounding), which may itself make the agent
  more tolerant of small changes. Testing that would need an ablation.
- **Data efficiency:** 32.8M real frames in total, versus 262M for the model-free S5 agent at a
  similar base-game level.
- Limitations: one seed evaluated so far (a second seed, `sqwm_c_s1`, is running); base-game
  performance fluctuates between rounds (round 7 was worse than round 6); the diver curriculum and
  the reward bookkeeping use knowledge of Seaquest's rules (how rescues and the diver countdown work).
