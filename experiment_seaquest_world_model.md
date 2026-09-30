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
```

## Results

Base-game evaluation after each round (32 real games from the normal start, sampled / greedy):

| run | round | rescues per game | divers per game | score |
|---|---|---|---|---|
| `sqwm_b` (imagined episodes <= 256 steps, 300 PPO updates / round) | 0 | 0.00 / 0.00 | 5.2 / 5.0 | 212 / 223 |
| | 1 | 0.09 / 0.09 | 7.6 / 9.0 | 422 / 532 |
| `sqwm_c` (imagined episodes <= 64 steps, 400 PPO updates / round) | 0 | 0.00 / 0.00 | 5.9 / 5.0 | 274 / 207 |
| | 1 | **1.38 / 1.34** | 16.8 / 17.9 | 2909 / 2882 |

After the fixes above, the agent trained only in imagination rescues 6 divers in the real base game.
Runs in progress.
