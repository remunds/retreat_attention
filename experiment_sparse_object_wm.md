# Experiment 1: sparse-gated object-interaction world model + PPO in imagination

Code: `experiment_sparse_object_wm.py` (training), `evaluate_lazy_enemy.py` (held-out evaluation),
`common.py` (OC env pipeline, data collection, evaluation).

## Goal

Train an actor **entirely inside a learned world model**, where the world model is fitted only to
data from **unmodified Pong** (JAXAtari object-centric inputs), so that the actor scores **more than
10 points per game** in Pong with the `lazy_enemy` mod active (which it never sees during training).

> Note: the success criterion was later clarified as a **final score (agent points minus enemy
> points) of at least +10**, i.e. the agent has to win. See the results below.

## Research question

Can a world model with per-object predictions and sparse, per-step gated interactions, trained on
unmodified Pong only, produce (via pure imagination training) an actor that stays strong when one
object's dynamics (the enemy paddle) are changed at test time? And do the learned gates actually
select only the objects each prediction needs?

## Setup

- **Environment / inputs:** `AtariWrapper` -> `ObjectCentricWrapper` (frame skip 4, frame stack 4,
  sticky actions 0.25, noop resets). From each OC frame we use (x, y) of player, enemy and ball.
- **World model `SparseObjectWM`:** per-object encoders over the object's own last K=16 frames
  (positions + velocities); for every receiving object i and every other token j (two other
  objects + action token), a message `LayerNorm(MLP(e_i, e_j))` scaled by a gate
  `g_ij = sigmoid(MLP(e_i, e_j))` recomputed every step; L1 penalty on the gates (coef 0.01);
  per-object heads predict the next displacement; the ball head also predicts the reward
  (point won / lost / none). Trained with an 8-step open-loop (autoregressive) MSE + reward CE loss.
- **Actor:** PPO (2 x 256 MLP actor and critic) on 2048 imagined environments whose step function
  is the world model (ball/enemy positions rounded to integers). Imagined episodes start from real
  16-frame histories of the replay buffer, end when the model predicts a point (one rally = one
  episode) or after 256 steps (value bootstrap). The actor never trains on real transitions.
- **Rounds (Dreamer-style):** round 0 collects 512k real steps with a random policy; rounds 1-4
  collect 512k steps each with the current actor (10 % random actions). Each round: fit the world
  model on all data (20k steps, batch 512), then 400 PPO updates in imagination
  (~105M imagined steps per round), then evaluate on real unmodified Pong (32 full games).
- **Model selection:** the round and action mode (greedy vs. sampled) with the best
  *unmodified-Pong* score difference. `lazy_enemy` was evaluated once, afterwards, on the selected
  checkpoint only.

## Experiments

Run `sow_v1` (seed 0), on 1 GPU, ~35 min:

```
CUDA_VISIBLE_DEVICES=4 uv run experiment_sparse_object_wm.py --name sow_v1 --wm_steps 20000 --ppo_updates 400
CUDA_VISIBLE_DEVICES=4 uv run evaluate_lazy_enemy.py --module experiment_sparse_object_wm --ckpt runs/sow_v1/best.pkl
```

All other hyperparameters are the defaults in the script (see `runs/sow_v1/results.json`).
Evaluation uses 32 full games (to 21), each with a different seed derived from `PRNGKey(12345)`,
the same seeds for Pong and `lazy_enemy`.

## Results

World model (validation, held-out envs, 8-step open-loop, MSE in units of (8 px)^2) and actor score
on **real unmodified Pong** per round (player : enemy points per game, mean over 32 games):

| round | data | WM val loss | Pong (sampled) | Pong (greedy) |
|---|---|---|---|---|
| 0 | 512k random | 1.43 | 20.81 : 15.25 | 20.97 : 10.25 |
| 1 | +512k actor | 1.36 | 21.00 : 9.97 | 21.00 : 9.91 |
| 2 | +512k actor | 1.48 | 21.00 : 3.16 | **20.59 : 1.59** (selected) |
| 3 | +512k actor | 1.35 | 21.00 : 6.19 | 21.00 : 4.62 |
| 4 | +512k actor | 1.30 | 21.00 : 8.28 | 20.97 : 7.69 |

Held-out evaluation of the selected checkpoint (round 2, greedy), 32 games each:

| environment | player points / game | enemy points / game | games with > 10 points |
|---|---|---|---|
| Pong (unmodified) | 21.00 ± 0.00 | 2.03 | 32 / 32 |
| **`lazy_enemy`** | **12.62 ± 2.81** (min 7, max 18) | 21.00 | 25 / 32 |

This met the originally written criterion (> 10 points per game: 12.62), but **not the clarified
criterion**: the final score with `lazy_enemy` is 12.62 - 21.00 = **-8.38** (the actor loses every
game), versus +18.97 on unmodified Pong. Experiment 2 (`experiment_intervention_actor.md`) addresses this.

World-model open-loop RMSE in pixels on trajectories of the actor (1 / 10 steps ahead):

| object | Pong h1 | Pong h10 | `lazy_enemy` h1 | `lazy_enemy` h10 |
|---|---|---|---|---|
| player | 2.4 | 6.7 | 2.5 | 7.4 |
| enemy | 2.6 | 10.8 | 7.5 | 33.7 |
| ball | 3.4 | 13.0 | 7.6 | 26.5 |

Average gates (how much object *row* uses token *column*), on Pong:

| receiver | player | enemy | ball | action |
|---|---|---|---|---|
| player | 1 | 0.04 | 0.11 | 0.48 |
| enemy | 0.01 | 1 | 0.37 | 0.13 |
| ball | 0.04 | 0.60 | 1 | 0.08 |

Plots: `runs/sow_v1/gates_pong.png`, `runs/sow_v1/gates_lazy_enemy.png`.

### Interpretation

- Pure imagination training works well: already after round 0 (world model fitted to random-play
  data only) the actor wins real Pong, and after round 2 it wins 21 : 2.
- The actor transfers to `lazy_enemy` well enough to meet the criterion (12.6 points), but it is
  clearly hurt: it now *loses* the games (12.6 : 21). In the gate plots the actor often misses
  balls when the enemy paddle is stuck at a position it never had in training, so the policy
  itself reacts to the out-of-distribution enemy position.
- The gates are sparse and meaningful for the player (it uses the action, and the ball only near
  hits) and for the ball's use of the **player** and **action** (only around the player's hits).
  But the ball also relies strongly on the **enemy** (gate ~0.6, high whenever the ball moves
  toward the enemy), and the enemy uses the ball. In unmodified Pong the enemy tracks `ball_y`,
  so the enemy's position is a shortcut for the ball's position. Under `lazy_enemy` this shortcut
  breaks, and the ball's prediction error roughly doubles (3.4 -> 7.6 px at 1 step). So the world
  model is **not yet** robust in the intended, object-contained way: the change to the enemy
  leaks into the ball's prediction.
- Next steps: make the ball's dependence on the enemy contextual (e.g. stronger or hard-concrete
  gates, gate penalties that favour the object's own history, or interventional data
  augmentation such as randomly perturbing the enemy's history during world-model training on
  unmodified Pong), and make the actor ignore objects it does not need (e.g. an object-attention
  policy with the same sparsity bias).
