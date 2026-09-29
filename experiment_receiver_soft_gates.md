# Experiment 5: receiver-driven soft / straight-through gates in the world model

Code: `experiment_receiver_soft_gates.py` (full pipeline of `experiment_intervention_actor.py` with
a different world model), diagnostic `ablation_ball_enemy.py`.

## Summary

Receiver-driven soft (sigmoid + L1) and binary straight-through gates, plus an ablation of how much the ball's prediction needs the enemy. No variant closed ball <- enemy in free flight, although removing it entirely costs little accuracy.

## Goal

Make the world model object-contained (the ball ignores the enemy except near a bounce), as a
prerequisite for training the actor under enemy interventions (experiment 2), with the final aim of
a final score of at least +10 with `lazy_enemy`.

## Research question

1. How much does the ball's prediction actually need the enemy? (ablation)
2. Do receiver-driven gates with a deterministic sigmoid (L1 penalty, thresholded at 0.1 when used)
   or with binary straight-through gates remove the ball<-enemy leakage?

## Experiments

Ablation (experiment 1's world model on 512k random-policy transitions, with and without the
ball<-enemy message; output in `runs/ablation_ball_enemy.log`):

```
CUDA_VISIBLE_DEVICES=4 uv run ablation_ball_enemy.py
```

World-model-only runs on round-0 (random-policy) data:

```
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run experiment_receiver_soft_gates.py --name rsg_wmonly_g0.02 --gate_coef 0.02 --wm_only 1
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run experiment_receiver_soft_gates.py --name rsg_wmonly_g0.1 --gate_coef 0.1 --wm_only 1
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run experiment_receiver_soft_gates.py --gate_type ste --name rste_wmonly_g0.02 --gate_coef 0.02 --wm_only 1
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run experiment_receiver_soft_gates.py --gate_type ste --name rste_wmonly_g0.1 --gate_coef 0.1 --wm_only 1
```

No full (actor) run was done, because no variant removed the leakage.

## Results

**Ablation**: the ball's 1-step error (pixels, mean) with and without messages from the enemy:

| ball position / direction | with enemy | without enemy |
|---|---|---|
| x < 30, towards enemy (bounce zone) | 3.08 | 3.86 |
| 30 <= x < 60 | 0.52 / 0.74 | 0.58 / 0.87 |
| 60 <= x < 120 | 0.77 / 0.37 | 0.86 / 0.44 |
| x >= 120 (player side) | 2.20 / 1.44 | 2.60 / 2.01 |
| **all** | **0.71** | **0.90** |

The enemy helps the ball's prediction mainly at the enemy's bounce, and only slightly elsewhere.

In unmodified Pong the enemy's height is always close to the ball's (99 % of gaps < 19 px when the
ball is far from the enemy). Swapping in an enemy with a larger gap is out of distribution; for
experiment 1's model the ball's change grows from 5 px (gap < 10 px) to 19 px (gap > 60 px).

**Receiver-driven gates** (leakage in pixels, ball<-enemy):

| gates | gate_coef | 8-step ball MSE | open gates / receiver | near | far (moving) | waiting |
|---|---|---|---|---|---|---|
| soft (sigmoid + L1, thr. 0.1) | 0.02 | 0.36 | 0.77 | 6.03 | 12.16 | 1.76 |
| soft (sigmoid + L1, thr. 0.1) | 0.1 | 0.35 | 0.28 | 5.66 | 12.38 | 0.31 |
| straight-through binary | 0.02 | 0.33 | 0.80 | 7.82 | 12.93 | 0.30 |
| straight-through binary | 0.1 | 0.34 | 0.49 | 5.01 | 13.32 | 0.27 |

### Interpretation

None of the gating variants closes ball<-enemy in free flight, even though the ablation shows that
it gains the ball little there, and even with binary gates where an open gate costs the full
penalty. The gate penalty only acts on the training distribution, where an enemy that tracks the
ball is a harmless, slightly useful input. Nothing in the unmodified-Pong data tells the model that
an enemy far from the ball's height should not matter, so sparsity pressure alone does not give
invariance to out-of-distribution enemy behaviour. This motivated experiment 6, which applies the
interventions to the actor's observation instead of passing them through the world model.
