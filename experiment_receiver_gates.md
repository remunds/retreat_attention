# Experiment 4: receiver-driven L0 gates in the world model

Code: `experiment_receiver_gates.py` (full pipeline of `experiment_intervention_actor.py` with a
different world model).

## Goal

Make the world model object-contained, so that the actor can be trained under enemy interventions
(experiment 2) without the imagined ball reacting to the enemy, and ultimately reach a final score
of at least +10 with `lazy_enemy`.

## Research question

If whether object i listens to object j is decided from i's own state only
(g_ij = HC(f_i(e_i))_j), can an unusual enemy no longer open the ball's gate, so that the ball's
free flight is exactly invariant to the enemy?

## Experiments

World-model-only run on round-0 (random-policy) data, gate_coef 0.05:

```
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run experiment_receiver_gates.py --name rg_wmonly --wm_only 1
```

No full (actor) run was done, because the world model already failed (below).

## Results

| model | val loss | 8-step ball MSE | open gates / receiver | ball<-player leak | ball<-enemy near / far / waiting |
|---|---|---|---|---|---|
| exp. 2 L0 pair gates (reference) | 1.59 | 0.35 | 1.82 | 1.35 | 4.75 / 11.02 / — |
| receiver-driven L0 gates | 2.11 | 0.61 | 1.33 (constant) | 0.00 | 6.06 / 11.29 / 1.88 |

The gates collapsed to constant values that do not depend on the state (exactly 4 of 9 gates open
for every sample). The ball *never* listens to the player (so it cannot bounce off the player's
paddle correctly; the ball's error nearly doubled), and it *always* listens to the enemy.

### Interpretation

Receiver-driven hard-concrete gates got stuck in a bad optimum. In random-policy data the player
rarely hits the ball, so ball<-player is rarely useful and the penalty closes it; ball<-enemy stays
open everywhere. This variant was abandoned in favour of receiver-driven soft / straight-through
gates (experiment 5).
