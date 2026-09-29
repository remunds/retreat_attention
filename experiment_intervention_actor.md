# Experiment 2: L0-gated object world model + actor trained under enemy interventions

Code: `experiment_intervention_actor.py` (reuses PPO, data collection and rounds from
`experiment_sparse_object_wm.py`), evaluation with `evaluate_lazy_enemy.py`.

## Goal

Reach a **final score (agent points minus enemy points) of at least +10** with `lazy_enemy`, using
an actor trained only inside a world model fitted to unmodified Pong. Experiment 1 lost with
`lazy_enemy` (12.6 : 21 = -8.4).

## Research question

1. Do hard-concrete (L0) gates make the world model object-contained, i.e. is the ball's
   prediction invariant to the enemy far from a bounce?
2. Does training the actor in imagination under random interventions on the enemy's mechanism
   (freeze / scaled motion / random walk, all applied to the world model's enemy prediction) make
   it robust to a changed enemy?

## Experiments

- World-model-only runs on round-0 (random-policy) data, to check gates and leakage (the
  counterfactual leakage test `common.leakage`: swap one object's history with another sample's
  and measure the change of the other objects' predictions, in pixels):

  ```
  CUDA_VISIBLE_DEVICES=4 uv run experiment_intervention_actor.py --name wmonly_g0.05 --gate_coef 0.05 --wm_only 1
  CUDA_VISIBLE_DEVICES=5 uv run experiment_intervention_actor.py --name wmonly_g0.2 --gate_coef 0.2 --wm_only 1
  ```
- Full run `ia_v1` (seed 0, default config: gate_coef 0.05; intervention probabilities none 0.25,
  freeze 0.35, scale 0.2, random walk 0.2; 5 rounds; ~40 min on one GPU):

  ```
  CUDA_VISIBLE_DEVICES=4 uv run experiment_intervention_actor.py --name ia_v1
  CUDA_VISIBLE_DEVICES=4 uv run evaluate_lazy_enemy.py --module experiment_intervention_actor --ckpt runs/ia_v1/best.pkl
  ```

## Results

Leakage of the ball's 1-step prediction when the enemy's history is swapped (pixels; the reference
is experiment 1's final model: far 4.3 on actor data, 14.3 for its round-0 model on random data):

| model | ball<-enemy near | ball<-enemy far (ball moving) | ball waiting to be served |
|---|---|---|---|
| WM-only, gate_coef 0.05 | 4.75 | 11.02 | — |
| WM-only, gate_coef 0.2 | 5.70 | 10.56 | — |
| `ia_v1` round 0 | 4.36 | 12.38 | 1.28 |
| `ia_v1` round 4 | 4.77 | 6.56 | 0.80 |

Real unmodified Pong during training (player : enemy, 32 games):

| round | WM val loss | sampled | greedy |
|---|---|---|---|
| 0 | 1.60 | 18.53 : 18.31 | 18.66 : 17.09 |
| 1 | 1.47 | 20.53 : 15.31 | 20.78 : 14.75 |
| 2 | 1.59 | 21.00 : 9.03 | 20.75 : 5.25 |
| 3 | 1.51 | 21.00 : 5.72 | 21.00 : 4.53 |
| 4 | 1.31 | 21.00 : 3.78 | **21.00 : 2.53** (selected) |

Held-out evaluation of the selected checkpoint (32 games, seeds from `PRNGKey(12345)`):

| environment | player : enemy | final score | games won |
|---|---|---|---|
| Pong | 21.00 : 3.25 | +17.75 | 32 / 32 |
| **`lazy_enemy`** | 19.31 : 19.12 | **+0.19** | 17 / 32 |

World-model 1-step RMSE on `lazy_enemy` trajectories: ball 5.1 px (1.5 on Pong), enemy 6.7 px.

**Criterion not met** (+0.19 < +10), but a large improvement over experiment 1 (-8.38).

### Interpretation

- The L0 gates did **not** make the ball object-contained. The player's gates became clean
  (it uses only the action), but the ball kept *all* gates open at every step
  (`runs/ia_v1/gates_*.png`), so the ball's prediction reacts to the enemy even in free flight.
- The enemy interventions still helped a lot (-8.4 -> +0.2): the actor now wins half of the
  `lazy_enemy` games. But because the world model's ball depends on the enemy, intervening on the
  enemy also changes the imagined ball's flight. The actor is taught a false causal link
  ("the enemy's position moves the ball") and keeps reacting to the enemy. For interventions in
  imagination to teach the right lesson, the world model has to be causally correct (see
  experiments 4 and 5), or the intervention must not pass through the world model (experiment 6).
