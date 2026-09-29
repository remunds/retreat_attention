# Experiment 3: object-gated actor (sparse attention over objects) trained in imagination

Code: `experiment_gated_actor.py` (world model, data collection, imagination and rounds from
`experiment_sparse_object_wm.py`), evaluation with `evaluate_lazy_enemy.py`.

## Summary

Experiment 1's world model with an actor that gates its object inputs (ball, enemy) per step with hard-concrete gates and an expected-L0 penalty. The gates stayed open because the enemy is useful in unmodified Pong; `lazy_enemy` final score -4.9.

## Goal

Reach a **final score of at least +10** with `lazy_enemy` with an actor trained only in a world
model of unmodified Pong, by letting the *actor* decide per step which objects it looks at (as the
world model does), with a penalty for looking, so that it stops depending on the enemy.

## Research question

Does an expected-L0 penalty on per-step hard-concrete gates over the actor's object tokens (ball,
enemy; the player's own token is always used) make the actor ignore the enemy, and does that make
it robust to `lazy_enemy`?

## Experiments

- `ga_v1`: gate_coef 0.02 from the start, gate log-odds initialised at 0. The penalty closed both
  gates (ball and enemy) within 200 PPO updates, before the actor had learned anything; the blind
  actor lost real Pong 0.3 : 21. Stopped after round 0 (run directory not kept).
- `ga_v2`: gates initialised open (log-odds 3), penalty 0.005 after a 400-update warm-up. The
  gates stayed at 1.0 (saturated sigmoid, the penalty gradient was ~100x too small). Stopped after
  round 2 without evaluation (run directory not kept).
- `ga_v3` (full run, reported below): log-odds initialised at 1, penalty 0.05 after a 400-update
  warm-up, 5 rounds.

  ```
  CUDA_VISIBLE_DEVICES=5 uv run experiment_gated_actor.py --name ga_v3
  CUDA_VISIBLE_DEVICES=5 uv run evaluate_lazy_enemy.py --module experiment_gated_actor --ckpt runs/ga_v3/best.pkl
  ```

## Results

`ga_v3` on real unmodified Pong during training (player : enemy, 32 games):

| round | sampled | greedy |
|---|---|---|
| 0 | 4.41 : 21.00 | 2.31 : 21.00 |
| 1 | 20.72 : 14.47 | 20.97 : 14.00 |
| 2 | 21.00 : 6.66 | 21.00 : 6.06 |
| 3 | 21.00 : 4.94 | 21.00 : 2.91 |
| 4 | **21.00 : 1.66** (selected) | 21.00 : 1.88 |

The actor's gates stayed fully open during all rounds (gate_ball = gate_enemy = 1.0): the actor's
return improvement from reading the enemy outweighed the penalty.

Held-out evaluation of the selected checkpoint (32 games):

| environment | player : enemy | final score | games won |
|---|---|---|---|
| Pong | 21.00 : 1.59 | +19.41 | 32 / 32 |
| **`lazy_enemy`** | 15.97 : 20.91 | **-4.94** | 2 / 32 |

**Criterion not met.**

### Interpretation

A learned, reward-driven sparsity penalty on the actor's attention did not make it drop the
enemy: in unmodified Pong the enemy's position is genuinely useful for the actor (and it is highly
correlated with the ball), so the gate stays open, and the actor behaves like experiment 1's MLP
actor (-8.4 there, -4.9 here). Either the penalty is too weak to remove useful information, or too
strong and removes the ball too (`ga_v1`). Being useful in training is not the same as being
causally needed, which a penalty on the training objective cannot tell apart.
