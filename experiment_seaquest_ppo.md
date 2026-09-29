# Experiment S1: PPO on base Seaquest with a task-shaped reward

Code: `experiment_seaquest_ppo.py`, `seaquest_common.py`; held-out evaluation with `evaluate_gravity.py`.

## Summary

Stopped after the MLP agent stalled at about 0.1 rescues per base game. Model-free PPO with an MLP actor-critic on the object-centric state of base Seaquest, with the reward
shaped towards the goal behaviour (divers picked up, rescues with 6 divers). This first tests whether
the agent can learn to rescue divers at all, and then how it holds up under the held-out `gravity` mod.

## Goal

An agent trained on the base Seaquest game only that, with the held-out `gravity` mod, collects 6
divers and surfaces with them at least twice per game (`successful_rescues` >= 2), on average over
at least 10 games.

## Research question

Can PPO on the base game, with a reward shaped towards rescuing divers, learn the long-horizon
"collect 6 divers, then surface" behaviour? How well does the resulting agent transfer to a changed
dynamics of its own submarine (`gravity`)?

## Setup

- Environment: JAXAtari `seaquest`, `AtariWrapper` (sticky actions 0.25, noop resets, loss of a
  life ends the value horizon) and `ObjectCentricWrapper` (frame skip 4, 4 stacked frames of the
  284-dim object-centric state, fixed feature scaling, score dropped). 18 actions.
- Reward (from the base game's state): score gain / 100 + `diver_bonus` per diver picked up +
  `rescue_bonus` (10) per successful rescue - `surface_penalty` when surfacing costs a diver
  (1-5 divers) - `life_penalty` per life lost.
- PPO: 512 parallel base games, 128 steps per rollout, 4 epochs, 8 minibatches, lr 2.5e-4 (linear
  decay to 10%), gamma 0.995, GAE lambda 0.95, clip 0.2, separate 2 x 512 ReLU MLPs for actor and critic.
- Every 250 updates: 32 full base games (sampled and greedy actions), checkpoint `round<k>.pkl`.
  Selection: the checkpoint and action mode with the most base-game rescues (then score), `best.pkl`.
  `gravity` is only evaluated afterwards on `best.pkl` with `evaluate_gravity.py`.

## Experiments

A debug run with the default `life_penalty 2` collapsed within 50 updates into staying at the
surface (few divers, entropy 1.4), so both main runs use `--life_penalty 0` (losing a life already
ends the value horizon):

```
CUDA_VISIBLE_DEVICES=4 uv run experiment_seaquest_ppo.py --name sq_ppo_a --life_penalty 0
CUDA_VISIBLE_DEVICES=5 uv run experiment_seaquest_ppo.py --name sq_ppo_b --life_penalty 0 --surface_penalty 0 --diver_bonus 2 --ent_coef 0.02
```

## Results

- `sq_ppo_a` (surface penalty 0.5, diver bonus 1, entropy 0.01) collapsed by update 250 (66M frames)
  to picking up almost no divers (0.1-0.4 per rollout): the penalty for surfacing with fewer than 6
  divers made the whole dive-and-surface cycle look bad. It was stopped there to free the GPU.
  Base-game evaluation at update 250: 0 rescues, 1.5 divers, score 233 per game.
- `sq_ppo_b` (no surface penalty, diver bonus 2, entropy 0.02) started rescuing divers but then
  stalled. Base-game evaluations (32 games; sampled / greedy rescues per game): update 250: 0.09 / 0.03
  (6.1 divers, score 239); update 500: 0.12 / 0.09 (8.0 divers, score 430); update 750: 0.03 / 0.06
  (8.7 divers, score 356). The policy entropy stayed high (about 2.3 of a maximum of 2.89). In the base
  game the agent dies often in encounters with enemies and surfaces early (at 50-60 of 64 oxygen),
  losing a diver each time. It was stopped after about 800 updates (210M frames) in favour of S3.

Conclusion: the flat MLP learns to pick up divers (about 8 per game) but not to keep 6 of them and
surface, and it does not survive long enough. Not evaluated on `gravity`: the base-game behaviour is
far below the goal. See S3 (object-centric attention agent).
