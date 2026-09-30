# Experiment S5: start-state curriculum on the divers carried (with random currents)

Code: `experiment_seaquest_diver_curriculum.py` (PPO loop from S1, agents from S1/S3, currents from
S2); held-out evaluation with `evaluate_gravity.py`.

## Summary

The agents so far picked up 6-9 divers per game but almost never held 6 at once, so they rarely saw
the rescue reward. Here, at the start of each game and after each lost life in training, the divers
carried are set to a random 0-6 with probability 0.5 (a base-game start-state curriculum). S2's random
currents are also applied. Evaluation and selection use the unmodified base game.

## Goal

An agent trained on the base Seaquest game only that, with the held-out `gravity` mod, collects 6
divers and surfaces with them at least twice per game (`successful_rescues` >= 2), on average over
at least 10 games.

## Research question

Does starting training from states that already carry several divers make the long-horizon rescue
behaviour learnable, and does it then transfer, together with random currents, to the held-out
`gravity` game?

## Setup

- Curriculum: at every game start and after every lost life, with probability 0.5, divers carried
  := k ~ U{0..6}. The pickup bonus is not paid for these gifted divers.
- Currents (as S2/S4): per game, with probability 0.25 none, otherwise one of 8 directions with a
  per-step probability q ~ U(0, 1) of a 1 px displacement.
- Reward: S3's (S1's shaped reward with diver bonus 2 and rescue bonus 10, plus potential shaping
  towards the surface with 6 divers or with oxygen < 16).
- Agents: `--agent attention` (S3's object-attention agent, orientation bug fixed) and `--agent mlp`
  (S1's 2 x 512 MLP on `obs_features_v2`).
- PPO: 512 envs, 128 steps, 4000 updates (1.05B frames), lr 3e-4, entropy 0.01, gamma 0.995.
- Every 250 updates: 32 full games of the unmodified base game from its normal start. Selection:
  most base-game rescues, then score.

## Experiments

```
CUDA_VISIBLE_DEVICES=4 uv run experiment_seaquest_diver_curriculum.py --agent attention --name sq_dc_att
CUDA_VISIBLE_DEVICES=5 uv run experiment_seaquest_diver_curriculum.py --agent mlp --name sq_dc_mlp
```

## Results

Progress (base game from the normal start, 32 games, sampled / greedy actions):

| run | update (frames) | rescues per game | divers per game | score |
|---|---|---|---|---|
| `sq_dc_att` | 250 (66M) | 4.53 / 4.84 | 33.0 / 34.3 | 11634 / 13603 |
| `sq_dc_mlp` | 250 (66M) | 0.28 / 0.44 | 9.9 / 10.1 | 693 / 992 |

With the curriculum, the object-attention agent learns the rescue behaviour within 66M frames (all
earlier agents stayed below 0.2 rescues per game). The MLP learns it much more slowly. Runs in
progress; `gravity` is evaluated once on the finally selected checkpoint.
