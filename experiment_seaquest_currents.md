# Experiment S2: PPO on base Seaquest with random currents on the submarine

Code: `experiment_seaquest_currents.py` (training loop, agent and reward from `experiment_seaquest_ppo.py`),
held-out evaluation with `evaluate_gravity.py`.

## Summary

Same agent, reward and PPO settings as S1's run `sq_ppo_b`. During training, each base game gets
a random current that displaces the submarine by 1 px per step in one of 8 directions with a random
probability. This is a generic intervention on the agent's own dynamics, so that the agent learns to
counteract external forces. The held-out `gravity` mod is never used for training or selection.

## Goal

An agent trained on the base Seaquest game only that, with the held-out `gravity` mod, collects 6
divers and surfaces with them at least twice per game (`successful_rescues` >= 2), on average over
at least 10 games.

## Research question

Does training with random, generic perturbations of the submarine's own motion (currents in all
directions) make the agent robust to a changed dynamics of the submarine at test time, without
losing the rescue behaviour in the base game?

## Setup

- As S1 (`sq_ppo_b` settings: diver bonus 2, rescue bonus 10, no surface or life penalty, entropy
  0.02, 512 envs, 3000 PPO updates).
- Currents: per game, with probability `p_none` = 0.25 no current, otherwise a direction drawn
  uniformly from the 8 compass directions and a strength q ~ U(0, 1). After every agent step the
  submarine is displaced by 1 px along the direction with probability q (clamped to the playfield,
  not while exploding). A new game draws a new current.
- Evaluation during training and checkpoint selection use the plain base game (no currents).
- Design note: the perturbation family is generic (all directions, all strengths up to 1 px per
  step), and a downward current is one of its 8 directions. It was not fitted to the mod's rule.

## Experiments

```
CUDA_VISIBLE_DEVICES=4 uv run experiment_seaquest_currents.py --name sq_cur_a
```

## Results

Stopped at update 375 (98M frames), together with S1: the MLP agent it builds on stalled at about
0.1 rescues per base game. Base-game evaluation at update 250: 0.16 / 0.06 rescues per game (sampled
/ greedy), 8.1 divers, score 427. So the currents did not hurt base-game learning early on. The same
currents are used on top of the stronger object-attention agent in S4
(`experiment_seaquest_attention_currents.py`). Not evaluated on `gravity`.
