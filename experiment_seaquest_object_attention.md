# Experiment S3: object-centric attention agent with potential-based surfacing shaping

Code: `experiment_seaquest_object_attention.py` (PPO loop from `experiment_seaquest_ppo.py`),
held-out evaluation with `evaluate_gravity.py`.

## Summary

Replaces S1's flat MLP with an attention policy over per-object tokens (positions relative to the
submarine, velocities, types; inactive objects masked). It adds policy-invariant potential shaping
that rewards rising towards the surface when carrying 6 divers or when low on oxygen. Trained on the
base game only.

## Goal

An agent trained on the base Seaquest game only that, with the held-out `gravity` mod, collects 6
divers and surfaces with them at least twice per game (`successful_rescues` >= 2), on average over
at least 10 games.

## Research question

Does an object-centric attention agent, which sees every object relative to its own submarine and
shares weights across all slots of an object type, learn the rescue behaviour in the base game much
faster than an MLP on the flat observation? Is it robust to `gravity` without any perturbation during
training?

## Setup

- Tokens: player, 4 divers, 12 sharks, 12 enemy subs, surface sub, torpedo, 4 enemy missiles (type
  one-hot, position relative to the submarine and absolute, velocity from the last two frames, size,
  orientation, visual id, active flag) and one global token (oxygen, divers carried, lives, own
  position, oxygen trend). Inactive objects are masked.
- Network: linear embedding (d = 64), 2 pre-LayerNorm self-attention blocks (4 heads), readout of
  the global token, the player token and a masked mean into 256-unit policy and value heads.
- Reward: S1's shaped reward (diver bonus 2, rescue bonus 10, no surface or life penalty) +
  gamma * Phi(s') - Phi(s) with Phi = height * (2 * [6 divers] + 1 * [oxygen < 16]), where height is
  0 at the bottom and 1 at the surface, and Phi = 0 at terminals.
- PPO: 512 base games, 128 steps, 4000 updates, lr 3e-4, entropy 0.01, otherwise as S1.
- Selection on the base game (most rescues, then score). `gravity` only afterwards.

## Experiments

```
CUDA_VISIBLE_DEVICES=5 uv run experiment_seaquest_object_attention.py --name sq_att_a
```

## Results

Run in progress.
