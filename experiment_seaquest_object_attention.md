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

- First run `sq_att_a` (stopped at about update 550, 144M frames; run directory removed): **input bug**.
  The objects' orientation was fed raw (0 / 90 / 270 degrees) next to features in [-1, 1], and the
  oxygen-trend feature reached -16 at resets. Base-game evaluations at updates 250 / 500: 0 rescues,
  6.5-6.9 divers, score 70-220 per game, behind S1's MLP at the same number of frames. Fixed in the
  code: orientation is encoded as (sin, cos) and the oxygen trend is clipped to [-1, 1]. (S1's MLP
  scaling had the same orientation issue; `seaquest_common.obs_features_v2` fixes it for new
  experiments.)
- The fixed agent is used in S5 (`experiment_seaquest_diver_curriculum.py --agent attention`), which
  adds a start-state curriculum. S3 without the curriculum was not rerun yet.
