# Experiment S4: object-attention agent trained with random currents

Code: `experiment_seaquest_attention_currents.py` (agent and reward from S3, currents from S2),
held-out evaluation with `evaluate_gravity.py`.

## Summary

S3's object-centric attention agent, trained in base Seaquest with S2's random per-game currents
(1 px per step in one of 8 directions, random strength) as a generic intervention on the agent's own
dynamics. Selection on the plain base game; the held-out `gravity` mod is only used for evaluation.

## Goal

An agent trained on the base Seaquest game only that, with the held-out `gravity` mod, collects 6
divers and surfaces with them at least twice per game (`successful_rescues` >= 2), on average over
at least 10 games.

## Research question

Do random perturbations of the submarine's own motion during training make the object-attention
agent robust to changed submarine dynamics at test time, compared with S3 trained without them?

## Setup

As S3, plus currents as in S2: per game, with probability 0.25 no current, otherwise a direction
uniformly from 8 compass directions and a strength q ~ U(0, 1) (probability of a 1 px displacement
after each agent step). A new game draws a new current.

## Experiments

```
CUDA_VISIBLE_DEVICES=4 uv run experiment_seaquest_attention_currents.py --name sq_attcur_a
```

## Results

- First run `sq_attcur_a` (stopped at about update 550, 144M frames; run directory removed): **input bug**.
  The objects' orientation was fed raw (0 / 90 / 270 degrees) next to features in [-1, 1], and the
  oxygen-trend feature reached -16 at resets. Base-game evaluations at updates 250 / 500: 0 rescues,
  6.5-6.9 divers, score 70-220 per game, behind S1's MLP at the same number of frames. Fixed in the
  code: orientation is encoded as (sin, cos) and the oxygen trend is clipped to [-1, 1]. (S1's MLP
  scaling had the same orientation issue; `seaquest_common.obs_features_v2` fixes it for new
  experiments.)
- The fixed agent is used in S5 (`experiment_seaquest_diver_curriculum.py --agent attention`), which
  adds a start-state curriculum. S4 without the curriculum was not rerun yet.
