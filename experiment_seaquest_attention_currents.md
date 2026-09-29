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

Run in progress.
