# Experiment S5: start-state curriculum on the divers carried (with random currents)

Code: `experiment_seaquest_diver_curriculum.py` (PPO loop from S1, agents from S1/S3, currents from
S2); held-out evaluation with `evaluate_gravity.py`.

## Summary

**Goal met.** The object-attention agent trained with this curriculum (and random currents) on the
base game makes 8.38 rescues with 6 divers per game under the held-out `gravity` mod (all 32 games
with at least 2); without currents it makes 7.22. The agents so far picked up 6-9 divers per game but almost never held 6 at once, so they rarely saw
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

Base-game progress during training (unmodified base game from its normal start, 32 games per
evaluation, sampled / greedy actions):

| run | update (frames) | rescues per game | divers per game | score |
|---|---|---|---|---|
| `sq_dc_att` | 250 (66M) | 4.53 / 4.84 | 33.0 / 34.3 | 11634 / 13603 |
| `sq_dc_att` | 500 (131M) | 7.50 / 9.22 | 52.1 / 63.3 | 26729 / 38196 |
| `sq_dc_att` | 750 (197M) | 11.00 / 12.34 | 72.3 / 81.5 | 49024 / 60612 |
| `sq_dc_att` | 1000 (262M) | 12.50 / **18.00** | 81.6 / 115.8 | 59796 / 102202 |
| `sq_dc_mlp` | 250 (66M) | 0.28 / 0.44 | 9.9 / 10.1 | 693 / 992 |
| `sq_dc_mlp` | 500 (131M) | 1.03 / 1.19 | 12.3 / 13.2 | 1867 / 2120 |
| `sq_dc_att_nocur` | 250 (66M) | 2.62 / 2.44 | 21.3 / 21.8 | 5700 / 5133 |
| `sq_dc_att_nocur` | 500 (131M) | 6.28 / 7.12 | 44.9 / 49.3 | 21358 / 25294 |
| `sq_dc_att_nocur` | 750 (197M) | 7.31 / 10.25 | 51.1 / 69.8 | 27880 / 45534 |
| `sq_dc_att_nocur` | 1000 (262M) | 11.84 / 13.44 | 78.3 / 88.8 | 57308 / 68281 |

`sq_dc_att` was stopped after the evaluation at update 1000 (of the planned 4000). The base-game
rescue rate was already far above what the goal needs, and each evaluation took longer and longer as
games got longer. The decision used only base-game numbers and compute. The ablation `sq_dc_att_nocur` got the same
budget (stopped after update 1000). `sq_dc_mlp` was stopped after update 500 to free a GPU. Selected
checkpoints (base-game rule: most rescues, then score): `sq_dc_att` and `sq_dc_att_nocur` round 3
(update 1000), greedy; `sq_dc_mlp` round 1 (update 500), greedy.

**Held-out evaluation** (`evaluate_gravity.py`, 32 full games per environment with the same seeds
from `PRNGKey(12345)`, at most 27,000 agent steps per game), each selected checkpoint evaluated once:

| run | agent | base: rescues / game | base: games with >= 2 | **gravity: rescues / game** | gravity: games with >= 2 | gravity: divers / game | gravity: score |
|---|---|---|---|---|---|---|---|
| `sq_dc_att` | attention + curriculum + currents | 15.69 | 32 / 32 | **8.38** | **32 / 32** | 57.7 | 31472 |
| `sq_dc_att_nocur` | attention + curriculum, **no currents** | 13.97 | 32 / 32 | **7.22** | 31 / 32 | 53.8 | 24908 |
| `sq_dc_mlp` | MLP + curriculum + currents | 1.06 | 6 / 32 | 0.75 | 2 / 32 | 10.2 | 1274 |
| `sq_ppo_b` (S1, reference) | MLP, no curriculum, no currents | 0.09 | 0 / 32 | 0.00 | 0 / 32 | 5.7 | 267 |

**The goal is met**: the object-attention agent trained with the diver curriculum and random
currents, on the base game only, rescues 6 divers 8.38 times per game on average with the held-out
`gravity` mod, and at least twice in every one of the 32 games.

```
CUDA_VISIBLE_DEVICES=4 uv run evaluate_gravity.py --module experiment_seaquest_diver_curriculum --ckpt runs/sq_dc_att/best.pkl
CUDA_VISIBLE_DEVICES=4 uv run evaluate_gravity.py --module experiment_seaquest_diver_curriculum --ckpt runs/sq_dc_att_nocur/best.pkl
CUDA_VISIBLE_DEVICES=4 uv run evaluate_gravity.py --module experiment_seaquest_diver_curriculum --ckpt runs/sq_dc_mlp/best.pkl
CUDA_VISIBLE_DEVICES=4 uv run evaluate_gravity.py --module experiment_seaquest_ppo --ckpt runs/sq_ppo_b/best.pkl
```

### Interpretation

- The start-state curriculum is what makes the rescue behaviour learnable: without it (S1-S4) no
  agent exceeded about 0.2 rescues per base game. With it, the object-attention agent reaches
  4.5-4.8 after 66M frames and 18 after 262M.
- The object-attention agent learns far faster than the MLP (4.8 vs 0.4 rescues per game after 66M
  frames with the same curriculum), in line with S3's motivation.
- Gravity still costs performance (15.7 -> 8.4 rescues per game, shorter games: 6164 -> 3820 steps
  on average), but the agent keeps rescuing reliably.
- **The random currents are not what makes the agent robust.** Without them (`sq_dc_att_nocur`) the
  agent also meets the goal with `gravity` (7.22 rescues per game, 31 / 32 games with >= 2). Both
  agents lose about half their base-game rescues under gravity (-47% with currents, -48% without).
  In this single-seed comparison the currents add a little in absolute terms (8.38 vs 7.22, 32 vs
  31 games), but the robustness comes mainly from having a strong, reliable rescuer, i.e. from the
  curriculum and the object-centric agent. More seeds would be needed to show any effect of the
  currents themselves.
- Limitations: one seed per configuration; the S5 runs were stopped at a quarter of the planned
  budget (on base-game grounds). Rescues per game under gravity vary from game to game (see
  `runs/*/gravity_eval.json` for per-game numbers).
