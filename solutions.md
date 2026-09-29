# Candidate Solutions

Working notes on possible architectures for the per-object world model described in `project.md`. Goal: predict each object's (player, enemy, ball) next state independently, with a dependency structure on other objects that is decided fresh at every timestep and can be inspected afterward.

## Core requirements an architecture must satisfy

1. **Per-object predictor.** Each object gets its own next-state prediction, computed from information specific to that object's needs — not a shared decoder over a joint scene embedding.
2. **Dynamic, discrete dependencies.** At every timestep, for every object, we need a decision of *which other objects (if any) it attends to*. This should be inspectable as a simple per-step, per-object-pair yes/no (or weight), not buried in a dense soft-attention matrix that "sort of" depends on everything a little.
3. **History use.** Each object's predictor conditions on its own past states, plus the past states of whatever it currently depends on.
4. **Robustness under intervention.** Changing one object's dynamics (e.g. `lazy_enemy`) at inference time should only degrade predictions that genuinely depend on that object.

## Candidate approach A: Slot predictors + hard/sparse gating

- Each object (player, enemy, ball) has its own small predictor (MLP or GRU) over its own history.
- A separate lightweight gating module looks at the current states (and maybe short histories) of all objects and outputs, per target object, a **discrete or sparse selection** of which other objects to include this step — e.g. Gumbel-Softmax / straight-through top-k over the other objects, not plain softmax attention.
- Selected objects' (history) representations are fed into the target object's predictor as extra input (e.g. concatenation or cross-attention restricted to the selected set).
- Dependency visualization = just log the gate's discrete selections per step.
- Risk: gating module itself could become a "hidden" joint model if not constrained; may need a sparsity penalty or hard top-k cap to keep it honest.

## Candidate approach B: Cross-attention with sparsity-inducing penalty

- Standard per-object query attending over all other objects' keys/values (transformer-style), but train with an entropy/L1 penalty on attention weights to push them toward near-one-hot per step.
- Simpler to implement (no discrete sampling / straight-through gradients), but dependency structure is only ever approximately discrete — visualization needs a threshold to call something "attended" or not, which is less clean than approach A's hard decisions.
- Easier to get working first as a baseline; could motivate moving to hard gating (A) if soft attention doesn't stay sparse enough in practice.

## Candidate approach C: Rule/heuristic-gated experts

- Instead of learning the gate, hand-derive a small set of interaction conditions (e.g. "ball near paddle" by distance threshold) and switch between "independent" and "interacting" predictor branches per object based on that condition.
- Pro: dependencies are trivially inspectable and guaranteed sparse/discrete by construction.
- Con: defeats part of the point of the project (learning *when* dependencies matter, rather than hand-specifying it) — probably only useful as a sanity-check baseline for what "ideal" gating should look like, or as an ablation to compare a learned gate against.

## Decisions

- **Gating is not symmetric.** Each target object's gate is independent — if the ball attends to the enemy paddle at a given step, that does not imply the enemy paddle attends to the ball.
- **Training is one-step prediction on ground truth.** During training, each object's predictor conditions on ground-truth history (no rollout, no feeding back its own predictions). Multi-step rollout only happens at prediction/evaluation time, where predicted states get fed back in as history — this is where compounding error and robustness under intervention (e.g. `lazy_enemy`) actually get tested.

## Open questions

- How do we quantify "robustness" beyond the final points-scored criterion — e.g. per-object prediction error broken down by whether `lazy_enemy` is active, to directly check that only ball/enemy-adjacent predictions degrade?
- At rollout time, does the gate also condition on predicted (rather than ground-truth) history for the objects being considered as dependencies? If so, gating errors early in a rollout could compound alongside state-prediction errors.

## Progress

**Plain (no gating) baseline is running end-to-end** — see `world_model/`:

- `data.py` collects random-policy rollouts from JAXAtari Pong (`collect_rollout`), recording each object's own state fields only (player: `y, speed`; enemy: `y, speed`; ball: `x, y, vel_x, vel_y`) plus the action and an episode id per step.
- `windows.py` turns a rollout into per-object (history window → next state) supervised pairs, dropping any window/target span that crosses an episode reset.
- `model.py` / `train.py` train one independent 2-layer MLP per object via one-step prediction on ground-truth history (per the training decision above). The player's predictor also takes its own current action as input (exogenous control, not a cross-object dependency); enemy and ball see only their own history.
- `evaluate.py` runs an open-loop autoregressive rollout (feeding predictions back in as history, per the rollout decision above) and reports per-object MSE, split into first/second half of the horizon to show error compounding.
- `tests/test_world_model.py` is a fast smoke test over this whole pipeline.

**Findings from the first run** (20k random-policy steps, 50 epochs): one-step validation MSE for the player predictor kept dropping thoughout training (2.6 at epoch 50, still decreasing), while enemy and ball plateaued much higher (~9.5 and ~11 respectively) after ~20 epochs. This matches the hypothesis: enemy AI tracks the ball, and ball bounces depend on paddle position — an independent, history-only model structurally cannot close that gap for either. Under open-loop rollout (200 steps), error compounds sharply for all three objects (e.g. ball first-half MSE ~1234, ball is somewhat calmer in the second half here but the effect is noisy over a single 200-step window and should be re-checked over more rollouts/seeds) — confirming this plain baseline is the "non-robust" reference point the dependency-aware approaches (A/B) need to beat.

**Approach B (soft cross-attention + entropy penalty) is implemented** — `world_model/gated_model.py` + `world_model/train_gated.py`, built on `windows.build_joint_dataset` (adds each other object's *current-step* state alongside the target's own history window):

- Each target object has its own query projection (from its own current state) and its own key/value projections for each *other* object — gating is per-target, not shared/symmetric, per the earlier decision.
- Candidates for a target are the other two objects plus a learned **null** vector (key + value), so the gate can choose "no dependency" instead of being forced to always draw from something.
- `attention_entropy` is added to the MSE loss, scaled by `--sparsity-weight`, to push weights toward peaked/near-discrete rather than uniformly blended.
- `tests/test_gated_model.py` smoke-tests the gate's shape/weight invariants and one training epoch per object.

### Findings — approach B does not yet produce sparse, meaningful dependencies

Trained on the same 20k-step rollout, 50 epochs, `sparsity_weight=0.01`:

| object | baseline val MSE | gated val MSE | mean attention (other, other, null) |
|---|---|---|---|
| player | 2.61 (still falling) | 4.02 | enemy=0.40, ball=0.30, **null=0.29** |
| enemy | 9.53 | 9.42 | player=0.36, ball=0.22, **null=0.41** |
| ball | ~10.9–11.0 | 11.10 | player=0.43, enemy=0.21, **null=0.36** |

Two honest negative results:

1. **The player "null-dependency" sanity check fails.** Player structurally shouldn't depend on enemy/ball at all, so a working sparse gate should push its null weight to ~1. Instead it sits at ~0.29 — barely above the other two candidates — and the extra (irrelevant) attention context makes player prediction *worse* than the plain baseline (4.02 vs 2.61), not better.
2. **The entropy penalty barely moves attention even at 100x strength.** Re-running with `sparsity_weight=1.0` (vs 0.01) left the mean attention distribution and entropy almost unchanged for all three objects. The gate converges to a static, input-independent-looking blend rather than a per-step switch, and a much stronger penalty doesn't visibly break it out of that.
3. **Enemy/ball MSE barely improved** over the no-dependency baseline despite the gate having access to exactly the object it needs (ball for enemy, paddles for ball) — consistent with the gate not actually learning to route information sharply through the correct candidate.

### A real hidden dependency we hadn't modeled: enemy has a duty-cycle skip

While digging into why enemy MSE didn't drop further, `jaxatari/games/jax_pong.py::_enemy_step` shows:

```python
should_move = state.step_counter % 8 != 0
direction = jnp.sign(state.ball_y - state.enemy_y)
new_y = state.enemy_y + (direction * ENEMY_STEP_SIZE) if should_move else state.enemy_y
```

The enemy doesn't just track the ball — it also **freezes on every 8th step**, gated by `step_counter`, which we deliberately excluded from all object state as "episode bookkeeping." This is a genuine dependency, just not an *object* dependency — it's a hidden global-clock signal outside the player/enemy/ball framing in `project.md`. With `WINDOW=4` (less than the period-8 cycle) and no explicit access to step parity, some of enemy's residual error is close to irreducible for both the baseline and the gated model, and this affects both equally, so it doesn't change the *comparison* between them — but it's worth remembering before reading too much into either model's absolute enemy MSE.

## Diagnosing the entropy penalty's weak effect

Added two more direct levers to `train_gated.py`: an explicit softmax **temperature**, annealed geometrically from `--temperature-start` (2.0) to `--temperature-end` (0.1) over training (`gate_forward` now divides scores by it before the softmax), and a **separate, higher learning rate for the gate's parameters** (`--gate-lr`, via `optax.multi_transform`) in case the predictor MLP was simply out-competing the gate for gradient signal.

Re-ran all three objects, 50 epochs, same data:

| object | val MSE (ep 10 → 50) | entropy (ep 10 → 50) | mean attn @ ep50 |
|---|---|---|---|
| player | 16.1 → 3.6 | 0.881 → **0.235** | enemy=0.45, ball=0.18, null=0.37 |
| enemy | 10.8 → 9.7 (got *worse* after ep20) | 0.990 → **0.600** | player=0.39, ball=0.16, null=0.44 |
| ball | 12.9 → 10.9 | 1.027 → **0.615** | player=0.36, enemy=0.30, null=0.34 |

**Temperature annealing works as a sharpening mechanism** — entropy drops substantially for all three objects this time (vs. barely moving before), confirming the earlier problem really was that nothing was forcing the softmax to sharpen; the entropy penalty alone wasn't providing enough gradient pressure on its own.

**But this exposes a deeper, more important problem: sharpening is not the same as being *correct*.** Two symptoms:

- **Player** (which should learn "always null") instead becomes *confidently* split — a low population-level entropy with `enemy=0.45` mean weight means individual examples are picking `enemy` sharply and often, not hedging. That's a confident wrong dependency, and player's val MSE (3.6) is still worse than the plain baseline (2.6).
- **Enemy's val MSE got worse, not better, as temperature dropped** (8.98 at epoch 30 → 9.74 at epoch 50), even though it has real, valid access to the ball it needs. Its attention on `ball` actually *shrank* over training (0.24 → 0.16) rather than growing toward it.

So forcing a discrete choice (via temperature or, presumably, approach A's hard gating) doesn't by itself teach the gate the *right* choice — it just forces commitment to whatever the query/key geometry currently (weakly, mostly incorrectly) encodes, and appears to lock that in earlier rather than letting it keep improving under a softer regime. This looks like a premature-commitment problem: the query/key vectors haven't learned meaningful "is this dependency relevant right now" geometry before the annealing schedule forces near-one-hot decisions on them.

## Action as its own attention token

Previously the player's action was just concatenated as a raw scalar into its own-history input — invisible to the gate entirely. Reworked `gated_model.py`/`windows.build_joint_dataset`/`train_gated.py` so the action is instead one-hot embedded and given its own query/key/value projection, as a full candidate alongside enemy/ball/null (`init_gate_params` now takes an explicit `candidate_dims` dict rather than assuming candidates are always objects).

Re-ran the same 50-epoch, temperature-annealed setup:

| candidate (player's gate, ep50) | weight |
|---|---|
| enemy (spurious) | 0.31 |
| ball (spurious) | 0.06 |
| **action (real dependency)** | **0.23** |
| null | 0.40 |

Player's val MSE improved somewhat (3.61 → **2.95**, still a bit worse than the plain baseline's 2.61) — a cleaner one-hot action signal helps some. But this sharpens the earlier finding rather than resolving it: even with the *true* dependency directly available as a candidate, the gate still gives the spurious "enemy" candidate *more* weight (0.31) than the real "action" dependency (0.23), and this ratio stays fairly stable across epochs/temperatures rather than correcting itself.

The likely mechanism: the predictor MLP downstream of the gate can partially null out an irrelevant/wrong context contribution with its own weights, so getting the *right* candidate costs little MSE either way — only the entropy penalty pushes for sharpness, and it doesn't care *which* candidate becomes dominant, just that something does. Whatever the query/key parameters happen to prefer early (plausibly influenced by random init) can get amplified into a confident, incorrect choice with no strong corrective pressure. This is consistent with, and reinforces, the "sharpening ≠ correctness" finding from the temperature-annealing experiment above.

## Testing a linear predictor head

Added `model.init_linear_params`/`model.linear_forward` (no hidden layer) and a `--head {mlp,linear}` flag to `train_gated.py`, to directly test the hypothesis that the MLP head was quietly nulling out bad/irrelevant context, removing the gate's incentive to pick the *correct* candidate. Same 50-epoch, temperature-annealed setup, `--head linear`:

| object | val MSE trend (ep 10 → 50) | attention trend |
|---|---|---|
| player | 6.1 → **0.71** (ep30) → 1.46 (ep50, got *worse* after) | roughly flat/even blend across enemy/ball/action/null (~0.2–0.3 each), entropy stayed high (~1.2+) |
| enemy | 13.7 → 9.7 (roughly flat) | player~0.35, ball~0.2–0.28, no clear sharpening |
| ball | 27.6 → 21.5 → 18.1 → 15.8 → **13.4**, improving every checkpoint | attention on `enemy` climbed monotonically: 0.52 → 0.57 → 0.60 → 0.68 → 0.69, `null` shrank 0.25 → 0.11 |

Two genuinely new results:

1. **Player's true dynamics are close to linear**, and a linear head fits them far better than the MLP did (0.71 vs. baseline's 2.61) — makes sense given `_player_step`'s update rule (`new_speed = old_speed + (target_speed - old_speed) * 0.3`, `target_speed` a simple function of the action) is linear except for a couple of boundary clamps. But **performance got *worse* again as the temperature kept annealing past that optimum** (0.71 → 0.86 → 1.46) — forcing near-one-hot selection past the point where a soft blend was already working well actively hurt it. This is concrete evidence for a gentler/earlier-stopping anneal schedule, not just a weaker one.
2. **Ball's attention on `enemy` sharpening *and* MSE improving together, for the first time** — looked like exactly the result we'd been hoping for.

**But a quick diagnostic (`world_model/inspect_gate.py`, buckets attention by the ball's court position — enemy paddle sits near `x≈16`, player near `x≈140`) shows this "positive" result is spurious:**

```
Attention breakdown for 'ball' (n=19977 examples):
  near enemy  (x<40)       (n=  924): player=0.07, enemy=0.93, null=0.00
  mid-court (40<=x<=120)   (n=14567): player=0.28, enemy=0.59, null=0.13
  near player (x>120)      (n= 4486): player=0.00, enemy=0.95, null=0.05
```

Attention on `enemy` is *just as high near the player's side of the court as near the enemy's* — including a near-zero `player` weight exactly where the ball would need player-paddle info to predict a bounce. The gate isn't learning "attend to whichever paddle is nearby"; it's exploiting `enemy_y` as a proxy for something else useful about recent ball trajectory (plausible mechanism: the enemy AI already tracks the ball, so `enemy_y` is itself a lagged, smoothed function of recent ball position — informative for one-step MSE, but not the bounce-proximity dependency the project actually wants to recover, and not something that would degrade correctly if enemy's tracking rule changed, e.g. under `lazy_enemy`).

This is actually a clean, small-scale demonstration of exactly why `project.md`'s success criterion is a *robustness* test under intervention, not a prediction-accuracy benchmark: a confounded shortcut can lower validation MSE and even look like "sharp, improving, sensible-sounding" attention, while being the opposite of what the project needs. Low MSE and plausible-looking attention are not sufficient evidence of a correct dependency — this diagnostic (bucketing/behavior-conditioning attention by a known ground-truth-relevant variable) is a cheap, necessary check before trusting any gate's learned weights, and should be run on any future gating result before treating it as progress.

## A second environment: Seaquest, with a "no enemies" test set

To eventually check whether any dependency-gating approach generalizes beyond Pong, added JAXAtari's **Seaquest** as a second environment — `world_model/seaquest_objects.py` + `world_model/seaquest_data.py`, kept as parallel modules rather than touching the Pong pipeline.

**The "no enemies" test set uses JAXAtari's existing `disable_enemies` mod** (`jaxatari.make("seaquest", mods=["disable_enemies"])`), which zeroes `shark_positions`/`sub_positions`/`enemy_missile_positions` every step — directly analogous to Pong's `lazy_enemy`: train on the default distribution, evaluate under this intervention. Verified this holds throughout a full rollout, not just at reset (`tests/test_seaquest.py`).

**A real design problem, not present in Pong: Seaquest's enemies are variable-cardinality.** Up to 12 sharks and 12 enemy submarines can be active simultaneously — there's no single "the enemy" object the way Pong has one fixed paddle. As a first cut, `seaquest_objects.py` reduces this to a single **"threat" object**: the closest active shark-or-sub to the player, by Euclidean distance, each step. This is a real simplification (every other simultaneously-active enemy is thrown away), chosen only so the same player/enemy-shaped 2-object framing already built for Pong could, in principle, be pointed at this domain later without a redesign. **This is not resolved, just deferred**: a proper treatment would need a genuinely variable-cardinality (set-based) dependency mechanism, which is a bigger architectural question than anything tackled so far.

**A caveat worth knowing before training anything on this data**: under a uniform-random policy, enemies only start spawning after a long warm-up (fill oxygen to 64, start diving past the start position, then a further ~277-step spawn-timer countdown per lane) — in a 20k-step / 15-episode rollout, only **31% of steps** have an active threat at all. And removing enemies didn't meaningfully extend episode length (mean **1333 steps with enemies vs. 1177 without**) — most random-policy deaths aren't enemy contact, they're something else (plausibly oxygen mismanagement). So this random-policy data source is fairly sparse in exactly the signal we'd want a "does the player depend on the threat" gate to learn from; a smarter/heuristic data-collection policy (e.g. one that reliably dives and survives) would likely be needed before training anything meaningful here.

**Not yet done**: `train.py`/`train_gated.py` still hardcode Pong's `OBJECT_DIMS` import and aren't parameterized by environment, so none of the gating experiments above have been run on Seaquest data yet — this was scoped as "add the environment + test set," not "port the training pipeline." That's a natural next step once Pong's gate-correctness problem (see above) is actually resolved, since retesting a fixed approach on a second domain before that would just double the noise.

## Giving the gate an explicit distance feature — it got worse, not better

Implemented the top next-step above: `world_model/objects.py` now has `object_xy`/`distance` (using the paddles' fixed x-constants, `PLAYER_X=140`/`ENEMY_X=16`, confirmed from `env.consts`), and `windows.build_joint_dataset` takes an optional `distance_fn` that appends each candidate's current Euclidean distance to the target as an extra trailing feature. `train_gated.py`/`inspect_gate.py` now pass this by default — no other code needed to change, since candidate dims are already read from the data's shape rather than hardcoded.

Re-ran the exact same linear-head, 50-epoch, temperature-annealed setup with this feature added. Result: **the confound got more extreme, not fixed.**

```
Attention breakdown for 'ball' (n=19977 examples), WITH the distance feature:
  near enemy  (x<40)       (n=  924): player=0.00, enemy=1.00, null=0.00
  mid-court (40<=x<=120)   (n=14567): player=0.01, enemy=0.89, null=0.10
  near player (x>120)      (n= 4486): player=0.00, enemy=1.00, null=0.00
```

Attention on "enemy" is now ~1.00 in *every* bucket, including near the player's side — worse than before (was 0.93–0.95). Entropy also dropped further (0.62 → 0.12 at epoch 50) — the gate got *more* confident in the same wrong answer. Handing it `distance-to-enemy` as a raw input feature didn't teach it "attend to whichever candidate is closest" as a general rule; it just gave the existing `enemy_y`-as-ball-proxy shortcut an even richer feature to exploit (distance-to-enemy is itself correlated with recent ball trajectory, via the same confound, so it's *more* informative on average, not more *targeted*), and the optimization leaned on it harder instead of learning to condition on it.

**Why, most likely:** the architecture has no inductive bias tying "small distance" to "higher attention" — distance is just one more opaque number the key/value projections can use however they like. Nothing forces attention to actually *decrease monotonically* as distance grows; it's still fully at the mercy of whatever the loss landscape rewards, and evidently a strong-but-unconditional reliance on "enemy" is an easier optimum to find than the conditional, proximity-based rule we want.

## A learned, weak-by-default distance bias — closer, but still not fixed

Implemented the suggestion above, deliberately kept weak per discussion: `gated_model.gate_forward` now takes a `distances` dict and subtracts `softplus(dist_bias_raw[name]) * distance` from that candidate's score — `softplus` keeps the weight non-negative (so distance can only ever *suppress*, never inflate, a candidate's attention), and each candidate gets its **own learned scalar**, initialized strongly negative (`softplus(-4) ≈ 0.018`) so it starts almost inert and only grows if training rewards it. This is intentionally *not* a strong universal prior — Pong is only the first environment, and distance may not matter the same way (or at all) elsewhere. Distance is scaled by its own training-set std before reaching this term (`dist_scale` in the checkpoint), so "starts weak" is meaningful regardless of a game's raw coordinate units, not just Pong's ~200px court. This replaces the earlier attempt (feeding distance in as an opaque extra input feature, which made the confound worse) — see `tests/test_gated_model.py::test_distance_bias_is_weak_at_init_and_only_suppresses` for the two properties this needs to hold.

Re-ran the same setup. **The learned weights themselves look right**: for the ball predictor, `player`'s distance-bias weight grew substantially over training (0.036 → 0.356), while `enemy`'s shrank toward zero (0.006 → 0.002) — the model correctly learned that distance-to-player is a meaningful suppression signal and distance-to-enemy isn't. But the court-position check still shows the same confound:

```
Attention breakdown for 'ball', WITH the learned distance bias:
  near enemy  (x<40)       (n=  924): player=0.02, enemy=0.98, null=0.01
  mid-court (40<=x<=120)   (n=14567): player=0.00, enemy=0.90, null=0.10
  near player (x>120)      (n= 4486): player=0.00, enemy=0.93, null=0.07
```

`player` attention is still ~0.00–0.02 everywhere, including right next to the player's own paddle. **Why**: the bias term is purely subtractive — at its very best (distance = 0) it contributes nothing, so it can suppress a wrongly-dominant candidate but can never lift a correct one *above* one that's already scoring higher via the ordinary query/key term. Apparently "player" as a candidate for ball's predictor never had a competitive raw query-key score to begin with (plausibly because the strong, always-available `enemy_y` shortcut left little training pressure for the player key/value projection to develop a useful signal at all) — so even a real, correctly-learned distance-suppression rule on the *wrong* candidate can't hand attention back to the *right* one. A weak, purely-suppressive bias — which is what was asked for here, deliberately — isn't sufficient by itself to override an already-dominant wrong candidate; it can trim wrong attention at range, but can't independently promote a correct one that never got a foothold.

## Checking the data itself: does the training rollout even contain a player bounce?

Before pushing further on the gate, checked something more basic: does the random-policy training rollout actually contain enough player-ball bounce events to learn a proximity rule from at all? `world_model/bounce_stats.py` detects two kinds of events directly from the raw ball trajectory (no model involved) — a **bounce** (a sign flip in `ball_vel_x` with no position jump) and a **miss** (the ball resetting to `(BALL_START_X, BALL_START_Y)` after passing a paddle, i.e. a point scored against that side) — and attributes each to whichever paddle's fixed x the ball was nearest to beforehand.

On the same 20k-step rollout used for every gating experiment above:

```
player:   16 bounces,  113 misses (12.4% return rate, n=129)
enemy :   22 bounces,    6 misses (78.6% return rate, n=28)
```

**The player successfully returns the ball only 12.4% of the time it arrives — 16 clean examples in the entire dataset.** The enemy's scripted AI returns it 78.6% of the time. This is a real, structural data-scarcity problem, independent of anything about the gate architecture: there's barely a player-bounce signal in this data for anything to learn, sparse or otherwise.

It also explains the *quality* of what little signal exists. Binning each bounce's contact point (`dy = ball_y - paddle_y`, paddle half-height is 8px):

| dy bin | player (n=16) | enemy (n=22) |
|---|---|---|
| -8..-4 | 1 | 0 |
| -4..0 | 2 | 2 |
| 0..4 | 4 | **15** |
| 4..8 | 3 | 0 |
| 8..12 | 5 | 1 |
| 12..16 | 0 | 1 |
| 16..20 | 1 | 1 |

Enemy bounces cluster tightly at dy≈0 (15 of 22, since it actively tracks the ball, so contact is almost always dead-center) — a clean, consistent, easy-to-learn pattern. Player bounces are scattered almost uniformly from -8 to +17 (essentially uncorrelated with anything, since the random policy's paddle position at contact time is close to arbitrary) — even the 16 examples that exist don't share a learnable common structure. **This plausibly explains why "player" never developed a competitive raw query/key score in the ball predictor's gate** (see the distance-bias finding above): it's not just fewer examples, the examples are also individually much less informative.

Visualized alongside the gating results in the [Gating Diagnostics artifact](https://claude.ai/artifact/XyKZhMQoxKCaWZ1LcwPm4L) (return-rate stacked bar + dy-histogram comparison).

## An "intermediate agent" fixes the data problem — and reveals the confound just moves

Implemented the suggestion above: `world_model/policies.py` adds `track_ball_policy` (mirrors the enemy AI's own logic — move toward the ball's y — using the game's own action mapping, empirically verified: action 2 moves the paddle up, action 3 down) and `epsilon_track_ball_policy` (mostly tracks, `epsilon=0.2` uniform-random for exploration diversity, so the data isn't perfectly deterministic). `world_model/data.py` takes a `--policy` flag; default (`random`) is unchanged, so every prior result stays reproducible.

Collected a fresh 20k-step rollout with `epsilon_track_ball`. **The data problem is completely fixed**:

```
                     random policy              epsilon_track_ball policy
player:   16 bounces,  113 misses (12.4%)  ->   87 bounces,    1 misses (98.9%)
enemy :   22 bounces,    6 misses (78.6%)  ->   88 bounces,   20 misses (81.5%)
```

Only one episode occurred in the whole 20k steps this time (rallies now last long, instead of ending almost immediately) — 87 clean player bounces instead of 16, and the contact-point distribution tightened a lot (std 3.63 vs. 5.71 before), since the paddle is now usually near the ball instead of wherever chance left it.

**But retraining the gated linear-head model on this new data doesn't fix the gate — it moves the confound to the other paddle.** Ball's attention now goes overwhelmingly to **"player"** (0.88 at epoch 50, up from tiny before) instead of "enemy". Re-running the court-position check:

```
Attention breakdown for 'ball', trained on epsilon_track_ball data:
  near enemy  (x<40)       (n= 3621): player=0.95, enemy=0.02, null=0.03
  mid-court (40<=x<=120)   (n=13599): player=0.85, enemy=0.05, null=0.10
  near player (x>120)      (n= 2777): player=0.91, enemy=0.02, null=0.07
```

Still flat, still ~0.85–0.95 everywhere — including right next to the *enemy's* paddle, where "player" should now be the irrelevant one. **The underlying mechanism is the same as before, just relocated**: any paddle that consistently tracks the ball (previously only the enemy AI; now also the player, by construction of the new policy) makes its own y-position a globally-useful proxy for recent ball movement, and the gate keeps preferring "copy whichever paddle is a good ball-trajectory proxy" over "attend to whichever paddle is spatially near me right now" — regardless of which specific paddle currently has that property. Better data fixed the *data-scarcity* problem cleanly, but the *gate's preference for a global shortcut over a genuinely conditional rule* turns out to be a separate, deeper issue that persists across both dataset regimes.

One interesting side effect, worth noting rather than chasing further right now: the enemy predictor's attention on "ball" (0.47, entropy dropped to 0.047) is now plausibly *correct*, not a shortcut — the enemy AI's own logic genuinely is `direction = sign(ball_y - enemy_y)`, so "enemy depends on ball" is the real mechanism, not a confound. Distinguishing a genuinely-improved dependency from a relocated confound is exactly why the court-position (or, better, an actual intervention) check has to be run on *every* target, not just the one that looked wrong last time.

## An explicit prior toward null: L1 on the non-null attention mass

The entropy penalty only ever rewarded *sharpness*; it has no preference for *which* candidate the gate lands on, so a globally-useful-but-wrong shortcut was just as attractive to it as a correct, conditional dependency. Added `--null-prior-weight`: since attention weights always sum to 1, `sum(non-null weights) == 1 - null_weight`, so an L1 penalty on the non-null mass is exactly a direct, directional prior — "default to no dependency unless a real MSE improvement is worth paying for" — rather than "be peaked, on whatever." Unlike the distance bias, this doesn't presume *which* candidate should be suppressed; it just makes attending to *anything* cost something.

Tested at two strengths on the ball predictor, using the `epsilon_track_ball` data (where "player" was the relocated shortcut):

| `null_prior_weight` | val MSE @ ep50 | mean attn @ ep50 | entropy @ ep50 |
|---|---|---|---|
| 0 (baseline, no prior) | 3.27 | player=0.88, enemy=0.04, null=0.08 | 0.222 |
| 0.5 | 3.34 | player=0.83, enemy=0.05, null=0.12 | 0.289 |
| 5.0 | **3.14** | player=0.00, enemy=0.00, **null=1.00** | 0.008 |

At `0.5` the effect is real but mild (null mass 8% → 12%) — the MSE term (scale ~3-9) simply dominates a coefficient bounded by 1. At `5.0` the gate **collapses to essentially 100% null attention, and validation MSE doesn't get worse — it gets slightly *better***. That's a clean, independent confirmation that the "player"/"enemy" attention the gate had been relying on was providing little to no real predictive value beyond what the ball's own history already captures: forcing it away costs almost nothing, which is exactly what you'd expect if it really was shortcut correlation rather than a genuine, load-bearing dependency.

**This is informative, but not obviously a fix, and needed a caveat before treating it as progress.** A strong null-prior makes "attend to nothing, always" a very easy local optimum — L1 cost is paid every time *any* non-null candidate is used, even if that use is sparse and genuinely conditional (e.g. only right at the moment of an actual bounce, which is a small fraction of all steps). Collapsing to null everywhere is consistent with either "the confound was all there ever was" or "there was a little real signal too, and the prior swept it away along with the shortcut" — resolved below.

## Checking bounce-timed attention directly, not just static position

`world_model/inspect_bounce_attention.py` uses `bounce_stats.py`'s exact detected bounce timesteps (not the coarser "which side of the court" proxy `inspect_gate.py` uses) and compares mean attention at the row that predicts each real bounce against every other row. (`bounce_stats.analyze`'s events now carry the raw timestep `t`; `windows.build_joint_dataset` now returns a parallel `"t"` array so a row can be matched back to it — the bounce at raw index `t` is predicted by the row whose window ends at `t - 1`.)

Ran this across the same three checkpoints:

| `null_prior_weight` | player-bounce rows (n=87) | player elsewhere (n=19910) | enemy-bounce rows (n=88) | enemy elsewhere (n=19909) |
|---|---|---|---|---|
| 0 | player=0.92 | player=0.88 | player=0.96, enemy=0.02 | player=0.88, enemy=0.04 |
| 0.5 | player=0.93 | player=0.83 | player=0.92, enemy=0.03 | player=0.83, enemy=0.06 |
| 5.0 | null=1.00 | null=1.00 | enemy=0.05, null=0.95 | null=1.00 |

**This resolves the open question, and not in the direction that would have made the null-prior look risky.** At `0` and `0.5`, attention on "player" is *not* meaningfully higher right at real player-bounce moments than everywhere else (0.92 vs. 0.88; 0.93 vs. 0.83 — a few points, not the sharp spike a genuine conditional dependency should produce). It's *also* not lower at enemy-bounce moments, where it should be irrelevant (0.96, 0.92 — if anything higher). **There was no meaningful bounce-localized structure to protect in the first place** — the shortcut wasn't "mostly confound with a little real signal mixed in," it was, as far as this gate/data combination could find, just the global confound, full stop. So the `null_prior_weight=5.0` collapse isn't losing anything real; it's correctly finding that there was nothing real there to keep.

One faint, worth-noting exception: at `5.0`, enemy-bounce rows show `enemy=0.05` versus `0.00` everywhere else — the only nonzero deviation from total collapse anywhere in this sweep. Given enemy's own dynamics genuinely do depend on the ball (not the reverse relationship being tested here, but a hint in the same spirit), this could be a faint trace of real structure surviving even heavy suppression — or just noise from a small sample (n=88). Not enough to act on, but worth flagging rather than rounding off to "nothing."

## Approach A: hard/discrete gating via Gumbel-Softmax

Implemented the recommendation above. `gated_model.gate_forward` gained `hard`/`key` arguments: a straight-through estimator (`weights = stop_gradient(onehot - soft) + soft` — **note the term order**, since `stop_gradient` doesn't change a forward value, only its gradient, and getting this backwards silently reduces the whole mechanism back to plain softmax; caught by a unit test asserting the output is *exactly* one-hot, not just close to it) forces every forward pass to commit to exactly one candidate (or null), with Gumbel noise added before the argmax during training (`key=<PRNGKey>`, resampled every step) for exploration, and no noise for a fully deterministic pick during inspection (`key=None`). `train_gated.py --gate-mode hard` wires this up, reusing the existing temperature schedule as the Gumbel-softmax temperature (anneal high→low, exactly standard practice) — no new hyperparameter needed.

Trained the ball predictor this way (linear head, `epsilon_track_ball` data, same 50 epochs). The printed "mean attn" numbers now mean something different and more literal than before: since every single example is an exact one-hot, they're the population **fraction of examples that discretely chose each candidate** (e.g. player=0.78 = "78% of examples picked 'player' as their one dependency"), not a soft blend. Checked both diagnostics:

```
Court position:
  near enemy  (x<40)  : player=0.82, enemy=0.17, null=0.01
  mid-court            : player=0.81, enemy=0.12, null=0.07
  near player (x>120)  : player=0.59, enemy=0.38, null=0.03

Exact bounce timing:
  player-bounce (n=87) : player=0.71, enemy=0.28   | elsewhere: player=0.78, enemy=0.16
  enemy-bounce  (n=88) : player=0.67, enemy=0.31   | elsewhere: player=0.78, enemy=0.16
```

**For the first time in this entire investigation, attention varies meaningfully with context instead of being flat everywhere** — court-position buckets differ by 20+ points, bounce-vs-elsewhere by 10-15 points. Every soft-attention variant (temperature annealing, distance bias, null-prior) produced numbers that were essentially the same regardless of ball position or timing; this doesn't.

**But the direction is mixed, not cleanly correct.** At real *enemy*-bounce moments, "enemy" choice frequency roughly doubles (16% → 31%) — plausible, in the right direction. At real *player*-bounce moments, "player" choice frequency goes the **wrong way**, dropping slightly (78% → 71%) while "enemy" rises (16% → 28%) — exactly backwards from what a genuine dependency should do. So hard gating broke the "identical everywhere" pattern that characterized every prior approach, which is real progress toward the kind of per-step-varying, inspectable dependency `project.md` asks for — but it hasn't (yet, at 50 epochs) landed on the *correct* rule for the player side specifically, only the enemy side. Plausibly because the enemy's true dependency (`direction = sign(ball_y - enemy_y)`) is a clean, deterministic rule to discover, while the player's bounce timing is entangled with the `epsilon_track_ball` policy's own stochastic exploration, making a clean discrete rule harder to find in the same number of epochs.

## Pushing hard gating further: 150 epochs

Re-ran the same three predictors for 150 epochs instead of 50 (same schedule, just stretched — temperature still anneals 2.0→0.1, just more gradually).

**Player and enemy both moved toward their plausibly-correct dependency with more training** — a genuinely encouraging sign:

| target | @ 50 epochs | @ 150 epochs |
|---|---|---|
| player | ball=0.90 (MSE 2.81) | action=0.53, null=0.26, ball=0.19 (MSE 0.67) |
| enemy | player=0.55 (MSE 2.07) | ball=0.46, player=0.11 (MSE 1.06) |

Player's real dependency is its own action (that's what actually moves the paddle); enemy's real dependency is the ball (`direction = sign(ball_y - enemy_y)`). Both predictors drifted *away* from an initially-dominant, plausibly-spurious candidate and *toward* the game-logic-correct one as training continued, with MSE dropping sharply alongside (2.81→0.67, 2.07→1.06) — real evidence that hard gating can, with enough time, find the right dependency instead of just any confident one.

**But ball's court-position/bounce-timing pattern got worse, not better, with more training — a sobering result that reframes the earlier "real progress" finding.**

```
Court position, 50 -> 150 epochs:
  near enemy  : player=0.82 -> 0.88
  mid-court   : player=0.81 -> 0.85
  near player : player=0.59 -> 0.82   (was the most-varied bucket; now nearly as high as the rest)

Bounce timing, 50 -> 150 epochs:
  player-bounce: player=0.71 -> 0.64 | elsewhere: 0.78 -> 0.85   (wrong-direction gap: -0.07 -> -0.21, THREE TIMES LARGER)
  enemy-bounce : enemy=0.31 -> 0.14  | elsewhere: 0.16 -> 0.07   (still ~2x at bounce vs elsewhere, ratio preserved)
```

More training didn't refine ball's court-position variation toward the correct localized rule — it **flattened it back out**, converging to a more confident, more globally-uniform reliance on "player" (0.82-0.88 everywhere, versus the more varied 0.59-0.88 spread at 50 epochs). And the *wrong-direction* dip at real player-bounces got substantially larger, not smaller. This suggests the earlier "context-varying attention" reading of the 50-epoch snapshot was likely an artifact of **incomplete convergence** rather than a sign of correctly-in-progress learning — letting it run longer reveals the actual endpoint is a strong, confident, *still-wrong* global habit, matching the concern already raised: Gumbel-softmax choices can get stuck once the temperature anneal narrows things down, amplifying whatever pattern (right or wrong) it locked onto early rather than self-correcting.

**Net read**: hard gating looks genuinely promising for player and enemy (both objects with a single, clean, deterministic true dependency), but not yet for ball, whose real dependency is switching between two candidates conditionally rather than settling on one fixed winner — exactly the harder case the whole "per-timestep dependency" idea was designed for, and exactly where it's still failing.

## Next steps
- **The player/enemy improvement suggests the mechanism itself works when there's one right answer to converge to; ball's regression suggests it doesn't yet handle "the right answer changes per step."** Worth trying an anneal schedule that doesn't force full commitment by a fixed epoch count — e.g. only sharpen once validation MSE plateaus, rather than on a fixed schedule, so the model has more room to keep exploring before committing early to whichever candidate happened to look best first.
- Re-run the full diagnostic suite (court-position, bounce-timing, return-rate) on player's and enemy's own final dependencies too, not just their printed mean-attention numbers, to confirm the "moved toward the correct candidate" reading holds up under the same scrutiny that caught ball's problem.
- **Actually wiring up `lazy_enemy` (or an analogous "lazy player" mod) is now the more informative test than ever, and complements the bounce-timed check rather than repeating it**: bounce-timing tells us whether a dependency is *conditionally localized*; intervention tells us whether it's *causally real* even if localized (enemy's "ball" dependency looks increasingly genuine by every measure so far, worth confirming it survives `lazy_enemy`) — a relocated shortcut should either fail the localization check (as ball's did) or break under intervention even if it passed localization.
- Re-run both the `inspect_gate.py` court-position check and `inspect_bounce_attention.py`'s exact-timestep check (plus the return-rate check) on every future variant before trusting any resulting numbers — standing rule, now with three independent confirmations of how necessary it is.
- Stop annealing temperature once validation MSE stops improving (player's linear-head run got *worse* past its epoch-30 optimum as temperature kept dropping) — an early-stopping or MSE-monitoring criterion on the anneal schedule, rather than a fixed epoch-based one.
- Average the open-loop rollout evaluation over multiple start points/seeds for both models — the single-window baseline numbers from before are too noisy to trust individually, and the same applies to any future gated-model rollout comparison.
- (Explicitly deprioritized per discussion: the `step_counter % 8` enemy duty-cycle dependency — noted for context but not being pursued right now.)
- Once Pong's gating actually works, decide how to collect *useful* Seaquest training data (current random policy barely reaches enemies — see above) and how to generalize `train.py`/`train_gated.py` beyond Pong's hardcoded `OBJECT_DIMS` before running any gating experiment on it.
- Decide how to handle Seaquest's variable-cardinality enemies for real, rather than the single-nearest-threat placeholder — e.g. a set/slot-attention style mechanism, vs. accepting the placeholder as a permanent simplification with known blind spots (it can only ever depend on one enemy at a time, even when several are relevant).
