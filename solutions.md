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

## Next steps

- Given the above, a purely suppressive distance term is structurally limited to reallocating away from things that are far, not toward things that are close, when the close candidate never developed a useful signal to begin with — with the explicit caveat from this session (don't over-bias, distance won't matter the same way in every game) staying in force, worth thinking about what would let the "correct" candidate develop a competitive raw score in the first place, rather than pushing the distance term harder.
- Re-run the `inspect_gate.py` court-position check after any further change before trusting any resulting numbers — standing rule.
- Stop annealing temperature once validation MSE stops improving (player's linear-head run got *worse* past its epoch-30 optimum as temperature kept dropping) — an early-stopping or MSE-monitoring criterion on the anneal schedule, rather than a fixed epoch-based one.
- The real test the confound-check above stands in for is the project's actual success criterion: **run the same court-position-style diagnostic under an actual intervention (e.g. `lazy_enemy`)** rather than only checking correlation with static position. A dependency that's genuinely about paddle proximity should degrade gracefully under `lazy_enemy`; a confounded one (like the current `ball → enemy` shortcut) should break, since `enemy_y` would no longer track the ball the same way. This is the first point where actually wiring up the `lazy_enemy` mod and re-evaluating would answer something we can't get from validation MSE alone.
- If neither of those closes the gap, that's real evidence for moving to approach A (hard/discrete gating, e.g. Gumbel-Softmax or straight-through top-k) — it's not obviously guaranteed to fix the "confident but wrong" issue either, but it's the more direct way to test whether the problem is soft-attention-specific or more fundamental to how the gate is supervised.
- Average the open-loop rollout evaluation over multiple start points/seeds for both models — the single-window baseline numbers from before are too noisy to trust individually, and the same applies to any future gated-model rollout comparison.
- (Explicitly deprioritized per discussion: the `step_counter % 8` enemy duty-cycle dependency — noted for context but not being pursued right now.)
- Once Pong's gating actually works, decide how to collect *useful* Seaquest training data (current random policy barely reaches enemies — see above) and how to generalize `train.py`/`train_gated.py` beyond Pong's hardcoded `OBJECT_DIMS` before running any gating experiment on it.
- Decide how to handle Seaquest's variable-cardinality enemies for real, rather than the single-nearest-threat placeholder — e.g. a set/slot-attention style mechanism, vs. accepting the placeholder as a permanent simplification with known blind spots (it can only ever depend on one enemy at a time, even when several are relevant).
