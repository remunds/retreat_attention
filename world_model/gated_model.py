"""Approach B from solutions.md: independent per-object predictors, each with a
soft cross-attention gate over candidate inputs plus a learned "null" option,
trained with an entropy penalty pushing the gate toward sparse/near-discrete
per-step choices.

Candidates are the OTHER objects, and, for the player only, its own action —
exposed as its own token (one-hot embedded and given its own key/value
projection) rather than baked directly into the predictor's input, so the
gate can show how much the player's next state depends on its action versus
on the other objects at each step.

Gating is NOT symmetric: each target object has its own query/key/value
parameters, so e.g. the ball attending to the enemy paddle at a given step
does not imply the enemy paddle attends to the ball.

Earlier, feeding a candidate's distance-to-target in as an extra *opaque*
input feature (run through the same learned key/value projection as
everything else) made a confounded shortcut worse, not better - the network
had no reason to treat "far" as "attend less", so it didn't. `distance_bias`
below is a **structural, per-candidate learned scalar** that subtracts
`softplus(raw) * distance` directly from that candidate's score, so
increasing distance can only ever suppress attention on it, never inflate it
- the direction is architectural, not hoped-for. It's initialized strongly
negative (softplus(-4) ~= 0.018) so it starts almost inert and only grows if
training actually rewards it: distance may matter a lot for Pong's ball, and
much less (or not at all) for other objects or other games, so nothing here
should force it to matter everywhere.
"""

import jax
import jax.numpy as jnp

ATTN_DIM = 16
VALUE_DIM = 16
DIST_BIAS_INIT = -4.0


def init_gate_params(key, dim_target: int, candidate_dims: dict, distance_candidates=()):
    """candidate_dims: candidate name -> its input dim (object state dim, or
    num_actions for the "action" token). distance_candidates: names (subset of
    candidate_dims) that get a learned distance-bias scalar."""
    params = {"key_w": {}, "value_w": {}, "dist_bias_raw": {}}

    key, k_query, k_nkey, k_nval = jax.random.split(key, 4)
    params["query_w"] = jax.random.normal(k_query, (dim_target, ATTN_DIM)) * jnp.sqrt(2.0 / dim_target)
    params["null_key"] = jax.random.normal(k_nkey, (ATTN_DIM,)) * 0.1
    params["null_value"] = jax.random.normal(k_nval, (VALUE_DIM,)) * 0.1

    for name, dim_s in candidate_dims.items():
        key, k_key, k_val = jax.random.split(key, 3)
        params["key_w"][name] = jax.random.normal(k_key, (dim_s, ATTN_DIM)) * jnp.sqrt(2.0 / dim_s)
        params["value_w"][name] = jax.random.normal(k_val, (dim_s, VALUE_DIM)) * jnp.sqrt(2.0 / dim_s)

    for name in distance_candidates:
        params["dist_bias_raw"][name] = jnp.array(DIST_BIAS_INIT)

    return params


def gate_forward(params, own_current, candidates: dict, candidate_names, temperature: float = 1.0, distances=None):
    """own_current: (B, dim_target). candidates[name]: (B, dim_name). All normalized
    (the "action" candidate is one-hot, which needs no separate normalization).

    `distances`, if given: dict of candidate name -> (B,) raw (unnormalized)
    distance-to-target, for candidates in `params["dist_bias_raw"]`. Subtracted
    from that candidate's score after the query/key dot product (see module
    docstring) - a structural nudge, not a feature the network could ignore.

    `temperature` divides the raw attention scores before the softmax: lower
    values force a sharper (more discrete-looking) distribution regardless of
    what the query/key parameters have learned, which is what lets us anneal
    toward hard selection instead of relying purely on the entropy penalty.

    Returns context (B, VALUE_DIM), weights (B, len(candidate_names) + 1) — the
    last column being the learned "null"/no-dependency option — and the raw
    (pre-softmax, pre-temperature) scores for diagnostics.
    """
    batch = own_current.shape[0]
    query = own_current @ params["query_w"]  # (B, ATTN_DIM)

    keys = [candidates[name] @ params["key_w"][name] for name in candidate_names]
    keys.append(jnp.broadcast_to(params["null_key"], (batch, ATTN_DIM)))
    keys = jnp.stack(keys, axis=1)  # (B, K, ATTN_DIM)

    values = [candidates[name] @ params["value_w"][name] for name in candidate_names]
    values.append(jnp.broadcast_to(params["null_value"], (batch, VALUE_DIM)))
    values = jnp.stack(values, axis=1)  # (B, K, VALUE_DIM)

    scores = jnp.einsum("bd,bkd->bk", query, keys) / jnp.sqrt(ATTN_DIM)

    if distances is not None:
        dist_bias_cols = []
        for name in candidate_names:
            if name in params["dist_bias_raw"]:
                w = jax.nn.softplus(params["dist_bias_raw"][name])
                dist_bias_cols.append(w * distances[name])
            else:
                dist_bias_cols.append(jnp.zeros(batch))
        dist_bias_cols.append(jnp.zeros(batch))  # null candidate: no distance to bias by
        scores = scores - jnp.stack(dist_bias_cols, axis=1)

    weights = jax.nn.softmax(scores / temperature, axis=-1)  # (B, K)
    context = jnp.einsum("bk,bkd->bd", weights, values)  # (B, VALUE_DIM)
    return context, weights, scores


def attention_entropy(weights, eps: float = 1e-8):
    return -jnp.sum(weights * jnp.log(weights + eps), axis=-1)
