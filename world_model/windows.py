"""Turn a collected rollout into supervised pairs, per object.

`build_dataset` is the plain baseline's (history window -> next state) pairs.
`build_joint_dataset` extends that with every OTHER object's current-step
state, for approach B's cross-object attention gate.
"""

import numpy as np


def _valid_timesteps(episode_ids: np.ndarray, window: int, num_steps: int):
    for t in range(window - 1, num_steps):
        # episode_ids is non-decreasing, so comparing the endpoints of the
        # span is enough to detect an episode reset anywhere inside it.
        span = episode_ids[t - window + 1 : t + 2]
        if span[0] == span[-1]:
            yield t


def build_dataset(data: dict, object_name: str, window: int, include_action: bool = False):
    states = data[object_name]  # (T+1, dim)
    episode_ids = data["episode_ids"]  # (T+1,)
    actions = data.get("actions")
    num_steps = states.shape[0] - 1

    xs, ys = [], []
    for t in _valid_timesteps(episode_ids, window, num_steps):
        x = states[t - window + 1 : t + 1].reshape(-1)
        if include_action:
            x = np.concatenate([x, [actions[t]]])
        xs.append(x)
        ys.append(states[t + 1])

    return np.stack(xs), np.stack(ys)


def build_joint_dataset(
    data: dict,
    target_name: str,
    window: int,
    object_names,
    include_action: bool = False,
    distance_fn=None,
):
    """Same (own history -> next state) pairs as `build_dataset`, plus each
    OTHER object's current-step state at the same timestep, so a gate can
    decide per-example how much the target should depend on each of them.

    If `distance_fn` is given (signature `(target_name, target_row, other_name,
    other_row) -> float`), each other-object also gets a `"<name>_distance"`
    entry: its current distance to the target. This is kept as its own array
    rather than appended into the candidate's feature vector — appending it as
    an opaque extra input feature (an earlier version of this) let a
    confounded gate exploit distance as "yet another correlated number"
    instead of learning to attend less at long range, so it made things worse
    (see solutions.md). Kept separate, it can only be used as a structural,
    monotonic bias (see `gated_model.gate_forward`'s `distances` argument).

    Returns a dict: "own" (own flattened window, no action mixed in), "y"
    (target), "t" (the raw timestep each row's window ends at, i.e. row i
    predicts `target_states[t[i] + 1]` — lets a row be cross-referenced
    against a raw-trajectory event like a detected bounce), one entry per
    other-object name (its current-step state) [+ "<name>_distance" if
    `distance_fn` given], and, if `include_action`, an "action" entry (raw
    action id, one per example) — kept separate from "own" so it can be
    exposed as its own attention token rather than baked into the
    predictor's input.
    """
    episode_ids = data["episode_ids"]
    actions = data.get("actions")
    target_states = data[target_name]
    num_steps = target_states.shape[0] - 1
    other_names = [name for name in object_names if name != target_name]

    own_xs, ys, acts, ts = [], [], [], []
    other_xs = {name: [] for name in other_names}
    other_dists = {name: [] for name in other_names}
    for t in _valid_timesteps(episode_ids, window, num_steps):
        target_row = target_states[t]
        own_xs.append(target_states[t - window + 1 : t + 1].reshape(-1))
        ys.append(target_states[t + 1])
        ts.append(t)
        for name in other_names:
            other_row = data[name][t]
            other_xs[name].append(other_row)
            if distance_fn is not None:
                other_dists[name].append(distance_fn(target_name, target_row, name, other_row))
        if include_action:
            acts.append(actions[t])

    result = {"own": np.stack(own_xs), "y": np.stack(ys), "t": np.array(ts, dtype=np.int64)}
    for name in other_names:
        result[name] = np.stack(other_xs[name])
        if distance_fn is not None:
            result[f"{name}_distance"] = np.array(other_dists[name], dtype=np.float32)
    if include_action:
        result["action"] = np.array(acts, dtype=np.int32)
    return result


def train_val_split(x, y, val_fraction: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    num_val = int(len(x) * val_fraction)
    val_idx, train_idx = idx[:num_val], idx[num_val:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx])


def split_indices(n: int, val_fraction: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    num_val = int(n * val_fraction)
    return idx[num_val:], idx[:num_val]
