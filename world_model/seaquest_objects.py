"""Per-object-ish state extraction for JAXAtari Seaquest.

Unlike Pong's single fixed enemy paddle, Seaquest's enemies are
variable-cardinality: up to 12 sharks and 12 enemy submarines can be active
at once (`state.shark_positions`/`state.sub_positions`, each `(12, 3)` arrays
of `(x, y, direction)`, with an all-zero row meaning "inactive"). There is no
single "the enemy" object the way there is in Pong.

As a first cut, this reduces that down to a single **"threat" object** per
step - the closest active shark or sub to the player - so the same
player/enemy-shaped dependency-gating machinery built for Pong could, in
principle, be pointed at this domain later without a redesign. This is a real
simplification (it throws away every other active enemy), not a full
multi-instance model - see solutions.md for the open design question of how
to represent a variable number of simultaneous objects.
"""

import numpy as np

OBJECT_DIMS = {"player": 3, "oxygen": 1, "threat": 3}  # (x, y, direction) / (level,) / (x, y, direction)


def player_state(state):
    return (float(state.player_x), float(state.player_y), float(state.player_direction))


def oxygen_state(state):
    return (float(state.oxygen),)


def _nearest_active(positions, player_x, player_y):
    """positions: (N, 3) array of (x, y, direction); an all-zero row is inactive."""
    positions = np.asarray(positions)
    active = positions[:, 2] != 0
    if not active.any():
        return None
    dx = positions[:, 0] - player_x
    dy = positions[:, 1] - player_y
    dist = np.where(active, dx * dx + dy * dy, np.inf)
    return positions[int(np.argmin(dist))]


def threat_state(state):
    player_x, player_y = float(state.player_x), float(state.player_y)
    candidates = [
        c for c in (
            _nearest_active(state.shark_positions, player_x, player_y),
            _nearest_active(state.sub_positions, player_x, player_y),
        )
        if c is not None
    ]
    if not candidates:
        return (0.0, 0.0, 0.0)
    dists = [(c[0] - player_x) ** 2 + (c[1] - player_y) ** 2 for c in candidates]
    nearest = candidates[int(np.argmin(dists))]
    return (float(nearest[0]), float(nearest[1]), float(nearest[2]))


STATE_FNS = {"player": player_state, "oxygen": oxygen_state, "threat": threat_state}
