"""Per-object state extraction for JAXAtari Pong.

Each object's state vector only includes the fields that belong to it;
`player_score`/`enemy_score`/`step_counter` are episode bookkeeping, not
object dynamics, so they are left out.
"""

OBJECT_DIMS = {"player": 2, "enemy": 2, "ball": 4}

# Paddles only move along y; their x is a fixed game constant (env.consts.PLAYER_X /
# ENEMY_X). Needed to compute real 2D distances between objects for the gate.
PLAYER_X = 140.0
ENEMY_X = 16.0


def player_state(state):
    return (float(state.player_y), float(state.player_speed))


def enemy_state(state):
    return (float(state.enemy_y), float(state.enemy_speed))


def ball_state(state):
    return (float(state.ball_x), float(state.ball_y), float(state.ball_vel_x), float(state.ball_vel_y))


STATE_FNS = {"player": player_state, "enemy": enemy_state, "ball": ball_state}


def object_xy(name, values):
    """(x, y) for an object given its own state feature vector (last axis = feature
    dim; works for a single (dim,) row or a batch (..., dim))."""
    if name == "ball":
        return values[..., 0], values[..., 1]
    if name == "player":
        return PLAYER_X, values[..., 0]
    if name == "enemy":
        return ENEMY_X, values[..., 0]
    raise ValueError(f"unknown object: {name}")


def distance(name_a, values_a, name_b, values_b):
    """Euclidean distance between two objects' current positions."""
    ax, ay = object_xy(name_a, values_a)
    bx, by = object_xy(name_b, values_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
