"""Per-object state extraction for JAXAtari Pong.

Each object's state vector only includes the fields that belong to it;
`player_score`/`enemy_score`/`step_counter` are episode bookkeeping, not
object dynamics, so they are left out.
"""

OBJECT_DIMS = {"player": 2, "enemy": 2, "ball": 4}


def player_state(state):
    return (float(state.player_y), float(state.player_speed))


def enemy_state(state):
    return (float(state.enemy_y), float(state.enemy_speed))


def ball_state(state):
    return (float(state.ball_x), float(state.ball_y), float(state.ball_vel_x), float(state.ball_vel_y))


STATE_FNS = {"player": player_state, "enemy": enemy_state, "ball": ball_state}
