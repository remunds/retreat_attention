"""Pong environment helpers: env construction (incl. mods) and object-state extraction."""

import jax.numpy as jnp
import jaxatari

OBJECTS = ("player", "enemy", "ball")
PLAYER, ENEMY, BALL = range(len(OBJECTS))
NUM_OBJECTS = len(OBJECTS)
NUM_ACTIONS = 6  # NOOP, FIRE, RIGHT(up), LEFT(down), RIGHTFIRE, LEFTFIRE
POS_DIM = 2  # (x, y) per object

# Pong geometry, mirrored from jaxatari.games.jax_pong.PongConstants.
SCREEN_W, SCREEN_H = 160.0, 210.0
PLAYER_X = 140
BALL_START_X = 78
PADDLE_H = 16
BALL_H = 4

# The intervention for the success criterion (JAXAtari's own mod): the enemy only tracks the ball
# while the ball moves toward it. Training uses the unmodified game.
EVAL_MOD = "lazy_enemy"


def make_env(mods=None):
    """Create Pong, optionally with JAXAtari mods, e.g. ["lazy_enemy"]."""
    return jaxatari.make("pong", mods=list(mods) if mods else None)


def object_positions(obs) -> jnp.ndarray:
    """Object-centric observation as a [NUM_OBJECTS, 2] array of (x, y) pixel positions."""
    objs = [obs.player, obs.enemy, obs.ball]
    return jnp.stack(
        [jnp.stack([jnp.asarray(o.x, jnp.float32), jnp.asarray(o.y, jnp.float32)]) for o in objs]
    )
