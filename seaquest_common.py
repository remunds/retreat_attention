"""Shared, stable code for the Seaquest experiments.

- `make_env(mods)`: the standard JAXAtari object-centric pipeline (`AtariWrapper` -> `ObjectCentricWrapper`,
  frame skip 4, frame stack 4, sticky actions 0.25). `mods=["gravity"]` must only be used for
  *evaluation* (see AGENTS.md).
- `obs_features(obs)`: fixed scaling of the flattened object-centric observation (284 features per
  frame: player, 4 divers, 25 enemies, 5 projectiles, oxygen, score, lives, collected divers).
- `evaluate(...)`: full games; per game the number of successful rescues (surfacing with 6 divers),
  divers collected, score, lives lost and length.
- `game_state(st)`: the base game's `SeaquestState` inside the wrapper state.

Layout of one flattened frame (ravel of the observation dataclass, field by field):
player [0:8] = x, y, w, h, active, visual_id, state, orientation; divers [8:40] (8 fields x 4);
enemies [40:240] (8 fields x 25: 12 sharks, 12 subs, 1 surface sub); projectiles [240:280]
(8 fields x 5: player torpedo, 4 enemy missiles); oxygen 280, score 281, lives 282, collected divers 283.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import jaxatari
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper

FRAME_STACK = 4
FRAME_SKIP = 4
NUM_ACTIONS = 18
FRAME_DIM = 284
PLAYER_Y_IDX = 1
OXYGEN_IDX, SCORE_IDX, LIVES_IDX, DIVERS_IDX = 280, 281, 282, 283


def make_env(mods=None):
    """Standard OC Seaquest pipeline. `mods=["gravity"]` is for evaluation only."""
    base = jaxatari.make("seaquest", mods=mods) if mods else jaxatari.make("seaquest")
    return ObjectCentricWrapper(AtariWrapper(base), frame_stack_size=FRAME_STACK, frame_skip=FRAME_SKIP)


def _frame_scale():
    """Per-feature scale so that all inputs are roughly in [-1, 1]."""
    s = np.ones(FRAME_DIM, np.float32)

    def group(start, n):  # 8 fields x n objects, field-major
        s[start + 0 * n:start + 1 * n] = 160.0  # x
        s[start + 1 * n:start + 2 * n] = 210.0  # y
        s[start + 2 * n:start + 3 * n] = 16.0  # width
        s[start + 3 * n:start + 4 * n] = 16.0  # height
        s[start + 4 * n:start + 5 * n] = 1.0  # active
        s[start + 5 * n:start + 6 * n] = 5.0  # visual id
        s[start + 6 * n:start + 7 * n] = 1.0  # state
        s[start + 7 * n:start + 8 * n] = 1.0  # orientation

    group(0, 1)
    group(8, 4)
    group(40, 25)
    group(240, 5)
    s[OXYGEN_IDX], s[SCORE_IDX], s[LIVES_IDX], s[DIVERS_IDX] = 64.0, 1e4, 4.0, 6.0
    return jnp.asarray(s)


FRAME_SCALE = _frame_scale()


def obs_features(obs):
    """(..., FRAME_STACK, 284) stacked OC frames -> (..., F) scaled features (score dropped)."""
    x = obs / FRAME_SCALE
    x = x.at[..., SCORE_IDX].set(0.0)  # the score is not needed to act
    return x.reshape(*x.shape[:-2], -1)


OBS_DIM = FRAME_STACK * FRAME_DIM


def game_state(st):
    """SeaquestState inside the ObjectCentricWrapper state."""
    return st.atari_state.env_state


@partial(jax.jit, static_argnums=(0, 1, 2))
def evaluate(env, act_fn, max_steps, act_params, keys):
    """Play full games (until game over or `max_steps` agent steps).

    Returns a dict of per-game arrays: rescues (successful rescues = surfacing with 6 divers),
    divers (divers picked up), score, lives_lost, steps, finished.
    """

    def run_one(k):
        k_reset, k_run = jax.random.split(k)
        obs, st = env.reset(k_reset)
        step_keys = jax.random.split(k_run, max_steps)

        def cond(c):
            return jnp.logical_and(c[0] < max_steps, jnp.logical_not(c[-1]))

        def body(c):
            t, obs, st, divers, lives_lost, over = c
            g0 = game_state(st)
            a = act_fn(act_params, obs, step_keys[t])
            obs2, st2, r, term, trunc, info = env.step(st, a)
            g1 = game_state(st2)
            live = jnp.logical_not(over)
            picked = jnp.maximum(g1.divers_collected - g0.divers_collected, 0)
            divers = divers + jnp.where(live & ~info["env_done"], picked, 0)
            lives_lost = lives_lost + jnp.where(live & ~info["env_done"] & (g1.lives < g0.lives), 1, 0)
            return t + 1, obs2, st2, divers, lives_lost, jnp.logical_or(over, info["env_done"])

        # the final score / rescues are read before the automatic reset, so track them in the loop
        def body_tracked(c):
            (t, obs, st, divers, lives_lost, over), (score, rescues) = c
            g = game_state(st)
            score = jnp.where(over, score, g.score)
            rescues = jnp.where(over, rescues, g.successful_rescues)
            return body((t, obs, st, divers, lives_lost, over)), (score, rescues)

        init = ((jnp.array(0), obs, st, jnp.array(0), jnp.array(0), jnp.array(False)), (jnp.array(0), jnp.array(0)))
        (t, _, st_end, divers, lives_lost, over), (score, rescues) = jax.lax.while_loop(
            lambda c: cond(c[0]), body_tracked, init)
        g = game_state(st_end)
        # if the game did not end, the last state holds the final numbers
        score = jnp.where(over, score, g.score)
        rescues = jnp.where(over, rescues, g.successful_rescues)
        return dict(rescues=rescues, divers=divers, score=score, lives_lost=lives_lost, steps=t, finished=over)

    return jax.vmap(run_one)(keys)
