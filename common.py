"""Shared, stable code for the object-centric Pong world-model experiments.

Contains:
- construction of the standard JAXAtari object-centric (OC) Pong pipeline
  (`AtariWrapper` -> `ObjectCentricWrapper`, frame skip 4, frame stack 4, sticky actions 0.25),
  optionally with game modifications (only ever `lazy_enemy` for *evaluation*),
- conversion of flattened OC frames to per-object positions,
- real-environment data collection for world-model training (unmodified Pong only),
- full-game evaluation of a policy (player / enemy points per game).

Object order everywhere is (player, enemy, ball); each object is described by (x, y) in pixels.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import jaxatari
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper

FRAME_STACK = 4
FRAME_SKIP = 4
NUM_ACTIONS = 6
NUM_OBJECTS = 3
OBJECT_NAMES = ("player", "enemy", "ball")
# Indices of (x, y) of player, enemy and ball in a flattened single OC frame (26 features:
# 3 objects x [x, y, w, h, active, visual_id, state, orientation] + 2 scores).
POS_IDX = np.array([[0, 1], [8, 9], [16, 17]])
# Scale used to normalise pixel positions for network inputs.
POS_SCALE = jnp.array([160.0, 210.0])


def make_env(mods=None):
    """Standard OC Pong pipeline. `mods=["lazy_enemy"]` must only be used for evaluation."""
    base = jaxatari.make("pong", mods=mods) if mods else jaxatari.make("pong")
    return ObjectCentricWrapper(AtariWrapper(base), frame_stack_size=FRAME_STACK, frame_skip=FRAME_SKIP)


def frames_to_pos(frames):
    """(..., 26) flattened OC frames -> (..., 3, 2) object positions (x, y) in pixels."""
    return frames[..., POS_IDX]


def policy_features(pos_hist):
    """Policy input from the last FRAME_STACK frames of object positions.

    pos_hist: (..., FRAME_STACK, 3, 2) pixels -> (..., F) normalised positions and velocities.
    Used identically in imagination and in the real environment.
    """
    p = pos_hist / POS_SCALE
    v = (pos_hist[..., 1:, :, :] - pos_hist[..., :-1, :, :]) / 8.0
    lead = p.shape[:-3]
    return jnp.concatenate([p.reshape(*lead, -1), v.reshape(*lead, -1)], axis=-1)


POLICY_FEATURE_DIM = FRAME_STACK * NUM_OBJECTS * 2 + (FRAME_STACK - 1) * NUM_OBJECTS * 2


@partial(jax.jit, static_argnums=(0, 1, 2))
def collect(env, act_fn, num_steps, act_params, keys):
    """Run `act_fn(act_params, obs_stack, key) -> action` in `len(keys)` parallel envs.

    Returns per-step arrays of shape (num_envs, num_steps, ...):
      pos:    object positions of the latest frame *before* the action (3, 2)
      next:   object positions of the frame after the action (3, 2)
      action, reward (clipped, per agent step), done (env_done: episode over, `next` is a reset frame)
    Also returns the initial frame stack so full histories can be reconstructed.
    """

    def run_one(k_env):
        k_reset, k_run = jax.random.split(k_env)
        obs, state = env.reset(k_reset)

        def body(carry, k):
            obs, state = carry
            k_act, _ = jax.random.split(k)
            a = act_fn(act_params, obs, k_act)
            obs2, state2, r, term, trunc, info = env.step(state, a)
            out = dict(
                pos=frames_to_pos(obs[-1]),
                action=a,
                reward=r,
                done=jnp.logical_or(info["env_done"], trunc),
                next=frames_to_pos(obs2[-1]),
            )
            return (obs2, state2), out

        (_, _), traj = jax.lax.scan(body, (obs, state), jax.random.split(k_run, num_steps))
        return traj

    return jax.vmap(run_one)(keys)


@partial(jax.jit, static_argnums=(0, 1, 2))
def evaluate(env, act_fn, max_steps, act_params, keys):
    """Play full games (until one side reaches 21 or `max_steps` agent steps).

    Returns (player_points, enemy_points, finished) per game, counted from the unclipped
    environment reward.
    """

    def run_one(k):
        k_reset, k_run = jax.random.split(k)
        obs, state = env.reset(k_reset)

        def body(carry, k):
            obs, state, ps, es, over = carry
            a = act_fn(act_params, obs, k)
            obs2, state2, r, term, trunc, info = env.step(state, a)
            er = info["env_reward"]
            live = jnp.logical_not(over)
            ps = ps + jnp.where(live, jnp.maximum(er, 0), 0)
            es = es + jnp.where(live, jnp.maximum(-er, 0), 0)
            over = jnp.logical_or(over, info["env_done"])
            return (obs2, state2, ps, es, over), None

        init = (obs, state, jnp.array(0.0), jnp.array(0.0), jnp.array(False))
        (_, _, ps, es, over), _ = jax.lax.scan(body, init, jax.random.split(k_run, max_steps))
        return ps, es, over

    return jax.vmap(run_one)(keys)


def random_act(params, obs, key):
    return jax.random.randint(key, (), 0, NUM_ACTIONS)


class ReplayBuffer:
    """Stores real (unmodified Pong) trajectories as (num_envs, T) chunks and samples windows.

    A window of `hist` frames followed by `horizon` transitions is valid if no episode boundary
    lies inside it.
    """

    def __init__(self):
        self.chunks = []

    def add(self, traj):
        traj = jax.device_get(traj)
        # frames[t] = pos before action t; frames[T] = pos after last action.
        frames = np.concatenate([traj["pos"], traj["next"][:, -1:]], axis=1)
        self.chunks.append(
            dict(
                frames=frames.astype(np.float32),
                action=traj["action"].astype(np.int32),
                reward=traj["reward"].astype(np.float32),
                done=traj["done"].astype(bool),
            )
        )

    @property
    def num_transitions(self):
        return sum(c["action"].size for c in self.chunks)

    def arrays(self):
        """Concatenate all chunks along the env axis (chunks must share T)."""
        keys = ("frames", "action", "reward", "done")
        return {k: np.concatenate([c[k] for c in self.chunks], axis=0) for k in keys}

    def valid_starts(self, hist, horizon):
        """Indices (env, t) such that frames[t-hist+1 .. t+horizon] lie in one episode.

        t is the index of the last history frame; transitions t .. t+horizon-1 are predicted.
        """
        d = self.arrays()["done"]  # done[s]: transition s -> s+1 crosses an episode reset
        n_env, T = d.shape
        c = np.concatenate([np.zeros((n_env, 1), np.int32), np.cumsum(d, axis=1)], axis=1)
        ts = np.arange(hist - 1, T - horizon + 1)
        # no done may occur at s in [t-hist+1, t+horizon-1].
        lo = ts - hist + 1
        hi = ts + horizon  # exclusive
        bad = c[:, hi] - c[:, lo]
        env_idx, t_idx = np.nonzero(bad == 0)
        return env_idx.astype(np.int32), ts[t_idx].astype(np.int32)
