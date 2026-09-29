"""Experiment S2 (Seaquest): PPO on the base game with random "ocean currents" acting on the submarine.

Builds on S1 (`experiment_seaquest_ppo.py`: PPO with an MLP actor-critic on the object-centric state
and a task-shaped reward). S1 trains in the unmodified base game, where the submarine only ever
moves as commanded. It never has to counteract an external force, and at the surface (while oxygen
refills, when movement is blocked) its action does not matter. A policy trained like that has no
reason to be robust to a change of its own submarine's dynamics.

Change relative to S1 (the only one): during training, each game gets a random **current**, a
generic intervention on the controlled object's dynamics inside the base game:
- with probability `p_none` there is no current, otherwise a direction (dx, dy) is drawn uniformly
  from the 8 compass directions and a strength q ~ U(q_min, q_max);
- after every agent step, with probability q the submarine is displaced by 1 px along (dx, dy)
  (clamped to the playfield), except while it is exploding.
Nothing else about the game changes. Evaluation during training, and checkpoint selection, use the
plain base game without currents. The held-out `gravity` mod is never used: it is only evaluated
afterwards with `evaluate_gravity.py`. (A downward current is one of the eight directions. The
family was chosen as generic perturbations of the agent's own motion, not around the mod's rule.)
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

import experiment_seaquest_ppo as s1
import seaquest_common as sc

DIRS = jnp.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], jnp.int32)
X_BOUNDS, Y_BOUNDS = (21, 134), (46, 141)

# interfaces for evaluate_gravity.py and the dashboard (same agent as S1)
AC, ACT_SAMPLE, ACT_GREEDY = s1.AC, s1.ACT_SAMPLE, s1.ACT_GREEDY


@struct.dataclass
class CurrentState:
    inner: Any  # ObjectCentricState of the base game
    d: jnp.ndarray  # (2,) current direction (dx, dy)
    q: jnp.ndarray  # per-step probability of a 1 px displacement (0 = no current)
    key: jnp.ndarray

    @property
    def atari_state(self):  # so that seaquest_common.game_state() works unchanged
        return self.inner.atari_state


class CurrentWrapper:
    """Base Seaquest with a random current per game (see module docstring)."""

    def __init__(self, env, cfg):
        self._env, self.cfg = env, cfg

    def _sample(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        d = DIRS[jax.random.randint(k1, (), 0, 8)]
        q = jax.random.uniform(k2, (), minval=self.cfg.q_min, maxval=self.cfg.q_max)
        q = jnp.where(jax.random.uniform(k3) < self.cfg.p_none, 0.0, q)
        return d, q

    def reset(self, key):
        k_env, k_cur, k_next = jax.random.split(key, 3)
        obs, st = self._env.reset(k_env)
        d, q = self._sample(k_cur)
        return obs, CurrentState(st, d, q, k_next)

    def step(self, state, action):
        obs, st, r, term, trunc, info = self._env.step(state.inner, action)
        key, k_push, k_new = jax.random.split(state.key, 3)
        g = sc.game_state(st)
        push = (jax.random.uniform(k_push) < state.q) & (g.death_counter == 0) & ~info["env_done"]
        x = jnp.clip(g.player_x + push * state.d[0], *X_BOUNDS).astype(g.player_x.dtype)
        y = jnp.clip(g.player_y + push * state.d[1], *Y_BOUNDS).astype(g.player_y.dtype)
        g = g.replace(player_x=x, player_y=y)
        st = st.replace(atari_state=st.atari_state.replace(env_state=g))
        d, q = self._sample(k_new)  # a new game gets a new current
        d = jnp.where(info["env_done"], d, state.d)
        q = jnp.where(info["env_done"], q, state.q)
        return obs, CurrentState(st, d, q, key), r, term, trunc, info


def get_parser():
    ap = s1.get_parser()
    ap.set_defaults(name="seaquest_currents", life_penalty=0.0, surface_penalty=0.0, diver_bonus=2.0, ent_coef=0.02)
    ap.add_argument("--p_none", type=float, default=0.25, help="probability of a game without current")
    ap.add_argument("--q_min", type=float, default=0.0)
    ap.add_argument("--q_max", type=float, default=1.0)
    return ap


def main():
    cfg = get_parser().parse_args()
    s1.main(cfg, make_train_env=lambda c: CurrentWrapper(sc.make_env(), c),
            experiment="experiment_seaquest_currents")


if __name__ == "__main__":
    main()
