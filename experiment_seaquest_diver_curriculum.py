"""Experiment S5 (Seaquest): start-state curriculum on the number of divers carried.

S1-S4 learned to pick up divers (6-9 per game) but rarely rescued 6 of them. A rescue needs 6 divers
at once, while every surfacing with fewer costs one and every death costs one. So the agent almost
never reaches the states where the large rescue reward is paid, and the value of "keep the divers
and surface with 6" is learned very slowly.

Change relative to S4 (S3's object-attention agent and potential shaping; random currents with
`--currents 1`; `--agent mlp` uses S1's MLP on orientation-fixed features instead): a
**start-state curriculum in the base game**. When a game starts and after every
lost life, with probability `p_curriculum` the number of divers carried is set to k ~ U{0, ..., 6}.
The agent therefore regularly carries 4-6 divers and experiences the rescue (and how to keep divers
on the way to 6) early in training. The game's rules are unchanged; only which reachable states
training starts from changes (as in reverse / Go-Explore-style curricula). Evaluation during training
and checkpoint selection use the unmodified base game from its normal start. The held-out `gravity`
mod is never used.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

import experiment_seaquest_currents as s2
import experiment_seaquest_object_attention as s3
import experiment_seaquest_ppo as s1
import seaquest_common as sc

# ----- agents: S3's object-attention agent (default) or S1's MLP on v2 features (`--agent mlp`)
MLP = s1.ActorCritic()


def mlp_apply(params, obs):
    return MLP.apply(params, sc.obs_features_v2(obs))


def mlp_init(key):
    return MLP.init(key, jnp.zeros((1, sc.OBS_DIM)))


def _mlp_act(params, obs, key, greedy):
    logits, _ = mlp_apply(params, obs[None])
    return jnp.where(greedy, jnp.argmax(logits[0]), jax.random.categorical(key, logits[0]))


AGENTS = {"attention": s3.AGENT,
          "mlp": (mlp_apply, mlp_init, lambda p, o, k: _mlp_act(p, o, k, False), lambda p, o, k: _mlp_act(p, o, k, True))}


def _act(params, obs, key, greedy):
    """Dispatch on the parameter tree (the MLP has "a_0" layers), so one module serves both agents."""
    name = "mlp" if "a_0" in params["params"] else "attention"
    return AGENTS[name][3 if greedy else 2](params, obs, key)


def ACT_SAMPLE(params, obs, key):
    return _act(params, obs, key, False)


def ACT_GREEDY(params, obs, key):
    return _act(params, obs, key, True)


@struct.dataclass
class CurState:
    inner: Any
    key: jnp.ndarray

    @property
    def atari_state(self):
        return self.inner.atari_state


class DiverCurriculum:
    """Sets the divers carried to a random k at game start and after a lost life (training only)."""

    def __init__(self, env, cfg):
        self._env, self.cfg = env, cfg

    def _set(self, st, key, apply):
        k1, k2 = jax.random.split(key)
        use = apply & (jax.random.uniform(k1) < self.cfg.p_curriculum)
        k = jax.random.randint(k2, (), 0, 7)
        # the wrapped state may itself be a wrapper state (currents): find the ObjectCentricState
        oc = st.inner if hasattr(st, "inner") else st
        g = oc.atari_state.env_state
        g = g.replace(divers_collected=jnp.where(use, k, g.divers_collected).astype(g.divers_collected.dtype))
        oc = oc.replace(atari_state=oc.atari_state.replace(env_state=g))
        return st.replace(inner=oc) if hasattr(st, "inner") else oc

    def reset(self, key):
        k_env, k_cur, k_next = jax.random.split(key, 3)
        obs, st = self._env.reset(k_env)
        return obs, CurState(self._set(st, k_cur, jnp.array(True)), k_next)

    def step(self, state, action):
        g0 = sc.game_state(state.inner)
        obs, st, r, term, trunc, info = self._env.step(state.inner, action)
        g1 = sc.game_state(st)
        key, k = jax.random.split(state.key)
        # after a lost life (the new life starts at the surface) or at the start of a new game
        new_life = (g1.lives < g0.lives) | info["env_done"]
        return obs, CurState(self._set(st, k, new_life), key), r, term, trunc, info


def reward_fn(cfg, g0, g1, env_done):
    """S3's reward, without counting the curriculum's gifted divers as picked up.

    The gift is applied at the end of a step in which a life was lost; no diver can be picked up in
    such a step, so the pickup bonus is removed there.
    """
    r = s3.reward_fn(cfg, g0, g1, env_done)
    lost_life = g1.lives < g0.lives
    gifted = jnp.maximum(g1.divers_collected - g0.divers_collected, 0).astype(jnp.float32)
    return r - jnp.where(lost_life & ~env_done, cfg.diver_bonus * gifted, 0.0)


def get_parser():
    ap = s3.get_parser(name="seaquest_diver_curriculum")
    ap.add_argument("--agent", default="attention", choices=list(AGENTS))
    ap.add_argument("--p_curriculum", type=float, default=0.5)
    ap.add_argument("--currents", type=int, default=1, help="also apply S2's random currents")
    ap.add_argument("--p_none", type=float, default=0.25)
    ap.add_argument("--q_min", type=float, default=0.0)
    ap.add_argument("--q_max", type=float, default=1.0)
    return ap


def make_train_env(cfg):
    env = sc.make_env()
    if cfg.currents:
        env = s2.CurrentWrapper(env, cfg)
    return DiverCurriculum(env, cfg)


def main():
    cfg = get_parser().parse_args()
    s1.main(cfg, make_train_env=make_train_env, experiment="experiment_seaquest_diver_curriculum",
            agent=AGENTS[cfg.agent], reward_fn=reward_fn)


if __name__ == "__main__":
    main()
