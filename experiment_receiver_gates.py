"""Experiment 4: receiver-driven L0 gates in the world model + actor trained under enemy interventions.

Motivation: in experiments 1 and 2 the gate of a pair (i <- j) was computed from *both* objects'
states, g_ij = f(e_i, e_j). In unmodified Pong the enemy's height is always close to the ball's
(99 % of gaps < 19 px), so a counterfactual enemy far from the ball is out of distribution for
f and for the message. It opens the gate and corrupts the ball's prediction even in free flight
(leakage up to ~20 px, see `common.leakage` and `experiment_intervention_actor.md`). This breaks
imagination under enemy interventions and, at test time, would break under `lazy_enemy`.

Change relative to experiment 2 (everything else, including the enemy interventions during actor
training, is identical and reused from `experiment_intervention_actor.py`):
- **Receiver-driven gates:** whether object i listens to token j is decided from i's own state
  only: g_ij = HC(f_i(e_i))_j (hard-concrete / L0 gate). The *content* of the message still depends
  on both objects. When the gate is closed it is exactly 0, so i's prediction is then exactly
  invariant to anything j does, including behaviour never seen in training. For example, the ball
  can decide from its own position and velocity that it is about to reach a paddle, and only then
  read that paddle.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

import common
import experiment_intervention_actor as e2
import experiment_sparse_object_wm as e1

HC_BETA, HC_GAMMA, HC_ZETA = e2.HC_BETA, e2.HC_GAMMA, e2.HC_ZETA


class ReceiverGatedWM(nn.Module):
    d: int = 128

    @nn.compact
    def __call__(self, hist, action, train: bool = False):
        """Same interface as `e2.L0ObjectWM`: (delta, reward_logits, gates (B,3,4), l0 (B,3))."""
        B = hist.shape[0]
        p = hist / common.POS_SCALE
        v = (hist[:, 1:] - hist[:, :-1]) / 8.0
        per_obj = jnp.concatenate(
            [p.transpose(0, 2, 1, 3).reshape(B, 3, -1), v.transpose(0, 2, 1, 3).reshape(B, 3, -1)], -1
        )
        enc = [e1.MLP([self.d, self.d], name=f"enc_{n}")(per_obj[:, i]) for i, n in enumerate(common.OBJECT_NAMES)]
        e_act = nn.Embed(common.NUM_ACTIONS, self.d, name="act_embed")(action)
        tokens = jnp.stack(enc + [e_act], axis=1)

        deltas, gates, l0s = [], [], []
        reward_logits = None
        for i, n in enumerate(common.OBJECT_NAMES):
            others = [j for j in range(4) if j != i]
            e_i = tokens[:, i]
            e_j = tokens[:, others]
            # gate logits from the receiver's own state only: one logit per sender
            alpha = e1.MLP([self.d, len(others)], name=f"gate_{n}")(e_i)  # (B, 3)
            if train:
                u = jax.random.uniform(self.make_rng("gate"), alpha.shape, minval=1e-6, maxval=1 - 1e-6)
                s = nn.sigmoid((jnp.log(u) - jnp.log1p(-u) + alpha) / HC_BETA)
            else:
                s = nn.sigmoid(alpha)
            g = jnp.clip(s * (HC_ZETA - HC_GAMMA) + HC_GAMMA, 0.0, 1.0)
            l0s.append(nn.sigmoid(alpha - HC_BETA * jnp.log(-HC_GAMMA / HC_ZETA)).sum(-1))
            pair = jnp.concatenate([jnp.broadcast_to(e_i[:, None], e_j.shape), e_j], -1)
            m = nn.LayerNorm(name=f"msg_ln_{n}")(e1.MLP([self.d, self.d], name=f"msg_{n}")(pair))
            msg = jnp.sum(g[..., None] * m, axis=1)
            h = e1.MLP([2 * self.d, self.d], act_last=True, name=f"dec_{n}")(jnp.concatenate([e_i, msg], -1))
            deltas.append(nn.Dense(2, name=f"out_{n}")(h))
            if n == "ball":
                reward_logits = nn.Dense(3, name="reward")(h)
            gates.append(jnp.ones((B, 4)).at[:, jnp.array(others)].set(g))
        return jnp.stack(deltas, 1), reward_logits, jnp.stack(gates, 1), jnp.stack(l0s, 1)


def build_model(cfg):
    """World model from a config dict (used by evaluate_lazy_enemy.py)."""
    return ReceiverGatedWM(d=cfg["d"])


wm_step = e2.wm_step
ACT_SAMPLE, ACT_GREEDY = e1.ACT_SAMPLE, e1.ACT_GREEDY


def main():
    e2.run(e2.get_parser(default_name="receiver_gates").parse_args(), ReceiverGatedWM)


if __name__ == "__main__":
    main()
