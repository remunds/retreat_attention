"""Experiment 5: receiver-driven *soft* gates (L1) in the world model + actor trained under enemy interventions.

History:
- Exp. 1 (pair sigmoid gates, L1): state-dependent gates, but the ball used the enemy as a shortcut
  far from any bounce; the gate and message of a pair depend on both objects, so an enemy in an
  unseen position (as under `lazy_enemy`) corrupts the ball's prediction (leakage).
- Exp. 2 (pair hard-concrete / L0 gates + enemy interventions for the actor): the ball kept *all*
  gates open, so under the enemy interventions the imagined ball reacted to the enemy and the actor
  learned that false dependence (lazy_enemy final score +0.2).
- Exp. 4 (receiver-driven L0 gates): the L0 gates got stuck at constant values (the ball never
  listened to the player), a bad optimum.
- Ablation (scratch, documented in `experiment_receiver_soft_gates.md`): removing ball<-enemy
  messages entirely only raises the ball's 1-step error from 0.71 to 0.90 px, mostly near the
  enemy's paddle, so far from it the enemy is not needed.

This experiment keeps experiment 2's pipeline (enemy interventions during imagination training,
MLP actor, rounds, selection on unmodified Pong) and changes only the world-model gates:
- **Receiver-driven soft gates:** g_ij = sigmoid(f_i(e_i))_j depends only on the receiver's own
  state (as in exp. 4) but is a deterministic sigmoid trained with an L1 penalty (as in exp. 1),
  which learned state-dependent gates reliably. Messages are layer-normalised, so their size is
  bounded.
- `--gate_type ste`: binary gates (exactly 0 or 1 in the forward pass, sigmoid gradient in the
  backward pass, L1 penalty on the gate probability). With soft gates, a small but nonzero gate
  times large decoder weights still leaked the enemy into the ball (a WM-only sweep found
  ball<-enemy far-field leakage of ~12 px even at gate_coef 0.1); a binary gate either costs the
  full penalty or passes nothing.
- **Exact zeros at use time (soft gates):** when the model is used (imagination, evaluation) gates below
  `GATE_THRESHOLD` are set to 0, so the receiver's prediction is then exactly invariant to the
  sender, however unusual the sender's state is. Training uses the soft gates.
"""

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

import common
import experiment_intervention_actor as e2
import experiment_sparse_object_wm as e1

GATE_THRESHOLD = 0.1


class ReceiverSoftGatedWM(nn.Module):
    d: int = 128
    gate_type: str = "soft"  # "soft": sigmoid gates; "ste": binary gates with straight-through gradient

    @nn.compact
    def __call__(self, hist, action, train: bool = False):
        """Same interface as `e2.L0ObjectWM`; the 4th output is the L1 of the gates per receiver."""
        B = hist.shape[0]
        p = hist / common.POS_SCALE
        v = (hist[:, 1:] - hist[:, :-1]) / 8.0
        per_obj = jnp.concatenate(
            [p.transpose(0, 2, 1, 3).reshape(B, 3, -1), v.transpose(0, 2, 1, 3).reshape(B, 3, -1)], -1
        )
        enc = [e1.MLP([self.d, self.d], name=f"enc_{n}")(per_obj[:, i]) for i, n in enumerate(common.OBJECT_NAMES)]
        e_act = nn.Embed(common.NUM_ACTIONS, self.d, name="act_embed")(action)
        tokens = jnp.stack(enc + [e_act], axis=1)

        deltas, gates, l1s = [], [], []
        reward_logits = None
        for i, n in enumerate(common.OBJECT_NAMES):
            others = [j for j in range(4) if j != i]
            e_i = tokens[:, i]
            e_j = tokens[:, others]
            g = nn.sigmoid(e1.MLP([self.d, len(others)], name=f"gate_{n}")(e_i))  # receiver-only
            l1s.append(g.sum(-1))
            if self.gate_type == "ste":
                # forward: exactly 0 or 1; backward: gradient of the sigmoid
                g = g + jax.lax.stop_gradient((g > 0.5).astype(g.dtype) - g)
            elif not train:
                g = jnp.where(g < GATE_THRESHOLD, 0.0, g)
            pair = jnp.concatenate([jnp.broadcast_to(e_i[:, None], e_j.shape), e_j], -1)
            m = nn.LayerNorm(name=f"msg_ln_{n}")(e1.MLP([self.d, self.d], name=f"msg_{n}")(pair))
            msg = jnp.sum(g[..., None] * m, axis=1)
            h = e1.MLP([2 * self.d, self.d], act_last=True, name=f"dec_{n}")(jnp.concatenate([e_i, msg], -1))
            deltas.append(nn.Dense(2, name=f"out_{n}")(h))
            if n == "ball":
                reward_logits = nn.Dense(3, name="reward")(h)
            gates.append(jnp.ones((B, 4)).at[:, jnp.array(others)].set(g))
        return jnp.stack(deltas, 1), reward_logits, jnp.stack(gates, 1), jnp.stack(l1s, 1)


def build_model(cfg):
    """World model from a config dict (used by evaluate_lazy_enemy.py)."""
    return ReceiverSoftGatedWM(d=cfg["d"], gate_type=cfg.get("gate_type", "soft"))


wm_step = e2.wm_step
ACT_SAMPLE, ACT_GREEDY = e1.ACT_SAMPLE, e1.ACT_GREEDY


def main():
    ap = e2.get_parser(default_name="receiver_soft_gates")
    ap.add_argument("--gate_type", default="soft", choices=["soft", "ste"])
    cfg = ap.parse_args()
    e2.run(cfg, partial(ReceiverSoftGatedWM, gate_type=cfg.gate_type))


if __name__ == "__main__":
    main()
