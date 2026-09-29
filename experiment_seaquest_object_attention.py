"""Experiment S3 (Seaquest): object-centric attention agent + potential-based surfacing shaping.

S1 (`experiment_seaquest_ppo.py`, MLP on the flat 1136-dim observation) learned to pick up divers
but stalled at ~0.1 rescues per base game after 200M frames. It dies often in encounters with sharks
and subs, and it surfaces early (each surfacing with fewer than 6 divers costs one). An MLP has to
learn from absolute coordinates, separately for each of the 25 enemy slots, where things are *relative
to the submarine*.

Changes relative to S1 (same PPO loop, data and base-game-only training):
- **Object-centric attention policy.** One token per object: player, 4 divers, 12 sharks, 12 enemy
  subs, the surface sub, the player's torpedo and 4 enemy missiles, plus a global token (oxygen,
  divers carried, lives). A token holds the object's type, position relative to the submarine,
  absolute position, velocity (last two frames), size and orientation. Inactive objects are masked
  out. Two pre-LayerNorm self-attention layers (d = 64, 4 heads) share weights across all slots of a
  type, and the global and player tokens are read out into the policy and value heads.
- **Potential-based shaping** (policy-invariant; Ng et al. 1999), on top of S1's reward:
  Phi(s) = c_surf * [6 divers carried] * height(s) + c_ox * [oxygen < ox_low] * height(s), where height
  is 0 at the bottom and 1 at the surface. The agent gets r + gamma * Phi(s') - Phi(s), so rising
  towards the surface is rewarded when it carries 6 divers or is low on oxygen. Phi = 0 at terminals
  (life lost).

The held-out `gravity` mod is not used; selection is on the base game (most rescues, then score).
"""

import argparse

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import experiment_seaquest_ppo as s1
import seaquest_common as sc

# object groups in a flattened frame: (start, count, type ids)
N_TYPES = 8  # player, diver, shark, enemy sub, surface sub, torpedo, enemy missile, global
TYPE_PLAYER, TYPE_DIVER, TYPE_SHARK, TYPE_SUB, TYPE_SURF, TYPE_TORP, TYPE_MISSILE, TYPE_GLOBAL = range(8)
GROUPS = [(0, 1, [TYPE_PLAYER]), (8, 4, [TYPE_DIVER] * 4),
          (40, 25, [TYPE_SHARK] * 12 + [TYPE_SUB] * 12 + [TYPE_SURF]),
          (240, 5, [TYPE_TORP] + [TYPE_MISSILE] * 4)]
TYPES = np.concatenate([np.array(t) for _, _, t in GROUPS] + [np.array([TYPE_GLOBAL])])
N_TOK = len(TYPES)  # 35 objects + 1 global token
TOK_DIM = N_TYPES + 12


def _fields(frame, start, n):
    """(..., 284) -> dict of (..., n) arrays for the 8 object fields."""
    names = ("x", "y", "w", "h", "active", "vid", "state", "orient")
    return {k: frame[..., start + i * n:start + (i + 1) * n] for i, k in enumerate(names)}


def tokens(obs):
    """Raw OC stack (..., 4, 284) -> tokens (..., N_TOK, TOK_DIM) and mask (..., N_TOK)."""
    cur, prev = obs[..., -1, :], obs[..., -2, :]
    px, py = cur[..., 0:1], cur[..., 1:2]
    feats, masks = [], []
    for start, n, _ in GROUPS:
        c, p = _fields(cur, start, n), _fields(prev, start, n)
        act = (c["active"] > 0).astype(jnp.float32)
        moving = act * (p["active"] > 0)
        f = jnp.stack([
            (c["x"] - px) / 80.0, (c["y"] - py) / 100.0, c["x"] / 160.0, c["y"] / 210.0,
            moving * jnp.clip(c["x"] - p["x"], -8, 8) / 8.0, moving * jnp.clip(c["y"] - p["y"], -8, 8) / 8.0,
            c["w"] / 16.0, c["h"] / 16.0, c["orient"], c["vid"] / 5.0, act, c["state"]], axis=-1)
        feats.append(f * act[..., None])
        masks.append(act)
    g = jnp.stack([cur[..., sc.OXYGEN_IDX] / 64.0, cur[..., sc.DIVERS_IDX] / 6.0, cur[..., sc.LIVES_IDX] / 4.0,
                   cur[..., 0] / 160.0, cur[..., 1] / 210.0, cur[..., 7],
                   (cur[..., sc.OXYGEN_IDX] - prev[..., sc.OXYGEN_IDX]) / 4.0,
                   (cur[..., sc.DIVERS_IDX] >= 6).astype(jnp.float32),
                   (cur[..., sc.OXYGEN_IDX] < 16).astype(jnp.float32), jnp.zeros_like(cur[..., 0]),
                   jnp.zeros_like(cur[..., 0]), jnp.ones_like(cur[..., 0])], axis=-1)[..., None, :]
    x = jnp.concatenate(feats + [g], axis=-2)  # (..., N_TOK, 12)
    onehot = jnp.broadcast_to(jax.nn.one_hot(jnp.asarray(TYPES), N_TYPES), x.shape[:-1] + (N_TYPES,))
    mask = jnp.concatenate(masks + [jnp.ones_like(masks[0][..., :1])], axis=-1)
    mask = mask.at[..., 0].set(1.0)  # the player token is always attended (also while exploding)
    return jnp.concatenate([onehot, x], axis=-1), mask


class Block(nn.Module):
    d: int
    heads: int

    @nn.compact
    def __call__(self, x, mask):
        att_mask = (mask[..., None, :] > 0)[..., None, :, :]  # (..., 1, 1, N) -> broadcast over heads/queries
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(num_heads=self.heads, qkv_features=self.d)(h, h, mask=att_mask)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.d)(nn.gelu(nn.Dense(2 * self.d)(h)))
        return x + h


class ObjectAttentionAC(nn.Module):
    d: int = 64
    heads: int = 4
    layers: int = 2
    hidden: int = 256

    @nn.compact
    def __call__(self, tok, mask):
        orth = nn.initializers.orthogonal
        x = nn.Dense(self.d)(tok)
        for _ in range(self.layers):
            x = Block(self.d, self.heads)(x, mask)
        x = nn.LayerNorm()(x)
        # readout: global token, player token and a masked mean over all objects
        pooled = (x * mask[..., None]).sum(-2) / jnp.maximum(mask.sum(-1, keepdims=True), 1.0)
        h = jnp.concatenate([x[..., -1, :], x[..., 0, :], pooled], axis=-1)
        h = nn.relu(nn.Dense(self.hidden, kernel_init=orth(np.sqrt(2)))(h))
        logits = nn.Dense(sc.NUM_ACTIONS, kernel_init=orth(0.01), name="pi")(
            nn.relu(nn.Dense(self.hidden, kernel_init=orth(np.sqrt(2)))(h)))
        value = nn.Dense(1, kernel_init=orth(1.0), name="v")(
            nn.relu(nn.Dense(self.hidden, kernel_init=orth(np.sqrt(2)))(h)))[..., 0]
        return logits, value


AC = ObjectAttentionAC()


def apply_fn(params, obs):
    tok, mask = tokens(obs)
    return AC.apply(params, tok, mask)


def init_fn(key):
    tok, mask = tokens(jnp.zeros((1, sc.FRAME_STACK, sc.FRAME_DIM)))
    return AC.init(key, tok, mask)


def ACT_SAMPLE(params, obs, key):
    logits, _ = apply_fn(params, obs[None])
    return jax.random.categorical(key, logits[0])


def ACT_GREEDY(params, obs, key):
    logits, _ = apply_fn(params, obs[None])
    return jnp.argmax(logits[0])


AGENT = (apply_fn, init_fn, ACT_SAMPLE, ACT_GREEDY)


def height(g):
    return (141.0 - g.player_y.astype(jnp.float32)) / 95.0


def potential(cfg, g):
    return height(g) * (cfg.c_surf * (g.divers_collected >= 6) + cfg.c_ox * (g.oxygen < cfg.ox_low))


def reward_fn(cfg, g0, g1, env_done):
    r = s1.shaped_reward(cfg, g0, g1, env_done)
    terminal = env_done | (g1.lives < g0.lives)
    phi1 = jnp.where(terminal, 0.0, potential(cfg, g1))
    return r + jnp.where(env_done, 0.0, cfg.gamma * phi1 - potential(cfg, g0))


def get_parser(name="seaquest_object_attention"):
    ap = s1.get_parser()
    ap.set_defaults(name=name, life_penalty=0.0, surface_penalty=0.0, diver_bonus=2.0, ent_coef=0.01,
                    updates=4000, lr=3e-4)
    ap.add_argument("--c_surf", type=float, default=2.0, help="potential: surfacing with 6 divers")
    ap.add_argument("--c_ox", type=float, default=1.0, help="potential: surfacing when low on oxygen")
    ap.add_argument("--ox_low", type=float, default=16.0)
    return ap


def main():
    cfg = get_parser().parse_args()
    s1.main(cfg, experiment="experiment_seaquest_object_attention", agent=AGENT, reward_fn=reward_fn)


if __name__ == "__main__":
    main()
