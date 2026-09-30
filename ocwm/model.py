"""Object-centric world model with per-step, per-object dependency gates.

For every target object i and every source s (the other objects and the action), a gate
g[i, s] in {0, 1} is decided anew at every step from the current situation. Object i's next
state is predicted from its own history plus messages from the sources whose gate is open:

    h_i  = self_enc(history_i)
    h_i += sum_s g[i, s] * message(history_i, source_s)
    Δ_i  = head(h_i)

With g[i, s] = 0, the prediction for i is exactly independent of source s's history, so the
gates are also the dependency visualization. Gates are trained with straight-through
Gumbel-sigmoid samples and a sparsity penalty on how often they open.
"""

import pickle
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ocwm.env import NUM_ACTIONS, NUM_OBJECTS, OBJECTS, POS_DIM, SCREEN_H, SCREEN_W

SOURCE_NAMES = OBJECTS + ("action",)
NUM_SOURCES = len(SOURCE_NAMES)  # all objects + the action
ACTION_SOURCE = NUM_OBJECTS
SCALE = jnp.array([SCREEN_W, SCREEN_H])


@dataclass(frozen=True)
class ModelConfig:
    history: int = 4
    hidden: int = 128
    gate_hidden: int = 64

    def to_dict(self):
        return asdict(self)


class Stat(nnx.Variable):
    """Non-trainable normalization statistics, saved with the checkpoint."""


def mlp(sizes, rngs):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nnx.Linear(a, b, rngs=rngs))
        if i < len(sizes) - 2:
            layers.append(nnx.gelu)
    return nnx.Sequential(*layers)


def object_tokens(hist):
    """Per-object history features. hist: [H, 3, 2] pixels -> [3, D_obj]."""
    pos = hist / SCALE  # [H, 3, 2] normalized absolute positions
    vel = (hist[1:] - hist[:-1]) / 4.0  # [H-1, 3, 2] pixel velocities, roughly unit scale
    feats = jnp.concatenate([pos.transpose(1, 0, 2).reshape(NUM_OBJECTS, -1), vel.transpose(1, 0, 2).reshape(NUM_OBJECTS, -1)], -1)
    return jnp.concatenate([feats, jnp.eye(NUM_SOURCES)[:NUM_OBJECTS]], -1)


def source_tokens(hist, action):
    """Tokens for every source (objects + action), zero-padded to a common width. -> [4, D]."""
    obj = object_tokens(hist)
    act = jnp.concatenate([jax.nn.one_hot(action, NUM_ACTIONS), jnp.eye(NUM_SOURCES)[ACTION_SOURCE]])
    width = max(obj.shape[-1], act.shape[-1])
    pad = lambda x: jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, width - x.shape[-1])])
    return jnp.concatenate([pad(obj), pad(act)[None]], 0)


def pair_features(hist, action):
    """[3 targets, 4 sources, D_pair]: target token, source token, relative position."""
    tgt = object_tokens(hist)  # [3, Dt]
    src = source_tokens(hist, action)  # [4, Ds]
    cur = hist[-1] / SCALE  # [3, 2]
    rel = jnp.concatenate([cur[None, :, :] - cur[:, None, :], jnp.zeros((NUM_OBJECTS, 1, POS_DIM))], 1)
    tgt_b = jnp.broadcast_to(tgt[:, None], (NUM_OBJECTS, NUM_SOURCES, tgt.shape[-1]))
    src_b = jnp.broadcast_to(src[None], (NUM_OBJECTS, NUM_SOURCES, src.shape[-1]))
    return jnp.concatenate([tgt_b, src_b, rel * 4.0], -1)


# Self-pairs are handled by the self encoder, never by a gate.
GATE_MASK = jnp.concatenate([1.0 - jnp.eye(NUM_OBJECTS), jnp.ones((NUM_OBJECTS, 1))], 1)  # [3, 4]


class WorldModel(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.config = config
        H, h = config.history, config.hidden
        d_obj = H * POS_DIM + (H - 1) * POS_DIM + NUM_SOURCES
        d_src = max(d_obj, NUM_ACTIONS + NUM_SOURCES)
        d_pair = d_obj + d_src + POS_DIM
        self.self_enc = mlp([d_obj, h, h], rngs)
        self.message = mlp([d_pair, h, h], rngs)
        self.gate = mlp([d_pair, config.gate_hidden, 1], rngs)
        # Start with all gates open so the sparsity penalty prunes unused dependencies
        # instead of the model never discovering the rare useful ones (e.g. bounces).
        self.gate.layers[-1].bias[...] = jnp.full((1,), 3.0)
        self.head = mlp([h + NUM_OBJECTS, h, POS_DIM], rngs)
        # Per-object statistics of Δ = next_pos - pos, set from data before training.
        self.delta_mean = Stat(jnp.zeros((NUM_OBJECTS, POS_DIM)))
        self.delta_std = Stat(jnp.ones((NUM_OBJECTS, POS_DIM)))

    def gate_logits(self, hist, action):
        """[3, 4] gate logits for one sample. hist: [H, 3, 2], action: []."""
        return self.gate(pair_features(hist, action))[..., 0]

    def predict_single(self, hist, action, gates):
        """Normalized Δ prediction [3, 2] for one sample given gates [3, 4]."""
        h = self.self_enc(object_tokens(hist))  # [3, h]
        msgs = self.message(pair_features(hist, action))  # [3, 4, h]
        h = h + jnp.einsum("ts,tsh->th", gates * GATE_MASK, msgs)
        return self.head(jnp.concatenate([h, jnp.eye(NUM_OBJECTS)], -1))

    def __call__(self, hist, action, key=None, temperature=0.5):
        """Batched forward pass. hist: [B, H, 3, 2] pixels, action: [B].

        With `key`, gates are straight-through Gumbel-sigmoid samples (training); without,
        they are hard thresholds of the logits (evaluation). Returns (Δ_norm, gates, logits).
        """
        logits = jax.vmap(self.gate_logits)(hist, action)  # [B, 3, 4]
        if key is None:
            gates = (logits > 0).astype(jnp.float32)
        else:
            u = jax.random.uniform(key, logits.shape, minval=1e-6, maxval=1 - 1e-6)
            soft = jax.nn.sigmoid((logits + jnp.log(u) - jnp.log1p(-u)) / temperature)
            gates = (soft > 0.5).astype(jnp.float32) + soft - jax.lax.stop_gradient(soft)
        gates = gates * GATE_MASK
        delta = jax.vmap(self.predict_single)(hist, action, gates)
        return delta, gates, logits

    def next_positions(self, hist, action):
        """Deterministic next object positions in pixels, plus gates. hist: [B, H, 3, 2]."""
        delta, gates, _ = self(hist, action)
        return hist[:, -1] + delta * self.delta_std[...] + self.delta_mean[...], gates


def save(model, path):
    state = jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(model)))
    with open(path, "wb") as f:
        pickle.dump({"config": model.config.to_dict(), "state": state}, f)


def load(path):
    """Load a neural world model (.pkl) or a synthesized program model (.json)."""
    if str(path).endswith(".json"):
        from ocwm.programs import ProgramModel

        return ProgramModel.load(path)
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    model = WorldModel(ModelConfig(**ckpt["config"]), nnx.Rngs(0))
    state = nnx.state(model)
    nnx.replace_by_pure_dict(state, ckpt["state"])
    nnx.update(model, state)
    return model


def imagine(model, hist, actions):
    """Autoregressive rollout. hist: [B, H, 3, 2], actions: [B, K] -> positions [B, K, 3, 2], gates [B, K, 3, 4]."""

    def step(hist, action):
        nxt, gates = model.next_positions(hist, action)
        return jnp.concatenate([hist[:, 1:], nxt[:, None]], 1), (nxt, gates)

    _, (pos, gates) = jax.lax.scan(step, hist, actions.T)
    return pos.swapaxes(0, 1), gates.swapaxes(0, 1)
