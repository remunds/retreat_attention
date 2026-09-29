"""A tiny 2-layer MLP used independently per object (no cross-object dependencies)."""

import jax
import jax.numpy as jnp


def init_params(key, in_dim: int, hidden_dim: int, out_dim: int):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (in_dim, hidden_dim)) * jnp.sqrt(2.0 / in_dim),
        "b1": jnp.zeros((hidden_dim,)),
        "w2": jax.random.normal(k2, (hidden_dim, out_dim)) * jnp.sqrt(2.0 / hidden_dim),
        "b2": jnp.zeros((out_dim,)),
    }


def forward(params, x):
    h = jax.nn.relu(x @ params["w1"] + params["b1"])
    return h @ params["w2"] + params["b2"]


def init_linear_params(key, in_dim: int, out_dim: int):
    """A plain linear head — no hidden layer, so it has far less room to
    quietly null out an irrelevant/wrong context vector than the MLP above.
    Used to test whether that "shrugging off" capacity is what's letting the
    gate in gated_model.py converge to confident-but-wrong dependencies."""
    return {
        "w": jax.random.normal(key, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim),
        "b": jnp.zeros((out_dim,)),
    }


def linear_forward(params, x):
    return x @ params["w"] + params["b"]
