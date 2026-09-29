"""Fast end-to-end smoke test for approach B: the gated (cross-attention) world model."""

import jax
import pytest

from world_model.data import collect_rollout
from world_model.gated_model import attention_entropy, gate_forward, init_gate_params
from world_model.objects import OBJECT_DIMS
from world_model.train_gated import train_object


@pytest.fixture(scope="module")
def data():
    return collect_rollout(num_steps=300, seed=0)


def test_gate_forward_weights_sum_to_one():
    key = jax.random.PRNGKey(0)
    candidate_names = ["enemy", "ball", "action"]
    candidate_dims = {"enemy": OBJECT_DIMS["enemy"], "ball": OBJECT_DIMS["ball"], "action": 6}
    params = init_gate_params(key, OBJECT_DIMS["player"], candidate_dims)

    own_current = jax.numpy.zeros((5, OBJECT_DIMS["player"]))
    candidates = {
        "enemy": jax.numpy.zeros((5, OBJECT_DIMS["enemy"])),
        "ball": jax.numpy.zeros((5, OBJECT_DIMS["ball"])),
        "action": jax.nn.one_hot(jax.numpy.zeros((5,), dtype=jax.numpy.int32), 6),
    }
    context, weights, scores = gate_forward(params, own_current, candidates, candidate_names, temperature=1.0)

    assert context.shape == (5, 16)
    assert weights.shape == (5, 4)  # enemy, ball, action, null
    assert scores.shape == (5, 4)
    assert jax.numpy.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
    assert (attention_entropy(weights) >= 0).all()


@pytest.mark.parametrize("head", ["mlp", "linear"])
def test_train_gated_object_smoke(data, head):
    key = jax.random.PRNGKey(0)
    for name in OBJECT_DIMS:
        key, subkey = jax.random.split(key)
        result = train_object(
            name,
            data,
            subkey,
            epochs=1,
            batch_size=32,
            lr=1e-3,
            gate_lr=5e-3,
            sparsity_weight=0.01,
            temperature_start=1.0,
            temperature_end=1.0,
            head=head,
        )
        assert "gate_params" in result
        assert "pred_params" in result
        assert result["head"] == head
