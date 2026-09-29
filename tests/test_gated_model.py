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


def test_distance_bias_is_weak_at_init_and_only_suppresses():
    key = jax.random.PRNGKey(0)
    candidate_names = ["enemy", "ball"]
    candidate_dims = {"enemy": OBJECT_DIMS["enemy"], "ball": OBJECT_DIMS["ball"]}
    params = init_gate_params(key, OBJECT_DIMS["player"], candidate_dims, distance_candidates=candidate_names)

    own_current = jax.numpy.zeros((4, OBJECT_DIMS["player"]))
    candidates = {
        "enemy": jax.numpy.zeros((4, OBJECT_DIMS["enemy"])),
        "ball": jax.numpy.zeros((4, OBJECT_DIMS["ball"])),
    }

    # Distances are always normalized (roughly unit scale) before reaching the
    # gate in real training (see train_gated.py's dist_scale) - this is what
    # makes the near-zero init actually weak, regardless of a game's raw
    # coordinate units.
    _, weights_no_dist, scores_no_dist = gate_forward(params, own_current, candidates, candidate_names, 1.0)
    distances = {"enemy": jax.numpy.array([0.0, 0.5, 1.0, 2.0]), "ball": jax.numpy.zeros(4)}
    _, weights_dist, scores_dist = gate_forward(params, own_current, candidates, candidate_names, 1.0, distances)

    # At init (dist_bias_raw = -4 -> softplus ~ 0.018) the bias barely moves attention.
    assert jax.numpy.allclose(weights_no_dist, weights_dist, atol=0.01)
    # But it strictly decreases the biased candidate's score as distance grows, never increases it.
    enemy_scores = scores_dist[:, candidate_names.index("enemy")]
    assert bool((jax.numpy.diff(enemy_scores) < 0).all())
    ball_scores = scores_dist[:, candidate_names.index("ball")]
    assert bool(jax.numpy.allclose(ball_scores, scores_no_dist[:, candidate_names.index("ball")]))


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
