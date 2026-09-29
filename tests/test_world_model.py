"""Fast end-to-end smoke test for the plain per-object world-model baseline."""

import jax
import pytest

from world_model.data import collect_rollout
from world_model.evaluate import rollout as eval_rollout
from world_model.objects import OBJECT_DIMS
from world_model.train import train_object
from world_model.windows import build_dataset


@pytest.fixture(scope="module")
def data():
    return collect_rollout(num_steps=300, seed=0)


def test_collect_rollout_shapes(data):
    for name, dim in OBJECT_DIMS.items():
        assert data[name].shape == (301, dim)
    assert data["actions"].shape == (300,)
    assert data["episode_ids"].shape == (301,)


def test_build_dataset_excludes_episode_boundaries(data):
    x, y = build_dataset(data, "ball", window=4)
    assert len(x) == len(y)
    assert x.shape[1] == 4 * OBJECT_DIMS["ball"]
    # every window+target span must stay within a single episode
    assert len(x) <= data["ball"].shape[0]


def test_train_and_rollout_smoke(data):
    key = jax.random.PRNGKey(0)
    results = {}
    for name in OBJECT_DIMS:
        key, subkey = jax.random.split(key)
        results[name] = train_object(name, data, subkey, epochs=1, batch_size=32, lr=1e-3)

    checkpoint = {"window": 4, "objects": results}
    predicted, truth = eval_rollout(checkpoint, data, start=10, horizon=5)

    for name in OBJECT_DIMS:
        assert predicted[name].shape == truth[name].shape == (5, OBJECT_DIMS[name])
