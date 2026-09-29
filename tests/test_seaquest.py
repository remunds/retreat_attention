"""Smoke tests for the Seaquest environment plumbing and the "no enemies" test set."""

import numpy as np

from world_model.seaquest_data import collect_rollout
from world_model.seaquest_objects import OBJECT_DIMS


def test_collect_rollout_shapes():
    data = collect_rollout(num_steps=300, seed=0)
    for name, dim in OBJECT_DIMS.items():
        assert data[name].shape == (301, dim)
    assert data["actions"].shape == (300,)
    assert data["episode_ids"].shape == (301,)


def test_disable_enemies_mod_zeroes_threat_object():
    data = collect_rollout(num_steps=300, seed=0, mods=["disable_enemies"])
    assert np.all(data["threat"] == 0.0)


def test_default_rollout_has_some_active_threats():
    # Enemies only start spawning after the player survives a long warm-up
    # (fill oxygen, start diving, then a ~277-step spawn-timer countdown), so
    # this needs enough steps for at least one episode to get that far.
    data = collect_rollout(num_steps=1500, seed=0, mods=[])
    assert np.any(data["threat"] != 0.0)
