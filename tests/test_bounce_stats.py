"""Smoke test for the player/enemy bounce-vs-miss data-availability check."""

from world_model.bounce_stats import analyze
from world_model.data import collect_rollout


def test_analyze_runs_and_classifies_by_side():
    data = collect_rollout(num_steps=5000, seed=0)
    bounces, misses = analyze(data)

    assert len(bounces) + len(misses) > 0
    for event in bounces + misses:
        assert event["side"] in ("player", "enemy")
    for b in bounces:
        assert "dy" in b
