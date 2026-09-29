"""Smoke test for the intermediate 'track the ball' player policies."""

import pytest

from world_model.bounce_stats import analyze
from world_model.data import collect_rollout


@pytest.mark.parametrize("policy", ["track_ball", "epsilon_track_ball"])
def test_intermediate_policy_returns_the_ball_far_more_than_random(policy):
    random_data = collect_rollout(num_steps=3000, seed=0, policy="random")
    better_data = collect_rollout(num_steps=3000, seed=0, policy=policy)

    def return_rate(data):
        bounces, misses = analyze(data)
        player_bounces = sum(1 for b in bounces if b["side"] == "player")
        player_total = player_bounces + sum(1 for m in misses if m["side"] == "player")
        return player_bounces / player_total if player_total else 0.0

    assert return_rate(better_data) > return_rate(random_data)
