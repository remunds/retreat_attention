"""Smoke test for cross-referencing bounce events against joint-dataset rows."""

import jax

from world_model.bounce_stats import analyze
from world_model.data import collect_rollout
from world_model.inspect_bounce_attention import recompute_weights
from world_model.objects import OBJECT_DIMS, distance
from world_model.train_gated import WINDOW, train_object
from world_model.windows import build_joint_dataset


def test_bounce_rows_are_found_and_weights_recomputed():
    data = collect_rollout(num_steps=3000, seed=0, policy="epsilon_track_ball")
    result = train_object(
        target_name="ball",
        data=data,
        key=jax.random.PRNGKey(0),
        epochs=1,
        batch_size=32,
        lr=1e-3,
        gate_lr=5e-3,
        sparsity_weight=0.01,
        temperature_start=1.0,
        temperature_end=1.0,
    )

    dataset = build_joint_dataset(
        data, "ball", WINDOW, list(OBJECT_DIMS), include_action=result["include_action"], distance_fn=distance
    )
    weights = recompute_weights(result, dataset)
    assert weights.shape[0] == len(dataset["t"])

    bounces, _ = analyze(data)
    assert len(bounces) > 0  # epsilon_track_ball should produce plenty of bounces even in 3000 steps

    row_t = dataset["t"]
    t_to_row = {int(t): i for i, t in enumerate(row_t)}
    matched = [t_to_row[e["t"] - 1] for e in bounces if (e["t"] - 1) in t_to_row]
    assert len(matched) > 0
