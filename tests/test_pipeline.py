import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ocwm import data as D
from ocwm import model as M
from ocwm import actor as A
from ocwm.env import BALL, ENEMY, EVAL_MOD, NUM_OBJECTS, make_env


def test_collect_shapes_and_windows():
    d = D.collect(make_env(), jax.random.PRNGKey(0), num_envs=2, num_steps=50)
    assert d["pos"].shape == (2, 51, NUM_OBJECTS, 2)
    assert d["action"].shape == (2, 50)
    idx = D.valid_indices(d["done"], history=4)
    hist, act, nxt = D.gather_windows(jnp.asarray(d["pos"]), jnp.asarray(d["action"]), jnp.asarray(idx[:8]), 4)
    assert hist.shape == (8, 4, NUM_OBJECTS, 2) and act.shape == (8,) and nxt.shape == (8, NUM_OBJECTS, 2)


def test_valid_indices_drop_steps_after_game_end():
    done = np.zeros((1, 10), bool)
    done[0, 5] = True
    t = D.valid_indices(done, history=2)[:, 1]
    assert t.min() == 1 and t.max() == 5


def test_lazy_enemy_only_moves_while_ball_approaches():
    d = D.collect(make_env([EVAL_MOD]), jax.random.PRNGKey(0), num_envs=4, num_steps=400)
    enemy_moved = np.diff(d["pos"][:, :, ENEMY, 1], axis=1) != 0
    ball_vx = np.diff(d["pos"][:, :, BALL, 0], axis=1)
    moving_away = ball_vx > 0
    assert enemy_moved.any()
    assert not (enemy_moved & moving_away).any()


def test_closed_gate_makes_prediction_independent_of_source():
    model = M.WorldModel(M.ModelConfig(history=4), nnx.Rngs(0))
    hist = jax.random.uniform(jax.random.PRNGKey(0), (4, NUM_OBJECTS, 2)) * 100
    gates = jnp.ones((NUM_OBJECTS, M.NUM_SOURCES)).at[2, ENEMY].set(0.0)  # ball ignores enemy
    moved = hist.at[:, ENEMY].add(37.0)
    a = model.predict_single(hist, jnp.int32(0), gates)
    b = model.predict_single(moved, jnp.int32(0), gates)
    np.testing.assert_allclose(a[2], b[2], atol=1e-6)  # ball unchanged
    assert not np.allclose(a[0], b[0])  # player still attends to enemy, so it changes


def test_imagine_and_checkpoint_roundtrip(tmp_path):
    model = M.WorldModel(M.ModelConfig(history=3), nnx.Rngs(0))
    hist = jnp.ones((2, 3, NUM_OBJECTS, 2)) * 50
    pos, gates = M.imagine(model, hist, jnp.zeros((2, 5), jnp.int32))
    assert pos.shape == (2, 5, NUM_OBJECTS, 2) and gates.shape == (2, 5, NUM_OBJECTS, M.NUM_SOURCES)
    M.save(model, tmp_path / "m.pkl")
    loaded = M.load(tmp_path / "m.pkl")
    np.testing.assert_allclose(M.imagine(loaded, hist, jnp.zeros((2, 5), jnp.int32))[0], pos, atol=1e-6)


def test_planners_run():
    env = make_env([EVAL_MOD])
    model = M.WorldModel(M.ModelConfig(history=3), nnx.Rngs(0))
    for plan, history in [(A.planner(A.oracle_rollout(env, 16)), 2), (A.planner(A.model_rollout(model, 16)), 3)]:
        player, enemy, hits, frames = A.play(env, plan, jax.random.PRNGKey(0), 2, history, max_steps=64)
        assert player.shape == enemy.shape == hits.shape == frames.shape == (2,)
        assert np.all(frames == 64)  # nobody reaches 21 in 64 frames


def test_score_prefers_player_goal():
    pos0 = jnp.array([[140.0, 100], [16, 100], [60, 100]])
    traj = jnp.tile(pos0, (10, 1, 1))
    player_goal = traj.at[5:, 2, 0].set(10.0)  # ball passes the enemy column
    enemy_goal = traj.at[5:, 2, 0].set(150.0)
    assert A.score_trajectory(player_goal, pos0) > A.score_trajectory(traj, pos0) > A.score_trajectory(enemy_goal, pos0)
