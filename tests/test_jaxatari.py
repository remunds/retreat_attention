"""Smoke tests that JAXAtari's Pong environment loads and steps correctly."""

import jax
import jaxatari
import pytest

NUM_STEPS = 10


@pytest.fixture(scope="module")
def env():
    return jaxatari.make("pong")


def test_reset_returns_valid_state(env):
    reset = jax.jit(env.reset)
    obs, state = reset(jax.random.PRNGKey(0))
    assert obs is not None
    assert state is not None


def test_step_runs_and_updates_state(env):
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    num_actions = env.action_space().n
    assert num_actions > 0

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    obs, state = reset(reset_key)

    for _ in range(NUM_STEPS):
        key, action_key = jax.random.split(key)
        action = jax.random.randint(action_key, (), 0, num_actions)
        obs, state, reward, done, info = step(state, action)

        assert hasattr(state, "player_y")
        assert hasattr(state, "enemy_y")
        assert hasattr(state, "ball_x")
        assert hasattr(state, "ball_y")
        assert isinstance(float(reward), float)

        if done:
            break


def test_render_produces_frame(env):
    reset = jax.jit(env.reset)
    obs, state = reset(jax.random.PRNGKey(0))

    frame = env.render(state)
    assert frame.ndim == 3
    assert frame.shape[-1] == 3
