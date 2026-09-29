"""Player action policies for Pong rollout collection.

`random_policy` is what every dataset used so far — uniform over all actions,
regardless of state. `bounce_stats.py` found this makes the player return the
ball only 12.4% of the time it arrives (16 clean bounce examples in 20k
steps), vs. 78.6% for the scripted enemy AI — a real data-scarcity problem,
not an architecture one.

`track_ball_policy` is an "intermediate" agent: it mirrors the enemy AI's own
logic (JaxPong._enemy_step's `direction = sign(ball_y - paddle_y)`) to move
the player paddle toward the ball, using the same action mapping as the game
itself (`RIGHT`/action 2 moves the paddle up, i.e. y decreases; `LEFT`/action
3 moves it down — verified empirically, not just read off the game's naming).
It is NOT a trained policy or anything close to optimal play; it's the
simplest thing that actually tries to hit the ball back, which is all that's
needed to test whether more/cleaner bounce examples fix the gating problem.

`epsilon_track_ball_policy` mixes in some uniform-random exploration so the
resulting data isn't perfectly deterministic (still useful diversity for a
world model - e.g. varied paddle speeds/positions at contact, not just one
canonical trajectory every time).
"""

import jax
import jax.numpy as jnp

UP_ACTION = 2  # ACTION_SET[2] = RIGHT -> player_y decreases (game's own naming, not ours)
DOWN_ACTION = 3  # ACTION_SET[3] = LEFT -> player_y increases
NOOP_ACTION = 0


def random_policy(key, state, num_actions):
    return jax.random.randint(key, (), 0, num_actions)


def track_ball_policy(key, state, num_actions):
    del key, num_actions
    return jnp.where(
        state.ball_y < state.player_y,
        UP_ACTION,
        jnp.where(state.ball_y > state.player_y, DOWN_ACTION, NOOP_ACTION),
    )


def make_epsilon_track_ball_policy(epsilon: float):
    def policy(key, state, num_actions):
        k_explore, k_random = jax.random.split(key)
        explore = jax.random.bernoulli(k_explore, epsilon)
        return jnp.where(
            explore,
            jax.random.randint(k_random, (), 0, num_actions),
            track_ball_policy(key, state, num_actions),
        )

    return policy


def get_policy(name: str, epsilon: float = 0.2):
    if name == "random":
        return random_policy
    if name == "track_ball":
        return track_ball_policy
    if name == "epsilon_track_ball":
        return make_epsilon_track_ball_policy(epsilon)
    raise ValueError(f"unknown policy: {name}")
