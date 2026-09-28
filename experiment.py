"""Load Pong in JAXAtari, run a few random steps, and print the object-centric state."""

import jax
import jaxatari

NUM_STEPS = 10


def main():
    env = jaxatari.make("pong")
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    num_actions = env.action_space().n

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    obs, state = reset(reset_key)
    print(f"JAX devices: {jax.devices()}")
    print(f"Pong loaded with {num_actions} actions")

    for t in range(NUM_STEPS):
        key, action_key = jax.random.split(key)
        action = jax.random.randint(action_key, (), 0, num_actions)
        obs, state, reward, done, info = step(state, action)
        print(
            f"step {t:2d} | action {int(action)} | reward {float(reward):+.0f} | "
            f"player_y {int(state.player_y)} | enemy_y {int(state.enemy_y)} | "
            f"ball ({int(state.ball_x)}, {int(state.ball_y)})"
        )
        if done:
            break

    frame = env.render(state)
    print(f"Rendered frame shape: {frame.shape}")
    print(f"Final observation: {obs}")


if __name__ == "__main__":
    main()
