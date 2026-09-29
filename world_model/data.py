"""Collect JAXAtari Pong rollouts for world-model training.

Defaults to a uniform-random player policy, as every dataset used so far has.
Pass `policy="track_ball"` or `"epsilon_track_ball"` (see `policies.py`) for
an "intermediate" agent that actually tries to hit the ball back - `random`
was found (see solutions.md) to make the player return the ball only 12.4%
of the time it arrives, which starves the player-ball dependency of examples
to learn from independent of anything about the gating architecture.
"""

import argparse
from pathlib import Path

import jax
import numpy as np

import jaxatari

from world_model.objects import STATE_FNS
from world_model.policies import get_policy

DEFAULT_OUT = Path("artifacts/rollout.npz")


def collect_rollout(num_steps: int, seed: int = 0, policy: str = "random", epsilon: float = 0.2):
    env = jaxatari.make("pong")
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    num_actions = env.action_space().n
    act = get_policy(policy, epsilon)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = reset(reset_key)

    states = {name: [fn(state)] for name, fn in STATE_FNS.items()}
    actions = []
    episode_ids = [0]
    episode_id = 0

    for _ in range(num_steps):
        key, action_key = jax.random.split(key)
        action = act(action_key, state, num_actions)
        obs, state, reward, done, info = step(state, action)

        actions.append(int(action))
        for name, fn in STATE_FNS.items():
            states[name].append(fn(state))
        # Label the just-recorded state with the CURRENT episode id before
        # bumping it, so a reset never falsely joins two episodes together.
        episode_ids.append(episode_id)

        if bool(done):
            key, reset_key = jax.random.split(key)
            obs, state = reset(reset_key)
            episode_id += 1

    data = {name: np.array(vals, dtype=np.float32) for name, vals in states.items()}
    data["actions"] = np.array(actions, dtype=np.int32)
    data["episode_ids"] = np.array(episode_ids, dtype=np.int32)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=["random", "track_ball", "epsilon_track_ball"], default="random")
    parser.add_argument("--epsilon", type=float, default=0.2, help="exploration rate for epsilon_track_ball")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = collect_rollout(args.num_steps, args.seed, args.policy, args.epsilon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **data)

    num_episodes = int(data["episode_ids"][-1]) + 1
    print(f"Collected {args.num_steps} steps across {num_episodes} episode(s) (policy={args.policy}) -> {args.out}")


if __name__ == "__main__":
    main()
