"""Collect random-policy rollouts from JAXAtari Seaquest.

Pass `--mods disable_enemies` to collect the "no enemies" test set: JAXAtari's
built-in Seaquest mod that zeroes all shark/sub/enemy-missile state on every
step. This is the Seaquest analogue of Pong's `lazy_enemy` robustness test -
train on the default distribution, evaluate under this intervention.
"""

import argparse
from pathlib import Path

import jax
import numpy as np

import jaxatari

from world_model.seaquest_objects import STATE_FNS

DEFAULT_OUT = Path("artifacts/seaquest_rollout.npz")


def collect_rollout(num_steps: int, seed: int = 0, mods=None):
    env = jaxatari.make("seaquest", mods=mods or [])
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    num_actions = env.action_space().n

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = reset(reset_key)

    states = {name: [fn(state)] for name, fn in STATE_FNS.items()}
    actions = []
    episode_ids = [0]
    episode_id = 0

    for _ in range(num_steps):
        key, action_key = jax.random.split(key)
        action = jax.random.randint(action_key, (), 0, num_actions)
        obs, state, reward, done, info = step(state, action)

        actions.append(int(action))
        for name, fn in STATE_FNS.items():
            states[name].append(fn(state))
        # Label the just-recorded state with the CURRENT episode id before
        # bumping it, so a reset never falsely joins two episodes together
        # (same reasoning as world_model/data.py's Pong collector).
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
    parser.add_argument("--mods", nargs="*", default=[], help="e.g. --mods disable_enemies")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = collect_rollout(args.num_steps, args.seed, mods=args.mods)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **data)

    num_episodes = int(data["episode_ids"][-1]) + 1
    mod_str = ",".join(args.mods) if args.mods else "none"
    print(f"Collected {args.num_steps} steps across {num_episodes} episode(s) (mods={mod_str}) -> {args.out}")


if __name__ == "__main__":
    main()
