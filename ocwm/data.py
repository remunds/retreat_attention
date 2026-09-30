"""Collect object-centric Pong trajectories and turn them into training windows.

Usage:
    uv run python -m ocwm.data --out data/train.npz
    uv run python -m ocwm.data --out data/lazy.npz --mods lazy_enemy
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from ocwm.env import BALL, NUM_ACTIONS, PADDLE_H, BALL_H, PLAYER, make_env, object_positions


def tracking_policy(key, pos, prev_pos, eps, aim):
    """Noisy ball tracker so the data covers rallies and paddle bounces, not only misses.

    pos, prev_pos: [3, 2]; eps: prob. of a random action; aim: paddle offset in pixels.
    """
    k_rand, k_act = jax.random.split(key)
    speed = pos[PLAYER, 1] - prev_pos[PLAYER, 1]
    target = pos[BALL, 1] + BALL_H / 2 - PADDLE_H / 2 + aim
    err = target - (pos[PLAYER, 1] + 2.3 * speed)
    greedy = jnp.where(err > 2.0, 3, jnp.where(err < -2.0, 2, 0))  # 3 = down, 2 = up
    random_action = jax.random.randint(k_act, (), 0, NUM_ACTIONS)
    return jnp.where(jax.random.uniform(k_rand) < eps, random_action, greedy).astype(jnp.int32)


def collect(env, key, num_envs, num_steps, eps_range=(0.05, 1.0)):
    """Roll out `num_envs` parallel games for `num_steps` steps.

    Returns numpy arrays: pos [N, T+1, 3, 2], action [N, T], reward [N, T], done [N, T].
    """
    k_reset, k_eps, k_aim, k_run = jax.random.split(key, 4)
    obs, state = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
    pos0 = jax.vmap(object_positions)(obs)
    eps = jax.random.uniform(k_eps, (num_envs,), minval=eps_range[0], maxval=eps_range[1])
    aim = jax.random.uniform(k_aim, (num_envs,), minval=-8.0, maxval=8.0)

    def step(carry, key):
        state, pos, prev_pos = carry
        keys = jax.random.split(key, num_envs)
        action = jax.vmap(tracking_policy)(keys, pos, prev_pos, eps, aim)
        obs, state, reward, done, _ = jax.vmap(env.step)(state, action)
        new_pos = jax.vmap(object_positions)(obs)
        return (state, new_pos, pos), (new_pos, action, reward, done)

    @jax.jit
    def run(state, pos0, key):
        _, out = jax.lax.scan(step, (state, pos0, pos0), jax.random.split(key, num_steps))
        return out

    pos, action, reward, done = run(state, pos0, k_run)
    pos = jnp.concatenate([pos0[None], pos], axis=0)  # [T+1, N, 3, 2]
    to_np = lambda x: np.asarray(jnp.swapaxes(x, 0, 1))
    return dict(pos=to_np(pos), action=to_np(action), reward=to_np(reward), done=to_np(done))


def valid_indices(done, history, rollout=1):
    """(env, t) pairs whose window pos[t-history+1 : t+rollout+1] lies inside one game.

    A game ends when done is first set; everything after it is dropped.
    """
    num_steps = done.shape[1]
    ended_before = np.cumsum(done, axis=1) - done > 0  # done happened at an earlier step
    n, t = np.nonzero(~ended_before)
    last = np.minimum(t + rollout - 1, num_steps - 1)
    keep = (t >= history - 1) & (t + rollout <= num_steps) & ~ended_before[n, last]
    return np.stack([n[keep], t[keep]], axis=1).astype(np.int32)


def surprise_masks(pos, idx, rollout=1, threshold=1.0):
    """[len(idx), 3] bool: object deviates > threshold px from constant velocity within the window.

    Surprises are the rare interactions (bounces, goals, enemy pauses) a world model must get right;
    training oversamples them.
    """
    n, t = idx[:, 0], idx[:, 1]
    out = np.zeros((len(idx), pos.shape[2]), bool)
    for r in range(rollout):
        cur, prev, nxt = pos[n, t + r], pos[n, np.maximum(t + r - 1, 0)], pos[n, t + r + 1]
        out |= (np.abs(nxt - (2 * cur - prev)) > threshold).any(-1)
    return out


def gather_windows(pos, action, idx, history):
    """Batch of (history [B, H, 3, 2], action [B], next_pos [B, 3, 2]) at indices idx [B, 2]."""

    def one(i):
        n, t = i[0], i[1]
        hist = jax.lax.dynamic_slice_in_dim(pos[n], t - history + 1, history, axis=0)
        return hist, action[n, t], pos[n, t + 1]

    return jax.vmap(one)(idx)


def gather_sequences(pos, action, idx, history, rollout):
    """(history [B, H, 3, 2], actions [B, R], future positions [B, R, 3, 2]) at indices idx [B, 2]."""

    def one(i):
        n, t = i[0], i[1]
        hist = jax.lax.dynamic_slice_in_dim(pos[n], t - history + 1, history, axis=0)
        acts = jax.lax.dynamic_slice_in_dim(action[n], t, rollout, axis=0)
        future = jax.lax.dynamic_slice_in_dim(pos[n], t + 1, rollout, axis=0)
        return hist, acts, future

    return jax.vmap(one)(idx)


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--mods", nargs="*", default=[], help="e.g. lazy_enemy (training data should use none)")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = collect(make_env(args.mods), jax.random.PRNGKey(args.seed), args.num_envs, args.num_steps)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, mods=np.array(args.mods, dtype=str), **data)
    print(
        f"saved {args.out}: {data['action'].size} transitions, "
        f"player points {int(data['reward'].clip(0).sum())}, enemy points {int((-data['reward']).clip(0).sum())}"
    )


if __name__ == "__main__":
    main()
