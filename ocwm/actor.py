"""Actors for Pong. The paddle follows the ball with a hit offset k = ball_y - paddle_y.

The offset decides which fifth of the paddle the ball hits, and so the rebound angle (JaxPong:
vy = -2 .. +2). The actors differ only in how they pick k:

  - "planner": every `replan_every` steps, imagine each candidate k in closed loop with a
    predictor and pick the one whose imagined future scores best (ball gets past the enemy).
    The predictor is the learned world model ("model") or the true env ("oracle", upper bound).
    All knowledge of how a hit angle plays out comes from the predictor.
  - "fixed": a constant k, no model at all. Used as model-free baselines.

Every real step uses sticky actions (prob. 0.25 of repeating the previous action), so games
are stochastic.

Usage:
    uv run python -m ocwm.actor --policy planner --predictor oracle --mods lazy_enemy
    uv run python -m ocwm.actor --policy planner --predictor model --ckpt runs/base/model.pkl --mods lazy_enemy
    uv run python -m ocwm.actor --policy fixed --offset 8 --mods lazy_enemy
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ocwm import model as M
from ocwm.env import BALL, BALL_START_X, ENEMY, EVAL_MOD, PLAYER, make_env, object_positions

NOOP, UP, DOWN = 0, 2, 3
# One candidate per paddle fifth (sections are 3.2 px; k in [-4, 20] is a hit), extremes doubled.
HIT_OFFSETS = jnp.array([-2.0, 1.6, 4.8, 8.0, 11.2, 14.4, 17.0])
CENTER_OFFSET = 8.0  # middle fifth: flat rebound
STICKY_PROB = 0.25


def steer(pos, prev_pos, k):
    """Move the paddle so that ball_y - paddle_y = k, compensating for paddle momentum."""
    speed = pos[PLAYER, 1] - prev_pos[PLAYER, 1]
    err = (pos[BALL, 1] - k) - (pos[PLAYER, 1] + speed * 0.7 / 0.3)  # 0.3 = JaxPong paddle smoothing
    return jnp.where(err > 1.0, DOWN, jnp.where(err < -1.0, UP, NOOP)).astype(jnp.int32)


def score_trajectory(traj, pos0, use_enemy=True):
    """Value of an imagined trajectory [K, 3, 2] from the player's point of view.

    Player goal (ball past the enemy) = +2, earlier is better; enemy goal = -2, later is better.
    Without a goal in the horizon: how far the ball passes from the enemy paddle's center.
    Goals are detected by the ball leaving the paddle columns or by its jump back to the center.
    With use_enemy=False (enemy locked away) the enemy term is dropped: only imagined goals count.
    """
    K = traj.shape[0]
    bx, by, ey = traj[:, BALL, 0], traj[:, BALL, 1], traj[:, ENEMY, 1]
    prev = jnp.concatenate([pos0[BALL, :1], bx[:-1]])
    jump = (jnp.abs(bx - prev) > 20) & (jnp.abs(bx - BALL_START_X) <= 4)  # reset to center after a goal
    first = lambda m: jnp.where(m.any(), jnp.argmax(m), K)
    t_player = first((bx <= 13) | (jump & (prev < 80)))
    t_enemy = first((bx >= 147) | (jump & (prev >= 80)))
    before = jnp.arange(K) < jnp.minimum(t_player, t_enemy)
    gap = jnp.abs((by + 2) - (ey + 8)) * ((bx <= 24) & before)
    dense = jnp.clip(gap.max() / 30.0, 0.0, 1.0) * use_enemy
    return jnp.where(
        t_player < t_enemy, 2.0 - t_player / K, jnp.where(t_enemy < t_player, -2.0 + t_enemy / K, dense)
    )


def model_rollout(model, horizon):
    """Rollout with a world model: a neural nnx model or a synthesized ProgramModel."""
    if isinstance(model, nnx.Module):
        graphdef, params = nnx.split(model)
        get_model = lambda: nnx.merge(graphdef, params)
    else:
        get_model = lambda: model

    def rollout(state, hist, k):
        m = get_model()

        def step(hist, _):
            action = steer(hist[-1], hist[-2], k)
            nxt, _ = m.next_positions(hist[None], action[None])
            return jnp.concatenate([hist[1:], nxt]), nxt[0]

        return jax.lax.scan(step, hist, None, length=horizon)[1]

    return rollout


def oracle_rollout(env, horizon):
    def rollout(state, hist, k):
        def step(carry, _):
            state, pos, prev = carry
            obs, state, *_ = env.step(state, steer(pos, prev, k))
            new = object_positions(obs)
            return (state, new, pos), new

        return jax.lax.scan(step, (state, hist[-1], hist[-2]), None, length=horizon)[1]

    return rollout


def planner(rollout, use_enemy=True):
    def plan(state, hist):
        trajs = jax.vmap(lambda k: rollout(state, hist, k))(HIT_OFFSETS)
        scores = jax.vmap(score_trajectory, in_axes=(0, None, None))(trajs, hist[-1], use_enemy)
        scores = scores - 1e-3 * jnp.abs(HIT_OFFSETS - CENTER_OFFSET)  # ties -> flat hit
        return HIT_OFFSETS[jnp.argmax(scores)]

    return plan


def model_planner(model, horizon):
    """Planner with a world model; if the model locked the enemy away, scoring ignores it too."""
    return planner(model_rollout(model, horizon), use_enemy="enemy" not in getattr(model, "excluded", ()))


def fixed(k):
    return lambda state, hist: jnp.float32(k)


def play(env, plan, key, num_episodes, history, max_steps, replan_every=8, sticky=STICKY_PROB):
    """Play games to 21 (or max_steps). Returns player score, enemy score, player hits and frames
    played (until the game ended or max_steps); each [num_episodes]."""
    k_reset, k_run = jax.random.split(key)
    obs, state = jax.vmap(env.reset)(jax.random.split(k_reset, num_episodes))
    hist0 = jnp.repeat(jax.vmap(object_positions)(obs)[:, None], history, axis=1)  # [E, H, 3, 2]

    def env_step(carry, key, k):
        state, hist, done, prev_action, hits, frames = carry
        action = steer(hist[-1], hist[-2], k)
        action = jnp.where(jax.random.uniform(key) < sticky, prev_action, action)
        obs, new_state, _, new_done, _ = env.step(state, action)
        new_hist = jnp.concatenate([hist[1:], object_positions(obs)[None]])
        hit = (state.ball_vel_x > 0) & (new_state.ball_vel_x < 0) & (new_state.ball_x > 80)
        keep = lambda old, new: jax.tree.map(lambda o, n: jnp.where(done, o, n), old, new)
        return (keep(state, new_state), keep(hist, new_hist), done | new_done, action, hits + (hit & ~done), frames + ~done)

    def chunk(carry, key):
        state, hist = carry[0], carry[1]
        k = jax.vmap(plan)(state, hist)

        def inner(carry, key):
            return jax.vmap(env_step)(carry, jax.random.split(key, num_episodes), k), None

        return jax.lax.scan(inner, carry, jax.random.split(key, replan_every))[0], None

    @jax.jit
    def run(state, hist, key):
        zeros = jnp.zeros(num_episodes, jnp.int32)
        carry = (state, hist, jnp.zeros(num_episodes, bool), zeros, zeros, zeros)
        keys = jax.random.split(key, max_steps // replan_every)
        final = jax.lax.scan(chunk, carry, keys)[0]
        return final[0].player_score, final[0].enemy_score, final[4], final[5]

    return run(state, hist0, k_run)


def make_plan(args, env):
    """(plan, history) for the CLI arguments."""
    if args.policy == "fixed":
        return fixed(args.offset), 2
    if args.predictor == "oracle":
        return planner(oracle_rollout(env, args.horizon)), 2
    model = M.load(args.ckpt)
    return model_planner(model, args.horizon), model.config.history


def add_args(p):
    p.add_argument("--policy", choices=["planner", "fixed"], default="planner")
    p.add_argument("--predictor", choices=["model", "oracle"], default="model")
    p.add_argument("--ckpt", help="world model checkpoint (for --predictor model)")
    p.add_argument("--offset", type=float, default=CENTER_OFFSET, help="hit offset for --policy fixed")
    p.add_argument("--horizon", type=int, default=256, help="imagined steps per candidate")
    p.add_argument("--replan-every", type=int, default=8)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    p.add_argument("--mods", nargs="*", default=[EVAL_MOD])
    args = p.parse_args()

    env = make_env(args.mods)
    plan, history = make_plan(args, env)
    player, enemy, hits, _ = map(np.asarray, play(env, plan, jax.random.PRNGKey(args.seed), args.episodes, history, args.max_steps, args.replan_every))
    print(f"policy={args.policy} predictor={args.predictor} mods={args.mods}")
    print("  scores (player-enemy): " + "  ".join(f"{a}-{b}" for a, b in zip(player, enemy)))
    print(f"  mean player score {player.mean():.2f} ± {player.std():.2f} | mean enemy score {enemy.mean():.2f} | winners per hit {player.sum() / max(hits.sum(), 1):.2f}")


if __name__ == "__main__":
    main()
