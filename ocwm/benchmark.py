"""Success criteria. The world model is trained on the unmodified game only; everything below runs
with the intervention it never saw (default: JAXAtari's lazy_enemy, the enemy only tracks the ball
while the ball moves toward it).
Games use sticky actions (p=0.25). Three candidate criteria are reported side by side:

  A. Original:    the world-model planner scores > 10 points on average.
  B. Actor+model: the planner wins >= 90% of games, its winners per hit (player points / player
                  returns) beats the best model-free fixed hit offset, and reaches >= 90% of the same
                  planner with the true env (oracle). Winners per hit is undefined when the player
                  never returns a ball; then those checks are reported as N/A.
  C. Prediction:  containment of the intervention, as a counterfactual on the same situations: in
                  windows where the ball stays far from the enemy (x > 40), replacing the enemy's
                  trajectory with a lazy enemy's must not raise the ball's or player's 1-step or
                  16-step open-loop error by more than 10%. (Comparing games played under the mod
                  instead would be confounded, because the mod changes which situations occur.)

Usage:
    uv run python -m ocwm.benchmark --ckpt runs/v2/model.pkl
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from ocwm import actor as A
from ocwm import data as D
from ocwm import model as M
from ocwm.env import BALL, ENEMY, EVAL_MOD, PLAYER, make_env

FAR_FROM_ENEMY_X = 40
TOLERANCE = 1.1


def game_stats(env, plan, history, args):
    player, enemy, hits, _ = map(
        np.asarray, A.play(env, plan, jax.random.PRNGKey(args.seed), args.episodes, history, args.max_steps)
    )
    wph = player.sum() / hits.sum() if hits.sum() > 0 else float("nan")
    return dict(win=float((player >= 21).mean()), wph=wph, hits=hits.mean(), player=player.mean(), enemy=enemy.mean())


def lazy_enemy_track(ball_x, ball_y, enemy_y0, step=2):
    """Enemy y that JAXAtari's lazy_enemy would follow given the ball's positions [T].

    LazyEnemyMod: move `step` px toward the ball, but only while the ball moves toward the enemy
    (vx < 0) and not on every 8th frame. The frame counter's phase is unknown here, so the
    pause is placed relative to the window start.
    """

    def f(e, inp):
        i, bx_prev, bx, by = inp
        move = (i % 8 != 0) & (bx < bx_prev)
        e = jnp.where(move, e + step * jnp.sign(by - e), e)
        return e, e

    idx = jnp.arange(1, len(ball_y))
    ys = jax.lax.scan(f, enemy_y0, (idx, ball_x[:-1], ball_x[1:], ball_y[:-1]))[1]
    return jnp.concatenate([enemy_y0[None], ys])


def rollout_forced_enemy(model, hist, acts, enemy_future):
    """Open-loop rollout where the enemy follows a given trajectory [B, R] instead of the model."""

    def step(hist, inp):
        act, ey = inp
        nxt, _ = model.next_positions(hist, act)
        nxt = nxt.at[:, ENEMY, 1].set(ey)
        return jnp.concatenate([hist[:, 1:], nxt[:, None]], 1), nxt

    return jax.lax.scan(step, hist, (acts.T, enemy_future.T))[1].swapaxes(0, 1)


def containment_errors(model, seed, rollout):
    """Counterfactual containment test on the same situations of the unmodified game.

    Windows where the ball stays far from the enemy (so the ball's and player's true futures do not
    depend on it). The rollout is run twice with the enemy forced along (a) its real trajectory and
    (b) the trajectory a lazy enemy would have taken. Returns ball/player errors
    [1-step ball, 1-step player, R-step ball, R-step player] for (a) and (b), in pixels.
    """
    H = model.config.history
    d = D.collect(make_env(), jax.random.PRNGKey(seed), 32, 2048)
    idx = D.valid_indices(d["done"], H, rollout)
    far = np.stack([d["pos"][idx[:, 0], idx[:, 1] - H + 1 + r, BALL, 0] > FAR_FROM_ENEMY_X for r in range(H + rollout)]).all(0)
    idx = jnp.asarray(idx[far])
    pos, action = jnp.asarray(d["pos"]), jnp.asarray(d["action"])
    hist, acts, future = D.gather_sequences(pos, action, idx, H, rollout)
    window = jnp.concatenate([hist, future], 1)  # [B, H+R, 3, 2]
    lazy = jax.vmap(lambda w: lazy_enemy_track(w[:, BALL, 0], w[:, BALL, 1], w[0, ENEMY, 1]))(window)
    hist_cf = hist.at[:, :, ENEMY, 1].set(lazy[:, :H])

    @jax.jit
    def errors(hist, enemy_future):
        pred = rollout_forced_enemy(model, hist, acts, enemy_future)
        err = jnp.abs(pred - future).mean(-1)  # [B, R, 3]
        return jnp.stack([err[:, 0, BALL].mean(), err[:, 0, PLAYER].mean(), err[:, :, BALL].mean(), err[:, :, PLAYER].mean()])

    return np.asarray(errors(hist, future[:, :, ENEMY, 1])), np.asarray(errors(hist_cf, lazy[:, H:]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mod", default=EVAL_MOD)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--horizon", type=int, default=256)
    p.add_argument("--rollout", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = make_env([args.mod])
    model = M.load(args.ckpt)
    fmt = lambda s: f"win {s['win']:.0%}  score {s['player']:.1f}-{s['enemy']:.1f}  hits/game {s['hits']:.1f}  winners/hit {s['wph']:.3f}"

    wm = game_stats(env, A.model_planner(model, args.horizon), model.config.history, args)
    print(f"world-model planner : {fmt(wm)}", flush=True)
    oracle = game_stats(env, A.planner(A.oracle_rollout(env, args.horizon)), 2, args)
    print(f"oracle planner      : {fmt(oracle)}", flush=True)
    fixed = {}
    for k in np.asarray(A.HIT_OFFSETS):
        fixed[float(k)] = game_stats(env, A.fixed(k), 2, args)
        print(f"fixed offset {k:5.1f}  : {fmt(fixed[float(k)])}", flush=True)
    # Best model-free baseline: most games won, then winners per hit. Winners per hit alone would
    # favour an offset that barely touches the ball and loses every game.
    measured = {k: v for k, v in fixed.items() if not np.isnan(v["wph"])}
    best_k = max(measured, key=lambda k: (measured[k]["win"], measured[k]["wph"])) if measured else None

    base, mod = containment_errors(model, args.seed + 1, args.rollout)
    print(f"containment errors px [1-step ball, 1-step player, {args.rollout}-step ball, {args.rollout}-step player]")
    print(f"  real enemy: {np.round(base, 3)}\n  lazy enemy: {np.round(mod, 3)}")

    na = np.isnan(wm["wph"]) or np.isnan(oracle["wph"]) or best_k is None
    criteria = {
        "A. original": [("world-model planner scores > 10", wm["player"] > 10, f"{wm['player']:.1f} points")],
        "B. actor + model": [
            ("wins >= 90% of games", wm["win"] >= 0.9, f"{wm['win']:.0%}"),
            ("winners/hit > best winning model-free", None if na else wm["wph"] > measured[best_k]["wph"],
             "no player hits" if na else f"{wm['wph']:.3f} vs {measured[best_k]['wph']:.3f} (offset {best_k})"),
            ("winners/hit >= 90% of oracle", None if na else wm["wph"] >= 0.9 * oracle["wph"],
             "no player hits" if na else f"{wm['wph']:.3f} vs {0.9 * oracle['wph']:.3f}"),
        ],
        "C. prediction": [
            (name, bool(m <= TOLERANCE * b), f"{m:.3f} vs {b:.3f} px")
            for name, m, b in zip(
                ["1-step ball", "1-step player", f"{args.rollout}-step ball", f"{args.rollout}-step player"], mod, base
            )
        ],
    }
    print(f"\nsuccess criteria under {args.mod}:")
    for group, checks in criteria.items():
        results = [ok for _, ok, _ in checks]
        overall = "N/A" if any(r is None for r in results) else ("PASS" if all(results) else "FAIL")
        print(f"{group}: {overall}")
        for name, ok, detail in checks:
            print(f"  [{'N/A ' if ok is None else ('PASS' if ok else 'FAIL')}] {name}: {detail}")


if __name__ == "__main__":
    main()
