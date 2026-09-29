"""Check whether the collected rollout actually contains player<->ball bounce
interactions to learn a dependency from - a data-availability question that's
independent of any architecture/optimization issue in the gate.

Two kinds of events are detected from the raw ball trajectory alone:
- A "bounce": ball_vel_x flips sign between consecutive steps with no big
  jump in position (a real physical bounce off a paddle or wall).
- A "goal reset": ball_x/ball_y jump back to (BALL_START_X, BALL_START_Y)
  from somewhere else - i.e. a point was scored and the ball was reset,
  rather than the paddle in front of it returning it.

Each event is attributed to the player or enemy side by whichever paddle's
fixed x the ball was nearest to just before the event.
"""

import argparse
from pathlib import Path

import numpy as np

from world_model.objects import ENEMY_X, PLAYER_X

DEFAULT_DATA = Path("artifacts/rollout.npz")
BALL_START_X = 78.0
BALL_START_Y = 115.0
PADDLE_HALF_HEIGHT = 8.0  # PLAYER_SIZE/ENEMY_SIZE height (16) / 2


def _side(ball_x_prev: float) -> str:
    return "player" if abs(ball_x_prev - PLAYER_X) < abs(ball_x_prev - ENEMY_X) else "enemy"


def analyze(data: dict):
    ball = data["ball"]
    player_y = data["player"][:, 0]
    enemy_y = data["enemy"][:, 0]
    episode_ids = data["episode_ids"]
    ball_x, ball_y, vel_x = ball[:, 0], ball[:, 1], ball[:, 2]

    bounces = []  # dicts: side, dy (ball_y - paddle_y just before contact)
    misses = []  # dicts: side (the side that FAILED to return it)

    for t in range(1, len(ball_x)):
        if episode_ids[t] != episode_ids[t - 1]:
            continue  # our own episode reset, not a real game transition

        is_reset_jump = (
            abs(ball_x[t] - BALL_START_X) < 1.0
            and abs(ball_y[t] - BALL_START_Y) < 1.0
            and (abs(ball_x[t - 1] - BALL_START_X) > 3.0 or abs(ball_y[t - 1] - BALL_START_Y) > 3.0)
        )
        if is_reset_jump:
            side = _side(ball_x[t - 1])
            misses.append({"side": side, "t": t})
            continue

        flipped_x = vel_x[t - 1] != 0 and vel_x[t] != 0 and np.sign(vel_x[t]) != np.sign(vel_x[t - 1])
        no_jump = abs(ball_x[t] - ball_x[t - 1]) < 10.0
        if flipped_x and no_jump:
            side = _side(ball_x[t - 1])
            paddle_y = player_y[t - 1] if side == "player" else enemy_y[t - 1]
            bounces.append({"side": side, "dy": float(ball_y[t - 1] - paddle_y), "t": t})

    return bounces, misses


def summarize(bounces, misses):
    lines = []
    for side in ("player", "enemy"):
        n_bounce = sum(1 for b in bounces if b["side"] == side)
        n_miss = sum(1 for m in misses if m["side"] == side)
        total = n_bounce + n_miss
        rate = (n_bounce / total * 100) if total else float("nan")
        lines.append(f"{side:6s}: {n_bounce:4d} bounces, {n_miss:4d} misses ({rate:.1f}% return rate, n={total})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    raw = np.load(args.data)
    data = {k: raw[k] for k in raw.files}
    bounces, misses = analyze(data)

    print(summarize(bounces, misses))
    player_dys = [b["dy"] for b in bounces if b["side"] == "player"]
    if player_dys:
        print(
            f"\nplayer bounce dy=ball_y-player_y: mean={np.mean(player_dys):.2f} "
            f"std={np.std(player_dys):.2f} range=[{min(player_dys):.1f}, {max(player_dys):.1f}] "
            f"(paddle half-height = {PADDLE_HALF_HEIGHT})"
        )


if __name__ == "__main__":
    main()
