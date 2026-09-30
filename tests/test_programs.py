import jax.numpy as jnp
import numpy as np

from ocwm import programs as P
from ocwm import synthesis as S
from ocwm.env import NUM_OBJECTS

BALL_VX = ("proj", "ball", "vx")


def toy_features(n=400, seed=0):
    rng = np.random.default_rng(seed)
    hist = rng.integers(20, 150, size=(n, 2, NUM_OBJECTS, 2)).astype(np.float32)
    return P.features(hist, rng.integers(0, 6, size=n), np)


def test_synthesis_recovers_conditional_bounce():
    f = toy_features()
    # Bounce off a wall at x = 100: velocity flips there, otherwise it is kept.
    y = np.where(f[("ball", "x")] <= 100, f[("ball", "vx")], -f[("ball", "vx")])
    terms = S.enumerate_terms(f, 3)
    prog = S.Synthesizer(f, terms, tol=0.5, max_depth=4, min_leaf=5).solve(y, np.arange(len(y)))
    assert np.all(np.abs(P.evaluate(prog, f, np) - y) <= 0.5)
    assert P.static_sources(prog) == {"ball"}


def test_fitted_constants_and_simplify():
    f = toy_features()
    y = 0.7 * f[("player", "vy")] + 1.725 * f["down"]
    terms = S.enumerate_terms(f, 3)
    prog = P.simplify(S.Synthesizer(f, terms, tol=0.01, max_depth=2, min_leaf=5).solve(y, np.arange(len(y))))
    assert np.allclose(P.evaluate(prog, f, np), y, atol=0.01)
    assert P.static_sources(prog) == {"player", "action"}


def test_used_sources_follow_executed_branch():
    f = toy_features(n=50)
    cond = ("le", ("proj", "ball", "x"), 100.0)
    prog = ("ite", cond, BALL_VX, ("proj", "enemy", "y"))
    used = P.used_sources(prog, f, np)  # [n, 4] over player, enemy, ball, action
    left = f[("ball", "x")] <= 100
    assert np.all(used[left, 1] == 0) and np.all(used[~left, 1] == 1)
    assert np.all(used[:, 2] == 1)  # the condition always reads the ball


def test_program_model_roundtrip(tmp_path):
    progs = {(o, d): ("const", 0.0) for o, d in P.TARGETS}
    progs[("ball", "dx")] = BALL_VX
    model = P.ProgramModel(progs)
    model.save(tmp_path / "p.json")
    loaded = P.ProgramModel.load(tmp_path / "p.json")
    hist = jnp.ones((3, 2, NUM_OBJECTS, 2)).at[:, 1, 2, 0].set(4.0)  # ball x: 1 -> 4, so vx = 3
    nxt, deps = loaded.next_positions(hist, jnp.zeros(3, jnp.int32))
    assert np.allclose(nxt[:, 2, 0], 7.0) and deps.shape == (3, NUM_OBJECTS, len(P.SOURCES))


def test_excluded_object_is_never_read(tmp_path):
    f = toy_features()
    terms = S.enumerate_terms(f, 3, excluded=("enemy",))
    assert all("enemy" not in P.static_sources(node) for node, _, _ in terms)
    progs = {(o, d): ("const", 0.0) for o, d in P.TARGETS}
    P.ProgramModel(progs, ("enemy",)).save(tmp_path / "p.json")
    assert P.ProgramModel.load(tmp_path / "p.json").excluded == ("enemy",)
