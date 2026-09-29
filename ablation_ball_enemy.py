"""Diagnostic ablation: experiment 1's world model trained on random-policy Pong data with vs. without
the ball<-enemy message, reporting the ball's 1-step error by ball position and direction.

Usage: CUDA_VISIBLE_DEVICES=4 uv run ablation_ball_enemy.py
"""
import sys, argparse, jax, jax.numpy as jnp, numpy as np, optax, flax.linen as nn
import common, experiment_sparse_object_wm as e1

class MaskedWM(e1.SparseObjectWM):
    mask_ball_enemy: bool = False
    @nn.compact
    def __call__(self, hist, action):
        B = hist.shape[0]
        p = hist / common.POS_SCALE; v = (hist[:, 1:] - hist[:, :-1]) / 8.0
        per_obj = jnp.concatenate([p.transpose(0, 2, 1, 3).reshape(B, 3, -1), v.transpose(0, 2, 1, 3).reshape(B, 3, -1)], -1)
        enc = [e1.MLP([self.d, self.d], name=f"enc_{n}")(per_obj[:, i]) for i, n in enumerate(common.OBJECT_NAMES)]
        tokens = jnp.stack(enc + [nn.Embed(6, self.d, name="act_embed")(action)], 1)
        deltas, gates = [], []; rl = None
        for i, n in enumerate(common.OBJECT_NAMES):
            others = [j for j in range(4) if j != i]
            e_i = tokens[:, i]; e_j = tokens[:, others]
            pair = jnp.concatenate([jnp.broadcast_to(e_i[:, None], e_j.shape), e_j], -1)
            g = nn.sigmoid(e1.MLP([self.d, 1], name=f"gate_{n}")(pair)[..., 0])
            if n == "ball" and self.mask_ball_enemy: g = g.at[:, 1].set(0.0)  # others for ball = [player, enemy, action]
            m = nn.LayerNorm(name=f"msg_ln_{n}")(e1.MLP([self.d, self.d], name=f"msg_{n}")(pair))
            h = e1.MLP([2 * self.d, self.d], act_last=True, name=f"dec_{n}")(jnp.concatenate([e_i, jnp.sum(g[..., None] * m, 1)], -1))
            deltas.append(nn.Dense(2, name=f"out_{n}")(h))
            if n == "ball": rl = nn.Dense(3, name="reward")(h)
            gates.append(jnp.ones((B, 4)).at[:, jnp.array(others)].set(g))
        return jnp.stack(deltas, 1), rl, jnp.stack(gates, 1)

env = common.make_env()
traj = common.collect(env, common.random_act, 2000, None, jax.random.split(jax.random.PRNGKey(0), 256))
buf = common.ReplayBuffer(); buf.add(traj)
ev = common.collect(env, common.random_act, 2000, None, jax.random.split(jax.random.PRNGKey(99), 32))
vb = common.ReplayBuffer(); vb.add(ev); ei, ti = vb.valid_starts(16, 1); arr = vb.arrays()
sel = np.random.default_rng(0).choice(len(ei), 16384, replace=False); fo = np.arange(-15, 1)
f = jnp.asarray(arr["frames"][ei[sel, None], ti[sel, None] + fo[None]]); a = jnp.asarray(arr["action"][ei[sel], ti[sel]])
tgt = arr["frames"][ei[sel], ti[sel] + 1]
bx = np.asarray(f[:, -1, 2, 0]); vx = np.sign(np.asarray(f[:, -1, 2, 0] - f[:, -2, 2, 0]))
for mask in [False, True]:
    model = MaskedWM(d=128, mask_ball_enemy=mask)
    cfg = argparse.Namespace(K=16, H=8, gate_coef=0.01, wm_batch=512, wm_steps=20000)
    params = model.init(jax.random.PRNGKey(1), jnp.zeros((1, 16, 3, 2)), jnp.zeros((1,), jnp.int32))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-4, weight_decay=1e-4))
    params, _, st = e1.train_world_model(model, params, tx.init(params), tx, buf, cfg, jax.random.PRNGKey(2), lambda s: None)
    pred = np.asarray(e1.wm_step(model, params, f, a)[0])
    err = np.linalg.norm(pred[:, 2] - tgt[:, 2], axis=-1)
    print(f"mask_ball_enemy={mask}: val_loss={st['val_loss']:.3f} mse_ball(8-step)={st['mse_ball']:.3f} 1-step ball err mean={err.mean():.2f}")
    for lo, hi in [(0, 30), (30, 60), (60, 120), (120, 165)]:
        for s in [-1, 1]:
            m = (bx >= lo) & (bx < hi) & (vx == s)
            print(f"   x[{lo},{hi}) dir{s:+.0f} n={m.sum():5d} err={err[m].mean():.2f} p90={np.percentile(err[m], 90):.2f}")
