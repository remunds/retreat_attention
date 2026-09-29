"""Train approach B: independent per-object predictors with a soft, learned
cross-object attention gate (see gated_model.py), trained via one-step
prediction on ground-truth history, same as the plain baseline in train.py.

Loss = one-step MSE + `sparsity_weight` * mean attention entropy. The first
run of this (see solutions.md) found the entropy penalty alone barely moved
attention even at 100x strength, so this version adds two more direct levers
for sharpening the gate:

- Softmax `temperature`, annealed geometrically from `--temperature-start` to
  `--temperature-end` over training: dividing scores by a small temperature
  mechanically forces a near-one-hot distribution regardless of how small the
  underlying score differences are, unlike the entropy penalty which only
  nudges gradients indirectly.
- A separate, higher learning rate (`--gate-lr`) for the gate's parameters,
  in case the gate was simply learning too slowly relative to the predictor
  MLP (which can partially compensate for a blurry context vector, weakening
  the gate's gradient signal).

The player's action is exposed as its own attention candidate (one-hot
embedded), alongside the other objects and the "null" option, rather than
being concatenated into the predictor's own-history input directly — so the
gate's weights show how much player prediction is attributed to the action
versus to (spurious) cross-object dependencies.

That experiment showed the gate can converge to a *confident but wrong*
dependency (e.g. player attending more to enemy than to its own action) with
little MSE cost, presumably because the MLP head has enough capacity to
quietly null out a bad context vector, leaving little gradient pressure on
the gate to pick the *correct* candidate rather than just *some* peaked one.
`--head linear` swaps in a plain linear predictor (`model.linear_forward`,
no hidden layer) to test that hypothesis directly: with no nonlinearity to
hide behind, a wrong context should cost real MSE, which should show up as
the gate shifting weight toward the actually-useful candidate.
"""

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from world_model.gated_model import VALUE_DIM, attention_entropy, gate_forward, init_gate_params
from world_model.model import forward, init_linear_params, init_params, linear_forward
from world_model.objects import OBJECT_DIMS
from world_model.windows import build_joint_dataset, split_indices

WINDOW = 4
HIDDEN_DIM = 64
DEFAULT_DATA = Path("artifacts/rollout.npz")
DEFAULT_OUT = Path("artifacts/gated_params.pkl")


def normalize_stats(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    return mean, std


def temperature_schedule(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    frac = epoch / (epochs - 1)
    return float(start * (end / start) ** frac)


def make_gate_optimizer(lr: float, gate_lr: float):
    def label_fn(params):
        return {
            "gate": jax.tree_util.tree_map(lambda _: "gate", params["gate"]),
            "pred": jax.tree_util.tree_map(lambda _: "pred", params["pred"]),
        }

    return optax.multi_transform({"gate": optax.adam(gate_lr), "pred": optax.adam(lr)}, label_fn)


def train_object(
    target_name: str,
    data: dict,
    key,
    epochs: int,
    batch_size: int,
    lr: float,
    gate_lr: float,
    sparsity_weight: float,
    temperature_start: float,
    temperature_end: float,
    head: str = "mlp",
):
    other_names = [name for name in OBJECT_DIMS if name != target_name]
    dim_t = OBJECT_DIMS[target_name]
    include_action = target_name == "player"
    num_actions = int(data["actions"].max()) + 1

    dataset = build_joint_dataset(data, target_name, WINDOW, list(OBJECT_DIMS), include_action=include_action)
    own, y = dataset["own"], dataset["y"]
    candidate_names = other_names + (["action"] if include_action else [])
    candidates_raw = {name: dataset[name] for name in other_names}
    if include_action:
        candidates_raw["action"] = dataset["action"]

    train_idx, val_idx = split_indices(len(own))
    own_train, own_val = own[train_idx], own[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    candidates_train = {n: v[train_idx] for n, v in candidates_raw.items()}
    candidates_val = {n: v[val_idx] for n, v in candidates_raw.items()}

    own_mean, own_std = normalize_stats(own_train)
    y_mean, y_std = normalize_stats(y_train)
    other_stats = {n: normalize_stats(candidates_train[n]) for n in other_names}

    def prepare_candidates(batch):
        result = {n: (batch[n] - other_stats[n][0]) / other_stats[n][1] for n in other_names}
        if include_action:
            result["action"] = jax.nn.one_hot(batch["action"], num_actions)
        return result

    key, k_gate, k_pred = jax.random.split(key, 3)
    candidate_dims = {n: dataset[n].shape[1] for n in other_names}
    if include_action:
        candidate_dims["action"] = num_actions
    gate_params = init_gate_params(k_gate, dim_t, candidate_dims)

    in_dim = own.shape[1] + VALUE_DIM  # own flattened window, plus attention context
    if head == "linear":
        pred_params = init_linear_params(k_pred, in_dim=in_dim, out_dim=dim_t)
        pred_forward = linear_forward
    else:
        pred_params = init_params(k_pred, in_dim=in_dim, hidden_dim=HIDDEN_DIM, out_dim=dim_t)
        pred_forward = forward

    params = {"gate": gate_params, "pred": pred_params}
    opt = make_gate_optimizer(lr, gate_lr)
    opt_state = opt.init(params)

    def forward_pass(params, own_batch, candidates_batch, temperature):
        own_norm = (own_batch - own_mean) / own_std
        own_current_norm = own_norm[:, -dim_t:]
        candidates_norm = prepare_candidates(candidates_batch)
        context, weights, scores = gate_forward(
            params["gate"], own_current_norm, candidates_norm, candidate_names, temperature
        )
        pred_input = jnp.concatenate([own_norm, context], axis=-1)
        pred_norm = pred_forward(params["pred"], pred_input)
        pred = pred_norm * y_std + y_mean
        return pred, weights, scores

    def loss_fn(params, own_batch, candidates_batch, y_batch, temperature):
        pred, weights, _ = forward_pass(params, own_batch, candidates_batch, temperature)
        mse = jnp.mean((pred - y_batch) ** 2)
        entropy = jnp.mean(attention_entropy(weights))
        return mse + sparsity_weight * entropy, (mse, entropy)

    @jax.jit
    def step(params, opt_state, own_batch, candidates_batch, y_batch, temperature):
        (loss, (mse, entropy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, own_batch, candidates_batch, y_batch, temperature
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, mse, entropy

    rng = np.random.default_rng(0)
    num_train = len(own_train)
    for epoch in range(epochs):
        temperature = temperature_schedule(epoch, epochs, temperature_start, temperature_end)
        perm = rng.permutation(num_train)
        for i in range(0, num_train, batch_size):
            idx = perm[i : i + batch_size]
            candidates_batch = {n: v[idx] for n, v in candidates_train.items()}
            params, opt_state, _, _ = step(
                params, opt_state, own_train[idx], candidates_batch, y_train[idx], temperature
            )

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            val_pred, val_weights, val_scores = forward_pass(params, own_val, candidates_val, temperature)
            val_mse = float(jnp.mean((val_pred - y_val) ** 2))
            val_entropy = float(jnp.mean(attention_entropy(val_weights)))
            score_std = float(jnp.mean(jnp.std(val_scores, axis=-1)))
            mean_weights = np.array(jnp.mean(val_weights, axis=0))
            labels = candidate_names + ["null"]
            weight_str = ", ".join(f"{lbl}={w:.2f}" for lbl, w in zip(labels, mean_weights))
            print(
                f"  [{target_name}] epoch {epoch + 1:3d}/{epochs} | T={temperature:.3f} | val MSE {val_mse:.4f} "
                f"| entropy {val_entropy:.3f} | score_std {score_std:.3f} | mean attn: {weight_str}"
            )

    return {
        "gate_params": params["gate"],
        "pred_params": params["pred"],
        "own_mean": own_mean,
        "own_std": own_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "other_stats": other_stats,
        "other_names": other_names,
        "candidate_names": candidate_names,
        "include_action": include_action,
        "num_actions": num_actions,
        "temperature": temperature_end,
        "head": head,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr", type=float, default=5e-3)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--temperature-start", type=float, default=2.0)
    parser.add_argument("--temperature-end", type=float, default=0.1)
    parser.add_argument("--head", choices=["mlp", "linear"], default="mlp")
    args = parser.parse_args()

    raw = np.load(args.data)
    data = {k: raw[k] for k in raw.files}

    key = jax.random.PRNGKey(0)
    results = {}
    for target_name in OBJECT_DIMS:
        print(f"Training gated {target_name} predictor...")
        key, subkey = jax.random.split(key)
        results[target_name] = train_object(
            target_name,
            data,
            subkey,
            args.epochs,
            args.batch_size,
            args.lr,
            args.gate_lr,
            args.sparsity_weight,
            args.temperature_start,
            args.temperature_end,
            args.head,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"window": WINDOW, "objects": results}, f)
    print(f"Saved trained gated model to {args.out}")


if __name__ == "__main__":
    main()
