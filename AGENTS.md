# AGENTS.md

Guidelines for coding agents working in this repository. See `project.md` for the project goals.

## Git workflow

- Work **only** on the branch `raban`. If it does not exist yet, create it (`git checkout -b raban`); otherwise switch to it (`git checkout raban`). Never commit to `main` or any other branch.
- Commit and push all changes, including every new experiment, to `raban` (`git push -u origin raban`). Do not leave finished work uncommitted or unpushed.

## Project management

- Use **uv** for everything: dependencies, environments, and running code.
  - Add dependencies with `uv add <package>` (never `pip install`).
  - Run scripts with `uv run <script>.py`.
  - Keep `pyproject.toml` and `uv.lock` in sync and commit both.

## ML / RL stack

- Use **JAX** for all ML and RL code (models, training loops, rollouts).
- Environments come from **JAXAtari** (`jaxatari.make("pong")`); keep the pipeline in JAX so environment steps and model updates can be `jax.jit`-compiled and `jax.vmap`-ed.
- Handle randomness explicitly with `jax.random` keys; split keys instead of reusing them.
- Prefer JAX-ecosystem libraries (e.g. Flax, Optax) over non-JAX frameworks. Do not introduce PyTorch or TensorFlow.

## Experiments

- For every new experiment setup (e.g. a new methodological approach, model architecture, or training scheme), **create a new file** rather than modifying an existing experiment.
- Give the file a descriptive name (e.g. `experiment_slot_attention.py`) and start it with a docstring explaining the approach and how it differs from earlier ones.
- Existing experiment files stay runnable so results remain reproducible and comparable.
- Shared, stable code (e.g. data collection, evaluation on the `lazy_enemy` mod) can be moved into common modules once it is used by more than one experiment.
