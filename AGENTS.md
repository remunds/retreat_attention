# AGENTS.md

Guidelines for coding agents working in this repository. See `project.md` for the project goals.

## Goal and stopping criterion

- Your task is finished **only** when an actor trained inside a learned world model is robust to the `lazy_enemy` modification: with `lazy_enemy` active, it must reach a **final score of at least +10** on average, over at least 10 full games with different random seeds. The final score is the Pong score: the agent's points minus the enemy's points at the end of a game (e.g. 21 : 11 = +10). The agent therefore has to win, and win clearly; scoring 10 points in a lost game does not count.
- Do not stop before this is reached. If an approach falls short, document the result in its experiment Markdown file, work out why it failed, and try a new approach in a new experiment file. Do not stop just because one approach failed, or to hand a partial result back.
- The criterion must be met under all the rules below. In particular, the world model and the actor are trained only on unmodified Pong, and every design decision is based on unmodified Pong (see "Robustness evaluation with `lazy_enemy`"). An actor that reaches the score only because `lazy_enemy` leaked into training, tuning, or model selection does not count.
- When the criterion is met, record the final numbers (scores with and without `lazy_enemy`, number of episodes, seeds) and the exact commands to reproduce them in the experiment's Markdown file, then commit and push.

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
- Use the **object-centric (OC) inputs** of JAXAtari, not pixel observations: models work on the per-object state (e.g. positions and sizes of the paddles and the ball), obtained via JAXAtari's object-centric wrappers (`AtariWrapper` followed by `ObjectCentricWrapper`) or its structured OC observations.
- Handle randomness explicitly with `jax.random` keys; split keys instead of reusing them.
- Prefer JAX-ecosystem libraries (e.g. Flax, Optax) over non-JAX frameworks. Do not introduce PyTorch or TensorFlow.

## GPU usage

- You may use **only GPUs 4 and 5**. Never use any other GPU on this machine.
- Occupy **at most 2 GPUs at the same time**, across all running jobs combined.
- Always set `CUDA_VISIBLE_DEVICES` explicitly when running code, e.g. `CUDA_VISIBLE_DEVICES=4 uv run experiment_x.py` or `CUDA_VISIBLE_DEVICES=4,5 uv run experiment_x.py`. Never run JAX code without it, since JAX would otherwise grab every visible GPU.
- Check `nvidia-smi` before starting a job, and do not start new GPU jobs while your earlier ones are still using both allowed GPUs.

## Experiments

- For every new experiment setup (e.g. a new methodological approach, model architecture, or training scheme), **create a new file** rather than modifying an existing experiment.
- Give the file a descriptive name (e.g. `experiment_slot_attention.py`) and start it with a docstring explaining the approach and how it differs from earlier ones.
- Existing experiment files stay runnable so results remain reproducible and comparable.
- Shared, stable code (e.g. data collection, evaluation on the `lazy_enemy` mod) can be moved into common modules once it is used by more than one experiment.
- Feel free to try several different approaches. For each one, check that it actually learns (e.g. falling training and validation loss, accurate rollouts) and then evaluate its robustness as described below.

### Experiment documentation

Every experiment file gets an accompanying Markdown file with the same name (e.g. `experiment_slot_attention.py` → `experiment_slot_attention.md`). It must contain:

- **Goal:** what the experiment is trying to achieve.
- **Research question:** the specific question the experiment answers.
- **Experiments:** which runs were done, with their configurations and the exact commands used.
- **Results:** a summary of the outcomes, including key numbers (losses, scores with and without `lazy_enemy`) and what they mean for the research question.

Keep this file up to date as runs finish, and commit it together with the code.

### Experiment dashboard

- `dashboard/` holds a dashboard with one card per experiment (status, summary, research question, scores, training curve, runs, and three videos: the actor in the real `lazy_enemy` game and inside its world model started from Pong or `lazy_enemy` histories). It finds experiments and runs automatically; see `dashboard/README.md`.
- Start each experiment's Markdown file with a `## Summary` section (1-3 sentences: the approach and its outcome) and list every run's exact command (with `--name <run>`) in it, so the dashboard can show and attribute it.
- After a run finishes or an experiment's Markdown changes, rebuild with `CUDA_VISIBLE_DEVICES= uv run dashboard/build_dashboard.py` (CPU only) and commit `dashboard/` with the experiment. If you have the Artifact tool, also republish the hosted copy as described in `dashboard/README.md`.
- The dashboard only displays `lazy_enemy` results. Do not use it for model selection or tuning.

## Robustness evaluation with `lazy_enemy`

- The JAXAtari **`lazy_enemy`** mod can be used to check whether a learned world model is robust to changed object dynamics (see `project.md`; the goal is an actor reaching a final score (own points minus enemy points) of at least +10 with the mod active).
- **NEVER train on `lazy_enemy`.** Do not use it for training data, for training the world model or the actor, for tuning hyperparameters, or for model selection. It is for validation only.
- All training uses the unmodified Pong environment. `lazy_enemy` is only used to evaluate finished models.
