# Paper: observation interventions for imagination-trained agents

NeurIPS 2025 style (`preprint` option), `main.tex` → `main.pdf`.

## Rebuild

From the repository root:

```bash
# 1. model-dependent figure data (trajectories, gates, rollouts, leakage); needs the selected checkpoints
CUDA_VISIBLE_DEVICES=4 uv run paper/compute_figure_data.py
# 2. all figures (PDF), numbers.tex (LaTeX macros for every number quoted in the text) and the main table
uv run paper/make_figures.py
# 3. PDF (any LaTeX distribution works; we used tectonic 0.15)
cd paper && tectonic -X compile main.tex        # or: latexmk -pdf main.tex
```

`numbers.tex` is generated. Do not edit it by hand. Results come from the
`runs/<run>/robust/lazy_enemy_eval.json` and `best_selection.json` files, which the repository scripts
write:

- training: `experiment_*.py` (see the corresponding `experiment_*.md` for commands),
- selection on unmodified Pong: `select_robust_checkpoint.py`,
- held-out evaluation: `evaluate_lazy_enemy.py`.

## Figure colours

The categorical colours come from a validated palette: they pass the lightness, chroma,
colour-vision-deficiency and normal-vision separation checks. Because some colours have low contrast
against white, every figure identifies series with direct labels, a legend or line styles, not with
colour alone.
