# Experiments outside the paper

These files preserve work present before the paper-alignment cleanup. They are not the five models evaluated in the paper.

- `trace_residual.py`: XGBoost corrections using forward-chained HGBM predictions. Requires `xgboost`; its baseline is `hgbm_with_elo.py`.
- `hgbm_with_elo.py`: HGBM with pre-match Elo features.
- `hgbm_with_elo_accuracy.csv`: existing results from that Elo-enabled baseline.
- `point_serve_model.py`: learned next-point serve probabilities for Markov inputs.

Run scripts from the repository root. New experimental plots/tables go to `experiments/outputs/`, which is ignored. Existing experiment results have not been relabeled as paper results.

`main_before_cleanup/` preserves main-specific manuscript and legacy exports from the merge. See its `SNAPSHOT.md` for provenance.
