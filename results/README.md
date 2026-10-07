# Result provenance

`paper_reported_metrics.csv` is a transcription of Tables 4 and 5 in `../forecasting_the_winner_of_a_live_tennis_match.pdf`. It contains reported values, not newly computed predictions.

The other CSV files were already present before the cleanup. They were moved from `images/`, and model labels/column names were normalized (`Markov Ensemble` → `Trace`, `residual_*` → `trace_*`). Numerical measurements and parameter records were preserved.

- `model_accuracy.csv`: five-model progress curve used for Figure 2. Its checkpoint exports differ slightly from Table 4: for example, Trace at 25% is approximately 0.76161 here versus 0.7606 in the paper. The original fractions include floating-point values near 0.50; do not interpret those differences as separate checkpoints.
- `trace_accuracy.csv`: historical structural-model/Trace export, retained for provenance; it overlaps with the corresponding columns of the five-model table.
- `atp_wta_match_fraction_accuracy.csv`: tour-specific evaluations used for Figure 4. The figure plots Elo-asymmetric Markov, serve-shrink Markov, and Trace.
- `calibration_bins.csv`, `calibration_summary.csv`, `calibration_tuning.csv`: historical calibration outputs. They do not exactly reproduce Table 5: for example, the saved Trace log loss at 75% is approximately 0.22933, while the paper reports 0.2002. The new calibration command uses the paper-facing Trace implementation and its checkpoint seeds; these saved historical numbers have not been silently replaced.
- `year_split_2011_2019_dev2017_test2014.csv`: supplementary non-chronological comparison, separate from the primary test period.

The Elo-enabled HGBM result table is preserved in `../experiments/hgbm_with_elo_accuracy.csv`, because its model inputs differ from the paper's baseline.

No full retraining was performed during cleanup. Fresh runs may overwrite result tables and generated figures. Library versions, source-data revisions, and checkpoint selection should be recorded when investigating numerical reproducibility. The supplied PDF remains the reference for the reported results.
