# Forecasting the Winner of a Live Tennis Match

Research by Charles Xie and Aneesh Muppidi on live tennis match-win probabilities, using score state, pre-match Elo, and serving performance observed before the next point.

Read the [full paper](forecasting_the_winner_of_a_live_tennis_match.pdf) or its [LaTeX source](forecasting_the_winner_of_a_live_tennis_match.tex). The supplied paper is the reference for the terminology and reported results below.

## Models

| Paper model | Script | Inputs and method |
|---|---|---|
| Symmetric Markov | `models/symmetric_markov.py` | Score recursion with equal serve probabilities for both players. |
| Elo-asymmetric Markov | `models/elo_asymmetric_markov.py` | Score recursion with serve probabilities shifted by pre-match Elo difference. |
| Serve-shrink Markov | `models/serve_shrink_markov.py` | Elo serve priors blended with observed serve rates using a validation-tuned pseudo-count. |
| HGBM | `models/hgbm.py` | Histogram gradient boosting on score and live features, without Elo or Markov predictions. |
| Trace | `models/trace.py` | Histogram gradient boosting on Elo-asymmetric and serve-shrink Markov probabilities, their logits and differences, score state, and live features. |

Trace is the HGBM hybrid described in Section 3.6. Earlier names such as “Markov Ensemble” and “residual model” have been replaced in the paper-facing code. The separate XGBoost residual implementation and Elo-enabled HGBM are preserved in [experiments](experiments/README.md).

## Data and evaluation

The paper uses Jeff Sackmann's Grand Slam point-by-point data and ATP/WTA match histories. After filtering, it reports 8,222 matches and 1,505,355 point states, with Elo coverage of 96.0% of prepared matches. Features describe the state **before a point is played**; the eventual match winner is the target.

The primary split is training in 2011–2021, validation in 2022, and testing in 2023–2024. Fixed checkpoints are 25%, 50%, and 75% match progress, with additional all-point evaluation. Match fractions are retrospective evaluation checkpoints determined using each match's recorded length; they are not features available in a live deployment.

The separate comparison in Table 6 uses 2011–2019, development in 2017, and testing in 2014, with those two years excluded from training. This non-chronological comparison is kept separate from the primary results.

## Reported results

These values are transcribed from Tables 4 and 5 of the supplied paper, not from a fresh training run.

| Model | 25% accuracy | 50% accuracy | 75% accuracy | All-point accuracy | 25% log loss | 50% log loss | 75% log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Symmetric Markov | 0.6851 | 0.7703 | 0.8544 | 73.21% | 0.6292 | 0.5363 | 0.3498 |
| Elo-asymmetric Markov | 0.7575 | 0.8050 | 0.8648 | 77.56% | 0.5210 | 0.4549 | 0.3096 |
| Serve-shrink Markov | 0.7564 | 0.7946 | 0.8720 | 77.68% | 0.5142 | 0.4266 | 0.2842 |
| HGBM | 0.6944 | 0.7918 | 0.8738 | 73.98% | 0.5376 | 0.3788 | 0.2299 |
| Trace | 0.7606 | 0.8215 | 0.8834 | 77.84% | 0.4753 | 0.3530 | 0.2002 |

The same reported values are stored in `results/paper_reported_metrics.csv`. Existing evaluation exports are retained separately; see [result provenance](results/README.md) for discrepancies with the paper.

## Repository layout

```text
live-winprob/
├── forecasting_the_winner_of_a_live_tennis_match.pdf
├── forecasting_the_winner_of_a_live_tennis_match.tex
├── models/          # five paper models and evaluation utilities
├── src/             # data preparation, Elo, scoring recursion, live features
├── images/          # only Figures 1–4 used by the manuscript
├── results/         # result tables, provenance, and generated evaluations
├── experiments/     # preserved work outside the paper's model comparison
├── archive/         # preserved legacy work and data
├── data/            # ignored raw datasets
├── artifacts/       # ignored prepared points and Elo tables
└── requirements.txt
```

The manuscript figures use consistent names:

1. `images/figure_1_trace_pipeline.png` — Trace pipeline, extracted from the supplied PDF.
2. `images/figure_2_model_accuracy.png` — accuracy of the five models across match progress.
3. `images/figure_3_trace_calibration.png` — Trace reliability curves at the three checkpoints.
4. `images/figure_4_tour_accuracy.png` — ATP/WTA accuracy curves.

Duplicate and obsolete image exports were removed. Source data tables were retained in `results/`.

## Running the pipeline

Install the core dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Place raw datasets in `data/slam/`, `data/atp/`, and `data/wta/`. Data and prepared artifacts can also be located using `TENNIS_DATA` and `TENNIS_ARTIFACTS`.

From the repository root, prepare points and pre-match Elo:

```bash
python3 src/prepare_data.py
python3 src/elo.py
```

These commands produce `artifacts/points.parquet` and `artifacts/elo.parquet`. Then run the models:

```bash
python3 models/symmetric_markov.py
python3 models/elo_asymmetric_markov.py
python3 models/serve_shrink_markov.py
python3 models/hgbm.py
python3 models/trace.py
```

The standalone Markov scripts use the `MATCH_FRACTION` constant, initially 0.50. HGBM evaluates the three checkpoints. Trace evaluates the accuracy curve at 5% intervals from 5% to 95%; set `PLOT_ACCURACY_CURVE = False` and adjust `MATCH_FRACTION` for a single checkpoint. Fresh Trace outputs go to `results/model_accuracy.csv` and the Figure 2 PNG/PDF paths.

Additional evaluations:

```bash
python3 models/calibration.py
python3 models/comparison_split.py
python3 models/tour_accuracy.py
```

Calibration uses the same Trace fitter and checkpoint seeds, producing calibration tables and Figure 3. The comparison script produces the Table 6 split results. The tour script renders Figure 4 from the retained tour table. These commands can overwrite the corresponding saved exports; they do not update the manuscript's reported numerical tables.

Build the LaTeX manuscript from the repository root, with a local LaTeX installation:

```bash
pdflatex forecasting_the_winner_of_a_live_tennis_match.tex
pdflatex forecasting_the_winner_of_a_live_tennis_match.tex
```

## Limitations

The study is limited to Grand Slams and does not explicitly model return strength, surface-specific Elo, fatigue, injury, weather, handedness, or playing style. Real-time deployment and broader tour-level testing remain future work. The cleanup did not rerun the full training pipeline or establish exact reproduction of every reported result.

## Acknowledgements

Public datasets are maintained by Jeff Sackmann. Dataset licensing and source documentation remain in `data/`; the paper contains the full bibliography.
