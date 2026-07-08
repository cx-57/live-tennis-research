# Live Tennis Win Probability

This repository contains a live tennis win-probability modeling project. The goal is to estimate the probability that a player wins a tennis match while the match is already in progress, using the current score state, pre-match player strength, and in-match serving performance.

The project combines a rule-based Markov recursion with learned player-specific inputs. Tennis scoring has a fixed recursive structure across points, games, sets, tiebreaks, and matches, so the model does not need to learn the rules of tennis from scratch. Instead, the main modeling problem is estimating the point-level serve probabilities that feed into the recursion.

## Overview

The project compares several live win-probability models:

1. **Symmetric Markov model**  
   A score-only baseline that treats both players as equally strong. The model uses a single global serve-point win probability and computes match win probability from the current score.

2. **Elo-asymmetric Markov model**  
   A structural Markov model where each player's serve probability is shifted using the pre-match Elo difference. This allows the model to start matches away from 50/50 when one player is stronger.

3. **Serve-shrink Markov model**  
   A live model that blends the Elo-based prior with serve points observed earlier in the same match. A pseudo-count parameter controls how quickly the model trusts in-match serve performance.

4. **Residual / calibrated model**  
   A machine-learning layer on top of the structural Markov prediction. This model uses the Markov probability plus selected live features to improve probability calibration.

## Data

The project is designed around public tennis datasets:

- Grand Slam point-by-point data
- ATP match results
- WTA match results

The point-level data is transformed into live match states. Each row represents a point in a match and includes the score after that point, server information, set/game/point state, and whether player 1 eventually won the match.

Pre-match Elo ratings are built from ATP and WTA match results and joined to the Grand Slam point-by-point data.

Large raw data files are not intended to be stored directly in this repository.

## Repository Structure

```text
live-tennis-research/
├── models/
│   ├── asymmetric_markov.py
│   ├── baseline_markov.py
│   ├── baseline_xgboost.py
│   ├── residual_markov.py
│   └── serve_shrink_model.py
├── src/
│   ├── __init__.py
│   ├── common.py
│   ├── elo.py
│   ├── markov.py
│   └── prepare_data.py
├── images/
│   ├── model_accuracy.csv
│   ├── model_accuracy.png
│   └── other saved result plots
├── paper.tex
├── README.md
└── .gitignore
```

## Main Components

### `src/prepare_data.py`

Builds the point-level modeling table from raw Grand Slam point-by-point data. It creates the live score state and running match features used by the models.

### `src/elo.py`

Builds pre-match Elo ratings from ATP and WTA match results. These ratings are used to estimate each player's prior strength before a match begins.

### `src/markov.py`

Contains the Markov recursion for tennis scoring. Given the current score and each player's serve-point win probability, it computes the probability that player 1 wins the match.

### `src/common.py`

Stores shared paths, data loading functions, train/validation/test splitting, and evaluation metrics.

### `models/baseline_markov.py`

Runs the symmetric Markov baseline. This model uses only the score state and a global serve-point probability.

### `models/asymmetric_markov.py`

Runs the Elo-asymmetric Markov model. This model adjusts serve probabilities based on the pre-match Elo gap.

### `models/serve_shrink_model.py`

Runs the serve-shrink model. This model combines the Elo prior with observed in-match serve performance.

### `models/baseline_xgboost.py`

Runs a machine-learning baseline using score and live context features.

### `models/residual_markov.py`

Runs a residual or calibrated model that builds on the structural Markov prediction using selected live features.

## Method

The core model uses a nested Markov recursion:

- Point probabilities determine game probabilities.
- Game probabilities determine set probabilities.
- Set probabilities determine match probabilities.

The structural model requires two main inputs:

- Probability player 1 wins a point on player 1's serve
- Probability player 2 wins a point on player 2's serve

The symmetric baseline uses the same serve probability for both players. The asymmetric model shifts these probabilities using Elo difference. The serve-shrink model updates them using serve results observed earlier in the match.

The serve-shrink update has the form:

```text
updated serve probability =
(observed serve points won + prior pseudo-count contribution)
/
(observed serve points + pseudo-count)
```

This prevents the model from overreacting to a small number of early serve points while still allowing it to adjust as more in-match evidence becomes available.

## Evaluation

The project uses a time-based split:

- Training: matches through 2021
- Validation: 2022
- Testing: 2023 and later

Models are evaluated using:

- Log loss
- Brier score
- Accuracy

Log loss is the most important metric because this is a probability prediction problem. Accuracy only measures whether the model is on the correct side of 50%, while log loss rewards well-calibrated probabilities.

## Current Results

The models were evaluated at 25%, 50%, and 75% of the way through each match. These checkpoints show how live win-probability performance changes as more score and in-match serving information becomes available.

| Model | 25% Accuracy | 25% Log Loss | 25% Brier | 50% Accuracy | 50% Log Loss | 50% Brier | 75% Accuracy | 75% Log Loss | 75% Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Markov | 0.6851 | 0.6292 | 0.2182 | 0.7703 | 0.5363 | 0.1745 | 0.8544 | 0.3498 | 0.1040 |
| Asymmetric Markov | 0.7575 | 0.5210 | 0.1738 | 0.8050 | 0.4549 | 0.1459 | 0.8648 | 0.3096 | 0.0949 |
| Serve-Shrink Markov | 0.7564 | 0.5142 | — | 0.7946 | 0.4266 | — | 0.8720 | 0.2842 | — |
| Residual Markov | 0.7606 | 0.4753 | — | 0.8215 | 0.3530 | — | 0.8834 | 0.2002 | — |

The baseline Markov model uses only the tennis score state. The asymmetric Markov model adds pre-match player strength through Elo-based serve probabilities. The serve-shrink model updates those probabilities using observed in-match serving data. The residual Markov model adds a machine-learning correction on top of the structural Markov prediction and performs best overall in this comparison, especially by log loss.

The full result table is saved in:

images/residual_markov_live_features_accuracy.csv

## How to Run

First, install the main Python dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib xgboost pyarrow
```

Then prepare the data and Elo artifacts. The expected structure is:

```text
data/
├── slam/
├── atp/
└── wta/

artifacts/
├── points.parquet
└── elo.parquet
```

Run the model scripts from the repository root:

```bash
python models/baseline_markov.py
python models/asymmetric_markov.py
python models/serve_shrink_model.py
python models/baseline_xgboost.py
python models/residual_markov.py


Some scripts save result plots and CSV files into the `images/` folder.

## Project Motivation

Pregame tennis prediction only uses information available before the match starts. Live win probability is more dynamic: the model must update after the score changes and after new information about player performance becomes available.

This project focuses on that live setting. The main idea is that tennis scoring should be handled structurally, while machine learning and statistical estimation should be used to estimate the player-specific inputs to that structure.
## Limitations

The current version has several limitations:

- Some evaluations use fixed match fractions instead of every point in the match.
- Surface, fatigue, injury, tactics, and pressure are not modeled directly.
- Elo matching depends on player-name joins, which may miss some matches.
- Serve and return strength are simplified into serve-point probabilities.
- The residual model is still relatively lightweight.

## Future Improvements

Possible next steps:

- Evaluate every point on a common held-out test set.
- Add surface-specific Elo ratings.
- Separate serve strength from return strength.
- Add pressure-point features.
- Improve calibration across different match stages.
- Compare the structural model against stronger machine-learning baselines.
- Build a simple live visualization that updates win probability point by point.

## Acknowledgments

This project uses public tennis datasets maintained by Jeff Sackmann, including Grand Slam point-by-point data and ATP/WTA match results.
