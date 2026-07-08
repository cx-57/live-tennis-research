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

The models were evaluated at different match fractions, where the match fraction represents how far into the match the live prediction is made. Later match fractions generally produce higher accuracy and lower log loss because more score and serve-performance information is available.

### Markov Baselines

| Model | Match Fraction | Log Loss | Brier Score | Accuracy |
|---|---:|---:|---:|---:|
| Baseline Markov | 25% | 0.6292 | 0.2182 | 0.6851 |
| Baseline Markov | 50% | 0.5363 | 0.1745 | 0.7703 |
| Baseline Markov | 75% | 0.3498 | 0.1040 | 0.8544 |
| Asymmetric Markov | 25% | 0.5210 | 0.1738 | 0.7575 |
| Asymmetric Markov | 50% | 0.4549 | 0.1459 | 0.8050 |
| Asymmetric Markov | 75% | 0.3096 | 0.0949 | 0.8648 |

The asymmetric Markov model improves over the baseline Markov model at each listed match fraction. This suggests that adding pre-match player strength through Elo-based serve probabilities gives the model a stronger starting point than using score alone.

### Live Feature and Residual Model Results

| Match Fraction | Markov Accuracy | Serve-Shrink Accuracy | Residual Accuracy | Markov Log Loss | Serve-Shrink Log Loss | Residual Log Loss |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 0.7090 | 0.7110 | 0.7090 | 0.5515 | 0.5509 | 0.5553 |
| 10% | 0.7255 | 0.7296 | 0.7451 | 0.5399 | 0.5352 | 0.5320 |
| 15% | 0.7358 | 0.7368 | 0.7379 | 0.5287 | 0.5246 | 0.5151 |
| 20% | 0.7430 | 0.7420 | 0.7523 | 0.5295 | 0.5281 | 0.5035 |
| 25% | 0.7575 | 0.7564 | 0.7606 | 0.5155 | 0.5142 | 0.4753 |
| 30% | 0.7492 | 0.7513 | 0.7606 | 0.5414 | 0.5390 | 0.4604 |
| 35% | 0.7534 | 0.7544 | 0.7668 | 0.5319 | 0.5295 | 0.4397 |
| 40% | 0.7595 | 0.7626 | 0.7812 | 0.4836 | 0.4803 | 0.4046 |
| 45% | 0.7843 | 0.7853 | 0.8060 | 0.4683 | 0.4629 | 0.3921 |
| 50% | 0.8039 | 0.7946 | 0.8215 | 0.4336 | 0.4266 | 0.3530 |
| 55% | 0.8184 | 0.8111 | 0.8338 | 0.4201 | 0.4057 | 0.3248 |
| 60% | 0.8338 | 0.8318 | 0.8431 | 0.4120 | 0.4002 | 0.2926 |
| 65% | 0.8411 | 0.8328 | 0.8617 | 0.3570 | 0.3413 | 0.2725 |
| 70% | 0.8504 | 0.8607 | 0.8679 | 0.3345 | 0.3086 | 0.2335 |
| 75% | 0.8648 | 0.8720 | 0.8834 | 0.3139 | 0.2842 | 0.2002 |
| 80% | 0.8741 | 0.8875 | 0.8916 | 0.2903 | 0.2602 | 0.1791 |
| 85% | 0.9061 | 0.9092 | 0.9195 | 0.2550 | 0.2254 | 0.1521 |
| 90% | 0.9360 | 0.9556 | 0.9536 | 0.1834 | 0.1628 | 0.1114 |
| 95% | 0.9598 | 0.9701 | 0.9649 | 0.1348 | 0.1188 | 0.0648 |

The residual model generally produces the lowest validation log loss across most match fractions, especially later in matches. This supports the idea that a structural Markov model provides a strong foundation, while a residual machine-learning layer can improve probability calibration using additional live features.

The full result table is saved in:

```text
images/residual_markov_live_features_accuracy.csv

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
