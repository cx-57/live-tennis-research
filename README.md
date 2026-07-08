# Live Tennis Win Probability

This repository contains a live tennis win-probability modeling project. The goal is to estimate the probability that a player wins a match while the match is already in progress, using the current score state, pre-match player strength, and in-match serving performance.

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
