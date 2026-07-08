"""Serve-shrink Markov model using Elo priors and live serve performance.

This model starts with an Elo-based estimate of each player's serve strength, then blends it
with each player's serve results from the match so far.

Kappa is the shrinkage strength: a larger kappa makes the model trust the pre-match prior longer,
while a smaller kappa makes it adjust more quickly to live serve performance.
"""

import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common import load, split, report, val_logloss, STATE
from markov import predict

MATCH_FRACTION = 0.50

# Candidate values for the Elo prior and kappa shrinkage strength
BASE_GRID = [0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65]
SLOPE_GRID = [4e-5, 6e-5, 9e-5, 1.3e-4, 1.8e-4, 2.2e-4]
KAPPA_GRID = [40, 80, 160, 320, 640]


def serve_probs(df, base, slope, kappa):
    # Blend the Elo-based prior with each player's live serve win rate
    edge = np.clip(slope * df.elo_diff.to_numpy(), -0.15, 0.15)

    prior_a = base + edge
    prior_b = base - edge

    p1_rate = df.p1_serve_rate.to_numpy()
    p2_rate = df.p2_serve_rate.to_numpy()

    p1_n = df.p1_serve_n.to_numpy()
    p2_n = df.p2_serve_n.to_numpy()

    pa = (p1_n * p1_rate + kappa * prior_a) / (p1_n + kappa)
    pb = (p2_n * p2_rate + kappa * prior_b) / (p2_n + kappa)

    return np.clip(pa, 0.45, 0.88), np.clip(pb, 0.45, 0.88)
