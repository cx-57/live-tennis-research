"""Asymmetric Markov with a pre-match Elo prior.

The recursion is the same as the anchor, but the two players are given different serve-win
probabilities set by the pre-match Elo gap: base plus or minus a slope times the rating
difference. This lets a match start away from fifty-fifty when the players are unequal, which is
most of what a live model is missing at the start of a match. The base and slope are calibrated
on the validation season.
"""
import numpy as np

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load, split, report, val_logloss, STATE
from src.markov import predict

MATCH_FRACTION = 0.50


def serve_probs(df, base, slope):
    # Convert the Elo gap into separate serve probabilities for player 1 and player 2
    edge = np.clip(slope * df.elo_diff.to_numpy(), -0.15, 0.15)
    return np.clip(base + edge, 0.45, 0.88), np.clip(base - edge, 0.45, 0.88)


def main():
    train, val, test = split(load(with_elo=True, match_fraction=MATCH_FRACTION))

    best, best_ll = None, float("inf")

    # Tune base serve strength and Elo slope on the validation season
    for base in [0.61, 0.62, 0.63, 0.64]:
        for slope in [4e-5, 6e-5, 9e-5, 1.3e-4, 1.8e-4]:
            ll = val_logloss(val.y.values, predict(val, *serve_probs(val, base, slope), STATE))
            if ll < best_ll:
                best, best_ll = (base, slope), ll

    base, slope = best
    print(f"calibrated base={base} slope={slope}")

    report("asymmetric Markov (Elo prior)", test.y.values,
           predict(test, *serve_probs(test, base, slope), STATE))


if __name__ == "__main__":
    main()
