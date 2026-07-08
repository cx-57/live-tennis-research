"""Symmetric Markov anchor

Points are treated as independent and the two players as equally strong, so the live win
probability is purely based on the score. The single serve-win probability is calibrated on
the validation season. This acts as the baseline that the other models are measured against.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load, split, report, val_logloss, STATE
from src.markov import predict

MATCH_FRACTION = 0.50


def main():
    train, val, test = split(load(with_elo=False, match_fraction=MATCH_FRACTION))

    # Tune one shared serve-win probability because both players are treated equally
    best_p, best_ll = None, float("inf")
    for p in [0.60, 0.61, 0.62, 0.63, 0.64, 0.65]:
        ll = val_logloss(val.y.values, predict(val, p, p, STATE))
        if ll < best_ll:
            best_p, best_ll = p, ll

    print(f"calibrated serve-win probability p = {best_p}")

    report("symmetric Markov", test.y.values, predict(test, best_p, best_p, STATE))


if __name__ == "__main__":
    main()
