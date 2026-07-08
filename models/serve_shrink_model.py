"""Serve-shrink Markov model at one match fraction."""

import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common import load, split, report, val_logloss, STATE
from markov import predict

MATCH_FRACTION = 0.50

BASE_GRID = [0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65]
SLOPE_GRID = [4e-5, 6e-5, 9e-5, 1.3e-4, 1.8e-4, 2.2e-4]
KAPPA_GRID = [40, 80, 160, 320, 640]


def serve_probs(df, base, slope, kappa):
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


def tune_params(val):
    best_params = None
    best_ll = float("inf")

    for base in BASE_GRID:
        for slope in SLOPE_GRID:
            for kappa in KAPPA_GRID:
                pa, pb = serve_probs(val, base, slope, kappa)
                pred = predict(val, pa, pb, STATE)
                ll = val_logloss(val.y.values, pred)

                if np.isfinite(ll) and ll < best_ll:
                    best_params = (base, slope, kappa)
                    best_ll = ll

    if best_params is None:
        raise ValueError("Could not tune serve-shrink parameters.")

    return best_params, best_ll


def main():
    train, val, test = split(load(with_elo=True, match_fraction=MATCH_FRACTION))

    print(f"match_fraction={MATCH_FRACTION}")

    (base, slope, kappa), val_ll = tune_params(val)

    print(
        f"serve-shrink params: "
        f"base={base} slope={slope} kappa={kappa} val_logloss={val_ll:.4f}"
    )

    pa, pb = serve_probs(test, base, slope, kappa)
    pred = predict(test, pa, pb, STATE)

    report("serve-shrink Markov", test.y.values, pred)


if __name__ == "__main__":
    main()