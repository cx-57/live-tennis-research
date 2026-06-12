"""Live win-probability model combining the pre-match prior with in-match serving.

Each player's serve-win probability is a blend of the Elo-implied prior and the serve-won rate
observed so far in the match. The blend is a Bayesian shrink with a pseudo-count kappa: early in
the match, when few serve points have been seen, the prior dominates; as serve points accumulate
the observed rate takes over. The Markov recursion turns the two serve probabilities and the
current score into a win probability. The base and slope come from the asymmetric-Markov
calibration; kappa is calibrated here on the validation season.
"""
import numpy as np

from common import load, split, report, val_logloss, STATE
from markov import predict

BASE, SLOPE = 0.61, 1.3e-4


def serve_probs(df, base, slope, kappa):
    edge = np.clip(slope * df.elo_diff.to_numpy(), -0.15, 0.15)
    prior_a, prior_b = base + edge, base - edge
    r1, n1 = df.p1_serve_rate.to_numpy(), df.p1_serve_n.to_numpy()
    r2, n2 = df.p2_serve_rate.to_numpy(), df.p2_serve_n.to_numpy()
    pa = (n1 * r1 + kappa * prior_a) / (n1 + kappa)
    pb = (n2 * r2 + kappa * prior_b) / (n2 + kappa)
    return np.clip(pa, 0.45, 0.88), np.clip(pb, 0.45, 0.88)


def main():
    train, val, test = split(load(with_elo=True))

    best_k, best_ll = None, float("inf")
    for kappa in [40, 80, 160, 320, 640]:
        ll = val_logloss(val.y.values, predict(val, *serve_probs(val, BASE, SLOPE, kappa), STATE))
        if ll < best_ll:
            best_k, best_ll = kappa, ll
    print(f"calibrated kappa = {best_k}")

    report("serve-shrink model", test.y.values,
           predict(test, *serve_probs(test, BASE, SLOPE, best_k), STATE))


if __name__ == "__main__":
    main()
